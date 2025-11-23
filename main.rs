use clap::ArgAction;
use clap::Parser;
use dashmap::DashSet;
use mimalloc::MiMalloc;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::cell::LazyCell;
use std::cmp::Reverse;
use std::error::Error;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::fmt::Write;
use std::fs::DirEntry;
use std::fs::Metadata;
use std::io::Write as _;
use std::io::{self, IsTerminal, Result, stderr};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::result::Result::Ok;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::RecvTimeoutError::{Disconnected, Timeout};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};
use std::{thread, u64};

/// A simple [`u64`] acting like a [`Vec<bool>`]
#[derive(Copy, Clone, Default)]
struct BoolsVec64 {
    val: u64,
    len: u8,
}

impl BoolsVec64 {
    fn push(&mut self, value: bool) {
        self.set(self.len, value);
        self.len += 1;
    }
    fn pop(&mut self) -> Option<bool> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.get(self.len))
    }
    fn get(&self, index: u8) -> bool {
        (self.val & (1 << index)) != 0
    }
    fn set(&mut self, index: u8, value: bool) {
        if value {
            self.val |= 1 << index;
        } else {
            self.val &= !(1 << index);
        }
    }
    fn join(mut self, value: bool) -> Self {
        self.push(value);
        self
    }
}

impl Extend<bool> for BoolsVec64 {
    fn extend<T: IntoIterator<Item = bool>>(&mut self, iter: T) {
        for x in iter {
            self.push(x)
        }
    }
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

struct Node {
    name: Box<OsStr>,
    total_size: u64,
    total_count: usize,
    content: NodeContent,
}

enum NodeContent {
    Directory(Box<[Node]>),
    Symlink(Box<Path>),
    File,
    Other,
}

struct FlatNode {
    total_size: u64,
    total_count: usize,
    is_last_child_list: BoolsVec64,
    has_children: bool,
    path: Box<Path>,
    target: Option<Box<Path>>,
}

trait NodeLike {
    fn total_size(&self) -> u64;
    fn total_count(&self) -> usize;
    fn path_or_name(&self) -> &Path;
}

impl NodeLike for Node {
    fn total_size(&self) -> u64 {
        self.total_size
    }
    fn total_count(&self) -> usize {
        self.total_count
    }
    fn path_or_name(&self) -> &Path {
        Path::new(&self.name)
    }
}

impl NodeLike for FlatNode {
    fn total_size(&self) -> u64 {
        self.total_size
    }
    fn total_count(&self) -> usize {
        self.total_count
    }
    fn path_or_name(&self) -> &Path {
        &self.path
    }
}

type FxDashSet<T> = DashSet<T, FxBuildHasher>;

fn get_disk_size(metadata: &Metadata) -> u64 {
    #[cfg(unix)]
    return metadata.blocks() * 512;
    #[cfg(not(unix))]
    return metadata.len();
}

static BLOCK_SIZE: LazyLock<u64> = LazyLock::new(|| {
    for var in ["DU_BLOCK_SIZE", "BLOCK_SIZE", "BLOCKSIZE"] {
        if let Some(value) = std::env::var_os(var) {
            return value
                .into_string()
                .unwrap_or_else(|x| panic!("{var} is not valid UTF-8: {x:?}"))
                .parse()
                .unwrap_or_else(|e| panic!("{var} is not a valid integer: {e}"));
        }
    }

    let posix_mode = std::env::var_os("POSIXLY_CORRECT").is_some();
    if posix_mode { 512 } else { 1024 }
});

impl Node {
    fn flatten_tree_root(self, output: &mut Vec<FlatNode>, args: &DuArgs) {
        if args.should_output(&self, 0) {
            self.flatten_tree(output, BoolsVec64::default(), args, 0);
        }
    }

    fn flatten_tree(
        self,
        output: &mut Vec<FlatNode>,
        is_last_child_list: BoolsVec64,
        args: &DuArgs,
        depth: usize,
    ) {
        let mut children = Vec::new();
        let mut target = None;

        match self.content {
            NodeContent::Directory(x) => {
                children = x
                    .into_iter()
                    .filter(|child| args.should_output(child, depth + 1))
                    .collect()
            }
            NodeContent::Symlink(x) => target = Some(x),
            _ => {}
        };

        output.push(FlatNode {
            is_last_child_list,
            total_size: self.total_size,
            total_count: self.total_count,
            path: PathBuf::from(self.name.into_os_string()).into(),
            has_children: !children.is_empty(),
            target,
        });

        if let Some(last_child) = children.pop() {
            for child in children {
                child.flatten_tree(output, is_last_child_list.join(false), args, depth + 1);
            }

            last_child.flatten_tree(output, is_last_child_list.join(true), args, depth + 1);
        }
    }

    fn flatten_joined(
        self,
        output: &mut Vec<FlatNode>,
        parent_path: &Path,
        args: &DuArgs,
        depth: usize,
    ) {
        if !args.should_output(&self, depth) {
            return;
        }

        let path = fast_path_join(parent_path, &self.name);
        let mut target = None;
        let mut has_children = false;

        match self.content {
            NodeContent::Directory(children) => {
                has_children = !children.is_empty();
                for child in children {
                    child.flatten_joined(output, &path, args, depth + 1);
                }
            }
            NodeContent::Symlink(x) => target = Some(x),
            _ => {}
        }

        output.push(FlatNode {
            path: path.into(),
            total_size: self.total_size,
            total_count: self.total_count,
            has_children,
            is_last_child_list: Default::default(),
            target,
        });
    }

    fn is_dir(&self) -> bool {
        match self.content {
            NodeContent::Directory(_) => true,
            _ => false,
        }
    }

    fn sort_children(&mut self, args: &DuArgs) {
        if let NodeContent::Directory(children) = &mut self.content {
            args.sort_nodelikes(children);

            children
                .par_iter_mut()
                .for_each(|child| child.sort_children(args));
        };
    }
}

struct Scanner<'a> {
    output: &'a dyn Output,
    args: &'a DuArgs,
    #[allow(unused)]
    seen: FxDashSet<(u64, u64)>,
}

impl<'a> Scanner<'a> {
    fn new(args: &'a DuArgs, output: &'a dyn Output) -> Self {
        let seen = FxDashSet::with_hasher(FxBuildHasher);
        Self { args, output, seen }
    }

    fn scan_impl(
        &self,
        parent_path: &Path,
        entry: Option<&DirEntry>,
        #[allow(unused_mut)] mut root_dev: Option<u64>,
    ) -> Result<Option<impl FnOnce(&Scanner, &Path) -> Result<Node> + use<> + Send>> {
        let dereference =
            self.args.dereference_all || (entry.is_none() && self.args.dereference_args);

        // We don't always need the full path, and it costs to create it from a DirEntry so do so lazily
        let path = LazyCell::new(|| -> Cow<Path> {
            match entry {
                Some(entry) => entry.path().into(),
                None => parent_path.into(),
            }
        });

        // We don't need metadata with -lz or -li and no -x or -L, so load it as needed.
        let metadata = LazyCell::new(|| {
            if dereference {
                path.metadata()
            } else if let Some(entry) = entry {
                entry.metadata()
            } else {
                path.symlink_metadata()
            }
        });

        let metadata = || metadata.as_ref().map_err(|e| add_context(e, &path));

        #[cfg(unix)]
        if self.args.one_file_system {
            let dev = metadata()?.dev();
            if let Some(root) = root_dev {
                if dev != root {
                    return Ok(None);
                }
            } else {
                root_dev = Some(dev)
            }
        }

        let file_type = match entry {
            Some(entry) if !dereference => {
                // In theory this could do an lstat() again, but only on some obscure
                // platforms or for some obscure file types.
                entry.file_type().map_err(|e| add_context(&e, &path))?
            }
            _ => metadata()?.file_type(),
        };

        #[allow(unused)]
        let is_dir = file_type.is_dir();

        // This is the exact number of children on macOS and BSD, but on Linux it only counts
        // subdirectories. On Windows its not available at all.
        #[cfg(unix)]
        let num_children = if is_dir { metadata()?.nlink() - 2 } else { 0 } as usize;
        #[cfg(not(unix))]
        let num_children = 0usize;

        #[cfg(unix)]
        if !self.args.count_links && !is_dir {
            let stat = metadata()?;
            if stat.nlink() > 1 && !self.seen.insert((stat.dev(), stat.ino())) {
                return Ok(None);
            }
        }

        let name: Option<Box<OsStr>> = entry.map(|e| e.file_name().into());

        let mut total_size = if self.args.no_size || self.args.inodes {
            0
        } else if self.args.apparent_size || self.args.bytes {
            metadata()?.len()
        } else {
            get_disk_size(metadata()?)
        };

        // Drop references to the DirEntry and return a continuation
        // that will read childen and construct a complete Node.
        Ok(Some(move |scanner: &Scanner, parent_path: &Path| {
            let path = LazyCell::new(|| -> Cow<Path> {
                match &name {
                    Some(name) => fast_path_join(parent_path, name).into(),
                    None => parent_path.into(),
                }
            });

            let mut total_count = 1;

            let content = if file_type.is_file() {
                NodeContent::File
            } else if file_type.is_dir() {
                let path = &*path;

                let mut funs = Vec::with_capacity(num_children);
                // opendir() can produce EINTR on macOS when reading dirs in ~/Library/{Group ,}Containers
                let mut iter = retry_if_interrupted(|| path.read_dir());
                // as_mut() to avoid moving out of the Result<ReadDir> because it can be large
                let iter = iter.as_mut().map_err(|e| add_context(e, &path))?;

                loop {
                    match iter.next() {
                        Some(Ok(ref entry)) => match scanner.scan_impl(path, Some(entry), root_dev) {
                            Ok(Some(f)) => funs.push(f),
                            Ok(None) => continue,
                            Err(e) => scanner.output.log_error(&e),
                        },
                        Some(Err(e)) => scanner.output.log_error(&add_context(&e, &path)),
                        None => break,
                    }
                }

                // num_children is only exact on macOS, so use funs.len() from here
                let mut results = Vec::<Result<Node>>::with_capacity(funs.len());
                let mut nodes = Vec::<Node>::with_capacity(funs.len());

                funs.into_par_iter()
                    .map(|f| f(scanner, &path))
                    .collect_into_vec(&mut results);

                let mut iter = results.into_iter();
                loop {
                    // Match the Option<Result<Node>> as a whole to avoid moving the Result<Node> out first
                    match iter.next() {
                        Some(Ok(node)) => nodes.push(node),
                        Some(Err(e)) => scanner.output.log_error(&e),
                        None => break,
                    }
                }

                scanner.output.add_total(nodes.len());

                for node in &nodes {
                    total_size += node.total_size;
                    total_count += node.total_count;
                }

                NodeContent::Directory(nodes.into())
            } else if file_type.is_symlink() {
                let target = if scanner.args.tree {
                    path.read_link().map_err(|e| add_context(&e, &path))?
                } else {
                    PathBuf::new()
                };
                NodeContent::Symlink(target.into())
            } else {
                NodeContent::Other
            };

            Ok(Node {
                name: name.unwrap_or_else(|| parent_path.to_owned().into_os_string().into()),
                total_size,
                total_count,
                content,
            })
        }))
    }

    fn scan(&self, path: &Path) -> Result<Option<Node>> {
        Ok(self
            .scan_impl(path, None, None)?
            .map(|f| f(self, path))
            .transpose()?
            .inspect(|_| self.output.add_total(1)))
    }
}

fn fast_path_join(path: &Path, name: &OsStr) -> PathBuf {
    // Assume the result will be {path}/{name} so that in most
    // cases .into_boxed_path() doesn't reallocate.
    let mut result = OsString::with_capacity(path.as_os_str().len() + 1 + name.len());
    // Push first onto an OsString to skip the PathBuf::push logic.
    result.push(path);
    let mut result = PathBuf::from(result);
    result.push(name);
    result
}

fn add_context(error: &io::Error, path: &Path) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {}", path.display(), error))
}

#[derive(Parser)]
#[command(disable_help_flag = true)]
struct DuArgs {
    #[arg(
        short = 'A',
        long = "apparent-size",
        help = "Print apparent sizes rather than disk usage. This is always true on Windows."
    )]
    apparent_size: bool,

    #[arg(
        short = 'D',
        visible_short_alias = 'H',
        long = "dereference-args",
        help = "Dereference only symlinks that are listed on the command line",
        group = "symlink-mode"
    )]
    dereference_args: bool,

    #[arg(
        short = 'L',
        long = "dereference",
        help = "Dereference all symbolic links",
        group = "symlink-mode"
    )]
    dereference_all: bool,

    #[arg(
        short = 'P',
        long = "no-dereference",
        help = "Don't dereference any symbolic links. This is the default.",
        group = "symlink-mode"
    )]
    dereference_none: bool,

    #[arg(
        short = 'b',
        long = "bytes",
        help = "Print size in bytes. Implies --apparent-size.",
        group = "size-format"
    )]
    bytes: bool,

    #[arg(
        short = 'B',
        long = "block-size",
        value_name = "SIZE",
        help = "Print size in the specified block size (in bytes)",
        group = "size-format"
    )]
    block_size: Option<u64>,

    #[arg(
        short = 'k',
        long = "kilos",
        visible_aliases = ["kibis"],
        help = "Print size in 1024-byte blocks",
        group = "size-format"
    )]
    kibis: bool,

    #[arg(
        short = 'm',
        long = "megas",
        visible_alias = "mebis",
        help = "Print size in 1048576-byte blocks",
        group = "size-format"
    )]
    mebis: bool,

    #[arg(
        short = 'g',
        long = "gigas",
        visible_alias = "gibis",
        help = "Print size in 1073741824-byte blocks",
        group = "size-format"
    )]
    gibis: bool,

    #[arg(short = 'r', help = "Unused, accepted for conformance with XPG4")]
    messages: bool,

    #[arg(
        short = 'h',
        long = "human-readable",
        help = "Print sizes in human readable format (KiB, MiB etc)",
        group = "size-format"
    )]
    human: bool,

    #[arg(short = 'S', long = "sort", help = "Sort output (ascending order)")]
    sort: bool,

    #[arg(
        long = "si",
        help = "Like -h, but use powers of 1000 (KB, MB etc) instead of 1024",
        group = "size-format"
    )]
    si: bool,

    #[arg(
        short = 'i',
        long = "inodes",
        help = "Count inodes instead of size",
        group = "size-format"
    )]
    inodes: bool,

    #[arg(
        short = 'l',
        long = "count-links",
        help = "Count sizes many times if hard linked. This is always true on Windows."
    )]
    count_links: bool,

    #[arg(
        short = 'a',
        long = "all",
        help = "Write counts for all files, not just directories"
    )]
    all: bool,

    #[arg(
        short = 'x',
        long = "one-file-system",
        help = "Skip nodes on different file systems. No effect on Windows."
    )]
    one_file_system: bool,

    #[arg(
        short = 'n',
        long = "limit",
        help = "Show only first N results (after sorting)",
        value_name = "N"
    )]
    limit: Option<usize>,

    #[arg(short = '?', long = "help", action = ArgAction::Help, help = "Print help")]
    help: (),

    #[arg(
        short = 'R',
        long = "reverse",
        default_value_t = false,
        help = "Sort output (descending order)"
    )]
    reverse: bool,

    #[arg(
        short = 'z',
        long = "no-size",
        help = "Do not print or sort by sizes, only names/paths",
        group = "size-format"
    )]
    no_size: bool,

    #[arg(
        short = 'T',
        long = "tree",
        help = "Show a tree view of the directory structure"
    )]
    tree: bool,

    #[arg(
        short = 'j',
        long = "num-threads",
        help = "Thread count, defaults to 2x the number of CPU cores"
    )]
    num_threads: Option<usize>,

    #[arg(long = "no-progress", help = "Disable progress output")]
    no_progress: bool,

    #[arg(help = "Files or directories to scan")]
    files_or_directories: Vec<String>,

    #[arg(
        long = "du",
        help = "Show output in the same format as \"du\". See also --no-progress."
    )]
    du_compatible: bool,

    #[arg(
        short = '0',
        long = "null",
        help = "End each line with a null byte instead of newline"
    )]
    null: bool,

    #[arg(
        short = 'd',
        long = "max-depth",
        help = "Only show entries up to this maximum depth",
        group = "depth"
    )]
    max_depth: Option<usize>,

    #[arg(
        short = 's',
        long = "summarize",
        help = "Same as --max-depth=0",
        group = "depth"
    )]
    summarize: bool,

    #[arg(short = 'c', long = "total", help = "Include a grand total")]
    show_total: bool,

    #[arg(
        short = 't',
        long = "threshold",
        help = "Only include entries with size over threshold. If negative, only display values with size below."
    )]
    threshold: Option<i64>,
}

static STDERR_IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stderr().is_terminal());

impl DuArgs {
    fn size_format(&self) -> Option<SizeFormat> {
        match () {
            _ if self.no_size => None,
            _ if self.inodes => Some(SizeFormat::Count),
            _ if self.bytes => Some(SizeFormat::Bytes),
            _ if self.si => Some(SizeFormat::SI),
            _ if self.human => Some(SizeFormat::Human),
            _ if self.kibis => Some(SizeFormat::Blocks(1024)),
            _ if self.mebis => Some(SizeFormat::Blocks(1024 * 1024)),
            _ if self.gibis => Some(SizeFormat::Blocks(1024 * 1024 * 1024)),
            _ => match self.block_size.unwrap_or_else(|| *BLOCK_SIZE) {
                bs @ (1024 | 512) => Some(SizeFormat::Blocks(bs)),
                bs if self.apparent_size || self.bytes => Some(SizeFormat::Blocks(bs)),
                bs => Some(SizeFormat::Blocks(div_round_up(bs, 512))),
            },
        }
    }

    fn with_output<T>(&self, body: impl FnOnce(&dyn Output) -> T) -> T {
        if self.no_progress || !*STDERR_IS_TERMINAL {
            body(&SimpleOutput)
        } else {
            let total_count = &AtomicUsize::new(0);
            thread::scope(|scope| {
                let (sender, receiver) = channel();
                scope.spawn(|| ui_thread(total_count, receiver));
                body(&TerminalOutput {
                    ui_wakeups: sender,
                    total_count,
                })
            })
        }
    }

    fn should_output(&self, node: &Node, depth: usize) -> bool {
        (self.all || node.is_dir())
            && (!self.summarize || depth == 0)
            && (self.max_depth.map_or(true, |max_depth| depth <= max_depth))
    }

    fn sort_nodelikes(&self, children: &mut [impl NodeLike + Send]) {
        if self.no_size && self.reverse {
            children.par_sort_unstable_by(|a, b| a.path_or_name().cmp(&b.path_or_name()).reverse());
        } else if self.no_size {
            children.par_sort_unstable_by(|a, b| a.path_or_name().cmp(&b.path_or_name()));
        } else if self.inodes && self.reverse {
            children.par_sort_unstable_by_key(|x| Reverse(x.total_count()));
        } else if self.inodes {
            children.par_sort_unstable_by_key(|x| x.total_count());
        } else if self.reverse {
            children.par_sort_unstable_by_key(|x| Reverse(x.total_size()));
        } else {
            children.par_sort_unstable_by_key(|x| x.total_size());
        }
    }
}

enum SizeFormat {
    Count,
    Bytes,
    Human,
    SI,
    Blocks(u64),
}

trait Output: Sync {
    fn add_total(&self, num: usize);
    fn log_error(&self, error: &dyn Error);
}

struct TerminalOutput<'a> {
    ui_wakeups: Sender<()>,
    total_count: &'a AtomicUsize,
}

impl Output for TerminalOutput<'_> {
    fn add_total(&self, num: usize) {
        self.total_count.fetch_add(num, Relaxed);
    }

    fn log_error(&self, error: &dyn Error) {
        eprintln!("{CLEAR_LINE}{error}");
        let _ = self.ui_wakeups.send(());
    }
}

struct SimpleOutput;

impl Output for SimpleOutput {
    fn add_total(&self, _num: usize) {}

    fn log_error(&self, error: &dyn Error) {
        eprintln!("{error}")
    }
}

fn ui_thread(total_count: &AtomicUsize, wakups: Receiver<()>) {
    let start = Instant::now();
    let mut next_due = start;
    let mut stop = false;
    let mut line = String::new();
    while !stop {
        stop = loop {
            match wakups.recv_timeout(next_due - Instant::now()) {
                Ok(()) => eprint!("{CLEAR_LINE}{line}"),
                Err(Timeout) => break false,
                Err(Disconnected) => break true,
            }
        };
        let count = total_count.load(Relaxed);
        let secs = (Instant::now() - start).as_secs_f64();
        line = format!(
            "Scanned {count} nodes in {secs:.3} seconds (avg. {:.0} nodes/s)",
            count as f64 / secs
        );
        eprint!("{CLEAR_LINE}{line}");
        next_due += Duration::from_millis(1000);
    }
    eprintln!();
}

const CLEAR_LINE: &str = "\x1B[2K\r";

fn retry_if_interrupted<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    loop {
        let result = f();
        match &result {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            _ => return result,
        }
    }
}

fn main() -> Result<()> {
    let args = DuArgs::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.num_threads.unwrap_or(0))
        .use_current_thread()
        .build_global()
        .expect("Failed to set thread pool");

    let mut roots: Vec<Node> = args.with_output(|output| {
        let scanner = Scanner::new(&args, output);

        args.files_or_directories
            .par_iter()
            .flat_map_iter(|path| {
                scanner
                    .scan(path.as_ref())
                    .inspect_err(|e| output.log_error(e))
            })
            .flatten()
            .collect()
    });

    let total = FlatNode {
        path: PathBuf::from("total").into(),
        total_size: roots.iter().map(|x| x.total_size).sum(),
        total_count: roots.iter().map(|x| x.total_count).sum(),
        target: None,
        has_children: false,
        is_last_child_list: Default::default(),
    };

    let mut items = Vec::new();

    if !args.summarize && args.max_depth.is_none() {
        items.reserve_exact(total.total_count + 1);
    }

    if args.show_total {
        items.push(total);
    }

    if args.tree {
        // When outputting a tree, sort nodes in the tree first before flattening
        if args.sort || args.reverse {
            roots
                .par_iter_mut()
                .for_each(|root| root.sort_children(&args));
        }

        for root in roots {
            root.flatten_tree_root(&mut items, &args);
        }
    } else {
        for root in roots {
            root.flatten_joined(&mut items, Path::new(""), &args, 0);
        }

        // When not outputting a tree, sort items after flattening
        if args.sort || args.reverse {
            args.sort_nodelikes(&mut items);
        }
    }

    if let Some(mut threshold) = args.threshold {
        if threshold < 0 {
            threshold *= -1;
            items.retain(|node| node.total_size <= threshold as u64);
        } else {
            items.retain(|node| node.total_size >= threshold as u64);
        }
    }

    if let Some(limit) = args.limit {
        items.truncate(limit);
    }

    let mut output = String::with_capacity(
        items
            .iter()
            .map(|item| item.path.as_os_str().len() + 48)
            .sum(),
    );

    for item in &items {
        item.format_line(&args, &mut output).unwrap();
    }

    std::io::stdout().lock().write_all(output.as_bytes())
}

const fn div_round_up(size: u64, block_size: u64) -> u64 {
    let div = size / block_size;
    let rem = size % block_size;
    if rem == 0 { div } else { div + 1 }
}

impl FlatNode {
    fn format_line(&self, args: &DuArgs, out: &mut String) -> fmt::Result {
        let size = self.total_size;
        let count = self.total_count;
        let sep = if args.du_compatible { "\t" } else { "  " };

        if let Some(format) = args.size_format() {
            if args.du_compatible {
                match format {
                    SizeFormat::Count => write!(out, "{count}")?,
                    SizeFormat::Bytes => write!(out, "{size}")?,
                    SizeFormat::Human => format_size(size, true, true, out)?,
                    SizeFormat::SI => format_size(size, false, true, out)?,
                    SizeFormat::Blocks(bs) => write!(out, "{}", div_round_up(size, bs))?,
                }
            } else {
                match format {
                    SizeFormat::Count => write!(out, "{count:>16} inodes")?,
                    SizeFormat::Bytes => write!(out, "{size:>16} bytes")?,
                    SizeFormat::Human => format_size(size, true, false, out)?,
                    SizeFormat::SI => format_size(size, false, false, out)?,
                    SizeFormat::Blocks(bs) => write!(out, "{:>12} blocks", div_round_up(size, bs))?,
                }
            }
            out.push_str(sep);
        }

        if args.tree {
            let mut list = self.is_last_child_list;
            let is_last_child = list.pop();
            for depth in 0..list.len {
                out.push_str(if list.get(depth) { "  " } else { "│ " });
            }

            out.push_str(match (is_last_child, self.has_children) {
                (None, false) => "• ",
                (None, true) => "╷ ",
                (Some(false), false) => "├─╴ ",
                (Some(false), true) => "├─╮ ",
                (Some(true), false) => "╰─╴ ",
                (Some(true), true) => "╰─╮ ",
            });
        }

        write!(out, "{}", self.path.display())?;

        if args.tree {
            if let Some(target) = &self.target {
                write!(out, " → {}", target.display())?
            }
        }

        out.push_str(if args.null { "\0" } else { "\n" });

        Ok(())
    }
}

fn format_size(bytes: u64, binary: bool, du_compatible: bool, out: &mut String) -> fmt::Result {
    let mut factor: u64 = 1;
    let mut power = 0;
    let mut result = bytes;
    let kilo = if binary { 1024 } else { 1000 };

    while result >= if du_compatible { 1000 } else { kilo } {
        factor *= kilo;
        power += 1;
        result /= kilo;
    }

    let result_float = bytes as f64 / factor as f64;

    let suffix = if binary && power > 0 { "iB" } else { "B" };
    let prefix = match power {
        0 => "",
        1 if !binary => "k",
        _ => &"KMGTPEZYRQ"[power - 1..][..1],
    };

    if du_compatible {
        if power == 0 {
            write!(out, "{:>3}B", bytes)
        } else {
            // We have only 3 chars of space, so format with one decimal point if <10
            let rounded = (result_float * 10.0).round() / 10.0;
            if rounded < 10.0 {
                write!(out, "{:>3.1}{prefix}", rounded)
            } else {
                write!(out, "{:>3.0}{prefix}", result_float.round())
            }
        }
    } else {
        // "1017.234 KiB".len() == 12
        let num_len = 12 - 1 - prefix.len() - suffix.len();
        let precision = if power == 0 { 0 } else { 3 };
        write!(out, "{result_float:>num_len$.precision$} {prefix}{suffix}")
    }
}
