use clap::ArgAction;
use clap::Parser;
use dashmap::DashSet;
use mimalloc::MiMalloc;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::error::Error;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt::{self, Debug, Write};
use std::fs;
use std::fs::Metadata;
use std::io::{self, IsTerminal, Write as _, stderr};
use std::path::{Path, PathBuf};
use std::result::Result::Ok;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::RecvTimeoutError::{Disconnected, Timeout};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::available_parallelism;
use std::time::{Duration, Instant};
use std::{thread, u64};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug)]
struct Node {
    name: Box<OsStr>,
    /// Always 1 if counting inodes
    self_size: u64,
    total_size: u64,
    file_type: fs::FileType,
    children: Box<[Node]>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    path: Box<str>,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy)]
struct InodeKey {
    ino: u64,
    dev: u64,
}

impl InodeKey {
    #[cfg(unix)]
    fn create(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                ino: metadata.ino(),
                dev: metadata.dev(),
            }
        }
        #[cfg(not(unix))]
        {
            Self { ino: 0, dev: 0 }
        }
    }
}

fn os_string_into_string_lossy(s: OsString) -> String {
    s.into_string()
        .unwrap_or_else(|s| s.to_string_lossy().into_owned())
}

type FxDashSet<T> = DashSet<T, FxBuildHasher>;

fn get_blocks(metadata: &Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.blocks() * 512
    }
    #[cfg(not(unix))]
    {
        metadata.len()
    }
}

impl Node {
    fn flatten_tree(
        &self,
        output: &mut Vec<FlatNode>,
        self_prefix: [&str; 2],
        child_prefix: [&str; 2],
        args: &DuArgs,
        depth: usize,
    ) {
        let name = self.name.to_string_lossy();
        let children = self
            .children
            .iter()
            .filter(|child| args.accept(child, depth + 1))
            .collect::<Vec<_>>();

        output.push(FlatNode {
            size: self.total_size,
            path: match (self_prefix, &*children) {
                (["", ""], []) => join_strs(["· ", &name]).into(),
                (["", ""], _) => join_strs(["╷ ", &name]).into(),
                ([a, b], []) => join_strs([a, b, "╴ ", &name]).into(),
                ([a, b], _) => join_strs([a, b, "╮ ", &name]).into(),
            },
        });

        if let [children @ .., last_child] = &*children {
            let prefix = join_strs(child_prefix);

            for child in children {
                child.flatten_tree(output, [&prefix, "├─"], [&prefix, "│ "], args, depth + 1);
            }

            last_child.flatten_tree(output, [&prefix, "╰─"], [&prefix, "  "], args, depth + 1);
        }
    }

    fn flatten_joined(
        &self,
        output: &mut Vec<FlatNode>,
        parent_path: &Path,
        args: &DuArgs,
        depth: usize,
    ) {
        if !args.accept(&self, depth) {
            return;
        }

        let path = fast_path_join(parent_path, &self.name);

        for child in &self.children {
            child.flatten_joined(output, &path, args, depth + 1);
        }

        output.push(FlatNode {
            path: os_string_into_string_lossy(path.into()).into(),
            size: self.total_size,
        });
    }

    fn total_count(&self) -> usize {
        let mut total = 1;
        for node in &self.children {
            total += node.total_count();
        }
        total
    }

    fn read_children(
        path: &Path,
        args: &DuArgs,
        root: &InodeKey,
        output: &dyn Output,
        seen: &FxDashSet<InodeKey>,
    ) -> io::Result<Box<[Node]>> {
        // Build nodes before recursing on subdirectories so we close the
        // directory handle and don't get a "Too many open files" error.

        // opendir() can produce EINTR on macOS when reading dirs in ~/Library/{Group ,}Containers
        let mut nodes = retry_if_interrupted(|| path.read_dir())
            .as_mut()
            .map_err(|e| add_context(e, path))?
            .map(|result| -> io::Result<_> {
                let entry = result.as_ref().map_err(|e| add_context(e, path))?;
                let file_type;
                let size;

                if args.dereference_all || args.one_file_system || !args.count_links || !args.inodes
                {
                    let metadata = if args.dereference_all {
                        entry.path().metadata()
                    } else {
                        entry.metadata()
                    };

                    let metadata = metadata
                        .as_ref()
                        .map_err(|e| add_context(e, &entry.path()))?;

                    if cfg!(unix) {
                        let inode = InodeKey::create(metadata);

                        if args.one_file_system && inode.dev != root.dev {
                            return Ok(None);
                        }

                        if !args.count_links && !seen.insert(inode) {
                            return Ok(None);
                        }
                    }

                    file_type = metadata.file_type();
                    size = if args.inodes {
                        1
                    } else if args.use_apparent_size() {
                        metadata.len()
                    } else {
                        get_blocks(metadata)
                    };
                } else {
                    file_type = entry
                        .file_type()
                        .map_err(|e| add_context(&e, &entry.path()))?;
                    size = 1;
                }

                Ok(Some(Node {
                    name: entry.file_name().into(),
                    self_size: size,
                    total_size: size,
                    file_type,
                    children: Default::default(),
                }))
            })
            .flat_map(|result| result.inspect_err(|e| output.log_error(e)))
            .flatten()
            .collect::<Box<_>>();

        output.add_total(nodes.len());

        nodes
            .iter_mut()
            .filter(|node| node.is_dir())
            .collect::<Vec<_>>()
            .into_par_iter()
            .with_max_len(1)
            .for_each(|node| {
                let path = fast_path_join(path, &node.name);

                node.children = Node::read_children(&path, args, root, output, seen)
                    .inspect_err(|e| output.log_error(e))
                    .unwrap_or_default();

                node.recalculate_total_size();
            });

        Ok(nodes)
    }

    fn read(
        path: PathBuf,
        args: &DuArgs,
        output: &dyn Output,
        seen: &FxDashSet<InodeKey>,
    ) -> io::Result<Option<Node>> {
        let metadata = if args.dereference_args || args.dereference_all {
            path.metadata()
        } else {
            path.symlink_metadata()
        };
        let metadata = metadata.as_ref().map_err(|e| add_context(e, &path))?;

        let inode = InodeKey::create(metadata);

        if cfg!(unix) && !seen.insert(inode) {
            return Ok(None);
        }

        let size = if args.inodes {
            1
        } else if args.use_apparent_size() {
            metadata.len()
        } else {
            get_blocks(metadata)
        };

        let mut node = Node {
            name: OsString::from(path).into(),
            self_size: size,
            total_size: size,
            file_type: metadata.file_type(),
            children: Default::default(),
        };

        if metadata.is_dir() {
            let path = Path::new(&node.name);
            node.children = Node::read_children(path, args, &inode, output, seen)?;
            node.recalculate_total_size();
        }

        output.add_total(1);

        Ok(Some(node))
    }

    fn is_dir(&self) -> bool {
        self.file_type.is_dir()
    }

    fn recalculate_total_size(&mut self) {
        self.total_size = self
            .children
            .iter()
            .map(|child| child.total_size)
            .sum::<u64>()
            + self.self_size;
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
        help = "Print apparent sizes rather than device usage. This is always true on Windows."
    )]
    apparent_size: bool,

    #[arg(
        short = 'D',
        visible_short_alias = 'H',
        long = "dereference-args",
        help = "Dereference only symlinks that are listed on the command line"
    )]
    dereference_args: bool,

    #[arg(
        short = 'L',
        long = "dereference",
        help = "Dereference all symbolic links"
    )]
    dereference_all: bool,

    #[arg(
        short = 'b',
        long = "bytes",
        help = "Print size in bytes. Implies --apparent-size unles --blocks is also passed."
    )]
    bytes: bool,

    #[arg(short = None, long = "blocks", help = "Print size in 512 byte blocks. This is the default.")]
    blocks: bool,

    #[arg(
        short = 'h',
        long = "human-readable",
        help = "Print sizes in human readable format (KiB, MiB etc)"
    )]
    human: bool,

    #[arg(short = 'S', long = "sort", help = "Sort output (ascending order)")]
    sort: bool,

    #[arg(
        short = None,
        long = "si",
        help = "Like -h, but use powers of 1000 (KB, MB etc) instead of 1024"
    )]
    si: bool,

    #[arg(short = 'i', long = "inodes", help = "Count inodes instead of size")]
    inodes: bool,

    #[arg(
        short = 'l',
        long = "count-links",
        help = "Count sizes many times if hard linked. No effect on Windows."
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
        short = 'r',
        long = "reverse",
        default_value_t = false,
        help = "Sort output (descending order)"
    )]
    reverse: bool,

    #[arg(
        short = 't',
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

    #[arg(
        long = "no-progress",
        help = "Disable progress output even if stderr is a terminal"
    )]
    no_progress: bool,

    #[arg(help = "Files or directories to scan")]
    files_or_directories: Vec<String>,

    #[arg(
        long = "du-compatible",
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
        help = "Only show entries up to this maximum depth"
    )]
    max_depth: Option<usize>,

    #[arg(short = 's', long = "summarize", help = "Same as --max-depth=0")]
    summarize: bool,

    #[arg(short = 'c', long = "total", help = "Include a grand total")]
    show_total: bool,
}

impl DuArgs {
    fn output_format(&self) -> OutputFormat {
        if self.inodes {
            OutputFormat::Count
        } else if self.bytes {
            OutputFormat::Bytes
        } else if self.si {
            OutputFormat::SI
        } else if self.human {
            OutputFormat::Human
        } else {
            OutputFormat::Blocks
        }
    }

    fn use_apparent_size(&self) -> bool {
        self.apparent_size || (self.bytes && !self.blocks)
    }

    fn with_output<T>(&self, body: impl FnOnce(&dyn Output) -> T) -> T {
        if self.no_progress || !stderr().is_terminal() {
            return body(&SimpleOutput);
        }

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

    fn accept(&self, node: &Node, depth: usize) -> bool {
        if !self.all && !node.is_dir() {
            return false;
        }

        if depth > 0 && self.summarize {
            return false;
        }

        if let Some(max_depth) = self.max_depth {
            if depth > max_depth {
                return false;
            }
        }

        return true;
    }
}

fn join_strs<const N: usize>(strs: [&str; N]) -> Cow<'_, str> {
    match &strs as &[&str] {
        [] => "".into(),
        [s] | [s, ""] | ["", s] => (*s).into(),
        _ => {
            let mut result = String::with_capacity(strs.iter().copied().map(str::len).sum());
            for s in strs {
                result.push_str(s);
            }
            result.into()
        }
    }
}

enum OutputFormat {
    Count,
    Bytes,
    Human,
    SI,
    Blocks,
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

fn retry_if_interrupted<T>(mut f: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    loop {
        let result = f();
        match &result {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            _ => return result,
        }
    }
}

fn main() -> std::io::Result<()> {
    let args: DuArgs = DuArgs::parse();

    ThreadPoolBuilder::new()
        .num_threads(
            args.num_threads
                .or_else(|| available_parallelism().ok().map(|x| x.get() * 2))
                .unwrap_or(0),
        )
        .build_global()
        .expect("Failed to set thread pool");

    let seen = FxDashSet::default();
    let roots: Vec<Node> = args.with_output(|output| {
        args.files_or_directories
            .par_iter()
            .with_max_len(1)
            .flat_map_iter(|path| {
                Node::read(path.into(), &args, output, &seen)
                    .inspect_err(|e| output.log_error(e))
                    .ok()
                    .flatten()
            })
            .collect()
    });

    let mut count = roots.iter().map(Node::total_count).sum();

    let mut total = FlatNode {
        path: "total".into(),
        size: 0,
    };

    if args.show_total {
        count += 1;
    }

    let mut items = Vec::with_capacity(count);

    for root in roots {
        total.size += root.total_size;

        if args.accept(&root, 0) {
            if args.tree {
                root.flatten_tree(&mut items, ["", ""], ["", ""], &args, 0);
            } else {
                root.flatten_joined(&mut items, Path::new(""), &args, 0);
            }
        }
    }

    if args.show_total {
        items.push(total);
    }

    if !args.tree {
        if args.reverse {
            items.par_sort_unstable_by_key(|x| Reverse(x.size));
        } else if args.sort {
            items.par_sort_unstable_by_key(|x| x.size);
        }
    }

    if let Some(limit) = args.limit {
        items.truncate(limit);
    }

    let mut lines = Vec::with_capacity(items.len());

    items
        .into_par_iter()
        .map(|item| {
            let mut buffer = String::with_capacity(item.path.len() + 32);
            item.format_line(&args, &mut buffer).unwrap();
            buffer
        })
        .collect_into_vec(&mut lines);

    let mut stdout = io::stdout().lock();
    for line in lines {
        let result = stdout.write_all(line.as_bytes());
        // Ignore broken pipe from e.g. pipe into less
        match result {
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => break,
            x => x?,
        }
    }

    Ok(())
}

impl FlatNode {
    fn format_line(&self, args: &DuArgs, out: &mut dyn Write) -> fmt::Result {
        let format = args.output_format();
        let newline = if args.null { '\0' } else { '\n' };
        let size = self.size;
        let blocks = size / 512;

        if args.du_compatible {
            match format {
                OutputFormat::Count => write!(out, "{size}"),
                OutputFormat::Bytes => write!(out, "{size}"),
                OutputFormat::Human => format_size(size, true, true, out),
                OutputFormat::SI => format_size(size, false, true, out),
                OutputFormat::Blocks => write!(out, "{blocks}"),
            }
        } else {
            match format {
                OutputFormat::Count => write!(out, "{size:>16} inodes"),
                OutputFormat::Bytes => write!(out, "{size:>16} bytes"),
                OutputFormat::Human => format_size(size, true, false, out),
                OutputFormat::SI => format_size(size, false, false, out),
                OutputFormat::Blocks => write!(out, "{blocks:>16} blocks"),
            }
        }?;

        if args.du_compatible {
            write!(out, "\t{}{newline}", self.path)
        } else {
            write!(out, "  {}{newline}", self.path)
        }
    }
}

fn format_size(bytes: u64, binary: bool, du_compatible: bool, out: &mut dyn Write) -> fmt::Result {
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
