use clap::ArgAction;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt::Debug;
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

use clap::Parser;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

#[derive(Debug)]
struct Node {
    name: Box<OsStr>,
    is_dir: bool,
    size: u64,
    inode: u64,
    device: u64,
    children: Box<[Node]>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    count: usize,
    is_dir: bool,
    path: Box<Path>,
    depth: usize,
}

impl Node {
    #[cfg(unix)]
    fn new(name: Box<OsStr>, metadata: &Metadata, args: &DuArgs) -> Node {
        use std::os::unix::fs::MetadataExt;
        Node {
            name,
            is_dir: metadata.is_dir(),
            size: if args.use_apparent_size() {
                metadata.len()
            } else {
                metadata.blocks() * 512
            },
            inode: metadata.ino(),
            device: metadata.dev(),
            children: Default::default(),
        }
    }

    #[cfg(not(unix))]
    fn new(name: Box<OsStr>, metadata: &Metadata, _args: &DuArgs) -> Node {
        Node {
            name,
            children: Default::default(),
            is_dir: metadata.is_dir(),
            size: metadata.len(),
            inode: 0,
            device: 0,
        }
    }

    fn flatten(
        self,
        output: &mut Vec<FlatNode>,
        parent: &mut FlatNode,
        seen: &mut Option<HashSet<(u64, u64)>>,
        depth: usize,
    ) {
        if let Some(seen) = seen {
            if !seen.insert((self.device, self.inode)) {
                return;
            }
        }

        let mut result = FlatNode {
            path: fast_path_join(&parent.path, &self.name),
            is_dir: self.is_dir,
            size: self.size,
            count: 1,
            depth,
        };

        for child in self.children {
            child.flatten(output, &mut result, seen, depth + 1);
        }

        parent.size += result.size;
        parent.count += result.count;

        output.push(result);
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
        root: &Node,
        output: &dyn Output,
    ) -> io::Result<Box<[Node]>> {
        // Build nodes before recursing on subdirectories so we close the
        // directory handle and don't get a "Too many open files" error.

        // opendir() can produce EINTR on macOS when reading dirs in ~/Library/{Group ,}Containers
        let mut nodes = retry_if_interrupted(|| path.read_dir())
            .as_mut()
            .map_err(|e| add_context(e, path))?
            .map(|result| -> io::Result<_> {
                let entry = result.as_ref().map_err(|e| add_context(e, path))?;

                let metadata = if args.dereference_all {
                    entry.path().metadata()
                } else {
                    entry.metadata()
                };

                let metadata = metadata
                    .as_ref()
                    .map_err(|e| add_context(e, &entry.path()))?;

                let name = entry.file_name();

                assert_eq!(
                    name.len(),
                    name.capacity(),
                    "DirEntry::file_name allocated extra capacity"
                );

                let node = Node::new(name.into(), metadata, args);

                if args.one_file_system && node.device != root.device {
                    return Ok(None);
                }

                Ok(Some(node))
            })
            .flat_map(|result| result.inspect_err(|e| output.log_error(e)))
            .flatten()
            .collect::<Box<_>>();

        output.add_total(nodes.len());

        nodes
            .iter_mut()
            .filter(|node| node.is_dir)
            .collect::<Vec<_>>()
            .into_par_iter()
            .with_max_len(1)
            .for_each(|node| {
                node.children =
                    Node::read_children(&fast_path_join(path, &node.name), args, root, output)
                        .inspect_err(|e| output.log_error(e))
                        .unwrap_or_default()
            });

        Ok(nodes)
    }

    fn read(path: PathBuf, args: &DuArgs, output: &dyn Output) -> io::Result<Node> {
        let metadata = if args.dereference_args || args.dereference_all {
            path.metadata()
        } else {
            path.symlink_metadata()
        };
        let metadata = metadata.as_ref().map_err(|e| add_context(e, &path))?;
        let mut node = Node::new(OsString::from(path).into(), metadata, args);

        if node.is_dir {
            node.children = Node::read_children(Path::new(&node.name), args, &node, output)?
        }

        output.add_total(1);

        Ok(node)
    }
}

fn fast_path_join(path: &Path, name: &OsStr) -> Box<Path> {
    // Assume the result will be {path}/{name} so that in most
    // cases .into_boxed_path() doesn't reallocate.
    let mut result = OsString::with_capacity(path.as_os_str().len() + 1 + name.len());
    // Push first onto an OsString to skip the PathBuf::push logic.
    result.push(path);
    let mut result = PathBuf::from(result);
    result.push(name);
    result.into_boxed_path()
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

    #[arg(
        short = None,
        long = "inodes",
        help = "Count inodes instead of size"
    )]
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

    fn max_depth(&self) -> Option<usize> {
        if self.summarize {
            Some(0)
        } else {
            self.max_depth
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

    let roots: Vec<Node> = args.with_output(|output| {
        args.files_or_directories
            .par_iter()
            .with_max_len(1)
            .flat_map_iter(|path| {
                Node::read(path.into(), &args, output).inspect_err(|e| output.log_error(e))
            })
            .collect()
    });

    let mut count = roots.iter().map(Node::total_count).sum();

    let mut total = FlatNode {
        count: 0,
        is_dir: true,
        path: PathBuf::new().into(),
        size: 0,
        depth: 0,
    };

    if args.show_total {
        count += 1;
    }

    let mut items = Vec::with_capacity(count);

    let mut seen = if !args.count_links && cfg!(unix) {
        Some(HashSet::with_capacity(count))
    } else {
        None
    };

    for root in roots {
        root.flatten(&mut items, &mut total, &mut seen, 0);
    }

    if let Some(max_depth) = args.max_depth() {
        items.retain(|x| x.depth <= max_depth);
    }

    if args.show_total {
        total.path = PathBuf::from("total").into();
        items.push(total);
    }

    if !args.all {
        items.retain(|x| x.is_dir);
    }

    if args.reverse {
        if args.inodes {
            items.par_sort_unstable_by_key(|x| Reverse(x.count));
        } else {
            items.par_sort_unstable_by_key(|x| Reverse(x.size));
        }
    } else if args.sort {
        if args.inodes {
            items.par_sort_unstable_by_key(|x| x.count);
        } else {
            items.par_sort_unstable_by_key(|x| x.size);
        }
    }

    if let Some(limit) = args.limit {
        items.truncate(limit);
    }

    let mut lines = Vec::with_capacity(items.len());

    items
        .into_par_iter()
        .map(|item| item.format_line(&args))
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
    fn format_line(&self, args: &DuArgs) -> String {
        let format = args.output_format();
        let newline = if args.null { '\0' } else { '\n' };
        let count = self.count;
        let size = self.size;
        let blocks = size / 512;

        let size_string = if args.du_compatible {
            match format {
                OutputFormat::Count => format!("{count}"),
                OutputFormat::Bytes => format!("{size}"),
                OutputFormat::Human => format_bytes(size, true, true),
                OutputFormat::SI => format_bytes(size, false, true),
                OutputFormat::Blocks => format!("{blocks}"),
            }
        } else {
            match format {
                OutputFormat::Count => format!("{count} inodes"),
                OutputFormat::Bytes => format!("{size} bytes"),
                OutputFormat::Human => format_bytes(size, true, false),
                OutputFormat::SI => format_bytes(size, false, false),
                OutputFormat::Blocks => format!("{blocks} blocks"),
            }
        };

        if args.du_compatible {
            format!("{size_string}\t{}{newline}", self.path.display())
        } else {
            format!("{size_string:>18}  {}{newline}", self.path.display())
        }
    }
}

fn format_bytes(bytes: u64, binary: bool, du_compatible: bool) -> String {
    let mut factor: u64 = 1;
    let mut power = 0;
    let mut result = bytes as f64;

    while result >= 1000.0 {
        factor *= if binary { 1024 } else { 1000 };
        power += 1;
        result = bytes as f64 / factor as f64;
    }

    let suffix = if binary { "iB" } else { "B" };
    let prefix = match power {
        0 => "",
        1 if !binary => "k",
        _ => &"KMGTPEZYRQ"[power - 1..][..1],
    };

    match power {
        0 if du_compatible => format!("{:>3}B", bytes),
        _ if du_compatible => format!("{:>3}{}", format_float_du_compat(result), prefix),
        0 => format!("{} B", bytes),
        _ => format!("{:.3} {}{}", result, prefix, suffix),
    }
}

fn format_float_du_compat(num: f64) -> String {
    // We have only 3 chars of space, so format with one decimal point if <10
    let rounded = (num * 10.0).round() / 10.0;
    if rounded < 10.0 {
        format!("{:.1}", rounded)
    } else {
        format!("{:.0}", num.round())
    }
}
