use clap::ArgAction;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fs::Metadata;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::result::Result::Ok;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::RecvTimeoutError::{Disconnected, Timeout};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::available_parallelism;
use std::time::{Duration, Instant};
use std::{result, thread};

use clap::{Parser, arg};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

#[derive(Debug)]
struct Node {
    path: PathBuf,
    is_dir: bool,
    size: u64,
    inode: u64,
    device: u64,
    children: Vec<Node>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    is_dir: bool,
    path: PathBuf,
}

#[cfg(unix)]
fn create_node(path: PathBuf, metadata: &Metadata, args: &DuArgs) -> Node {
    use std::os::unix::fs::MetadataExt;
    Node {
        path,
        is_dir: metadata.is_dir(),
        size: if args.inodes {
            1
        } else if args.apparent_size || args.bytes {
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
fn create_node(path: PathBuf, metadata: &Metadata, args: &DuArgs) -> Node {
    Node {
        path,
        children: Default::default(),
        is_dir: metadata.is_dir(),
        size: if args.inodes { 1 } else { metadata.len() },
        inode: 0,
        device: 0,
    }
}

fn flatten(node: Node, output: &mut Vec<FlatNode>, parent: Option<&mut FlatNode>) {
    let mut result = FlatNode {
        path: node.path,
        is_dir: node.is_dir,
        size: node.size,
    };

    for child in node.children {
        flatten(child, output, Some(&mut result));
    }

    if let Some(parent) = parent {
        parent.size += result.size;
    }

    output.push(result);
}

/// to use if -l isn't set
fn dedupe_inodes(nodes: Vec<Node>, seen: &mut HashSet<(u64, u64)>) -> Vec<Node> {
    nodes
        .into_iter()
        .flat_map(|node| {
            if !seen.insert((node.device, node.inode)) {
                return None;
            }
            Some(Node {
                children: dedupe_inodes(node.children, seen),
                ..node
            })
        })
        .collect()
}

fn retry_if_interrupted<T>(mut f: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    loop {
        let result = f();
        if let Err(e) = &result
            && e.kind() == io::ErrorKind::Interrupted
        {
            continue;
        }
        return result;
    }
}

fn parse_dir(path: &Path, args: &DuArgs, root: &Node, output: &Output) -> io::Result<Vec<Node>> {
    let mut nodes = Vec::new();

    for entry in add_context(path.read_dir().as_mut(), path)? {
        let entry = add_context(entry.as_ref(), path)?;
        let path = entry.path();
        let metadata = retry_if_interrupted(|| {
            if args.dereference_all {
                path.metadata()
            } else if cfg!(target_vendor = "apple") {
                path.symlink_metadata()
            } else {
                entry.metadata()
            }
        });
        let metadata = add_context(metadata.as_ref(), &path)?;
        nodes.push(create_node(path, metadata, args));
    }

    output.add_total(nodes.len());

    if args.one_file_system {
        nodes.retain(|node| node.device == root.device);
    }

    nodes
        .iter_mut()
        .filter(|node| node.is_dir)
        .collect::<Vec<_>>()
        .into_par_iter()
        .with_max_len(1)
        .for_each(|node| {
            node.children = output
                .handle_error(retry_if_interrupted(|| {
                    parse_dir(&node.path, args, root, output)
                }))
                .unwrap_or_default()
        });

    Ok(nodes)
}

fn parse(path: PathBuf, args: &DuArgs, output: &Output) -> io::Result<Node> {
    let metadata = if args.dereference_args || args.dereference_all {
        path.metadata()
    } else {
        path.symlink_metadata()
    };
    let metadata = add_context(metadata.as_ref(), &path)?;
    let mut node = create_node(path, metadata, args);

    if node.is_dir {
        node.children = output
            .handle_error(parse_dir(&node.path, args, &node, output))
            .unwrap_or_default()
    }

    output.add_total(1);

    Ok(node)
}

fn add_context<T, E: Borrow<io::Error>>(result: Result<T, E>, p: &Path) -> io::Result<T> {
    match result {
        Ok(x) => Ok(x),
        Err(e) => {
            let e = e.borrow();
            Err(io::Error::new(e.kind(), format!("{}: {}", p.display(), e)))
        }
    }
}

#[derive(Parser)]
#[command(disable_help_flag = true)]
struct DuArgs {
    #[arg(
        short = 'A',
        long = "apparent-size",
        help = "Print apparent sizes rather than device usage"
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

    #[arg(short = 'b', long = "bytes", help = "Print size in bytes")]
    bytes: bool,

    #[arg(short = None, long = "blocks", help = "Print size in 512 byte blocks. This is the default.")]
    blocks: bool,

    #[arg(
        short = 'h',
        long = "human-readable",
        help = "Print sizes in human readable format (KiB, MiB etc)."
    )]
    human: bool,

    #[arg(short = 'S', long = "sort", help = "Sort output (ascending order)")]
    sort: bool,

    #[arg(
        short = None,
        long = "si",
        help = "Like -h, but use powers of 1000 (KB, MB etc) not 1024"
    )]
    si: bool,

    #[arg(
        short = None,
        long = "inodes",
        help = "Count inodes instead of size."
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
        long = "threads",
        help = "Thread count, defaults to 2x the number of CPU cores"
    )]
    num_threads: Option<usize>,

    #[arg(help = "Files or directories to scan")]
    files_or_directories: Vec<String>,
}

struct Output<'a> {
    ui_wakeups: Sender<()>,
    total_count: &'a AtomicUsize,
}

impl Output<'_> {
    fn add_total(&self, num: usize) {
        self.total_count.fetch_add(num, Relaxed);
    }

    fn handle_error<T, E: Display>(&self, result: result::Result<T, E>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{CLEAR_LINE}{e}");
                let _ = self.ui_wakeups.send(());
                None
            }
        }
    }
}

fn ui_thread(total_count: &AtomicUsize, wakups: Receiver<()>) {
    let start = Instant::now();
    let mut next_due = start;
    let mut stop = false;
    loop {
        let count = total_count.load(Relaxed);
        let secs = (Instant::now() - start).as_secs_f64();
        let line = format!(
            "Scanned {count} nodes in {secs:.3} seconds (avg. {:.0} nodes/s)",
            count as f64 / secs
        );
        eprint!("{CLEAR_LINE}{line}");
        if stop {
            break
        }
        next_due += Duration::from_millis(1000);
        while !stop {
            match wakups.recv_timeout(next_due - Instant::now()) {
                Ok(()) => eprint!("{CLEAR_LINE}{line}"),
                Err(Timeout) => break,
                Err(Disconnected) => stop = true,
            }
        }
    }
    eprintln!();
}

const CLEAR_LINE: &str = "\x1B[2K\r";

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

    let total_count = &AtomicUsize::new(0);

    let mut roots = thread::scope(|scope| {
        let (sender, receiver) = channel();

        scope.spawn(|| ui_thread(total_count, receiver));

        let output = Output {
            ui_wakeups: sender,
            total_count,
        };

        args.files_or_directories
            .par_iter()
            .with_max_len(1)
            .flat_map_iter(|path| {
                output.handle_error(retry_if_interrupted(|| {
                    parse(PathBuf::from(path), &args, &output)
                }))
            })
            .collect()
    });

    if !args.count_links && cfg!(unix) {
        roots = dedupe_inodes(roots, &mut HashSet::new());
    }

    let mut items = vec![];
    for root in roots {
        flatten(root, &mut items, None);
    }

    if !args.all {
        items.retain(|x| x.is_dir);
    }

    if args.reverse {
        items.par_sort_by_key(|x| Reverse(x.size));
    } else if args.sort {
        items.par_sort_by_key(|x| x.size);
    }

    if let Some(limit) = args.limit {
        items = items.into_iter().take(limit).collect();
    }

    let output: Vec<String> = items
        .par_iter()
        .map(|item| {
            let size = if args.inodes {
                format!("{} inodes", item.size)
            } else if args.bytes {
                format!("{} bytes", item.size)
            } else if args.si {
                format_bytes(item.size, false)
            } else if args.human {
                format_bytes(item.size, true)
            } else {
                format!("{} blocks", item.size / 512)
            };

            format!("{:>18}  {}\n", size, item.path.display())
        })
        .collect();

    drop(items);

    let mut stdout = io::stdout().lock();
    for line in output {
        let result = stdout.write_all(line.as_bytes());
        // Ignore broken pipe from e.g. pipe into less
        match result {
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => break,
            x => x?,
        }
    }

    Ok(())
}

fn format_bytes(bytes: u64, binary: bool) -> String {
    let kilo = if binary { 1024 } else { 1000 };

    let mut factor = 1;
    let mut power = 0;
    while bytes >= (kilo * factor) {
        factor *= kilo;
        power += 1;
    }

    match power {
        0 => format!("{} B", bytes),
        1 => format!("{} {}", bytes, if binary { "KiB" } else { "kB" }),
        _ => format!(
            "{:.3} {}{}",
            bytes as f64 / factor as f64,
            b" KMGTPEZYRQ"[power] as char,
            if binary { "iB" } else { "B" },
        ),
    }
}
