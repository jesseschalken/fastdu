mod scanner;

use clap::ArgAction;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fs::DirEntry;
use std::io::{ErrorKind, Write, stdout};
use std::path::{Path, PathBuf};
use std::result::Result;
use std::result::Result::Ok;
use std::thread::available_parallelism;
use std::time::Instant;

use clap::{Parser, arg};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::scanner::DirEntryLike;
use crate::scanner::FsNodeState;
use crate::scanner::scan_tree;

type Error = std::io::Error;

#[derive(Debug)]
struct Node {
    path: Box<Path>,
    is_dir: bool,
    size: u64,
    inode: u64,
    device: u64,
    children: Box<[Node]>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    is_dir: bool,
    path: Box<Path>,
}

#[cfg(unix)]
fn create_node<T: DirEntryLike + ?Sized>(
    entry: &mut FsNodeState<T>,
    args: &DuArgs,
) -> Result<Node, Error> {
    use std::os::unix::fs::MetadataExt;
    let (inode, device) = if !args.count_links || args.one_file_system {
        let metadata = entry.metadata()?;
        (metadata.ino(), metadata.dev())
    } else {
        (0, 0)
    };
    Ok(Node {
        path: entry.take_path().into(),
        is_dir: entry.file_type()?.is_dir(),
        size: if args.inodes {
            1
        } else if args.apparent_size || args.bytes {
            entry.metadata()?.len()
        } else {
            entry.metadata()?.blocks() * 512
        },
        inode,
        device,
        children: Default::default(),
    })
}

#[cfg(not(unix))]
fn create_node<T: DirEntryLike + ?Sized>(
    entry: &mut FsNodeState<T>,
    args: &DuArgs,
) -> Result<Node, Error> {
    Node {
        path: entry.take_path().into(),
        children: Default::default(),
        is_dir: entry.file_type()?.is_dir(),
        size: if args.inodes {
            1
        } else {
            entry.metadata()?.len()
        },
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
                children: dedupe_inodes(node.children.into(), seen).into(),
                ..node
            })
        })
        .collect()
}

fn total_count(node: &Node) -> usize {
    1 + node.children.iter().map(total_count).sum::<usize>()
}

#[inline]
fn handle_error<T, E: Display>(result: Result<T, E>) -> Option<T> {
    match result {
        Ok(v) => return Some(v),
        Err(e) => eprintln!("{}", e),
    }
    return None;
}

fn parse_dir(path: &Path, args: &DuArgs, root: &Node) -> Result<Vec<Node>, Error> {
    let handler = |entry: &mut FsNodeState<DirEntry>| {
        let mut node = create_node(entry, args)?;

        if args.one_file_system && node.device != root.device {
            return Ok(None);
        }

        Ok(Some(move |children: Result<Vec<Node>, Error>| {
            node.children = handle_error(children).unwrap_or_default().into();
            Ok(Some(node))
        }))
    };

    scan_tree(path, &handler, args.dereference_all)
}

fn parse(path: PathBuf, args: &DuArgs) -> Result<Node, Error> {
    let mut state = FsNodeState::new(
        path.as_path(),
        args.dereference_args || args.dereference_all,
    );

    let mut node = create_node(&mut state, args)?;

    if node.is_dir {
        node.children = handle_error(parse_dir(&node.path, args, &node))
            .unwrap_or_default()
            .into()
    }

    Ok(node)
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

fn default_num_threads() -> usize {
    available_parallelism()
        .map(|cores| cores.get() * 2)
        .unwrap_or(1)
}

fn main() -> std::io::Result<()> {
    let args: DuArgs = DuArgs::parse();

    ThreadPoolBuilder::new()
        // Include the current thread so with -j1 no threads need to be spawned
        .use_current_thread()
        .num_threads(args.num_threads.unwrap_or_else(default_num_threads))
        .build_global()
        .expect("Failed to set thread pool");

    let start = Instant::now();

    let mut roots: Vec<Node> = args
        .files_or_directories
        .par_iter()
        .with_max_len(1)
        .flat_map_iter(|path| handle_error(parse(path.into(), &args)))
        .collect();

    let end = Instant::now();
    let count: usize = roots.iter().map(total_count).sum();
    let secs = (end - start).as_secs_f64();

    eprintln!(
        "Scanned {} nodes in {:.3} seconds ({:.0} nodes/s)",
        count,
        secs,
        count as f64 / secs
    );

    if !args.count_links && cfg!(unix) {
        roots = dedupe_inodes(roots, &mut HashSet::new());
    }

    let mut items = vec![];
    for root in roots {
        flatten(root, &mut items, None);
    }

    if !args.all {
        items = items.into_iter().filter(|x| x.is_dir).collect();
    }

    if args.reverse {
        items.par_sort_by_key(|x| Reverse(x.size));
    } else if args.sort {
        items.par_sort_by_key(|x| x.size);
    }

    if let Some(limit) = args.limit {
        items = items.into_iter().take(limit).collect();
    }

    items.shrink_to_fit();

    let output: Box<[Box<str>]> = items
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

            format!("{:>18}  {}\n", size, item.path.display()).into()
        })
        .collect();

    drop(items);

    let mut stdout = stdout().lock();
    for line in output {
        let result = stdout.write_all(line.as_bytes());
        // Ignore broken pipe from e.g. pipe into less
        match result {
            Err(e) if e.kind() == ErrorKind::BrokenPipe => break,
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
