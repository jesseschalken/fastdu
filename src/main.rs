use clap::ArgAction;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fs::Metadata;
use std::io::{ErrorKind, Write, stdout};
use std::result::Result::Ok;
use std::time::Instant;

use anyhow::*;
use camino::*;
use clap::{Parser, arg};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

#[derive(Debug)]
struct Node {
    path: Box<Utf8Path>,
    size: u64,
    is_dir: bool,
    inode: Option<u64>,
    children: Box<[Node]>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    is_dir: bool,
    path: Box<Utf8Path>,
}

impl Node {
    fn new(path: Box<Utf8Path>, metadata: &Metadata, args: &DuArgs) -> Self {
        Node {
            path,
            size: get_size(metadata, args),
            is_dir: metadata.is_dir(),
            inode: get_inode(metadata),
            children: Default::default(),
        }
    }

    fn flatten(
        self,
        parent: Option<&mut FlatNode>,
        output: &mut Vec<FlatNode>,
        args: &DuArgs,
    ) {
        let mut result = FlatNode {
            path: self.path,
            is_dir: self.is_dir,
            size: self.size,
        };

        for child in self.children {
            child.flatten(Some(&mut result), output, args);
        }

        if let Some(parent) = parent {
            parent.size += result.size;
        }

        output.push(result);
    }

    /// to use if -l isn't set
    fn dedupe_inodes(self, seen: &mut HashSet<u64>) -> Option<Node> {
        // Skip this node if it has an inode and its already seen
        if let Some(inode) = self.inode {
            if !seen.insert(inode) {
                return None;
            }
        }

        Some(Node {
            children: self
                .children
                .into_iter()
                .flat_map(|x| x.dedupe_inodes(seen))
                .collect(),
            ..self
        })
    }

    fn total_count(&self) -> usize {
        1 + self.children.iter().map(Node::total_count).sum::<usize>()
    }
}

fn get_size(metadata: &Metadata, args: &DuArgs) -> u64 {
    if args.inodes {
        1
    } else if !args.apparent_size && !args.bytes {
        get_disk_size(&metadata)
    } else if metadata.is_file() {
        metadata.len()
    } else {
        0
    }
}

#[cfg(unix)]
fn get_disk_size(metadata: &Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.blocks() * 512
}

#[cfg(not(unix))]
fn get_disk_size(metadata: &Metadata) -> u64 {
    // No easy way to do this on Windows
    metadata.len()
}

#[cfg(unix)]
fn get_device(metadata: &Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.dev())
}

#[cfg(not(unix))]
fn get_device(_: &Metadata) -> Option<u64> {
    // No easy way to do this on Windows
    None
}

#[cfg(unix)]
fn get_inode(metadata: &Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.ino())
}

#[cfg(not(unix))]
fn get_inode(_: &Metadata) -> Option<u64> {
    // No easy way to do this on Windows
    None
}

fn handle_error<T, E: Display>(x: Result<T, E>) -> Option<T> {
    match x {
        Ok(y) => Some(y),
        Err(e) => {
            eprintln!("{:#}", e);
            None
        }
    }
}

fn parse_dir(
    path: &Utf8Path,
    args: &DuArgs,
    root_dev: Option<u64>,
) -> Result<Box<[Node]>> {
    let nodes = retry_interrupted(|| path.read_dir_utf8())
        .with_context(|| format!("readdir({})", path))?
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map_iter(|entry| -> Option<_> {
            let entry = handle_error(
                entry.with_context(|| format!("readdir({})", path)),
            )?;
            let metadata = handle_error(if args.dereference_all {
                retry_interrupted(|| entry.path().metadata())
                    .with_context(|| format!("stat({})", path))
            } else {
                retry_interrupted(|| entry.metadata())
                    .with_context(|| format!("lstat({})", path))
            })?;

            if args.one_file_system && get_device(&metadata) != root_dev {
                return None;
            }

            let path = entry.into_path();
            let node = Node::new(path.into(), &metadata, args);

            // We have to drop entry here, otherwise it will hold the
            // directory handle open.
            Some(node)
        })
        // Collect into a vector so that we drop the directory handle before
        // traversing the children and don't get a "Too many open files" error.
        .collect::<Vec<Node>>()
        .into_par_iter()
        .map(|mut node| {
            if node.is_dir {
                node.children =
                    handle_error(parse_dir(&node.path, args, root_dev))
                        .unwrap_or_default();
            }
            node
        })
        .collect();

    Ok(nodes)
}

fn retry_interrupted<T>(
    mut f: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    loop {
        match f() {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            x => break x,
        }
    }
}

fn parse(path: Box<Utf8Path>, args: &DuArgs) -> Result<Node> {
    let metadata = if args.dereference_args || args.dereference_all {
        retry_interrupted(|| path.metadata())
            .with_context(|| format!("stat({})", path))?
    } else {
        retry_interrupted(|| path.symlink_metadata())
            .with_context(|| format!("lstat({})", path))?
    };

    let mut node = Node::new(path, &metadata, args);

    if node.is_dir {
        node.children =
            handle_error(parse_dir(&node.path, args, get_device(&metadata)))
                .unwrap_or_default();
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
        help = "Count sizes many times if hard linked"
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
        help = "Skip nodes on different file systems"
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
        help = "Thread count, defaults to number of CPU cores"
    )]
    threads: Option<usize>,

    #[arg(help = "Files or directories to scan")]
    files_or_directories: Vec<String>,
}

fn main() -> Result<()> {
    let args: DuArgs = DuArgs::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.threads.unwrap_or(0))
        .build_global()
        .expect("Failed to set thread pool");

    let start = Instant::now();

    let mut roots: Vec<Node> = args
        .files_or_directories
        .par_iter()
        .flat_map_iter(|path| handle_error(parse(path.into(), &args)))
        .collect();

    let end = Instant::now();
    let count = roots.iter().map(Node::total_count).sum::<usize>();
    let secs = (end - start).as_secs_f64();

    eprintln!(
        "Scanned {} nodes in {:.3} seconds ({:.0} nodes/s)",
        count,
        secs,
        count as f64 / secs
    );

    if !args.count_links {
        let mut seen = HashSet::new();
        roots = roots
            .into_iter()
            .flat_map(|x| x.dedupe_inodes(&mut seen))
            .collect();
    }

    let mut items = vec![];
    for root in roots {
        root.flatten(None, &mut items, &args);
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

    let mut stdout = stdout().lock();
    for item in items {
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

        let result = writeln!(stdout, "{:>18}  {}", size, item.path);

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
