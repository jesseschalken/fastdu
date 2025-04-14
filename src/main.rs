use clap::ArgAction;
use std::cmp::Reverse;
use std::fmt::{Debug, Display};
use std::fs::Metadata;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::result::Result::Ok;
use std::time::Instant;

use anyhow::*;
use camino::*;
use clap::{arg, Parser};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

#[derive(Debug)]
struct Node {
    path: Utf8PathBuf,
    metadata: Metadata,
    children: Vec<Node>,
}

#[derive(Debug)]
struct FlatNode {
    size: u64,
    metadata: Metadata,
    path: Utf8PathBuf,
}

impl Node {
    fn flatten(
        self,
        parent: Option<&mut FlatNode>,
        output: &mut Vec<FlatNode>,
        args: &DuArgs,
    ) {
        let size = get_size(&self.metadata, args);

        let mut result = FlatNode {
            path: self.path,
            metadata: self.metadata,
            size,
        };

        for child in self.children {
            child.flatten(Some(&mut result), output, args);
        }

        if let Some(parent) = parent {
            parent.size += result.size;
        }

        output.push(result);
    }
}

fn get_size(metadata: &Metadata, args: &DuArgs) -> u64 {
    if args.count {
        1
    } else if !args.apparent_size {
        get_disk_size(&metadata)
    } else if metadata.is_file() {
        metadata.len()
    } else {
        0
    }
}

#[cfg(unix)]
fn get_disk_size(metadata: &Metadata) -> u64 {
    metadata.blocks() * 512
}

#[cfg(not(unix))]
fn get_disk_size(metadata: &Metadata) -> u64 {
    // No easy way to do this on Windows
    metadata.len()
}

#[cfg(unix)]
fn get_device(metadata: &Metadata) -> Option<u64> {
    Some(metadata.dev())
}

#[cfg(not(unix))]
fn get_device(_: &Metadata) -> Option<u64> {
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
    root: &Metadata,
) -> Result<Vec<Node>> {
    let entries = path
        .read_dir_utf8()
        .with_context(|| format!("Failed to read directory {}", path))?
        .par_bridge()
        .map(|entry| -> Result<_> {
            let entry = entry.with_context(|| {
                format!("Failed to read entry in directory {}", path)
            })?;
            let metadata = if args.dereference_all {
                entry
                    .path()
                    .metadata()
                    .with_context(|| format!("Failed to stat {}", path))?
            } else {
                entry
                    .metadata()
                    .with_context(|| format!("Failed to lstat {}", path))?
            };

            // We have to drop entry here, otherwise it will hold the
            // directory handle open.
            Ok((entry.into_path(), metadata))
        })
        .flat_map(handle_error)
        .filter(|(_, metadata)| {
            if args.one_file_system {
                get_device(metadata) == get_device(root)
            } else {
                true
            }
        })
        // Collect into a vector so that we drop the directory handle before
        // traversing the children and don't get a "Too many open files" error.
        .collect::<Vec<_>>();

    let nodes = entries
        .into_par_iter()
        .map(|(path, metadata)| -> Result<_> {
            let children = if metadata.is_dir() {
                parse_dir(&path, args, root)?
            } else {
                Vec::new()
            };
            Ok(Node {
                path,
                metadata,
                children,
            })
        })
        .flat_map(handle_error)
        .collect::<Vec<Node>>();

    Ok(nodes)
}

fn parse(path: &Utf8Path, args: &DuArgs) -> Result<Node> {
    let metadata = if args.dereference_args || args.dereference_all {
        path.metadata()
            .with_context(|| format!("Failed to stat {}", path))?
    } else {
        path.symlink_metadata()
            .with_context(|| format!("Failed to lstat {}", path))?
    };

    let children = if metadata.is_dir() {
        parse_dir(path, args, &metadata)?
    } else {
        Vec::new()
    };

    Ok(Node {
        path: path.to_owned(),
        metadata,
        children,
    })
}

#[derive(Parser)]
#[command(
    disable_help_flag = true,
    about = "Parallel, cross-platform version of the 'du' utility"
)]
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

    #[arg(
        short = 'b',
        long = "bytes",
        help = "Print size in bytes",
        group = "format"
    )]
    bytes: bool,

    #[arg(
        short = 'h',
        long = "human-readable",
        help = "Print sizes in human readable format (KiB, MiB etc). This is the default.",
        group = "format"
    )]
    human: bool,

    #[arg(
        long = "si",
        help = "Like -h, but use powers of 1000 (KB, MB etc) not 1024",
        group = "format"
    )]
    si: bool,

    #[arg(
        short = 'c',
        long = "count",
        help = "Count nodes instead of size. Note hard links to the same inode are counted twice.",
        group = "format",
        visible_alias = "count"
    )]
    count: bool,

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
        help = "Show only first N results (after sorting)"
    )]
    limit: Option<usize>,

    #[arg(short = '?', long = "help", action = ArgAction::Help, help = "Print help")]
    help: (),

    #[arg(
        short = 'r',
        long = "reverse",
        default_value_t = false,
        help = "Sort descending instead of ascending"
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

fn main() {
    let args: DuArgs = DuArgs::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.threads.unwrap_or(0))
        .build_global()
        .expect("Failed to set thread pool");

    let start = Instant::now();
    let roots = args
        .files_or_directories
        .par_iter()
        .map(Utf8Path::new)
        .map(|p| parse(p, &args))
        .flat_map(|x| x.inspect_err(|e| eprintln!("{:#}", e)))
        .collect::<Vec<Node>>();
    eprintln!("Scanned in {:?}", Instant::now() - start);

    let start = Instant::now();
    let mut items = vec![];
    for root in roots {
        root.flatten(None, &mut items, &args);
    }
    eprintln!("Aggregated in {:?}", Instant::now() - start);

    if !args.all {
        items = items.into_iter().filter(|x| x.metadata.is_dir()).collect();
    }

    let start = Instant::now();
    if args.reverse {
        items.par_sort_unstable_by_key(|x| x.size);
    } else {
        items.par_sort_unstable_by_key(|x| Reverse(x.size));
    }
    eprintln!("Sorted in {:?}", Instant::now() - start);

    if let Some(limit) = args.limit {
        items = items.into_iter().take(limit).collect();
    }

    for item in items {
        let size = if args.bytes {
            format!("{:>16} bytes", item.size)
        } else if args.si {
            format_bytes(item.size, false)
        } else if args.count {
            format!("{:>8} nodes", item.size)
        } else {
            format_bytes(item.size, true)
        };
        // 12 chars for the bytes eg "1022.170 MiB"
        println!("{:>12}  {}", size, item.path);
    }
}

const BINARY_UNITS: [&str; 11] = [
    "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB",
];

const DECIMAL_UNITS: [&str; 11] = [
    "B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB",
];

fn format_bytes(bytes: u64, binary: bool) -> String {
    let kilo = if binary { 1024 } else { 1000 };

    let mut factor = 1;
    let mut power = 0;
    while bytes >= (kilo * factor) {
        factor *= kilo;
        power += 1;
    }

    let unit = if binary { BINARY_UNITS } else { DECIMAL_UNITS }[power];
    format!("{:.3} {}", bytes as f64 / factor as f64, unit)
}
