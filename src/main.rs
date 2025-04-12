use std::cmp::Reverse;
use std::fmt::Debug;
use std::time::Instant;

use anyhow::*;
use camino::*;
use clap::{arg, Parser, ValueEnum};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

#[derive(Debug)]
struct Node {
    path: Utf8PathBuf,
    size: u64,
    children: Vec<Node>,
}

#[derive(Debug)]
struct FlatNode {
    total_size: u64,
    total_count: u64,
    path: Utf8PathBuf,
}

impl Node {
    fn flatten(self, parent: Option<&mut FlatNode>, output: &mut Vec<FlatNode>) {
        let mut result = FlatNode {
            path: self.path,
            total_size: self.size,
            total_count: 1,
        };

        for child in self.children {
            child.flatten(Some(&mut result), output);
        }

        if let Some(parent) = parent {
            parent.total_count += result.total_count;
            parent.total_size += result.total_size;
        }

        output.push(result);
    }
}

fn parse_dir(path: &Utf8Path) -> Result<Vec<Node>> {
    let entries = path
        .read_dir_utf8()
        .with_context(|| format!("Failed to read directory {}", path))?
        .par_bridge()
        .map(|entry| -> Result<_> {
            let entry =
                entry.with_context(|| format!("Failed to read entry in directory {}", path))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("Failed to get file type of {}", entry.path()))?;
            let size = Some(&entry)
                .filter(|_| file_type.is_file())
                .map(Utf8DirEntry::metadata)
                .transpose()
                .with_context(|| format!("Failed to get size of {}", entry.path()))?
                .map(|x| x.len());

            // We have to drop entry here, otherwise it will hold the
            // directory handle open.
            Ok((entry.into_path(), file_type, size))
        })
        // Collect into a vector so that we drop the directory handle before
        // traversing the children and don't get a "Too many open files" error.
        .collect::<Vec<_>>();

    let nodes = entries
        .into_par_iter()
        .map(|entry| -> Result<_> {
            let (path, file_type, size) = entry?;
            let size = size.unwrap_or(0);
            let children = if file_type.is_dir() {
                parse_dir(&path)?
            } else {
                Vec::new()
            };
            Ok(Node {
                path,
                size,
                children,
            })
        })
        .flat_map(|x| x.inspect_err(|e| eprintln!("{:#}", e)))
        .collect::<Vec<Node>>();

    Ok(nodes)
}

fn parse(path: &Utf8Path) -> Result<Node> {
    let stat = path
        .symlink_metadata()
        .with_context(|| format!("Failed to lstat {}", path))?;
    Ok(Node {
        path: path.to_owned(),
        size: if stat.is_file() { stat.len() } else { 0 },
        children: if stat.is_dir() {
            parse_dir(path)?
        } else {
            Vec::new()
        },
    })
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Format {
    Binary,
    Decimal,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Sort {
    Count,
    Size,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, help = "Show only first N entries")]
    limit: Option<usize>,
    #[arg(short, long, value_enum, default_value_t = Format::Decimal, help = "Use binary (1024/KiB) or decimal (1000/kB) units")]
    units: Format,
    #[arg(short, long, value_enum, default_value_t = Sort::Size, help = "Sort by total size or total node count")]
    sort: Sort,
    #[arg(
        short,
        long,
        default_value_t = false,
        help = "Sort ascending instead of descending"
    )]
    reverse: bool,
    #[arg(short, long, help = "Thread count, defaults to number of CPU cores")]
    threads: Option<usize>,
    #[arg(help = "Files or directories to scan")]
    files_or_directories: Vec<String>,
}

fn main() {
    let args: Args = Args::parse();

    ThreadPoolBuilder::new()
        .num_threads(args.threads.unwrap_or(0))
        .build_global()
        .expect("Failed to set thread pool");

    let start = Instant::now();

    let roots = args
        .files_or_directories
        .par_iter()
        .map(Utf8Path::new)
        .map(parse)
        .flat_map(|x| x.inspect_err(|e| eprintln!("{:#}", e)))
        .collect::<Vec<Node>>();

    eprintln!("Scanned in {:?}", Instant::now() - start);

    let mut items = vec![];
    for root in roots {
        root.flatten(None, &mut items);
    }

    items.sort_by_key(|x| {
        Reverse(match args.sort {
            Sort::Count => x.total_count,
            Sort::Size => x.total_size,
        })
    });

    if args.reverse {
        items.reverse();
    }

    let binary = match args.units {
        Format::Binary => true,
        Format::Decimal => false,
    };

    for item in items.iter().take(args.limit.unwrap_or(usize::MAX)) {
        // 12 chars for the bytes eg "1022.170 MiB"
        println!(
            "{:>12} {:>8} nodes  {}",
            format_bytes(item.total_size as i64, binary),
            item.total_count,
            item.path
        );
    }
}

const BINARY_UNITS: [&str; 11] = [
    "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB",
];

const DECIMAL_UNITS: [&str; 11] = [
    "B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB",
];

fn format_bytes(bytes: i64, binary: bool) -> String {
    let sign = if bytes >= 0 { 1 } else { -1 };
    let kilo = if binary { 1024 } else { 1000 };

    let mut factor = 1;
    let mut power = 0;
    while (bytes * sign) >= (kilo * factor) {
        factor *= kilo;
        power += 1;
    }

    let unit = if binary { BINARY_UNITS } else { DECIMAL_UNITS }[power];
    format!("{:.3} {}", bytes as f64 / factor as f64, unit)
}
