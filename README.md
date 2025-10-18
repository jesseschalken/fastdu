# fastdu
parallel, cross-platform implementation of `du`

## Summary

`fastdu` is a Rust implementation of GNU `du`, supporting the majority of options and with the following additions:

1. Ability to sort output in ascending (`-s`) or descending (`-r`) order and take the first N entries (eg `-n 100`).
2. Display of progress and average speed while the filesystem is being scanned (can be disabled with `--no-progress`).
3. Parallel execution of filesystem operations using Rayon, with an adjustible thread count (eg `-j 10`).
4. Windows support, including efficient retrieval of file metadata using [`DirEntry::metadata`](https://doc.rust-lang.org/std/fs/struct.DirEntry.html#method.metadata).

## Installation

1. Install [rustup](https://rustup.rs/)
2. `cargo install --git https://github.com/jesseschalken/fastdu`
3. `fastdu -rha -n10 .`

## Usage

```
Usage: fastdu [OPTIONS] [FILES_OR_DIRECTORIES]...

Arguments:
  [FILES_OR_DIRECTORIES]...  Files or directories to scan

Options:
  -A, --apparent-size              Print apparent sizes rather than device usage. This is always true on Windows.
  -D, --dereference-args           Dereference only symlinks that are listed on the command line [aliases: -H]
  -L, --dereference                Dereference all symbolic links
  -b, --bytes                      Print size in bytes
      --blocks                     Print size in 512 byte blocks. This is the default.
  -h, --human-readable             Print sizes in human readable format (KiB, MiB etc)
  -S, --sort                       Sort output (ascending order)
      --si                         Like -h, but use powers of 1000 (KB, MB etc) instead of 1024
      --inodes                     Count inodes instead of size
  -l, --count-links                Count sizes many times if hard linked. No effect on Windows.
  -a, --all                        Write counts for all files, not just directories
  -x, --one-file-system            Skip nodes on different file systems. No effect on Windows.
  -n, --limit <N>                  Show only first N results (after sorting)
  -?, --help                       Print help
  -r, --reverse                    Sort output (descending order)
  -j, --num-threads <NUM_THREADS>  Thread count, defaults to 2x the number of CPU cores
      --no-progress                Disable progress output even if stderr is a terminal
      --du-compatible              Show output in the same format as "du". See also --no-progress.
  -0, --null                       End each line with a null byte instead of newline
  -d, --max-depth <MAX_DEPTH>      Only show entries up to this maximum depth
  -s, --summarize                  Same as --max-depth=0
  -c, --total                      Include a grand total
```
