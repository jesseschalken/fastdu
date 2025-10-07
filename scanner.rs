use rayon::prelude::*;
use std::borrow::Borrow;
use std::cell::OnceCell;
use std::fs::FileType;
use std::fs::Metadata;
use std::io::Error;
use std::io::Result;
use std::path::PathBuf;
use std::{fs::DirEntry, path::Path};

pub trait DirEntryLike {
    fn path(&self) -> PathBuf;
    fn metadata(&self) -> Result<Metadata>;
    fn file_type(&self) -> Option<Result<FileType>> {
        None
    }
}

impl DirEntryLike for Path {
    fn path(&self) -> PathBuf {
        self.to_path_buf()
    }

    fn metadata(&self) -> Result<Metadata> {
        self.symlink_metadata()
    }
}

impl DirEntryLike for DirEntry {
    fn path(&self) -> PathBuf {
        self.path()
    }

    fn metadata(&self) -> Result<Metadata> {
        self.metadata()
    }

    fn file_type(&self) -> Option<Result<FileType>> {
        Some(self.file_type())
    }
}

pub struct FsNodeState<'a, T: DirEntryLike + ?Sized> {
    entry: &'a T,
    follow_links: bool,
    path: OnceCell<PathBuf>,
    file_type: OnceCell<Result<FileType>>,
    metadata: OnceCell<Result<Metadata>>,
}

impl<'a, T: DirEntryLike + ?Sized> FsNodeState<'a, T> {
    pub fn new(entry: &'a T, follow_links: bool) -> Self {
        FsNodeState {
            entry,
            follow_links,
            path: OnceCell::new(),
            file_type: OnceCell::new(),
            metadata: OnceCell::new(),
        }
    }

    pub fn take_path(&mut self) -> PathBuf {
        self.path.take().unwrap_or_else(|| self.entry.path())
    }

    pub fn path(&self) -> &Path {
        self.path.get_or_init(|| self.entry.path())
    }

    pub fn file_type(&self) -> Result<FileType> {
        if self.follow_links {
            Ok(self.metadata()?.file_type())
        } else if let Some(Ok(metadata)) = self.metadata.get() {
            return Ok(metadata.file_type());
        } else {
            self.file_type
                .get_or_init(|| -> Result<_> {
                    self.entry
                        .file_type()
                        .unwrap_or_else(|| self.metadata().map(|m| m.file_type()))
                        .map_err(|e| add_context(&self.path())(e))
                })
                .as_ref()
                .map_err(copy_io_error)
                .copied()
        }
    }

    pub fn metadata(&self) -> Result<&Metadata> {
        self.metadata
            .get_or_init(|| {
                if self.follow_links {
                    self.path().metadata()
                } else if cfg!(target_vendor = "apple") {
                    // On macOS entry.metadata() just does entry.path().symlink_metadata()
                    // but we can reuse path in that case.
                    self.path().symlink_metadata()
                } else {
                    self.entry.metadata()
                }
                .map_err(|e| add_context(&self.path())(e))
            })
            .as_ref()
            .map_err(copy_io_error)
    }
}

pub fn scan_tree<T: Send + Sync, Fn1, Fn2>(
    path: &Path,
    handler: &Fn1,
    follow_links: bool,
) -> Result<Vec<T>>
where
    T: Send + Sync,
    Fn1: Fn(&mut FsNodeState<DirEntry>) -> Result<Option<Fn2>> + Sync,
    Fn2: FnOnce(Result<Vec<T>>) -> Result<Option<T>> + Send,
{
    let mut files = Vec::new();
    let mut dirs = Vec::new();

    for entry in path.read_dir().as_mut().map_err(add_context(path))? {
        let entry = entry.as_ref().map_err(add_context(path))?;
        let mut state = FsNodeState::new(entry, follow_links);
        let Some(fn2) = handler(&mut state)? else {
            continue;
        };
        if state.file_type()?.is_dir() {
            dirs.push((fn2, state.take_path()));
        } else if let Some(node) = fn2(Ok(Vec::new()))? {
            files.push(node);
        }
    }

    let dirs = dirs
        .into_par_iter()
        .with_max_len(1)
        .flat_map_iter(|(fn2, path)| fn2(scan_tree(&path, handler, follow_links)).transpose())
        .collect::<Result<Vec<T>>>()?;

    Ok(join_unordered(files, dirs))
}

fn join_unordered<T>(mut big: Vec<T>, mut small: Vec<T>) -> Vec<T> {
    let total = big.len() + small.len();
    // Put the larger list first so there are fewest elements to move
    if small.len() > big.len() {
        (big, small) = (small, big)
    }
    // If the small list has capacity and the big one doesn't, put that first regardless
    if small.capacity() >= total && big.capacity() < total {
        (big, small) = (small, big)
    }
    big.append(&mut small);
    big
}

pub fn add_context<E: Borrow<Error>>(p: &Path) -> impl FnOnce(E) -> Error {
    move |e| {
        let e: &Error = e.borrow();
        Error::new(e.kind(), format!("{}: {}", p.display(), e))
    }
}

pub fn copy_io_error(e: &Error) -> Error {
    Error::new(e.kind(), e.to_string())
}
