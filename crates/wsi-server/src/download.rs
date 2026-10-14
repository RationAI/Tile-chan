//! Raw slide download, ported from `wsi_service/utils/download_utils.py` and the
//! `/slides/download` endpoint (`api/v3/slides.py:375`).
//!
//! Python streams a `ZIP_STORED` archive with `zipfly`. Entry names are the file paths
//! with their deepest common parent directory stripped, so a DICOM slide arrives as a
//! folder of `.dcm` files rather than a flat list.

use std::io::Write;
use std::path::Path;

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use tokio::sync::mpsc;

use wsi_core::error::{ApiError, Result};

use crate::zipstream::StreamingZip;

/// `expand_folders` — `download_utils.py:20`.
///
/// Python appends to the list it is iterating, so nested directories keep expanding, then
/// de-duplicates with `set()`. The set makes the resulting order arbitrary; this port
/// sorts instead, so archives are byte-stable across runs.
pub fn expand_folders(paths: &[String]) -> Vec<String> {
    let mut queue: Vec<String> = paths.to_vec();
    let mut index = 0;
    while index < queue.len() {
        let current = queue[index].clone();
        if Path::new(&current).is_dir() {
            if let Ok(entries) = glob::glob(&format!("{}/*", current.trim_end_matches('/'))) {
                for entry in entries.flatten() {
                    queue.push(entry.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        index += 1;
    }
    queue.sort();
    queue.dedup();
    queue
}

/// `remove_folders` — `download_utils.py:27`.
pub fn remove_folders(paths: Vec<String>) -> Vec<String> {
    paths
        .into_iter()
        .filter(|p| !Path::new(p).is_dir())
        .collect()
}

/// `get_parent_folder` — `download_utils.py:13`.
///
/// Walks the first path's directory upwards until it is a substring of every other path's
/// directory, then appends a trailing slash. The substring test is what Python does — not
/// a proper path-prefix test — and is reproduced.
pub fn parent_folder(paths: &[String]) -> String {
    let dirname = |p: &str| -> String {
        match p.rfind('/') {
            Some(idx) => p[..idx].to_owned(),
            None => String::new(),
        }
    };

    let Some(first) = paths.first() else {
        return "/".to_owned();
    };
    let mut parent = dirname(first);
    while paths.iter().any(|p| !dirname(p).contains(&parent)) {
        let next = dirname(&parent);
        if next == parent {
            break;
        }
        parent = next;
    }
    format!("{parent}/")
}

/// `get_zipfly_paths` — `download_utils.py:5`. Returns `(filesystem path, entry name)`.
pub fn zip_entries(paths: &[String]) -> Vec<(String, String)> {
    let parent = parent_folder(paths);
    paths
        .iter()
        .map(|p| (p.clone(), p.replacen(&parent, "", 1)))
        .collect()
}

/// Streams the archive so a multi-gigabyte slide does not have to be buffered.
///
/// The zip is written on a blocking thread into a channel; the response body drains it.
/// `ZIP_STORED` matches `zipfly`, which does not compress — slide data is already
/// compressed, so deflating it would only burn CPU.
pub fn stream_zip(entries: Vec<(String, String)>, filename: &str) -> Result<Response> {
    if entries.is_empty() {
        return Err(ApiError::internal("No files to download"));
    }

    let (tx, rx) = mpsc::channel::<std::result::Result<bytes::Bytes, std::io::Error>>(8);

    std::thread::spawn(move || {
        let writer = ChannelWriter { tx };
        let mut zip = StreamingZip::new(writer);

        for (fs_path, name) in entries {
            let Ok(mut file) = std::fs::File::open(&fs_path) else {
                // A file that vanished between listing and reading is skipped, which is
                // also what zipfly's generator does.
                continue;
            };
            if zip.add_entry(&name, &mut file).is_err() {
                // The client hung up, or the file became unreadable mid-stream. Either
                // way the response is already truncated; dropping the writer closes it.
                return;
            }
        }
        let _ = zip.finish();
    });

    let stream = tokio_stream_from(rx);
    Ok(http::Response::builder()
        .status(http::StatusCode::OK)
        .header(http::header::CONTENT_TYPE, "application/zip")
        .header(
            http::header::CONTENT_DISPOSITION,
            format!("attachment;filename={filename}.zip"),
        )
        .body(Body::from_stream(stream))
        .map(IntoResponse::into_response)
        .unwrap_or_else(|_| http::StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

fn tokio_stream_from(
    rx: mpsc::Receiver<std::result::Result<bytes::Bytes, std::io::Error>>,
) -> impl futures::Stream<Item = std::result::Result<bytes::Bytes, std::io::Error>> {
    futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    })
}

/// `std::io::Write` that forwards to an async channel, so `ZipWriter` can run on a
/// blocking thread while the response body is consumed asynchronously.
struct ChannelWriter {
    tx: mpsc::Sender<std::result::Result<bytes::Bytes, std::io::Error>>,
}

impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let chunk = bytes::Bytes::copy_from_slice(buf);
        self.tx
            .blocking_send(Ok(chunk))
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receiver dropped"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_folder_is_the_deepest_common_directory() {
        let paths = vec![
            "/data/case/slide/a.dcm".to_owned(),
            "/data/case/slide/b.dcm".to_owned(),
        ];
        assert_eq!(parent_folder(&paths), "/data/case/slide/");

        let paths = vec![
            "/data/case/slide/a.dcm".to_owned(),
            "/data/case/other.svs".to_owned(),
        ];
        assert_eq!(parent_folder(&paths), "/data/case/");
    }

    #[test]
    fn a_single_file_uses_its_own_directory() {
        let paths = vec!["/data/case/a.svs".to_owned()];
        assert_eq!(parent_folder(&paths), "/data/case/");
    }

    #[test]
    fn entry_names_are_relative_to_the_common_parent() {
        let paths = vec![
            "/data/case/slide/a.dcm".to_owned(),
            "/data/case/slide/b.dcm".to_owned(),
        ];
        let entries = zip_entries(&paths);
        assert_eq!(entries[0].1, "a.dcm");
        assert_eq!(entries[1].1, "b.dcm");
    }

    #[test]
    fn entry_names_keep_the_folder_for_mixed_depths() {
        let paths = vec![
            "/data/case/slide.mrxs".to_owned(),
            "/data/case/slide/Index.dat".to_owned(),
        ];
        let entries = zip_entries(&paths);
        assert_eq!(entries[0].1, "slide.mrxs");
        assert_eq!(entries[1].1, "slide/Index.dat");
    }

    #[test]
    fn folders_expand_recursively_and_are_then_dropped() {
        let base = std::env::temp_dir().join("wsi-download-expand");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("slide/nested")).unwrap();
        std::fs::write(base.join("slide/a.dcm"), b"x").unwrap();
        std::fs::write(base.join("slide/nested/b.dcm"), b"x").unwrap();

        let root = base.join("slide").to_string_lossy().replace('\\', "/");
        let expanded = expand_folders(std::slice::from_ref(&root));
        // The folder itself, the nested folder, and both files.
        assert!(expanded.contains(&format!("{root}/a.dcm")));
        assert!(expanded.contains(&format!("{root}/nested/b.dcm")));

        let files = remove_folders(expanded);
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.ends_with(".dcm")));

        let _ = std::fs::remove_dir_all(base);
    }
}
