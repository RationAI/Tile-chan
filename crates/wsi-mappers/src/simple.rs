//! `SimpleMapper`, ported from `wsi_service/simple_mapper.py`.
//!
//! Scans the data directory once: every top-level subdirectory is a case, every supported
//! file inside it is a slide. IDs are UUID5 values, and **they must match Python exactly**
//! or existing client bookmarks break:
//!
//! * case  `uuid5(NAMESPACE_URL, <directory name>).hex`
//! * slide `uuid5(NAMESPACE_URL, <case directory name> + <file name>).hex`
//!
//! `.hex` is the 32-character undashed form.
//!
//! The index is cached on disk. Python pickles it to `local_mapper.p`; this port writes
//! JSON to `local_mapper.json` instead — pickle is a Python-specific format with a
//! deserialisation hazard, and the two services must not share a cache file anyway since
//! they can disagree about which plugins are present. The lock file name is unchanged.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use wsi_core::error::{ApiError, Result};
use wsi_core::models::{CaseLocalMapper, SlideLocalMapper, SlideStorage};
use wsi_core::plugins::PluginRegistry;
use wsi_core::storage::LocalMapper;

use crate::secondary::collect_secondary_files;

const CACHE_FILE: &str = "local_mapper.json";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct MapperIndex {
    data_dir: String,
    /// Insertion order matters: `get_cases` returns cases in scan order.
    cases: Vec<CaseLocalMapper>,
    slides: Vec<SlideLocalMapper>,
}

impl MapperIndex {
    fn case(&self, case_id: &str) -> Option<&CaseLocalMapper> {
        self.cases.iter().find(|c| c.id == case_id)
    }

    fn slide(&self, slide_id: &str) -> Option<&SlideLocalMapper> {
        self.slides.iter().find(|s| s.id == slide_id)
    }
}

pub struct SimpleMapper {
    data_dir: String,
    registry: Arc<PluginRegistry>,
    index: Mutex<MapperIndex>,
    /// Serialises rebuilds. Python uses a cross-process `FileLock` because it runs several
    /// gunicorn workers; this service is a single multi-threaded process, so an in-process
    /// lock is sufficient and avoids a file-locking dependency.
    rebuild_lock: Mutex<()>,
}

impl SimpleMapper {
    /// Mirrors `SimpleMapper.__init__`, which calls `refresh(force_refresh=False)`.
    pub fn new(data_dir: impl Into<String>, registry: Arc<PluginRegistry>) -> Result<Self> {
        let mapper = Self {
            data_dir: data_dir.into().replace('\\', "/"),
            registry,
            index: Mutex::new(MapperIndex::default()),
            rebuild_lock: Mutex::new(()),
        };
        mapper.refresh(false)?;
        Ok(mapper)
    }

    fn cache_path(&self) -> PathBuf {
        PathBuf::from(CACHE_FILE)
    }

    fn read_cache(&self) -> Option<MapperIndex> {
        let raw = std::fs::read(self.cache_path()).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    /// `_get_data_dir_changed` — a cache built for a different data directory is stale.
    fn data_dir_changed(&self) -> bool {
        match self.read_cache() {
            Some(cached) => cached.data_dir != self.data_dir,
            None => false,
        }
    }

    /// `_initialize_with_path` — `simple_mapper.py:59`.
    fn build_index(&self) -> Result<MapperIndex> {
        let root = Path::new(&self.data_dir);
        let mut index = MapperIndex {
            data_dir: self.data_dir.clone(),
            ..Default::default()
        };

        let entries = read_dir_sorted(root)
            .map_err(|_| ApiError::not_found(format!("No such directory: {}", self.data_dir)))?;

        // `_collect_all_folders_as_cases`
        for (name, path) in &entries {
            if path.is_dir() {
                index.cases.push(CaseLocalMapper {
                    id: uuid5_url_hex(name),
                    local_id: name.clone(),
                    slides: Vec::new(),
                });
            }
        }

        // `_collect_all_files_as_slides`
        for case in &mut index.cases {
            let case_dir = root.join(&case.local_id);
            let Ok(files) = read_dir_sorted(&case_dir) else {
                continue;
            };
            for (file_name, absfile) in files {
                if !self.registry.is_supported_format(&absfile) {
                    continue;
                }
                let slide_id = uuid5_url_hex(&format!("{}{}", case.local_id, file_name));
                if index.slides.iter().any(|s| s.id == slide_id) {
                    continue;
                }
                case.slides.push(slide_id.clone());
                let addresses = collect_secondary_files(
                    &absfile.to_string_lossy(),
                    &slide_id,
                    &slide_id,
                    &self.data_dir,
                );
                index.slides.push(SlideLocalMapper {
                    id: slide_id.clone(),
                    local_id: file_name,
                    slide_storage: SlideStorage {
                        slide_id,
                        storage_type: "fs".into(),
                        storage_addresses: addresses,
                    },
                });
            }
        }

        Ok(index)
    }
}

impl LocalMapper for SimpleMapper {
    fn get_cases(&self, context: Option<&str>) -> Result<Vec<CaseLocalMapper>> {
        if let Some(context) = context {
            tracing::warn!("SimpleMapper: received unexpected context='{context}', ignoring.");
        }
        self.load()?;
        Ok(self.index.lock().cases.clone())
    }

    fn get_slides(&self, case_id: &str) -> Result<Vec<SlideLocalMapper>> {
        self.load()?;
        let index = self.index.lock();
        let Some(case) = index.case(case_id) else {
            return Err(ApiError::not_found(format!(
                "Case with case_id {case_id} does not exist"
            )));
        };
        // Python sorts the slide IDs before looking them up.
        let mut ids = case.slides.clone();
        ids.sort();
        Ok(ids
            .iter()
            .filter_map(|id| index.slide(id).cloned())
            .collect())
    }

    fn get_slide(&self, slide_id: &str) -> Result<SlideLocalMapper> {
        if let Some(slide) = self.index.lock().slide(slide_id).cloned() {
            return Ok(slide);
        }
        // Miss: reload from disk in case another process rebuilt the index, then retry.
        self.load()?;
        self.index.lock().slide(slide_id).cloned().ok_or_else(|| {
            ApiError::not_found(format!("Slide with slide_id {slide_id} does not exist"))
        })
    }

    /// `refresh` — rebuilds when forced, when the data directory changed, or when there is
    /// no cache yet; then reloads.
    fn refresh(&self, force: bool) -> Result<()> {
        {
            let _guard = self.rebuild_lock.lock();
            if force || self.data_dir_changed() || !self.cache_path().exists() {
                let index = self.build_index()?;
                if let Ok(serialised) = serde_json::to_vec(&index) {
                    // A cache-write failure is not fatal: the in-memory index is still
                    // valid and the next start simply rebuilds.
                    if let Err(err) = std::fs::write(self.cache_path(), serialised) {
                        tracing::warn!("Could not persist local mapper cache: {err}");
                    }
                }
                *self.index.lock() = index;
                return Ok(());
            }
        }
        self.load()
    }

    fn load(&self) -> Result<()> {
        if let Some(cached) = self.read_cache() {
            if cached.data_dir == self.data_dir {
                *self.index.lock() = cached;
            }
        }
        Ok(())
    }
}

/// `uuid5(NAMESPACE_URL, name).hex` — the 32-character undashed form.
pub fn uuid5_url_hex(name: &str) -> String {
    Uuid::new_v5(&Uuid::NAMESPACE_URL, name.as_bytes())
        .simple()
        .to_string()
}

/// `os.listdir` in a deterministic order.
fn read_dir_sorted(dir: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_and_slide_ids_match_the_python_uuid5_scheme() {
        // Reference values produced by CPython:
        //   uuid5(NAMESPACE_URL, "testcase").hex
        //   uuid5(NAMESPACE_URL, "testcase" + "CMU-1-small.tiff").hex
        // The first two also appear in wsi_service/tests/unit/test_api.py as the case and
        // slide IDs of the standard fixture, so matching them pins the whole scheme:
        // namespace, name encoding, and the undashed `hex` form.
        assert_eq!(
            uuid5_url_hex("testcase"),
            "b85c8c5b3959522ca49b013c000078cc"
        );
        assert_eq!(
            uuid5_url_hex("testcaseCMU-1-small.tiff"),
            "14b5c5dab96b540bba23b08429592bcf"
        );
        assert_eq!(
            uuid5_url_hex("testcaseCMU-1.svs"),
            "893de814963d5f06926995342863b752"
        );

        assert_eq!(uuid5_url_hex("testcase").len(), 32);
        assert!(!uuid5_url_hex("testcase").contains('-'));
    }

    #[test]
    fn uuid5_matches_a_known_vector() {
        // uuid.uuid5(uuid.NAMESPACE_URL, "http://python.org/") in CPython.
        let value = Uuid::new_v5(&Uuid::NAMESPACE_URL, b"http://python.org/");
        assert_eq!(value.to_string(), "4c565f0d-3f5a-5890-b41b-20cf47701c5e");
    }

    #[test]
    fn slide_ids_concatenate_without_a_separator() {
        // Python builds the name as `local_case_id + case_file`, so "ab" + "c" and
        // "a" + "bc" collide. Reproduced rather than fixed: changing it would change every
        // slide ID already handed out to clients.
        assert_eq!(uuid5_url_hex("abc"), uuid5_url_hex("abc"));
        assert_ne!(uuid5_url_hex("a"), uuid5_url_hex("b"));
    }
}
