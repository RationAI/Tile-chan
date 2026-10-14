//! `PathsMapper`, ported from `wsi_service/paths_mapper.py`.
//!
//! Path-as-ID: a case is a directory relative to the data root and a slide is a file
//! inside it, so IDs are human-readable paths rather than UUIDs. `>` is accepted as a
//! separator alongside `/` for callers that cannot put slashes in a query parameter.
//!
//! Context-dependent: `/cases/` requires `?context=`, because there is no single root case
//! list to enumerate.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use wsi_core::error::{ApiError, Result};
use wsi_core::models::{CaseLocalMapper, SlideLocalMapper, SlideStorage};
use wsi_core::plugins::PluginRegistry;
use wsi_core::storage::LocalMapper;

use crate::secondary::collect_secondary_files;

pub struct PathsMapper {
    data_dir: String,
    registry: Arc<PluginRegistry>,
}

impl PathsMapper {
    pub fn new(data_dir: impl Into<String>, registry: Arc<PluginRegistry>) -> Self {
        Self {
            data_dir: into_normalised(data_dir.into()),
            registry,
        }
    }

    fn absolute(&self, relative: &str) -> PathBuf {
        PathBuf::from(format!(
            "{}/{}",
            self.data_dir.trim_end_matches('/'),
            relative.trim_start_matches('/')
        ))
    }

    /// Directory entries in a stable order.
    ///
    /// `os.listdir` returns entries in an unspecified, filesystem-dependent order and the
    /// Python mapper passes that through for cases. Sorting makes the response
    /// deterministic; `get_slides` already sorts in Python.
    fn entries(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
        let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
            .map_err(|_| ApiError::not_found(format!("No such directory: {}", dir.display())))?
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }
}

fn into_normalised(value: String) -> String {
    value.replace('\\', "/")
}

/// `context.replace(">", "/")` — both separators are accepted everywhere.
fn normalise_id(value: &str) -> String {
    value.replace('>', "/")
}

impl LocalMapper for PathsMapper {
    fn is_context_dependent(&self) -> bool {
        true
    }

    fn get_cases(&self, context: Option<&str>) -> Result<Vec<CaseLocalMapper>> {
        let Some(context) = context else {
            return Err(ApiError::not_found("context is required"));
        };
        let context_path = normalise_id(context);
        let abs_context = self.absolute(&context_path);

        if !abs_context.is_dir() {
            return Err(ApiError::not_found(format!(
                "Context folder '{context}' not found"
            )));
        }

        let mut cases = Vec::new();
        for (name, path) in Self::entries(&abs_context)? {
            if !path.is_dir() {
                continue;
            }
            let case_id = format!("{context}/{name}");

            // A directory that is itself a slide (DICOM) belongs to the parent case as a
            // slide, not as a case of its own.
            if self.registry.is_supported_format(&path) {
                continue;
            }

            let mut slide_ids = Vec::new();
            for (file_name, file_path) in Self::entries(&path)? {
                if self.registry.is_supported_format(&file_path) {
                    slide_ids.push(format!("{case_id}/{file_name}"));
                }
            }

            cases.push(CaseLocalMapper {
                id: case_id.clone(),
                local_id: case_id,
                slides: slide_ids,
            });
        }
        Ok(cases)
    }

    fn get_slides(&self, case_id: &str) -> Result<Vec<SlideLocalMapper>> {
        let case_path = normalise_id(case_id);
        let abs_case = self.absolute(&case_path);

        if !abs_case.is_dir() {
            return Err(ApiError::not_found(format!("Case '{case_id}' not found")));
        }

        // The case directory may itself be a slide, e.g. a DICOM folder.
        if self.registry.is_supported_format(&abs_case) {
            let slide_id = case_id.to_owned();
            let addresses = collect_secondary_files(
                &abs_case.to_string_lossy(),
                &slide_id,
                &slide_id,
                &self.data_dir,
            );
            return Ok(vec![SlideLocalMapper {
                id: slide_id.clone(),
                local_id: abs_case
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                slide_storage: SlideStorage {
                    slide_id,
                    storage_type: "fs".into(),
                    storage_addresses: addresses,
                },
            }]);
        }

        let mut slides = Vec::new();
        for (file_name, file_path) in Self::entries(&abs_case)? {
            if !self.registry.is_supported_format(&file_path) {
                continue;
            }
            let slide_id = format!("{case_id}/{file_name}");
            let addresses = collect_secondary_files(
                &file_path.to_string_lossy(),
                &slide_id,
                &slide_id,
                &self.data_dir,
            );
            slides.push(SlideLocalMapper {
                id: slide_id.clone(),
                local_id: file_name,
                slide_storage: SlideStorage {
                    slide_id,
                    storage_type: "fs".into(),
                    storage_addresses: addresses,
                },
            });
        }
        Ok(slides)
    }

    fn get_slide(&self, slide_id: &str) -> Result<SlideLocalMapper> {
        let slide_id = normalise_id(slide_id);
        let absfile = self.absolute(&slide_id);
        if !self.registry.is_supported_format(&absfile) {
            return Err(ApiError::not_found(format!(
                "Slide {slide_id} does not exist"
            )));
        }
        let addresses = collect_secondary_files(
            &absfile.to_string_lossy(),
            &slide_id,
            &slide_id,
            &self.data_dir,
        );
        Ok(SlideLocalMapper {
            id: slide_id.clone(),
            local_id: slide_id.clone(),
            slide_storage: SlideStorage {
                slide_id,
                storage_type: "fs".into(),
                storage_addresses: addresses,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use wsi_core::plugins::WsiPlugin;
    use wsi_core::slide::Slide;

    /// Accepts `.svs` files and directories containing a `.dcm`, mirroring what the real
    /// plugin set does closely enough to exercise the mapper's branching.
    struct StubPlugin;

    #[async_trait]
    impl WsiPlugin for StubPlugin {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn version(&self) -> &'static str {
            "0.0.0"
        }
        fn is_supported(&self, path: &Path) -> bool {
            if path.is_dir() {
                return std::fs::read_dir(path)
                    .map(|mut it| {
                        it.any(|e| {
                            e.map(|e| e.path().extension().and_then(|x| x.to_str()) == Some("dcm"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
            }
            path.extension().and_then(|e| e.to_str()) == Some("svs")
        }
        async fn open(&self, _: &Path) -> Result<Arc<dyn Slide>> {
            unimplemented!()
        }
    }

    struct Fixture {
        root: PathBuf,
        mapper: PathsMapper,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("wsi-paths-mapper-{name}"));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            let registry = Arc::new(PluginRegistry::new(vec![Arc::new(StubPlugin)]));
            let mapper = PathsMapper::new(root.to_string_lossy().replace('\\', "/"), registry);
            Self { root, mapper }
        }

        fn dir(&self, rel: &str) -> PathBuf {
            let p = self.root.join(rel);
            std::fs::create_dir_all(&p).unwrap();
            p
        }

        fn file(&self, rel: &str) {
            let p = self.root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn cases_require_a_context() {
        let f = Fixture::new("ctx");
        let err = f.mapper.get_cases(None).unwrap_err();
        assert_eq!(err.status, http::StatusCode::NOT_FOUND);
        assert_eq!(err.detail_text(), "context is required");
    }

    #[test]
    fn missing_context_folder_is_a_404() {
        let f = Fixture::new("missing");
        let err = f.mapper.get_cases(Some("nope")).unwrap_err();
        assert_eq!(err.detail_text(), "Context folder 'nope' not found");
    }

    #[test]
    fn cases_list_subdirectories_with_their_slides() {
        let f = Fixture::new("cases");
        f.dir("proj/caseA");
        f.file("proj/caseA/one.svs");
        f.file("proj/caseA/two.svs");
        f.file("proj/caseA/ignored.txt");
        f.dir("proj/caseB");

        let cases = f.mapper.get_cases(Some("proj")).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].id, "proj/caseA");
        assert_eq!(cases[0].local_id, "proj/caseA");
        assert_eq!(
            cases[0].slides,
            vec!["proj/caseA/one.svs", "proj/caseA/two.svs"]
        );
        assert!(cases[1].slides.is_empty());
    }

    #[test]
    fn a_slide_directory_is_not_listed_as_a_case() {
        let f = Fixture::new("dicomcase");
        f.dir("proj/dicomslide");
        f.file("proj/dicomslide/a.dcm");
        f.dir("proj/realcase");

        let cases = f.mapper.get_cases(Some("proj")).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].id, "proj/realcase");
    }

    #[test]
    fn angle_bracket_separator_is_accepted() {
        let f = Fixture::new("angle");
        f.dir("proj/caseA");
        f.file("proj/caseA/one.svs");

        let cases = f.mapper.get_cases(Some("proj")).unwrap();
        assert_eq!(cases.len(), 1);

        let slides = f.mapper.get_slides("proj>caseA").unwrap();
        assert_eq!(slides.len(), 1);
        assert_eq!(slides[0].local_id, "one.svs");
    }

    #[test]
    fn slides_of_a_dicom_directory_return_the_directory_itself() {
        let f = Fixture::new("dicomslides");
        f.dir("proj/slide");
        f.file("proj/slide/a.dcm");

        let slides = f.mapper.get_slides("proj/slide").unwrap();
        assert_eq!(slides.len(), 1);
        assert_eq!(slides[0].id, "proj/slide");
        assert_eq!(slides[0].local_id, "slide");
        // The DICOM file plus the main address.
        assert_eq!(slides[0].slide_storage.storage_addresses.len(), 2);
    }

    #[test]
    fn get_slide_rejects_unsupported_paths() {
        let f = Fixture::new("getslide");
        f.file("proj/caseA/one.svs");
        f.file("proj/caseA/notes.txt");

        let slide = f.mapper.get_slide("proj/caseA/one.svs").unwrap();
        assert_eq!(slide.id, "proj/caseA/one.svs");
        assert_eq!(slide.slide_storage.storage_type, "fs");
        assert_eq!(
            slide.slide_storage.storage_addresses[0].address,
            "proj/caseA/one.svs"
        );

        let err = f.mapper.get_slide("proj/caseA/notes.txt").unwrap_err();
        assert_eq!(err.status, http::StatusCode::NOT_FOUND);
        assert_eq!(
            err.detail_text(),
            "Slide proj/caseA/notes.txt does not exist"
        );
    }

    #[test]
    fn mapper_reports_itself_as_context_dependent() {
        let f = Fixture::new("flag");
        assert!(f.mapper.is_context_dependent());
    }
}
