//! `IteratorMapper`, ported from `wsi_service/mapper_iterator/`.
//!
//! Walks the data directory: every directory becomes a case, and every recognised file
//! inside it becomes a slide, with IDs derived from UUID5 values rather than paths.
//! Multi-file formats are grouped into a single slide:
//!
//! * `.mrxs` — the file plus a sibling directory of the same stem, which must contain
//!   `Slidedat.ini` and nothing but `.dat`/`.ini` files;
//! * `.dcm` — a directory of DICOM files, represented by the first one alphabetically,
//!   skipping `DICOMDIR`;
//! * `.vsf` — a `.vsf`/`.img`/`.jpg` trio sharing a stem.
//!
//! Settings come from the `WSIT_` environment prefix.
//!
//! Two behaviours worth stating, both reproduced:
//!
//! * `institution_pattern` and `project_pattern` default to *unset*, and the Python code
//!   tests them against `""` rather than `None`, so it calls `re.search(None, path)` and
//!   raises `TypeError` during construction. An unset pattern is therefore a startup
//!   failure, not a "skip this pattern".
//! * a matched pattern group longer than five characters trips an `assert`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use wsi_core::error::{ApiError, Result};
use wsi_core::models::{CaseLocalMapper, SlideLocalMapper, SlideStorage, StorageAddress};
use wsi_core::settings::EnvSource;
use wsi_core::storage::LocalMapper;

const CACHE_FILE: &str = "local_mapper_iterator.json";

/// `Extensions.get_allowed_extensions` — `iterator/extensions.py:3`.
const ALLOWED_EXTENSIONS: [&str; 19] = [
    ".bif",
    ".dcm",
    ".isyntax",
    ".mrxs",
    ".ndpi",
    ".nii",
    ".nii.gz",
    ".ome.btf",
    ".ome.tf2",
    ".ome.tf8",
    ".ome.tif",
    ".ome.tiff",
    ".scn",
    ".svs",
    ".tif",
    ".tiff",
    ".vsf",
    ".img",
    ".jpg",
];

/// `Extensions.get_multi_extensions` — formats made of more than one file.
const MULTI_EXTENSIONS: [&str; 5] = [".mrxs", ".dcm", ".vsf", ".img", ".jpg"];

/// `SettingsIterator` — `iterator/settings.py:3`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IteratorSettings {
    /// `None` means the environment variable was never set, which Python turns into a
    /// `TypeError`; `Some("")` means explicitly disabled.
    pub institution_pattern: Option<String>,
    pub project_pattern: Option<String>,
    pub source_path: String,
}

impl IteratorSettings {
    pub fn load() -> Self {
        Self::from_source(&EnvSource::load_prefixed(".env", "wsit_"))
    }

    pub fn from_source(src: &EnvSource) -> Self {
        Self {
            institution_pattern: src.get_raw("institution_pattern"),
            project_pattern: src.get_raw("project_pattern"),
            source_path: src.get_raw("source_path").unwrap_or_else(|| "/".to_owned()),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct MapperIndex {
    data_dir: String,
    cases: Vec<CaseLocalMapper>,
    slides: Vec<SlideLocalMapper>,
}

pub struct IteratorMapper {
    data_dir: String,
    settings: IteratorSettings,
    /// Where the index is cached. Python writes it to the working directory; that is the
    /// default here too, but making it explicit lets tests stay independent.
    cache_dir: PathBuf,
    index: Mutex<MapperIndex>,
    rebuild_lock: Mutex<()>,
}

impl IteratorMapper {
    pub fn new(data_dir: impl Into<String>, settings: IteratorSettings) -> Result<Self> {
        Self::with_cache_dir(data_dir, settings, ".")
    }

    pub fn with_cache_dir(
        data_dir: impl Into<String>,
        settings: IteratorSettings,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let mapper = Self {
            data_dir: data_dir.into().replace('\\', "/"),
            settings,
            cache_dir: cache_dir.into(),
            index: Mutex::new(MapperIndex::default()),
            rebuild_lock: Mutex::new(()),
        };
        mapper.refresh(false)?;
        Ok(mapper)
    }

    fn cache_path(&self) -> PathBuf {
        self.cache_dir.join(CACHE_FILE)
    }

    fn read_cache(&self) -> Option<MapperIndex> {
        let raw = std::fs::read(self.cache_path()).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    fn data_dir_changed(&self) -> bool {
        match self.read_cache() {
            Some(cached) => cached.data_dir != self.data_dir,
            None => false,
        }
    }

    /// `iterate` — `iterator/iterator.py:56`.
    ///
    /// `refresh` overrides `source_path` with the data directory before iterating, so the
    /// `WSIT_SOURCE_PATH` setting is effectively unused.
    fn build_index(&self) -> Result<MapperIndex> {
        let mut index = MapperIndex {
            data_dir: self.data_dir.clone(),
            ..Default::default()
        };

        let mut source = PathBuf::from(&self.data_dir);
        if !source.exists() {
            // Python prints and returns a single value from a function whose caller
            // unpacks two, raising `ValueError`. Startup fails either way.
            return Err(ApiError::internal(format!(
                "Path does not exist: {}",
                source.display()
            )));
        }
        if !source.is_dir() {
            source = source.parent().map(Path::to_path_buf).unwrap_or(source);
        }

        let mut walker = Walker {
            settings: &self.settings,
            cases: Vec::new(),
            slides: Vec::new(),
            controlled: BTreeSet::new(),
        };
        walker.control_files(&source)?;

        index.cases = walker.cases;
        index.slides = walker.slides;
        Ok(index)
    }
}

impl LocalMapper for IteratorMapper {
    fn get_cases(&self, context: Option<&str>) -> Result<Vec<CaseLocalMapper>> {
        if let Some(context) = context {
            tracing::warn!("IteratorMapper: received unexpected context='{context}', ignoring.");
        }
        self.load()?;
        Ok(self.index.lock().cases.clone())
    }

    fn get_slides(&self, case_id: &str) -> Result<Vec<SlideLocalMapper>> {
        self.load()?;
        let index = self.index.lock();
        let Some(case) = index.cases.iter().find(|c| c.id == case_id) else {
            return Err(ApiError::not_found(format!(
                "Case with case_id {case_id} does not exist"
            )));
        };
        let mut ids = case.slides.clone();
        ids.sort();
        Ok(ids
            .iter()
            .filter_map(|id| index.slides.iter().find(|s| s.id == *id).cloned())
            .collect())
    }

    fn get_slide(&self, slide_id: &str) -> Result<SlideLocalMapper> {
        if let Some(slide) = self
            .index
            .lock()
            .slides
            .iter()
            .find(|s| s.id == slide_id)
            .cloned()
        {
            return Ok(slide);
        }
        self.load()?;
        self.index
            .lock()
            .slides
            .iter()
            .find(|s| s.id == slide_id)
            .cloned()
            .ok_or_else(|| {
                ApiError::not_found(format!("Slide with slide_id {slide_id} does not exist"))
            })
    }

    fn refresh(&self, force: bool) -> Result<()> {
        {
            let _guard = self.rebuild_lock.lock();
            if force || self.data_dir_changed() || !self.cache_path().exists() {
                let index = self.build_index()?;
                if let Ok(serialised) = serde_json::to_vec(&index) {
                    if let Err(err) = std::fs::write(self.cache_path(), serialised) {
                        tracing::warn!("Could not persist iterator mapper cache: {err}");
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

/// Case metadata needed while walking, before it becomes a `CaseLocalMapper`.
struct CaseContext {
    local_id: String,
    namespace: Uuid,
    institution: String,
    project: String,
}

struct Walker<'a> {
    settings: &'a IteratorSettings,
    cases: Vec<CaseLocalMapper>,
    slides: Vec<SlideLocalMapper>,
    /// Paths already consumed by a multi-file format, so they are not offered again.
    controlled: BTreeSet<String>,
}

impl Walker<'_> {
    /// `control_files` — `iterator/iterator.py:26`.
    fn control_files(&mut self, source: &Path) -> Result<()> {
        let case = self.create_case(source)?;
        self.cases.push(CaseLocalMapper {
            id: case.local_id.clone(),
            local_id: case.local_id.clone(),
            slides: Vec::new(),
        });

        for entry in read_dir_sorted(source) {
            let file_path = normalise(&entry);
            let extension = get_extension(&file_path);

            if entry.is_dir() {
                self.control_dir(&entry, &case)?;
            } else if !self.controlled.contains(&file_path) && is_allowed_extension(&extension) {
                if MULTI_EXTENSIONS.contains(&extension.as_str()) {
                    self.multi_extension_control(&entry, &case)?;
                } else {
                    self.add_slide(&file_path, &case);
                    self.controlled.insert(file_path);
                }
            }
        }
        Ok(())
    }

    /// `control_dir` — `iterator/iterator.py:15`.
    ///
    /// A directory with a sibling `<name>.mrxs` is that slide's data folder; a directory
    /// containing `.dcm` files is a DICOM slide; anything else is a nested case.
    fn control_dir(&mut self, dir: &Path, case: &CaseContext) -> Result<()> {
        let mrxs = format!("{}.mrxs", normalise(dir));
        if Path::new(&mrxs).exists() && !self.controlled.contains(&mrxs) {
            return self.control_mrxs(Path::new(&mrxs), case);
        }
        if directory_file_extensions(dir).contains(".dcm") {
            return self.control_dcm(dir, case);
        }
        self.control_files(dir)
    }

    /// `control_mrxs` — `iterator/control_extensions.py:24`.
    fn control_mrxs(&mut self, file_path: &Path, case: &CaseContext) -> Result<()> {
        self.controlled.insert(normalise(file_path));

        let stem = file_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let corresponding = file_path
            .parent()
            .map(|p| p.join(&stem))
            .unwrap_or_else(|| PathBuf::from(&stem));

        if !corresponding.is_dir() {
            tracing::info!(
                "No corresponding directory found for the file {}.",
                file_path.display()
            );
            return Ok(());
        }
        if !corresponding.join("Slidedat.ini").exists() {
            tracing::info!(
                "Slidedat.ini file is missing in the directory {}.",
                corresponding.display()
            );
            return Ok(());
        }
        for entry in read_dir_sorted(&corresponding) {
            let extension = get_extension(&normalise(&entry));
            if extension != ".dat" && extension != ".ini" {
                tracing::info!("Unsupported file format for .mrxs {}.", entry.display());
                return Ok(());
            }
        }

        self.add_slide(&normalise(file_path), case);
        Ok(())
    }

    /// `control_dcm` — `iterator/control_extensions.py:53`.
    ///
    /// The slide is represented by the first `.dcm` in sorted order, skipping `DICOMDIR`.
    fn control_dcm(&mut self, path: &Path, case: &CaseContext) -> Result<()> {
        let directory = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().map(Path::to_path_buf).unwrap_or_default()
        };

        let mut names: Vec<String> = read_dir_sorted(&directory)
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect();

        for name in &names {
            self.controlled.insert(normalise(&directory.join(name)));
            if get_extension(name) != ".dcm" && name != "DICOMDIR" {
                tracing::info!("Unsupported file format for .dcm {name}.");
                return Ok(());
            }
        }

        names.sort();
        let Some(first) = names.first().cloned() else {
            // Python would raise IndexError on an empty directory.
            return Err(ApiError::internal("list index out of range"));
        };
        let chosen = if first == "DICOMDIR" {
            match names.get(1) {
                Some(second) => second.clone(),
                None => return Err(ApiError::internal("list index out of range")),
            }
        } else {
            first
        };

        self.add_slide(&normalise(&directory.join(chosen)), case);
        Ok(())
    }

    /// `control_vsf` — `iterator/control_extensions.py:76`. All three files must exist.
    fn control_vsf(&mut self, file_path: &Path, case: &CaseContext) -> Result<()> {
        let base = strip_final_extension(&normalise(file_path));
        let vsf = format!("{base}.vsf");
        let img = format!("{base}.img");
        let jpg = format!("{base}.jpg");

        if Path::new(&vsf).exists() && Path::new(&img).exists() && Path::new(&jpg).exists() {
            self.add_slide(&vsf, case);
            self.controlled.insert(vsf);
            self.controlled.insert(img);
            self.controlled.insert(jpg);
            return Ok(());
        }

        tracing::info!("Error: Folder must contain one .vsf, one .img, and one .jpg file.");
        self.controlled.insert(normalise(file_path));
        Ok(())
    }

    /// `multi_extension_control` — dispatch on the *final* extension, which is why
    /// `.ome.tif` never reaches here.
    fn multi_extension_control(&mut self, file_path: &Path, case: &CaseContext) -> Result<()> {
        let extension = final_extension(&normalise(file_path));
        match extension.as_str() {
            ".mrxs" => self.control_mrxs(file_path, case),
            ".dcm" => self.control_dcm(file_path, case),
            ".vsf" | ".img" | ".jpg" => self.control_vsf(file_path, case),
            _ => Ok(()),
        }
    }

    /// `create_case_object` — `iterator/local_id_creation.py:22`.
    fn create_case(&self, source_path: &Path) -> Result<CaseContext> {
        let path = normalise(source_path);
        let institution =
            self.match_pattern(self.settings.institution_pattern.as_deref(), &path)?;
        let project = self.match_pattern(self.settings.project_pattern.as_deref(), &path)?;

        let namespace = Uuid::new_v5(&Uuid::NAMESPACE_DNS, path.as_bytes());
        let local_id = format!("{institution}.{project}.c.{namespace}");

        Ok(CaseContext {
            local_id,
            namespace,
            institution,
            project,
        })
    }

    /// The Python guard is `if settings.institution_pattern != ""`, so an unset pattern
    /// (`None`) passes the guard and then fails inside `re.search`.
    fn match_pattern(&self, pattern: Option<&str>, path: &str) -> Result<String> {
        let Some(pattern) = pattern else {
            return Err(ApiError::internal(
                "expected string or bytes-like object, got 'NoneType'",
            ));
        };
        if pattern.is_empty() {
            return Ok(String::new());
        }
        let regex = regex::Regex::new(pattern)
            .map_err(|e| ApiError::internal(format!("Invalid pattern {pattern}: {e}")))?;
        let Some(captures) = regex.captures(path) else {
            return Ok(String::new());
        };
        let value = captures
            .get(1)
            .map(|m| m.as_str().to_owned())
            .unwrap_or_default();
        if value.len() > 5 {
            // `assert len(institution) <= 5`
            return Err(ApiError::internal(format!(
                "Pattern group '{value}' exceeds 5 characters"
            )));
        }
        Ok(value)
    }

    /// `create_slide_object` — `iterator/local_id_creation.py:60`.
    ///
    /// The storage address is the path as found, which for a walked tree is absolute.
    /// `os.path.join(data_dir, address)` leaves an absolute address unchanged, so this
    /// still resolves correctly.
    fn add_slide(&mut self, file_path: &str, case: &CaseContext) {
        let slide_uuid = Uuid::new_v5(&case.namespace, file_path.as_bytes());
        let local_id = format!("{}.{}.w.{}", case.institution, case.project, slide_uuid);

        let slide = SlideLocalMapper {
            id: local_id.clone(),
            local_id: local_id.clone(),
            slide_storage: SlideStorage {
                slide_id: local_id.clone(),
                storage_type: "fs".into(),
                storage_addresses: vec![StorageAddress {
                    address: file_path.to_owned(),
                    slide_id: local_id.clone(),
                    main_address: true,
                    storage_address_id: local_id.clone(),
                }],
            },
        };

        if let Some(case_entry) = self.cases.iter_mut().find(|c| c.id == case.local_id) {
            case_entry.slides.push(local_id.clone());
        }
        self.slides.push(slide);
    }
}

/// `get_extension` — `iterator/control_extensions.py:7`.
///
/// Recognises the compound suffixes `.ome.*` and `.nii.gz`; everything else is the final
/// extension, lowercased.
pub fn get_extension(file_path: &str) -> String {
    let last = final_extension(file_path);
    let base = strip_final_extension(file_path);
    let second = final_extension(&base);

    if second == ".ome" {
        return format!("{second}{last}");
    }
    if second == ".nii" && last == ".gz" {
        return ".nii.gz".to_owned();
    }
    last
}

/// `os.path.splitext` — the final extension including the dot, lowercased, or empty.
///
/// Like Python, a leading dot does not count as an extension separator, so `.env` has no
/// extension.
fn final_extension(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rfind('.') {
        Some(0) | None => String::new(),
        Some(index) => name[index..].to_ascii_lowercase(),
    }
}

fn strip_final_extension(path: &str) -> String {
    let extension = final_extension(path);
    if extension.is_empty() {
        return path.to_owned();
    }
    path[..path.len() - extension.len()].to_owned()
}

fn is_allowed_extension(extension: &str) -> bool {
    if ALLOWED_EXTENSIONS.contains(&extension) {
        return true;
    }
    tracing::info!("The extention is not supported: {extension}");
    false
}

/// `get_file_extensions` — the set of final suffixes of the files directly inside a
/// directory.
fn directory_file_extensions(dir: &Path) -> BTreeSet<String> {
    read_dir_sorted(dir)
        .iter()
        .filter(|p| p.is_file())
        .map(|p| final_extension(&normalise(p)))
        .filter(|e| !e.is_empty())
        .collect()
}

fn read_dir_sorted(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    out.sort();
    out
}

fn normalise(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Builds a mapper using settings read from the `WSIT_` environment prefix.
pub fn from_env(data_dir: &str) -> Result<Arc<IteratorMapper>> {
    Ok(Arc::new(IteratorMapper::new(
        data_dir,
        IteratorSettings::load(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        /// Inputs and the index cache both live under the fixture's own directory, so the
        /// tests do not race each other through the process working directory.
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("wsi-iterator-mapper-{name}"));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(root.join("data")).unwrap();
            Self { root }
        }

        fn data_dir(&self) -> String {
            normalise(&self.root.join("data"))
        }

        fn dir(&self, rel: &str) -> PathBuf {
            let path = self.root.join("data").join(rel);
            std::fs::create_dir_all(&path).unwrap();
            path
        }

        fn file(&self, rel: &str) {
            let path = self.root.join("data").join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }

        fn mapper(&self) -> Result<IteratorMapper> {
            self.mapper_with(IteratorSettings {
                // Explicitly empty: patterns disabled, but *set*, which is what the
                // Python guard actually requires.
                institution_pattern: Some(String::new()),
                project_pattern: Some(String::new()),
                source_path: self.data_dir(),
            })
        }

        fn mapper_with(&self, settings: IteratorSettings) -> Result<IteratorMapper> {
            IteratorMapper::with_cache_dir(self.data_dir(), settings, &self.root)
        }

        fn mapper_at(
            &self,
            data_dir: String,
            settings: IteratorSettings,
        ) -> Result<IteratorMapper> {
            IteratorMapper::with_cache_dir(data_dir, settings, &self.root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn compound_extensions_are_recognised() {
        assert_eq!(get_extension("/a/b.svs"), ".svs");
        assert_eq!(get_extension("/a/b.SVS"), ".svs");
        assert_eq!(get_extension("/a/b.ome.tiff"), ".ome.tiff");
        assert_eq!(get_extension("/a/b.ome.tif"), ".ome.tif");
        assert_eq!(get_extension("/a/b.nii.gz"), ".nii.gz");
        assert_eq!(get_extension("/a/b.nii"), ".nii");
        assert_eq!(get_extension("/a/noext"), "");
        // A leading dot is not an extension, matching os.path.splitext.
        assert_eq!(get_extension("/a/.env"), "");
    }

    #[test]
    fn allowed_extensions_match_the_python_list() {
        for allowed in [".svs", ".ndpi", ".ome.tiff", ".nii.gz", ".dcm", ".mrxs"] {
            assert!(is_allowed_extension(allowed), "{allowed}");
        }
        assert!(!is_allowed_extension(".png"));
        assert!(!is_allowed_extension(".txt"));
    }

    #[test]
    fn an_unset_pattern_fails_construction_like_the_python_type_error() {
        let f = Fixture::new("unset-pattern");
        f.file("case/a.svs");
        let err = match f.mapper_with(IteratorSettings {
            institution_pattern: None,
            project_pattern: None,
            source_path: f.data_dir(),
        }) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.detail_text().contains("NoneType"));
    }

    #[test]
    fn each_directory_becomes_a_case_with_dotted_uuid_ids() {
        let f = Fixture::new("cases");
        f.file("caseA/one.svs");
        f.file("caseB/two.ndpi");

        let mapper = f.mapper().unwrap();
        let cases = mapper.get_cases(None).unwrap();
        // The data directory itself, plus the two subdirectories.
        assert_eq!(cases.len(), 3);
        for case in &cases {
            assert!(case.id.starts_with("..c."), "unexpected id {}", case.id);
        }

        let slides: Vec<_> = cases.iter().flat_map(|c| c.slides.clone()).collect();
        assert_eq!(slides.len(), 2);
        for slide in &slides {
            assert!(slide.starts_with("..w."), "unexpected id {slide}");
        }
    }

    #[test]
    fn slide_ids_are_uuid5_of_the_path_under_the_case_namespace() {
        let f = Fixture::new("slide-ids");
        f.file("caseA/one.svs");

        let mapper = f.mapper().unwrap();
        let cases = mapper.get_cases(None).unwrap();
        let case = cases.iter().find(|c| !c.slides.is_empty()).unwrap();

        let case_path = normalise(&f.root.join("data/caseA"));
        let namespace = Uuid::new_v5(&Uuid::NAMESPACE_DNS, case_path.as_bytes());
        assert_eq!(case.id, format!("..c.{namespace}"));

        let slide_path = normalise(&f.root.join("data/caseA/one.svs"));
        let slide_uuid = Uuid::new_v5(&namespace, slide_path.as_bytes());
        assert_eq!(case.slides[0], format!("..w.{slide_uuid}"));

        let slide = mapper.get_slide(&case.slides[0]).unwrap();
        assert_eq!(slide.slide_storage.storage_addresses[0].address, slide_path);
        assert!(slide.slide_storage.storage_addresses[0].main_address);
    }

    #[test]
    fn unsupported_files_are_skipped() {
        let f = Fixture::new("skip");
        f.file("caseA/notes.txt");
        f.file("caseA/one.svs");

        let mapper = f.mapper().unwrap();
        let total: usize = mapper
            .get_cases(None)
            .unwrap()
            .iter()
            .map(|c| c.slides.len())
            .sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn a_valid_mirax_pair_becomes_one_slide() {
        let f = Fixture::new("mrxs-ok");
        f.file("caseA/slide.mrxs");
        f.dir("caseA/slide");
        f.file("caseA/slide/Slidedat.ini");
        f.file("caseA/slide/Data0000.dat");

        let mapper = f.mapper().unwrap();
        let total: usize = mapper
            .get_cases(None)
            .unwrap()
            .iter()
            .map(|c| c.slides.len())
            .sum();
        assert_eq!(total, 1, "the .mrxs file and its folder are one slide");
    }

    #[test]
    fn a_mirax_folder_without_slidedat_is_skipped() {
        let f = Fixture::new("mrxs-bad");
        f.file("caseA/slide.mrxs");
        f.dir("caseA/slide");
        f.file("caseA/slide/Data0000.dat");

        let mapper = f.mapper().unwrap();
        let total: usize = mapper
            .get_cases(None)
            .unwrap()
            .iter()
            .map(|c| c.slides.len())
            .sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn a_dicom_directory_becomes_one_slide_named_after_its_first_file() {
        let f = Fixture::new("dcm");
        f.file("caseA/study/DICOMDIR");
        f.file("caseA/study/b.dcm");
        f.file("caseA/study/a.dcm");

        let mapper = f.mapper().unwrap();
        let cases = mapper.get_cases(None).unwrap();
        let total: usize = cases.iter().map(|c| c.slides.len()).sum();
        assert_eq!(total, 1);

        let slide_id = cases.iter().flat_map(|c| c.slides.clone()).next().unwrap();
        let slide = mapper.get_slide(&slide_id).unwrap();
        // DICOMDIR sorts first and is skipped in favour of the next entry.
        assert!(
            slide.slide_storage.storage_addresses[0]
                .address
                .ends_with("a.dcm"),
            "got {}",
            slide.slide_storage.storage_addresses[0].address
        );
    }

    #[test]
    fn a_vsf_trio_becomes_one_slide() {
        let f = Fixture::new("vsf");
        f.file("caseA/scan.vsf");
        f.file("caseA/scan.img");
        f.file("caseA/scan.jpg");

        let mapper = f.mapper().unwrap();
        let total: usize = mapper
            .get_cases(None)
            .unwrap()
            .iter()
            .map(|c| c.slides.len())
            .sum();
        assert_eq!(total, 1, "the three files are one slide");
    }

    #[test]
    fn an_incomplete_vsf_trio_produces_no_slide() {
        let f = Fixture::new("vsf-partial");
        f.file("caseA/scan.vsf");
        f.file("caseA/scan.img");

        let mapper = f.mapper().unwrap();
        let total: usize = mapper
            .get_cases(None)
            .unwrap()
            .iter()
            .map(|c| c.slides.len())
            .sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn a_matched_pattern_lands_in_the_id() {
        let f = Fixture::new("patterns");
        f.file("mmci/one.svs");

        let mapper = f
            .mapper_with(IteratorSettings {
                institution_pattern: Some(r"data/([a-z]+)".into()),
                project_pattern: Some(String::new()),
                source_path: f.data_dir(),
            })
            .unwrap();

        let cases = mapper.get_cases(None).unwrap();
        let case = cases.iter().find(|c| !c.slides.is_empty()).unwrap();
        assert!(case.id.starts_with("mmci..c."), "got {}", case.id);
        assert!(case.slides[0].starts_with("mmci..w."));
    }

    #[test]
    fn a_pattern_group_longer_than_five_characters_is_rejected() {
        let f = Fixture::new("pattern-len");
        f.file("institution/one.svs");

        let result = f.mapper_with(IteratorSettings {
            institution_pattern: Some(r"data/([a-z]+)".into()),
            project_pattern: Some(String::new()),
            source_path: f.data_dir(),
        });
        let err = match result {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert!(err.detail_text().contains("exceeds 5 characters"));
    }

    #[test]
    fn a_missing_data_directory_fails_construction() {
        let f = Fixture::new("missing-dir");
        let result = f.mapper_at(
            format!("{}/nope", f.data_dir()),
            IteratorSettings {
                institution_pattern: Some(String::new()),
                project_pattern: Some(String::new()),
                source_path: String::new(),
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn unknown_ids_are_404s() {
        let f = Fixture::new("unknown");
        f.file("caseA/one.svs");
        let mapper = f.mapper().unwrap();

        let err = match mapper.get_slide("nope") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Slide with slide_id nope does not exist");
    }
}
