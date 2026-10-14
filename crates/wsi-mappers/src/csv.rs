//! `CSVMapper`, ported from `wsi_service/csv_mapper.py`.
//!
//! Reads a tab- or comma-separated definition file with configurable column indices:
//!
//! ```text
//! GROUP_1   GROUP_2   SLIDE_ID   CASE      PATH
//! <ID>      <VALUE>   <ID>       <CASE_ID> <WSI-PATH>
//! ```
//!
//! IDs are built as `{group_1}.{group_2}.{c|w}.{id}` — the same shape the LSAAI
//! integrations parse when deciding access, which is why the dotted form matters.
//!
//! Settings come from the `CSWS_` environment prefix, not `WS_`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use wsi_core::error::{ApiError, Result};
use wsi_core::models::{CaseLocalMapper, SlideLocalMapper, SlideStorage};
use wsi_core::settings::EnvSource;

use crate::secondary::collect_secondary_files;

const CACHE_FILE: &str = "local_mapper_csv.json";

/// `CSVMapperSettings` — `csv_mapper.py:18`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvMapperSettings {
    /// A `.csv`/`.tsv` file, or a directory searched recursively for them.
    pub source: String,
    pub separator: char,
    pub group_1: usize,
    pub group_2: usize,
    pub slide_id: usize,
    pub case_id: usize,
    pub path: usize,
}

impl Default for CsvMapperSettings {
    fn default() -> Self {
        Self {
            source: "data.csv".into(),
            separator: '\t',
            group_1: 0,
            group_2: 1,
            slide_id: 2,
            case_id: 3,
            path: 4,
        }
    }
}

impl CsvMapperSettings {
    pub fn load() -> Self {
        Self::from_source(&EnvSource::load_prefixed(".env", "csws_"))
    }

    pub fn from_source(src: &EnvSource) -> Self {
        let mut settings = Self::default();
        if let Some(value) = src.get_raw("source") {
            settings.source = value;
        }
        if let Some(value) = src.get_raw("separator") {
            // pydantic coerces to `str`; only the first character is used as a delimiter.
            if let Some(first) = unescape_separator(&value).chars().next() {
                settings.separator = first;
            }
        }
        let column = |name: &str| -> Option<usize> {
            src.get_raw(name)
                .and_then(|v| v.trim().parse::<usize>().ok())
        };
        if let Some(value) = column("group_1") {
            settings.group_1 = value;
        }
        if let Some(value) = column("group_2") {
            settings.group_2 = value;
        }
        if let Some(value) = column("slide_id") {
            settings.slide_id = value;
        }
        if let Some(value) = column("case_id") {
            settings.case_id = value;
        }
        if let Some(value) = column("path") {
            settings.path = value;
        }
        settings
    }
}

/// `\t` written literally in a `.env` file arrives as two characters.
fn unescape_separator(raw: &str) -> String {
    match raw {
        "\\t" => "\t".to_owned(),
        "\\n" => "\n".to_owned(),
        other => other.to_owned(),
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct MapperIndex {
    data_dir: String,
    cases: Vec<CaseLocalMapper>,
    slides: Vec<SlideLocalMapper>,
}

pub struct CsvMapper {
    data_dir: String,
    settings: CsvMapperSettings,
    /// Where the parsed index is cached. Python writes it to the working directory; that
    /// is the default here too, but making it explicit lets tests stay independent.
    cache_dir: PathBuf,
    index: Mutex<MapperIndex>,
    rebuild_lock: Mutex<()>,
}

impl CsvMapper {
    pub fn new(data_dir: impl Into<String>, settings: CsvMapperSettings) -> Result<Self> {
        Self::with_cache_dir(data_dir, settings, ".")
    }

    pub fn with_cache_dir(
        data_dir: impl Into<String>,
        settings: CsvMapperSettings,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let mapper = Self {
            data_dir: data_dir.into().replace('\\', "/"),
            settings,
            cache_dir: cache_dir.into(),
            index: Mutex::new(MapperIndex::default()),
            rebuild_lock: Mutex::new(()),
        };
        // `__init__` calls `refresh(force_refresh=False)`.
        wsi_core::storage::LocalMapper::refresh(&mapper, false)?;
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

    /// `_read_csv_data` — a file is read directly; a directory is walked for `.csv`/`.tsv`.
    fn build_index(&self) -> Result<MapperIndex> {
        let mut index = MapperIndex {
            data_dir: self.data_dir.clone(),
            ..Default::default()
        };
        let source = Path::new(&self.settings.source);

        if source.is_dir() {
            let mut found_data = false;
            let mut last_error: Option<ApiError> = None;
            for file in walk_definition_files(source) {
                match self.read_definition_file(&file, &mut index) {
                    Ok(()) => found_data = true,
                    Err(err) => last_error = Some(err),
                }
            }
            if !found_data {
                if last_error.is_some() {
                    tracing::error!(
                        "Directory {} does not contain a valid data definition .tsv or .csv files!",
                        self.settings.source
                    );
                    return Err(ApiError::internal("Invalid CSV source data!"));
                }
                tracing::info!(
                    "Directory {} does not contain any data.",
                    self.settings.source
                );
            }
        } else if source.is_file() {
            self.read_definition_file(source, &mut index).map_err(|_| {
                ApiError::internal("Target CSV data definition is not a valid file!")
            })?;
        } else {
            tracing::error!(
                "Path {} is neither a file nor a directory.",
                self.settings.source
            );
            return Err(ApiError::internal("Invalid CSV source data!"));
        }

        Ok(index)
    }

    /// `_read_csv_file` — one row per slide.
    fn read_definition_file(&self, path: &Path, index: &mut MapperIndex) -> Result<()> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            ApiError::internal(format!(
                "Failed to parse the CSV file! Is your syntax correct? ({e})"
            ))
        })?;

        let settings = &self.settings;
        let required = [
            settings.group_1,
            settings.group_2,
            settings.slide_id,
            settings.case_id,
            settings.path,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        for row in parse_delimited(&contents, settings.separator) {
            if row.len() <= required {
                // Python would raise IndexError, which `refresh` turns into a 500.
                return Err(ApiError::internal(
                    "Failed to parse the CSV file! Is your syntax correct?",
                ));
            }

            let group_1 = row[settings.group_1].clone();
            let group_2 = row[settings.group_2].clone();
            if group_1.len() > 5 || group_2.len() > 5 {
                // `assert len(group_1) <= 5` in `create_case_object`.
                return Err(ApiError::internal(
                    "Failed to parse the CSV file! Is your syntax correct?",
                ));
            }

            let context_id = row[settings.case_id].clone();
            let case_local_id = format!("{group_1}.{group_2}.c.{context_id}");
            let slide_local_id = format!("{group_1}.{group_2}.w.{}", row[settings.slide_id]);

            let addresses = collect_secondary_files(
                &row[settings.path],
                &slide_local_id,
                &slide_local_id,
                &self.data_dir,
            );
            let slide = SlideLocalMapper {
                id: slide_local_id.clone(),
                local_id: slide_local_id.clone(),
                slide_storage: SlideStorage {
                    slide_id: slide_local_id.clone(),
                    storage_type: "fs".into(),
                    storage_addresses: addresses,
                },
            };

            // Python looks the case up by the *raw* case id but stores it under the dotted
            // local id, so the lookup never hits and every row replaces the previous case
            // object — leaving each case holding only its last slide. Reproduced; see
            // docs/divergences.md.
            let case = CaseLocalMapper {
                id: case_local_id.clone(),
                local_id: case_local_id.clone(),
                slides: vec![slide_local_id.clone()],
            };
            match index.cases.iter().position(|c| c.id == case_local_id) {
                Some(position) => index.cases[position] = case,
                None => index.cases.push(case),
            }

            match index.slides.iter().position(|s| s.id == slide_local_id) {
                Some(position) => index.slides[position] = slide,
                None => index.slides.push(slide),
            }
        }
        Ok(())
    }
}

impl wsi_core::storage::LocalMapper for CsvMapper {
    fn get_cases(&self, context: Option<&str>) -> Result<Vec<CaseLocalMapper>> {
        if let Some(context) = context {
            tracing::warn!("CSVMapper: received unexpected context='{context}', ignoring.");
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
                        tracing::warn!("Could not persist CSV mapper cache: {err}");
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

/// Recursively collects `.csv` and `.tsv` files, in a deterministic order.
fn walk_definition_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut items: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        items.sort();
        for item in items {
            if item.is_dir() {
                stack.push(item);
            } else if matches!(
                item.extension().and_then(|e| e.to_str()),
                Some("csv") | Some("tsv")
            ) {
                found.push(item);
            }
        }
    }
    found.sort();
    found
}

/// A minimal RFC 4180 reader, matching Python's default `csv` dialect: double quotes
/// group a field, a doubled quote inside a quoted field is a literal quote, and newlines
/// inside quotes do not end the row.
pub fn parse_delimited(contents: &str, delimiter: char) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = contents.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(ch);
            }
            continue;
        }

        match ch {
            '"' => in_quotes = true,
            c if c == delimiter => row.push(std::mem::take(&mut field)),
            '\r' => {
                // Swallow CR; the following LF ends the row.
                if chars.peek() == Some(&'\n') {
                    continue;
                }
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            other => field.push(other),
        }
    }

    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }

    // Python's reader skips genuinely blank lines.
    rows.retain(|r| !(r.len() == 1 && r[0].is_empty()));
    rows
}

/// Builds a mapper using settings read from the `CSWS_` environment prefix.
pub fn from_env(data_dir: &str) -> Result<Arc<CsvMapper>> {
    Ok(Arc::new(CsvMapper::new(
        data_dir,
        CsvMapperSettings::load(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wsi_core::storage::LocalMapper;

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        /// Each fixture gets its own directory for both inputs and the index cache, so the
        /// tests stay independent under cargo's parallel runner. Changing the process
        /// working directory instead would make them race.
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("wsi-csv-mapper-{name}"));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.root.join(name);
            std::fs::write(&path, contents).unwrap();
            path
        }

        fn mapper(&self, source: &Path) -> Result<CsvMapper> {
            CsvMapper::with_cache_dir("/data", settings_for(source), &self.root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn settings_for(source: &Path) -> CsvMapperSettings {
        CsvMapperSettings {
            source: source.to_string_lossy().replace('\\', "/"),
            ..CsvMapperSettings::default()
        }
    }

    #[test]
    fn delimited_parsing_handles_quotes_and_embedded_newlines() {
        let rows = parse_delimited("a\tb\tc\n\"quoted\tfield\"\tx\ty\n", '\t');
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["a", "b", "c"]);
        assert_eq!(rows[1], vec!["quoted\tfield", "x", "y"]);

        let rows = parse_delimited("\"line\nbreak\",b\n", ',');
        assert_eq!(rows[0], vec!["line\nbreak", "b"]);

        let rows = parse_delimited("\"say \"\"hi\"\"\",b\n", ',');
        assert_eq!(rows[0], vec!["say \"hi\"", "b"]);
    }

    #[test]
    fn blank_lines_are_skipped_and_crlf_is_handled() {
        let rows = parse_delimited("a,b\r\n\r\nc,d\r\n", ',');
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["a", "b"]);
        assert_eq!(rows[1], vec!["c", "d"]);
    }

    #[test]
    fn ids_follow_the_dotted_group_scheme() {
        let f = Fixture::new("ids");
        let csv = f.write("data.tsv", "mmci\tadopt\tslide1\tcase1\tcase/one.svs\n");
        let mapper = f.mapper(&csv).unwrap();

        let cases = mapper.get_cases(None).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].id, "mmci.adopt.c.case1");

        let slide = mapper.get_slide("mmci.adopt.w.slide1").unwrap();
        assert_eq!(slide.id, "mmci.adopt.w.slide1");
        assert_eq!(slide.slide_storage.storage_type, "fs");
        assert_eq!(
            slide.slide_storage.storage_addresses[0].address,
            "case/one.svs"
        );
        assert!(slide.slide_storage.storage_addresses[0].main_address);
    }

    #[test]
    fn a_case_ends_up_with_only_its_last_slide() {
        // Python looks cases up by the raw case id but stores them under the dotted local
        // id, so each row replaces the case object. Reproduced deliberately.
        let f = Fixture::new("case-bug");
        let csv = f.write(
            "data.tsv",
            "mmci\tadopt\tslideA\tcase1\ta.svs\nmmci\tadopt\tslideB\tcase1\tb.svs\n",
        );
        let mapper = f.mapper(&csv).unwrap();

        let cases = mapper.get_cases(None).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(
            cases[0].slides,
            vec!["mmci.adopt.w.slideB"],
            "only the last row's slide survives"
        );

        // Both slides are still individually addressable.
        assert!(mapper.get_slide("mmci.adopt.w.slideA").is_ok());
        assert!(mapper.get_slide("mmci.adopt.w.slideB").is_ok());
    }

    #[test]
    fn a_group_longer_than_five_characters_is_rejected() {
        let f = Fixture::new("group-len");
        let csv = f.write("data.tsv", "toolonggroup\tadopt\ts\tc\tp.svs\n");
        let err = match f.mapper(&csv) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn a_short_row_is_a_parse_failure() {
        let f = Fixture::new("short-row");
        let csv = f.write("data.tsv", "mmci\tadopt\n");
        assert!(f.mapper(&csv).is_err());
    }

    #[test]
    fn a_missing_source_is_a_500() {
        let f = Fixture::new("missing-source");
        let err = match f.mapper(&f.root.join("nope.tsv")) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Invalid CSV source data!");
    }

    #[test]
    fn a_directory_source_is_walked_for_definition_files() {
        let f = Fixture::new("dir-source");
        std::fs::create_dir_all(f.root.join("defs/nested")).unwrap();
        std::fs::write(
            f.root.join("defs/a.tsv"),
            "mmci\tadopt\tslideA\tcase1\ta.svs\n",
        )
        .unwrap();
        std::fs::write(
            f.root.join("defs/nested/b.tsv"),
            "mmci\tadopt\tslideB\tcase2\tb.svs\n",
        )
        .unwrap();
        std::fs::write(f.root.join("defs/ignored.txt"), "not a definition").unwrap();

        let mapper = f.mapper(&f.root.join("defs")).unwrap();
        assert_eq!(mapper.get_cases(None).unwrap().len(), 2);
        assert!(mapper.get_slide("mmci.adopt.w.slideA").is_ok());
        assert!(mapper.get_slide("mmci.adopt.w.slideB").is_ok());
    }

    #[test]
    fn unknown_ids_are_404s_with_the_python_messages() {
        let f = Fixture::new("unknown");
        let csv = f.write("data.tsv", "mmci\tadopt\ts\tc\tp.svs\n");
        let mapper = f.mapper(&csv).unwrap();

        let err = match mapper.get_slide("nope") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Slide with slide_id nope does not exist");

        let err = match mapper.get_slides("nope") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Case with case_id nope does not exist");
    }

    #[test]
    fn separator_setting_accepts_an_escaped_tab() {
        assert_eq!(unescape_separator("\\t"), "\t");
        assert_eq!(unescape_separator(","), ",");
    }
}
