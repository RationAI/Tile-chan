//! The `mvt` plugin: passthrough vector and JSON tile datasets.
//!
//! Ported from `wsi_service_base_plugins/mvt/wsi_service_plugin_mvt/`.
//!
//! Four dataset shapes, all serving tile *bytes* rather than pixels:
//!
//! - a directory tree in `z/x/y.ext` layout;
//! - a single tile file (`.mvt` `.pbf` `.json` `.geojson`, gzipped or not);
//! - a manifest JSON pointing at a directory tree;
//! - an `.mbtiles` SQLite database.
//!
//! Service level 0 is the *highest* available zoom, so levels run coarse-ward as the zoom
//! number falls. `get_region` and `get_thumbnail` are 501 by design; `get_tile` returns
//! `Pixels::Raw`, which the response pipeline passes through untouched for the vector
//! formats and rejects for the raster ones — the same outcome Python's `make_response`
//! reaches.
//!
//! **The Python plugin cannot serve anything.** `_build_slide_info` omits `format`, which
//! `SlideInfo` requires, so constructing it raises `ValidationError` and every dataset
//! answers 500. This port supplies `format = "mvt"` and works; see
//! `docs/divergences.md`, divergence 26.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::{IccIntent, IccProfileCache};
use wsi_core::models::{SlideExtent, SlideInfo, SlideLevel, SlidePixelSizeNm};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::slide::{ReadOptions, Slide};

/// `supported_file_extensions` — the single suffixes plugin discovery matches.
const SUPPORTED_FILE_EXTENSIONS: [&str; 5] = ["json", "geojson", "mvt", "pbf", "mbtiles"];

/// `SUPPORTED_TILE_SUFFIXES` — the default lookup order inside a directory dataset.
const SUPPORTED_TILE_SUFFIXES: [&str; 8] = [
    ".mvt",
    ".pbf",
    ".json",
    ".geojson",
    ".mvt.gz",
    ".pbf.gz",
    ".json.gz",
    ".geojson.gz",
];

/// `MANIFEST_HINT_KEYS` — a JSON object carrying any of these is treated as a manifest
/// rather than as a tile payload.
const MANIFEST_HINT_KEYS: [&str; 12] = [
    "root",
    "tile_root",
    "tile_size",
    "max_zoom",
    "min_zoom",
    "tile_suffix",
    "tile_suffixes",
    "scheme",
    "zoom_levels",
    "levels",
    "bounds_by_zoom",
    "tiles",
];

const DEFAULT_TILE_SIZE: i64 = 256;

pub struct MvtPlugin;

impl MvtPlugin {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MvtPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WsiPlugin for MvtPlugin {
    fn name(&self) -> &'static str {
        "mvt"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// The `supported_file_extensions` membership test, which applies to directories too:
    /// `pathlib.Path` reports a suffix for any name, so a directory dataset is only
    /// discoverable when its own name ends in one of these.
    fn is_supported(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| SUPPORTED_FILE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let slide = tokio::task::spawn_blocking(move || MvtSlide::open(path))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

/// `max_x` / `max_y` for one zoom, in tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ZoomRange {
    max_x: i64,
    max_y: i64,
}

impl ZoomRange {
    /// The full grid a zoom level can hold: `2^zoom - 1` in both axes.
    fn full(zoom: i64) -> Self {
        let span = 1i64.checked_shl(zoom.clamp(0, 62) as u32).unwrap_or(1) - 1;
        Self {
            max_x: span,
            max_y: span,
        }
    }
}

/// Where tiles come from, after the dataset has been identified.
enum Backing {
    /// A `z/x/y.ext` tree rooted here.
    Directory(PathBuf),
    /// One file served as level 0, tile `(0, 0)`.
    SingleTile(Bytes),
    /// An `.mbtiles` database. `None` once closed.
    MbTiles(Mutex<Option<Connection>>),
}

pub struct MvtSlide {
    source: PathBuf,
    backing: Backing,
    /// `self.scheme` — `"tms"` flips the Y axis on lookup.
    scheme: String,
    tile_suffixes: Vec<String>,
    zoom_ranges: BTreeMap<i64, ZoomRange>,
    /// `self._level_to_zoom` — service level index to zoom number, highest zoom first.
    level_to_zoom: Vec<i64>,
    info: SlideInfo,
}

impl MvtSlide {
    fn open(source: PathBuf) -> Result<Self> {
        // Defaults, before a manifest or the MBTiles metadata table gets a say.
        let mut tile_size = DEFAULT_TILE_SIZE;
        let mut scheme = "xyz".to_owned();
        let mut tile_suffixes: Vec<String> = SUPPORTED_TILE_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect();

        let name = source
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();

        let (backing, zoom_ranges) = if source.is_dir() {
            let root = source.canonicalize().unwrap_or_else(|_| source.clone());
            let ranges = scan_zoom_ranges(&root, &tile_suffixes);
            if ranges.is_empty() {
                return Err(ApiError::bad_request(format!(
                    "No tile zoom directories found in {}.",
                    root.display()
                )));
            }
            (Backing::Directory(root), ranges)
        } else if name.ends_with(".mbtiles") {
            open_mbtiles(&source, &mut tile_size, &mut scheme)?
        } else {
            open_file_dataset(
                &source,
                &name,
                &mut tile_size,
                &mut scheme,
                &mut tile_suffixes,
            )?
        };

        let (info, level_to_zoom) = build_slide_info(&zoom_ranges, tile_size);
        Ok(Self {
            source,
            backing,
            scheme,
            tile_suffixes,
            zoom_ranges,
            level_to_zoom,
            info,
        })
    }

    /// `_to_storage_tile_y` — the TMS/XYZ flip, using the zoom's own extent when known.
    fn storage_tile_y(&self, zoom: i64, tile_y: i64) -> i64 {
        if self.scheme != "tms" {
            return tile_y;
        }
        let max_y = self
            .zoom_ranges
            .get(&zoom)
            .map(|r| r.max_y)
            .unwrap_or_else(|| ZoomRange::full(zoom).max_y);
        max_y - tile_y
    }
}

/// `_open_mbtiles_dataset`.
fn open_mbtiles(
    source: &Path,
    tile_size: &mut i64,
    scheme: &mut String,
) -> Result<(Backing, BTreeMap<i64, ZoomRange>)> {
    // Python opens read-write, but every statement here is a read; opening read-only
    // cannot create an empty database where `load_slide` already found a file.
    let connection = Connection::open_with_flags(source, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| ApiError::bad_request(format!("Failed to open MBTiles file: {e}")))?;

    // An MBTiles archive is TMS unless its metadata says otherwise.
    *scheme = "tms".to_owned();
    let metadata = read_mbtiles_metadata(&connection);
    if let Some(value) = metadata.get("tile_size") {
        *tile_size = python_int_from_str(value)?;
    }
    if let Some(value) = metadata.get("scheme").filter(|s| !s.is_empty()) {
        *scheme = value.to_ascii_lowercase();
    }

    let zooms = fetch_mbtiles_zooms(&connection)?;
    if zooms.is_empty() {
        return Err(ApiError::bad_request(
            "MBTiles file does not contain any tiles.",
        ));
    }
    let ranges = zooms
        .into_iter()
        .map(|zoom| (zoom, ZoomRange::full(zoom)))
        .collect();
    Ok((Backing::MbTiles(Mutex::new(Some(connection))), ranges))
}

/// `SELECT name, value FROM metadata`, with a missing or unreadable table treated as
/// empty — Python swallows `sqlite3.Error` here but nowhere else.
fn read_mbtiles_metadata(connection: &Connection) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(mut statement) = connection.prepare("SELECT name, value FROM metadata") else {
        return out;
    };
    let Ok(rows) = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0).unwrap_or_default(),
            row.get::<_, String>(1).unwrap_or_default(),
        ))
    }) else {
        return out;
    };
    for row in rows.flatten() {
        out.insert(row.0, row.1);
    }
    out
}

/// `_fetch_mbtiles_zooms`. Unlike the metadata read, a failure here is not caught in
/// Python, so it escapes as a 500.
fn fetch_mbtiles_zooms(connection: &Connection) -> Result<Vec<i64>> {
    let sql = "SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level";
    let mut statement = connection
        .prepare(sql)
        .map_err(|e| ApiError::internal(format!("{e}")))?;
    let rows = statement
        .query_map([], |row| row.get::<_, i64>(0))
        .map_err(|e| ApiError::internal(format!("{e}")))?;
    rows.collect::<rusqlite::Result<Vec<i64>>>()
        .map_err(|e| ApiError::internal(format!("{e}")))
}

/// `_open_file_dataset`.
fn open_file_dataset(
    source: &Path,
    name: &str,
    tile_size: &mut i64,
    scheme: &mut String,
    tile_suffixes: &mut Vec<String>,
) -> Result<(Backing, BTreeMap<i64, ZoomRange>)> {
    if !source.is_file() {
        return Err(ApiError::not_found(format!(
            "Tile dataset does not exist: {}",
            source.display()
        )));
    }

    if matches_any_suffix(name, tile_suffixes) {
        let raw = std::fs::read(source)
            .map_err(|e| ApiError::internal(format!("Failed to read tile dataset: {e}")))?;
        // Only an uncompressed `.json`/`.geojson` is examined for a manifest: a
        // `.json.gz` name does not end in `.json`, so it is always a tile payload.
        if name.ends_with(".json") || name.ends_with(".geojson") {
            if let Some(manifest) = try_parse_manifest(name, &raw) {
                return open_manifest_dataset(source, &manifest, tile_size, scheme, tile_suffixes);
            }
        }
        return Ok((Backing::SingleTile(Bytes::from(raw)), single_tile_ranges()));
    }

    // Unreachable through plugin discovery — every accepted single suffix also ends one of
    // the tile suffixes, except `.mbtiles`, which is handled before this — but a caller
    // reaching in directly still gets Python's behaviour.
    let text = std::fs::read_to_string(source)
        .map_err(|_| ApiError::bad_request(format!("Unsupported tile dataset file: {name}")))?;
    let manifest: Value = serde_json::from_str(&text)
        .map_err(|_| ApiError::bad_request(format!("Unsupported tile dataset file: {name}")))?;
    if !is_manifest_object(&manifest) {
        return Err(ApiError::bad_request(
            "JSON input is not recognized as a manifest. \
             Expected keys like root, tile_suffixes, max_zoom, min_zoom, scheme, zoom_levels, or levels.",
        ));
    }
    open_manifest_dataset(source, &manifest, tile_size, scheme, tile_suffixes)
}

/// `_set_single_tile` — one tile at level 0.
fn single_tile_ranges() -> BTreeMap<i64, ZoomRange> {
    BTreeMap::from([(0, ZoomRange { max_x: 0, max_y: 0 })])
}

/// `_open_manifest_dataset`.
fn open_manifest_dataset(
    source: &Path,
    manifest: &Value,
    tile_size: &mut i64,
    scheme: &mut String,
    tile_suffixes: &mut Vec<String>,
) -> Result<(Backing, BTreeMap<i64, ZoomRange>)> {
    // `manifest.get("root", manifest.get("tile_root", "."))` — a present-but-null `root`
    // reaches `Path / None` in Python and raises, so anything but a string is an error.
    let root_value = match manifest.get("root") {
        Some(value) => value,
        None => manifest.get("tile_root").unwrap_or(&Value::Null),
    };
    let root_value = match root_value {
        Value::Null => ".",
        Value::String(text) => text.as_str(),
        other => {
            return Err(ApiError::internal(format!(
                "unsupported operand type(s) for /: 'PosixPath' and '{}'",
                json_type_name(other)
            )))
        }
    };

    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    let joined = parent.join(root_value);
    let root = joined.canonicalize().unwrap_or(joined);
    if !root.is_dir() {
        return Err(ApiError::bad_request(format!(
            "Manifest root does not exist or is not a directory: {}",
            root.display()
        )));
    }

    if let Some(value) = manifest.get("tile_size") {
        *tile_size = python_int(value)?;
    }
    if let Some(value) = manifest.get("scheme") {
        *scheme = json_str(value).to_ascii_lowercase();
    }

    // `tile_suffixes`, or a single `tile_suffix`; each entry gains a leading dot.
    let declared = match manifest.get("tile_suffixes") {
        Some(Value::Array(items)) => Some(items.clone()),
        Some(Value::Null) | None => manifest
            .get("tile_suffix")
            .filter(|v| !v.is_null())
            .map(|v| vec![v.clone()]),
        // A non-list `tile_suffixes` is iterated in Python; a string would yield its
        // characters, which is not worth reproducing, and anything else raises.
        Some(other) => {
            return Err(ApiError::internal(format!(
                "'{}' object is not iterable",
                json_type_name(other)
            )))
        }
    };
    if let Some(items) = declared {
        *tile_suffixes = items
            .iter()
            .map(|item| {
                let text = json_str(item);
                if text.starts_with('.') {
                    text
                } else {
                    format!(".{text}")
                }
            })
            .collect();
    }

    let mut ranges = zoom_ranges_from_manifest(manifest)?;
    if ranges.is_empty() {
        ranges = scan_zoom_ranges(&root, tile_suffixes);
    }
    if ranges.is_empty() {
        return Err(ApiError::bad_request(
            "Manifest did not resolve to any readable tile zoom levels.",
        ));
    }
    Ok((Backing::Directory(root), ranges))
}

/// `_zoom_ranges_from_manifest`.
///
/// The four sources are applied in Python's order, and the difference between them
/// matters: `bounds_by_zoom` and the dict form of `levels` overwrite, while the integer
/// form of `levels`, `zoom_levels` and the `min_zoom`/`max_zoom` span only fill gaps.
fn zoom_ranges_from_manifest(manifest: &Value) -> Result<BTreeMap<i64, ZoomRange>> {
    let mut result: BTreeMap<i64, ZoomRange> = BTreeMap::new();

    if let Some(Value::Object(bounds)) = manifest.get("bounds_by_zoom") {
        for (zoom_key, dims) in bounds {
            let zoom = python_int_from_str(zoom_key)?;
            let range = match dims {
                Value::Object(map) => {
                    // `dims.get("x_tiles", 1) - 1` is evaluated eagerly, so a missing
                    // `x_tiles` contributes 0, not a skip.
                    let axis = |explicit: &str, tiles: &str| -> Result<i64> {
                        match map.get(explicit) {
                            Some(value) => python_int(value),
                            None => Ok(match map.get(tiles) {
                                Some(value) => python_int(value)? - 1,
                                None => 0,
                            }),
                        }
                    };
                    ZoomRange {
                        max_x: axis("max_x", "x_tiles")?,
                        max_y: axis("max_y", "y_tiles")?,
                    }
                }
                Value::Array(pair) if pair.len() >= 2 => ZoomRange {
                    max_x: python_int(&pair[0])?,
                    max_y: python_int(&pair[1])?,
                },
                other => {
                    return Err(ApiError::internal(format!(
                        "'{}' object is not subscriptable",
                        json_type_name(other)
                    )))
                }
            };
            result.insert(zoom, range);
        }
    }

    if let Some(Value::Array(levels)) = manifest.get("levels") {
        for item in levels {
            match item {
                Value::Object(map) => {
                    let Some(zoom) = map.get("zoom") else {
                        return Err(ApiError::internal("'zoom'"));
                    };
                    let zoom = python_int(zoom)?;
                    if let (Some(max_x), Some(max_y)) = (map.get("max_x"), map.get("max_y")) {
                        result.insert(
                            zoom,
                            ZoomRange {
                                max_x: python_int(max_x)?,
                                max_y: python_int(max_y)?,
                            },
                        );
                    } else if let (Some(x_tiles), Some(y_tiles)) =
                        (map.get("x_tiles"), map.get("y_tiles"))
                    {
                        result.insert(
                            zoom,
                            ZoomRange {
                                max_x: python_int(x_tiles)? - 1,
                                max_y: python_int(y_tiles)? - 1,
                            },
                        );
                    }
                }
                // Only a bare integer, not a float or a numeric string: Python's test is
                // `isinstance(item, int)`.
                Value::Number(number) if number.is_i64() || number.is_u64() => {
                    let zoom = python_int(item)?;
                    result.entry(zoom).or_insert_with(|| ZoomRange::full(zoom));
                }
                _ => {}
            }
        }
    }

    if let Some(Value::Array(zoom_levels)) = manifest.get("zoom_levels") {
        for item in zoom_levels {
            let zoom = python_int(item)?;
            result.entry(zoom).or_insert_with(|| ZoomRange::full(zoom));
        }
    }

    match (manifest.get("min_zoom"), manifest.get("max_zoom")) {
        (Some(min), Some(max)) if !min.is_null() && !max.is_null() => {
            for zoom in python_int(min)?..=python_int(max)? {
                result.entry(zoom).or_insert_with(|| ZoomRange::full(zoom));
            }
        }
        _ => {}
    }

    Ok(result)
}

/// `_scan_zoom_ranges` — walk `root/<zoom>/<x>/<y><suffix>` and record the observed maxima.
fn scan_zoom_ranges(root: &Path, tile_suffixes: &[String]) -> BTreeMap<i64, ZoomRange> {
    let mut zoom_ranges = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return zoom_ranges;
    };
    let mut zoom_dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    zoom_dirs.sort();

    for zoom_dir in zoom_dirs {
        let Some(zoom) = digits_only(&zoom_dir) else {
            continue;
        };
        if !zoom_dir.is_dir() {
            continue;
        }
        let (mut max_x, mut max_y) = (-1i64, -1i64);
        let Ok(x_entries) = std::fs::read_dir(&zoom_dir) else {
            continue;
        };
        for x_dir in x_entries.flatten().map(|e| e.path()) {
            let Some(x_value) = digits_only(&x_dir) else {
                continue;
            };
            if !x_dir.is_dir() {
                continue;
            }
            max_x = max_x.max(x_value);
            let Ok(tiles) = std::fs::read_dir(&x_dir) else {
                continue;
            };
            for tile in tiles.flatten().map(|e| e.path()) {
                let Some(file_name) = tile.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !tile.is_file() || !matches_any_suffix(file_name, tile_suffixes) {
                    continue;
                }
                if let Some(y_value) = parse_tile_y(file_name) {
                    max_y = max_y.max(y_value);
                }
            }
        }
        if max_x >= 0 && max_y >= 0 {
            zoom_ranges.insert(zoom, ZoomRange { max_x, max_y });
        }
    }
    zoom_ranges
}

/// `int(dir.name) if dir.name.isdigit()`.
fn digits_only(path: &Path) -> Option<i64> {
    let name = path.file_name()?.to_str()?;
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    name.parse::<i64>().ok()
}

/// `_TILE_Y_RE`: `^(?P<y>\d+)(?:\.(?:mvt|pbf|json|geojson))?(?:\.gz)?$`.
///
/// Note the pattern hard-codes the four standard extensions, so a manifest that declares
/// some other suffix passes the membership test and then fails to yield a `y` — which
/// leaves that zoom out of the scan. Reproduced rather than fixed.
fn parse_tile_y(file_name: &str) -> Option<i64> {
    let stem = file_name.strip_suffix(".gz").unwrap_or(file_name);
    let stem = ["mvt", "pbf", "json", "geojson"]
        .iter()
        .find_map(|ext| stem.strip_suffix(&format!(".{ext}")))
        .unwrap_or(stem);
    if stem.is_empty() || !stem.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    stem.parse::<i64>().ok()
}

fn matches_any_suffix(name: &str, suffixes: &[String]) -> bool {
    suffixes
        .iter()
        .any(|suffix| name.ends_with(suffix.as_str()))
}

/// `_try_parse_manifest`.
fn try_parse_manifest(name: &str, raw: &[u8]) -> Option<Value> {
    // Retained from the original even though it is unreachable: the caller only passes
    // names ending in `.json`/`.geojson`, which never also end in `.gz`.
    let payload = if name.ends_with(".gz") {
        gunzip(raw).ok()?
    } else {
        raw.to_vec()
    };
    let text = String::from_utf8(payload).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    is_manifest_object(&value).then_some(value)
}

/// `_is_manifest_object`.
fn is_manifest_object(value: &Value) -> bool {
    value
        .as_object()
        .map(|map| MANIFEST_HINT_KEYS.iter().any(|key| map.contains_key(*key)))
        .unwrap_or(false)
}

fn gunzip(raw: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(raw).read_to_end(&mut out)?;
    Ok(out)
}

/// `data[:2] == b"\x1f\x8b"` then `gzip.decompress`.
fn inflate_if_gzipped(data: Vec<u8>) -> Result<Bytes> {
    if data.starts_with(&[0x1f, 0x8b]) {
        let inflated = gunzip(&data)
            .map_err(|e| ApiError::internal(format!("Failed to decompress tile: {e}")))?;
        return Ok(Bytes::from(inflated));
    }
    Ok(Bytes::from(data))
}

/// `int(value)` for a JSON value: numbers truncate toward zero, strings must be integral,
/// and anything else is the `TypeError` Python would raise.
fn python_int(value: &Value) -> Result<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|f| f.trunc() as i64))
            .ok_or_else(|| ApiError::internal("int() argument out of range")),
        Value::String(text) => python_int_from_str(text),
        Value::Bool(flag) => Ok(i64::from(*flag)),
        other => Err(ApiError::internal(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            json_type_name(other)
        ))),
    }
}

fn python_int_from_str(text: &str) -> Result<i64> {
    text.trim().parse::<i64>().map_err(|_| {
        ApiError::internal(format!("invalid literal for int() with base 10: '{text}'"))
    })
}

/// `str(value)` for the fields Python stringifies before lowercasing.
fn json_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) => {
            if number.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `_build_slide_info`.
///
/// `format` is the one field Python leaves out — and the reason the Python plugin answers
/// 500 for every dataset. It is supplied here.
fn build_slide_info(
    zoom_ranges: &BTreeMap<i64, ZoomRange>,
    tile_size: i64,
) -> (SlideInfo, Vec<i64>) {
    let fallback = single_tile_ranges();
    let ranges = if zoom_ranges.is_empty() {
        &fallback
    } else {
        zoom_ranges
    };

    // Highest zoom first: service level 0 is the finest available zoom.
    let level_to_zoom: Vec<i64> = ranges.keys().rev().copied().collect();
    let max_zoom = level_to_zoom[0];
    let levels: Vec<SlideLevel> = level_to_zoom
        .iter()
        .map(|zoom| {
            let range = ranges[zoom];
            SlideLevel {
                extent: SlideExtent::new(
                    (range.max_x + 1) * tile_size,
                    (range.max_y + 1) * tile_size,
                    1,
                ),
                downsample_factor: 2f64.powi((max_zoom - zoom) as i32),
            }
        })
        .collect();

    let info = SlideInfo {
        id: String::new(),
        // An empty list, not `null`: Python passes `channels=[]`.
        channels: Some(Vec::new()),
        channel_depth: Some(8),
        extent: levels[0].extent,
        num_levels: levels.len() as i64,
        pixel_size_nm: SlidePixelSizeNm::xy(-1.0, -1.0),
        tile_extent: SlideExtent::new(tile_size, tile_size, 1),
        levels,
        format: "mvt".into(),
        raw_download: false,
    };
    (info, level_to_zoom)
}

#[async_trait]
impl Slide for MvtSlide {
    async fn close(&self) -> Result<()> {
        if let Backing::MbTiles(connection) = &self.backing {
            // Dropping the handle is what `sqlite3.Connection.close()` does.
            *connection.lock() = None;
        }
        Ok(())
    }

    async fn get_info(&self) -> Result<SlideInfo> {
        Ok(self.info.clone())
    }

    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        _opts: ReadOptions,
    ) -> Result<Pixels> {
        let Some(&zoom) = self.level_to_zoom.get(level) else {
            return Err(ApiError::not_found("Requested level does not exist."));
        };

        match &self.backing {
            Backing::SingleTile(bytes) => {
                if level == 0 && tile_x == 0 && tile_y == 0 {
                    Ok(Pixels::Raw(bytes.clone()))
                } else {
                    Err(ApiError::not_found("Tile does not exist."))
                }
            }
            Backing::MbTiles(connection) => {
                let storage_y = self.storage_tile_y(zoom, tile_y);
                let guard = connection.lock();
                let Some(connection) = guard.as_ref() else {
                    return Err(ApiError::internal("MBTiles connection is not initialized."));
                };
                let data = connection
                    .query_row(
                        "SELECT tile_data FROM tiles \
                         WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?",
                        (zoom, tile_x, storage_y),
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .map_err(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => {
                            ApiError::not_found("Tile does not exist.")
                        }
                        other => ApiError::internal(format!("{other}")),
                    })?;
                Ok(Pixels::Raw(inflate_if_gzipped(data)?))
            }
            Backing::Directory(root) => {
                let lookup_y = self.storage_tile_y(zoom, tile_y);
                for suffix in &self.tile_suffixes {
                    let candidate = root
                        .join(zoom.to_string())
                        .join(tile_x.to_string())
                        .join(format!("{lookup_y}{suffix}"));
                    if candidate.is_file() {
                        let data = std::fs::read(&candidate)
                            .map_err(|e| ApiError::internal(format!("Failed to read tile: {e}")))?;
                        return Ok(Pixels::Raw(inflate_if_gzipped(data)?));
                    }
                }
                Err(ApiError::not_found("Tile does not exist."))
            }
        }
    }

    async fn get_region(
        &self,
        _level: usize,
        _start_x: i64,
        _start_y: i64,
        _size_x: u32,
        _size_y: u32,
        _opts: ReadOptions,
    ) -> Result<Pixels> {
        Err(ApiError::not_implemented(
            "Region access is not supported for vector tile datasets.",
        ))
    }

    async fn get_thumbnail(
        &self,
        _max_x: u32,
        _max_y: u32,
        _icc_profile_intent: Option<IccIntent>,
        _icc_profile_strict: bool,
    ) -> Result<Pixels> {
        Err(ApiError::not_implemented(
            "Thumbnail is not supported for vector tile datasets.",
        ))
    }

    async fn get_label(&self) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image label does not exist.",
        ))
    }

    async fn get_macro(&self, _: Option<IccIntent>, _: bool) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image macro does not exist.",
        ))
    }

    /// Python returns `b""`, which `get_for_payload` then rejects — as the plain-text 500
    /// of divergence 11, because it raises `http.client.HTTPException`, not FastAPI's.
    async fn get_icc_profile(&self) -> Result<Bytes> {
        IccProfileCache::payload(Some(&[] as &[u8]))
    }

    fn filepath(&self) -> &Path {
        &self.source
    }

    fn plugin_name(&self) -> &str {
        "mvt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory per test, so the scans see only what the test put there.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("wsi-mvt-plugin-tests").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tile(root: &Path, zoom: i64, x: i64, name: &str, body: &[u8]) {
        let dir = root.join(zoom.to_string()).join(x.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn raw(pixels: Pixels) -> Bytes {
        match pixels {
            Pixels::Raw(bytes) => bytes,
            other => panic!("expected raw bytes, got {other:?}"),
        }
    }

    fn gzipped(body: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use std::io::Write;
        let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(body).unwrap();
        encoder.finish().unwrap()
    }

    /// `unwrap_err` needs `T: Debug`, which an open slide cannot provide.
    fn open_err(path: PathBuf) -> ApiError {
        match MvtSlide::open(path) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        }
    }

    #[test]
    fn discovery_matches_the_single_suffix_list() {
        let plugin = MvtPlugin::new();
        for name in ["a.json", "a.geojson", "a.mvt", "a.pbf", "a.mbtiles"] {
            assert!(plugin.is_supported(Path::new(name)), "{name}");
        }
        // Only the final suffix counts, which is why the README tells operators to select
        // the plugin explicitly for `.mvt.gz`.
        assert!(!plugin.is_supported(Path::new("a.mvt.gz")));
        assert!(!plugin.is_supported(Path::new("a.svs")));
        assert!(!plugin.is_supported(Path::new("noextension")));
        // A directory is discoverable only through its own name.
        assert!(plugin.is_supported(Path::new("tiles.mvt")));
    }

    #[tokio::test]
    async fn a_directory_dataset_maps_level_zero_to_the_highest_zoom() {
        let dir = scratch("directory");
        write_tile(&dir, 0, 0, "0.mvt", b"z0");
        write_tile(&dir, 1, 0, "0.mvt", b"z1-0-0");
        write_tile(&dir, 1, 1, "1.mvt", b"z1-1-1");

        let slide = MvtSlide::open(dir.clone()).unwrap();
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.num_levels, 2);
        assert_eq!(info.format, "mvt");
        assert_eq!(info.tile_extent, SlideExtent::new(256, 256, 1));
        // Zoom 1 is level 0: two tiles per axis at 256 each.
        assert_eq!(info.extent, SlideExtent::new(512, 512, 1));
        assert_eq!(info.levels[0].downsample_factor, 1.0);
        assert_eq!(info.levels[1].extent, SlideExtent::new(256, 256, 1));
        assert_eq!(info.levels[1].downsample_factor, 2.0);
        assert_eq!(info.channels.as_ref().unwrap().len(), 0);
        assert_eq!(info.pixel_size_nm.x, -1.0);

        assert_eq!(
            &raw(slide
                .get_tile(0, 1, 1, ReadOptions::default())
                .await
                .unwrap())[..],
            b"z1-1-1"
        );
        assert_eq!(
            &raw(slide
                .get_tile(1, 0, 0, ReadOptions::default())
                .await
                .unwrap())[..],
            b"z0"
        );
        assert_eq!(
            slide
                .get_tile(0, 9, 9, ReadOptions::default())
                .await
                .unwrap_err()
                .status,
            http::StatusCode::NOT_FOUND
        );
        assert_eq!(
            slide
                .get_tile(5, 0, 0, ReadOptions::default())
                .await
                .unwrap_err()
                .detail_text(),
            "Requested level does not exist."
        );
    }

    #[tokio::test]
    async fn a_gzipped_tile_is_inflated_on_the_way_out() {
        let dir = scratch("gzipped");
        write_tile(&dir, 0, 0, "0.mvt.gz", &gzipped(b"inflated"));
        let slide = MvtSlide::open(dir).unwrap();
        assert_eq!(
            &raw(slide
                .get_tile(0, 0, 0, ReadOptions::default())
                .await
                .unwrap())[..],
            b"inflated"
        );
    }

    #[test]
    fn a_directory_without_zoom_folders_is_rejected() {
        let dir = scratch("empty");
        std::fs::write(dir.join("readme.txt"), b"nothing here").unwrap();
        let err = open_err(dir);
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert!(err.detail_text().starts_with("No tile zoom directories"));
    }

    #[tokio::test]
    async fn a_single_tile_file_serves_only_level_zero_at_the_origin() {
        let dir = scratch("single");
        let path = dir.join("tile.mvt");
        std::fs::write(&path, b"payload").unwrap();

        let slide = MvtSlide::open(path).unwrap();
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.num_levels, 1);
        assert_eq!(info.extent, SlideExtent::new(256, 256, 1));
        assert_eq!(
            &raw(slide
                .get_tile(0, 0, 0, ReadOptions::default())
                .await
                .unwrap())[..],
            b"payload"
        );
        assert_eq!(
            slide
                .get_tile(0, 1, 0, ReadOptions::default())
                .await
                .unwrap_err()
                .detail_text(),
            "Tile does not exist."
        );
    }

    #[tokio::test]
    async fn a_geojson_payload_without_manifest_keys_is_a_single_tile() {
        let dir = scratch("geojson");
        let path = dir.join("features.geojson");
        std::fs::write(&path, br#"{"type":"FeatureCollection","features":[]}"#).unwrap();
        let slide = MvtSlide::open(path).unwrap();
        assert_eq!(slide.get_info().await.unwrap().num_levels, 1);
        assert!(matches!(slide.backing, Backing::SingleTile(_)));
    }

    #[tokio::test]
    async fn a_manifest_resolves_its_root_and_zoom_span() {
        let dir = scratch("manifest");
        let tiles = dir.join("tiles");
        write_tile(&tiles, 2, 3, "1.pbf", b"deep");
        let path = dir.join("dataset.json");
        std::fs::write(
            &path,
            br#"{"root": "tiles", "tile_size": 512, "min_zoom": 1, "max_zoom": 2,
                 "tile_suffixes": ["pbf", ".mvt"]}"#,
        )
        .unwrap();

        let slide = MvtSlide::open(path).unwrap();
        assert_eq!(slide.tile_suffixes, vec![".pbf", ".mvt"]);
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.num_levels, 2);
        assert_eq!(info.tile_extent, SlideExtent::new(512, 512, 1));
        // `min_zoom`/`max_zoom` only fill gaps, with the full 2^zoom grid.
        assert_eq!(info.extent, SlideExtent::new(4 * 512, 4 * 512, 1));
        assert_eq!(
            &raw(slide
                .get_tile(0, 3, 1, ReadOptions::default())
                .await
                .unwrap())[..],
            b"deep"
        );
    }

    #[tokio::test]
    async fn bounds_by_zoom_overrides_and_levels_fill_gaps() {
        let dir = scratch("bounds");
        let tiles = dir.join("t");
        write_tile(&tiles, 1, 0, "0.mvt", b"x");
        let path = dir.join("m.json");
        std::fs::write(
            &path,
            br#"{"root": "t",
                 "bounds_by_zoom": {"1": {"max_x": 1, "max_y": 2}, "2": [3, 3]},
                 "levels": [{"zoom": 1, "x_tiles": 9, "y_tiles": 9}, 5],
                 "zoom_levels": [2, 7]}"#,
        )
        .unwrap();

        let slide = MvtSlide::open(path).unwrap();
        // `levels` in dict form overwrites `bounds_by_zoom`; `zoom_levels` does not.
        assert_eq!(slide.zoom_ranges[&1], ZoomRange { max_x: 8, max_y: 8 });
        assert_eq!(slide.zoom_ranges[&2], ZoomRange { max_x: 3, max_y: 3 });
        assert_eq!(slide.zoom_ranges[&5], ZoomRange::full(5));
        assert_eq!(slide.zoom_ranges[&7], ZoomRange::full(7));
        // Level 0 is zoom 7, the highest present.
        assert_eq!(slide.level_to_zoom.first(), Some(&7));
        assert_eq!(
            slide.get_info().await.unwrap().levels[0].downsample_factor,
            1.0
        );
    }

    #[test]
    fn x_tiles_alone_in_bounds_by_zoom_yields_the_eager_default() {
        // `dims.get("max_y", dims.get("y_tiles", 1) - 1)` with neither key present is 0.
        let manifest: Value = serde_json::from_str(r#"{"bounds_by_zoom": {"3": {}}}"#).unwrap();
        let ranges = zoom_ranges_from_manifest(&manifest).unwrap();
        assert_eq!(ranges[&3], ZoomRange { max_x: 0, max_y: 0 });
    }

    #[test]
    fn a_manifest_pointing_at_a_missing_root_is_rejected() {
        let dir = scratch("badroot");
        let path = dir.join("m.json");
        std::fs::write(&path, br#"{"root": "nope"}"#).unwrap();
        let err = open_err(path);
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert!(err
            .detail_text()
            .starts_with("Manifest root does not exist"));
    }

    #[test]
    fn a_manifest_that_resolves_no_zoom_levels_is_rejected() {
        let dir = scratch("nozooms");
        std::fs::create_dir_all(dir.join("t")).unwrap();
        let path = dir.join("m.json");
        std::fs::write(&path, br#"{"root": "t"}"#).unwrap();
        let err = open_err(path);
        assert_eq!(
            err.detail_text(),
            "Manifest did not resolve to any readable tile zoom levels."
        );
    }

    #[test]
    fn a_missing_file_is_a_404() {
        let dir = scratch("missing");
        let err = open_err(dir.join("gone.mvt"));
        assert_eq!(err.status, http::StatusCode::NOT_FOUND);
        assert!(err.detail_text().starts_with("Tile dataset does not exist"));
    }

    #[test]
    fn the_tile_y_pattern_only_accepts_the_four_standard_extensions() {
        assert_eq!(parse_tile_y("12.mvt"), Some(12));
        assert_eq!(parse_tile_y("12.json.gz"), Some(12));
        assert_eq!(parse_tile_y("12.gz"), Some(12));
        assert_eq!(parse_tile_y("12"), Some(12));
        // A manifest-declared suffix outside the pattern yields no `y`.
        assert_eq!(parse_tile_y("12.vector"), None);
        assert_eq!(parse_tile_y("tile.mvt"), None);
    }

    // --- MBTiles ------------------------------------------------------------------------

    fn make_mbtiles(path: &Path, rows: &[(i64, i64, i64, Vec<u8>)], metadata: &[(&str, &str)]) {
        let connection = Connection::open(path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE metadata (name text, value text);
                 CREATE TABLE tiles (zoom_level integer, tile_column integer,
                                     tile_row integer, tile_data blob);",
            )
            .unwrap();
        for (name, value) in metadata {
            connection
                .execute("INSERT INTO metadata VALUES (?, ?)", (name, value))
                .unwrap();
        }
        for (zoom, column, row, data) in rows {
            connection
                .execute(
                    "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                    (zoom, column, row, data),
                )
                .unwrap();
        }
    }

    #[tokio::test]
    async fn an_mbtiles_dataset_flips_y_because_it_defaults_to_tms() {
        let dir = scratch("mbtiles");
        let path = dir.join("tiles.mbtiles");
        // Zoom 1 holds rows 0 and 1; TMS row 1 is XYZ tile_y 0.
        make_mbtiles(
            &path,
            &[
                (1, 0, 0, b"row0".to_vec()),
                (1, 0, 1, b"row1".to_vec()),
                (0, 0, 0, b"zoom0".to_vec()),
            ],
            &[],
        );

        let slide = MvtSlide::open(path).unwrap();
        assert_eq!(slide.scheme, "tms");
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.num_levels, 2);
        // Zoom 1 advertises the full 2x2 grid regardless of which tiles exist.
        assert_eq!(info.extent, SlideExtent::new(512, 512, 1));

        assert_eq!(
            &raw(slide
                .get_tile(0, 0, 0, ReadOptions::default())
                .await
                .unwrap())[..],
            b"row1"
        );
        assert_eq!(
            &raw(slide
                .get_tile(0, 0, 1, ReadOptions::default())
                .await
                .unwrap())[..],
            b"row0"
        );
        assert_eq!(
            slide
                .get_tile(0, 1, 0, ReadOptions::default())
                .await
                .unwrap_err()
                .detail_text(),
            "Tile does not exist."
        );

        slide.close().await.unwrap();
        assert_eq!(
            slide
                .get_tile(0, 0, 0, ReadOptions::default())
                .await
                .unwrap_err()
                .status,
            http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[tokio::test]
    async fn mbtiles_metadata_can_override_the_scheme_and_tile_size() {
        let dir = scratch("mbtiles-meta");
        let path = dir.join("xyz.mbtiles");
        make_mbtiles(
            &path,
            &[(1, 0, 0, gzipped(b"payload"))],
            &[("scheme", "XYZ"), ("tile_size", "512")],
        );

        let slide = MvtSlide::open(path).unwrap();
        assert_eq!(slide.scheme, "xyz");
        assert_eq!(
            slide.get_info().await.unwrap().tile_extent,
            SlideExtent::new(512, 512, 1)
        );
        // No flip, and the gzipped blob is inflated.
        assert_eq!(
            &raw(slide
                .get_tile(0, 0, 0, ReadOptions::default())
                .await
                .unwrap())[..],
            b"payload"
        );
    }

    #[test]
    fn an_mbtiles_file_with_no_tiles_is_rejected() {
        let dir = scratch("mbtiles-empty");
        let path = dir.join("empty.mbtiles");
        make_mbtiles(&path, &[], &[]);
        let err = open_err(path);
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert_eq!(
            err.detail_text(),
            "MBTiles file does not contain any tiles."
        );
    }

    #[test]
    fn a_file_that_is_not_a_database_is_rejected() {
        let dir = scratch("mbtiles-bogus");
        let path = dir.join("bogus.mbtiles");
        std::fs::write(&path, b"definitely not sqlite").unwrap();
        // SQLite reports "file is not a database" either at open or at the first
        // statement, depending on its version; Python surfaces the first as a 400 and the
        // second as a 500, and so does this.
        let err = open_err(path);
        assert!(
            err.status == http::StatusCode::BAD_REQUEST
                || err.status == http::StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected status {}",
            err.status
        );
    }

    // --- unsupported operations ---------------------------------------------------------

    #[tokio::test]
    async fn region_and_thumbnail_are_501_and_associated_images_are_404() {
        let dir = scratch("unsupported");
        let path = dir.join("tile.pbf");
        std::fs::write(&path, b"x").unwrap();
        let slide = MvtSlide::open(path).unwrap();

        assert_eq!(
            slide
                .get_region(0, 0, 0, 1, 1, ReadOptions::default())
                .await
                .unwrap_err()
                .status,
            http::StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            slide
                .get_thumbnail(10, 10, None, false)
                .await
                .unwrap_err()
                .status,
            http::StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            slide.get_label().await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );
        assert_eq!(
            slide.get_macro(None, false).await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );
        // `b""` reaches `get_for_payload`, which raises the unhandled 404 of divergence 11.
        let icc = slide.get_icc_profile().await.unwrap_err();
        assert_eq!(icc.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(icc.unhandled);
    }

    #[test]
    fn python_int_accepts_what_python_accepts() {
        assert_eq!(python_int(&serde_json::json!(7)).unwrap(), 7);
        assert_eq!(python_int(&serde_json::json!(7.9)).unwrap(), 7);
        assert_eq!(python_int(&serde_json::json!("7")).unwrap(), 7);
        assert!(python_int(&serde_json::json!("7.5")).is_err());
        assert!(python_int(&serde_json::json!(null)).is_err());
    }
}
