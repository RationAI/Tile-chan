//! Response comparison.
//!
//! Not every response can be compared byte for byte, and pretending otherwise would drown
//! real differences in encoder noise. The rules, and the reason for each:
//!
//! | media type | comparison | why |
//! |---|---|---|
//! | `application/json` | structural, with a float tolerance | key order is not guaranteed |
//! | `image/png`, `image/bmp` | decoded pixels | lossless, but zlib settings differ |
//! | `image/tiff` | decoded pixels | tifffile writes extra tags |
//! | `image/jpeg` | decoded pixels, max channel delta | same encoder, but check anyway |
//! | `image/gif` | dimensions only | Pillow's adaptive palette is not reproducible |
//! | `application/zip` | entry names, then each entry recursively | archives embed timestamps |
//! | anything else | bytes | |
//!
//! See `docs/divergences.md` for what each relaxation is covering for.

use std::io::Read;

/// How far a decoded pixel may drift before it counts as a difference.
///
/// Zero for lossless formats. JPEG goes through libjpeg-turbo on both sides at the same
/// quality, so it should also be exact; a small allowance keeps the harness from failing
/// on a libjpeg version bump, and the actual maximum is always reported.
const JPEG_MAX_CHANNEL_DELTA: u8 = 1;

/// Relative tolerance for floating-point JSON values, covering `pixel_size_nm`.
const FLOAT_TOLERANCE: f64 = 1e-9;

#[derive(Debug)]
pub enum Verdict {
    Match,
    /// Equal within the documented tolerance, with a note about what was relaxed.
    MatchWithTolerance(String),
    Differ(String),
}

impl Verdict {
    /// Used by the tests; the run loop matches on the variants directly so it can tell a
    /// tolerated match from an exact one.
    #[cfg(test)]
    pub fn is_failure(&self) -> bool {
        matches!(self, Verdict::Differ(_))
    }
}

pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub content_encoding: Option<String>,
    pub body: Vec<u8>,
    /// The body hit the harness's size cap, so only a prefix is present. Set for
    /// `/v3/slides/download`, which streams a whole slide file.
    pub truncated: bool,
}

/// Per-channel allowances for one run.
///
/// Both default to the openslide configuration, where the two services share a decoder and
/// every sample has to agree. They are raised only for a configuration whose *decoder*
/// differs — disabling openslide makes `tiffslide` serve the pyramidal slides, and that is
/// divergence 28, a measured one-per-channel disagreement in the inverse DCT.
#[derive(Clone, Copy)]
pub struct Tolerance {
    /// Allowed per-channel difference where the response format is lossless, so this is
    /// the decoder difference itself and nothing else.
    pub lossless: u8,
    /// Allowed per-channel difference after a JPEG round trip, which amplifies whatever
    /// the decoder fed it: a one-per-channel input becomes up to sixteen in a large
    /// region. Checking the lossless formats is what actually guards the pixels; this
    /// bound only keeps the JPEG responses from drowning out the report.
    pub jpeg: u8,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            lossless: 0,
            jpeg: JPEG_MAX_CHANNEL_DELTA,
        }
    }
}

pub fn compare_with(python: &Response, rust: &Response, tolerance: Tolerance) -> Verdict {
    if python.status != rust.status {
        return Verdict::Differ(format!(
            "status {} != {} (python body: {})",
            python.status,
            rust.status,
            preview(&python.body)
        ));
    }

    let python_media = media_type(&python.content_type);
    let rust_media = media_type(&rust.content_type);
    if python_media != rust_media {
        return Verdict::Differ(format!("content-type {python_media} != {rust_media}"));
    }

    // Error bodies are JSON when FastAPI raised the error, and plain text when a bare
    // Python exception escaped it. The content-type check above already separated those
    // two cases, so it is safe to branch on status here.
    if python.status >= 400 {
        return match python_media.as_str() {
            "application/json" => compare_json(&python.body, &rust.body),
            // `text/plain` is Starlette's unhandled-exception response; compare it exactly.
            _ => {
                if python.body == rust.body {
                    Verdict::Match
                } else {
                    Verdict::Differ(format!(
                        "error body differs: {} vs {}",
                        preview(&python.body),
                        preview(&rust.body)
                    ))
                }
            }
        };
    }

    if python.content_encoding != rust.content_encoding {
        return Verdict::Differ(format!(
            "content-encoding {:?} != {:?}",
            python.content_encoding, rust.content_encoding
        ));
    }

    // A capped read cannot support a full-body verdict, so say what was actually checked
    // rather than letting a prefix comparison masquerade as one.
    if python.truncated || rust.truncated {
        return compare_body_prefix(&python.body, &rust.body);
    }

    match python_media.as_str() {
        "application/json" | "application/geo+json" => compare_json(&python.body, &rust.body),
        "image/png" | "image/bmp" => compare_pixels(
            &python.body,
            &rust.body,
            tolerance.lossless,
            python_media.as_str(),
        ),
        "image/tiff" => compare_tiff(&python.body, &rust.body, tolerance),
        "image/jpeg" => compare_pixels(&python.body, &rust.body, tolerance.jpeg, "image/jpeg"),
        "image/gif" => compare_gif(&python.body, &rust.body),
        "application/zip" => compare_zip(&python.body, &rust.body, tolerance),
        _ => {
            if python.body == rust.body {
                Verdict::Match
            } else {
                Verdict::Differ(format!(
                    "body differs ({} vs {} bytes)",
                    python.body.len(),
                    rust.body.len()
                ))
            }
        }
    }
}

/// Compares as much of two bodies as the harness kept.
///
/// Used only for `/v3/slides/download`. The two archives are *not* byte-identical and are
/// not meant to be: the port's ZIP writer emits zip64 headers with a data descriptor and a
/// zero timestamp where Python's `zipfile` writes a classic header with the file's mtime —
/// see `docs/divergences.md`. What is compared is the entry's name and compression method
/// and the payload that follows it, which is where a wrong file or a wrong offset would
/// show up.
fn compare_body_prefix(python: &[u8], rust: &[u8]) -> Verdict {
    let (Some(a), Some(b)) = (first_zip_entry(python), first_zip_entry(rust)) else {
        return Verdict::Differ("could not read the archive's first entry".into());
    };
    if a.name != b.name {
        return Verdict::Differ(format!("first zip entry {} != {}", a.name, b.name));
    }
    if a.method != b.method {
        return Verdict::Differ(format!(
            "first zip entry {} compression {} != {}",
            a.name, a.method, b.method
        ));
    }
    let shared = a.payload.len().min(b.payload.len());
    if a.payload[..shared] != b.payload[..shared] {
        let at = (0..shared)
            .find(|&i| a.payload[i] != b.payload[i])
            .unwrap_or(0);
        return Verdict::Differ(format!(
            "first zip entry {} payload differs at byte {at}",
            a.name
        ));
    }
    Verdict::MatchWithTolerance(format!(
        "capped read: entry {} and its first {shared} payload bytes match; \
         the ZIP container itself differs by design",
        a.name
    ))
}

struct ZipEntryHead<'a> {
    name: String,
    method: u16,
    payload: &'a [u8],
}

/// Parses just the first local file header of a ZIP stream.
fn first_zip_entry(body: &[u8]) -> Option<ZipEntryHead<'_>> {
    // signature(4) version(2) flags(2) method(2) time(2) date(2) crc(4) sizes(8)
    // name_len(2) extra_len(2) = 30 bytes, then the name and the extra field.
    if body.len() < 30 || &body[..4] != b"PK\x03\x04" {
        return None;
    }
    let read_u16 = |at: usize| u16::from_le_bytes([body[at], body[at + 1]]);
    let method = read_u16(8);
    let name_len = read_u16(26) as usize;
    let extra_len = read_u16(28) as usize;
    let name_at = 30;
    let payload_at = name_at + name_len + extra_len;
    if body.len() < payload_at {
        return None;
    }
    Some(ZipEntryHead {
        name: String::from_utf8_lossy(&body[name_at..name_at + name_len]).into_owned(),
        method,
        payload: &body[payload_at..],
    })
}

// --- TIFF ---------------------------------------------------------------------------------

/// Compares two TIFF responses by their structure and their decompressed samples.
///
/// The `image` crate cannot read what either service emits here: `app_utils.py` writes the
/// raw region with `tifffile.imwrite(..., photometric="minisblack",
/// planarconfig="separate")`, which is three *separate* planes under a greyscale
/// photometric with two extra samples. A general-purpose decoder rejects that, so a pixel
/// comparison through one reports "failed to decode" on both sides and can conclude
/// nothing — which is what the first parity run did, fourteen times.
///
/// Parsing the IFD directly settles the question the decoder could not: the geometry tags
/// have to agree exactly, and so do the sample bytes after each strip is decompressed. The
/// container around them is allowed to differ, and does — see `docs/divergences.md`.
fn compare_tiff(python: &[u8], rust: &[u8], tolerance: Tolerance) -> Verdict {
    let (a, b) = match (read_tiff(python), read_tiff(rust)) {
        (Ok(a), Ok(b)) => (a, b),
        // Anything this reader cannot parse is not something either service writes, so fall
        // back rather than reporting a difference the format is not responsible for.
        _ => return compare_pixels(python, rust, 0, "image/tiff"),
    };

    if a.geometry != b.geometry {
        return Verdict::Differ(format!(
            "tiff geometry {:?} != {:?}",
            a.geometry, b.geometry
        ));
    }
    if a.samples != b.samples {
        let differing = a
            .samples
            .iter()
            .zip(b.samples.iter())
            .filter(|(x, y)| x != y)
            .count();
        let max_delta = a
            .samples
            .iter()
            .zip(b.samples.iter())
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        let note = format!(
            "image/tiff: {differing} of {} sample bytes differ, max delta {max_delta}",
            a.samples.len()
        );
        if max_delta > tolerance.lossless {
            return Verdict::Differ(format!("{note} (allowed {})", tolerance.lossless));
        }
        return Verdict::MatchWithTolerance(note);
    }

    if python == rust {
        return Verdict::Match;
    }
    Verdict::MatchWithTolerance(
        "image/tiff: container bytes differ, decompressed samples identical".into(),
    )
}

/// The parts of a TIFF that have to match: the shape of the data and the data itself.
#[derive(Debug, PartialEq, Eq)]
struct TiffGeometry {
    width: u64,
    height: u64,
    bits_per_sample: Vec<u64>,
    samples_per_pixel: u64,
    photometric: u64,
    planar_config: u64,
    rows_per_strip: u64,
    /// Decompressed length of each strip, which pins the strip layout without requiring
    /// the two writers to have chosen the same offsets.
    strip_lengths: Vec<usize>,
}

struct Tiff {
    geometry: TiffGeometry,
    samples: Vec<u8>,
}

/// A single-IFD, strip-based TIFF reader — only as much as both services emit.
fn read_tiff(body: &[u8]) -> Result<Tiff, String> {
    if body.len() < 8 {
        return Err("too short".into());
    }
    let little = match &body[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Err("not a TIFF".into()),
    };
    let u16_at = |at: usize| -> Result<u16, String> {
        let bytes: [u8; 2] = body
            .get(at..at + 2)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let u32_at = |at: usize| -> Result<u32, String> {
        let bytes: [u8; 4] = body
            .get(at..at + 4)
            .ok_or("truncated")?
            .try_into()
            .map_err(|_| "truncated")?;
        Ok(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };

    if u16_at(2)? != 42 {
        return Err("not a classic TIFF".into());
    }
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;

    let mut tags: std::collections::HashMap<u16, Vec<u64>> = std::collections::HashMap::new();
    for index in 0..count {
        let entry = ifd + 2 + index * 12;
        let tag = u16_at(entry)?;
        let kind = u16_at(entry + 2)?;
        let n = u32_at(entry + 4)? as usize;
        let width = match kind {
            1 | 2 | 6 | 7 => 1usize,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => continue,
        };
        let total = width * n;
        let at = if total > 4 {
            u32_at(entry + 8)? as usize
        } else {
            entry + 8
        };
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let value = match width {
                1 => u64::from(*body.get(at + i).ok_or("truncated")?),
                2 => u64::from(u16_at(at + i * 2)?),
                4 => u64::from(u32_at(at + i * 4)?),
                // A rational is two LONGs; neither its numerator nor denominator is part
                // of what has to match, so only the numerator is kept.
                _ => u64::from(u32_at(at + i * 8)?),
            };
            values.push(value);
        }
        tags.insert(tag, values);
    }

    let first = |tag: u16, default: u64| {
        tags.get(&tag)
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(default)
    };
    let offsets = tags.get(&273).cloned().unwrap_or_default();
    let counts = tags.get(&279).cloned().unwrap_or_default();
    if offsets.is_empty() || offsets.len() != counts.len() {
        return Err("not strip-based".into());
    }
    let compression = first(259, 1);

    let mut samples = Vec::new();
    let mut strip_lengths = Vec::with_capacity(offsets.len());
    for (offset, length) in offsets.iter().zip(counts.iter()) {
        let start = *offset as usize;
        let end = start + *length as usize;
        let raw = body.get(start..end).ok_or("strip out of range")?;
        let before = samples.len();
        match compression {
            // Uncompressed.
            1 => samples.extend_from_slice(raw),
            // `DEFLATE` (32946) and `ADOBE_DEFLATE` (8) are both zlib streams; the two
            // writers pick different codes for the same encoding.
            8 | 32946 => {
                use std::io::Read;
                let mut out = Vec::new();
                flate2::read::ZlibDecoder::new(raw)
                    .read_to_end(&mut out)
                    .map_err(|e| format!("inflate failed: {e}"))?;
                samples.extend_from_slice(&out);
            }
            other => return Err(format!("unsupported compression {other}")),
        }
        strip_lengths.push(samples.len() - before);
    }

    Ok(Tiff {
        geometry: TiffGeometry {
            width: first(256, 0),
            height: first(257, 0),
            bits_per_sample: tags.get(&258).cloned().unwrap_or_else(|| vec![1]),
            samples_per_pixel: first(277, 1),
            photometric: first(262, 0),
            planar_config: first(284, 1),
            rows_per_strip: first(278, u64::from(u32::MAX)),
            strip_lengths,
        },
        samples,
    })
}

fn media_type(header: &str) -> String {
    header
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

fn preview(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(&body[..body.len().min(200)]);
    text.replace('\n', " ")
}

// --- JSON ---------------------------------------------------------------------------------

fn compare_json(python: &[u8], rust: &[u8]) -> Verdict {
    let (Ok(a), Ok(b)) = (
        serde_json::from_slice::<serde_json::Value>(python),
        serde_json::from_slice::<serde_json::Value>(rust),
    ) else {
        return if python == rust {
            Verdict::Match
        } else {
            Verdict::Differ(format!(
                "unparseable JSON differs: {} vs {}",
                preview(python),
                preview(rust)
            ))
        };
    };

    let mut differences = Vec::new();
    diff_json("", &a, &b, &mut differences);
    if differences.is_empty() {
        Verdict::Match
    } else {
        Verdict::Differ(differences.join("; "))
    }
}

fn diff_json(path: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                let child = format!("{path}.{key}");
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => diff_json(&child, x, y, out),
                    (Some(_), None) => out.push(format!("{child} missing in rust")),
                    (None, Some(_)) => out.push(format!("{child} unexpected in rust")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                out.push(format!("{path} array length {} != {}", a.len(), b.len()));
                return;
            }
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                diff_json(&format!("{path}[{index}]"), x, y, out);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (Some(x), Some(y)) = (x.as_f64(), y.as_f64()) else {
                if x != y {
                    out.push(format!("{path} {x} != {y}"));
                }
                return;
            };
            let scale = x.abs().max(y.abs()).max(1.0);
            if (x - y).abs() > FLOAT_TOLERANCE * scale {
                out.push(format!("{path} {x} != {y}"));
            }
        }
        _ => {
            if a != b {
                out.push(format!("{path} {a} != {b}"));
            }
        }
    }
}

// --- images -------------------------------------------------------------------------------

fn compare_pixels(python: &[u8], rust: &[u8], max_delta: u8, label: &str) -> Verdict {
    if python == rust {
        return Verdict::Match;
    }

    let (Ok(a), Ok(b)) = (
        image::load_from_memory(python),
        image::load_from_memory(rust),
    ) else {
        return Verdict::Differ(format!(
            "{label}: bytes differ and at least one side failed to decode"
        ));
    };

    if a.width() != b.width() || a.height() != b.height() {
        return Verdict::Differ(format!(
            "{label}: dimensions {}x{} != {}x{}",
            a.width(),
            a.height(),
            b.width(),
            b.height()
        ));
    }

    let a = a.to_rgba8();
    let b = b.to_rgba8();
    let mut worst = 0u8;
    let mut differing = 0usize;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let mut pixel_worst = 0u8;
        for channel in 0..4 {
            let delta = pa.0[channel].abs_diff(pb.0[channel]);
            pixel_worst = pixel_worst.max(delta);
        }
        if pixel_worst > 0 {
            differing += 1;
            worst = worst.max(pixel_worst);
        }
    }

    if worst == 0 {
        return Verdict::MatchWithTolerance(format!(
            "{label}: container bytes differ, pixels identical"
        ));
    }
    if worst <= max_delta {
        return Verdict::MatchWithTolerance(format!(
            "{label}: {differing} pixels differ, max channel delta {worst} (allowed {max_delta})"
        ));
    }
    Verdict::Differ(format!(
        "{label}: {differing} pixels differ, max channel delta {worst} (allowed {max_delta})"
    ))
}

/// Pillow's adaptive-palette quantisation cannot be reproduced, so only the frame geometry
/// is checked. Documented as divergence 9.
fn compare_gif(python: &[u8], rust: &[u8]) -> Verdict {
    let (Ok(a), Ok(b)) = (
        image::load_from_memory(python),
        image::load_from_memory(rust),
    ) else {
        return Verdict::Differ("gif: at least one side failed to decode".into());
    };
    if a.width() != b.width() || a.height() != b.height() {
        return Verdict::Differ(format!(
            "gif: dimensions {}x{} != {}x{}",
            a.width(),
            a.height(),
            b.width(),
            b.height()
        ));
    }
    Verdict::MatchWithTolerance(
        "gif: dimensions only — Pillow's adaptive palette is not reproducible".into(),
    )
}

// --- archives -----------------------------------------------------------------------------

fn read_archive(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes.to_vec()))
        .map_err(|e| format!("not a readable zip: {e}"))?;
    let mut entries = Vec::new();
    for index in 0..archive.len() {
        let mut file = archive
            .by_index(index)
            .map_err(|e| format!("unreadable zip entry: {e}"))?;
        let name = file.name().to_owned();
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("unreadable zip entry {name}: {e}"))?;
        entries.push((name, data));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn compare_zip(python: &[u8], rust: &[u8], tolerance: Tolerance) -> Verdict {
    let (a, b) = match (read_archive(python), read_archive(rust)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return Verdict::Differ(e),
    };

    let a_names: Vec<&String> = a.iter().map(|(n, _)| n).collect();
    let b_names: Vec<&String> = b.iter().map(|(n, _)| n).collect();
    if a_names != b_names {
        return Verdict::Differ(format!("zip entries {a_names:?} != {b_names:?}"));
    }

    let mut notes = Vec::new();
    for ((name, a_data), (_, b_data)) in a.iter().zip(b.iter()) {
        // Entry names carry the format, which is how the content is interpreted.
        let media = match name.rsplit('.').next().unwrap_or("") {
            "png" => "image/png",
            "jpeg" | "jpg" => "image/jpeg",
            "tiff" | "tif" => "image/tiff",
            "bmp" => "image/bmp",
            "gif" => "image/gif",
            "json" | "geojson" => "application/json",
            // `.err` entries hold an exception repr; compare them exactly, since the text
            // is the whole point.
            _ => "application/octet-stream",
        };
        let verdict = compare_with(
            &Response {
                status: 200,
                content_type: media.into(),
                content_encoding: None,
                body: a_data.clone(),
                truncated: false,
            },
            &Response {
                status: 200,
                content_type: media.into(),
                content_encoding: None,
                body: b_data.clone(),
                truncated: false,
            },
            tolerance,
        );
        match verdict {
            Verdict::Match => {}
            Verdict::MatchWithTolerance(note) => notes.push(format!("{name}: {note}")),
            Verdict::Differ(reason) => {
                return Verdict::Differ(format!("zip entry {name}: {reason}"))
            }
        }
    }

    if notes.is_empty() {
        Verdict::Match
    } else {
        Verdict::MatchWithTolerance(notes.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allowances a run uses unless told otherwise.
    fn compare_default(python: &Response, rust: &Response) -> Verdict {
        compare_with(python, rust, Tolerance::default())
    }

    fn json(status: u16, body: &str) -> Response {
        Response {
            status,
            content_type: "application/json".into(),
            content_encoding: None,
            body: body.as_bytes().to_vec(),
            truncated: false,
        }
    }

    fn png(image: &image::RgbImage) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image.clone())
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn image_response(media: &str, body: Vec<u8>) -> Response {
        Response {
            status: 200,
            content_type: media.into(),
            content_encoding: None,
            body,
            truncated: false,
        }
    }

    #[test]
    fn identical_json_matches_regardless_of_key_order() {
        let a = json(200, r#"{"a":1,"b":{"c":2}}"#);
        let b = json(200, r#"{"b":{"c":2},"a":1}"#);
        assert!(matches!(compare_default(&a, &b), Verdict::Match));
    }

    #[test]
    fn json_differences_are_reported_with_a_path() {
        let a = json(200, r#"{"extent":{"x":10}}"#);
        let b = json(200, r#"{"extent":{"x":11}}"#);
        match compare_default(&a, &b) {
            Verdict::Differ(reason) => assert!(reason.contains(".extent.x")),
            other => panic!("expected a difference, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_key_is_a_difference() {
        let a = json(200, r#"{"a":1,"b":2}"#);
        let b = json(200, r#"{"a":1}"#);
        match compare_default(&a, &b) {
            Verdict::Differ(reason) => assert!(reason.contains("missing in rust")),
            other => panic!("expected a difference, got {other:?}"),
        }
    }

    #[test]
    fn floats_compare_within_a_relative_tolerance() {
        let a = json(200, r#"{"pixel_size_nm":{"x":499.0000000001}}"#);
        let b = json(200, r#"{"pixel_size_nm":{"x":499.0}}"#);
        assert!(matches!(compare_default(&a, &b), Verdict::Match));

        let a = json(200, r#"{"pixel_size_nm":{"x":499.1}}"#);
        assert!(compare_default(&a, &b).is_failure());
    }

    #[test]
    fn a_status_mismatch_is_reported_first() {
        let a = json(404, r#"{"detail":"nope"}"#);
        let b = json(200, r#"{"detail":"nope"}"#);
        match compare_default(&a, &b) {
            Verdict::Differ(reason) => assert!(reason.starts_with("status 404 != 200")),
            other => panic!("expected a difference, got {other:?}"),
        }
    }

    #[test]
    fn a_json_error_body_never_matches_a_plain_text_one() {
        // FastAPI's HTTPException vs a bare Python exception escaping to Starlette: the
        // status is the same, but they are not the same response.
        let json_error = Response {
            status: 500,
            content_type: "application/json".into(),
            content_encoding: None,
            body: br#"{"detail":"ICC Profile not available."}"#.to_vec(),
            truncated: false,
        };
        let text_error = Response {
            status: 500,
            content_type: "text/plain; charset=utf-8".into(),
            content_encoding: None,
            body: b"Internal Server Error".to_vec(),
            truncated: false,
        };
        assert!(compare_default(&json_error, &text_error).is_failure());

        // Two identical plain-text 500s match.
        let other = Response {
            status: 500,
            content_type: "text/plain; charset=utf-8".into(),
            content_encoding: None,
            body: b"Internal Server Error".to_vec(),
            truncated: false,
        };
        assert!(matches!(
            compare_default(&text_error, &other),
            Verdict::Match
        ));
    }

    #[test]
    fn error_bodies_are_compared_as_json_whatever_the_format() {
        let a = Response {
            status: 400,
            content_type: "application/json".into(),
            content_encoding: None,
            body: br#"{"detail":"Provided image format parameter not supported"}"#.to_vec(),
            truncated: false,
        };
        let b = Response {
            status: 400,
            content_type: "application/json".into(),
            content_encoding: None,
            body: br#"{"detail":"Provided image format parameter not supported"}"#.to_vec(),
            truncated: false,
        };
        assert!(matches!(compare_default(&a, &b), Verdict::Match));
    }

    /// Same pixels, deliberately different PNG encoder settings — the situation the
    /// harness exists to tolerate, since Pillow and the `image` crate pick different zlib
    /// levels and filters.
    fn png_with_compression(
        image: &image::RgbImage,
        compression: image::codecs::png::CompressionType,
    ) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            &mut out,
            compression,
            image::codecs::png::FilterType::NoFilter,
        );
        image::ImageEncoder::write_image(
            encoder,
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
        out.into_inner()
    }

    #[test]
    fn identical_pixels_match_even_when_the_container_differs() {
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([1, 2, 3]));
        let fast = png_with_compression(&img, image::codecs::png::CompressionType::Fast);
        let best = png_with_compression(&img, image::codecs::png::CompressionType::Best);
        assert_ne!(fast, best, "the two encodings should differ");

        let verdict = compare_default(
            &image_response("image/png", fast),
            &image_response("image/png", best),
        );
        match verdict {
            Verdict::MatchWithTolerance(note) => assert!(note.contains("pixels identical")),
            other => panic!("expected a tolerated match, got {other:?}"),
        }
    }

    #[test]
    fn differing_pixels_fail_for_lossless_formats() {
        let a = image_response(
            "image/png",
            png(&image::RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3]))),
        );
        let b = image_response(
            "image/png",
            png(&image::RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 9]))),
        );
        match compare_default(&a, &b) {
            Verdict::Differ(reason) => assert!(reason.contains("max channel delta 6")),
            other => panic!("expected a difference, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_dimensions_fail() {
        let a = image_response(
            "image/png",
            png(&image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]))),
        );
        let b = image_response(
            "image/png",
            png(&image::RgbImage::from_pixel(8, 4, image::Rgb([0, 0, 0]))),
        );
        match compare_default(&a, &b) {
            Verdict::Differ(reason) => assert!(reason.contains("dimensions 4x4 != 8x4")),
            other => panic!("expected a difference, got {other:?}"),
        }
    }

    #[test]
    fn content_encoding_must_agree() {
        let a = Response {
            status: 200,
            content_type: "application/octet-stream".into(),
            content_encoding: Some("gzip".into()),
            body: vec![1, 2, 3],
            truncated: false,
        };
        let b = Response {
            status: 200,
            content_type: "application/octet-stream".into(),
            content_encoding: None,
            body: vec![1, 2, 3],
            truncated: false,
        };
        assert!(compare_default(&a, &b).is_failure());
    }

    #[test]
    fn zip_entry_names_must_agree() {
        let build = |names: &[&str]| -> Vec<u8> {
            let mut buffer = std::io::Cursor::new(Vec::new());
            {
                let mut zip = zip::ZipWriter::new(&mut buffer);
                let options = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                for name in names {
                    zip.start_file(*name, options).unwrap();
                    std::io::Write::write_all(&mut zip, b"payload").unwrap();
                }
                zip.finish().unwrap();
            }
            buffer.into_inner()
        };

        let a = image_response("application/zip", build(&["t1.png", "t2.png"]));
        let b = image_response("application/zip", build(&["t1.png", "t2.png"]));
        assert!(!compare_default(&a, &b).is_failure());

        let c = image_response("application/zip", build(&["t1.png", "t2.err"]));
        assert!(compare_default(&a, &c).is_failure());
    }

    #[test]
    fn a_content_type_mismatch_is_a_difference() {
        let a = image_response("image/png", vec![]);
        let b = image_response("image/jpeg", vec![]);
        assert!(compare_default(&a, &b).is_failure());
    }

    #[test]
    fn charset_parameters_are_ignored_on_the_content_type() {
        let a = json(200, "{}");
        let mut b = json(200, "{}");
        b.content_type = "application/json; charset=utf-8".into();
        assert!(matches!(compare_default(&a, &b), Verdict::Match));
    }
}
