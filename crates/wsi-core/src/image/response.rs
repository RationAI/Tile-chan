//! Response construction and request validation, ported from
//! `wsi_service/utils/app_utils.py`.
//!
//! Encoders are chosen for parity, not convenience:
//!
//! * **JPEG** is produced through libjpeg-turbo (the `jpeg-turbo` feature, on by default),
//!   the same library Pillow links against, so quality settings and subsampling produce
//!   the same bitstream. The `jpeg-pure` fallback exists so the crate still builds without
//!   a C toolchain, at the cost of byte-level parity.
//! * **PNG** and **BMP** are lossless, so a different encoder changes the container bytes
//!   but not the pixels; the parity harness compares them decoded.
//! * **GIF** requires palette quantisation, and Pillow's adaptive median cut is not
//!   reproducible by any Rust crate. Documented as a known divergence.
//! * **TIFF** goes through [`super::tiffout`].

use std::io::Cursor;

use bytes::Bytes;
use image::DynamicImage;

use crate::error::{ApiError, Result};
use crate::models::SlideInfo;
use crate::pixels::{Pixels, PlanarData};
use crate::settings::Settings;

use super::convert::{
    mask_rgb_channels, planar_to_image, requested_channels_as_array,
    requested_channels_as_rgb_array, PilMode,
};
use super::tiffout::encode_planar_tiff;

/// `supported_image_formats` — `app_utils.py:26`.
pub fn raster_media_type(format: &str) -> Option<&'static str> {
    Some(match format {
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "jpeg" => "image/jpeg",
        "png" => "image/png",
        "tiff" => "image/tiff",
        _ => return None,
    })
}

/// `supported_passthrough_formats` — `app_utils.py:34`.
pub fn passthrough_media_type(format: &str) -> Option<&'static str> {
    Some(match format {
        "raw" => "application/octet-stream",
        "json" => "application/json",
        "geojson" => "application/geo+json",
        "mvt" | "pbf" => "application/vnd.mapbox-vector-tile",
        _ => return None,
    })
}

/// `alternative_spellings` + `normalize_image_format` — `app_utils.py:42,50`.
pub fn normalize_image_format(format: &str) -> String {
    match format {
        "jpg" => "jpeg".to_owned(),
        "tif" => "tiff".to_owned(),
        "geo+json" => "geojson".to_owned(),
        "mapbox-vector-tile" => "mvt".to_owned(),
        other => other.to_owned(),
    }
}

pub fn is_passthrough_format(format: &str) -> bool {
    passthrough_media_type(&normalize_image_format(format)).is_some()
}

/// A fully encoded response body, ready for the framework layer to wrap.
#[derive(Debug, Clone)]
pub struct EncodedResponse {
    pub body: Bytes,
    pub media_type: &'static str,
    /// Extra headers Python sets explicitly (`Content-Length`, `Content-Encoding`).
    pub headers: Vec<(&'static str, String)>,
}

impl EncodedResponse {
    fn new(body: Vec<u8>, media_type: &'static str) -> Self {
        Self {
            body: Bytes::from(body),
            media_type,
            headers: Vec::new(),
        }
    }
}

// --- passthrough ------------------------------------------------------------------------

fn is_gzipped(payload: &[u8]) -> bool {
    payload.len() >= 2 && payload[0] == 0x1f && payload[1] == 0x8b
}

/// `coerce_passthrough_payload` — `app_utils.py:64`.
///
/// Bytes and strings pass straight through. Anything else is JSON-serialised, but only for
/// the `json`/`geojson` formats; other formats raise 400. Rust plugins always hand back
/// `Pixels::Raw`, so the JSON branch is only reachable for a plugin that returns a decoded
/// image where a passthrough payload was expected — which Python also rejects.
pub fn coerce_passthrough_payload(pixels: &Pixels, format: &str) -> Result<Bytes> {
    coerce_passthrough_payload_opt(Some(pixels), format)
}

/// As [`coerce_passthrough_payload`], for a payload that may be absent.
///
/// Python passes `None` straight in — `safe_get_slide_icc_profile` returns it for any slide
/// whose profile could not be read — and `None` matches none of the byte or string cases,
/// so it falls through to the same 400. That matters because the call is not inside
/// `batch_safe_make_response`'s `try`: one unreadable profile fails the entire
/// `/files/icc_profile` request rather than becoming a `.err` entry.
pub fn coerce_passthrough_payload_opt(pixels: Option<&Pixels>, format: &str) -> Result<Bytes> {
    match pixels {
        Some(Pixels::Raw(bytes)) => Ok(bytes.clone()),
        _ => Err(ApiError::bad_request(format!(
            "Failed to read passthrough payload for format '{format}'. Expected bytes or string payload."
        ))),
    }
}

/// `make_passthrough_response` — `app_utils.py:91`.
pub fn make_passthrough_response(
    pixels: &Pixels,
    format: &str,
    compress_raw: bool,
) -> Result<EncodedResponse> {
    let format = normalize_image_format(format);
    let Some(media_type) = passthrough_media_type(&format) else {
        return Err(ApiError::bad_request(
            "Provided image format parameter not supported",
        ));
    };

    let payload = coerce_passthrough_payload(pixels, &format)?;
    let already_gzipped = is_gzipped(&payload);

    if format == "raw" && compress_raw && !already_gzipped {
        let compressed = gzip(&payload)?;
        let len = compressed.len();
        return Ok(EncodedResponse {
            body: Bytes::from(compressed),
            media_type,
            headers: vec![
                ("content-encoding", "gzip".to_owned()),
                ("content-length", len.to_string()),
            ],
        });
    }

    let mut headers = vec![("content-length", payload.len().to_string())];
    if already_gzipped {
        headers.insert(0, ("content-encoding", "gzip".to_owned()));
    }
    Ok(EncodedResponse {
        body: payload,
        media_type,
        headers,
    })
}

fn gzip(payload: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(payload)
        .map_err(|e| ApiError::internal(format!("Failed to gzip payload: {e}")))?;
    encoder
        .finish()
        .map_err(|e| ApiError::internal(format!("Failed to gzip payload: {e}")))
}

// --- raster -----------------------------------------------------------------------------

/// `process_image_region` — `app_utils.py:113`.
///
/// Turns whatever the plugin returned into a Pillow-equivalent image, applying channel
/// selection. Note the asymmetry: for planar data with an explicit channel list the
/// contrast window is taken from that selected data's own min/max, so the output depends
/// on the pixel values in the requested region.
pub fn process_image_region(
    pixels: &Pixels,
    image_channels: Option<&[usize]>,
) -> Result<DynamicImage> {
    match pixels {
        Pixels::Image(img) => Ok(match image_channels {
            None => img.clone(),
            Some(chs) => mask_rgb_channels(img, chs),
        }),
        Pixels::Planar(data) => match image_channels {
            None => {
                // Workaround in the original: return the first three channels as RGB.
                let result = requested_channels_as_rgb_array(data, None);
                planar_to_image(&result, None, None, PilMode::Rgb)
            }
            Some(chs) => {
                let result = requested_channels_as_rgb_array(data, Some(chs));
                let mode = if chs.len() == 1 {
                    PilMode::L
                } else {
                    PilMode::Rgb
                };
                let (lo, hi) = result.min_max().ok_or_else(|| {
                    ApiError::bad_request("Image conversion to Pillow failed: empty region")
                })?;
                planar_to_image(&result, Some(lo), Some(hi), mode)
            }
        },
        Pixels::Raw(_) => Err(ApiError::bad_request(
            "Failed to read region in an appropriate internal representation.",
        )),
    }
}

/// `process_image_region_raw` — `app_utils.py:136`.
///
/// For a Pillow image Python does `np.asarray(image).transpose(2, 0, 1)`. A greyscale
/// image yields a 2-D array and the transpose raises `numpy.AxisError`, which escapes as
/// an unhandled 500 — reproduced here rather than silently promoted, because the failure
/// is observable (`?image_format=tiff` on a greyscale label image).
pub fn process_image_region_raw(
    pixels: &Pixels,
    image_channels: Option<&[usize]>,
) -> Result<PlanarData> {
    match pixels {
        Pixels::Image(img) => match img {
            // `np.asarray(L-mode image)` is 2-D, so `transpose(2, 0, 1)` raises
            // numpy.AxisError, which escapes FastAPI as a plain-text 500.
            DynamicImage::ImageLuma8(_) | DynamicImage::ImageLuma16(_) => Err(ApiError::unhandled(
                "AxisError('axis 2 is out of bounds for array of dimension 2')",
            )),
            other => {
                let rgb = other.to_rgb8();
                let (w, h) = rgb.dimensions();
                let mut arr = ndarray::Array3::<u8>::zeros((3, h as usize, w as usize));
                for (x, y, px) in rgb.enumerate_pixels() {
                    for c in 0..3 {
                        arr[[c, y as usize, x as usize]] = px.0[c];
                    }
                }
                Ok(PlanarData::U8(arr))
            }
        },
        Pixels::Planar(data) => Ok(match image_channels {
            None => data.clone(),
            Some(chs) => requested_channels_as_array(data, chs),
        }),
        Pixels::Raw(_) => Err(ApiError::bad_request(
            "Failed to read region in an apropriate internal representation.",
        )),
    }
}

/// `make_response` — `app_utils.py:153`.
pub fn make_response(
    pixels: &Pixels,
    image_format: &str,
    image_quality: Option<u8>,
    image_channels: Option<&[usize]>,
) -> Result<EncodedResponse> {
    let format = normalize_image_format(image_format);

    if is_passthrough_format(&format) {
        return make_passthrough_response(pixels, &format, true);
    }

    // Raw JPEG tiles are handed back untouched when JPEG was requested — the fast path
    // that lets a plugin skip re-encoding entirely.
    let decoded;
    let pixels = match pixels {
        Pixels::Raw(bytes) => {
            if format == "jpeg" {
                return Ok(EncodedResponse::new(bytes.to_vec(), "image/jpeg"));
            }
            let img = image::load_from_memory(bytes).map_err(|e| {
                ApiError::bad_request(format!("Failed to decode plugin payload: {e}"))
            })?;
            decoded = Pixels::Image(img);
            &decoded
        }
        other => other,
    };

    if format == "tiff" {
        let narray = process_image_region_raw(pixels, image_channels)?;
        return make_tif_response(&narray);
    }

    let img = process_image_region(pixels, image_channels)?;
    make_image_response(&img, &format, image_quality)
}

/// `make_image_response` — `app_utils.py:174`.
pub fn make_image_response(
    img: &DynamicImage,
    image_format: &str,
    image_quality: Option<u8>,
) -> Result<EncodedResponse> {
    let format = normalize_image_format(image_format);
    let Some(media_type) = raster_media_type(&format) else {
        return Err(ApiError::bad_request(
            "Provided image format parameter not supported",
        ));
    };
    let body = encode_image(img, &format, image_quality)?;
    Ok(EncodedResponse::new(body, media_type))
}

/// `make_tif_response` — `app_utils.py:184`.
pub fn make_tif_response(data: &PlanarData) -> Result<EncodedResponse> {
    let body = encode_planar_tiff(data)?;
    Ok(EncodedResponse::new(body, "image/tiff"))
}

/// `pil_image.save(mem, format=..., quality=...)`.
fn encode_image(img: &DynamicImage, format: &str, quality: Option<u8>) -> Result<Vec<u8>> {
    match format {
        "jpeg" => encode_jpeg(img, quality.unwrap_or(90)),
        "png" => {
            let mut out = Cursor::new(Vec::new());
            img.write_to(&mut out, image::ImageFormat::Png)
                .map_err(|e| ApiError::bad_request(format!("Error writing png: {e}")))?;
            Ok(out.into_inner())
        }
        "bmp" => {
            let mut out = Cursor::new(Vec::new());
            img.write_to(&mut out, image::ImageFormat::Bmp)
                .map_err(|e| ApiError::bad_request(format!("Error writing bmp: {e}")))?;
            Ok(out.into_inner())
        }
        "gif" => {
            // Pillow quantises RGB to a 256-entry adaptive palette (median cut). No Rust
            // crate reproduces that exactly; see docs/divergences.md.
            let mut out = Cursor::new(Vec::new());
            img.write_to(&mut out, image::ImageFormat::Gif)
                .map_err(|e| ApiError::bad_request(format!("Error writing gif: {e}")))?;
            Ok(out.into_inner())
        }
        other => Err(ApiError::bad_request(format!(
            "Provided image format parameter not supported: {other}"
        ))),
    }
}

/// libjpeg's `jpeg_set_quality` opens with "Safety limit on quality factor. Convert 0 to 1
/// to avoid zero divide", so Pillow's `quality=0` reaches the encoder as 1. turbojpeg
/// validates the parameter instead and rejects 0 outright, which would answer 400 where
/// Python answers with an image — and `image_quality=0` is a documented value, the one
/// that makes a TIFF response lossy.
fn clamp_jpeg_quality(quality: u8) -> u8 {
    quality.clamp(1, 100)
}

#[cfg(feature = "jpeg-turbo")]
fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let quality = clamp_jpeg_quality(quality);
    use turbojpeg::{compress_image, Subsamp};

    // Pillow leaves libjpeg's defaults in place (`subsampling=-1`), which is 4:2:0.
    let subsamp = Subsamp::Sub2x2;
    match img {
        DynamicImage::ImageLuma8(buf) => {
            let out = compress_image(buf, i32::from(quality), Subsamp::Gray)
                .map_err(|e| ApiError::bad_request(format!("Error writing jpeg: {e}")))?;
            Ok(out.to_vec())
        }
        other => {
            let rgb = other.to_rgb8();
            let out = compress_image(&rgb, i32::from(quality), subsamp)
                .map_err(|e| ApiError::bad_request(format!("Error writing jpeg: {e}")))?;
            Ok(out.to_vec())
        }
    }
}

#[cfg(not(feature = "jpeg-turbo"))]
fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    // Pure-Rust fallback. Pixel output is close but not bit-identical to libjpeg-turbo;
    // build with the `jpeg-turbo` feature for parity.
    let mut out = Cursor::new(Vec::new());
    let mut encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, clamp_jpeg_quality(quality));
    encoder
        .encode_image(img)
        .map_err(|e| ApiError::bad_request(format!("Error writing jpeg: {e}")))?;
    Ok(out.into_inner())
}

// --- validation -------------------------------------------------------------------------

/// `validate_image_request` — `app_utils.py:203`.
pub fn validate_image_request(image_format: &str, image_quality: Option<i64>) -> Result<()> {
    let format = normalize_image_format(image_format);
    if raster_media_type(&format).is_none() && passthrough_media_type(&format).is_none() {
        return Err(ApiError::bad_request(
            "Provided image format parameter not supported",
        ));
    }
    if let Some(q) = image_quality {
        if !(0..=100).contains(&q) {
            return Err(ApiError::bad_request(
                "Provided image quality parameter not supported",
            ));
        }
    }
    Ok(())
}

/// `validate_hex_color_string` — `app_utils.py:211`.
///
/// The regex accepts both `#RGB` and `#RRGGBB`, but the parser slices as if the value were
/// always six digits, so a three-digit colour raises `ValueError` and escapes as a 500.
/// That defect is reproduced by default; set `WS_LENIENT_PADDING_COLOR=true` to expand
/// short-form colours instead. Anything the regex rejects falls back to the configured
/// default, silently.
pub fn validate_hex_color_string(
    padding_color: Option<&str>,
    settings: &Settings,
) -> Result<(u8, u8, u8)> {
    let Some(raw) = padding_color.filter(|s| !s.is_empty()) else {
        return Ok(settings.padding_color);
    };

    let Some(digits) = raw.strip_prefix('#') else {
        return Ok(settings.padding_color);
    };
    let hex_ok = matches!(digits.len(), 3 | 6) && digits.chars().all(|c| c.is_ascii_hexdigit());
    if !hex_ok {
        return Ok(settings.padding_color);
    }

    if digits.len() == 3 {
        if !settings.lenient_padding_color {
            // `int("", 16)` raises ValueError, which FastAPI does not handle.
            return Err(ApiError::unhandled(
                "ValueError(\"invalid literal for int() with base 16: ''\")",
            ));
        }
        let expand = |c: char| {
            let v = c.to_digit(16).expect("checked above") as u8;
            v * 16 + v
        };
        let mut chars = digits.chars();
        let r = expand(chars.next().unwrap());
        let g = expand(chars.next().unwrap());
        let b = expand(chars.next().unwrap());
        return Ok((r, g, b));
    }

    let parse = |at: usize| u8::from_str_radix(&digits[at..at + 2], 16).unwrap_or(0);
    Ok((parse(0), parse(2), parse(4)))
}

/// `validate_image_channels` — `app_utils.py:221`.
pub fn validate_image_channels(info: &SlideInfo, image_channels: Option<&[usize]>) -> Result<()> {
    let Some(channels) = image_channels else {
        return Ok(());
    };
    let available = info.channel_count();
    for &i in channels {
        if i >= available {
            // The message is multi-line in Python, indentation included. Reproduced
            // verbatim so clients matching on it keep working.
            return Err(ApiError::bad_request(format!(
                "\n                Selected image channel exceeds channel bounds\n                (selected: {i} max: {})\n                ",
                available as i64 - 1
            )));
        }
    }
    let mut seen = std::collections::HashSet::new();
    if !channels.iter().all(|c| seen.insert(*c)) {
        return Err(ApiError::bad_request("No duplicates allowed in channels"));
    }
    Ok(())
}

/// `validate_image_size` — `app_utils.py:237`.
pub fn validate_image_size(size_x: i64, size_y: i64, settings: &Settings) -> Result<()> {
    if size_x.saturating_mul(size_y) > settings.max_returned_region_size {
        return Err(ApiError::unprocessable(format!(
            "Requested region may not contain more than {} pixels.",
            settings.max_returned_region_size
        )));
    }
    Ok(())
}

/// `validate_image_z` — `app_utils.py:245`.
pub fn validate_image_z(info: &SlideInfo, z: i64) -> Result<()> {
    if z > 0 && info.extent.z == 1 {
        return Err(ApiError::unprocessable(format!(
            "Invalid ZStackQuery z={z}. The image does not support multiple z-layers."
        )));
    }
    if z > 0 && z >= info.extent.z {
        return Err(ApiError::unprocessable(format!(
            "Invalid ZStackQuery z={z}. The image has only {} z-layers.",
            info.extent.z
        )));
    }
    Ok(())
}

/// `validate_image_level` — `app_utils.py:258`.
pub fn validate_image_level(info: &SlideInfo, level: usize) -> Result<()> {
    if level >= info.levels.len() {
        return Err(ApiError::unprocessable(format!(
            "The requested pyramid level is not available. The coarsest available level is {}.",
            info.levels.len() as i64 - 1
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SlideChannel, SlideColor, SlideExtent, SlideLevel, SlidePixelSizeNm};
    use ndarray::arr3;

    fn settings() -> Settings {
        Settings::default()
    }

    fn info_with(channels: usize, z: i64, levels: usize) -> SlideInfo {
        SlideInfo {
            id: "s".into(),
            channels: Some(
                (0..channels)
                    .map(|i| SlideChannel {
                        id: i as i32,
                        name: format!("C{i}"),
                        color: SlideColor::new(0, 0, 0, 0),
                    })
                    .collect(),
            ),
            channel_depth: Some(8),
            extent: SlideExtent::new(10, 10, z),
            num_levels: levels as i64,
            pixel_size_nm: SlidePixelSizeNm::xy(1.0, 1.0),
            tile_extent: SlideExtent::new(4, 4, 1),
            levels: (0..levels)
                .map(|_| SlideLevel {
                    extent: SlideExtent::new(10, 10, z),
                    downsample_factor: 1.0,
                })
                .collect(),
            format: "f".into(),
            raw_download: false,
        }
    }

    #[test]
    fn format_normalisation_matches_the_alternative_spellings_table() {
        assert_eq!(normalize_image_format("jpg"), "jpeg");
        assert_eq!(normalize_image_format("tif"), "tiff");
        assert_eq!(normalize_image_format("geo+json"), "geojson");
        assert_eq!(normalize_image_format("mapbox-vector-tile"), "mvt");
        assert_eq!(normalize_image_format("png"), "png");
    }

    #[test]
    fn passthrough_formats_are_recognised() {
        for f in [
            "raw",
            "json",
            "geojson",
            "mvt",
            "pbf",
            "geo+json",
            "mapbox-vector-tile",
        ] {
            assert!(is_passthrough_format(f), "{f}");
        }
        for f in ["jpeg", "png", "tiff", "bmp", "gif"] {
            assert!(!is_passthrough_format(f), "{f}");
        }
    }

    #[test]
    fn media_types_match_python() {
        assert_eq!(raster_media_type("jpeg"), Some("image/jpeg"));
        assert_eq!(raster_media_type("tiff"), Some("image/tiff"));
        assert_eq!(
            passthrough_media_type("raw"),
            Some("application/octet-stream")
        );
        assert_eq!(
            passthrough_media_type("geojson"),
            Some("application/geo+json")
        );
        assert_eq!(
            passthrough_media_type("pbf"),
            Some("application/vnd.mapbox-vector-tile")
        );
    }

    #[test]
    fn raw_passthrough_is_gzipped_once() {
        let payload = Pixels::Raw(Bytes::from_static(b"hello hello hello"));
        let res = make_passthrough_response(&payload, "raw", true).unwrap();
        assert_eq!(res.media_type, "application/octet-stream");
        assert!(res
            .headers
            .iter()
            .any(|(k, v)| *k == "content-encoding" && v == "gzip"));
        assert!(is_gzipped(&res.body));

        // Already-gzipped payloads are passed through with the header set, not re-gzipped.
        let already = Pixels::Raw(Bytes::from(res.body.to_vec()));
        let res2 = make_passthrough_response(&already, "raw", true).unwrap();
        assert_eq!(res2.body, res.body);
        assert!(res2
            .headers
            .iter()
            .any(|(k, v)| *k == "content-encoding" && v == "gzip"));
    }

    #[test]
    fn non_raw_passthrough_is_not_compressed() {
        let payload = Pixels::Raw(Bytes::from_static(br#"{"a":1}"#));
        let res = make_passthrough_response(&payload, "json", true).unwrap();
        assert_eq!(res.media_type, "application/json");
        assert!(!res.headers.iter().any(|(k, _)| *k == "content-encoding"));
        assert_eq!(res.body.as_ref(), br#"{"a":1}"#);
    }

    #[test]
    fn raw_jpeg_tiles_bypass_re_encoding() {
        // A plugin that already has JPEG bytes must not be decoded and re-encoded.
        let jpeg_bytes = Bytes::from_static(&[0xFF, 0xD8, 0xFF, 0xD9]);
        let res = make_response(&Pixels::Raw(jpeg_bytes.clone()), "jpeg", Some(90), None).unwrap();
        assert_eq!(res.body, jpeg_bytes);
        assert_eq!(res.media_type, "image/jpeg");
    }

    #[test]
    fn hex_colour_parses_six_digit_form() {
        let s = settings();
        assert_eq!(
            validate_hex_color_string(Some("#FF8000"), &s).unwrap(),
            (255, 128, 0)
        );
        assert_eq!(
            validate_hex_color_string(Some("#ff8000"), &s).unwrap(),
            (255, 128, 0)
        );
    }

    #[test]
    fn hex_colour_falls_back_to_the_default_when_the_regex_rejects() {
        let s = settings();
        for bad in ["nonsense", "#12345", "FF8000", "#GGGGGG", ""] {
            assert_eq!(
                validate_hex_color_string(Some(bad), &s).unwrap(),
                s.padding_color,
                "{bad}"
            );
        }
        assert_eq!(
            validate_hex_color_string(None, &s).unwrap(),
            s.padding_color
        );
    }

    #[test]
    fn three_digit_hex_reproduces_the_python_500_by_default() {
        let s = settings();
        let err = validate_hex_color_string(Some("#FFF"), &s).unwrap_err();
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        // A bare ValueError, so the body is Starlette's plain text, not a JSON detail.
        assert!(err.unhandled);
    }

    #[test]
    fn three_digit_hex_expands_when_lenient() {
        let mut s = settings();
        s.lenient_padding_color = true;
        assert_eq!(
            validate_hex_color_string(Some("#F0A"), &s).unwrap(),
            (255, 0, 170)
        );
    }

    #[test]
    fn image_request_validation_covers_format_and_quality() {
        assert!(validate_image_request("jpeg", Some(90)).is_ok());
        assert!(validate_image_request("jpg", Some(0)).is_ok());
        assert!(validate_image_request("raw", None).is_ok());

        let err = validate_image_request("xcf", Some(90)).unwrap_err();
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert_eq!(
            err.detail_text(),
            "Provided image format parameter not supported"
        );

        let err = validate_image_request("jpeg", Some(101)).unwrap_err();
        assert_eq!(
            err.detail_text(),
            "Provided image quality parameter not supported"
        );
        assert!(validate_image_request("jpeg", Some(-1)).is_err());
    }

    #[test]
    fn channel_validation_rejects_out_of_bounds_and_duplicates() {
        let info = info_with(3, 1, 1);
        assert!(validate_image_channels(&info, None).is_ok());
        assert!(validate_image_channels(&info, Some(&[0, 1, 2])).is_ok());

        let err = validate_image_channels(&info, Some(&[3])).unwrap_err();
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert!(err.detail_text().contains("selected: 3 max: 2"));

        let err = validate_image_channels(&info, Some(&[0, 0])).unwrap_err();
        assert_eq!(err.detail_text(), "No duplicates allowed in channels");
    }

    #[test]
    fn size_validation_uses_the_configured_maximum() {
        let s = settings();
        assert!(validate_image_size(5000, 5000, &s).is_ok());
        let err = validate_image_size(5001, 5000, &s).unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            err.detail_text(),
            "Requested region may not contain more than 25000000 pixels."
        );
    }

    #[test]
    fn z_validation_distinguishes_flat_and_short_stacks() {
        let flat = info_with(3, 1, 1);
        assert!(validate_image_z(&flat, 0).is_ok());
        let err = validate_image_z(&flat, 1).unwrap_err();
        assert!(err
            .detail_text()
            .contains("does not support multiple z-layers"));

        let stack = info_with(3, 4, 1);
        assert!(validate_image_z(&stack, 3).is_ok());
        let err = validate_image_z(&stack, 4).unwrap_err();
        assert!(err.detail_text().contains("has only 4 z-layers"));
    }

    #[test]
    fn level_validation_reports_the_coarsest_level() {
        let info = info_with(3, 1, 3);
        assert!(validate_image_level(&info, 2).is_ok());
        let err = validate_image_level(&info, 3).unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            err.detail_text(),
            "The requested pyramid level is not available. The coarsest available level is 2."
        );
    }

    #[test]
    fn planar_region_to_tiff_round_trips_channel_selection() {
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]], [[3]], [[4]]]));
        let res = make_response(&Pixels::Planar(data), "tiff", None, Some(&[1, 2])).unwrap();
        assert_eq!(res.media_type, "image/tiff");
        assert!(res.body.starts_with(b"II"));
    }

    #[test]
    fn greyscale_to_tiff_reproduces_the_numpy_axis_error() {
        let img = DynamicImage::ImageLuma8(image::GrayImage::new(2, 2));
        let err = process_image_region_raw(&Pixels::Image(img), None).unwrap_err();
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.unhandled);
    }
}
