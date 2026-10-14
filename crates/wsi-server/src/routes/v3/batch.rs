//! The multi-slide endpoints, ported from
//! `wsi_service/api/v3/slides_batch_api_helpers.py` and
//! `wsi_service/utils/app_batch_utils.py`.
//!
//! Every route exists twice: under `/files/` with a `paths` parameter, and under `/batch/`
//! with a `slides` parameter, both comma-separated and both delegating here. The `/batch/`
//! forms are the older spelling, kept for compatibility.
//!
//! Failure handling is deliberately uneven, and reproduced as-is:
//!
//! * `info` propagates the first failure — one bad ID fails the whole request;
//! * `thumbnail`, `label` and `macro` fetch slides through a swallowing wrapper but then
//!   call a method on the result unconditionally, so a failed slide still fails the whole
//!   request;
//! * `tile`, `batch` and `icc_profile` swallow per-item failures and write them into the
//!   archive as `t{n}.err`.
//!
//! Batch reads also skip the edge-padding wrappers entirely (`app_batch_utils.py:137`) —
//! deliberately, to cut transfer size for viewer prefetching.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use axum::extract::{Path, RawQuery, State};
use axum::response::{IntoResponse, Json, Response};
use http::HeaderMap;

use wsi_core::error::{ApiError, Result};
use wsi_core::image::{
    make_response, normalize_image_format, passthrough_media_type, process_image_region,
    process_image_region_raw, raster_media_type, validate_hex_color_string,
    validate_image_channels, validate_image_level, validate_image_request, validate_image_z,
};
use wsi_core::models::SlideInfo;
use wsi_core::pixels::Pixels;
use wsi_core::slide::Slide;

use crate::query::QueryParams;
use crate::routes::common::{authenticate, ImageQuery, Validation};
use crate::state::AppState;

/// Which query parameter carries the slide list: `paths` under `/files/`, `slides` under
/// `/batch/`. Python declares them separately so the OpenAPI document can name both.
#[derive(Clone, Copy)]
pub enum IdParam {
    Paths,
    Slides,
}

impl IdParam {
    fn name(self) -> &'static str {
        match self {
            IdParam::Paths => "paths",
            IdParam::Slides => "slides",
        }
    }
}

fn slide_ids(params: &QueryParams, id_param: IdParam) -> Result<Vec<String>> {
    let raw = params.required_str(id_param.name())?;
    // `paths.split(",")` — an empty segment stays, and becomes a slide ID that fails to
    // resolve, exactly as in Python.
    Ok(raw.split(',').map(str::to_owned).collect())
}

/// `info` — `slides_batch_api_helpers.py:22`.
pub async fn info(
    state: AppState,
    headers: HeaderMap,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/info").await?;
    let params = QueryParams::parse(raw.as_deref());
    let ids = slide_ids(&params, id_param)?;
    let plugin = params.optional_nonempty("plugin");

    let mut infos: Vec<SlideInfo> = Vec::with_capacity(ids.len());
    for id in &ids {
        infos.push(
            state
                .slide_manager
                .get_slide_info(id, plugin.as_deref())
                .await?,
        );
    }
    for slide in &infos {
        state
            .auth
            .allow_access_slide(&payload, &slide.id, plugin.as_deref(), Some(slide))
            .await?;
    }

    Ok(Json(infos).into_response())
}

/// `thumbnail` — `slides_batch_api_helpers.py:36`.
pub async fn thumbnail(
    state: AppState,
    headers: HeaderMap,
    path: HashMap<String, String>,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/thumbnail").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let ids = slide_ids(&params, id_param)?;
    let query = validation.image_query(&params);

    let max = state.settings.max_thumbnail_size as i64;
    let max_x = validation.path_int(&path, "max_x", Some(1), Some(max), None);
    let max_y = validation.path_int(&path, "max_y", Some(1), Some(max), None);
    validation.finish()?;

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, query.plugin.as_deref(), None)
            .await?;
    }
    validate_image_request(&query.image_format, Some(query.image_quality))?;

    let slides = open_all(&state, &ids, query.plugin.as_deref()).await;
    let mut items = Vec::with_capacity(slides.len());
    for slide in &slides {
        // Python calls `slide.get_thumbnail(...)` without a None check, so a slide that
        // failed to open takes the whole request down with a 500.
        let slide = slide.as_ref().ok_or_else(missing_slide_attribute_error)?;
        items.push(
            slide
                .get_thumbnail(
                    max_x as u32,
                    max_y as u32,
                    query.icc_profile_intent,
                    query.icc_profile_strict,
                )
                .await
                .ok(),
        );
    }

    batch_zip(
        items,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )
}

/// `label` — `slides_batch_api_helpers.py:58`.
///
/// The resize Python intends but never performs (a discarded lazy `map` calling the
/// removed `Image.ANTIALIAS`) is performed here; see `docs/divergences.md`.
pub async fn label(
    state: AppState,
    headers: HeaderMap,
    path: HashMap<String, String>,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/label").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let ids = slide_ids(&params, id_param)?;
    let query = validation.image_query(&params);
    let max_x = validation.path_int(&path, "max_x", None, None, None);
    let max_y = validation.path_int(&path, "max_y", None, None, None);
    validation.finish()?;

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, query.plugin.as_deref(), None)
            .await?;
    }
    validate_image_request(&query.image_format, Some(query.image_quality))?;

    let slides = open_all(&state, &ids, query.plugin.as_deref()).await;
    let mut items = Vec::with_capacity(slides.len());
    for slide in &slides {
        let slide = slide.as_ref().ok_or_else(missing_slide_attribute_error)?;
        let label = slide.get_label().await.ok();
        items.push(label.map(|p| shrink(p, max_x, max_y)));
    }

    batch_zip(
        items,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )
}

/// `macro` — `slides_batch_api_helpers.py:93`.
pub async fn macro_image(
    state: AppState,
    headers: HeaderMap,
    path: HashMap<String, String>,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/macro").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let ids = slide_ids(&params, id_param)?;
    let query = validation.image_query(&params);
    let max_x = validation.path_int(&path, "max_x", None, None, None);
    let max_y = validation.path_int(&path, "max_y", None, None, None);
    validation.finish()?;

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, query.plugin.as_deref(), None)
            .await?;
    }
    validate_image_request(&query.image_format, Some(query.image_quality))?;

    let slides = open_all(&state, &ids, query.plugin.as_deref()).await;
    let mut items = Vec::with_capacity(slides.len());
    for slide in &slides {
        let slide = slide.as_ref().ok_or_else(missing_slide_attribute_error)?;
        let macro_image = slide
            .get_macro(query.icc_profile_intent, query.icc_profile_strict)
            .await
            .ok();
        items.push(macro_image.map(|p| shrink(p, max_x, max_y)));
    }

    batch_zip(
        items,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )
}

/// `tile` — `slides_batch_api_helpers.py:129`. One level and tile coordinate for every
/// slide in the list.
pub async fn tile(
    state: AppState,
    headers: HeaderMap,
    path: HashMap<String, String>,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/tile").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let ids = slide_ids(&params, id_param)?;
    let query = validation.image_query(&params);

    let level = validation.path_int(&path, "level", Some(0), None, None);
    let tile_x = validation.path_int(&path, "tile_x", None, None, None);
    let tile_y = validation.path_int(&path, "tile_y", None, None, None);
    validation.finish()?;

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, query.plugin.as_deref(), None)
            .await?;
    }
    let padding = validate_hex_color_string(query.padding_color.as_deref(), &state.settings)?;
    validate_image_request(&query.image_format, Some(query.image_quality))?;

    let slides = open_all(&state, &ids, query.plugin.as_deref()).await;
    let coordinates: Vec<(i64, i64, i64)> =
        (0..slides.len()).map(|_| (level, tile_x, tile_y)).collect();
    read_tiles_into_zip(&query, padding, &slides, &coordinates).await
}

/// `batch` — `slides_batch_api_helpers.py:170`. A distinct level and coordinate per slide.
pub async fn batch(
    state: AppState,
    headers: HeaderMap,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/batch").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let ids = slide_ids(&params, id_param)?;
    let query = validation.image_query(&params);
    validation.finish()?;

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, query.plugin.as_deref(), None)
            .await?;
    }
    let padding = validate_hex_color_string(query.padding_color.as_deref(), &state.settings)?;
    validate_image_request(&query.image_format, Some(query.image_quality))?;

    let slides = open_all(&state, &ids, query.plugin.as_deref()).await;

    let xs = params.required_int_list("xs")?;
    let ys = params.required_int_list("ys")?;
    let levels = params.required_int_list("levels")?;

    // Python indexes the coordinate lists by slide position, so a short list raises
    // IndexError inside the gather and surfaces as a 500.
    if xs.len() < slides.len() || ys.len() < slides.len() || levels.len() < slides.len() {
        return Err(ApiError::unhandled("IndexError('list index out of range')"));
    }

    let coordinates: Vec<(i64, i64, i64)> = (0..slides.len())
        .map(|i| (levels[i], xs[i], ys[i]))
        .collect();
    read_tiles_into_zip(&query, padding, &slides, &coordinates).await
}

/// `icc_profile` — `slides_batch_api_helpers.py:215`.
pub async fn icc_profile(
    state: AppState,
    headers: HeaderMap,
    raw: Option<String>,
    id_param: IdParam,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/files/icc_profile").await?;
    let params = QueryParams::parse(raw.as_deref());
    let ids = slide_ids(&params, id_param)?;
    let plugin = params.optional_nonempty("plugin");

    for id in &ids {
        state
            .auth
            .allow_access_slide(&payload, id, plugin.as_deref(), None)
            .await?;
    }

    let slides = open_all(&state, &ids, plugin.as_deref()).await;
    let mut items = Vec::with_capacity(slides.len());
    for slide in &slides {
        items.push(match slide {
            Some(slide) => slide.get_icc_profile().await.ok().map(Pixels::Raw),
            None => None,
        });
    }

    batch_zip(items, "raw", None, None)
}

// --- shared helpers ---------------------------------------------------------------------

/// `safe_get_slide` — swallows open failures and yields `None`.
async fn open_all(
    state: &AppState,
    ids: &[String],
    plugin: Option<&str>,
) -> Vec<Option<Arc<dyn Slide>>> {
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        match state.slide_manager.get_slide(id, plugin).await {
            Ok(slide) => out.push(Some(slide)),
            Err(err) => {
                tracing::error!("{err}");
                out.push(None);
            }
        }
    }
    out
}

/// The failure Python produces when it calls a method on a slide that failed to open.
///
/// `safe_get_slide` swallows the open error and yields `None`, but the very next line
/// calls a method on it, so an AttributeError escapes the gather and FastAPI never sees an
/// HTTP error — the client gets Starlette's plain-text 500.
fn missing_slide_attribute_error() -> ApiError {
    ApiError::unhandled("AttributeError(\"'NoneType' object has no attribute 'get_thumbnail'\")")
}

/// `batch_safe_get_tile` for every slide, then `batch_safe_make_response`.
async fn read_tiles_into_zip(
    query: &ImageQuery,
    padding: (u8, u8, u8),
    slides: &[Option<Arc<dyn Slide>>],
    coordinates: &[(i64, i64, i64)],
) -> Result<Response> {
    let opts = query.read_options(padding);
    let channels = query.channels_usize();

    let mut items: Vec<Option<Pixels>> = Vec::with_capacity(slides.len());
    for (index, slide) in slides.iter().enumerate() {
        let (level, tile_x, tile_y) = coordinates[index];
        items.push(match slide {
            None => None,
            Some(slide) => read_one_tile(
                slide.as_ref(),
                level,
                tile_x,
                tile_y,
                query.z,
                channels.as_deref(),
                opts,
            )
            .await
            .map_err(|err| tracing::error!("{err}"))
            .ok(),
        });
    }

    batch_zip(
        items,
        &query.image_format,
        Some(query.image_quality as u8),
        channels.as_deref(),
    )
}

async fn read_one_tile(
    slide: &dyn Slide,
    level: i64,
    tile_x: i64,
    tile_y: i64,
    z: i64,
    channels: Option<&[usize]>,
    opts: wsi_core::slide::ReadOptions,
) -> Result<Pixels> {
    let info = slide.get_info().await?;
    validate_image_level(&info, level as usize)?;
    validate_image_z(&info, z)?;
    validate_image_channels(&info, channels)?;
    // No edge extension here, by design.
    slide.get_tile(level as usize, tile_x, tile_y, opts).await
}

fn shrink(pixels: Pixels, max_x: i64, max_y: i64) -> Pixels {
    match pixels {
        Pixels::Image(img) => {
            if max_x <= 0 || max_y <= 0 {
                return Pixels::Image(img);
            }
            Pixels::Image(wsi_core::image::pillow_thumbnail(
                &img,
                max_x as u32,
                max_y as u32,
                wsi_core::image::Filter::Lanczos,
            ))
        }
        other => other,
    }
}

/// `batch_safe_make_response` — `app_batch_utils.py:64`.
///
/// A stored (uncompressed) archive with one entry per requested slide, named
/// `t{n}.{format}` starting at 1. Most per-item failures become `t{n}.err` holding the
/// exception's `repr`, so a partially successful batch still returns 200.
///
/// Not all of them, though: two steps of `batch_safe_make_response` sit *outside* its
/// `try`, so an item that fails there takes the whole response down with a 400 and no
/// archive at all. [`encode_batch_item`] draws that line, and this returns `Err` for it.
pub fn batch_zip(
    items: Vec<Option<Pixels>>,
    image_format: &str,
    image_quality: Option<u8>,
    image_channels: Option<&[usize]>,
) -> Result<Response> {
    let format = normalize_image_format(image_format);
    let mut buffer = std::io::Cursor::new(Vec::<u8>::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        for (index, item) in items.into_iter().enumerate() {
            let entry = index + 1;
            match encode_batch_item(item, &format, image_quality, image_channels)? {
                BatchItem::Bytes(bytes) => {
                    let _ = zip.start_file(format!("t{entry}.{format}"), options);
                    let _ = zip.write_all(&bytes);
                }
                BatchItem::Failed(err) => {
                    let _ = zip.start_file(format!("t{entry}.err"), options);
                    let _ = zip.write_all(python_repr(&err).as_bytes());
                }
            }
        }
        let _ = zip.finish();
    }

    Ok(http::Response::builder()
        .status(http::StatusCode::OK)
        .header(http::header::CONTENT_TYPE, "application/zip")
        .body(axum::body::Body::from(buffer.into_inner()))
        .map(IntoResponse::into_response)
        .unwrap_or_else(|_| http::StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

/// One archive entry, and which of Python's two failure shapes it took.
enum BatchItem {
    /// Written as `t{n}.{format}`.
    Bytes(Vec<u8>),
    /// Written as `t{n}.err`, carrying the `repr` of the exception Python caught.
    Failed(ApiError),
}

/// `batch_safe_make_response`'s per-item body — `app_batch_utils.py:64`.
///
/// Returning `Err` means the failure is *fatal*: Python raises it outside the `try` that
/// turns everything else into a `.err` entry, so FastAPI answers it and no archive is
/// produced. There are exactly two such places, and both are reachable:
///
/// * the passthrough branch, where `coerce_passthrough_payload` is called directly — so
///   `/files/icc_profile` over a slide with no readable profile is a 400 for the *whole*
///   request, not an archive with one `.err` in it;
/// * decoding an already-encoded payload for a non-JPEG raster format, where
///   `Image.open(BytesIO(...))` is likewise unguarded.
fn encode_batch_item(
    item: Option<Pixels>,
    format: &str,
    quality: Option<u8>,
    channels: Option<&[usize]>,
) -> Result<BatchItem> {
    // The passthrough branch comes first in Python and is not inside the `try`. A missing
    // item reaches `coerce_passthrough_payload(None, ...)`, which raises its "Expected
    // bytes or string payload" 400 — fatal, and the reason `safe_get_slide_icc_profile`
    // swallowing an error still loses the response.
    if passthrough_media_type(format).is_some() {
        let payload =
            wsi_core::image::response::coerce_passthrough_payload_opt(item.as_ref(), format)?;
        return Ok(BatchItem::Bytes(payload.to_vec()));
    }

    // Everything below is inside the `try`, so a missing item is a `.err` entry: Python
    // reaches `process_image_region(slide, None, ...)`, which raises the "appropriate
    // internal representation" 400.
    let Some(pixels) = item else {
        return Ok(BatchItem::Failed(ApiError::bad_request(
            "Failed to read region in an appropriate internal representation.",
        )));
    };

    if let Pixels::Raw(bytes) = &pixels {
        if format == "jpeg" {
            return Ok(BatchItem::Bytes(bytes.to_vec()));
        }
        // `Image.open(BytesIO(image_region))` for any other raster format, also outside
        // the `try`. The port never produces raw bytes for a non-JPEG format, so there is
        // nothing to decode and nothing that can fail here; if that ever changes this is
        // where the failure has to stay fatal.
    }

    let encoded = (|| -> Result<Vec<u8>> {
        if format == "tiff" {
            if raster_media_type(format).is_none() {
                return Err(ApiError::bad_request(
                    "Provided image format parameter not supported for OME tiff",
                ));
            }
            let narray = process_image_region_raw(&pixels, channels)?;
            return wsi_core::image::encode_planar_tiff(&narray);
        }

        if raster_media_type(format).is_none() {
            return Err(ApiError::bad_request(
                "Provided image format parameter not supported",
            ));
        }
        let img = process_image_region(&pixels, channels)?;
        Ok(make_response(&Pixels::Image(img), format, quality, None)?
            .body
            .to_vec())
    })();

    Ok(match encoded {
        Ok(bytes) => BatchItem::Bytes(bytes),
        Err(err) => BatchItem::Failed(err),
    })
}

/// What `repr(ex)` yields for the exception Python caught, which is the text written into
/// a `t{n}.err` entry.
///
/// An exception that would otherwise escape FastAPI already carries its own `repr`; a
/// FastAPI `HTTPException` gets Starlette's.
fn python_repr(err: &ApiError) -> String {
    if err.unhandled {
        return err.detail_text();
    }
    format!(
        "HTTPException(status_code={}, detail={})",
        err.status.as_u16(),
        python_repr_str(&err.detail_text())
    )
}

/// Python's `repr()` for a `str`: single quotes unless the value contains one.
fn python_repr_str(value: &str) -> String {
    if value.contains('\'') && !value.contains('"') {
        format!("\"{value}\"")
    } else {
        format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

// --- axum wiring ------------------------------------------------------------------------
//
// Each endpoint is registered twice with a different id parameter, so these thin wrappers
// bind the parameter name.

macro_rules! bind {
    ($name:ident, $inner:ident, $param:expr, with_path) => {
        pub async fn $name(
            State(state): State<AppState>,
            headers: HeaderMap,
            Path(path): Path<HashMap<String, String>>,
            RawQuery(raw): RawQuery,
        ) -> Result<Response> {
            $inner(state, headers, path, raw, $param).await
        }
    };
    ($name:ident, $inner:ident, $param:expr) => {
        pub async fn $name(
            State(state): State<AppState>,
            headers: HeaderMap,
            RawQuery(raw): RawQuery,
        ) -> Result<Response> {
            $inner(state, headers, raw, $param).await
        }
    };
}

bind!(files_info, info, IdParam::Paths);
bind!(files_thumbnail, thumbnail, IdParam::Paths, with_path);
bind!(files_label, label, IdParam::Paths, with_path);
bind!(files_macro, macro_image, IdParam::Paths, with_path);
bind!(files_tile, tile, IdParam::Paths, with_path);
bind!(files_batch, batch, IdParam::Paths);
bind!(files_icc_profile, icc_profile, IdParam::Paths);

bind!(batch_info, info, IdParam::Slides);
bind!(batch_thumbnail, thumbnail, IdParam::Slides, with_path);
bind!(batch_label, label, IdParam::Slides, with_path);
bind!(batch_macro, macro_image, IdParam::Slides, with_path);
bind!(batch_tile, tile, IdParam::Slides, with_path);
bind!(batch_batch, batch, IdParam::Slides);
// The legacy `/batch/icc_profile` route reads `paths`, not `slides` — a Python
// inconsistency (`api/v3/slides.py:647`) that clients may already rely on.
bind!(batch_icc_profile, icc_profile, IdParam::Paths);

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    fn read_entries(body: &[u8]) -> Vec<(String, Vec<u8>)> {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(body.to_vec())).unwrap();
        let mut out = Vec::new();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let name = file.name().to_owned();
            let mut bytes = Vec::new();
            std::io::copy(&mut file, &mut bytes).unwrap();
            out.push((name, bytes));
        }
        out
    }

    async fn body_of(response: Response) -> Vec<u8> {
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    #[tokio::test]
    async fn entries_are_numbered_from_one_with_the_format_suffix() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let response = batch_zip(
            vec![Some(Pixels::Image(img.clone())), Some(Pixels::Image(img))],
            "png",
            Some(90),
            None,
        );
        let entries = read_entries(&body_of(response.unwrap()).await);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "t1.png");
        assert_eq!(entries[1].0, "t2.png");
        assert!(entries[0].1.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[tokio::test]
    async fn a_failed_item_becomes_a_dot_err_entry_and_the_rest_still_succeed() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let response = batch_zip(vec![None, Some(Pixels::Image(img))], "png", Some(90), None);
        let entries = read_entries(&body_of(response.unwrap()).await);
        assert_eq!(entries[0].0, "t1.err");
        assert_eq!(entries[1].0, "t2.png");
        let message = String::from_utf8(entries[0].1.clone()).unwrap();
        assert!(message.starts_with("HTTPException(status_code=400, detail="));
    }

    #[tokio::test]
    async fn jpeg_payloads_pass_through_without_re_encoding() {
        let raw = bytes::Bytes::from_static(&[0xFF, 0xD8, 0xFF, 0xD9]);
        let response = batch_zip(vec![Some(Pixels::Raw(raw.clone()))], "jpeg", Some(90), None);
        let entries = read_entries(&body_of(response.unwrap()).await);
        assert_eq!(entries[0].0, "t1.jpeg");
        assert_eq!(entries[0].1, raw.to_vec());
    }

    #[tokio::test]
    async fn icc_profiles_are_written_as_raw_entries() {
        let profile = bytes::Bytes::from_static(b"fake-icc-profile");
        let response = batch_zip(vec![Some(Pixels::Raw(profile.clone()))], "raw", None, None);
        let entries = read_entries(&body_of(response.unwrap()).await);
        assert_eq!(entries[0].0, "t1.raw");
        assert_eq!(entries[0].1, profile.to_vec());
    }

    /// `coerce_passthrough_payload` is called outside `batch_safe_make_response`'s `try`,
    /// so a slide whose ICC profile could not be read does not become a `.err` entry — the
    /// whole request is a 400 and there is no archive. `/files/icc_profile` over a
    /// `tifffile` slide hits this, since that plugin's profile endpoint always raises.
    #[tokio::test]
    async fn a_missing_passthrough_payload_fails_the_whole_request() {
        let error = batch_zip(vec![None], "raw", None, None).unwrap_err();
        assert_eq!(error.status, http::StatusCode::BAD_REQUEST);
        assert_eq!(
            error.detail_text(),
            "Failed to read passthrough payload for format 'raw'. Expected bytes or string payload."
        );

        // One good entry alongside the bad one does not save it.
        let profile = bytes::Bytes::from_static(b"fake-icc-profile");
        assert!(batch_zip(vec![Some(Pixels::Raw(profile)), None], "raw", None, None).is_err());
    }

    #[test]
    fn python_repr_matches_starlette() {
        let err = ApiError::bad_request("boom");
        assert_eq!(
            python_repr(&err),
            "HTTPException(status_code=400, detail='boom')"
        );
    }

    #[test]
    fn an_unhandled_exception_keeps_its_own_repr_in_the_archive() {
        let err = ApiError::unhandled("IndexError('list index out of range')");
        assert_eq!(python_repr(&err), "IndexError('list index out of range')");
    }

    #[test]
    fn python_repr_switches_quotes_like_cpython() {
        assert_eq!(python_repr_str("plain"), "'plain'");
        assert_eq!(python_repr_str("it's"), "\"it's\"");
        assert_eq!(python_repr_str("both ' and \""), "'both \\' and \"'");
    }

    #[test]
    fn slide_ids_split_on_commas_including_empty_segments() {
        let params = QueryParams::parse(Some("paths=a,,b"));
        assert_eq!(
            slide_ids(&params, IdParam::Paths).unwrap(),
            vec!["a".to_owned(), String::new(), "b".to_owned()]
        );
    }

    #[test]
    fn the_two_id_parameter_spellings_are_distinct() {
        let params = QueryParams::parse(Some("slides=a,b"));
        assert!(slide_ids(&params, IdParam::Paths).is_err());
        assert_eq!(slide_ids(&params, IdParam::Slides).unwrap().len(), 2);
    }
}
