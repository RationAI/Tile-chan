//! The single-slide image endpoints, ported from `wsi_service/api/v3/slides.py:47-406`.
//!
//! The order of operations inside each handler matters and is preserved exactly: it
//! decides which error a client sees when more than one applies. Authentication runs
//! first — FastAPI resolves the security dependency before validating request parameters,
//! so an unauthenticated request with a bad `image_format` gets 401, not 400.

use std::collections::HashMap;

use axum::extract::{Path, RawQuery, State};
use axum::response::{IntoResponse, Json, Response};
use http::HeaderMap;

use wsi_core::error::{ApiError, Result};
use wsi_core::image::{
    check_complete_region_overlap, check_complete_tile_overlap, get_extended_region,
    get_extended_tile, is_passthrough_format, make_response, validate_hex_color_string,
    validate_image_channels, validate_image_level, validate_image_request, validate_image_size,
    validate_image_z,
};
use wsi_core::pixels::Pixels;

use crate::download::{expand_folders, remove_folders, stream_zip, zip_entries};
use crate::query::QueryParams;
use crate::reply;
use crate::routes::common::{authenticate, Validation};
use crate::state::AppState;

/// `GET /slides/info`
pub async fn info(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/info").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    validation.finish()?;
    let plugin = params.optional_nonempty("plugin");

    // Python fetches the slide info *before* the access check on this endpoint, so the
    // integration can inspect it — `allow_access_slide(..., slide=slide)`.
    let info = state
        .slide_manager
        .get_slide_info(&slide_id, plugin.as_deref())
        .await?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, plugin.as_deref(), Some(&info))
        .await?;

    Ok(Json(info).into_response())
}

/// `GET /slides/thumbnail/max_size/{max_x}/{max_y}`
pub async fn thumbnail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<HashMap<String, String>>,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/thumbnail").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    let query = validation.image_query(&params);

    let max = state.settings.max_thumbnail_size as i64;
    let max_x = validation.path_int(&path, "max_x", Some(1), Some(max), None);
    let max_y = validation.path_int(&path, "max_y", Some(1), Some(max), None);
    validation.finish()?;

    validate_image_request(&query.image_format, Some(query.image_quality))?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, query.plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, query.plugin.as_deref())
        .await?;
    let thumbnail = slide
        .get_thumbnail(
            max_x as u32,
            max_y as u32,
            query.icc_profile_intent,
            query.icc_profile_strict,
        )
        .await?;

    // No `image_channels` on this endpoint, matching Python.
    Ok(reply::encoded(make_response(
        &thumbnail,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )?))
}

/// `GET /slides/label/max_size/{max_x}/{max_y}`
///
/// `max_x`/`max_y` carry no bounds here — Python declares them as bare `Path(...)`.
pub async fn label(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<HashMap<String, String>>,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/label").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    let query = validation.image_query(&params);
    let max_x = validation.path_int(&path, "max_x", None, None, None);
    let max_y = validation.path_int(&path, "max_y", None, None, None);
    validation.finish()?;

    validate_image_request(&query.image_format, Some(query.image_quality))?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, query.plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, query.plugin.as_deref())
        .await?;
    let label = slide.get_label().await?;
    let label = shrink_to_box(label, max_x, max_y)?;

    Ok(reply::encoded(make_response(
        &label,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )?))
}

/// `GET /slides/macro/max_size/{max_x}/{max_y}`
pub async fn macro_image(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<HashMap<String, String>>,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/macro").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    let query = validation.image_query(&params);
    let max_x = validation.path_int(&path, "max_x", None, None, None);
    let max_y = validation.path_int(&path, "max_y", None, None, None);
    validation.finish()?;

    validate_image_request(&query.image_format, Some(query.image_quality))?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, query.plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, query.plugin.as_deref())
        .await?;
    let macro_image = slide
        .get_macro(query.icc_profile_intent, query.icc_profile_strict)
        .await?;
    let macro_image = shrink_to_box(macro_image, max_x, max_y)?;

    Ok(reply::encoded(make_response(
        &macro_image,
        &query.image_format,
        Some(query.image_quality as u8),
        None,
    )?))
}

/// `GET /slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}`
pub async fn region(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<HashMap<String, String>>,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/region").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    let query = validation.image_query(&params);

    let level = validation.path_int(&path, "level", Some(0), None, None);
    let start_x = validation.path_int(&path, "start_x", None, None, None);
    let start_y = validation.path_int(&path, "start_y", None, None, None);
    let size_x = validation.path_int(&path, "size_x", None, None, Some(0));
    let size_y = validation.path_int(&path, "size_y", None, None, Some(0));
    validation.finish()?;

    let padding = validate_hex_color_string(query.padding_color.as_deref(), &state.settings)?;
    validate_image_request(&query.image_format, Some(query.image_quality))?;
    validate_image_size(size_x, size_y, &state.settings)?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, query.plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, query.plugin.as_deref())
        .await?;
    let info = slide.get_info().await?;
    validate_image_level(&info, level as usize)?;
    validate_image_z(&info, query.z)?;
    let channels = query.channels_usize();
    validate_image_channels(&info, channels.as_deref())?;

    let opts = query.read_options(padding);
    let level = level as usize;

    let region = if is_passthrough_format(&query.image_format)
        || !state.settings.apply_padding
        || check_complete_region_overlap(&info, level, start_x, start_y, size_x, size_y)
    {
        slide
            .get_region(level, start_x, start_y, size_x as u32, size_y as u32, opts)
            .await?
    } else {
        // Only reachable when `apply_padding` is true, so `extend` is always true here —
        // the Python expression `True if settings.apply_padding else False` is a no-op
        // guard against the branch being reordered.
        get_extended_region(
            slide.as_ref(),
            &info,
            level,
            start_x,
            start_y,
            size_x,
            size_y,
            opts,
            state.settings.apply_padding,
        )
        .await?
    };

    Ok(reply::encoded(make_response(
        &region,
        &query.image_format,
        Some(query.image_quality as u8),
        channels.as_deref(),
    )?))
}

/// `GET /slides/tile/level/{level}/tile/{tile_x}/{tile_y}`
pub async fn tile(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<HashMap<String, String>>,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/tile").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    let query = validation.image_query(&params);

    let level = validation.path_int(&path, "level", Some(0), None, None);
    let tile_x = validation.path_int(&path, "tile_x", None, None, None);
    let tile_y = validation.path_int(&path, "tile_y", None, None, None);
    validation.finish()?;

    let padding = validate_hex_color_string(query.padding_color.as_deref(), &state.settings)?;
    validate_image_request(&query.image_format, Some(query.image_quality))?;
    state
        .auth
        .allow_access_slide(&payload, &slide_id, query.plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, query.plugin.as_deref())
        .await?;
    let info = slide.get_info().await?;
    validate_image_level(&info, level as usize)?;
    validate_image_z(&info, query.z)?;
    let channels = query.channels_usize();
    validate_image_channels(&info, channels.as_deref())?;

    let opts = query.read_options(padding);
    let level = level as usize;

    let result = async {
        let tile = if is_passthrough_format(&query.image_format)
            || check_complete_tile_overlap(&info, level, tile_x, tile_y)
        {
            slide.get_tile(level, tile_x, tile_y, opts).await?
        } else {
            get_extended_tile(
                slide.as_ref(),
                &info,
                level,
                tile_x,
                tile_y,
                opts,
                state.settings.get_tile_apply_padding,
            )
            .await?
        };
        make_response(
            &tile,
            &query.image_format,
            Some(query.image_quality as u8),
            channels.as_deref(),
        )
    }
    .await;

    match result {
        Ok(encoded) => Ok(reply::encoded(encoded)),
        Err(err) => {
            // Python logs the full request context before re-raising, which is what makes
            // tile failures diagnosable in production.
            tracing::error!(
                "Tile request failed: slide_id={} plugin={} level={} tile=({},{}) z={} \
                 format={} channels={:?} detail={}",
                slide_id,
                slide.plugin_name(),
                level,
                tile_x,
                tile_y,
                query.z,
                query.image_format,
                query.image_channels,
                err.detail_text(),
            );
            Err(err)
        }
    }
}

/// `GET /slides/download`
pub async fn download(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/download").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    validation.finish()?;
    let plugin = params.optional_nonempty("plugin");

    state
        .auth
        .allow_access_slide(&payload, &slide_id, plugin.as_deref(), None)
        .await?;

    let paths = state.slide_manager.get_slide_file_paths(&slide_id).await?;
    // Storage addresses sometimes name a folder rather than its files — DICOM, for
    // instance. Expand those, then drop the folder entries themselves.
    let paths = remove_folders(expand_folders(&paths));
    stream_zip(zip_entries(&paths), &slide_id)
}

/// `GET /slides/icc_profile`
pub async fn icc_profile(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/icc_profile").await?;
    let params = QueryParams::parse(raw.as_deref());
    let mut validation = Validation::new();
    let slide_id = validation.required_str(&params, "slide_id");
    validation.finish()?;
    let plugin = params.optional_nonempty("plugin");

    state
        .auth
        .allow_access_slide(&payload, &slide_id, plugin.as_deref(), None)
        .await?;

    let slide = state
        .slide_manager
        .get_slide(&slide_id, plugin.as_deref())
        .await?;
    let profile = slide.get_icc_profile().await?;

    Ok(reply::encoded(make_response(
        &Pixels::Raw(profile),
        "raw",
        None,
        None,
    )?))
}

/// `Image.thumbnail((max_x, max_y), Image.Resampling.LANCZOS)`.
///
/// Pillow shrinks in place, preserving the aspect ratio, and never enlarges. A plugin that
/// returns a numpy array here would hit `AttributeError` in Python and surface as a 500;
/// reproduced rather than silently handled, since no shipped plugin does it.
fn shrink_to_box(pixels: Pixels, max_x: i64, max_y: i64) -> Result<Pixels> {
    match pixels {
        Pixels::Image(img) => {
            if max_x <= 0 || max_y <= 0 {
                // Pillow raises ValueError on a non-positive box, and this endpoint
                // declares no bounds on max_x/max_y, so it escapes as a plain-text 500.
                return Err(ApiError::unhandled(
                    "ValueError('height and width must be > 0')",
                ));
            }
            Ok(Pixels::Image(wsi_core::image::pillow_thumbnail(
                &img,
                max_x as u32,
                max_y as u32,
                // `api/v3/slides.py` passes `Image.Resampling.LANCZOS` for label and macro.
                wsi_core::image::Filter::Lanczos,
            )))
        }
        // Python calls `.thumbnail(...)` unconditionally, so a plugin that returns an
        // array raises AttributeError, which FastAPI does not handle.
        Pixels::Planar(_) => Err(ApiError::unhandled(
            "AttributeError(\"'numpy.ndarray' object has no attribute 'thumbnail'\")",
        )),
        raw @ Pixels::Raw(_) => Ok(raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn shrink_preserves_aspect_ratio() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(400, 200));
        let out = shrink_to_box(Pixels::Image(img), 100, 100).unwrap();
        match out {
            Pixels::Image(img) => assert_eq!((img.width(), img.height()), (100, 50)),
            _ => panic!("expected an image"),
        }
    }

    #[test]
    fn shrink_never_enlarges() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(10, 10));
        let out = shrink_to_box(Pixels::Image(img), 500, 500).unwrap();
        match out {
            Pixels::Image(img) => assert_eq!((img.width(), img.height()), (10, 10)),
            _ => panic!("expected an image"),
        }
    }

    #[test]
    fn shrink_rejects_a_non_positive_box() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(10, 10));
        assert!(shrink_to_box(Pixels::Image(img), 0, 10).is_err());
    }

    #[test]
    fn planar_label_reproduces_the_python_attribute_error() {
        let data = wsi_core::pixels::PlanarData::U8(ndarray::Array3::zeros((1, 1, 1)));
        let err = match shrink_to_box(Pixels::Planar(data), 10, 10) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.unhandled);
    }
}
