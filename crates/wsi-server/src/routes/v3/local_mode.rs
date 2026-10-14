//! Standalone-mode routes, ported from `wsi_service/api/v3/local_mode.py`.
//!
//! Registered only when `WS_LOCAL_MODE` names a mapper — `add_routes_v3` gates them the
//! same way (`api/v3/__init__.py:8`).

use axum::extract::{RawQuery, State};
use axum::response::{IntoResponse, Json, Response};
use http::HeaderMap;

use wsi_core::error::Result;

use crate::query::QueryParams;
use crate::routes::common::authenticate;
use crate::state::AppState;

/// `GET /cases/`
///
/// `context` is only meaningful for context-dependent mappers; `SimpleMapper` logs a
/// warning and ignores it.
pub async fn cases(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    authenticate(&state, &headers, "/cases/").await?;
    let params = QueryParams::parse(raw.as_deref());
    let context = params.optional_str("context");

    let mapper = require_mapper(&state)?;
    let cases = mapper.get_cases(context.as_deref())?;
    Ok(Json(cases).into_response())
}

/// `GET /cases/slides/`
pub async fn case_slides(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    authenticate(&state, &headers, "/cases/slides/").await?;
    let params = QueryParams::parse(raw.as_deref());
    let case_id = params.required_str("case_id")?;

    let mapper = require_mapper(&state)?;
    let slides = mapper.get_slides(&case_id)?;
    Ok(Json(slides).into_response())
}

/// `GET /slides`
pub async fn slide(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides").await?;
    let params = QueryParams::parse(raw.as_deref());
    let slide_id = params.required_str("slide_id")?;

    // Python performs the access check before consulting the mapper on this route.
    state
        .auth
        .allow_access_slide(&payload, &slide_id, None, None)
        .await?;

    let mapper = require_mapper(&state)?;
    let slide = mapper.get_slide(&slide_id)?;
    Ok(Json(slide).into_response())
}

/// `GET /slides/storage`
///
/// Python omits the per-slide access check here, even though the sibling `/slides` route
/// two functions up performs it — so under an integration that enforces per-slide rules
/// (`LsaaiAuth`) any authenticated caller can read back the storage addresses of a slide
/// it is not allowed to open. The port performs the check; divergence 35.
pub async fn slide_storage(
    State(state): State<AppState>,
    headers: HeaderMap,
    RawQuery(raw): RawQuery,
) -> Result<Response> {
    let payload = authenticate(&state, &headers, "/slides/storage").await?;
    let params = QueryParams::parse(raw.as_deref());
    let slide_id = params.required_str("slide_id")?;

    // Not in Python. See the note above: it is a no-op for every integration that has no
    // per-slide rules, so it costs nothing where Python was already equivalent.
    state
        .auth
        .allow_access_slide(&payload, &slide_id, None, None)
        .await?;

    let mapper = require_mapper(&state)?;
    let slide = mapper.get_slide(&slide_id)?;
    Ok(Json(slide.slide_storage).into_response())
}

/// `GET /refresh_local_mapper`
pub async fn refresh_local_mapper(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response> {
    authenticate(&state, &headers, "/refresh_local_mapper").await?;

    let mapper = require_mapper(&state)?;
    let mapper = std::sync::Arc::clone(mapper);
    // Rescanning a data directory is blocking filesystem work.
    tokio::task::spawn_blocking(move || mapper.refresh(true))
        .await
        .map_err(|e| wsi_core::error::ApiError::internal(format!("Refresh failed: {e}")))??;

    Ok((
        http::StatusCode::OK,
        Json(serde_json::json!({ "detail": "Local mapper has been refreshed." })),
    )
        .into_response())
}

/// These routes are only registered when a mapper exists, so this is a guard against a
/// wiring mistake rather than a reachable error.
fn require_mapper(state: &AppState) -> Result<&std::sync::Arc<dyn wsi_core::storage::LocalMapper>> {
    state
        .local_mapper
        .as_ref()
        .ok_or_else(|| wsi_core::error::ApiError::not_found("Local mode is not enabled"))
}
