//! The `/v3` sub-application, ported from `wsi_service/api/v3/__init__.py`.
//!
//! FastAPI mounts this as a *separate* app, which is why it has its own
//! `/v3/openapi.json`. `Router::nest` reproduces the path prefixing; the separate OpenAPI
//! document is produced in [`crate::openapi`].

use axum::routing::get;
use axum::Router;

use crate::state::AppState;

pub mod batch;
pub mod local_mode;
pub mod slides;

/// `add_routes_v3` — slide routes always, local-mode routes only when a mapper exists.
pub fn router(state: &AppState) -> Router<AppState> {
    let mut router = Router::new()
        // --- single slide -------------------------------------------------------------
        .route("/slides/info", get(slides::info))
        .route(
            "/slides/thumbnail/max_size/{max_x}/{max_y}",
            get(slides::thumbnail),
        )
        .route("/slides/label/max_size/{max_x}/{max_y}", get(slides::label))
        .route(
            "/slides/macro/max_size/{max_x}/{max_y}",
            get(slides::macro_image),
        )
        .route(
            "/slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}",
            get(slides::region),
        )
        .route(
            "/slides/tile/level/{level}/tile/{tile_x}/{tile_y}",
            get(slides::tile),
        )
        .route("/slides/download", get(slides::download))
        .route("/slides/icc_profile", get(slides::icc_profile))
        // --- multi slide, `paths` parameter -------------------------------------------
        .route("/files/info", get(batch::files_info))
        .route(
            "/files/thumbnail/max_size/{max_x}/{max_y}",
            get(batch::files_thumbnail),
        )
        .route(
            "/files/label/max_size/{max_x}/{max_y}",
            get(batch::files_label),
        )
        .route(
            "/files/macro/max_size/{max_x}/{max_y}",
            get(batch::files_macro),
        )
        .route(
            "/files/tile/level/{level}/tile/{tile_x}/{tile_y}",
            get(batch::files_tile),
        )
        .route("/files/batch/", get(batch::files_batch))
        .route("/files/icc_profile", get(batch::files_icc_profile))
        // --- multi slide, legacy `slides` parameter -----------------------------------
        .route("/batch/info", get(batch::batch_info))
        .route(
            "/batch/thumbnail/max_size/{max_x}/{max_y}",
            get(batch::batch_thumbnail),
        )
        .route(
            "/batch/label/max_size/{max_x}/{max_y}",
            get(batch::batch_label),
        )
        .route(
            "/batch/macro/max_size/{max_x}/{max_y}",
            get(batch::batch_macro),
        )
        .route(
            "/batch/tile/level/{level}/tile/{tile_x}/{tile_y}",
            get(batch::batch_tile),
        )
        .route("/batch/batch/", get(batch::batch_batch))
        .route("/batch/icc_profile", get(batch::batch_icc_profile));

    if state.local_mode_enabled() {
        router = router
            // Starlette redirects `/cases` to `/cases/` automatically; axum does not, so
            // both spellings are registered rather than leaving one to 404.
            .route("/cases/", get(local_mode::cases))
            .route("/cases", get(local_mode::cases))
            .route("/cases/slides/", get(local_mode::case_slides))
            .route("/cases/slides", get(local_mode::case_slides))
            .route("/slides", get(local_mode::slide))
            .route("/slides/storage", get(local_mode::slide_storage))
            .route(
                "/refresh_local_mapper",
                get(local_mode::refresh_local_mapper),
            );
    }

    router
}
