//! The HTTP service: routers, state, and the application factory.
//!
//! Ported from `wsi_service/app.py`, which builds a root FastAPI app, mounts a second app
//! at `/v3`, and applies CORS to both.

use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use wsi_auth::AuthIntegration;
use wsi_core::error::Result;
use wsi_core::plugins::{PluginRegistry, WsiPlugin};
use wsi_core::settings::Settings;
use wsi_core::slide_manager::SlideManager;
use wsi_core::storage::StorageResolver;

pub mod download;
pub mod openapi;
pub mod query;
pub mod reply;
pub mod routes;
pub mod state;
pub mod zipstream;

pub use state::AppState;

/// Builds the plugin registry.
///
/// Python discovers plugins by scanning for `wsi_service_plugin_*` modules. Rust cannot, so
/// the set is fixed at compile time and selected with cargo features. The registry sorts by
/// name, so the order of these pushes does not matter.
// Each push is behind a cargo feature, so the vector cannot be built as a literal, and
// with every plugin feature disabled the binding is never mutated.
#[allow(unused_mut, clippy::vec_init_then_push)]
pub fn build_registry(settings: &Arc<Settings>) -> Arc<PluginRegistry> {
    let mut plugins: Vec<Arc<dyn WsiPlugin>> = Vec::new();

    #[cfg(feature = "openslide")]
    plugins.push(Arc::new(wsi_plugin_openslide::OpenSlidePlugin::new(
        Arc::clone(settings),
    )));

    #[cfg(feature = "pil")]
    plugins.push(Arc::new(wsi_plugin_pil::PilPlugin::new(Arc::clone(
        settings,
    ))));

    #[cfg(feature = "nifti")]
    plugins.push(Arc::new(wsi_plugin_nifti::NiftiPlugin::new(Arc::clone(
        settings,
    ))));

    #[cfg(feature = "mvt")]
    plugins.push(Arc::new(wsi_plugin_mvt::MvtPlugin::new()));

    #[cfg(feature = "tifffile")]
    plugins.push(Arc::new(wsi_plugin_tifffile::TiffFilePlugin::new(
        Arc::clone(settings),
    )));

    #[cfg(feature = "tiffslide")]
    plugins.push(Arc::new(wsi_plugin_tiffslide::TiffSlidePlugin::new(
        Arc::clone(settings),
    )));

    #[cfg(feature = "wsidicom")]
    plugins.push(Arc::new(wsi_plugin_wsidicom::WsiDicomPlugin::new(
        Arc::clone(settings),
    )));

    Arc::new(PluginRegistry::new(plugins))
}

/// Assembles application state from settings, mirroring the module-level construction in
/// `app.py` and `api/v3/singletons.py`.
pub fn build_state(settings: Settings) -> Result<AppState> {
    let settings = Arc::new(settings);
    let registry = build_registry(&settings);

    let local_mapper = if settings.local_mode.is_empty() {
        None
    } else {
        Some(wsi_mappers::from_descriptor(
            &settings.local_mode,
            &settings.data_dir,
            Arc::clone(&registry),
        )?)
    };

    // A local mapper short-circuits the storage-mapper service entirely
    // (`SlideManager.with_local_mapper`).
    let resolver = match &local_mapper {
        Some(mapper) => StorageResolver::local(Arc::clone(mapper)),
        None => StorageResolver::remote(
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(settings.request_timeout))
                .pool_max_idle_per_host(settings.connection_limit_per_host)
                .build()
                .map_err(|e| {
                    wsi_core::error::ApiError::internal(format!("Could not build HTTP client: {e}"))
                })?,
            settings.mapper_address.clone(),
        ),
    };

    let slide_manager = Arc::new(SlideManager::new(
        resolver,
        Arc::clone(&registry),
        settings.data_dir.clone(),
        settings.inactive_histo_image_timeout_seconds,
        settings.image_handle_cache_size,
    ));

    let auth: Arc<dyn AuthIntegration> = wsi_auth::from_descriptor(&settings.api_v3_integration)?;

    Ok(AppState {
        settings,
        registry,
        slide_manager,
        auth,
        local_mapper,
    })
}

/// Builds the router. `app.py` mounts `/v3` as a separate application; nesting reproduces
/// the path prefixing and the second OpenAPI document.
pub fn build_router(state: AppState) -> Router {
    let mut root = routes::root::router(&state);
    let mut v3 = routes::v3::router(&state);

    if !state.settings.disable_openapi {
        root = root
            .route("/openapi.json", get(openapi::root_json))
            .route("/docs", get(openapi::docs));
        v3 = v3
            .route("/openapi.json", get(openapi::v3_json))
            .route("/docs", get(openapi::docs));
    }

    let mut app = root.nest("/v3", v3);

    // `if settings.cors_allow_origins:` — the middleware is only added when the list is
    // non-empty, so an unset value leaves responses without CORS headers entirely.
    if !state.settings.cors_allow_origins.is_empty() {
        app = app.layer(cors_layer(&state.settings));
    }

    app.with_state(state)
}

fn cors_layer(settings: &Settings) -> CorsLayer {
    let mut layer = CorsLayer::new().allow_methods(Any).allow_headers(Any);

    if settings.cors_allow_origins.contains("*") {
        // `allow_credentials` and a wildcard origin are mutually exclusive in the CORS
        // spec, and tower-http panics on the combination. Starlette resolves it by
        // reflecting the request origin, which is what `allow_credentials` with `["*"]`
        // effectively does there.
        if settings.cors_allow_credentials {
            layer = layer
                .allow_origin(tower_http::cors::AllowOrigin::mirror_request())
                .allow_credentials(true);
        } else {
            layer = layer.allow_origin(Any);
        }
    } else {
        let origins: Vec<http::HeaderValue> = settings
            .cors_allow_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        layer = layer
            .allow_origin(origins)
            .allow_credentials(settings.cors_allow_credentials);
    }

    layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_mode_descriptor_selects_a_mapper_and_bypasses_the_storage_service() {
        let settings = Settings {
            data_dir: std::env::temp_dir().to_string_lossy().replace('\\', "/"),
            local_mode: "wsi_service.paths_mapper:PathsMapper".into(),
            api_v3_integration: "wsi_service.api.v3.integrations.disable_auth:DisableAuth".into(),
            ..Settings::default()
        };

        let state = build_state(settings).unwrap();
        assert!(state.local_mode_enabled());
        assert!(matches!(
            state.slide_manager.resolver(),
            StorageResolver::Local(_)
        ));
    }

    #[test]
    fn without_local_mode_the_storage_mapper_service_is_used() {
        let settings = Settings {
            api_v3_integration: "wsi_service.api.v3.integrations.disable_auth:DisableAuth".into(),
            ..Settings::default()
        };
        let state = build_state(settings).unwrap();
        assert!(!state.local_mode_enabled());
        assert!(matches!(
            state.slide_manager.resolver(),
            StorageResolver::Remote { .. }
        ));
    }

    #[test]
    fn an_unknown_integration_descriptor_refuses_to_start() {
        let settings = Settings {
            api_v3_integration: "not.a:Thing".into(),
            ..Settings::default()
        };
        assert!(build_state(settings).is_err());
    }
}
