//! Shared application state.
//!
//! Mirrors the module-level singletons Python builds in `wsi_service/singletons.py`,
//! `wsi_service/app.py` and `wsi_service/api/v3/singletons.py`, but passed explicitly
//! rather than imported, so tests can construct a service without touching the process
//! environment.

use std::sync::Arc;

use wsi_auth::AuthIntegration;
use wsi_core::plugins::PluginRegistry;
use wsi_core::settings::Settings;
use wsi_core::slide_manager::SlideManager;
use wsi_core::storage::LocalMapper;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub registry: Arc<PluginRegistry>,
    pub slide_manager: Arc<SlideManager>,
    pub auth: Arc<dyn AuthIntegration>,
    /// Present only when `WS_LOCAL_MODE` is set; gates the local-mode routes exactly as
    /// `add_routes_v3` does.
    pub local_mapper: Option<Arc<dyn LocalMapper>>,
}

impl AppState {
    pub fn local_mode_enabled(&self) -> bool {
        self.local_mapper.is_some()
    }
}
