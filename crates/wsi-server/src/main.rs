//! Service entry point.
//!
//! Replaces `uvicorn wsi_service.app:app`. Startup and shutdown mirror the FastAPI
//! lifespan in `wsi_service/app.py:26`: warn when the data directory is missing, start the
//! plugins, and on shutdown stop the plugins and close every cached slide handle.

use std::net::SocketAddr;

use tracing_subscriber::EnvFilter;

use wsi_core::settings::Settings;
use wsi_server::{build_router, build_state};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `pydantic-settings` reads `.env` from the working directory; so does this.
    let _ = dotenvy::dotenv();
    let settings = Settings::load();

    init_tracing(settings.debug);

    tracing::info!("🚀 LIFESPAN: startup (data dir: {})", settings.data_dir);
    if !std::path::Path::new(&settings.data_dir).is_dir() {
        tracing::warn!("Data directory does not exist: {}", settings.data_dir);
    }

    let port: u16 = std::env::var("WS_PORT")
        .or_else(|_| std::env::var("PORT"))
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let address = SocketAddr::from(([0, 0, 0, 0], port));

    let state = build_state(settings)?;
    state.registry.start_all();

    let slide_manager = std::sync::Arc::clone(&state.slide_manager);
    let registry = std::sync::Arc::clone(&state.registry);
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!("Listening on http://{address}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("💥 LIFESPAN: shutdown");
    registry.stop_all();
    slide_manager.close().await;

    Ok(())
}

fn init_tracing(debug: bool) {
    let default = if debug { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("wsi_server={default},wsi_core={default}")));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

/// Ctrl-C, plus SIGTERM on Unix so container shutdown runs the same path.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
