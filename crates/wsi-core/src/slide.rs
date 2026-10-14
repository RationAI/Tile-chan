//! The plugin-facing slide interface, mirroring `wsi_service/slide.py`.

use std::path::Path;

use async_trait::async_trait;
use bytes::Bytes;

use crate::error::{ApiError, Result};
use crate::icc::IccIntent;
use crate::models::SlideInfo;
use crate::pixels::Pixels;

/// Read parameters shared by `get_region` and `get_tile`.
///
/// Bundled into a struct because Python threads the same five optional arguments through
/// every plugin method and every wrapper; passing them positionally in Rust would be easy
/// to get subtly wrong.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadOptions {
    /// Resolved padding colour. `None` means "plugin decides", which every plugin
    /// resolves to `settings.padding_color`.
    pub padding_color: Option<(u8, u8, u8)>,
    pub z: u32,
    pub icc_profile_intent: Option<IccIntent>,
    pub icc_profile_strict: bool,
}

#[async_trait]
pub trait Slide: Send + Sync {
    async fn close(&self) -> Result<()>;

    async fn get_info(&self) -> Result<SlideInfo>;

    async fn get_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels>;

    async fn get_label(&self) -> Result<Pixels>;

    async fn get_macro(
        &self,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels>;

    /// `start_x`/`start_y` are signed: `get_extended_region` clamps out-of-bounds requests
    /// before calling, but plugins are still reached with coordinates at the extent edge,
    /// and the batch endpoints bypass clamping entirely.
    async fn get_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<Pixels>;

    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels>;

    /// Raw ICC profile bytes. Python's default raises 404 when there is no profile;
    /// plugins that cannot carry one never override this.
    async fn get_icc_profile(&self) -> Result<Bytes> {
        Err(ApiError::not_found("ICC Profile not available."))
    }

    /// Re-read the slide if the backing file changed.
    ///
    /// `SlideManager.get_slide` calls this inside a `try/except AttributeError`, so a
    /// plugin without the method is simply skipped. A defaulted no-op is the direct
    /// equivalent.
    async fn refresh(&self) -> Result<()> {
        Ok(())
    }

    fn filepath(&self) -> &Path;

    /// Set by the loader after a successful open (`plugins.py:93`), so the slide can report
    /// which plugin produced it — used when building the slide format identifier.
    fn plugin_name(&self) -> &str;
}

/// A boxed, shareable open slide.
pub type SharedSlide = std::sync::Arc<dyn Slide>;
