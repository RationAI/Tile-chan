//! Plugin registry and slide loading, ported from `wsi_service/plugins.py`.
//!
//! Python discovers plugins at import time by scanning `sys.path` for modules named
//! `wsi_service_plugin_*`. Rust has no equivalent, so the registry is built explicitly in
//! `wsi-server`, one entry per plugin, each behind a cargo feature. Everything else —
//! support detection, the priority gate, the ordering, the error strings — is reproduced.
//!
//! One ordering detail worth stating: Python sorts the supported plugins by priority with
//! `sorted(..., reverse=True)`, which is stable, so equal priorities fall back to the
//! discovery order of `pkgutil.iter_modules()`. That order is filesystem-dependent and
//! therefore unspecified. Here the registry is sorted by plugin name at construction and
//! the priority sort is stable on top of it, which makes ties deterministic.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{ApiError, Result};
use crate::models::PluginInfo;
use crate::settings::Settings;
use crate::slide::Slide;

#[async_trait]
pub trait WsiPlugin: Send + Sync {
    /// Name without the `wsi_service_plugin_` prefix, e.g. `openslide`.
    fn name(&self) -> &'static str;

    /// Reported by `/alive`; Python reads it from the plugin's package metadata.
    fn version(&self) -> &'static str;

    /// The plugin's own `priority` module attribute. Absent in Python means 0.
    fn default_priority(&self) -> i64 {
        0
    }

    /// `is_supported(filepath)`, or the `supported_file_extensions` membership test for
    /// plugins that only declare extensions.
    fn is_supported(&self, path: &Path) -> bool;

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>>;

    /// Optional lifespan hooks; `app.py:31` calls them if present.
    fn start(&self) {}
    fn stop(&self) {}
}

pub struct PluginRegistry {
    plugins: Vec<Arc<dyn WsiPlugin>>,
}

impl PluginRegistry {
    /// Sorts by name so that equal-priority ties resolve deterministically.
    pub fn new(mut plugins: Vec<Arc<dyn WsiPlugin>>) -> Self {
        plugins.sort_by_key(|p| p.name());
        Self { plugins }
    }

    pub fn plugins(&self) -> &[Arc<dyn WsiPlugin>] {
        &self.plugins
    }

    /// `_get_plugin_priority` — `plugins.py:81`. The environment override wins.
    pub fn priority(plugin: &dyn WsiPlugin) -> i64 {
        Settings::plugin_priority_override(plugin.name())
            .unwrap_or_else(|| plugin.default_priority())
    }

    /// `_get_supported_plugins` — `plugins.py:63`. Plugins with a negative priority are
    /// excluded outright, which is how an operator disables one.
    pub fn supported(&self, path: &Path) -> Vec<Arc<dyn WsiPlugin>> {
        self.plugins
            .iter()
            .filter(|p| Self::priority(p.as_ref()) >= 0)
            .filter(|p| p.is_supported(path))
            .cloned()
            .collect()
    }

    /// `_get_sorted_plugins` — `plugins.py:77`. Highest priority first, stable.
    fn sorted_supported(&self, path: &Path) -> Vec<Arc<dyn WsiPlugin>> {
        let mut supported = self.supported(path);
        supported.sort_by_key(|p| std::cmp::Reverse(Self::priority(p.as_ref())));
        supported
    }

    /// `is_supported_format` — `plugins.py:59`. Used by the local mappers to decide which
    /// directory entries are slides.
    pub fn is_supported_format(&self, path: &Path) -> bool {
        !self.supported(path).is_empty()
    }

    /// `get_plugins_overview` — `plugins.py:48`.
    pub fn overview(&self) -> Vec<PluginInfo> {
        self.plugins
            .iter()
            .map(|p| PluginInfo {
                name: p.name().to_owned(),
                version: p.version().to_owned(),
                priority: Self::priority(p.as_ref()),
            })
            .collect()
    }

    pub fn start_all(&self) {
        for p in &self.plugins {
            p.start();
        }
    }

    pub fn stop_all(&self) {
        for p in &self.plugins {
            p.stop();
        }
    }

    /// `load_slide` — `plugins.py:19`.
    pub async fn load_slide(
        &self,
        filepath: &Path,
        plugin: Option<&str>,
    ) -> Result<Arc<dyn Slide>> {
        if !filepath.exists() {
            return Err(ApiError::internal(format!(
                "File {} not found.",
                filepath.display()
            )));
        }

        let supported = self.sorted_supported(filepath);
        tracing::info!(
            "Slide supports {:?}",
            supported.iter().map(|p| p.name()).collect::<Vec<_>>()
        );

        if supported.is_empty() {
            return Err(ApiError::internal(
                "There is no plugin available that does support this slide.",
            ));
        }

        if let Some(requested) = plugin.filter(|p| !p.is_empty()) {
            let Some(chosen) = supported.iter().find(|p| p.name() == requested) else {
                return Err(ApiError::internal(format!(
                    "Selected plugin {requested} is not available or does not support this slide. Please specify another plugin."
                )));
            };
            return open_slide(chosen.as_ref(), filepath).await;
        }

        // Try each supported plugin in priority order, accumulating failures exactly the
        // way Python does: `exception_details += e.detail + ". "`.
        let mut exception_details = String::new();
        for candidate in &supported {
            tracing::info!("ATTEMPT TO USE {}", candidate.name());
            match open_slide(candidate.as_ref(), filepath).await {
                Ok(slide) => return Ok(slide),
                Err(err) => {
                    exception_details.push_str(&err.detail_text());
                    exception_details.push_str(". ");
                }
            }
        }
        Err(ApiError::internal(exception_details))
    }
}

/// `_open_slide` — `plugins.py:89`.
async fn open_slide(plugin: &dyn WsiPlugin, filepath: &Path) -> Result<Arc<dyn Slide>> {
    tracing::info!("Using plugin {}", plugin.name());
    plugin.open(filepath).await.map_err(|e| {
        ApiError::internal(format!(
            "Plugin {} unable to open image ({})",
            plugin.name(),
            e.detail_text()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SlideInfo;
    use crate::pixels::Pixels;
    use crate::slide::ReadOptions;
    use std::path::PathBuf;

    struct StubSlide(&'static str);

    #[async_trait]
    impl Slide for StubSlide {
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        async fn get_info(&self) -> Result<SlideInfo> {
            unimplemented!()
        }
        async fn get_thumbnail(
            &self,
            _: u32,
            _: u32,
            _: Option<crate::icc::IccIntent>,
            _: bool,
        ) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_label(&self) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_macro(&self, _: Option<crate::icc::IccIntent>, _: bool) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_region(
            &self,
            _: usize,
            _: i64,
            _: i64,
            _: u32,
            _: u32,
            _: ReadOptions,
        ) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_tile(&self, _: usize, _: i64, _: i64, _: ReadOptions) -> Result<Pixels> {
            unimplemented!()
        }
        fn filepath(&self) -> &Path {
            Path::new("")
        }
        fn plugin_name(&self) -> &str {
            self.0
        }
    }

    struct StubPlugin {
        name: &'static str,
        priority: i64,
        supports: bool,
        fails_with: Option<&'static str>,
    }

    #[async_trait]
    impl WsiPlugin for StubPlugin {
        fn name(&self) -> &'static str {
            self.name
        }
        fn version(&self) -> &'static str {
            "1.0.0"
        }
        fn default_priority(&self) -> i64 {
            self.priority
        }
        fn is_supported(&self, _: &Path) -> bool {
            self.supports
        }
        async fn open(&self, _: &Path) -> Result<Arc<dyn Slide>> {
            match self.fails_with {
                Some(detail) => Err(ApiError::internal(detail)),
                None => Ok(Arc::new(StubSlide(self.name))),
            }
        }
    }

    fn plugin(
        name: &'static str,
        priority: i64,
        supports: bool,
        fails_with: Option<&'static str>,
    ) -> Arc<dyn WsiPlugin> {
        Arc::new(StubPlugin {
            name,
            priority,
            supports,
            fails_with,
        })
    }

    fn existing_file() -> PathBuf {
        // Any path guaranteed to exist for the `exists()` guard.
        std::env::current_exe().expect("test binary path")
    }

    /// `Result::unwrap_err` needs `T: Debug`, and `Arc<dyn Slide>` cannot provide it.
    fn expect_err(result: Result<Arc<dyn Slide>>) -> ApiError {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        }
    }

    #[test]
    fn registry_is_sorted_by_name_for_deterministic_ties() {
        let reg = PluginRegistry::new(vec![
            plugin("zeta", 0, true, None),
            plugin("alpha", 0, true, None),
        ]);
        assert_eq!(
            reg.plugins().iter().map(|p| p.name()).collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
    }

    #[test]
    fn negative_priority_excludes_a_plugin_entirely() {
        let reg = PluginRegistry::new(vec![
            plugin("off", -1, true, None),
            plugin("on", 0, true, None),
        ]);
        let supported = reg.supported(Path::new("x.svs"));
        assert_eq!(supported.len(), 1);
        assert_eq!(supported[0].name(), "on");
    }

    #[tokio::test]
    async fn highest_priority_plugin_wins() {
        let reg = PluginRegistry::new(vec![
            plugin("low", 0, true, None),
            plugin("high", 10, true, None),
        ]);
        let slide = reg.load_slide(&existing_file(), None).await.unwrap();
        assert_eq!(slide.plugin_name(), "high");
    }

    #[tokio::test]
    async fn failures_fall_through_to_the_next_plugin() {
        let reg = PluginRegistry::new(vec![
            plugin("broken", 10, true, Some("boom")),
            plugin("working", 0, true, None),
        ]);
        let slide = reg.load_slide(&existing_file(), None).await.unwrap();
        assert_eq!(slide.plugin_name(), "working");
    }

    #[tokio::test]
    async fn all_failures_are_concatenated_into_one_500() {
        let reg = PluginRegistry::new(vec![
            plugin("a", 1, true, Some("first")),
            plugin("b", 0, true, Some("second")),
        ]);
        let err = expect_err(reg.load_slide(&existing_file(), None).await);
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            err.detail_text(),
            "Plugin a unable to open image (first). Plugin b unable to open image (second). "
        );
    }

    #[tokio::test]
    async fn explicit_unsupported_plugin_is_rejected_with_the_python_message() {
        let reg = PluginRegistry::new(vec![plugin("a", 0, true, None)]);
        let err = expect_err(reg.load_slide(&existing_file(), Some("b")).await);
        assert_eq!(
            err.detail_text(),
            "Selected plugin b is not available or does not support this slide. Please specify another plugin."
        );
    }

    #[tokio::test]
    async fn no_supporting_plugin_is_a_500() {
        let reg = PluginRegistry::new(vec![plugin("a", 0, false, None)]);
        let err = expect_err(reg.load_slide(&existing_file(), None).await);
        assert_eq!(
            err.detail_text(),
            "There is no plugin available that does support this slide."
        );
    }

    #[tokio::test]
    async fn missing_file_is_reported_before_plugin_selection() {
        let reg = PluginRegistry::new(vec![plugin("a", 0, true, None)]);
        let err = expect_err(
            reg.load_slide(Path::new("definitely/not/here.svs"), None)
                .await,
        );
        assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.detail_text().ends_with("not found."));
    }
}
