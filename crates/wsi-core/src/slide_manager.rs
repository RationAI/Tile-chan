//! Open-slide cache and slide-info assembly, ported from `wsi_service/slide_manager.py`.
//!
//! Two independent lifetimes govern a cached slide handle:
//!
//! * an **LRU bound** of `image_handle_cache_size` entries — evicting closes the slide;
//! * an **inactivity timer** of `inactive_histo_image_timeout_seconds`, reset on every
//!   access — firing closes the slide and drops it from the cache.
//!
//! Python implements the timer with `loop.call_later` and cancels the handle on each
//! access. Here it is a spawned task per entry with its `JoinHandle` aborted and respawned,
//! which has the same observable behaviour.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use lru::LruCache;
use tokio::sync::Mutex;

use crate::error::{ApiError, Result};
use crate::models::SlideInfo;
use crate::plugins::PluginRegistry;
use crate::slide::Slide;
use crate::storage::{join_data_dir, StorageResolver, StorageResolverExt};

struct CachedSlide {
    slide: Arc<dyn Slide>,
    /// Aborting this cancels the pending close.
    expiry: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for CachedSlide {
    fn drop(&mut self) {
        if let Some(handle) = self.expiry.take() {
            handle.abort();
        }
    }
}

struct Inner {
    data_dir: String,
    timeout: Duration,
    cache: Mutex<LruCache<String, CachedSlide>>,
    /// One lock per cache id so concurrent first-opens of the same slide do not race.
    storage_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    resolver: StorageResolver,
    registry: Arc<PluginRegistry>,
}

pub struct SlideManager {
    inner: Arc<Inner>,
}

impl SlideManager {
    pub fn new(
        resolver: StorageResolver,
        registry: Arc<PluginRegistry>,
        data_dir: impl Into<String>,
        timeout_seconds: u64,
        cache_size: usize,
    ) -> Self {
        let capacity = NonZeroUsize::new(cache_size.max(1)).expect("non-zero");
        Self {
            inner: Arc::new(Inner {
                data_dir: data_dir.into(),
                timeout: Duration::from_secs(timeout_seconds),
                cache: Mutex::new(LruCache::new(capacity)),
                storage_locks: Mutex::new(HashMap::new()),
                resolver,
                registry,
            }),
        }
    }

    pub fn registry(&self) -> &Arc<PluginRegistry> {
        &self.inner.registry
    }

    pub fn resolver(&self) -> &StorageResolver {
        &self.inner.resolver
    }

    pub fn data_dir(&self) -> &str {
        &self.inner.data_dir
    }

    /// `cache_id = slide_id + f" ({plugin})" if plugin else slide_id` —
    /// `slide_manager.py:30`.
    fn cache_id(slide_id: &str, plugin: Option<&str>) -> String {
        match plugin.filter(|p| !p.is_empty()) {
            Some(p) => format!("{slide_id} ({p})"),
            None => slide_id.to_owned(),
        }
    }

    /// `get_slide` — `slide_manager.py:29`.
    pub async fn get_slide(&self, slide_id: &str, plugin: Option<&str>) -> Result<Arc<dyn Slide>> {
        let cache_id = Self::cache_id(slide_id, plugin);

        if let Some(slide) = self.take_cached(&cache_id).await {
            self.reset_expiration(&cache_id).await;
            // Python calls `refresh()` if the plugin defines it, swallowing AttributeError.
            slide.refresh().await.ok();
            return Ok(slide);
        }

        // Serialise opens of the same slide.
        let lock = self.storage_lock(&cache_id).await;
        let _guard = lock.lock().await;

        // Another task may have finished opening while we waited.
        if let Some(slide) = self.take_cached(&cache_id).await {
            self.reset_expiration(&cache_id).await;
            slide.refresh().await.ok();
            return Ok(slide);
        }

        let main = self.inner.resolver.main_storage_address(slide_id).await?;
        let storage_address = join_data_dir(&self.inner.data_dir, &main.address);
        tracing::debug!("Storage address for slide {slide_id}: {storage_address}");

        let slide = self
            .inner
            .registry
            .load_slide(Path::new(&storage_address), plugin)
            .await?;

        {
            let mut cache = self.inner.cache.lock().await;
            // `LruCache::push` returns the entry it displaced; dropping it aborts that
            // entry's timer, and the slide is closed below.
            if let Some((_, evicted)) = cache.push(
                cache_id.clone(),
                CachedSlide {
                    slide: Arc::clone(&slide),
                    expiry: None,
                },
            ) {
                let evicted_slide = Arc::clone(&evicted.slide);
                drop(evicted);
                let _ = evicted_slide.close().await;
            }
        }
        tracing::debug!("New slide handle opened for storage address: {storage_address}");

        self.reset_expiration(&cache_id).await;
        slide.refresh().await.ok();
        Ok(slide)
    }

    /// `get_slide_info` — `slide_manager.py:60`.
    ///
    /// Overwrites the plugin's placeholder id with the requested one, flags raw download
    /// when the file is on disk, and extends the format identifier.
    pub async fn get_slide_info(&self, slide_id: &str, plugin: Option<&str>) -> Result<SlideInfo> {
        let slide = self.get_slide(slide_id, plugin).await?;
        let mut info = slide.get_info().await?;
        info.id = slide_id.to_owned();
        extend_slide_format_identifier(slide.as_ref(), &mut info);
        if slide.filepath().exists() {
            info.raw_download = true;
        }
        Ok(info)
    }

    /// `get_slide_file_paths` — `slide_manager.py:74`.
    pub async fn get_slide_file_paths(&self, slide_id: &str) -> Result<Vec<String>> {
        self.inner
            .resolver
            .file_paths(&self.inner.data_dir, slide_id)
            .await
    }

    /// `close` — `slide_manager.py:78`. Cancels every timer and closes every handle.
    pub async fn close(&self) {
        let entries: Vec<(String, Arc<dyn Slide>)> = {
            let mut cache = self.inner.cache.lock().await;
            let ids: Vec<String> = cache.iter().map(|(k, _)| k.clone()).collect();
            ids.into_iter()
                .filter_map(|id| cache.pop(&id).map(|entry| (id, Arc::clone(&entry.slide))))
                .collect()
        };
        for (cache_id, slide) in entries {
            let _ = slide.close().await;
            tracing::debug!("Closed slide with storage address: {cache_id}");
        }
        self.inner.storage_locks.lock().await.clear();
    }

    async fn take_cached(&self, cache_id: &str) -> Option<Arc<dyn Slide>> {
        let mut cache = self.inner.cache.lock().await;
        cache.get(cache_id).map(|entry| Arc::clone(&entry.slide))
    }

    async fn storage_lock(&self, cache_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.inner.storage_locks.lock().await;
        Arc::clone(
            locks
                .entry(cache_id.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    /// `_reset_slide_expiration` — `slide_manager.py:88`.
    async fn reset_expiration(&self, cache_id: &str) {
        let weak: Weak<Inner> = Arc::downgrade(&self.inner);
        let id = cache_id.to_owned();
        let timeout = self.inner.timeout;

        let handle = tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            let Some(inner) = weak.upgrade() else { return };
            let evicted = {
                let mut cache = inner.cache.lock().await;
                cache.pop(&id)
            };
            if let Some(entry) = evicted {
                inner.storage_locks.lock().await.remove(&id);
                let slide = Arc::clone(&entry.slide);
                // Prevent the Drop impl from aborting the task we are running inside.
                let mut entry = entry;
                entry.expiry = None;
                drop(entry);
                let _ = slide.close().await;
                tracing::debug!("Closed slide with storage address: {id}");
            }
        });

        let mut cache = self.inner.cache.lock().await;
        if let Some(entry) = cache.get_mut(cache_id) {
            if let Some(previous) = entry.expiry.replace(handle) {
                previous.abort();
            }
            tracing::debug!(
                "Set expiration timer for storage address ({cache_id}): {}",
                timeout.as_secs()
            );
        } else {
            // The entry vanished between the read and here; nothing to arm.
            handle.abort();
        }
    }
}

/// `_extend_slide_format_identifier` — `slide_manager.py:150`.
///
/// Builds `{file|folder}-{extension}-{plugin identifier}-{plugin name}`, e.g.
/// `file-svs-aperio-openslide` or `folder-dicom-wsidicom`, then lowercases the result.
pub fn extend_slide_format_identifier(slide: &dyn Slide, info: &mut SlideInfo) {
    let path = slide.filepath();

    if !info.format.contains("file") && !info.format.contains("folder") {
        if path.is_file() {
            let extension = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();
            info.format = format!("file-{extension}-{}", info.format);
        } else if path.is_dir() {
            info.format = format!("folder-{}", info.format);
        }
    }

    let plugin = slide.plugin_name();
    if !info.format.contains(plugin) {
        if !info.format.is_empty() && !info.format.ends_with('-') {
            info.format.push('-');
        }
        info.format.push_str(plugin);
    }

    info.format = info.format.to_lowercase();
}

/// Convenience used by the endpoints: a missing slide is a 404, everything else passes
/// through unchanged.
pub fn require_slide(result: Result<Arc<dyn Slide>>) -> Result<Arc<dyn Slide>> {
    result.map_err(|err| {
        if err.status == http::StatusCode::NOT_FOUND {
            ApiError::not_found(err.detail_text())
        } else {
            err
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icc::IccIntent;
    use crate::models::{SlideExtent, SlidePixelSizeNm};
    use crate::pixels::Pixels;
    use crate::slide::ReadOptions;
    use async_trait::async_trait;
    use std::path::PathBuf;

    struct FakeSlide {
        path: PathBuf,
        plugin: &'static str,
        format: String,
    }

    #[async_trait]
    impl Slide for FakeSlide {
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        async fn get_info(&self) -> Result<SlideInfo> {
            Ok(SlideInfo {
                id: String::new(),
                channels: None,
                channel_depth: Some(8),
                extent: SlideExtent::new(1, 1, 1),
                num_levels: 1,
                pixel_size_nm: SlidePixelSizeNm::xy(1.0, 1.0),
                tile_extent: SlideExtent::new(1, 1, 1),
                levels: vec![],
                format: self.format.clone(),
                raw_download: false,
            })
        }
        async fn get_thumbnail(
            &self,
            _: u32,
            _: u32,
            _: Option<IccIntent>,
            _: bool,
        ) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_label(&self) -> Result<Pixels> {
            unimplemented!()
        }
        async fn get_macro(&self, _: Option<IccIntent>, _: bool) -> Result<Pixels> {
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
            &self.path
        }
        fn plugin_name(&self) -> &str {
            self.plugin
        }
    }

    fn fake(path: &str, plugin: &'static str, format: &str) -> FakeSlide {
        FakeSlide {
            path: PathBuf::from(path),
            plugin,
            format: format.to_owned(),
        }
    }

    async fn format_of(slide: &FakeSlide) -> String {
        let mut info = slide.get_info().await.unwrap();
        extend_slide_format_identifier(slide, &mut info);
        info.format
    }

    #[test]
    fn cache_id_appends_the_plugin_when_one_is_pinned() {
        assert_eq!(SlideManager::cache_id("abc", None), "abc");
        assert_eq!(SlideManager::cache_id("abc", Some("")), "abc");
        assert_eq!(
            SlideManager::cache_id("abc", Some("openslide")),
            "abc (openslide)"
        );
    }

    #[tokio::test]
    async fn format_identifier_appends_the_plugin_name() {
        // The path does not exist, so neither the file nor the folder prefix is added and
        // only the plugin suffix appears — the same as Python on a missing path.
        let slide = fake("/nowhere/CMU-1.svs", "openslide", "aperio");
        assert_eq!(format_of(&slide).await, "aperio-openslide");
    }

    #[tokio::test]
    async fn format_identifier_is_lowercased() {
        let slide = fake("/nowhere/x.svs", "openslide", "APERIO");
        assert_eq!(format_of(&slide).await, "aperio-openslide");
    }

    #[tokio::test]
    async fn format_identifier_does_not_repeat_the_plugin_name() {
        let slide = fake("/nowhere/x.svs", "openslide", "openslide");
        assert_eq!(format_of(&slide).await, "openslide");
    }

    #[tokio::test]
    async fn empty_plugin_format_yields_just_the_plugin_name() {
        let slide = fake("/nowhere/x.svs", "pil", "");
        assert_eq!(format_of(&slide).await, "pil");
    }

    #[tokio::test]
    async fn existing_file_gets_the_file_and_extension_prefix() {
        let exe = std::env::current_exe().unwrap();
        let extension = exe
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_lowercase();
        let slide = FakeSlide {
            path: exe,
            plugin: "openslide",
            format: "aperio".into(),
        };
        let expected = format!("file-{extension}-aperio-openslide");
        assert_eq!(format_of(&slide).await, expected);
    }

    #[tokio::test]
    async fn existing_directory_gets_the_folder_prefix() {
        let dir = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let slide = FakeSlide {
            path: dir,
            plugin: "wsidicom",
            format: "dicom".into(),
        };
        assert_eq!(format_of(&slide).await, "folder-dicom-wsidicom");
    }
}
