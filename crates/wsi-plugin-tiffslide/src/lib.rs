//! The `tiffslide` plugin: Aperio SVS and generic pyramidal TIFF.
//!
//! Ported from `wsi_service_base_plugins/tiffslide/wsi_service_plugin_tiffslide/`.
//!
//! `priority = 1`, below `openslide`'s 10, so this is the fallback that serves these
//! formats when libopenslide is not available or when a client asks for `plugin=tiffslide`
//! explicitly.
//!
//! The plugin itself is thin; almost everything it reports comes from `tiffslide`, whose
//! decisions [`reader`] reproduces. Three consequences are worth stating up front:
//!
//! - Only Aperio, Ventana and *vendorless* slides get past `supported_vendors`. A Leica
//!   SCN or Philips TIFF reaches `__get_pixel_size`'s dead `else` branch and comes back as
//!   `404 Failed to gather slide infos`. A Ventana slide does not, because tiffslide's
//!   `parse_ventana` delegates to `parse_generic_tiff` and leaves the vendor unset.
//! - Hamamatsu NDPI is claimed by `is_supported`, as in Python, but refused by `open`:
//!   its pages cannot be decoded here at all. See `docs/divergences.md`, divergence 33.
//! - `is_supported` declines any name *ending* in `ome.tif` — no dot required, so
//!   `metabolome.tif` goes with it. Divergence 34.

pub mod reader;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use image::DynamicImage;
use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::{IccIntent, IccProfileCache};
use wsi_core::image::{pillow_thumbnail, rgba_to_rgb_with_background_color, Filter};
use wsi_core::models::{
    original_levels, rgb_channel_list, tile_read_size, SlideExtent, SlideInfo, SlidePixelSizeNm,
};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

use reader::TiffSlideReader;

/// `suffix in [".svs", ".tiff", ".tif", ".ndpi"]`
const SUPPORTED_EXTENSIONS: [&str; 4] = ["svs", "tiff", "tif", "ndpi"];

const DEFAULT_TILE_SIZE: i64 = 256;

/// Reported when the file records no pixel size — the same sentinel `pil` and `mvt` use.
const UNKNOWN_PIXEL_SIZE_NM: SlidePixelSizeNm = SlidePixelSizeNm {
    x: -1.0,
    y: -1.0,
    z: None,
};

pub struct TiffSlidePlugin {
    settings: Arc<Settings>,
}

impl TiffSlidePlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for TiffSlidePlugin {
    fn name(&self) -> &'static str {
        "tiffslide"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// `priority = 1`
    fn default_priority(&self) -> i64 {
        1
    }

    /// `is_supported`: an OME-TIFF is explicitly declined, so it falls to `tifffile`.
    fn is_supported(&self, path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let name = name.to_ascii_lowercase();
        if name.ends_with("ome.tif") || name.ends_with("ome.tiff") {
            return false;
        }
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || TiffSlideSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

pub struct TiffSlideSlide {
    path: PathBuf,
    reader: Mutex<Option<TiffSlideReader>>,
    info: SlideInfo,
    icc_profile: Option<Vec<u8>>,
    icc: IccProfileCache,
    settings: Arc<Settings>,
    /// `self.thumbnail`, computed once and then reused.
    ///
    /// Python caches with `hasattr`, so the *first* call's `icc_profile_intent` decides
    /// the colours every later call sees. Reproduced.
    thumbnail: Mutex<Option<DynamicImage>>,
}

impl TiffSlideSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        let reader = TiffSlideReader::open(&path)?;
        let info = slide_info(&reader)?;
        let icc_profile = reader.icc_profile.clone();
        Ok(Self {
            path,
            reader: Mutex::new(Some(reader)),
            info,
            icc_profile,
            icc: IccProfileCache::new(),
            settings,
            thumbnail: Mutex::new(None),
        })
    }

    fn with_reader<T>(&self, f: impl FnOnce(&mut TiffSlideReader) -> Result<T>) -> Result<T> {
        let mut guard = self.reader.lock();
        match guard.as_mut() {
            Some(reader) => f(reader),
            None => Err(ApiError::unhandled(
                "AttributeError(\"'NoneType' object has no attribute 'read_region'\")",
            )),
        }
    }

    fn apply_icc(
        &self,
        image: DynamicImage,
        intent: Option<IccIntent>,
        strict: bool,
        cache: bool,
    ) -> Result<DynamicImage> {
        match intent {
            None => Ok(image),
            Some(intent) => self.icc.process_image_lenient(
                image,
                self.icc_profile.as_deref(),
                strict,
                intent,
                cache,
            ),
        }
    }

    /// `__get_associated_image`.
    fn associated_image(
        &self,
        name: &str,
        intent: Option<IccIntent>,
        strict: bool,
    ) -> Result<DynamicImage> {
        let image = self.with_reader(|reader| match reader.associated_image(name) {
            Some(result) => result.map(DynamicImage::ImageRgb8),
            None => Err(ApiError::not_found(format!(
                "Associated image {name} does not exist."
            ))),
        })?;
        // `cache=False`: an associated image's profile need not be the region profile.
        self.apply_icc(image, intent, strict, false)
    }

    /// `get_region` without the `Pixels` wrapper, so the thumbnail path can reuse it.
    #[allow(clippy::too_many_arguments)]
    fn read_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        padding_color: (u8, u8, u8),
        intent: Option<IccIntent>,
        strict: bool,
    ) -> Result<DynamicImage> {
        let Some(slide_level) = self.info.levels.get(level) else {
            return Err(ApiError::unhandled("IndexError('list index out of range')"));
        };
        let downsample = slide_level.downsample_factor;
        let location = adapt_level_0_location(start_x, start_y, downsample);

        let image = self.with_reader(|reader| {
            reader
                .read_region(location, level, (size_x, size_y))
                .map(DynamicImage::ImageRgb8)
        })?;
        let image = self.apply_icc(image, intent, strict, true)?;
        // A no-op for an RGB image, which is what `read_region` always returns — the
        // padding tiffslide added is already black.
        Ok(rgba_to_rgb_with_background_color(image, padding_color))
    }

    /// `__get_best_level_for_thumbnail` — the same off-by-one as openslide's, and the same
    /// negative-index consequence. See divergence 7.
    fn best_level_for_thumbnail(&self, max_x: u32, max_y: u32) -> usize {
        let levels = &self.info.levels;
        for (index, level) in levels.iter().enumerate() {
            if level.extent.x < max_x as i64 && level.extent.y < max_y as i64 {
                return if index == 0 {
                    // Python's `levels[-1]`.
                    levels.len().saturating_sub(1)
                } else {
                    index - 1
                };
            }
        }
        levels.len().saturating_sub(1)
    }

    /// `__get_thumbnail` — a whole level, shrunk to the box.
    fn level_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        intent: Option<IccIntent>,
        strict: bool,
    ) -> Result<DynamicImage> {
        let level = self.best_level_for_thumbnail(max_x, max_y);
        let extent = self
            .info
            .levels
            .get(level)
            .map(|l| l.extent)
            .unwrap_or(self.info.extent);
        let region = self
            .read_region(
                level,
                0,
                0,
                extent.x as u32,
                extent.y as u32,
                self.settings.padding_color,
                intent,
                strict,
            )
            .map_err(|e| {
                ApiError::internal(format!(
                    "Failed to extract thumbnail from WSI [{}].",
                    e.detail_text()
                ))
            })?;
        Ok(pillow_thumbnail(&region, max_x, max_y, Filter::Bicubic))
    }
}

#[async_trait]
impl Slide for TiffSlideSlide {
    async fn close(&self) -> Result<()> {
        *self.reader.lock() = None;
        self.icc.free_cache();
        Ok(())
    }

    async fn get_info(&self) -> Result<SlideInfo> {
        Ok(self.info.clone())
    }

    async fn get_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let padding_color = opts.padding_color.unwrap_or(self.settings.padding_color);
        let image = self.read_region(
            level,
            start_x,
            start_y,
            size_x,
            size_y,
            padding_color,
            opts.icc_profile_intent,
            opts.icc_profile_strict,
        )?;
        Ok(Pixels::Image(image))
    }

    /// `get_tile` — unlike openslide, the read is shrunk to the part of the tile that is
    /// inside the level before it is issued.
    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let Some(level_extent) = self.info.levels.get(level).map(|l| l.extent) else {
            return Err(ApiError::unhandled("IndexError('list index out of range')"));
        };
        let tile = self.info.tile_extent;
        let (width, height) = tile_read_size(level_extent, tile, tile_x, tile_y);
        // `z` is dropped: the Python call passes a literal 0.
        self.get_region(
            level,
            tile_x * tile.x,
            tile_y * tile.y,
            width as u32,
            height as u32,
            ReadOptions { z: 0, ..opts },
        )
        .await
    }

    async fn get_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        let cached = {
            let guard = self.thumbnail.lock();
            guard.clone()
        };
        let source = match cached {
            Some(image) => image,
            None => {
                // The embedded thumbnail first; a level read only if there is none.
                let image = match self.associated_image(
                    "thumbnail",
                    icc_profile_intent,
                    icc_profile_strict,
                ) {
                    Ok(image) => image,
                    Err(_) => self.level_thumbnail(
                        self.settings.max_thumbnail_size,
                        self.settings.max_thumbnail_size,
                        icc_profile_intent,
                        icc_profile_strict,
                    )?,
                };
                *self.thumbnail.lock() = Some(image.clone());
                image
            }
        };
        Ok(Pixels::Image(pillow_thumbnail(
            &source,
            max_x,
            max_y,
            Filter::Bicubic,
        )))
    }

    /// `get_label` passes no intent, so a label is never colour-managed.
    async fn get_label(&self) -> Result<Pixels> {
        Ok(Pixels::Image(self.associated_image("label", None, false)?))
    }

    async fn get_macro(
        &self,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        Ok(Pixels::Image(self.associated_image(
            "macro",
            icc_profile_intent,
            icc_profile_strict,
        )?))
    }

    async fn get_icc_profile(&self) -> Result<Bytes> {
        IccProfileCache::payload(self.icc_profile.as_deref())
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "tiffslide"
    }
}

/// `__adapt_level_0_location`.
///
/// The level-0 origin is nudged so that dividing it back down lands on the requested level
/// coordinate, which is what openslide does and tiffslide does not — see tiffslide issue
/// #63.
fn adapt_level_0_location(start_x: i64, start_y: i64, downsample: f64) -> (i64, i64) {
    let mut x = (start_x as f64 * downsample) as i64;
    let mut y = (start_y as f64 * downsample) as i64;
    if (x as f64 / downsample) as i64 != start_x {
        x += (((x as f64 / downsample) as i64 - start_x).abs() as f64 * downsample) as i64;
    }
    if (y as f64 / downsample) as i64 != start_y {
        y += (((y as f64 / downsample) as i64 - start_y).abs() as f64 * downsample) as i64;
    }
    (x, y)
}

/// `__get_slide_info`.
fn slide_info(reader: &TiffSlideReader) -> Result<SlideInfo> {
    let (width, height) = reader.dimensions();
    let dimensions = reader.level_dimensions();
    let downsamples = reader.level_downsamples();
    let levels = original_levels(&dimensions, &downsamples);

    Ok(SlideInfo {
        id: String::new(),
        channels: Some(rgb_channel_list()),
        channel_depth: Some(8),
        extent: SlideExtent::new(width, height, 1),
        num_levels: levels.len() as i64,
        pixel_size_nm: pixel_size(reader),
        tile_extent: tile_extent(reader),
        levels,
        format: reader.format.as_str().to_owned(),
        raw_download: false,
    })
}

/// `__get_pixel_size`.
///
/// In Python the `else` branch is a dead `SlidePixelSizeNm()` that raises, and a supported
/// vendor with no `mpp` reaches `float(None)`, which raises too — both surface as a 404 on
/// a slide that is otherwise perfectly readable. A TIFF written with `ResolutionUnit` NONE
/// hits exactly that. Here both report `UNKNOWN_PIXEL_SIZE_NM` instead, the `-1` the `pil`
/// and `mvt` plugins already use. See `docs/divergences.md`, divergence 39.
fn pixel_size(reader: &TiffSlideReader) -> SlidePixelSizeNm {
    if !reader::vendor_is_supported(reader.properties.vendor.as_deref()) {
        return UNKNOWN_PIXEL_SIZE_NM;
    }
    match (reader.properties.mpp_x, reader.properties.mpp_y) {
        (Some(x), Some(y)) => SlidePixelSizeNm::xy(1000.0 * x, 1000.0 * y),
        _ => UNKNOWN_PIXEL_SIZE_NM,
    }
}

/// `__get_tile_extent`.
///
/// The two properties are always present — `collect_level_info` sets them for every
/// level — so the 256 default is only reached when the file has no level at all.
fn tile_extent(reader: &TiffSlideReader) -> SlideExtent {
    match reader.properties.level_tiles.first() {
        Some(&(width, height)) => SlideExtent::new(width, height, 1),
        None => SlideExtent::new(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin() -> TiffSlidePlugin {
        TiffSlidePlugin::new(Arc::new(Settings::default()))
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("wsi-tiffslide-plugin-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, b"not a tiff").unwrap();
        path
    }

    #[test]
    fn the_supported_extensions_match_the_python_list() {
        let plugin = plugin();
        for name in ["slide.svs", "slide.tif", "slide.tiff", "slide.ndpi"] {
            assert!(plugin.is_supported(&scratch(name)), "{name}");
        }
        assert!(!plugin.is_supported(&scratch("slide.png")));
        assert!(!plugin.is_supported(&scratch("slide.mrxs")));
        assert!(!plugin.is_supported(Path::new("missing.svs")));
    }

    #[test]
    fn an_ome_tiff_is_declined_so_tifffile_can_have_it() {
        let plugin = plugin();
        assert!(!plugin.is_supported(&scratch("fluorescence.ome.tif")));
        assert!(!plugin.is_supported(&scratch("fluorescence.ome.tiff")));
        // The test is `filename.endswith("ome.tif")`, with no dot, so any name whose stem
        // happens to end in "ome" is declined too. Reproduced; see divergence 34.
        assert!(!plugin.is_supported(&scratch("metabolome.tif")));
        assert!(!plugin.is_supported(&scratch("chromosome.tiff")));
        assert!(plugin.is_supported(&scratch("genomics.tif")));
    }

    #[test]
    fn a_file_that_is_not_a_tiff_is_a_500() {
        let error = match TiffSlideSlide::open(scratch("broken.svs"), Arc::new(Settings::default()))
        {
            Ok(_) => panic!("expected an error"),
            Err(error) => error,
        };
        assert_eq!(error.status, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(error.detail_text().starts_with("TiffFileError"));
    }

    #[test]
    fn the_level_zero_location_is_nudged_to_match_openslide() {
        // An exact downsample needs no adjustment.
        assert_eq!(adapt_level_0_location(100, 200, 4.0), (400, 800));
        // A fractional one can round back to the wrong level coordinate, which is what the
        // adjustment exists to fix.
        let downsample = 4.000121536217793;
        let (x, _) = adapt_level_0_location(1000, 0, downsample);
        assert_eq!((x as f64 / downsample) as i64, 1000);
    }

    #[test]
    fn the_vendor_gate_is_the_one_python_applies() {
        // The predicate `pixel_size` consults, pinned here because it decides which
        // formats this plugin can serve at all.
        assert!(reader::vendor_is_supported(None));
        assert!(reader::vendor_is_supported(Some("aperio")));
        assert!(!reader::vendor_is_supported(Some("leica")));
    }
}
