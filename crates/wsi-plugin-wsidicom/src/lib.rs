//! The `wsidicom` plugin: DICOM VL Whole Slide Microscopy Image directories.
//!
//! Ported from `wsi_service_base_plugins/wsidicom/wsi_service_plugin_wsidicom/`.
//!
//! `priority = 0`, and `openslide` claims the same directories at 10 through the same
//! libopenslide the Python service links — so in a default build DICOM is already served
//! pixel-exactly and this plugin answers an explicit `plugin=wsidicom`, or everything in a
//! build without libopenslide. The same position `tiffslide` is in.
//!
//! Four behaviours of the Python plugin are worth stating up front, because they are not
//! what the other plugins do:
//!
//! - **`is_supported` matches a directory**, never a file: one containing at least one
//!   `*.dcm`. No other plugin here claims a directory.
//! - **A region outside the level is not padded.** `WsiDicom.read_region` raises
//!   `WsiDicomOutOfBoundsError`, the plugin does not catch it, and FastAPI does not
//!   recognise it — so the response is Starlette's plain-text 500. Divergence 37.
//! - **`get_tile` catches everything.** Its `except Exception` answers
//!   `500 Failed to extract tile from WSI.` for any failure, so no tile error is ever
//!   specific — not an out-of-range tile, not a corrupt frame.
//! - **The thumbnail is cached on first use**, so the `icc_profile_intent` of whichever
//!   request arrives first fixes the colours for every later one. The same `hasattr`
//!   caching quirk `tiffslide` and `openslide` have.

pub mod dicom;
pub mod reader;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use image::{DynamicImage, RgbImage};
use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::{IccIntent, IccProfileCache};
use wsi_core::image::{pillow_thumbnail, rgba_to_rgb_with_background_color, Filter};
use wsi_core::models::{
    original_levels, rgb_channel_list, SlideExtent, SlideInfo, SlidePixelSizeNm,
};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

use reader::WsiDicomReader;
use wsi_plugin_tifffile::tiff::page::Raster;

pub struct WsiDicomPlugin {
    settings: Arc<Settings>,
}

impl WsiDicomPlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for WsiDicomPlugin {
    fn name(&self) -> &'static str {
        "wsidicom"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// `priority = 0`
    fn default_priority(&self) -> i64 {
        0
    }

    /// `is_supported`: a *directory* holding at least one `*.dcm`.
    ///
    /// Python globs for `*.dcm` and takes any non-empty result, so an unreadable directory
    /// is simply unsupported rather than an error.
    fn is_supported(&self, path: &Path) -> bool {
        if !path.is_dir() {
            return false;
        }
        dicom::dicom_files(path).is_ok_and(|files| !files.is_empty())
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || WsiDicomSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

pub struct WsiDicomSlide {
    path: PathBuf,
    reader: Mutex<Option<WsiDicomReader>>,
    info: SlideInfo,
    icc_profile: Option<Vec<u8>>,
    icc: IccProfileCache,
    settings: Arc<Settings>,
    /// `self.thumbnail`, computed once and then reused — see the module note.
    thumbnail: Mutex<Option<DynamicImage>>,
}

impl WsiDicomSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        // `WsiDicom.open` failing is a 500 `WsiDicomNotFoundError: ...` in Python, and
        // everything `__get_slide_info_dicom` touches is wrapped in a 404 instead. The
        // open is attributed to the former.
        let reader = WsiDicomReader::open(&path)
            .map_err(|e| ApiError::internal(format!("WsiDicomNotFoundError: {e}")))?;
        let info = slide_info(&reader)?;
        let icc_profile = reader.icc_profile().map(<[u8]>::to_vec);
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

    fn with_reader<T>(&self, f: impl FnOnce(&WsiDicomReader) -> Result<T>) -> Result<T> {
        let guard = self.reader.lock();
        match guard.as_ref() {
            Some(reader) => f(reader),
            // After `close`, Python's `self.dicom_slide` is a closed `WsiDicom` whose
            // reads raise. The shape of the message is what the batch endpoints record.
            None => Err(ApiError::unhandled(
                "AttributeError(\"'NoneType' object has no attribute 'read_region'\")",
            )),
        }
    }

    /// The DICOM level index — the *exponent* — for a list index.
    ///
    /// `self.dicom_slide.levels[level].level`. An index past the end is Python's
    /// `IndexError`, which FastAPI does not handle.
    fn exponent_for(&self, level: usize) -> Result<i32> {
        self.with_reader(|reader| {
            reader
                .levels
                .get(level)
                .map(|l| l.exponent)
                .ok_or_else(|| ApiError::unhandled("IndexError('list index out of range')"))
        })
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

    /// `__get_thumbnail_dicom` — `read_thumbnail((max, max))`, wrapped so that any failure
    /// is `500 Failed to extract thumbnail from WSI.`
    fn dicom_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        intent: Option<IccIntent>,
        strict: bool,
    ) -> Result<DynamicImage> {
        let source = self
            .with_reader(|reader| {
                reader
                    .thumbnail_source((max_x, max_y))
                    .map_err(|e| ApiError::internal(e.0))
            })
            .map_err(|_| ApiError::internal("Failed to extract thumbnail from WSI."))?;
        let image = raster_to_image(&source)?;
        // `read_thumbnail` shrinks to the box itself, with Pillow's default filter.
        let image = pillow_thumbnail(&image, max_x, max_y, Filter::Bicubic);
        // `cache=False`: a thumbnail's profile need not be the region profile.
        self.apply_icc(image, intent, strict, false)
    }
}

#[async_trait]
impl Slide for WsiDicomSlide {
    async fn close(&self) -> Result<()> {
        *self.reader.lock() = None;
        self.icc.free_cache();
        Ok(())
    }

    async fn get_info(&self) -> Result<SlideInfo> {
        Ok(self.info.clone())
    }

    /// `get_region` — `start_x`/`start_y` are in **this level's** pixels.
    ///
    /// Unlike the `tiffslide` and `openslide` plugins, nothing is scaled: `read_region`
    /// takes the location in the level's own coordinates, and no `__adapt_level_0_location`
    /// nudge is applied. A region reaching outside the level is not padded — see
    /// divergence 37.
    async fn get_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let exponent = self.exponent_for(level)?;
        let padding_color = opts.padding_color.unwrap_or(self.settings.padding_color);

        // A negative origin cannot be expressed in `read_region`'s unsigned location, and
        // `valid_pixels` would reject it anyway; the error is the one `wsidicom` raises.
        if start_x < 0 || start_y < 0 {
            return Err(out_of_bounds(format!(
                "Region origin ({start_x}, {start_y})"
            )));
        }

        let raster = self.with_reader(|reader| {
            reader
                .read_region((start_x as u32, start_y as u32), exponent, (size_x, size_y))
                .map_err(|e| out_of_bounds(e.0))
        })?;

        let image = raster_to_image(&raster)?;
        let image = self.apply_icc(
            image,
            opts.icc_profile_intent,
            opts.icc_profile_strict,
            true,
        )?;
        // A no-op for the RGB image `read_region` returns, as in Python.
        Ok(Pixels::Image(rgba_to_rgb_with_background_color(
            image,
            padding_color,
        )))
    }

    /// `get_tile` — a tile *index*, and every failure is the same 500.
    ///
    /// `read_tile` takes the tile coordinate directly, so there is no size arithmetic and
    /// no padding: a tile is whatever the frame holds, which `TILED_FULL` pads to the full
    /// tile size at the edges of the image.
    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let failed = || ApiError::internal("Failed to extract tile from WSI.");

        // The `except Exception` covers the level lookup too, so an out-of-range level is
        // this 500 rather than the `IndexError` the region path gives.
        let exponent = self.exponent_for(level).map_err(|_| failed())?;
        if tile_x < 0 || tile_y < 0 {
            return Err(failed());
        }

        let raster = self
            .with_reader(|reader| {
                reader
                    .read_tile(exponent, (tile_x as u32, tile_y as u32))
                    .map_err(|e| ApiError::internal(e.0))
            })
            .map_err(|_| failed())?;
        let image = raster_to_image(&raster).map_err(|_| failed())?;

        // The ICC step is the one failure that keeps its own status. Python catches
        // `ICCProfileError` *before* the bare `except Exception`, so a strict request
        // without a profile is still the 412 every other endpoint gives rather than this
        // method's catch-all 500. Nothing else `apply_icc` can return is an error, so the
        // distinction is which step failed rather than what the status happens to be.
        let image = self.apply_icc(
            image,
            opts.icc_profile_intent,
            opts.icc_profile_strict,
            true,
        )?;

        let padding_color = opts.padding_color.unwrap_or(self.settings.padding_color);
        Ok(Pixels::Image(rgba_to_rgb_with_background_color(
            image,
            padding_color,
        )))
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
                let image = self.dicom_thumbnail(
                    self.settings.max_thumbnail_size,
                    self.settings.max_thumbnail_size,
                    icc_profile_intent,
                    icc_profile_strict,
                )?;
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
        let raster = self.with_reader(|reader| {
            reader
                .read_label()
                .map_err(|_| ApiError::not_found("Label image does not exist."))
        })?;
        Ok(Pixels::Image(raster_to_image(&raster)?))
    }

    async fn get_macro(
        &self,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        let raster = self.with_reader(|reader| {
            reader
                .read_overview()
                .map_err(|_| ApiError::not_found("Macro image does not exist."))
        })?;
        let image = raster_to_image(&raster)?;
        // `cache=False`, as in Python.
        Ok(Pixels::Image(self.apply_icc(
            image,
            icc_profile_intent,
            icc_profile_strict,
            false,
        )?))
    }

    async fn get_icc_profile(&self) -> Result<Bytes> {
        IccProfileCache::payload(self.icc_profile.as_deref())
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "wsidicom"
    }
}

/// `WsiDicomOutOfBoundsError`, which FastAPI does not recognise.
///
/// The plugin does not catch it, `wsi_service` registers no handler for it, and Starlette
/// answers an unhandled exception with a plain-text `Internal Server Error`. The `repr` is
/// kept because the batch endpoints write it into the archive.
fn out_of_bounds(what: impl Into<String>) -> ApiError {
    ApiError::unhandled(format!(
        "WsiDicomOutOfBoundsError('{} is outside of the image', 'level size')",
        what.into()
    ))
}

/// A decoded raster as an RGB image.
fn raster_to_image(raster: &Raster) -> Result<DynamicImage> {
    let (width, height) = (raster.width as u32, raster.height as u32);
    match raster.samples {
        3 => RgbImage::from_raw(width, height, raster.bytes.clone())
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| ApiError::internal("decoded frame does not match its geometry")),
        1 => image::GrayImage::from_raw(width, height, raster.bytes.clone())
            .map(|gray| DynamicImage::ImageLuma8(gray).to_rgb8())
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| ApiError::internal("decoded frame does not match its geometry")),
        other => Err(ApiError::internal(format!(
            "a frame with {other} samples per pixel is not supported"
        ))),
    }
}

/// `__get_slide_info_dicom`, whose every failure is a 404.
fn slide_info(reader: &WsiDicomReader) -> Result<SlideInfo> {
    let failed =
        |what: &str| ApiError::not_found(format!("Failed to gather slide infos. [{what}]"));

    let base = reader.levels.first().ok_or_else(|| failed("no levels"))?;

    let level_dimensions: Vec<(i64, i64)> = reader
        .levels
        .iter()
        .map(|level| (i64::from(level.width), i64::from(level.height)))
        .collect();
    // `level_downsamples.append(2 ** level.level)` — the exponent, not a size ratio.
    let level_downsamples: Vec<f64> = reader
        .levels
        .iter()
        .map(|level| 2f64.powi(level.exponent))
        .collect();

    let levels = original_levels(&level_dimensions, &level_downsamples);

    // `pixel_size_nm = 1000 * mpp`, and `mpp = pixel_spacing * 1000`, so this is the
    // spacing in millimetres times a million. A level without one is
    // `WsiDicomNoResolutionError`, which `__get_slide_info_dicom` catches as its 404.
    let (mpp_x, mpp_y) = base
        .mpp
        .ok_or_else(|| failed("WsiDicomNoResolutionError()"))?;

    Ok(SlideInfo {
        id: String::new(),
        channels: Some(rgb_channel_list()),
        channel_depth: Some(8),
        extent: SlideExtent::new(i64::from(base.width), i64::from(base.height), 1),
        num_levels: levels.len() as i64,
        pixel_size_nm: SlidePixelSizeNm::xy(1000.0 * mpp_x, 1000.0 * mpp_y),
        tile_extent: SlideExtent::new(i64::from(base.tile_width), i64::from(base.tile_height), 1),
        levels,
        // Python sets this literally, which is why a DICOM slide reports a format at all —
        // unlike the `mvt` plugin, which omits the field. See divergence 26.
        format: "dicom".into(),
        raw_download: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Arc<Settings> {
        Arc::new(Settings::default())
    }

    #[test]
    fn the_plugin_reports_the_python_name_priority_and_version() {
        let plugin = WsiDicomPlugin::new(settings());
        assert_eq!(plugin.name(), "wsidicom");
        assert_eq!(plugin.default_priority(), 0);
        // `/alive` reports this and the parity harness compares it.
        assert_eq!(plugin.version(), "0.4.4");
    }

    #[test]
    fn only_a_directory_holding_dcm_files_is_supported() {
        let plugin = WsiDicomPlugin::new(settings());
        let base = std::env::temp_dir().join(format!(
            "wsi-dicom-support-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let with_dcm = base.join("with");
        std::fs::create_dir_all(&with_dcm).unwrap();
        std::fs::write(with_dcm.join("a.dcm"), b"not really").unwrap();
        assert!(plugin.is_supported(&with_dcm));

        // Case does not matter, matching the glob's behaviour on a case-insensitive host.
        let upper = base.join("upper");
        std::fs::create_dir_all(&upper).unwrap();
        std::fs::write(upper.join("a.DCM"), b"not really").unwrap();
        assert!(plugin.is_supported(&upper));

        let empty = base.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(!plugin.is_supported(&empty));

        let other = base.join("other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("a.svs"), b"not really").unwrap();
        assert!(!plugin.is_supported(&other));

        // A *file* is never supported, not even one named `.dcm` — Python globs inside a
        // directory, so a lone instance is not a slide.
        assert!(!plugin.is_supported(&with_dcm.join("a.dcm")));

        let _ = std::fs::remove_dir_all(&base);
    }
}
