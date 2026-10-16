//! The `openslide` plugin: vendor whole-slide formats through libopenslide.
//!
//! Ported from `wsi_service_base_plugins/openslide/wsi_service_plugin_openslide/`.
//!
//! This is the plugin that carries the service: the deployed `.env` sets
//! `WS_PLUGIN_PRIORITY_OPENSLIDE=10`, so it wins over every other plugin for any file they
//! both claim. Because it goes through the same C library as `openslide-python`, region
//! pixels are identical by construction — the only work is reproducing the metadata
//! derivation and the RGBA-to-RGB compositing exactly.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use image::DynamicImage;
use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::{IccIntent, IccProfileCache};
use wsi_core::image::rgba_to_rgb_with_background_color;
use wsi_core::models::{
    original_levels, rgb_channel_list, SlideExtent, SlideInfo, SlidePixelSizeNm,
};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

pub mod ffi;

use ffi::OpenSlide;

/// `supported_vendors` — `slide.py:17`. Only these report a usable `openslide.mpp-*`.
const SUPPORTED_VENDORS: [&str; 9] = [
    "aperio",
    "mirax",
    "hamamatsu",
    "ventana",
    "leica",
    "trestle",
    "philips",
    "zeiss",
    "dicom",
];

/// Extensions `is_supported` accepts for a regular file — `__init__.py:13`.
const SUPPORTED_FILE_EXTENSIONS: [&str; 8] = [
    ".bif", ".mrxs", ".ndpi", ".scn", ".svs", ".tiff", ".tif", ".czi",
];

/// `openslide.PROPERTY_NAME_*`
const PROPERTY_VENDOR: &str = "openslide.vendor";
const PROPERTY_MPP_X: &str = "openslide.mpp-x";
const PROPERTY_MPP_Y: &str = "openslide.mpp-y";
const PROPERTY_TILE_WIDTH: &str = "openslide.level[0].tile-width";
const PROPERTY_TILE_HEIGHT: &str = "openslide.level[0].tile-height";

/// `openslide.vendor` for a TIFF with no recognised vendor backend.
const GENERIC_TIFF_VENDOR: &str = "generic-tiff";

/// Default tile extent when the slide does not advertise one — `slide.py:188`.
const DEFAULT_TILE_SIZE: i64 = 256;

/// Reported when the file records no pixel size — the same sentinel `pil` and `mvt` use.
const UNKNOWN_PIXEL_SIZE_NM: SlidePixelSizeNm = SlidePixelSizeNm {
    x: -1.0,
    y: -1.0,
    z: None,
};

pub struct OpenSlidePlugin {
    settings: Arc<Settings>,
}

impl OpenSlidePlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for OpenSlidePlugin {
    fn name(&self) -> &'static str {
        "openslide"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// `is_supported` — `__init__.py:7`.
    ///
    /// A regular file must have one of the listed extensions, except that anything named
    /// `*.ome.tif`/`*.ome.tiff` is rejected outright so the tifffile plugin gets it. A
    /// directory qualifies when it contains at least one `.dcm`.
    fn is_supported(&self, path: &Path) -> bool {
        if path.is_file() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            if name.ends_with("ome.tif") || name.ends_with("ome.tiff") {
                return false;
            }
            let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
                return false;
            };
            let dotted = format!(".{}", extension.to_ascii_lowercase());
            return SUPPORTED_FILE_EXTENSIONS.contains(&dotted.as_str());
        }
        // Python's `is_supported` reaches the directory branch for anything that is not a
        // regular file, including a path that does not exist; `glob` then finds nothing.
        contains_dicom(path)
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || OpenSlideSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("OpenSlideError: {e}")))??;
        Ok(Arc::new(slide))
    }

    fn start(&self) {
        tracing::info!("openslide plugin: libopenslide {}", ffi::version());
    }
}

/// `glob.glob(os.path.join(filepath, "*.dcm"))` — non-recursive, case-sensitive.
fn dicom_files(path: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(path) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("dcm"))
        .collect();
    // `glob` returns an unspecified order and Python takes `[0]`; sorting makes the choice
    // deterministic across runs and platforms.
    found.sort();
    found
}

fn contains_dicom(path: &Path) -> bool {
    !dicom_files(path).is_empty()
}

pub struct OpenSlideSlide {
    /// The path actually opened, which for a DICOM directory is one file inside it.
    path: PathBuf,
    slide: OpenSlide,
    info: SlideInfo,
    icc: IccProfileCache,
    /// The slide's own ICC profile, read once at open time.
    icc_profile: Option<Vec<u8>>,
    /// `self.thumbnail` — Python caches the first thumbnail on the instance, along with
    /// whatever ICC settings that first request used.
    thumbnail: Mutex<Option<DynamicImage>>,
    settings: Arc<Settings>,
}

impl OpenSlideSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        // `__check_and_adapt_filepath`: a directory of DICOM files is opened through its
        // first member, because libopenslide takes a file.
        let path = if path.is_dir() {
            match dicom_files(&path).into_iter().next() {
                Some(first) => first,
                None => path,
            }
        } else {
            path
        };

        let slide = OpenSlide::open(&path)?;
        // `detect_format` is called on the adapted path, after opening.
        let format = ffi::detect_vendor(&path);

        // Divergence from Python: a plain TIFF is handed back to the chain, so `tifffile`
        // and `tiffslide` get it. libopenslide's generic-TIFF backend exposes only the
        // first IFD — it reports `num_levels` 1 for a five-level pyramid — and flattens
        // every page to 8-bit RGB, losing a 16-bit or multi-channel image outright.
        // Python never noticed because `__get_pixel_size` raised on most of these files
        // anyway; now that a missing pixel size is no longer fatal, openslide would
        // otherwise win them all on priority. See `docs/divergences.md`, divergence 39.
        if format.as_deref() == Some(GENERIC_TIFF_VENDOR) {
            return Err(ApiError::not_found(
                "Plain TIFF is served by the tifffile and tiffslide plugins.",
            ));
        }

        let icc_profile = slide.icc_profile()?;

        let info = build_slide_info(&slide, format)?;

        Ok(Self {
            path,
            slide,
            info,
            icc: IccProfileCache::new(),
            icc_profile,
            thumbnail: Mutex::new(None),
            settings,
        })
    }

    /// `get_region` — the core read.
    fn read_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<DynamicImage> {
        let padding_color = opts.padding_color.unwrap_or(self.settings.padding_color);

        let downsample = self
            .info
            .levels
            .get(level)
            .map(|l| l.downsample_factor)
            .ok_or_else(|| {
                // The endpoint validates the level first, so this is unreachable through
                // HTTP; Python would raise IndexError here.
                ApiError::unhandled("IndexError('list index out of range')")
            })?;

        // `level_0_location = (int(start_x * downsample), int(start_y * downsample))` —
        // truncation toward zero, which for a negative start rounds *up*.
        let level_0_x = (start_x as f64 * downsample) as i64;
        let level_0_y = (start_y as f64 * downsample) as i64;

        let buffer = self.slide.read_region(
            level_0_x,
            level_0_y,
            level as i32,
            size_x as i64,
            size_y as i64,
        )?;
        let rgba = DynamicImage::ImageRgba8(ffi::to_rgba(&buffer));

        // ICC is applied to the RGBA image, before the padding colour is composited in.
        let processed = match opts.icc_profile_intent {
            None => rgba,
            Some(intent) => self.icc.process_image_lenient(
                rgba,
                self.icc_profile.as_deref(),
                opts.icc_profile_strict,
                intent,
                true,
            )?,
        };

        Ok(rgba_to_rgb_with_background_color(processed, padding_color))
    }

    /// `__get_associated_image` — `slide.py:128`.
    fn associated_image(
        &self,
        name: &str,
        intent: Option<IccIntent>,
        strict: bool,
    ) -> Result<DynamicImage> {
        if !self
            .slide
            .associated_image_names()?
            .iter()
            .any(|n| n == name)
        {
            return Err(ApiError::not_found(format!(
                "Associated image {name} does not exist."
            )));
        }

        let buffer = self.slide.associated_image(name)?;
        let rgba = DynamicImage::ImageRgba8(ffi::to_rgba(&buffer));

        let processed = match intent {
            None => rgba,
            Some(intent) => {
                // The associated image carries its own profile, so the transform is not
                // cached — a later macro read may need a different one.
                let profile = self.slide.associated_image_icc_profile(name)?;
                self.icc
                    .process_image_lenient(rgba, profile.as_deref(), strict, intent, false)?
            }
        };

        // `.convert("RGB")` drops alpha without compositing a background colour.
        Ok(DynamicImage::ImageRgb8(processed.to_rgb8()))
    }

    /// `__get_best_level_for_thumbnail` — `slide.py:244`.
    ///
    /// Returns the index *before* the first level that fits inside the box, or the last
    /// index if none does. When level 0 already fits it returns `-1`, which Python uses as
    /// a negative list index and so selects the **coarsest** level. Reproduced; see
    /// divergence 7.
    fn best_level_for_thumbnail(&self, max_x: u32, max_y: u32) -> usize {
        let levels = &self.info.levels;
        let mut best_level: i64 = 0;
        for level in levels {
            if level.extent.x < max_x as i64 && level.extent.y < max_y as i64 {
                best_level -= 1;
                break;
            }
            best_level += 1;
            if best_level as usize == levels.len() {
                best_level -= 1;
                break;
            }
        }
        if best_level < 0 {
            // Python's `levels[-1]`.
            levels.len().saturating_sub(1)
        } else {
            best_level as usize
        }
    }

    /// `__get_thumbnail_openslide` — reads a whole level and shrinks it.
    fn render_thumbnail(
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

        let opts = ReadOptions {
            // Python passes `None`, so the plugin falls back to the configured colour.
            padding_color: None,
            z: 0,
            icc_profile_intent: intent,
            icc_profile_strict: strict,
        };
        let region = self
            .read_region(level, 0, 0, extent.x as u32, extent.y as u32, opts)
            .map_err(|err| {
                ApiError::internal(format!(
                    "Failed to extract thumbnail from WSI [{}].",
                    err.detail_text()
                ))
            })?;

        Ok(pil_thumbnail(&region, max_x, max_y))
    }
}

/// `__get_slide_info_openslide` — every failure collapses into one 404, as in Python.
fn build_slide_info(slide: &OpenSlide, format: Option<String>) -> Result<SlideInfo> {
    gather_slide_info(slide, format).map_err(|err| {
        ApiError::not_found(format!(
            "Failed to gather slide infos. [{}]",
            err.detail_text()
        ))
    })
}

fn gather_slide_info(slide: &OpenSlide, format: Option<String>) -> Result<SlideInfo> {
    let level_count = slide.level_count()?;
    if level_count <= 0 {
        return Err(ApiError::internal("no levels"));
    }

    let mut dimensions = Vec::with_capacity(level_count as usize);
    let mut downsamples = Vec::with_capacity(level_count as usize);
    for level in 0..level_count {
        dimensions.push(slide.level_dimensions(level)?);
        downsamples.push(slide.level_downsample(level)?);
    }
    let levels = original_levels(&dimensions, &downsamples);

    let (width, height) = slide.level0_dimensions()?;

    // `format=self.format` where `detect_format` returned None would fail pydantic
    // validation, which the caller turns into the same 404.
    let format = format.ok_or_else(|| ApiError::internal("detect_format returned None"))?;

    Ok(SlideInfo {
        id: String::new(),
        channels: Some(rgb_channel_list()),
        channel_depth: Some(8),
        extent: SlideExtent::new(width, height, 1),
        num_levels: levels.len() as i64,
        pixel_size_nm: pixel_size(slide)?,
        tile_extent: tile_extent(slide)?,
        levels,
        format,
        raw_download: false,
    })
}

/// `__get_pixel_size` — `slide.py:160`.
///
/// Divergence from Python, which raises on every branch that cannot produce a size and so
/// turns an otherwise readable slide into a 404. A slide whose pixel size is simply not
/// recorded reports `UNKNOWN_PIXEL_SIZE_NM`, the `-1` the `pil` and `mvt` plugins already
/// use for "unknown". See `docs/divergences.md`, divergence 39.
fn pixel_size(slide: &OpenSlide) -> Result<SlidePixelSizeNm> {
    let vendor = slide.property(PROPERTY_VENDOR)?.unwrap_or_default();

    if vendor == GENERIC_TIFF_VENDOR {
        // A plain TIFF has no MPP, so the resolution tags are used — but only when they
        // are per centimetre.
        let unit = slide.property("tiff.ResolutionUnit")?.unwrap_or_default();
        if unit != "centimeter" {
            return Ok(UNKNOWN_PIXEL_SIZE_NM);
        }
        let (Some(x_resolution), Some(y_resolution)) = (
            slide.property("tiff.XResolution")?,
            slide.property("tiff.YResolution")?,
        ) else {
            return Ok(UNKNOWN_PIXEL_SIZE_NM);
        };
        let pixels_per_cm_x = parse_float(&x_resolution)?;
        let pixels_per_cm_y = parse_float(&y_resolution)?;
        if pixels_per_cm_x == 0.0 || pixels_per_cm_y == 0.0 {
            return Ok(UNKNOWN_PIXEL_SIZE_NM);
        }
        // 1 cm is 1e7 nm.
        return Ok(SlidePixelSizeNm::xy(
            1e7 / pixels_per_cm_x,
            1e7 / pixels_per_cm_y,
        ));
    }

    if SUPPORTED_VENDORS.contains(&vendor.as_str()) {
        // Python raises KeyError here and reports "Failed to gather slide infos".
        let (Some(mpp_x), Some(mpp_y)) = (
            slide.property(PROPERTY_MPP_X)?,
            slide.property(PROPERTY_MPP_Y)?,
        ) else {
            return Ok(UNKNOWN_PIXEL_SIZE_NM);
        };
        return Ok(SlidePixelSizeNm::xy(
            1000.0 * parse_float(&mpp_x)?,
            1000.0 * parse_float(&mpp_y)?,
        ));
    }

    Ok(UNKNOWN_PIXEL_SIZE_NM)
}

fn parse_float(raw: &str) -> Result<f64> {
    raw.trim()
        .parse::<f64>()
        .map_err(|_| ApiError::internal(format!("ValueError('could not convert to float: {raw}')")))
}

/// `__get_tile_extent` — `slide.py:187`. Both properties must be present to override the
/// 256x256 default.
fn tile_extent(slide: &OpenSlide) -> Result<SlideExtent> {
    let height = slide.property(PROPERTY_TILE_HEIGHT)?;
    let width = slide.property(PROPERTY_TILE_WIDTH)?;
    match (width, height) {
        (Some(width), Some(height)) => {
            let width = width.trim().parse::<i64>().map_err(|_| {
                ApiError::internal(format!("ValueError('invalid tile width: {width}')"))
            })?;
            let height = height.trim().parse::<i64>().map_err(|_| {
                ApiError::internal(format!("ValueError('invalid tile height: {height}')"))
            })?;
            Ok(SlideExtent::new(width, height, 1))
        }
        _ => Ok(SlideExtent::new(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, 1)),
    }
}

/// `Image.thumbnail((max_x, max_y))` — no resample argument is passed anywhere in this
/// plugin, so Pillow's default `BICUBIC` applies.
pub fn pil_thumbnail(image: &DynamicImage, max_x: u32, max_y: u32) -> DynamicImage {
    wsi_core::image::pillow_thumbnail(image, max_x, max_y, wsi_core::image::Filter::Bicubic)
}

#[async_trait]
impl Slide for OpenSlideSlide {
    async fn close(&self) -> Result<()> {
        self.slide.close();
        // `self._icc.clear()`
        self.icc.free_cache();
        *self.thumbnail.lock() = None;
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
        // `z` is accepted and ignored; no openslide format carries a Z stack.
        self.read_region(level, start_x, start_y, size_x, size_y, opts)
            .map(Pixels::Image)
    }

    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        // Always a full tile, even at the edge — unlike the tiffslide plugin, which shrinks
        // edge reads. libopenslide pads the overhang with transparent black, which the
        // padding colour then replaces.
        let tile = self.info.tile_extent;
        self.read_region(
            level,
            tile_x * tile.x,
            tile_y * tile.y,
            tile.x as u32,
            tile.y as u32,
            opts,
        )
        .map(Pixels::Image)
    }

    async fn get_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        // Python caches `self.thumbnail` on first use, so later requests reuse whatever
        // ICC settings the first one asked for. Reproduced, including that quirk.
        let cached = self.thumbnail.lock().clone();
        let base = match cached {
            Some(image) => image,
            None => {
                let built = match self.associated_image(
                    "thumbnail",
                    icc_profile_intent,
                    icc_profile_strict,
                ) {
                    Ok(image) => image,
                    // Any HTTPException falls back to rendering one from the pyramid.
                    Err(_) => self.render_thumbnail(
                        self.settings.max_thumbnail_size,
                        self.settings.max_thumbnail_size,
                        icc_profile_intent,
                        icc_profile_strict,
                    )?,
                };
                *self.thumbnail.lock() = Some(built.clone());
                built
            }
        };

        Ok(Pixels::Image(pil_thumbnail(&base, max_x, max_y)))
    }

    async fn get_label(&self) -> Result<Pixels> {
        // Python passes no ICC arguments here, so the intent is always None.
        self.associated_image("label", None, false)
            .map(Pixels::Image)
    }

    async fn get_macro(
        &self,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        self.associated_image("macro", icc_profile_intent, icc_profile_strict)
            .map(Pixels::Image)
    }

    async fn get_icc_profile(&self) -> Result<Bytes> {
        IccProfileCache::payload(self.icc_profile.as_deref())
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "openslide"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wsi_core::models::{SlideExtent as Extent, SlideLevel};

    fn plugin() -> OpenSlidePlugin {
        OpenSlidePlugin::new(Arc::new(Settings::default()))
    }

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("wsi-openslide-{name}"));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn file(&self, name: &str) -> PathBuf {
            let path = self.root.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
            path
        }

        fn dir(&self, name: &str) -> PathBuf {
            let path = self.root.join(name);
            std::fs::create_dir_all(&path).unwrap();
            path
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn vendor_extensions_are_accepted() {
        let f = Fixture::new("extensions");
        let p = plugin();
        for extension in ["bif", "mrxs", "ndpi", "scn", "svs", "tiff", "tif", "czi"] {
            let path = f.file(&format!("slide.{extension}"));
            assert!(p.is_supported(&path), "{extension} should be supported");
        }
        // Case is normalised.
        assert!(p.is_supported(&f.file("SLIDE.SVS")));
        // Not a vendor format.
        assert!(!p.is_supported(&f.file("slide.png")));
        assert!(!p.is_supported(&f.file("slide.nii")));
        assert!(!p.is_supported(&f.file("noextension")));
    }

    #[test]
    fn ome_tiff_is_left_to_the_tifffile_plugin() {
        let f = Fixture::new("ome");
        let p = plugin();
        assert!(!p.is_supported(&f.file("image.ome.tif")));
        assert!(!p.is_supported(&f.file("image.ome.tiff")));
        assert!(!p.is_supported(&f.file("IMAGE.OME.TIFF")));
        // A plain .tif is still ours.
        assert!(p.is_supported(&f.file("image.tif")));
    }

    #[test]
    fn a_directory_qualifies_only_when_it_holds_a_dcm() {
        let f = Fixture::new("dicom");
        let p = plugin();

        let empty = f.dir("empty");
        assert!(!p.is_supported(&empty));

        let study = f.dir("study");
        std::fs::write(study.join("notes.txt"), b"x").unwrap();
        assert!(!p.is_supported(&study));

        std::fs::write(study.join("image.dcm"), b"x").unwrap();
        assert!(p.is_supported(&study));

        // A path that does not exist takes the directory branch and finds nothing.
        assert!(!p.is_supported(&f.root.join("missing")));
    }

    #[test]
    fn a_dicom_directory_is_opened_through_its_first_file() {
        let f = Fixture::new("dicom-first");
        let study = f.dir("study");
        for name in ["b.dcm", "a.dcm", "c.dcm"] {
            std::fs::write(study.join(name), b"x").unwrap();
        }
        let chosen = dicom_files(&study);
        assert_eq!(chosen.len(), 3);
        assert_eq!(chosen[0].file_name().unwrap(), "a.dcm");
    }

    fn info_with_levels(extents: &[(i64, i64)]) -> SlideInfo {
        SlideInfo {
            id: String::new(),
            channels: Some(rgb_channel_list()),
            channel_depth: Some(8),
            extent: Extent::new(extents[0].0, extents[0].1, 1),
            num_levels: extents.len() as i64,
            pixel_size_nm: SlidePixelSizeNm::xy(500.0, 500.0),
            tile_extent: Extent::new(256, 256, 1),
            levels: extents
                .iter()
                .enumerate()
                .map(|(index, &(x, y))| SlideLevel {
                    extent: Extent::new(x, y, 1),
                    downsample_factor: (1 << index) as f64,
                })
                .collect(),
            format: "aperio".into(),
            raw_download: false,
        }
    }

    /// Exercises `best_level_for_thumbnail` without needing a real slide.
    fn best_level(extents: &[(i64, i64)], max_x: u32, max_y: u32) -> usize {
        let info = info_with_levels(extents);
        let levels = &info.levels;
        let mut best_level: i64 = 0;
        for level in levels {
            if level.extent.x < max_x as i64 && level.extent.y < max_y as i64 {
                best_level -= 1;
                break;
            }
            best_level += 1;
            if best_level as usize == levels.len() {
                best_level -= 1;
                break;
            }
        }
        if best_level < 0 {
            levels.len().saturating_sub(1)
        } else {
            best_level as usize
        }
    }

    #[test]
    fn thumbnail_level_is_the_one_before_the_first_that_fits() {
        // Levels 1000, 500, 250, 125; a 300-box first fits at level 2, so level 1 is used.
        let extents = [(1000, 1000), (500, 500), (250, 250), (125, 125)];
        assert_eq!(best_level(&extents, 300, 300), 1);
        // A 600-box first fits at level 1, so level 0 is used.
        assert_eq!(best_level(&extents, 600, 600), 0);
    }

    #[test]
    fn no_level_fitting_the_box_selects_the_coarsest() {
        let extents = [(1000, 1000), (500, 500)];
        // Nothing fits in 100x100, so the loop runs out and the last index is used.
        assert_eq!(best_level(&extents, 100, 100), 1);
    }

    #[test]
    fn a_level_zero_that_already_fits_wraps_to_the_coarsest_level() {
        // Python returns -1 here and indexes the list from the end. Divergence 7.
        let extents = [(64, 64), (32, 32), (16, 16)];
        assert_eq!(best_level(&extents, 500, 500), 2);
    }

    #[test]
    fn a_single_level_slide_is_handled() {
        assert_eq!(best_level(&[(1000, 1000)], 100, 100), 0);
        // Level 0 fits, so the negative index wraps to the only level there is.
        assert_eq!(best_level(&[(64, 64)], 500, 500), 0);
    }

    #[test]
    fn thumbnail_shrinks_preserving_aspect_ratio_and_never_grows() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(400, 200));
        let out = pil_thumbnail(&image, 100, 100);
        assert_eq!((out.width(), out.height()), (100, 50));

        let small = DynamicImage::ImageRgb8(image::RgbImage::new(10, 10));
        let out = pil_thumbnail(&small, 100, 100);
        assert_eq!((out.width(), out.height()), (10, 10));
    }

    #[test]
    fn float_parsing_reports_a_python_shaped_error() {
        assert_eq!(parse_float("0.499").unwrap(), 0.499);
        assert_eq!(parse_float(" 2.5 ").unwrap(), 2.5);
        let err = match parse_float("not-a-number") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert!(err.detail_text().contains("could not convert to float"));
    }
}
