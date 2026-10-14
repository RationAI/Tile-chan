//! The `tifffile` plugin: multichannel, Z-stacked and OME TIFFs.
//!
//! Ported from `wsi_service_base_plugins/tifffile/wsi_service_plugin_tifffile/`.
//!
//! This is the plugin that serves channel *stacks* rather than pictures. A read returns a
//! `(channels, height, width)` array which the response pipeline then windows and maps to
//! RGB — except on the plain-RGB fast path, where a page's three chunky samples become a
//! Pillow image instead, matching what `tiffslide` and `openslide` return for the same
//! slide.
//!
//! [`tiff`] is the low-level reader underneath: `tifffile`'s `series[0]`, its level
//! pyramid, and per-chunk decoding. [`layout`] works out how channels and Z planes are
//! spread over pages, [`region`] assembles a read out of them, and [`thumb`] carries the
//! one resize in this service that goes through scikit-image rather than Pillow.
//!
//! Two of the Python plugin's behaviours are defects that clients can see, so they are
//! reproduced rather than fixed — both stem from `page.tags.get("ICCProfile")` returning
//! a `TiffTag` object rather than the profile bytes:
//!
//! - `get_icc_profile` hands that object to the response, which cannot serialise it, so
//!   the endpoint is a 500 whether or not the slide has a profile;
//! - an `icc_profile_intent` on a region is silently ignored, and with
//!   `icc_profile_strict` it is a 500.
//!
//! See `docs/divergences.md`, divergences 30 and 31.

pub mod imagej;
pub mod layout;
pub mod ome;
pub mod region;
pub mod thumb;
pub mod tiff;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::IccIntent;
use wsi_core::image::{planar_to_image, PilMode};
use wsi_core::models::{
    original_levels, rgb_channel_list, SlideChannel, SlideColor, SlideExtent, SlideInfo,
    SlideLevel, SlidePixelSizeNm,
};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

use layout::{has_nontrivial_axis, has_supported_axis_layout, LayoutInfo, RGB_PHOTOMETRICS};
use tiff::page::Page;
use tiff::TiffFile;

/// `_OME_SUFFIXES` — an OME file is claimed on its name alone.
const OME_SUFFIXES: [&str; 5] = [".ome.tif", ".ome.tiff", ".ome.tf2", ".ome.tf8", ".ome.btf"];
/// `_TIFF_SUFFIXES` — anything else has to be inspected.
const TIFF_SUFFIXES: [&str; 5] = [".tif", ".tiff", ".tf2", ".tf8", ".btf"];

const UNTILED_TILE_MAX_DIMENSION: i64 = 2048;
const UNTILED_TILE_SIZE: i64 = 512;

/// `_DEFAULT_PALETTE` — cycled through for channels with no colour of their own.
const DEFAULT_PALETTE: [(i32, i32, i32); 7] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 255, 255),
];

pub struct TiffFilePlugin {
    settings: Arc<Settings>,
}

impl TiffFilePlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for TiffFilePlugin {
    fn name(&self) -> &'static str {
        "tifffile"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// `priority = 2` — above `tiffslide`, below `openslide`.
    fn default_priority(&self) -> i64 {
        2
    }

    /// `is_supported`.
    ///
    /// The interesting case is the one that says *no*: a plain RGB(A) page with no
    /// non-trivial Z axis belongs to `tiffslide`/`openslide`, which expose it as a
    /// picture. An RGB *Z-stack* is different — those two cannot address its Z planes —
    /// so it stays here.
    fn is_supported(&self, path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let name = name.to_ascii_lowercase();
        if OME_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
            return true;
        }
        if !TIFF_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
            return false;
        }

        // Python wraps the whole inspection in `except Exception: return False`.
        let Ok(file) = TiffFile::open(path) else {
            return false;
        };
        if file.is_ome() {
            return true;
        }
        let keyframe = file.series.keyframe();
        let samples = keyframe.samples_per_pixel.max(1);
        if RGB_PHOTOMETRICS.contains(&keyframe.photometric)
            && (samples == 3 || samples == 4)
            && !has_nontrivial_axis(&file.series, 'Z')
        {
            return false;
        }
        if samples > 1 {
            return true;
        }
        has_supported_axis_layout(&file.series, samples)
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || TiffFileSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

pub struct TiffFileSlide {
    path: PathBuf,
    /// `None` once closed.
    file: Mutex<Option<TiffFile>>,
    info: SlideInfo,
    layout: LayoutInfo,
    settings: Arc<Settings>,
    /// Whether the first page carries an `ICCProfile` tag, which decides *which* 500 the
    /// profile endpoint produces.
    has_icc_profile: bool,
}

impl TiffFileSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        let file = TiffFile::open(&path)
            .map_err(|e| ApiError::not_found(format!("Failed to load tiff file. [{e}]")))?;

        let (info, layout) = if file.is_ome() {
            // `xml.fromstring(self.tif_slide.ome_metadata)` failing is a 400; here the
            // parse already happened, so an absent `Pixels` element is the same failure.
            if file.ome_pixels().is_none() {
                return Err(ApiError::bad_request(
                    "Could not obtain ome metadata (no Pixels element)",
                ));
            }
            let layout = layout::inspect(&file.series)?;
            (slide_info_ome(&file, &layout)?, layout)
        } else {
            let layout = layout::inspect(&file.series)?;
            (slide_info_generic(&file, &layout)?, layout)
        };

        let has_icc_profile = file.series.keyframe().icc_profile().is_some();
        Ok(Self {
            path,
            file: Mutex::new(Some(file)),
            info,
            layout,
            settings,
            has_icc_profile,
        })
    }

    fn with_file<T>(&self, f: impl FnOnce(&mut TiffFile) -> Result<T>) -> Result<T> {
        let mut guard = self.file.lock();
        match guard.as_mut() {
            Some(file) => f(file),
            None => Err(ApiError::unhandled(
                "AttributeError(\"'NoneType' object has no attribute 'series'\")",
            )),
        }
    }

    /// `__get_tif_level_for_slide_level` — match a TIFF level by its extent.
    fn tif_level_for(&self, file: &TiffFile, level: &SlideLevel) -> Result<usize> {
        for (index, tif_level) in file.series.levels.iter().enumerate() {
            let keyframe = tif_level.keyframe();
            if keyframe.image_length as i64 == level.extent.y
                && keyframe.image_width as i64 == level.extent.x
            {
                return Ok(index);
            }
        }
        // Python returns `None` and then dereferences it.
        Err(ApiError::unhandled(
            "AttributeError(\"'NoneType' object has no attribute 'pages'\")",
        ))
    }

    /// The shared body of `get_region`, minus the ICC step.
    #[allow(clippy::too_many_arguments)]
    fn read_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        padding_color: (u8, u8, u8),
        z: u32,
    ) -> Result<Pixels> {
        let Some(level_slide) = self.info.levels.get(level).cloned() else {
            return Err(ApiError::unhandled("IndexError('list index out of range')"));
        };
        let channel_depth = self.info.channel_depth.unwrap_or(8);
        let layout = self.layout.clone();

        self.with_file(|file| {
            let tif_level = self.tif_level_for(file, &level_slide)?;
            let data = region::assemble(
                file,
                tif_level,
                &layout,
                channel_depth,
                start_x,
                start_y,
                size_x as usize,
                size_y as usize,
                padding_color,
                z as usize,
            )?;
            if layout.is_rgb {
                // Plain RGB slides skip the channel-stack path entirely.
                Ok(Pixels::Image(planar_to_image(
                    &data,
                    None,
                    None,
                    PilMode::Rgb,
                )?))
            } else {
                Ok(Pixels::Planar(data))
            }
        })
    }

    /// The ICC step of `get_region`, reproduced including its failure.
    ///
    /// Which failure depends on whether the page carries the tag at all, because
    /// `tags.get("ICCProfile")` returns `None` when it does not and a `TiffTag` when it
    /// does — and `process_pil_image` branches on that value's truthiness:
    ///
    /// * no tag: the `elif strict` branch raises `412 Error: ICC Profile not available.`,
    ///   the same as any other plugin without a profile;
    /// * a tag: the object is handed to `ImageCms.buildTransform`, which raises, so a
    ///   strict request is a `500`.
    ///
    /// Either way no colour transform is ever applied — see divergence 31.
    fn apply_icc(&self, pixels: Pixels, intent: Option<IccIntent>, strict: bool) -> Result<Pixels> {
        if intent.is_none() || !strict {
            return Ok(pixels);
        }
        Err(strict_icc_failure(self.has_icc_profile))
    }
}

/// Which error a strict colour-managed request gets, given whether the page carries the
/// `ICCProfile` tag.
///
/// Both outcomes come from `icc_profile.py::process_pil_image` branching on the truthiness
/// of `tags.get("ICCProfile")`: `None` takes the `elif strict` branch and is a 412, while a
/// `TiffTag` is handed to `ImageCms.buildTransform` and raises, which the `except` turns
/// into a 500.
fn strict_icc_failure(has_icc_profile: bool) -> ApiError {
    if has_icc_profile {
        ApiError::internal("Error: cannot build transform from a tifffile TiffTag")
    } else {
        ApiError::precondition_failed("Error: ICC Profile not available.")
    }
}

#[async_trait]
impl Slide for TiffFileSlide {
    async fn close(&self) -> Result<()> {
        *self.file.lock() = None;
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
        let pixels = self.read_region(
            level,
            start_x,
            start_y,
            size_x,
            size_y,
            padding_color,
            opts.z,
        )?;
        self.apply_icc(pixels, opts.icc_profile_intent, opts.icc_profile_strict)
    }

    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let tile = self.info.tile_extent;
        self.get_region(
            level,
            tile_x * tile.x,
            tile_y * tile.y,
            tile.x as u32,
            tile.y as u32,
            opts,
        )
        .await
    }

    /// `get_thumbnail`.
    ///
    /// The level choice is the *first* level smaller than the box in either axis, which is
    /// not the same as the closest fit; then the box is narrowed so the limiting axis keeps
    /// its requested value exactly.
    async fn get_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        let mut thumb_level = self.info.levels.len().saturating_sub(1);
        for (index, level) in self.info.levels.iter().enumerate() {
            if level.extent.x < max_x as i64 || level.extent.y < max_y as i64 {
                thumb_level = index;
                break;
            }
        }
        let Some(level) = self.info.levels.get(thumb_level) else {
            return Err(ApiError::unhandled("IndexError('list index out of range')"));
        };
        let (extent_x, extent_y) = (level.extent.x, level.extent.y);

        let (mut max_x, mut max_y) = (max_x as f64, max_y as f64);
        if extent_x as f64 / max_x >= extent_y as f64 / max_y {
            max_y = (extent_y as f64 * max_x / extent_x as f64)
                .round_ties_even()
                .max(1.0);
        } else {
            max_x = (extent_x as f64 * max_y / extent_y as f64)
                .round_ties_even()
                .max(1.0);
        }
        let (target_x, target_y) = (max_x as u32, max_y as u32);

        // The whole level, at the service's own padding colour rather than the caller's.
        let source = self.read_region(
            thumb_level,
            0,
            0,
            extent_x as u32,
            extent_y as u32,
            self.settings.padding_color,
            0,
        )?;
        let source = self.apply_icc(source, icc_profile_intent, icc_profile_strict)?;

        Ok(match source {
            Pixels::Planar(data) => Pixels::Planar(thumb::resize_to_uint16(
                &data,
                target_y as usize,
                target_x as usize,
            )?),
            Pixels::Image(image) => Pixels::Image(wsi_core::image::pillow_resize(
                &image,
                target_x,
                target_y,
                wsi_core::image::Filter::Lanczos,
                None,
            )),
            raw => raw,
        })
    }

    async fn get_label(&self) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image label does not exist.",
        ))
    }

    async fn get_macro(&self, _: Option<IccIntent>, _: bool) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image macro does not exist.",
        ))
    }

    /// Always a 500.
    ///
    /// With no profile, `get_for_payload` raises `http.client.HTTPException`, which
    /// FastAPI does not recognise (divergence 11). With one, it returns the `TiffTag`
    /// object the tag lookup produced, and the response layer cannot render that either.
    async fn get_icc_profile(&self) -> Result<Bytes> {
        if self.has_icc_profile {
            Err(ApiError::unhandled(
                "AttributeError(\"'TiffTag' object has no attribute 'encode'\")",
            ))
        } else {
            Err(ApiError::unhandled(
                "HTTPException(404, 'ICC Profile not available.')",
            ))
        }
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "tifffile"
    }
}

/// `__get_levels_from_series`.
fn levels_from_series(file: &TiffFile, num_z: usize) -> Vec<SlideLevel> {
    let dimensions: Vec<(i64, i64)> = file
        .series
        .levels
        .iter()
        .map(|level| {
            let keyframe = level.keyframe();
            (keyframe.image_width as i64, keyframe.image_length as i64)
        })
        .collect();
    let downsamples: Vec<f64> = dimensions
        .iter()
        .enumerate()
        .map(|(index, &(width, _))| {
            if index == 0 {
                1.0
            } else {
                dimensions[0].0 as f64 / width as f64
            }
        })
        .collect();
    let mut levels = original_levels(&dimensions, &downsamples);
    for level in &mut levels {
        level.extent.z = num_z as i64;
    }
    levels
}

/// `__get_untiled_tile_extent`.
fn untiled_tile_extent(keyframe: &Page) -> SlideExtent {
    let width = keyframe.image_width as i64;
    let height = keyframe.image_length as i64;
    if width.max(height) <= UNTILED_TILE_MAX_DIMENSION {
        SlideExtent::new(width, height, 1)
    } else {
        SlideExtent::new(UNTILED_TILE_SIZE, UNTILED_TILE_SIZE, 1)
    }
}

fn tile_extent(keyframe: &Page) -> SlideExtent {
    if keyframe.is_tiled() {
        SlideExtent::new(
            keyframe.tile_width as i64,
            keyframe.tile_length as i64,
            keyframe.tile_depth as i64,
        )
    } else {
        untiled_tile_extent(keyframe)
    }
}

/// `__get_slide_info_ome_tif`.
fn slide_info_ome(file: &TiffFile, layout: &LayoutInfo) -> Result<SlideInfo> {
    let keyframe = file.series.keyframe();
    Ok(SlideInfo {
        id: String::new(),
        channels: Some(channels_ome(file, layout)),
        channel_depth: Some(keyframe.bits_per_sample as i32),
        extent: SlideExtent::new(
            keyframe.image_width as i64,
            keyframe.image_length as i64,
            layout.num_z as i64,
        ),
        num_levels: file.series.levels.len() as i64,
        pixel_size_nm: pixel_size_ome(file),
        tile_extent: tile_extent(keyframe),
        levels: levels_from_series(file, layout.num_z),
        format: "ome-tiff".into(),
        raw_download: false,
    })
}

/// `__get_slide_info_generic_tif`.
fn slide_info_generic(file: &TiffFile, layout: &LayoutInfo) -> Result<SlideInfo> {
    let keyframe = file.series.keyframe();
    Ok(SlideInfo {
        id: String::new(),
        channels: Some(channels_generic(file, layout)),
        channel_depth: Some(keyframe.bits_per_sample as i32),
        extent: SlideExtent::new(
            keyframe.image_width as i64,
            keyframe.image_length as i64,
            layout.num_z as i64,
        ),
        num_levels: file.series.levels.len() as i64,
        pixel_size_nm: pixel_size_generic(file),
        tile_extent: tile_extent(keyframe),
        levels: levels_from_series(file, layout.num_z),
        format: "tiff-multichannel".into(),
        raw_download: false,
    })
}

/// `__get_channels_ome_tif`.
///
/// An OME `<Channel>` can describe several planes at once through `SamplesPerPixel`, so
/// each element is expanded into that many entries before the list is indexed.
fn channels_ome(file: &TiffFile, layout: &LayoutInfo) -> Vec<SlideChannel> {
    if layout.is_rgb {
        return rgb_channel_list();
    }
    let mut expanded: Vec<&ome::OmeChannel> = Vec::new();
    if let Some(pixels) = file.ome_pixels() {
        for channel in &pixels.channels {
            for _ in 0..channel.samples_per_pixel.max(1) {
                expanded.push(channel);
            }
        }
    }

    (0..layout.num_channels)
        .map(|index| {
            let declared = expanded.get(index).copied();
            let name = declared
                .and_then(|channel| channel.name.clone())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| format!("Channel {index}"));
            SlideChannel {
                id: index as i32,
                name,
                color: ome_channel_color(declared, index),
            }
        })
        .collect()
}

/// `__get_ome_channel_color`.
fn ome_channel_color(channel: Option<&ome::OmeChannel>, index: usize) -> SlideColor {
    if let Some(value) = channel.and_then(|c| ome::parse_color(c.color.as_deref())) {
        let rgba = wsi_core::image::int_to_rgba_array(value);
        return SlideColor::new(rgba[0], rgba[1], rgba[2], rgba[3]);
    }
    let (r, g, b) = DEFAULT_PALETTE[index % DEFAULT_PALETTE.len()];
    SlideColor::new(r, g, b, 0)
}

/// `__get_channels_generic`.
fn channels_generic(file: &TiffFile, layout: &LayoutInfo) -> Vec<SlideChannel> {
    if layout.is_rgb {
        return rgb_channel_list();
    }
    let names = channel_names(file, layout.num_channels);
    let colors = channel_colors(file, layout.num_channels);
    (0..layout.num_channels)
        .map(|index| {
            let name = names
                .get(index)
                .filter(|name| !name.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("Channel {index}"));
            let (r, g, b) = colors[index];
            SlideChannel {
                id: index as i32,
                name,
                color: SlideColor::new(r, g, b, 0),
            }
        })
        .collect()
}

/// `__extract_channel_names`: ImageJ `Labels` first, then scanner hints in the
/// `ImageDescription`.
fn channel_names(file: &TiffFile, num_channels: usize) -> Vec<String> {
    if let Some(metadata) = file.imagej() {
        if !metadata.labels.is_empty() {
            return metadata.labels.iter().take(num_channels).cloned().collect();
        }
    }

    let description = &file.series.keyframe().description;
    if description.is_empty() {
        return Vec::new();
    }
    // Bio-Formats: "Channel:0:Name=DAPI".
    let bio_formats = capture_all(description, "Channel:", ":Name=");
    if !bio_formats.is_empty() {
        return bio_formats.into_iter().take(num_channels).collect();
    }
    // Generic "ChannelName=Foo" or "Channel 0 Name = Foo".
    let generic = capture_generic_channel_names(description);
    generic.into_iter().take(num_channels).collect()
}

/// `re.findall(r"Channel:\d+:Name=([^\r\n]+)", description)`.
fn capture_all(text: &str, prefix: &str, infix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(prefix) {
        let after = &rest[at + prefix.len()..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        rest = after;
        if digits.is_empty() {
            continue;
        }
        let after_digits = &after[digits.len()..];
        if !after_digits.starts_with(infix) {
            continue;
        }
        let value = &after_digits[infix.len()..];
        let value: String = value
            .chars()
            .take_while(|&c| c != '\r' && c != '\n')
            .collect();
        out.push(value.trim().to_owned());
        rest = &after_digits[infix.len() + value.len()..];
    }
    out
}

/// `re.findall(r"Channel(?:Name|\s*\d+\s*Name)\s*=\s*([^\r\n]+)", description)`.
fn capture_generic_channel_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("Channel") {
        let after = &rest[at + "Channel".len()..];
        rest = after;
        let tail = if let Some(tail) = after.strip_prefix("Name") {
            Some(tail)
        } else {
            // "Channel 0 Name"
            let trimmed = after.trim_start_matches([' ', '\t']);
            let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                None
            } else {
                let after_digits = trimmed[digits.len()..].trim_start_matches([' ', '\t']);
                after_digits.strip_prefix("Name")
            }
        };
        let Some(tail) = tail else { continue };
        let tail = tail.trim_start_matches([' ', '\t']);
        let Some(tail) = tail.strip_prefix('=') else {
            continue;
        };
        let tail = tail.trim_start_matches([' ', '\t']);
        let value: String = tail
            .chars()
            .take_while(|&c| c != '\r' && c != '\n')
            .collect();
        out.push(value.trim().to_owned());
        rest = &tail[value.len()..];
    }
    out
}

/// `__extract_channel_colors`: ImageJ `LUTs`, else the palette cycle.
fn channel_colors(file: &TiffFile, num_channels: usize) -> Vec<(i32, i32, i32)> {
    let palette = |index: usize| DEFAULT_PALETTE[index % DEFAULT_PALETTE.len()];
    let Some(metadata) = file.imagej() else {
        return (0..num_channels).map(palette).collect();
    };
    if metadata.luts.is_empty() {
        return (0..num_channels).map(palette).collect();
    }
    (0..num_channels)
        .map(|index| {
            match metadata
                .luts
                .get(index)
                .and_then(|lut| imagej::lut_color(lut))
            {
                Some((r, g, b)) => (i32::from(r), i32::from(g), i32::from(b)),
                None => palette(index),
            }
        })
        .collect()
}

/// `__get_pixel_size_ome_tif`, falling back to the resolution tags.
fn pixel_size_ome(file: &TiffFile) -> SlidePixelSizeNm {
    let pixels = file.ome_pixels();
    let read = |size: Option<&String>, unit: Option<&String>| {
        ome::length_to_nm(size.map(String::as_str), unit.map(String::as_str))
    };
    let (x, y, z) = match pixels {
        Some(pixels) => (
            read(
                pixels.physical_size_x.as_ref(),
                pixels.physical_size_x_unit.as_ref(),
            ),
            read(
                pixels.physical_size_y.as_ref(),
                pixels.physical_size_y_unit.as_ref(),
            ),
            read(
                pixels.physical_size_z.as_ref(),
                pixels.physical_size_z_unit.as_ref(),
            ),
        ),
        None => (None, None, None),
    };

    match (x, y) {
        (Some(x), Some(y)) => SlidePixelSizeNm::xyz(x, y, z),
        _ => {
            let generic = pixel_size_generic(file);
            SlidePixelSizeNm::xyz(generic.x, generic.y, z.or(generic.z))
        }
    }
}

/// `__get_pixel_size_generic` — the TIFF resolution tags, or 1 µm/px.
fn pixel_size_generic(file: &TiffFile) -> SlidePixelSizeNm {
    let z = file.imagej().and_then(imagej::z_spacing_nm);
    let keyframe = file.series.keyframe();
    let fallback = || SlidePixelSizeNm::xyz(1000.0, 1000.0, z);

    let Some(x_res) = keyframe.ifd.get(tiff::ifd::tag::X_RESOLUTION) else {
        return fallback();
    };
    let Some(y_res) = keyframe.ifd.get(tiff::ifd::tag::Y_RESOLUTION) else {
        return fallback();
    };
    // `ResolutionUnit` defaults to 2 (inch) when the tag is absent.
    let unit_nm = match keyframe
        .ifd
        .get(tiff::ifd::tag::RESOLUTION_UNIT)
        .and_then(|entry| entry.as_u64())
        .unwrap_or(2)
    {
        2 => 2.54e7,
        3 => 1e7,
        _ => return fallback(),
    };
    let (Some(x_per_unit), Some(y_per_unit)) = (x_res.as_rational(), y_res.as_rational()) else {
        return fallback();
    };
    if x_per_unit == 0.0 || y_per_unit == 0.0 {
        return fallback();
    }
    SlidePixelSizeNm::xyz(unit_nm / x_per_unit, unit_nm / y_per_unit, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    /// The two strict-ICC outcomes are *different* errors, and which one a slide gets is
    /// decided by nothing but the presence of the tag. Reproducing only the 500 made every
    /// profile-less slide answer 500 where Python answers 412.
    #[test]
    fn a_strict_icc_request_fails_differently_with_and_without_the_tag() {
        let without = strict_icc_failure(false);
        assert_eq!(without.status, http::StatusCode::PRECONDITION_FAILED);
        assert_eq!(without.detail_text(), "Error: ICC Profile not available.");

        let with = strict_icc_failure(true);
        assert_eq!(with.status, http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    fn plugin() -> TiffFilePlugin {
        TiffFilePlugin::new(Arc::new(Settings::default()))
    }

    fn slide(name: &str) -> TiffFileSlide {
        TiffFileSlide::open(fixture(name), Arc::new(Settings::default()))
            .unwrap_or_else(|e| panic!("{name}: {}", e.detail_text()))
    }

    #[test]
    fn a_plain_rgb_tiff_is_left_to_tiffslide() {
        assert!(!plugin().is_supported(&fixture("plain_rgb.tif")));
    }

    #[test]
    fn imagej_stacks_are_claimed() {
        assert!(plugin().is_supported(&fixture("rgb_zstack_imagej.tif")));
        assert!(plugin().is_supported(&fixture("zstack_imagej.tif")));
        assert!(plugin().is_supported(&fixture("channels_only_imagej.tif")));
    }

    #[test]
    fn every_ome_file_is_claimed_on_its_name_alone() {
        assert!(plugin().is_supported(&fixture("rgb_contig.ome.tif")));
        assert!(plugin().is_supported(&fixture("fluorescence.ome.tif")));
    }

    #[test]
    fn a_non_tiff_extension_is_not_claimed() {
        assert!(!plugin().is_supported(Path::new("slide.svs")));
        assert!(!plugin().is_supported(Path::new("missing.tif")));
    }

    #[test]
    fn the_generic_channel_name_patterns_match_the_python_regexes() {
        let bio = "Channel:0:Name=DAPI\r\nChannel:1:Name=FITC\r\n";
        assert_eq!(capture_all(bio, "Channel:", ":Name="), vec!["DAPI", "FITC"]);
        assert_eq!(
            capture_generic_channel_names("ChannelName=Red\nChannel 1 Name = Green\n"),
            vec!["Red", "Green"]
        );
        assert!(capture_all("nothing here", "Channel:", ":Name=").is_empty());
    }

    #[test]
    fn an_untiled_level_serves_itself_whole_until_it_gets_large() {
        let small = slide("fluorescence.ome.tif");
        assert_eq!(small.info.tile_extent, SlideExtent::new(8, 6, 1));
    }

    #[test]
    fn the_untiled_extent_rule_switches_at_2048() {
        // Exercised directly, because a 2049-pixel fixture would be megabytes of zeros.
        let extent = |width: usize, height: usize| {
            let mut page = slide("fluorescence.ome.tif")
                .with_file(|file| Ok(file.series.keyframe().clone()))
                .unwrap();
            page.image_width = width;
            page.image_length = height;
            untiled_tile_extent(&page)
        };
        assert_eq!(extent(2048, 2048), SlideExtent::new(2048, 2048, 1));
        assert_eq!(extent(2049, 10), SlideExtent::new(512, 512, 1));
        assert_eq!(extent(10, 2049), SlideExtent::new(512, 512, 1));
    }
}
