//! The surface of `WsiDicom` that the plugin uses, reproduced.
//!
//! `wsidicom` is a large library; the plugin touches very little of it. What it needs is a
//! pyramid of levels with sizes, tile sizes and pixel spacings, the label and overview
//! images, a thumbnail, an ICC profile, and two reads — a region and a tile. Everything
//! here is traced to the method it stands in for.
//!
//! Three of its decisions are easy to get wrong and are reproduced deliberately:
//!
//! - **A level's index is an exponent**, assigned from the ratio of pixel spacings:
//!   `round(log2(this.spacing.width / base.spacing.width))` — `Level._assign_level`,
//!   `group/level.py:234`. The Aperio pyramid reports 0, 2, 4 across list indices 0, 1, 2.
//!   `wsidicom` refuses a ratio more than `level_scale_tolerance` from an integer, and so
//!   does this.
//! - **`read_region` does not pad.** It checks the region against the level and raises
//!   `WsiDicomOutOfBoundsError` — `wsidicom.py:605`. The Python plugin does not catch that,
//!   so it escapes FastAPI; see `docs/divergences.md`.
//! - **`read_thumbnail` picks the smallest image at least as wide as the box**
//!   (`Series.get_closest_by_size`, `series/series.py:154`), preferring a `THUMBNAIL`
//!   instance over a level, and only then calls Pillow's `thumbnail`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use wsi_plugin_tifffile::tiff::codec::{compression, decode_chunk, ChunkGeometry};
use wsi_plugin_tifffile::tiff::ifd::ByteOrder;
use wsi_plugin_tifffile::tiff::page::{Raster, SampleDtype};

use crate::dicom::{self, tags, Dataset, Instance};

/// `settings.level_scale_tolerance` — `wsidicom/config.py`.
const LEVEL_SCALE_TOLERANCE: f64 = 1e-2;

/// The largest region this will allocate for, in bytes.
///
/// `MAX_FRAME_BYTES` in [`crate::dicom`] bounds one *frame*; this bounds the stitched
/// output, which is sized from `TotalPixelMatrixColumns`/`Rows` — 32-bit fields in the file.
/// A crafted or corrupt instance can declare a 100000x100000 matrix with a single frame,
/// and `read_full` would then ask for 30 GB. The route's own `max_returned_region_size`
/// does not help here: `read_label`, `read_overview` and the thumbnail source all read a
/// whole image without passing through it.
///
/// The largest legitimate read in the collection is the Aperio thumbnail source, a
/// 2875x2057 level at 17.7 MB; a region request that reaches the plugin has already been
/// bounded to `max_returned_region_size`, 25 million pixels by default. 256 MB is far above
/// both and far below anything that would exhaust a host.
const MAX_REGION_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
pub struct ReaderError(pub String);

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub type Result<T> = std::result::Result<T, ReaderError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(ReaderError(message.into()))
}

impl From<dicom::DicomError> for ReaderError {
    fn from(error: dicom::DicomError) -> Self {
        ReaderError(error.0)
    }
}

/// Which group an instance belongs to, from `ImageType[2]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    Volume,
    Label,
    Overview,
    Thumbnail,
}

impl Flavour {
    fn parse(image_type: &[String]) -> Option<Self> {
        match image_type.get(2).map(String::as_str) {
            Some("VOLUME") => Some(Flavour::Volume),
            Some("LABEL") => Some(Flavour::Label),
            Some("OVERVIEW") => Some(Flavour::Overview),
            Some("THUMBNAIL") => Some(Flavour::Thumbnail),
            _ => None,
        }
    }
}

/// One frame set: an instance plus the geometry needed to read tiles out of it.
#[derive(Debug)]
struct FrameSet {
    instance: Instance,
    /// `TotalPixelMatrixColumns` / `Rows` — the image this frame set covers.
    width: u32,
    height: u32,
    /// `Columns` / `Rows` — one frame.
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
    samples: usize,
    bits_allocated: usize,
    /// The *TIFF* photometric number, which is what `decode_chunk` takes.
    photometric: u16,
    /// `compression::JPEG` or `compression::NONE`.
    compression: u16,
    /// `PixelSpacing` in millimetres, when the instance declares one.
    spacing: Option<(f64, f64)>,
}

impl FrameSet {
    fn build(instance: Instance) -> Result<Self> {
        let dataset = &instance.dataset;

        let organization = dataset
            .string(tags::DIMENSION_ORGANIZATION_TYPE)
            .unwrap_or_default();
        if organization != "TILED_FULL" {
            return err(format!(
                "DimensionOrganizationType is {organization:?}; only TILED_FULL is \
                 supported, because any other layout needs the per-frame functional groups"
            ));
        }

        let integer = |tag, what: &str| {
            dataset
                .integer(tag)
                .ok_or_else(|| ReaderError(format!("{what} is missing")))
        };

        let tile_width = integer(tags::COLUMNS, "Columns")? as u32;
        let tile_height = integer(tags::ROWS, "Rows")? as u32;
        // A LABEL or OVERVIEW instance is a single frame and may omit the matrix size, in
        // which case the frame *is* the image.
        let width = dataset
            .integer(tags::TOTAL_PIXEL_MATRIX_COLUMNS)
            .unwrap_or(i64::from(tile_width)) as u32;
        let height = dataset
            .integer(tags::TOTAL_PIXEL_MATRIX_ROWS)
            .unwrap_or(i64::from(tile_height)) as u32;
        if tile_width == 0 || tile_height == 0 {
            return err("Rows or Columns is zero");
        }

        let photometric_name = dataset
            .string(tags::PHOTOMETRIC_INTERPRETATION)
            .unwrap_or_default();
        // DICOM says the frame's colour space here; `decode_chunk` wants TIFF's number for
        // it. `RGB` means the JPEG samples already are R, G and B, which is the case the
        // APP14 splice in `codec.rs` exists for.
        let photometric = match photometric_name.as_str() {
            "RGB" => 2,
            name if name.starts_with("YBR") => 6,
            "MONOCHROME2" => 1,
            other => {
                return err(format!(
                    "PhotometricInterpretation {other:?} is not supported"
                ))
            }
        };

        let compression = match instance.transfer_syntax.as_str() {
            dicom::transfer_syntax::JPEG_BASELINE => compression::JPEG,
            dicom::transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN => compression::NONE,
            other => return err(format!("transfer syntax {other} is not supported")),
        };

        let planar = dataset.integer(tags::PLANAR_CONFIGURATION).unwrap_or(0);
        if planar != 0 {
            return err(format!(
                "PlanarConfiguration is {planar}; only interleaved (0) is supported"
            ));
        }

        let spacing = pixel_spacing(dataset);

        Ok(Self {
            width,
            height,
            tile_width,
            tile_height,
            tiles_across: width.div_ceil(tile_width),
            tiles_down: height.div_ceil(tile_height),
            samples: integer(tags::SAMPLES_PER_PIXEL, "SamplesPerPixel")? as usize,
            bits_allocated: integer(tags::BITS_ALLOCATED, "BitsAllocated")? as usize,
            photometric,
            compression,
            spacing,
            instance,
        })
    }

    /// Decodes one tile, always at the full tile size — `TILED_FULL` pads the edge frames
    /// rather than shortening them.
    fn decode_tile(&self, tile_x: u32, tile_y: u32) -> Result<Raster> {
        if tile_x >= self.tiles_across || tile_y >= self.tiles_down {
            return err(format!(
                "tile ({tile_x}, {tile_y}) is outside the {}x{} tile grid",
                self.tiles_across, self.tiles_down
            ));
        }
        // TILED_FULL is row-major over the tile grid, with one focal plane and one optical
        // path here, so the frame index is arithmetic.
        let index = (tile_y * self.tiles_across + tile_x) as usize;
        let frame = self.instance.read_frame(index)?;

        let geometry = ChunkGeometry {
            width: self.tile_width as usize,
            height: self.tile_height as usize,
            samples: self.samples,
            bits_per_sample: self.bits_allocated,
            photometric: self.photometric,
        };
        let bytes = decode_chunk(
            self.compression,
            1,
            ByteOrder::Little,
            geometry,
            &frame,
            None,
        )
        .map_err(|e| ReaderError(format!("frame {index}: {e}")))?;

        Ok(Raster {
            height: self.tile_height as usize,
            width: self.tile_width as usize,
            samples: self.samples,
            dtype: SampleDtype::U8,
            bytes,
        })
    }

    /// `ImageData.stitch_tiles` — `instance/image_data.py:452`.
    ///
    /// The region has to be inside the image; `wsidicom` raises rather than padding, and
    /// the caller has already checked, so this only guards against a bug here.
    fn read_region(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Raster> {
        if width == 0 || height == 0 {
            return err("region has a zero dimension");
        }
        // Widened before adding: both halves come from a request, and `x + width` in `u32`
        // wraps for a large enough size, which would turn an out-of-bounds region into an
        // in-bounds one.
        let ends_x = u64::from(x) + u64::from(width);
        let ends_y = u64::from(y) + u64::from(height);
        if ends_x > u64::from(self.width) || ends_y > u64::from(self.height) {
            return err(format!(
                "region {width}x{height} at ({x}, {y}) is outside the image {}x{}",
                self.width, self.height
            ));
        }

        // The output is sized from the file's own declared geometry, so it is bounded
        // before it is allocated rather than after.
        let bytes = u64::from(width) * u64::from(height) * self.samples as u64;
        if bytes > MAX_REGION_BYTES {
            return err(format!(
                "region {width}x{height} needs {bytes} bytes, past the \
                 {MAX_REGION_BYTES} limit"
            ));
        }

        let first_tile_x = x / self.tile_width;
        let first_tile_y = y / self.tile_height;
        let last_tile_x = (x + width - 1) / self.tile_width;
        let last_tile_y = (y + height - 1) / self.tile_height;

        let mut out = Raster::filled(
            height as usize,
            width as usize,
            self.samples,
            SampleDtype::U8,
            0.0,
        );

        for tile_y in first_tile_y..=last_tile_y {
            for tile_x in first_tile_x..=last_tile_x {
                let tile = self.decode_tile(tile_x, tile_y)?;
                // Where this tile starts in image coordinates, and the overlap with the
                // requested region.
                let tile_origin_x = tile_x * self.tile_width;
                let tile_origin_y = tile_y * self.tile_height;
                let src_x = x.saturating_sub(tile_origin_x);
                let src_y = y.saturating_sub(tile_origin_y);
                let dst_x = tile_origin_x.saturating_sub(x);
                let dst_y = tile_origin_y.saturating_sub(y);
                let copy_width = (self.tile_width - src_x).min(width - dst_x);
                let copy_height = (self.tile_height - src_y).min(height - dst_y);

                out.blit(
                    &tile,
                    dst_y as usize,
                    dst_x as usize,
                    src_y as usize,
                    src_x as usize,
                    copy_height as usize,
                    copy_width as usize,
                    0,
                );
            }
        }
        Ok(out)
    }

    fn read_full(&self) -> Result<Raster> {
        self.read_region(0, 0, self.width, self.height)
    }
}

/// `PixelSpacing`, from wherever the instance put it.
///
/// It belongs in `SharedFunctionalGroupsSequence > PixelMeasuresSequence`, and the nested
/// search is what finds it; a `LABEL` instance may have none at all.
fn pixel_spacing(dataset: &Dataset) -> Option<(f64, f64)> {
    let values = dataset.find_nested(tags::PIXEL_SPACING)?.decimals();
    // PS3.3: `PixelSpacing` is row spacing then column spacing — height, then width.
    match values.as_slice() {
        [row, column, ..] if *row > 0.0 && *column > 0.0 => Some((*column, *row)),
        _ => None,
    }
}

/// One level of the pyramid.
#[derive(Debug)]
pub struct Level {
    /// The *exponent*, not the list index: `2 ** exponent` is the downsample.
    pub exponent: i32,
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    /// Micrometres per pixel, which is `PixelSpacing * 1000`.
    pub mpp: Option<(f64, f64)>,
    frames: FrameSet,
}

pub struct WsiDicomReader {
    pub directory: PathBuf,
    pub levels: Vec<Level>,
    labels: Vec<FrameSet>,
    overviews: Vec<FrameSet>,
    thumbnails: Vec<FrameSet>,
    icc_profile: Option<Vec<u8>>,
}

impl WsiDicomReader {
    /// `WsiDicom.open(path)`.
    pub fn open(directory: &Path) -> Result<Self> {
        let files = dicom::dicom_files(directory)?;
        if files.is_empty() {
            return err(format!("{} holds no .dcm files", directory.display()));
        }

        let mut volumes = Vec::new();
        let mut labels = Vec::new();
        let mut overviews = Vec::new();
        let mut thumbnails = Vec::new();
        let mut icc_profile = None;

        for file in &files {
            let instance = dicom::read_instance(file)?;
            let image_type = instance
                .dataset
                .get(tags::IMAGE_TYPE)
                .map(|v| v.strings())
                .unwrap_or_default();
            let Some(flavour) = Flavour::parse(&image_type) else {
                // An instance that is none of the four flavours is not part of the image;
                // `wsidicom` ignores it rather than failing.
                tracing::debug!(
                    file = %file.display(),
                    image_type = ?image_type,
                    "ignoring an instance that is not a volume, label, overview or thumbnail"
                );
                continue;
            };

            // `__get_icc_bytes`: the first non-empty profile wins, whichever instance it
            // came from.
            if icc_profile.is_none() {
                if let Some(value) = instance.dataset.find_nested(tags::ICC_PROFILE) {
                    if !value.bytes.is_empty() {
                        icc_profile = Some(value.bytes.clone());
                    }
                }
            }

            let frames = FrameSet::build(instance)?;
            match flavour {
                Flavour::Volume => volumes.push(frames),
                Flavour::Label => labels.push(frames),
                Flavour::Overview => overviews.push(frames),
                Flavour::Thumbnail => thumbnails.push(frames),
            }
        }

        if volumes.is_empty() {
            return err(format!(
                "{} holds no VOLUME instances, so it has no pyramid",
                directory.display()
            ));
        }

        let levels = assign_levels(volumes)?;

        // Both series are kept sorted largest first, which is the order
        // `get_closest_by_size` relies on: it starts from the first group's width as its
        // ceiling and narrows.
        labels.sort_by_key(|f| std::cmp::Reverse(f.width));
        overviews.sort_by_key(|f| std::cmp::Reverse(f.width));
        thumbnails.sort_by_key(|f| std::cmp::Reverse(f.width));

        Ok(Self {
            directory: directory.to_owned(),
            levels,
            labels,
            overviews,
            thumbnails,
            icc_profile,
        })
    }

    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// `WsiDicom.read_region(location, level, size)` — `wsidicom.py:561`.
    ///
    /// `level` is the exponent, and `location` is in that level's own pixels.
    pub fn read_region(
        &self,
        location: (u32, u32),
        level: i32,
        size: (u32, u32),
    ) -> Result<Raster> {
        let level = self.level_at_or_below(level)?;
        level
            .frames
            .read_region(location.0, location.1, size.0, size.1)
    }

    /// `WsiDicom.read_tile(level, tile)` — `wsidicom.py:716`.
    pub fn read_tile(&self, level: i32, tile: (u32, u32)) -> Result<Raster> {
        let level = self.level_at_or_below(level)?;
        level.frames.decode_tile(tile.0, tile.1)
    }

    /// `WsiDicom.read_label()`, a 404 in the plugin when there is none.
    pub fn read_label(&self) -> Result<Raster> {
        let Some(label) = self.labels.first() else {
            return err("no label image");
        };
        label.read_full()
    }

    /// `WsiDicom.read_overview()`, which the plugin surfaces as the macro image.
    pub fn read_overview(&self) -> Result<Raster> {
        let Some(overview) = self.overviews.first() else {
            return err("no overview image");
        };
        overview.read_full()
    }

    /// The image `read_thumbnail` would downscale, and its size.
    ///
    /// `THUMBNAIL` instances are preferred, and within either series the choice is
    /// `get_closest_by_size`: the *smallest* image at least as wide as the box. A box wider
    /// than everything available selects nothing in that series, which is why the levels
    /// are tried second rather than instead.
    pub fn thumbnail_source(&self, size: (u32, u32)) -> Result<Raster> {
        if let Some(frames) = closest_by_size(&self.thumbnails, size.0) {
            return frames.read_full();
        }
        let level_frames: Vec<&FrameSet> = self.levels.iter().map(|l| &l.frames).collect();
        if let Some(frames) = closest_by_size_refs(&level_frames, size.0) {
            return frames.read_full();
        }
        err(format!(
            "no image is large enough to generate a thumbnail of {}x{}",
            size.0, size.1
        ))
    }

    /// `Pyramid.get_closest_by_level`, which the plugin always calls with a level that
    /// exists — it reads the exponent out of the level list first.
    fn level_at_or_below(&self, exponent: i32) -> Result<&Level> {
        self.levels
            .iter()
            .find(|level| level.exponent == exponent)
            .ok_or_else(|| {
                ReaderError(format!(
                    "no level with exponent {exponent}; the pyramid has {:?}",
                    self.levels.iter().map(|l| l.exponent).collect::<Vec<_>>()
                ))
            })
    }
}

fn closest_by_size(frames: &[FrameSet], width: u32) -> Option<&FrameSet> {
    let refs: Vec<&FrameSet> = frames.iter().collect();
    closest_by_size_refs(&refs, width).map(|f| {
        // Re-find by identity so the borrow outlives the temporary vector.
        frames
            .iter()
            .find(|candidate| std::ptr::eq(*candidate, f))
            .expect("the match came from this slice")
    })
}

/// `Series.get_closest_by_size` — `series/series.py:154`.
fn closest_by_size_refs<'a>(frames: &[&'a FrameSet], width: u32) -> Option<&'a FrameSet> {
    let first = frames.first()?;
    let mut ceiling = first.width;
    let mut closest = None;
    for candidate in frames {
        if width <= candidate.width && candidate.width <= ceiling {
            ceiling = candidate.width;
            closest = Some(*candidate);
        }
    }
    closest
}

/// Groups the volume instances into levels and assigns each an exponent.
///
/// `Pyramid.__init__` keys levels by `level.level` and sorts by it, so two instances that
/// resolve to the same exponent collapse into one level — which is how different optical
/// paths or focal planes of the same level are meant to combine. Here the first one wins,
/// since the collection has a single path and plane throughout.
fn assign_levels(volumes: Vec<FrameSet>) -> Result<Vec<Level>> {
    // The base is the finest spacing, which is also the largest image.
    let base_spacing = volumes
        .iter()
        .filter_map(|frames| frames.spacing)
        .map(|(width, _)| width)
        .min_by(f64::total_cmp);

    let mut by_exponent: BTreeMap<i32, Level> = BTreeMap::new();
    for frames in volumes {
        let exponent = match (frames.spacing, base_spacing) {
            (Some((spacing, _)), Some(base)) if base > 0.0 => {
                let ratio = (spacing / base).log2();
                let rounded = ratio.round();
                if (ratio - rounded).abs() > LEVEL_SCALE_TOLERANCE {
                    return err(format!(
                        "level scale {ratio} is more than the tolerance \
                         {LEVEL_SCALE_TOLERANCE} from the nearest integer {rounded}; \
                         base spacing {base}, this level {spacing}"
                    ));
                }
                rounded as i32
            }
            // Without a pixel spacing there is nothing to assign from. `wsidicom` raises
            // `WsiDicomNoResolutionError` reaching for `mpp` in that case; refusing here
            // keeps the message about the cause.
            _ => {
                return err(
                    "a VOLUME instance has no PixelSpacing, so its pyramid level cannot \
                     be assigned",
                )
            }
        };

        let mpp = frames
            .spacing
            .map(|(width, height)| (width * 1000.0, height * 1000.0));
        by_exponent.entry(exponent).or_insert(Level {
            exponent,
            width: frames.width,
            height: frames.height,
            tile_width: frames.tile_width,
            tile_height: frames.tile_height,
            mpp,
            frames,
        });
    }

    Ok(by_exponent.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flavour_comes_from_the_third_element_of_image_type() {
        let parse = |parts: &[&str]| {
            Flavour::parse(&parts.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
        };
        assert_eq!(
            parse(&["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]),
            Some(Flavour::Volume)
        );
        assert_eq!(
            parse(&["DERIVED", "PRIMARY", "THUMBNAIL", "RESAMPLED"]),
            Some(Flavour::Thumbnail)
        );
        // wsidicomizer writes "ORGINAL" — the typo is in the files, and it is the *third*
        // element that matters, so it makes no difference.
        assert_eq!(
            parse(&["ORGINAL", "PRIMARY", "LABEL", "RESAMPLED"]),
            Some(Flavour::Label)
        );
        assert_eq!(parse(&["ORIGINAL", "PRIMARY"]), None);
        assert_eq!(parse(&["ORIGINAL", "PRIMARY", "OTHER"]), None);
    }

    #[test]
    fn closest_by_size_takes_the_smallest_image_that_still_covers_the_box() {
        // Standing in for the frame sets, since only the width matters here.
        fn widths(widths: &[u32]) -> Vec<u32> {
            widths.to_vec()
        }
        // The algorithm, applied to bare widths: largest first, narrow from there.
        fn closest(sorted_descending: &[u32], want: u32) -> Option<u32> {
            let mut ceiling = *sorted_descending.first()?;
            let mut closest = None;
            for candidate in sorted_descending {
                if want <= *candidate && *candidate <= ceiling {
                    ceiling = *candidate;
                    closest = Some(*candidate);
                }
            }
            closest
        }

        let mirax = widths(&[94794, 47397, 23698, 11849, 5924, 2962, 1481, 740, 370, 185]);
        assert_eq!(closest(&mirax, 500), Some(740));
        let aperio = widths(&[46000, 11500, 2875]);
        assert_eq!(closest(&aperio, 500), Some(2875));
        // A box wider than every image selects nothing, which is what sends the caller on
        // to the next series.
        assert_eq!(closest(&aperio, 50000), None);
        // An exact match is allowed, not excluded.
        assert_eq!(closest(&aperio, 2875), Some(2875));
    }

    #[test]
    fn a_level_exponent_comes_from_the_spacing_ratio() {
        // Aperio's three levels, as the ground truth records them.
        for (spacing, expected) in [(0.000499, 0), (0.001996, 2), (0.007984, 4)] {
            let ratio: f64 = (spacing / 0.000499f64).log2();
            assert!((ratio - f64::from(expected)).abs() <= LEVEL_SCALE_TOLERANCE);
            assert_eq!(ratio.round() as i32, expected);
        }
        // MIRAX runs 0 through 9 off a base of 0.00023387573964.
        let base = 0.00023387573964f64;
        for (spacing, expected) in [
            (0.00046775147929, 1),
            (0.00093550295858, 2),
            (0.11974437869822, 9),
        ] {
            let ratio = (spacing / base).log2();
            assert!(
                (ratio - f64::from(expected)).abs() <= LEVEL_SCALE_TOLERANCE,
                "{ratio} is not within tolerance of {expected}"
            );
        }
    }

    /// The bounds check has to survive a size that wraps `u32`, because wrapping would
    /// turn an out-of-bounds region into an in-bounds one and read the wrong frames.
    #[test]
    fn a_region_size_that_wraps_u32_is_still_out_of_bounds() {
        // The arithmetic the check performs, in isolation: the frame set itself needs a
        // real file to build.
        let (image_width, image_height) = (46000u32, 32914u32);
        let (x, width) = (10u32, u32::MAX - 5);
        // In `u32` this wraps to 4, which is inside the image.
        assert!(x.wrapping_add(width) < image_width);
        // Widened, it does not.
        assert!(u64::from(x) + u64::from(width) > u64::from(image_width));
        assert!(u64::from(0u32) + u64::from(u32::MAX) > u64::from(image_height));
    }

    #[test]
    fn a_region_past_the_allocation_cap_is_refused() {
        // The cap is on the stitched output, which the file's declared matrix size drives.
        let bytes = |w: u64, h: u64| w * h * 3;
        // The largest legitimate read: Aperio's thumbnail source.
        assert!(bytes(2875, 2057) < MAX_REGION_BYTES);
        // A route-bounded region at `max_returned_region_size`, 25 million pixels.
        assert!(bytes(5000, 5000) < MAX_REGION_BYTES);
        // A crafted matrix.
        assert!(bytes(100_000, 100_000) > MAX_REGION_BYTES);
    }

    #[test]
    fn pixel_spacing_is_row_then_column() {
        // PS3.3 orders it height, width. Reading it the other way round would swap mpp on
        // any slide with non-square pixels — MIRAX has them.
        let mut dataset = Dataset::default();
        assert_eq!(pixel_spacing(&dataset), None);

        dataset = parse_spacing("0.00093550295858\\0.00093732283465");
        let (width, height) = pixel_spacing(&dataset).unwrap();
        assert_eq!(height, 0.00093550295858);
        assert_eq!(width, 0.00093732283465);
    }

    /// Builds a dataset carrying just a `PixelSpacing`, through the real parser so the test
    /// exercises the same path the files do.
    fn parse_spacing(value: &str) -> Dataset {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&tags::PIXEL_SPACING.0.to_le_bytes());
        bytes.extend_from_slice(&tags::PIXEL_SPACING.1.to_le_bytes());
        bytes.extend_from_slice(b"DS");
        let mut value = value.as_bytes().to_vec();
        if value.len() % 2 == 1 {
            value.push(b' ');
        }
        bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&value);

        let directory = std::env::temp_dir().join(format!(
            "wsi-dicom-spacing-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("spacing.dcm");

        let mut file = vec![0u8; 128];
        file.extend_from_slice(b"DICM");
        let uid = b"1.2.840.10008.1.2.1\0";
        let mut meta = Vec::new();
        meta.extend_from_slice(&[0x02, 0x00, 0x10, 0x00]);
        meta.extend_from_slice(b"UI");
        meta.extend_from_slice(&(uid.len() as u16).to_le_bytes());
        meta.extend_from_slice(uid);
        file.extend_from_slice(&[0x02, 0x00, 0x00, 0x00]);
        file.extend_from_slice(b"UL");
        file.extend_from_slice(&4u16.to_le_bytes());
        file.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        file.extend_from_slice(&meta);
        file.extend_from_slice(&bytes);
        std::fs::write(&path, &file).unwrap();

        let instance = dicom::read_instance(&path).unwrap();
        let _ = std::fs::remove_dir_all(&directory);
        instance.dataset
    }
}
