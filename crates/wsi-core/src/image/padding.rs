//! Edge and out-of-bounds handling, ported from `wsi_service/utils/image_utils.py:154-400`.
//!
//! Two shapes exist, and they behave differently on purpose:
//!
//! * **regions** clamp the request to the level, do exactly one read, and then either pad
//!   the result back up to the requested size (`extend = true`) or return the smaller real
//!   overlap (`extend = false`);
//! * **tiles** always read from the plugin first — the plugin is expected to pad — and
//!   then either crop away the padded area (`extend = false`) or re-pad to a full tile
//!   (`extend = true`).
//!
//! The `<` rather than `<=` in [`check_complete_region_overlap`] is deliberate and comes
//! straight from the Python: a region that ends exactly on the extent takes the padding
//! path. Changing it would change output for edge requests.

use image::DynamicImage;

use crate::error::{ApiError, Result};
use crate::models::SlideInfo;
use crate::pixels::{Pixels, PlanarData};
use crate::slide::{ReadOptions, Slide};

/// `check_complete_region_overlap` — `image_utils.py:154`.
pub fn check_complete_region_overlap(
    info: &SlideInfo,
    level: usize,
    start_x: i64,
    start_y: i64,
    size_x: i64,
    size_y: i64,
) -> bool {
    let extent = info.levels[level].extent;
    start_x >= 0 && start_y >= 0 && start_x + size_x < extent.x && start_y + size_y < extent.y
}

/// `_region_intersection` — `image_utils.py:165`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intersection {
    pub overlap_w: i64,
    pub overlap_h: i64,
    /// Clamped coordinates to request from the slide.
    pub src_x: i64,
    pub src_y: i64,
    /// Paste offsets in the output canvas when extending.
    pub dst_x: i64,
    pub dst_y: i64,
}

pub fn region_intersection(
    info: &SlideInfo,
    level: usize,
    start_x: i64,
    start_y: i64,
    size_x: i64,
    size_y: i64,
) -> Intersection {
    let extent = info.levels[level].extent;

    let src_x = start_x.max(0);
    let src_y = start_y.max(0);
    let src_x2 = extent.x.min(start_x + size_x);
    let src_y2 = extent.y.min(start_y + size_y);

    Intersection {
        overlap_w: (src_x2 - src_x).max(0),
        overlap_h: (src_y2 - src_y).max(0),
        src_x,
        src_y,
        dst_x: (-start_x).max(0),
        dst_y: (-start_y).max(0),
    }
}

/// `check_complete_tile_overlap` — `image_utils.py:299`.
pub fn check_complete_tile_overlap(
    info: &SlideInfo,
    level: usize,
    tile_x: i64,
    tile_y: i64,
) -> bool {
    let tile = info.tile_extent;
    let extent = info.levels[level].extent;
    let x0 = tile_x * tile.x;
    let y0 = tile_y * tile.y;
    x0 >= 0 && y0 >= 0 && x0 + tile.x <= extent.x && y0 + tile.y <= extent.y
}

/// `_overlap_wh` — `image_utils.py:311`. Clamped overlap of a tile with the level.
pub fn tile_overlap(info: &SlideInfo, level: usize, tile_x: i64, tile_y: i64) -> (i64, i64) {
    let tile = info.tile_extent;
    let extent = info.levels[level].extent;
    let x0 = tile_x * tile.x;
    let y0 = tile_y * tile.y;
    (
        tile.x.min(extent.x - x0).max(0),
        tile.y.min(extent.y - y0).max(0),
    )
}

/// Decodes a `Raw` payload so it can be padded or cropped, mirroring
/// `Image.open(BytesIO(...))` in both wrappers.
fn decode_raw(pixels: Pixels) -> Result<Pixels> {
    match pixels {
        Pixels::Raw(bytes) => {
            let img = image::load_from_memory(&bytes).map_err(|e| {
                ApiError::bad_request(format!("Failed to decode plugin tile payload: {e}"))
            })?;
            Ok(Pixels::Image(img))
        }
        other => Ok(other),
    }
}

/// Builds an all-padding canvas matching `prototype`'s layout.
///
/// For images this is `Image.new(mode, size, padding_color)`; for planar data it is
/// `np.empty(...)` filled with the padding colour, or zeros when no colour was given.
fn padded_canvas(
    prototype: &Pixels,
    width: i64,
    height: i64,
    padding_color: Option<(u8, u8, u8)>,
) -> Result<Pixels> {
    match prototype {
        Pixels::Image(img) => Ok(Pixels::Image(new_image_like(
            img,
            width as u32,
            height as u32,
            padding_color,
        ))),
        Pixels::Planar(data) => {
            let channels = data.channels();
            match padding_color {
                // `out.fill(0)`
                None => Ok(Pixels::Planar(data.filled_like(
                    channels,
                    height as usize,
                    width as usize,
                    0.0,
                ))),
                // `col = np.asarray(padding_color).reshape(-1, 1, 1)` broadcasts one value
                // per channel, and mismatched lengths raise ValueError.
                Some((r, g, b)) => {
                    if channels != 3 {
                        return Err(ApiError::internal(format!(
                            "padding_color channels (3) != region channels ({channels})"
                        )));
                    }
                    Ok(Pixels::Planar(data.filled_per_channel(
                        height as usize,
                        width as usize,
                        &[r as f64, g as f64, b as f64],
                    )))
                }
            }
        }
        Pixels::Raw(_) => Err(ApiError::internal(
            "Cannot build a padded canvas from a raw payload",
        )),
    }
}

fn new_image_like(
    prototype: &DynamicImage,
    width: u32,
    height: u32,
    padding_color: Option<(u8, u8, u8)>,
) -> DynamicImage {
    let (r, g, b) = padding_color.unwrap_or((0, 0, 0));
    match prototype {
        DynamicImage::ImageLuma8(_) => DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
            width,
            height,
            image::Luma([r]),
        )),
        DynamicImage::ImageRgba8(_) => DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            width,
            height,
            // Pillow's `Image.new("RGBA", size, (r, g, b))` fills alpha with 255.
            image::Rgba([r, g, b, 255]),
        )),
        _ => DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([r, g, b]),
        )),
    }
}

fn paste(dst: &mut Pixels, src: &Pixels, dst_x: i64, dst_y: i64) -> Result<()> {
    match (dst, src) {
        (Pixels::Image(d), Pixels::Image(s)) => {
            let mut buf = d.clone();
            image::imageops::replace(&mut buf, s, dst_x, dst_y);
            *d = buf;
            Ok(())
        }
        (Pixels::Planar(d), Pixels::Planar(s)) => {
            d.blit(s, dst_y as usize, dst_x as usize);
            Ok(())
        }
        _ => Err(ApiError::internal(
            "Cannot paste between mismatched pixel representations",
        )),
    }
}

/// `get_extended_region` — `image_utils.py:205`.
///
/// Guarantees a single read for partial overlaps. For a fully out-of-bounds request,
/// `extend = true` returns a padded canvas and `extend = false` raises 416.
#[allow(clippy::too_many_arguments)]
pub async fn get_extended_region(
    slide: &dyn Slide,
    info: &SlideInfo,
    level: usize,
    start_x: i64,
    start_y: i64,
    size_x: i64,
    size_y: i64,
    opts: ReadOptions,
    extend: bool,
) -> Result<Pixels> {
    let hit = region_intersection(info, level, start_x, start_y, size_x, size_y);

    if hit.overlap_w == 0 || hit.overlap_h == 0 {
        if !extend {
            return Err(ApiError::range_not_satisfiable(
                "Requested region is outside the image extent",
            ));
        }
        // Fetch a 1x1 sample so the empty canvas keeps the plugin's layout and dtype.
        let prototype = decode_raw(slide.get_region(level, 0, 0, 1, 1, opts).await?)?;
        return padded_canvas(&prototype, size_x, size_y, opts.padding_color);
    }

    let src = decode_raw(
        slide
            .get_region(
                level,
                hit.src_x,
                hit.src_y,
                hit.overlap_w as u32,
                hit.overlap_h as u32,
                opts,
            )
            .await?,
    )?;

    // Full overlap that exactly matches the request. The endpoint normally guards this,
    // but the check is cheap and avoids a copy.
    if hit.overlap_w == size_x && hit.overlap_h == size_y && start_x >= 0 && start_y >= 0 {
        return Ok(src);
    }

    if !extend {
        // SHRINK: hand back just the real content.
        return Ok(src);
    }

    let mut canvas = padded_canvas(&src, size_x, size_y, opts.padding_color)?;
    paste(&mut canvas, &src, hit.dst_x, hit.dst_y)?;
    Ok(canvas)
}

/// `get_extended_tile` — `image_utils.py:322`.
///
/// Always performs exactly one `get_tile` call, then normalises the result.
pub async fn get_extended_tile(
    slide: &dyn Slide,
    info: &SlideInfo,
    level: usize,
    tile_x: i64,
    tile_y: i64,
    opts: ReadOptions,
    extend: bool,
) -> Result<Pixels> {
    let tile = decode_raw(slide.get_tile(level, tile_x, tile_y, opts).await?)?;

    let tile_extent = info.tile_extent;
    let (overlap_w, overlap_h) = tile_overlap(info, level, tile_x, tile_y);

    // Interior tile: already exactly a full tile.
    if overlap_w == tile_extent.x && overlap_h == tile_extent.y {
        return Ok(tile);
    }

    // Completely outside the level. Plugins return a padded tile here; keep it non-empty
    // so the encoders can handle it.
    if overlap_w == 0 || overlap_h == 0 {
        return Ok(tile);
    }

    if !extend {
        // SHRINK: crop away the padded area.
        return Ok(match tile {
            Pixels::Image(img) => {
                Pixels::Image(img.crop_imm(0, 0, overlap_w as u32, overlap_h as u32))
            }
            Pixels::Planar(data) => {
                Pixels::Planar(data.crop_top_left(overlap_h as usize, overlap_w as usize))
            }
            raw @ Pixels::Raw(_) => raw,
        });
    }

    // EXTEND: copy the valid region into a full-size buffer.
    //
    // Python's fill here is `padding_color if padding_color is not None else 0`, which is
    // black rather than the settings default — the endpoints always resolve a colour
    // before calling, so this only matters for direct library use.
    let cropped = match &tile {
        Pixels::Image(img) => Pixels::Image(img.crop_imm(0, 0, overlap_w as u32, overlap_h as u32)),
        Pixels::Planar(data) => {
            Pixels::Planar(data.crop_top_left(overlap_h as usize, overlap_w as usize))
        }
        Pixels::Raw(_) => return Ok(tile),
    };

    let fill = opts.padding_color.or(Some((0, 0, 0)));
    let mut canvas = padded_canvas(&cropped, tile_extent.x, tile_extent.y, fill)?;
    paste(&mut canvas, &cropped, 0, 0)?;
    Ok(canvas)
}

/// Convenience for plugins that need a zero-filled planar buffer of a known dtype.
pub fn empty_like(data: &PlanarData, channels: usize, height: usize, width: usize) -> PlanarData {
    data.filled_like(channels, height, width, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SlideExtent, SlideLevel, SlidePixelSizeNm};

    fn info(level_w: i64, level_h: i64, tile: i64) -> SlideInfo {
        SlideInfo {
            id: "t".into(),
            channels: None,
            channel_depth: Some(8),
            extent: SlideExtent::new(level_w, level_h, 1),
            num_levels: 1,
            pixel_size_nm: SlidePixelSizeNm::xy(1.0, 1.0),
            tile_extent: SlideExtent::new(tile, tile, 1),
            levels: vec![SlideLevel {
                extent: SlideExtent::new(level_w, level_h, 1),
                downsample_factor: 1.0,
            }],
            format: "t".into(),
            raw_download: false,
        }
    }

    #[test]
    fn complete_overlap_uses_strict_less_than_on_the_far_edge() {
        let i = info(100, 100, 256);
        assert!(check_complete_region_overlap(&i, 0, 0, 0, 99, 99));
        // Ends exactly on the extent -> NOT considered complete, so the padding path runs.
        assert!(!check_complete_region_overlap(&i, 0, 0, 0, 100, 100));
        assert!(!check_complete_region_overlap(&i, 0, -1, 0, 10, 10));
    }

    #[test]
    fn intersection_clamps_and_reports_paste_offsets() {
        let i = info(100, 100, 256);
        let hit = region_intersection(&i, 0, -10, -20, 50, 60);
        assert_eq!(hit.src_x, 0);
        assert_eq!(hit.src_y, 0);
        assert_eq!(hit.overlap_w, 40);
        assert_eq!(hit.overlap_h, 40);
        assert_eq!(hit.dst_x, 10);
        assert_eq!(hit.dst_y, 20);
    }

    #[test]
    fn intersection_reports_no_overlap_when_fully_outside() {
        let i = info(100, 100, 256);
        let hit = region_intersection(&i, 0, 500, 500, 10, 10);
        assert_eq!(hit.overlap_w, 0);
        assert_eq!(hit.overlap_h, 0);
    }

    #[test]
    fn tile_overlap_matches_python() {
        let i = info(300, 300, 256);
        assert_eq!(tile_overlap(&i, 0, 0, 0), (256, 256));
        assert_eq!(tile_overlap(&i, 0, 1, 1), (44, 44));
        assert_eq!(tile_overlap(&i, 0, 2, 2), (0, 0));
    }

    #[test]
    fn complete_tile_overlap_uses_less_or_equal() {
        // Unlike regions, tiles use <=, so a tile ending exactly on the extent is complete.
        let i = info(512, 512, 256);
        assert!(check_complete_tile_overlap(&i, 0, 1, 1));
        assert!(!check_complete_tile_overlap(&i, 0, 2, 0));
    }
}
