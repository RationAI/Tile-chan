//! `__assemble_region` and the two page readers underneath it.
//!
//! A note on the Python this ports, because it is confusing to read: the call
//!
//! ```python
//! self.__read_region_of_page(page, 0, start_y, start_x, size_y, size_x, padding_color)
//! ```
//!
//! passes `start_y` into a parameter *named* `start_x`, and the swap holds all the way
//! down. The functions are therefore row-major throughout — the parameter called `start_x`
//! is a row and the one called `size_x` a row count — and everything lines up. The names
//! here say `row` and `column` instead.
//!
//! Two behaviours that look like bugs and are not fixed, because a client sees them:
//!
//! - A tiled read is assembled out of **whole tiles** and then sliced, so a region that
//!   reaches past the level's right or bottom edge comes back *smaller than requested*
//!   rather than padded, and the caller's `get_extended_region` pads it.
//! - The part of an edge tile that lies outside the image is whatever the writer stored
//!   there, not the padding colour.

use std::collections::BTreeSet;

use ndarray::Array3;

use wsi_core::error::{ApiError, Result};
use wsi_core::pixels::PlanarData;

use crate::layout::{Layout, LayoutInfo};
use crate::tiff::page::{Page, Raster, SampleDtype};
use crate::tiff::TiffFile;

/// `__get_color_for_channel`.
///
/// Above 8 bits the padding is black: channel values are mapped to RGB by stretching
/// between the observed minimum and maximum, so a colour could not survive the round trip.
pub fn color_for_channel(
    channel_index: usize,
    channel_depth: i32,
    padding_color: (u8, u8, u8),
    dtype: SampleDtype,
) -> f64 {
    let value = if channel_depth == 8 {
        let (r, g, b) = padding_color;
        f64::from(match channel_index {
            0 => r,
            1 => g,
            _ => b,
        })
    } else {
        0.0
    };
    // Signed 8-bit pages cannot hold 255, and numpy raises rather than wrapping.
    match dtype.integer_bounds() {
        Some((low, high)) => value.clamp(low, high),
        None => value,
    }
}

/// The region a read produced, before it becomes a channel stack.
struct PageRegion {
    raster: Raster,
}

/// `__assemble_region`.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    file: &mut TiffFile,
    level_index: usize,
    layout: &LayoutInfo,
    channel_depth: i32,
    start_x: i64,
    start_y: i64,
    size_x: usize,
    size_y: usize,
    padding_color: (u8, u8, u8),
    z: usize,
) -> Result<PlanarData> {
    match layout.mode {
        Layout::ChunkySamples => {
            let index = layout.page_index(z, 0)?;
            let page = page_at(file, level_index, index)?;
            if !page.is_tiled() {
                return chunky_untiled(
                    file,
                    &page,
                    channel_depth,
                    start_x,
                    start_y,
                    size_x,
                    size_y,
                    padding_color,
                );
            }
            // The tiled path reads rows first, so the arguments swap.
            let region = read_region_of_page(
                file,
                &page,
                0,
                start_y,
                start_x,
                size_y,
                size_x,
                padding_color,
                channel_depth,
            )?;
            // `(H, W, S)` becomes `(S, H, W)`.
            let samples = region.raster.samples;
            Ok(stack(
                &[(&region.raster, 0)],
                samples,
                region.raster.height,
                region.raster.width,
                true,
            ))
        }
        Layout::SingleChannel | Layout::PagePerZ => {
            let index = layout.page_index(z, 0)?;
            let page = page_at(file, level_index, index)?;
            let region = read_region_of_page(
                file,
                &page,
                0,
                start_y,
                start_x,
                size_y,
                size_x,
                padding_color,
                channel_depth,
            )?;
            // Drop the trailing sample axis and keep the single leading plane.
            Ok(stack(
                &[(&region.raster, 0)],
                1,
                region.raster.height,
                region.raster.width,
                false,
            ))
        }
        Layout::PagePerChannel | Layout::PagePerZChannel => {
            let mut regions = Vec::with_capacity(layout.num_channels);
            for channel_index in 0..layout.num_channels {
                let index = layout.page_index(z, channel_index)?;
                let page = page_at(file, level_index, index)?;
                regions.push(read_region_of_page(
                    file,
                    &page,
                    channel_index,
                    start_y,
                    start_x,
                    size_y,
                    size_x,
                    padding_color,
                    channel_depth,
                )?);
            }
            let (height, width) = regions
                .first()
                .map(|r| (r.raster.height, r.raster.width))
                .unwrap_or((0, 0));
            if regions
                .iter()
                .any(|r| r.raster.height != height || r.raster.width != width)
            {
                // `np.concatenate` refuses a ragged stack.
                return Err(ApiError::unhandled(
                    "ValueError('all the input array dimensions except for the concatenation axis must match exactly')",
                ));
            }
            let sources: Vec<(&Raster, usize)> =
                regions.iter().map(|r| (&r.raster, 0usize)).collect();
            Ok(stack(&sources, regions.len(), height, width, false))
        }
    }
}

fn page_at(file: &TiffFile, level_index: usize, page_index: usize) -> Result<Page> {
    file.series
        .levels
        .get(level_index)
        .and_then(|level| level.pages.get(page_index))
        .cloned()
        .ok_or_else(|| {
            ApiError::unhandled(format!(
                "IndexError('list index out of range') for page {page_index}"
            ))
        })
}

/// The chunky branch for an untiled page, which reads the whole page and crops.
///
/// Python special-cases this because `__read_region_of_page_untiled` allocates a 2-D
/// output and would fail to take the `(H, W, S)` slice.
#[allow(clippy::too_many_arguments)]
fn chunky_untiled(
    file: &mut TiffFile,
    page: &Page,
    channel_depth: i32,
    start_x: i64,
    start_y: i64,
    size_x: usize,
    size_y: usize,
    padding_color: (u8, u8, u8),
) -> Result<PlanarData> {
    let page_width = page.image_width as i64;
    let page_height = page.image_length as i64;
    require_non_negative(start_x, start_y)?;

    let page_array = read_full(file, page)?;
    let new_height = if start_y + size_y as i64 > page_height {
        page_height - start_y
    } else {
        size_y as i64
    };
    let new_width = if start_x + size_x as i64 > page_width {
        page_width - start_x
    } else {
        size_x as i64
    };

    let samples = page.samples_per_pixel;
    let mut out = Raster::filled(size_y, size_x, samples, page.dtype, 0.0);
    for sample in 0..samples {
        let value = color_for_channel(sample, channel_depth, padding_color, page.dtype);
        fill_sample(&mut out, sample, value);
    }
    if new_height > 0 && new_width > 0 {
        out.blit(
            &page_array,
            0,
            0,
            start_y as usize,
            start_x as usize,
            new_height as usize,
            new_width as usize,
            0,
        );
    }
    Ok(stack(&[(&out, 0)], samples, out.height, out.width, true))
}

/// `__read_region_of_page`: dispatch, then substitute a padding-only region when the read
/// found nothing.
#[allow(clippy::too_many_arguments)]
fn read_region_of_page(
    file: &mut TiffFile,
    page: &Page,
    channel_index: usize,
    row0: i64,
    column0: i64,
    rows: usize,
    columns: usize,
    padding_color: (u8, u8, u8),
    channel_depth: i32,
) -> Result<PageRegion> {
    let fill = color_for_channel(channel_index, channel_depth, padding_color, page.dtype);
    let raster = if !page.is_tiled() {
        match read_untiled(file, page, row0, column0, rows, columns, fill)? {
            Some(raster) => raster,
            None => Raster::filled(rows, columns, 1, page.dtype, fill),
        }
    } else {
        match read_tiled(file, page, row0, column0, rows, columns, fill)? {
            Some(raster) => raster,
            None => Raster::filled(rows, columns, page.samples_per_pixel, page.dtype, fill),
        }
    };
    Ok(PageRegion { raster })
}

/// `__read_region_of_page_untiled`.
fn read_untiled(
    file: &mut TiffFile,
    page: &Page,
    row0: i64,
    column0: i64,
    rows: usize,
    columns: usize,
    fill: f64,
) -> Result<Option<Raster>> {
    let page_width = page.image_width as i64;
    let page_height = page.image_length as i64;
    if row0 >= page_height
        || column0 >= page_width
        || row0 + rows as i64 <= 0
        || column0 + columns as i64 <= 0
    {
        return Ok(None);
    }
    require_non_negative(column0, row0)?;

    let page_array = read_full(file, page)?;
    let new_height = if row0 + rows as i64 > page_height {
        page_height - row0
    } else {
        rows as i64
    };
    let new_width = if column0 + columns as i64 > page_width {
        page_width - column0
    } else {
        columns as i64
    };

    let mut out = Raster::filled(rows, columns, 1, page.dtype, fill);
    out.blit(
        &page_array,
        0,
        0,
        row0 as usize,
        column0 as usize,
        new_height.max(0) as usize,
        new_width.max(0) as usize,
        0,
    );
    Ok(Some(out))
}

/// `__read_region_of_page_tiled`.
#[allow(clippy::too_many_arguments)]
fn read_tiled(
    file: &mut TiffFile,
    page: &Page,
    row0: i64,
    column0: i64,
    rows: usize,
    columns: usize,
    fill: f64,
) -> Result<Option<Raster>> {
    let image_width = page.image_width as i64;
    let image_height = page.image_length as i64;
    if row0 >= image_height
        || column0 >= image_width
        || row0 + rows as i64 <= 0
        || column0 + columns as i64 <= 0
    {
        return Ok(None);
    }

    let tile_width = page.tile_width as i64;
    let tile_height = page.tile_length as i64;
    let end_row = (row0 + rows as i64).min(image_height);
    let end_column = (column0 + columns as i64).min(image_width);

    // Floor division, so a negative origin lands on a negative tile index exactly as
    // Python's `//` does.
    let first_tile_row = row0.div_euclid(tile_height);
    let first_tile_column = column0.div_euclid(tile_width);
    let last_tile_row = ceil_div(end_row, tile_height);
    let last_tile_column = ceil_div(end_column, tile_width);

    let tiles_per_line = ceil_div(image_width, tile_width);
    let tiles_per_plane = tiles_per_line * ceil_div(image_height, tile_height);
    let separate = page.separate_planes();
    let planes = if separate { page.samples_per_pixel } else { 1 };

    let out_height = ((last_tile_row - first_tile_row) * tile_height).max(0) as usize;
    let out_width = ((last_tile_column - first_tile_column) * tile_width).max(0) as usize;
    // A wildly negative origin would ask for an enormous canvas; refuse rather than
    // attempt the allocation Python would also fail on.
    let footprint = out_height
        .saturating_mul(out_width)
        .saturating_mul(page.samples_per_pixel)
        .saturating_mul(page.dtype.size());
    if footprint > 1 << 30 {
        return Err(ApiError::unhandled(
            "MemoryError('Unable to allocate array for the requested region')",
        ));
    }

    let mut out = Raster::filled(
        out_height,
        out_width,
        page.samples_per_pixel,
        page.dtype,
        fill,
    );

    for plane in 0..planes {
        let mut used_offsets = BTreeSet::new();
        for tile_row in first_tile_row..last_tile_row {
            for tile_column in first_tile_column..last_tile_column {
                let index =
                    plane as i64 * tiles_per_plane + tile_row * tiles_per_line + tile_column;
                let Some(index) = resolve_index(index, page.data_offsets.len()) else {
                    continue;
                };
                let offset = page.data_offsets[index];
                if !used_offsets.insert(offset) {
                    continue;
                }
                let Some(tile) = page
                    .read_chunk(&mut file.source, index)
                    .map_err(|e| ApiError::unhandled(format!("TiffFileError({e:?})")))?
                else {
                    continue;
                };
                let dst_row = ((tile_row - first_tile_row) * tile_height) as usize;
                let dst_column = ((tile_column - first_tile_column) * tile_width) as usize;
                out.blit(
                    &tile,
                    dst_row,
                    dst_column,
                    0,
                    0,
                    page.tile_length,
                    page.tile_width,
                    plane,
                );
            }
        }
    }

    // Restrict to the requested region — which can be shorter than asked for, because
    // `out` only spans the tiles that exist.
    let new_row0 = (row0 - first_tile_row * tile_height).max(0) as usize;
    let new_column0 = (column0 - first_tile_column * tile_width).max(0) as usize;
    let height = rows.min(out.height.saturating_sub(new_row0));
    let width = columns.min(out.width.saturating_sub(new_column0));
    let mut result = Raster::filled(height, width, out.samples, out.dtype, 0.0);
    result.blit(&out, 0, 0, new_row0, new_column0, height, width, 0);
    Ok(Some(result))
}

/// `np.ceil(a / b)` for positive `b`, which is what the tile-grid arithmetic needs.
///
/// `i64::div_ceil` is still unstable, and the signed rounding it would give differs from
/// Python's float `ceil` for negative numerators anyway.
fn ceil_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let quotient = a.div_euclid(b);
    if a.rem_euclid(b) == 0 {
        quotient
    } else {
        quotient + 1
    }
}

/// Python's negative list indexing, then the `len(dataoffsets) <= index` skip.
fn resolve_index(index: i64, len: usize) -> Option<usize> {
    let resolved = if index < 0 { len as i64 + index } else { index };
    if resolved < 0 || resolved >= len as i64 {
        return None;
    }
    Some(resolved as usize)
}

/// A negative origin makes numpy slice from the end, and the assignment that follows then
/// fails on a shape mismatch. Python answers 500; so does this.
fn require_non_negative(column0: i64, row0: i64) -> Result<()> {
    if column0 < 0 || row0 < 0 {
        return Err(ApiError::unhandled(
            "ValueError('could not broadcast input array into the output shape')",
        ));
    }
    Ok(())
}

fn read_full(file: &mut TiffFile, page: &Page) -> Result<Raster> {
    page.read_full(&mut file.source)
        .map_err(|e| ApiError::unhandled(format!("TiffFileError({e:?})")))
}

fn fill_sample(raster: &mut Raster, sample: usize, value: f64) {
    let size = raster.dtype.size();
    let samples = raster.samples;
    let mut encoded = vec![0u8; size];
    raster.dtype.fill(&mut encoded, value);
    for pixel in 0..raster.height * raster.width {
        let at = (pixel * samples + sample) * size;
        raster.bytes[at..at + size].copy_from_slice(&encoded);
    }
}

/// Builds the `(channels, height, width)` stack the response pipeline expects.
///
/// `by_sample` selects between "one channel per source raster, sample 0" and "one channel
/// per sample of a single raster".
fn stack(
    sources: &[(&Raster, usize)],
    channels: usize,
    height: usize,
    width: usize,
    by_sample: bool,
) -> PlanarData {
    let dtype = sources
        .first()
        .map(|(raster, _)| raster.dtype)
        .unwrap_or(SampleDtype::U8);

    macro_rules! build {
        ($ty:ty, $n:literal, $variant:path) => {{
            let mut out = Array3::<$ty>::zeros((channels, height, width));
            for channel in 0..channels {
                let (raster, sample) = if by_sample {
                    (sources[0].0, channel)
                } else {
                    let (raster, sample) = sources[channel.min(sources.len() - 1)];
                    (raster, sample)
                };
                if sample >= raster.samples {
                    continue;
                }
                for y in 0..height.min(raster.height) {
                    for x in 0..width.min(raster.width) {
                        let mut buf = [0u8; $n];
                        buf.copy_from_slice(raster.sample_bytes(y, x, sample));
                        out[[channel, y, x]] = <$ty>::from_ne_bytes(buf);
                    }
                }
            }
            $variant(out)
        }};
    }

    match dtype {
        SampleDtype::U8 => build!(u8, 1, PlanarData::U8),
        SampleDtype::I8 => build!(i8, 1, PlanarData::I8),
        SampleDtype::U16 => build!(u16, 2, PlanarData::U16),
        SampleDtype::I16 => build!(i16, 2, PlanarData::I16),
        SampleDtype::U32 => build!(u32, 4, PlanarData::U32),
        SampleDtype::I32 => build!(i32, 4, PlanarData::I32),
        SampleDtype::U64 => build!(u64, 8, PlanarData::U64),
        SampleDtype::I64 => {
            // `Pixels`/`PlanarData` has no 64-bit signed variant, because
            // `convert_narray_to_pil_image` has no branch for it either; widening to f64
            // keeps the values and reaches the same 400.
            let mut out = Array3::<f64>::zeros((channels, height, width));
            for channel in 0..channels {
                let (raster, sample) = if by_sample {
                    (sources[0].0, channel)
                } else {
                    sources[channel.min(sources.len() - 1)]
                };
                for y in 0..height.min(raster.height) {
                    for x in 0..width.min(raster.width) {
                        out[[channel, y, x]] = raster.get(y, x, sample);
                    }
                }
            }
            PlanarData::F64(out)
        }
        SampleDtype::F32 => build!(f32, 4, PlanarData::F32),
        SampleDtype::F64 => build!(f64, 8, PlanarData::F64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_bit_padding_uses_the_requested_colour_per_channel() {
        let colour = (10, 20, 30);
        assert_eq!(color_for_channel(0, 8, colour, SampleDtype::U8), 10.0);
        assert_eq!(color_for_channel(1, 8, colour, SampleDtype::U8), 20.0);
        // Channel 2 and beyond all take the blue component.
        assert_eq!(color_for_channel(2, 8, colour, SampleDtype::U8), 30.0);
        assert_eq!(color_for_channel(7, 8, colour, SampleDtype::U8), 30.0);
    }

    #[test]
    fn deeper_pages_pad_black_because_the_colour_could_not_survive_the_mapping() {
        assert_eq!(
            color_for_channel(0, 16, (255, 255, 255), SampleDtype::U16),
            0.0
        );
    }

    #[test]
    fn padding_clamps_into_a_signed_page() {
        // 255 does not fit int8, and numpy raises instead of wrapping.
        assert_eq!(
            color_for_channel(0, 8, (255, 255, 255), SampleDtype::I8),
            127.0
        );
        assert_eq!(
            color_for_channel(0, 8, (255, 255, 255), SampleDtype::F32),
            255.0
        );
    }

    #[test]
    fn negative_indices_wrap_the_way_python_lists_do() {
        assert_eq!(resolve_index(2, 6), Some(2));
        assert_eq!(resolve_index(-1, 6), Some(5));
        assert_eq!(resolve_index(-7, 6), None);
        assert_eq!(resolve_index(6, 6), None);
    }
}
