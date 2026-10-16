//! `__inspect_series_layout` and `__get_series_page_index`.
//!
//! The series axes say how the channel and Z dimensions are spread over pages, and every
//! read depends on getting that mapping right. Five layouts cover it:
//!
//! | layout | how the data is stored |
//! |---|---|
//! | `chunky-samples`     | one page, all channels in `SamplesPerPixel` |
//! | `page-per-channel`   | one page per channel |
//! | `page-per-z`         | one page per Z plane, one channel |
//! | `page-per-z-channel` | one page per (Z, channel) pair |
//! | `single-channel`     | one page, one channel |

use wsi_core::error::{ApiError, Result};

use crate::tiff::series::Series;

/// The five page layouts, named as Python names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    ChunkySamples,
    PagePerChannel,
    PagePerZ,
    PagePerZChannel,
    SingleChannel,
}

/// `tifffile.PHOTOMETRIC` values that denote a directly displayable colour image.
pub const RGB_PHOTOMETRICS: [u16; 2] = [2, 6];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInfo {
    pub mode: Layout,
    /// The axes that select a page, in the order they appear in `series.axes`.
    pub page_axes: Vec<char>,
    pub page_dims: Vec<usize>,
    pub num_channels: usize,
    pub num_z: usize,
    /// Plain RGB brightfield: served as a Pillow image rather than a channel stack.
    pub is_rgb: bool,
}

/// `__inspect_series_layout`.
pub fn inspect(series: &Series) -> Result<LayoutInfo> {
    let keyframe = series.keyframe();
    let samples = keyframe.samples_per_pixel.max(1);
    let page_count = series.pages().len();

    let mut page_axes: Vec<char> = Vec::new();
    let mut page_dims: Vec<usize> = Vec::new();
    let mut num_channels = if samples > 1 { samples } else { 1 };
    let mut num_z = 1usize;
    let mut unsupported: Vec<(char, usize)> = Vec::new();

    for (index, axis) in series.axes.chars().enumerate() {
        let axis_size = series.shape.get(index).copied().unwrap_or(1);
        match axis {
            'X' | 'Y' => continue,
            'Z' => {
                num_z = axis_size;
                if axis_size > 1 {
                    page_axes.push(axis);
                    page_dims.push(axis_size);
                }
            }
            'C' | 'S' => {
                num_channels = axis_size;
                // A chunky page already carries its channels in the samples, so the axis
                // does not select a page.
                if samples == 1 && axis_size > 1 {
                    page_axes.push(axis);
                    page_dims.push(axis_size);
                }
            }
            other => {
                if axis_size > 1 {
                    unsupported.push((other, axis_size));
                }
            }
        }
    }

    if !unsupported.is_empty() {
        // Python formats a list of tuples, which is what a client sees.
        let rendered = unsupported
            .iter()
            .map(|(axis, size)| format!("('{axis}', {size})"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ApiError::bad_request(format!(
            "Unsupported TIFF axes for tifffile plugin: [{rendered}]"
        )));
    }

    let expected_pages: usize = if page_dims.is_empty() {
        1
    } else {
        page_dims.iter().product()
    };
    if page_count < expected_pages {
        return Err(ApiError::bad_request(format!(
            "Unexpected TIFF page layout. Expected at least {expected_pages} pages, got {page_count}."
        )));
    }

    let has_channel_axis = page_axes.iter().any(|axis| matches!(axis, 'C' | 'S'));
    let has_z_axis = page_axes.contains(&'Z');
    let mode = if samples > 1 {
        Layout::ChunkySamples
    } else if has_z_axis && has_channel_axis {
        Layout::PagePerZChannel
    } else if has_channel_axis {
        Layout::PagePerChannel
    } else if has_z_axis {
        Layout::PagePerZ
    } else {
        Layout::SingleChannel
    };

    // Restricted to exactly three samples because `convert_narray_to_pil_image` only
    // builds mode "RGB".
    let is_rgb = mode == Layout::ChunkySamples
        && samples == 3
        && num_z == 1
        && RGB_PHOTOMETRICS.contains(&keyframe.photometric);

    Ok(LayoutInfo {
        mode,
        page_axes,
        page_dims,
        num_channels,
        num_z,
        is_rgb,
    })
}

impl LayoutInfo {
    /// `__get_series_page_index` — `np.ravel_multi_index` over the page-selecting axes.
    pub fn page_index(&self, z: usize, channel: usize) -> Result<usize> {
        if self.page_axes.is_empty() {
            return Ok(0);
        }
        let mut indices = Vec::with_capacity(self.page_axes.len());
        for (axis, &size) in self.page_axes.iter().zip(self.page_dims.iter()) {
            let selected = match axis {
                'Z' => z,
                'C' | 'S' => channel,
                _ => 0,
            };
            if selected >= size {
                return Err(ApiError::unprocessable(format!(
                    "Requested axis index {selected} exceeds available TIFF axis size {size}."
                )));
            }
            indices.push(selected);
        }
        // C-order ravel: each index is scaled by the product of the dimensions after it.
        let mut flat = 0usize;
        for (position, index) in indices.iter().enumerate() {
            let stride: usize = self.page_dims[position + 1..].iter().product();
            flat += index * stride;
        }
        Ok(flat)
    }
}

/// `_has_nontrivial_axis` — used by `is_supported` to spot an RGB Z-stack.
pub fn has_nontrivial_axis(series: &Series, axis: char) -> bool {
    match series.axes.find(axis) {
        Some(index) => series.shape.get(index).copied().unwrap_or(1) > 1,
        None => false,
    }
}

/// `_has_supported_axis_layout`.
///
/// Note that both `C`/`S`/`Z` branches return `true` immediately, so the first
/// non-trivial one decides; an axis outside `XYCSZ` with size above 1 rejects the file.
///
/// Divergence from Python, which ends on `samples > 1` and so declines a plain
/// single-channel `YX` page. That page is a layout this plugin reads at any bit depth,
/// and declining it leaves a 16-bit or float greyscale TIFF with no plugin that can open
/// it at all. See `docs/divergences.md`, divergence 39.
pub fn has_supported_axis_layout(series: &Series, _samples: usize) -> bool {
    for (index, axis) in series.axes.chars().enumerate() {
        if matches!(axis, 'X' | 'Y') {
            continue;
        }
        let size = series.shape.get(index).copied().unwrap_or(1);
        if matches!(axis, 'C' | 'S' | 'Z') && size > 1 {
            return true;
        }
        if !matches!(axis, 'X' | 'Y' | 'C' | 'S' | 'Z') && size > 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(page_axes: &[char], page_dims: &[usize]) -> LayoutInfo {
        LayoutInfo {
            mode: Layout::PagePerZChannel,
            page_axes: page_axes.to_vec(),
            page_dims: page_dims.to_vec(),
            num_channels: 1,
            num_z: 1,
            is_rgb: false,
        }
    }

    #[test]
    fn page_index_ravels_in_c_order() {
        // ZCYX with Z=3, C=2: page = z * 2 + channel, which is ImageJ's own page order.
        let info = layout(&['Z', 'C'], &[3, 2]);
        assert_eq!(info.page_index(0, 0).unwrap(), 0);
        assert_eq!(info.page_index(0, 1).unwrap(), 1);
        assert_eq!(info.page_index(1, 0).unwrap(), 2);
        assert_eq!(info.page_index(2, 1).unwrap(), 5);
    }

    #[test]
    fn page_index_of_a_single_page_series_is_zero() {
        let info = layout(&[], &[]);
        assert_eq!(info.page_index(7, 9).unwrap(), 0);
    }

    #[test]
    fn an_out_of_range_axis_index_is_a_422() {
        let info = layout(&['Z'], &[3]);
        let error = info.page_index(3, 0).unwrap_err();
        assert_eq!(error.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error.detail_text(),
            "Requested axis index 3 exceeds available TIFF axis size 3."
        );
    }

    #[test]
    fn a_channel_axis_alone_indexes_by_channel() {
        let info = layout(&['C'], &[3]);
        assert_eq!(info.page_index(0, 2).unwrap(), 2);
    }
}
