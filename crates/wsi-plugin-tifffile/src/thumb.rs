//! `get_thumbnail`'s channel-stack branch.
//!
//! ```python
//! thumbnail_resized = util.img_as_uint(
//!     transform.resize(thumbnail_org, (thumbnail_org.shape[0], max_y, max_x))
//! )
//! ```
//!
//! This is the one resize in the whole service that does **not** go through Pillow, and it
//! is not a resize in Pillow's sense either. `skimage.transform.resize` with its defaults:
//!
//! 1. converts the input to float — `img_as_float`, which divides an integer array by its
//!    dtype's maximum, so a `uint16` page lands in `0..1`;
//! 2. anti-aliases with a Gaussian of `sigma = max(0, (factor - 1) / 2)` per axis, in
//!    `reflect` mode, where `factor` is the input extent over the output extent;
//! 3. interpolates bilinearly on the half-pixel grid, `in = (out + 0.5) * factor - 0.5`,
//!    with mirrored edges — `mode="reflect"` is scipy's `"mirror"`, so the edge sample is
//!    not repeated and index `-1` reads index `1`;
//! 4. clips the result back into the *input's* observed range, not into `0..1`.
//!
//! `img_as_uint` then multiplies by 65535, rounds half to even and clips — and refuses a
//! float image outside `-1..1` outright. So a multichannel thumbnail is always 16-bit,
//! whatever the page held, and a float page whose intensities are not already normalised
//! makes the endpoint fail, as it does in Python.
//!
//! The algorithm is reproduced; exact agreement with scipy's separable filters is not
//! claimed. See `docs/divergences.md`, divergence 29.

use ndarray::Array3;

use wsi_core::error::{ApiError, Result};
use wsi_core::pixels::{Dtype, PlanarData};

/// `img_as_float`'s scale for a dtype: the value that maps to 1.0.
fn float_scale(dtype: Dtype) -> f64 {
    match dtype {
        Dtype::U8 => 255.0,
        Dtype::U16 => 65_535.0,
        Dtype::U32 => 4_294_967_295.0,
        Dtype::U64 => u64::MAX as f64,
        // Signed types map onto -1..1 by their positive maximum.
        Dtype::I8 => 127.0,
        Dtype::I16 => 32_767.0,
        Dtype::I32 => 2_147_483_647.0,
        // Floats are already in range.
        Dtype::F32 | Dtype::F64 => 1.0,
    }
}

/// `img_as_uint(transform.resize(stack, (channels, height, width)))`.
pub fn resize_to_uint16(
    source: &PlanarData,
    out_height: usize,
    out_width: usize,
) -> Result<PlanarData> {
    let (channels, height, width) = source.shape();
    let scale = float_scale(source.dtype());

    // Step 1: to float, per `img_as_float`.
    let mut plane = Array3::<f64>::zeros((channels, height, width));
    let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let value = source.get_f64(c, y, x) / scale;
                plane[[c, y, x]] = value;
                bounds.0 = bounds.0.min(value);
                bounds.1 = bounds.1.max(value);
            }
        }
    }
    if bounds.0 < -1.0 || bounds.1 > 1.0 {
        // `img_as_uint` refuses a float image outside -1..1 rather than rescaling it.
        return Err(ApiError::unhandled(
            "ValueError('Images of type float must be between -1 and 1.')",
        ));
    }

    // Step 2: anti-alias. The channel axis keeps its extent, so its factor is 1 and its
    // sigma 0 — no blurring across channels.
    let factor_y = height as f64 / out_height.max(1) as f64;
    let factor_x = width as f64 / out_width.max(1) as f64;
    let sigma_y = ((factor_y - 1.0) / 2.0).max(0.0);
    let sigma_x = ((factor_x - 1.0) / 2.0).max(0.0);
    if sigma_y > 0.0 {
        plane = blur_axis(&plane, sigma_y, Axis::Rows);
    }
    if sigma_x > 0.0 {
        plane = blur_axis(&plane, sigma_x, Axis::Columns);
    }

    // Steps 3 and 4: bilinear on the half-pixel grid, then `_clip_warp_output` back into
    // the input's own range, then `img_as_uint`'s scale-and-round.
    let mut out = Array3::<u16>::zeros((channels, out_height, out_width));
    for c in 0..channels {
        for y in 0..out_height {
            let source_y = (y as f64 + 0.5) * factor_y - 0.5;
            for x in 0..out_width {
                let source_x = (x as f64 + 0.5) * factor_x - 0.5;
                let value = bilinear(&plane, c, source_y, source_x).clamp(bounds.0, bounds.1);
                // `np.rint` rounds halves to even.
                let scaled = (value * 65_535.0).round_ties_even();
                out[[c, y, x]] = scaled.clamp(0.0, 65_535.0) as u16;
            }
        }
    }
    Ok(PlanarData::U16(out))
}

enum Axis {
    Rows,
    Columns,
}

/// A one-dimensional Gaussian convolution along one axis, with mirrored edges.
///
/// `scipy.ndimage.gaussian_filter` truncates the kernel at four standard deviations and
/// normalises what remains, which is what this does.
fn blur_axis(source: &Array3<f64>, sigma: f64, axis: Axis) -> Array3<f64> {
    let radius = (4.0 * sigma + 0.5) as usize;
    if radius == 0 {
        return source.clone();
    }
    let kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let offset = i as f64 - radius as f64;
            (-(offset * offset) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|v| v / total).collect();

    let (channels, height, width) = source.dim();
    let mut out = Array3::<f64>::zeros((channels, height, width));
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (i, weight) in kernel.iter().enumerate() {
                    let offset = i as isize - radius as isize;
                    let (sy, sx) = match axis {
                        Axis::Rows => (mirror(y as isize + offset, height), x),
                        Axis::Columns => (y, mirror(x as isize + offset, width)),
                    };
                    sum += weight * source[[c, sy, sx]];
                }
                out[[c, y, x]] = sum;
            }
        }
    }
    out
}

/// `mode="mirror"`: the edge sample is not repeated, so `-1` reads index 1.
fn mirror(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let mut wrapped = index.rem_euclid(period);
    if wrapped >= length as isize {
        wrapped = period - wrapped;
    }
    wrapped as usize
}

fn bilinear(source: &Array3<f64>, channel: usize, y: f64, x: f64) -> f64 {
    let (_, height, width) = source.dim();
    let y0 = y.floor();
    let x0 = x.floor();
    let (fy, fx) = (y - y0, x - x0);
    let (y0, x0) = (y0 as isize, x0 as isize);

    let mut value = 0.0;
    for (dy, wy) in [(0isize, 1.0 - fy), (1, fy)] {
        if wy == 0.0 {
            continue;
        }
        let sy = mirror(y0 + dy, height);
        for (dx, wx) in [(0isize, 1.0 - fx), (1, fx)] {
            if wx == 0.0 {
                continue;
            }
            let sx = mirror(x0 + dx, width);
            value += wy * wx * source[[channel, sy, sx]];
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrored_indices_do_not_repeat_the_edge() {
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(-2, 5), 2);
        assert_eq!(mirror(5, 5), 3);
        assert_eq!(mirror(0, 5), 0);
        assert_eq!(mirror(4, 5), 4);
        assert_eq!(mirror(3, 1), 0);
    }

    #[test]
    fn a_constant_stack_survives_the_resize_unchanged() {
        let source = PlanarData::U8(Array3::from_elem((2, 8, 8), 128u8));
        let resized = resize_to_uint16(&source, 4, 4).unwrap();
        assert_eq!(resized.shape(), (2, 4, 4));
        // 128/255 scaled onto 0..65535.
        let expected: f64 = (128.0f64 / 255.0 * 65_535.0).round();
        for c in 0..2 {
            for y in 0..4 {
                for x in 0..4 {
                    assert!(
                        (resized.get_f64(c, y, x) - expected).abs() <= 1.0,
                        "at ({x}, {y}, {c}): {}",
                        resized.get_f64(c, y, x)
                    );
                }
            }
        }
    }

    #[test]
    fn upscaling_interpolates_on_the_half_pixel_grid_with_mirrored_edges() {
        // Two pixels, 0 and 255, widened to four. The grid maps output x to
        // `(x + 0.5) * 0.5 - 0.5`, i.e. -0.25, 0.25, 0.75, 1.25 — so the first and last
        // samples read *past* the edge, where `mirror` sends them back to the other pixel.
        let source = PlanarData::U8(ndarray::arr3(&[[[0u8, 255]]]));
        let resized = resize_to_uint16(&source, 1, 4).unwrap();
        let values: Vec<f64> = (0..4).map(|x| resized.get_f64(0, 0, x)).collect();
        let full = 65_535.0;
        // The samples land at -0.25, 0.25, 0.75 and 1.25. The two that fall outside pick
        // up their neighbour through the mirror, which makes the result symmetric rather
        // than a straight ramp from 0 to 255.
        for (index, expected) in [0.25, 0.25, 0.75, 0.75].iter().enumerate() {
            assert!((values[index] - expected * full).abs() <= 1.0, "{values:?}");
        }
    }

    #[test]
    fn a_float_stack_outside_the_unit_range_is_refused() {
        // `img_as_uint` raises rather than rescaling, so the endpoint fails.
        let source = PlanarData::F32(Array3::from_elem((1, 2, 2), 42.0f32));
        assert!(resize_to_uint16(&source, 1, 1).is_err());
    }

    #[test]
    fn the_output_is_always_sixteen_bit() {
        let source = PlanarData::F32(Array3::from_elem((1, 4, 4), 0.5f32));
        let resized = resize_to_uint16(&source, 2, 2).unwrap();
        assert_eq!(resized.dtype(), Dtype::U16);
        assert!((resized.get_f64(0, 0, 0) - 32_768.0).abs() <= 1.0);
    }
}
