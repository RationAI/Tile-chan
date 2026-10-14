//! Numeric conversions, ported from `wsi_service/utils/image_utils.py`.
//!
//! This is the part of the port where a "reasonable" rewrite silently changes output
//! bytes, so every function below documents the numpy or Pillow behaviour it is standing
//! in for. Two rules dominate:
//!
//! 1. **`ndarray.astype(np.uint8)` truncates toward zero.** It does not round. Every
//!    conversion here therefore clamps in `f64` and casts with `as u8`.
//! 2. **Pillow's matrix `convert` adds `0.5` before truncating** (`Convert.c`,
//!    `ImagingConvertMatrix`), which is round-half-up, and clamps with `CLIP8`.
//!
//! Where the Python has surprising behaviour that callers depend on, it is reproduced and
//! called out in a comment rather than quietly corrected — see `docs/divergences.md`.

use image::{DynamicImage, GrayImage, Rgb, RgbImage};
use ndarray::Array3;

use crate::error::{ApiError, Result};
use crate::pixels::{Dtype, PlanarData};

/// Pillow's `MULDIV255`: `tmp = a*b + 128; ((tmp >> 8) + tmp) >> 8`.
///
/// This is the rounded `a*b/255` used by every alpha-blending path in Pillow. Writing it
/// as `(a * b + 127) / 255` would differ for some inputs, so the exact integer sequence is
/// reproduced.
#[inline]
fn muldiv255(a: u32, b: u32) -> u32 {
    let tmp = a * b + 128;
    ((tmp >> 8) + tmp) >> 8
}

/// Pillow's `BLEND(mask, bg, fg)` from `Paste.c`.
#[inline]
fn blend(mask: u8, bg: u8, fg: u8) -> u8 {
    let m = mask as u32;
    (muldiv255(bg as u32, 255 - m) + muldiv255(fg as u32, m)) as u8
}

/// `rgba_to_rgb_with_background_color` — `image_utils.py:11`.
///
/// Python checks `image_rgba.info.get("transparency")` as well as `mode == "RGBA"`. The
/// `transparency` key only appears on palette and greyscale PNGs, and the one plugin that
/// can produce those (`pil`) converts to RGB while opening, so an RGBA check is
/// equivalent here.
pub fn rgba_to_rgb_with_background_color(
    image: DynamicImage,
    padding_color: (u8, u8, u8),
) -> DynamicImage {
    match image {
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = rgba.dimensions();
            let mut out = RgbImage::from_pixel(
                w,
                h,
                Rgb([padding_color.0, padding_color.1, padding_color.2]),
            );
            for (x, y, px) in rgba.enumerate_pixels() {
                let [r, g, b, a] = px.0;
                let bg = out.get_pixel(x, y).0;
                out.put_pixel(
                    x,
                    y,
                    Rgb([blend(a, bg[0], r), blend(a, bg[1], g), blend(a, bg[2], b)]),
                );
            }
            DynamicImage::ImageRgb8(out)
        }
        // `Image.convert("RGB")` for everything else.
        other => DynamicImage::ImageRgb8(other.to_rgb8()),
    }
}

/// `convert_narray_uintX_to_uint8` — `image_utils.py:20`.
///
/// Returns the per-sample mapping applied to a planar array. `exp` must be one of
/// 8/16/32/64; `lower`/`upper` must lie in `[0, 2^exp)` when given.
///
/// Two behaviours are easy to lose in translation and are preserved here:
///
/// * the early return is `if not lower and not upper and exp == 8`, and Python's `not`
///   is true for both `None` **and** `0`. An explicit `lower=0, upper=0` on an 8-bit
///   array therefore returns the array untouched rather than dividing by zero.
/// * when `upper` is unset and `exp > 8`, the default is `2^exp / (exp/4)` — 16384 for
///   16-bit, not 65535. That is a deliberate contrast stretch, not a normalisation.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub lower: f64,
    pub upper: f64,
    /// True when Python would have returned the array unchanged.
    pub identity: bool,
}

pub fn resolve_window(exp: u32, lower: Option<f64>, upper: Option<f64>) -> Result<Window> {
    if !matches!(exp, 8 | 16 | 32 | 64) {
        // Python raises ValueError here; it escapes as a 500 through FastAPI.
        return Err(ApiError::internal(
            "Only exponent in range [8, 16, 32, 64] supported",
        ));
    }
    let limit = 2f64.powi(exp as i32);
    if let Some(l) = lower {
        if !(0.0..limit).contains(&l) {
            return Err(ApiError::internal(format!(
                "lower bound must be between 0 and 2**{exp}"
            )));
        }
    }
    if let Some(u) = upper {
        if !(0.0..limit).contains(&u) {
            return Err(ApiError::internal(format!(
                "upper bound must be between 0 and 2**{exp}"
            )));
        }
    }

    // `not lower and not upper` — falsy covers None and 0.0.
    let lower_falsy = lower.map_or(true, |l| l == 0.0);
    let upper_falsy = upper.map_or(true, |u| u == 0.0);
    if lower_falsy && upper_falsy && exp == 8 {
        return Ok(Window {
            lower: 0.0,
            upper: 0.0,
            identity: true,
        });
    }

    let lower = lower.unwrap_or(0.0);
    let upper = upper.unwrap_or_else(|| {
        if exp > 8 {
            // Default upper bound for bitness > 8, to enhance contrast/brightness.
            limit / (exp as f64 / 4.0)
        } else {
            limit - 1.0
        }
    });

    Ok(Window {
        lower,
        upper,
        identity: false,
    })
}

/// Applies a resolved window to one sample, reproducing
/// `clip(((v - lower) / (upper - lower)) * 255, 0, 255).astype(np.uint8)`.
///
/// When `upper == lower` numpy divides by zero: `0/0` is `nan`, and `nan.astype(uint8)`
/// is 0 on every platform this runs on. That case is reachable for a uniform region, so
/// it is handled explicitly rather than left to Rust's `as` conversion.
#[inline]
pub fn apply_window(value: f64, window: Window) -> u8 {
    let span = window.upper - window.lower;
    let scaled = if span == 0.0 {
        let numerator = value - window.lower;
        if numerator == 0.0 {
            // 0/0 -> nan -> 0
            return 0;
        }
        // x/0 -> +/-inf, clipped below
        if numerator > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        (value - window.lower) / span
    };
    let clipped = (scaled * 255.0).clamp(0.0, 255.0);
    if clipped.is_nan() {
        return 0;
    }
    // Truncation, matching numpy's astype.
    clipped as u8
}

/// Converts a planar array to 8-bit samples using the dtype-driven exponent.
/// Mirrors the dtype dispatch at the top of `convert_narray_to_pil_image`.
fn to_uint8_planes(
    data: &PlanarData,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<Array3<u8>> {
    let (c, h, w) = data.shape();
    let Some(exp) = data.dtype().conversion_exponent() else {
        return Err(ApiError::bad_request("Array conversion not supported"));
    };

    // uint8 with no window is returned untouched.
    if data.dtype() == Dtype::U8 {
        let window = resolve_window(8, lower, upper)?;
        if window.identity {
            let mut out = Array3::zeros((c, h, w));
            for ci in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        out[[ci, y, x]] = data.get_f64(ci, y, x) as u8;
                    }
                }
            }
            return Ok(out);
        }
        let mut out = Array3::zeros((c, h, w));
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out[[ci, y, x]] = apply_window(data.get_f64(ci, y, x), window);
                }
            }
        }
        return Ok(out);
    }

    let window = resolve_window(exp, lower, upper)?;
    let mut out = Array3::zeros((c, h, w));
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[[ci, y, x]] = apply_window(data.get_f64(ci, y, x), window);
            }
        }
    }
    Ok(out)
}

/// Requested output layout, mirroring the `mode` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PilMode {
    Rgb,
    L,
}

/// `convert_narray_to_pil_image` — `image_utils.py:65`.
///
/// A single-channel array, or an explicit `L` mode, produces a greyscale image from
/// channel 0. Anything else is transposed to `(H, W, C)` and handed to
/// `Image.fromarray(..., mode="RGB")`, which requires exactly three channels — a
/// mismatch raises `ValueError` in Python and becomes
/// `400 "Image conversion to Pillow failed: ..."`.
pub fn planar_to_image(
    data: &PlanarData,
    lower: Option<f64>,
    upper: Option<f64>,
    mode: PilMode,
) -> Result<DynamicImage> {
    let planes = to_uint8_planes(data, lower, upper)?;
    let (c, h, w) = data.shape();

    if mode == PilMode::L || c == 1 {
        if c == 0 {
            return Err(ApiError::bad_request(
                "Image conversion to Pillow failed: array has no channels",
            ));
        }
        let mut img = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x as u32, y as u32, image::Luma([planes[[0, y, x]]]));
            }
        }
        return Ok(DynamicImage::ImageLuma8(img));
    }

    if c != 3 {
        // `Image.fromarray` with mode="RGB" over a (H, W, C != 3) buffer raises
        // ValueError; Python wraps it as a 400.
        return Err(ApiError::bad_request(format!(
            "Image conversion to Pillow failed: cannot create RGB image from {c} channels"
        )));
    }

    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([planes[[0, y, x]], planes[[1, y, x]], planes[[2, y, x]]]),
            );
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

/// `convert_int_to_rgba_array` — `image_utils.py:42`. OME stores channel colours as a
/// signed 32-bit RGBA integer.
pub fn int_to_rgba_array(value: i64) -> [i32; 4] {
    [
        ((value >> 24) & 0xFF) as i32,
        ((value >> 16) & 0xFF) as i32,
        ((value >> 8) & 0xFF) as i32,
        (value & 0xFF) as i32,
    ]
}

/// `convert_rgba_array_to_int` — `image_utils.py:46`.
pub fn rgba_array_to_int(rgba: [u8; 4]) -> i32 {
    i32::from_be_bytes(rgba)
}

/// `convert_rgb_image_for_channels` — `image_utils.py:50`.
///
/// Python builds a Pillow conversion matrix with 0/1 gains and calls `convert("RGB", m)`.
/// With unit gains the matrix reduces to a per-band mask: Pillow computes
/// `CLIP8(1.0 * v + 0.5)`, which truncates straight back to `v`. A mask is therefore
/// exactly equivalent, and much cheaper.
pub fn mask_rgb_channels(image: &DynamicImage, channels: &[usize]) -> DynamicImage {
    let keep_r = channels.contains(&0);
    let keep_g = channels.contains(&1);
    let keep_b = channels.contains(&2);

    let mut buf = image.to_rgb8();
    for px in buf.pixels_mut() {
        if !keep_r {
            px.0[0] = 0;
        }
        if !keep_g {
            px.0[1] = 0;
        }
        if !keep_b {
            px.0[2] = 0;
        }
    }
    DynamicImage::ImageRgb8(buf)
}

/// `convert_rgb_image_by_color` — `image_utils.py:59`.
///
/// Fractional gains, so Pillow's `+ 0.5` before truncation is observable. Currently
/// unreferenced by the request pipeline but kept for behavioural completeness.
pub fn scale_rgb_image_by_color(image: &DynamicImage, rgba: [u8; 4]) -> DynamicImage {
    let gains = [
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
    ];
    let mut buf = image.to_rgb8();
    for px in buf.pixels_mut() {
        for (band, gain) in gains.iter().enumerate() {
            let v = gain * px.0[band] as f32 + 0.5;
            px.0[band] = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v as u8
            };
        }
    }
    DynamicImage::ImageRgb8(buf)
}

/// `get_requested_channels_as_rgb_array` — `image_utils.py:98`.
///
/// Three cases, faithful to the original including its quirks:
///
/// * exactly one requested channel -> that channel alone, so the caller renders greyscale;
/// * exactly two -> `[c0, c1, zeros]`. `np.zeros` produces **float64**, and
///   `np.concatenate` promotes the whole result to float64. That promotion is reproduced
///   because it changes which exponent branch the conversion takes;
/// * anything else (including "no channels requested") -> the **first three source
///   channels**, ignoring the requested indices entirely. Requesting `[3, 4, 5]` returns
///   channels 0, 1 and 2. This is a Python defect, kept for parity; see
///   `docs/divergences.md`.
pub fn requested_channels_as_rgb_array(
    data: &PlanarData,
    image_channels: Option<&[usize]>,
) -> PlanarData {
    match image_channels {
        Some(chs) if chs.len() == 1 => data.select_channels(&chs[..1]),
        Some(chs) if chs.len() == 2 => {
            let selected = data.select_channels(chs);
            // np.zeros() is float64, and concatenation promotes.
            let promoted = to_f64(&selected);
            promoted.append_zero_channels(1)
        }
        _ => multi_channel_as_rgb(data),
    }
}

/// `get_multi_channel_as_rgb` — `image_utils.py:119`.
/// Takes the first three channels; if the source has exactly two, appends a
/// `np.zeros_like` plane, which *preserves* dtype (unlike the two-channel branch above).
fn multi_channel_as_rgb(data: &PlanarData) -> PlanarData {
    let c = data.channels();
    let take = c.min(3);
    let indices: Vec<usize> = (0..take).collect();
    let selected = data.select_channels(&indices);
    if take == 2 {
        selected.append_zero_channels(1)
    } else {
        selected
    }
}

fn to_f64(data: &PlanarData) -> PlanarData {
    let (c, h, w) = data.shape();
    let mut out = Array3::<f64>::zeros((c, h, w));
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[[ci, y, x]] = data.get_f64(ci, y, x);
            }
        }
    }
    PlanarData::F64(out)
}

/// `get_requested_channels_as_array` — `image_utils.py:142`.
///
/// Note the short-circuit: when the requested count equals the source channel count the
/// array is returned **unselected**, so `image_channels=[2,1,0]` on a 3-channel image does
/// not reorder anything. Preserved for parity.
pub fn requested_channels_as_array(data: &PlanarData, image_channels: &[usize]) -> PlanarData {
    if data.channels() == image_channels.len() {
        return data.clone();
    }
    data.select_channels(image_channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn muldiv255_matches_pillow() {
        // Reference values computed from Pillow's macro definition.
        assert_eq!(muldiv255(255, 255), 255);
        assert_eq!(muldiv255(0, 255), 0);
        assert_eq!(muldiv255(128, 128), 64);
        assert_eq!(muldiv255(1, 255), 1);
        assert_eq!(muldiv255(200, 100), 78);
    }

    #[test]
    fn alpha_composite_uses_pillow_blend() {
        // Opaque foreground wins outright.
        assert_eq!(blend(255, 255, 10), 10);
        // Fully transparent keeps the background.
        assert_eq!(blend(0, 255, 10), 255);
        // Half alpha blends both.
        assert_eq!(blend(128, 255, 0), 127);
    }

    #[test]
    fn window_default_upper_is_a_contrast_stretch_not_a_normalisation() {
        let w = resolve_window(16, None, None).unwrap();
        assert_eq!(w.lower, 0.0);
        assert_eq!(w.upper, 65536.0 / 4.0); // 16384, not 65535
        let w = resolve_window(32, None, None).unwrap();
        assert_eq!(w.upper, 4294967296.0 / 8.0);
        let w = resolve_window(64, None, None).unwrap();
        assert_eq!(w.upper, 18446744073709551616.0 / 16.0);
        // 8-bit keeps the full range when a window is forced.
        let w = resolve_window(8, Some(1.0), None).unwrap();
        assert_eq!(w.upper, 255.0);
    }

    #[test]
    fn window_identity_covers_zero_as_well_as_none() {
        // `not lower and not upper` is true for 0.0, not just None.
        assert!(resolve_window(8, None, None).unwrap().identity);
        assert!(resolve_window(8, Some(0.0), Some(0.0)).unwrap().identity);
        assert!(!resolve_window(8, Some(1.0), None).unwrap().identity);
        // Only exp == 8 short-circuits.
        assert!(!resolve_window(16, None, None).unwrap().identity);
    }

    #[test]
    fn window_bounds_are_validated() {
        assert!(resolve_window(12, None, None).is_err());
        assert!(resolve_window(8, Some(-1.0), None).is_err());
        assert!(resolve_window(8, Some(256.0), None).is_err());
        assert!(resolve_window(16, None, Some(65536.0)).is_err());
        assert!(resolve_window(16, None, Some(65535.0)).is_ok());
    }

    #[test]
    fn apply_window_truncates_it_does_not_round() {
        let w = Window {
            lower: 0.0,
            upper: 255.0,
            identity: false,
        };
        // 200/255*255 = 200 exactly
        assert_eq!(apply_window(200.0, w), 200);

        let w = Window {
            lower: 0.0,
            upper: 2.0,
            identity: false,
        };
        // 1/2*255 = 127.5 -> truncates to 127, would round to 128
        assert_eq!(apply_window(1.0, w), 127);
    }

    #[test]
    fn apply_window_clips_out_of_range() {
        let w = Window {
            lower: 10.0,
            upper: 20.0,
            identity: false,
        };
        assert_eq!(apply_window(5.0, w), 0);
        assert_eq!(apply_window(1000.0, w), 255);
    }

    #[test]
    fn uniform_region_maps_to_zero_like_numpy_nan() {
        // upper == lower: numpy computes 0/0 = nan, and nan.astype(uint8) is 0.
        let w = Window {
            lower: 7.0,
            upper: 7.0,
            identity: false,
        };
        assert_eq!(apply_window(7.0, w), 0);
    }

    #[test]
    fn sixteen_bit_default_window_saturates_above_a_quarter_range() {
        let w = resolve_window(16, None, None).unwrap();
        assert_eq!(apply_window(0.0, w), 0);
        assert_eq!(apply_window(8192.0, w), 127); // 0.5 * 255 = 127.5 -> 127
        assert_eq!(apply_window(16384.0, w), 255);
        assert_eq!(apply_window(65535.0, w), 255); // clipped
    }

    #[test]
    fn single_channel_becomes_grayscale() {
        let data = PlanarData::U8(arr3(&[[[10u8, 20], [30, 40]]]));
        let img = planar_to_image(&data, None, None, PilMode::Rgb).unwrap();
        assert!(matches!(img, DynamicImage::ImageLuma8(_)));
        assert_eq!(img.to_luma8().get_pixel(1, 0).0[0], 20);
    }

    #[test]
    fn three_channels_become_rgb() {
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]], [[3]]]));
        let img = planar_to_image(&data, None, None, PilMode::Rgb).unwrap();
        assert_eq!(img.to_rgb8().get_pixel(0, 0).0, [1, 2, 3]);
    }

    #[test]
    fn two_channels_in_rgb_mode_is_a_400() {
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]]]));
        let err = planar_to_image(&data, None, None, PilMode::Rgb).unwrap_err();
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert!(err
            .detail_text()
            .starts_with("Image conversion to Pillow failed"));
    }

    #[test]
    fn signed_dtypes_are_rejected_like_python() {
        let data = PlanarData::I16(arr3(&[[[1i16]]]));
        let err = planar_to_image(&data, None, None, PilMode::Rgb).unwrap_err();
        assert_eq!(err.detail_text(), "Array conversion not supported");
    }

    #[test]
    fn one_requested_channel_yields_one_plane() {
        let data = PlanarData::U16(arr3(&[[[1u16]], [[2]], [[3]]]));
        let out = requested_channels_as_rgb_array(&data, Some(&[1]));
        assert_eq!(out.shape(), (1, 1, 1));
        assert_eq!(out.get_f64(0, 0, 0), 2.0);
    }

    #[test]
    fn two_requested_channels_pad_with_zeros_and_promote_to_f64() {
        let data = PlanarData::U16(arr3(&[[[1u16]], [[2]], [[3]]]));
        let out = requested_channels_as_rgb_array(&data, Some(&[2, 0]));
        assert_eq!(out.shape(), (3, 1, 1));
        assert_eq!(
            out.dtype(),
            Dtype::F64,
            "np.zeros promotes the concatenation"
        );
        assert_eq!(out.get_f64(0, 0, 0), 3.0);
        assert_eq!(out.get_f64(1, 0, 0), 1.0);
        assert_eq!(out.get_f64(2, 0, 0), 0.0);
    }

    #[test]
    fn three_or_more_requested_channels_ignore_the_requested_indices() {
        // Python passes the *unselected* channel list to get_multi_channel_as_rgb, so the
        // first three source channels come back regardless of what was asked for.
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]], [[3]], [[4]], [[5]]]));
        let out = requested_channels_as_rgb_array(&data, Some(&[3, 4, 0]));
        assert_eq!(out.shape(), (3, 1, 1));
        assert_eq!(out.get_f64(0, 0, 0), 1.0);
        assert_eq!(out.get_f64(1, 0, 0), 2.0);
        assert_eq!(out.get_f64(2, 0, 0), 3.0);
    }

    #[test]
    fn no_requested_channels_takes_the_first_three() {
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]], [[3]], [[4]]]));
        let out = requested_channels_as_rgb_array(&data, None);
        assert_eq!(out.shape(), (3, 1, 1));
        assert_eq!(out.get_f64(2, 0, 0), 3.0);
    }

    #[test]
    fn two_source_channels_pad_with_zeros_like_preserving_dtype() {
        let data = PlanarData::U16(arr3(&[[[1u16]], [[2]]]));
        let out = requested_channels_as_rgb_array(&data, None);
        assert_eq!(out.shape(), (3, 1, 1));
        assert_eq!(out.dtype(), Dtype::U16, "np.zeros_like preserves dtype");
    }

    #[test]
    fn requested_channels_as_array_short_circuits_on_matching_count() {
        let data = PlanarData::U8(arr3(&[[[1u8]], [[2]], [[3]]]));
        // Same count as the source: returned untouched, so no reordering happens.
        let out = requested_channels_as_array(&data, &[2, 1, 0]);
        assert_eq!(out.get_f64(0, 0, 0), 1.0);
        // Different count: actually selects.
        let out = requested_channels_as_array(&data, &[2, 0]);
        assert_eq!(out.shape(), (2, 1, 1));
        assert_eq!(out.get_f64(0, 0, 0), 3.0);
    }

    #[test]
    fn channel_mask_zeroes_unrequested_bands() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([10, 20, 30])));
        let out = mask_rgb_channels(&img, &[0, 2]);
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [10, 0, 30]);
    }

    #[test]
    fn int_to_rgba_array_decodes_ome_colours() {
        // OME's signed 32-bit RGBA. 0xFF0000FF -> opaque red.
        assert_eq!(
            int_to_rgba_array(0xFF0000FFu32 as i32 as i64),
            [255, 0, 0, 255]
        );
        assert_eq!(rgba_array_to_int([255, 0, 0, 255]), 0xFF0000FFu32 as i32);
    }
}
