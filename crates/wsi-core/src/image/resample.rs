//! Pillow's resampler, ported exactly.
//!
//! Every thumbnail, label and macro response goes through `Image.thumbnail` or
//! `Image.resize`, so this is on the critical path for pixel parity — and a generic
//! high-quality resize is *not* a substitute. Measured against real slides, the `image`
//! crate's Lanczos3 lands within 1–2 per channel of Pillow, which is enough to fail a
//! byte comparison on every downscaled image.
//!
//! Three details account for the difference, and all three are reproduced here:
//!
//! 1. **Fixed-point coefficients.** Pillow computes filter weights in `f64`, normalises
//!    them to sum to 1, then rounds each to a 22-bit fixed-point integer
//!    (`normalize_coeffs_8bpc`). The accumulator starts at half an output unit so the final
//!    shift rounds rather than truncates.
//! 2. **A box-average pre-pass.** `resize` defaults to `reducing_gap=2.0`, which for a
//!    downscale of more than about 4x first calls `reduce()` — an exact integer box average
//!    — and then resamples the smaller image over a fractional box.
//! 3. **`thumbnail`'s own aspect arithmetic**, which picks between `floor` and `ceil` by
//!    whichever lands closer to the source aspect ratio, rather than rounding.
//!
//! Sources: Pillow's `src/libImaging/Resample.c`, `src/libImaging/Reduce.c` and
//! `Image.resize` / `Image.thumbnail` in `src/PIL/Image.py`.

use image::DynamicImage;

/// `PRECISION_BITS` — `Resample.c:18`, `32 - 8 - 2`.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Pillow's default `reducing_gap` for `resize` and `thumbnail`.
pub const DEFAULT_REDUCING_GAP: f64 = 2.0;

/// The resampling filters this service reaches.
///
/// `Image.thumbnail` defaults to `BICUBIC`, which is what every plugin's own
/// `get_thumbnail` uses; the `label` and `macro` endpoints pass `LANCZOS` explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Bicubic,
    Lanczos,
}

impl Filter {
    /// `struct filter { double (*filter)(double x); double support; }`
    fn support(self) -> f64 {
        match self {
            Filter::Bicubic => 2.0,
            Filter::Lanczos => 3.0,
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Filter::Bicubic => bicubic(x),
            Filter::Lanczos => lanczos(x),
        }
    }
}

/// `bicubic_filter` — Catmull-Rom with `a = -0.5`.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// `lanczos_filter` — a sinc truncated at three lobes.
fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// `clip8` — shift the accumulator down and clamp into a byte.
///
/// Pillow reaches this through a lookup table offset into a padded array, which is a
/// branch-free clamp; a clamp is what it computes.
#[inline]
fn clip8(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Interleaved 8-bit samples, the layout Pillow resamples in.
///
/// Channel count matters: Pillow stores `RGB` as four bytes per pixel and resamples all
/// four, but the fourth is padding. Tracking the real count avoids resampling padding and
/// produces the same visible channels.
#[derive(Clone)]
pub struct Bands {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl Bands {
    fn new(width: u32, height: u32, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0u8; width as usize * height as usize * channels],
        }
    }

    #[inline]
    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize * self.width as usize + x as usize) * self.channels
    }

    #[inline]
    fn sample(&self, x: u32, y: u32, channel: usize) -> u8 {
        self.data[self.index(x, y) + channel]
    }

    /// Splits a `DynamicImage` into the layout Pillow would hold it in.
    pub fn from_image(image: &DynamicImage) -> Self {
        match image {
            DynamicImage::ImageLuma8(buffer) => Self {
                width: buffer.width(),
                height: buffer.height(),
                channels: 1,
                data: buffer.as_raw().clone(),
            },
            DynamicImage::ImageRgba8(buffer) => Self {
                width: buffer.width(),
                height: buffer.height(),
                channels: 4,
                data: buffer.as_raw().clone(),
            },
            other => {
                let buffer = other.to_rgb8();
                Self {
                    width: buffer.width(),
                    height: buffer.height(),
                    channels: 3,
                    data: buffer.into_raw(),
                }
            }
        }
    }

    pub fn into_image(self) -> DynamicImage {
        match self.channels {
            1 => DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(self.width, self.height, self.data)
                    .expect("dimensions match the buffer"),
            ),
            4 => DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(self.width, self.height, self.data)
                    .expect("dimensions match the buffer"),
            ),
            _ => DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(self.width, self.height, self.data)
                    .expect("dimensions match the buffer"),
            ),
        }
    }
}

/// A source rectangle, in fractional pixels — Pillow's `box` argument.
#[derive(Debug, Clone, Copy)]
pub struct Box {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Box {
    fn whole(width: u32, height: u32) -> Self {
        Self {
            x0: 0.0,
            y0: 0.0,
            x1: width as f64,
            y1: height as f64,
        }
    }
}

/// Per-output-pixel filter weights, in Pillow's fixed-point form.
struct Coeffs {
    /// `(xmin, count)` per output pixel.
    bounds: Vec<(u32, u32)>,
    /// `out_size * ksize` weights, row-major, zero-padded.
    weights: Vec<i64>,
    ksize: usize,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` — `Resample.c:154,232`.
fn precompute_coeffs(in_size: u32, in0: f64, in1: f64, out_size: u32, filter: Filter) -> Coeffs {
    // `precompute_coeffs(int inSize, float in0, float in1, ...)` — the box reaches C as
    // `float box[4]`, so both bounds are narrowed to single precision first, and their
    // difference is taken in single precision before it is widened again:
    //
    //     filterscale = scale = (double)(in1 - in0) / outSize;
    //     double center = in0 + (xx + 0.5) * scale;
    //
    // The box is only fractional after the reduce pre-pass — 463 / 3 becomes 154.33333
    // as an `f64` and 154.33333 as an `f32` — and the gap between those is enough to move
    // a coefficient or a window edge across a rounding boundary. Keeping the full `f64`
    // left two pixels of a 53x64 label one off Python's.
    let in0 = f64::from(in0 as f32);
    let in1 = f64::from(in1 as f32);
    let scale = f64::from(in1 as f32 - in0 as f32) / out_size as f64;
    // Upscaling keeps the filter at unit width; only downscaling widens it.
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter.support() * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut bounds = Vec::with_capacity(out_size as usize);
    let mut weights = vec![0i64; out_size as usize * ksize];
    let mut scratch = vec![0f64; ksize];

    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;

        // C casts truncate toward zero, and the results are then clamped.
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let count = (xmax - xmin).max(0) as usize;

        let mut total = 0.0f64;
        for (x, slot) in scratch.iter_mut().enumerate().take(count) {
            let w = filter.apply(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            *slot = w;
            total += w;
        }
        if total != 0.0 {
            for slot in scratch.iter_mut().take(count) {
                *slot /= total;
            }
        }

        let base = xx as usize * ksize;
        for x in 0..ksize {
            let value = if x < count { scratch[x] } else { 0.0 };
            // Round away from zero, matching `normalize_coeffs_8bpc`.
            let fixed = if value < 0.0 {
                -0.5 + value * (1i64 << PRECISION_BITS) as f64
            } else {
                0.5 + value * (1i64 << PRECISION_BITS) as f64
            };
            weights[base + x] = fixed as i64;
        }
        bounds.push((xmin as u32, count as u32));
    }

    Coeffs {
        bounds,
        weights,
        ksize,
    }
}

/// `ImagingResampleHorizontal_8bpc`.
fn resample_horizontal(src: &Bands, out_width: u32, coeffs: &Coeffs) -> Bands {
    let mut out = Bands::new(out_width, src.height, src.channels);
    // The accumulator starts at half an output unit, so the closing shift rounds.
    let initial: i64 = 1 << (PRECISION_BITS - 1);

    for yy in 0..src.height {
        for xx in 0..out_width {
            let (xmin, count) = coeffs.bounds[xx as usize];
            let base = xx as usize * coeffs.ksize;
            let out_index = out.index(xx, yy);
            for channel in 0..src.channels {
                let mut sum = initial;
                for x in 0..count {
                    sum += src.sample(xmin + x, yy, channel) as i64
                        * coeffs.weights[base + x as usize];
                }
                out.data[out_index + channel] = clip8(sum);
            }
        }
    }
    out
}

/// `ImagingResampleVertical_8bpc`.
fn resample_vertical(src: &Bands, out_height: u32, coeffs: &Coeffs) -> Bands {
    let mut out = Bands::new(src.width, out_height, src.channels);
    let initial: i64 = 1 << (PRECISION_BITS - 1);

    for yy in 0..out_height {
        let (ymin, count) = coeffs.bounds[yy as usize];
        let base = yy as usize * coeffs.ksize;
        for xx in 0..src.width {
            let out_index = out.index(xx, yy);
            for channel in 0..src.channels {
                let mut sum = initial;
                for y in 0..count {
                    sum += src.sample(xx, ymin + y, channel) as i64
                        * coeffs.weights[base + y as usize];
                }
                out.data[out_index + channel] = clip8(sum);
            }
        }
    }
    out
}

/// `ImagingResample` — the horizontal pass then the vertical one, each skipped when that
/// axis already has the target size.
fn resample(src: &Bands, out_width: u32, out_height: u32, filter: Filter, source: Box) -> Bands {
    let mut current = std::borrow::Cow::Borrowed(src);

    // Pillow skips a pass when the axis needs no change *and* the box covers it fully.
    let horizontal_needed =
        out_width != src.width || source.x0 != 0.0 || source.x1 != src.width as f64;
    if horizontal_needed {
        let coeffs = precompute_coeffs(src.width, source.x0, source.x1, out_width, filter);
        current = std::borrow::Cow::Owned(resample_horizontal(&current, out_width, &coeffs));
    }

    let vertical_needed =
        out_height != src.height || source.y0 != 0.0 || source.y1 != src.height as f64;
    if vertical_needed {
        let coeffs = precompute_coeffs(src.height, source.y0, source.y1, out_height, filter);
        current = std::borrow::Cow::Owned(resample_vertical(&current, out_height, &coeffs));
    }

    current.into_owned()
}

/// `division_UINT32(divider, 8)` — `Reduce.c:8`.
///
/// Pillow does not divide the block sum; it multiplies by a reciprocal and shifts. The
/// reciprocal is computed in **single** precision and then truncated, so it is slightly
/// *below* `2^24 / scale` and the shift can land one below the true quotient. Reproducing
/// the division exactly instead is off by one on roughly a tenth of the pixels of a
/// reduced image, which is what made every thumbnail, label and macro response differ.
fn reduce_multiplier(scale: u32) -> u64 {
    // `max_dividend = (1 << 8) * divider` and `max_int = (1 << 30) * 4.0`, which is 2^32
    // exactly. The quotient is an `f32`, then cast to an integer, which truncates.
    let max_dividend = (256u64 * u64::from(scale)) as f32;
    let max_int = 4_294_967_296f32;
    (max_int / max_dividend) as u64
}

/// One output sample of `ImagingReduce`: `((ss + amend) * multiplier) >> 24`, with
/// `amend = scale / 2` and `scale` the number of source pixels the block actually covers.
///
/// Pillow keeps this in a `UINT32`; the largest value reachable is
/// `(255 * scale + scale / 2) * (2^24 / scale)`, just under `u32::MAX`, so widening to
/// `u64` changes no result and removes the need to reason about that margin.
fn reduce_sample(sum: u64, scale: u32) -> u8 {
    let amend = u64::from(scale / 2);
    (((sum + amend) * reduce_multiplier(scale)) >> 24).min(255) as u8
}

/// `ImagingReduce` — the integer box average used as `resize`'s pre-pass.
///
/// The output is `ceil(size / factor)`: a partial block at the right or bottom edge is
/// averaged over however many pixels it actually covers, which is what
/// `ImagingReduceCorners` does with its own `scale`. Pillow has a specialised function per
/// small factor, but they all compute the same thing — see [`reduce_sample`].
pub fn reduce_bands(src: &Bands, factor_x: u32, factor_y: u32) -> Bands {
    if factor_x <= 1 && factor_y <= 1 {
        return src.clone();
    }
    let out_width = src.width.div_ceil(factor_x);
    let out_height = src.height.div_ceil(factor_y);
    let mut out = Bands::new(out_width, out_height, src.channels);

    for y in 0..out_height {
        let y_start = y * factor_y;
        let y_end = (y_start + factor_y).min(src.height);
        for x in 0..out_width {
            let x_start = x * factor_x;
            let x_end = (x_start + factor_x).min(src.width);
            let scale = (x_end - x_start) * (y_end - y_start);
            let out_index = out.index(x, y);
            for channel in 0..src.channels {
                let mut sum = 0u64;
                for yy in y_start..y_end {
                    for xx in x_start..x_end {
                        sum += u64::from(src.sample(xx, yy, channel));
                    }
                }
                out.data[out_index + channel] = reduce_sample(sum, scale);
            }
        }
    }
    out
}

/// `Image.resize(size, resample, box=None, reducing_gap=2.0)`.
pub fn resize(
    image: &DynamicImage,
    out_width: u32,
    out_height: u32,
    filter: Filter,
    reducing_gap: Option<f64>,
) -> DynamicImage {
    if out_width == 0 || out_height == 0 {
        // Pillow produces an empty image; nothing downstream can use one, so keep the
        // source rather than panicking on a zero-sized buffer.
        return image.clone();
    }
    let src = Bands::from_image(image);
    if src.width == out_width && src.height == out_height {
        // `Image.resize` returns a copy when nothing changes.
        return image.clone();
    }
    resize_bands(&src, out_width, out_height, filter, reducing_gap).into_image()
}

fn resize_bands(
    src: &Bands,
    out_width: u32,
    out_height: u32,
    filter: Filter,
    reducing_gap: Option<f64>,
) -> Bands {
    let mut source = Box::whole(src.width, src.height);
    let mut working = std::borrow::Cow::Borrowed(src);

    if let Some(gap) = reducing_gap {
        // `factor_x = int(box_width / size[0] / reducing_gap) or 1`
        let factor_x = (((source.x1 - source.x0) / out_width as f64 / gap) as u32).max(1);
        let factor_y = (((source.y1 - source.y0) / out_height as f64 / gap) as u32).max(1);
        if factor_x > 1 || factor_y > 1 {
            // `_get_safe_box` expands the box by the filter support, then clamps it to the
            // image — which, for a box that already covers everything, leaves it unchanged.
            let reduced = reduce_bands(src, factor_x, factor_y);
            source = Box {
                x0: source.x0 / factor_x as f64,
                y0: source.y0 / factor_y as f64,
                // Note the *true* division: when the size is not a multiple of the factor,
                // the reduced image is wider than the box the resample reads.
                x1: source.x1 / factor_x as f64,
                y1: source.y1 / factor_y as f64,
            };
            working = std::borrow::Cow::Owned(reduced);
        }
    }

    resample(&working, out_width, out_height, filter, source)
}

/// `Image.thumbnail(size, resample, reducing_gap=2.0)`.
///
/// Shrinks in place, preserves the aspect ratio, and never enlarges.
pub fn thumbnail(image: &DynamicImage, max_x: u32, max_y: u32, filter: Filter) -> DynamicImage {
    let Some((out_width, out_height)) =
        preserve_aspect_ratio(image.width(), image.height(), max_x, max_y)
    else {
        // Already inside the box: Pillow returns without touching the image.
        return image.clone();
    };
    if image.width() == out_width && image.height() == out_height {
        return image.clone();
    }
    resize(
        image,
        out_width,
        out_height,
        filter,
        Some(DEFAULT_REDUCING_GAP),
    )
}

/// `Image.thumbnail`'s inner `preserve_aspect_ratio`.
///
/// Returns `None` when the image already fits, which is Pillow's "do nothing" signal. The
/// chosen dimension is whichever of `floor` and `ceil` lands closer to the source aspect
/// ratio, with ties going to `floor` — not a plain `round`.
pub fn preserve_aspect_ratio(
    width: u32,
    height: u32,
    max_x: u32,
    max_y: u32,
) -> Option<(u32, u32)> {
    let (mut x, mut y) = (max_x, max_y);
    if x >= width && y >= height {
        return None;
    }
    if width == 0 || height == 0 || x == 0 || y == 0 {
        // Pillow would divide by zero; there is nothing sensible to resize to.
        return None;
    }

    let aspect = width as f64 / height as f64;
    if x as f64 / y as f64 >= aspect {
        x = round_aspect(y as f64 * aspect, |n| (aspect - n as f64 / y as f64).abs());
    } else {
        y = round_aspect(x as f64 / aspect, |n| {
            if n == 0 {
                0.0
            } else {
                (aspect - x as f64 / n as f64).abs()
            }
        });
    }
    Some((x, y))
}

/// `round_aspect(number, key) = max(min(floor(n), ceil(n), key=key), 1)`.
///
/// `min` keeps its first argument on a tie, so `floor` wins.
fn round_aspect(number: f64, key: impl Fn(u32) -> f64) -> u32 {
    let floor = number.floor().max(0.0) as u32;
    let ceil = number.ceil().max(0.0) as u32;
    let chosen = if key(ceil) < key(floor) { ceil } else { floor };
    chosen.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    fn gradient(width: u32, height: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Rgb([
                (x * 255 / width.max(1)) as u8,
                (y * 255 / height.max(1)) as u8,
                ((x + y) % 256) as u8,
            ]);
        }
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn bicubic_matches_pillows_catmull_rom() {
        // f(0) = 1, f(1) = 0, f(2) = 0, and the classic a = -0.5 shape between.
        assert_eq!(bicubic(0.0), 1.0);
        assert!(bicubic(1.0).abs() < 1e-12);
        assert_eq!(bicubic(2.0), 0.0);
        assert_eq!(bicubic(3.0), 0.0);
        // Symmetric.
        assert_eq!(bicubic(-0.5), bicubic(0.5));
        // a = -0.5 gives f(0.5) = 0.5 * (…) — check the published value.
        assert!((bicubic(0.5) - 0.5625).abs() < 1e-12);
    }

    #[test]
    fn lanczos_is_a_three_lobe_sinc() {
        assert_eq!(lanczos(0.0), 1.0);
        // Zero at every non-zero integer inside the support.
        for x in [1.0, 2.0] {
            assert!(lanczos(x).abs() < 1e-12, "lanczos({x})");
        }
        // Outside the support it is exactly zero, and the window is asymmetric at +3.
        assert_eq!(lanczos(3.0), 0.0);
        assert_eq!(lanczos(-3.0), lanczos(-3.0));
        assert_eq!(lanczos(4.0), 0.0);
    }

    #[test]
    fn clip8_rounds_and_clamps() {
        let unit = 1i64 << PRECISION_BITS;
        let half = 1i64 << (PRECISION_BITS - 1);
        // The accumulator's initial half-unit turns the shift into a round.
        assert_eq!(clip8(half + 200 * unit), 200);
        assert_eq!(clip8(half), 0);
        // Overshoot from a negative filter lobe is clamped, not wrapped.
        assert_eq!(clip8(-5 * unit), 0);
        assert_eq!(clip8(300 * unit), 255);
    }

    #[test]
    fn a_flat_image_survives_resampling_unchanged() {
        // Coefficients sum to one, so a constant image must come back constant — the
        // sharpest check that normalisation and rounding are right.
        for filter in [Filter::Bicubic, Filter::Lanczos] {
            let flat = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([37, 211, 5])));
            let out = resize(&flat, 17, 23, filter, Some(DEFAULT_REDUCING_GAP));
            assert_eq!((out.width(), out.height()), (17, 23));
            for pixel in out.to_rgb8().pixels() {
                assert_eq!(pixel.0, [37, 211, 5], "{filter:?} shifted a flat image");
            }
        }
    }

    #[test]
    fn a_flat_greyscale_image_survives_too() {
        let flat = DynamicImage::ImageLuma8(GrayImage::from_pixel(50, 50, Luma([128])));
        let out = resize(&flat, 7, 7, Filter::Lanczos, Some(DEFAULT_REDUCING_GAP));
        for pixel in out.to_luma8().pixels() {
            assert_eq!(pixel.0[0], 128);
        }
    }

    /// Pillow's reciprocal is a *truncated* `f32`, so whenever `sum + scale / 2` is an
    /// exact multiple of `scale` the shift lands one below the true quotient. Ground truth,
    /// from Pillow 12.3.0:
    ///
    /// ```text
    /// sum=5  over 3x3 -> 0   (exact division: 1)
    /// sum=9  over 3x3 -> 1
    /// sum=13 over 3x3 -> 1
    /// ```
    ///
    /// Dividing exactly instead disagreed with Pillow on roughly a tenth of the pixels of
    /// every reduced image, which is what made thumbnail, label and macro responses differ.
    #[test]
    fn reduce_reproduces_pillows_reciprocal_rather_than_dividing() {
        assert_eq!(reduce_sample(5, 9), 0);
        assert_eq!(reduce_sample(9, 9), 1);
        assert_eq!(reduce_sample(13, 9), 1);
        assert_eq!(reduce_sample(1152, 9), 128);
        assert_eq!(reduce_sample(2295, 9), 255);

        // The same shortfall at other factors: a block of nine 1s reduced 3x3, versus a
        // block of four 1s reduced 2x2, where `sum + 2 == 6` is not a multiple of 4.
        assert_eq!(reduce_sample(4, 4), 1);
        assert_eq!(reduce_sample(2, 4), 1);

        // A factor of 1 has to be the identity, and its multiplier is exactly 2^24.
        assert_eq!(reduce_multiplier(1), 1 << 24);
        for value in [0u64, 1, 128, 255] {
            assert_eq!(u64::from(reduce_sample(value, 1)), value);
        }
    }

    #[test]
    fn reduce_is_an_exact_rounded_box_average() {
        // A 2x2 of 10, 20, 30, 41 averages to (101 + 2) / 4 = 25.
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, Rgb([10, 10, 10]));
        image.put_pixel(1, 0, Rgb([20, 20, 20]));
        image.put_pixel(0, 1, Rgb([30, 30, 30]));
        image.put_pixel(1, 1, Rgb([41, 41, 41]));
        let bands = Bands::from_image(&DynamicImage::ImageRgb8(image));

        let reduced = reduce_bands(&bands, 2, 2);
        assert_eq!((reduced.width, reduced.height), (1, 1));
        assert_eq!(reduced.sample(0, 0, 0), 25);
    }

    #[test]
    fn reduce_rounds_up_the_output_size_and_averages_partial_blocks() {
        // 3 pixels reduced by 2 gives 2 outputs; the second covers one pixel only.
        let mut image = GrayImage::new(3, 1);
        image.put_pixel(0, 0, Luma([0]));
        image.put_pixel(1, 0, Luma([100]));
        image.put_pixel(2, 0, Luma([200]));
        let bands = Bands::from_image(&DynamicImage::ImageLuma8(image));

        let reduced = reduce_bands(&bands, 2, 1);
        assert_eq!((reduced.width, reduced.height), (2, 1));
        assert_eq!(reduced.sample(0, 0, 0), 50);
        assert_eq!(reduced.sample(1, 0, 0), 200);
    }

    #[test]
    fn a_large_downscale_takes_the_reduce_pre_pass() {
        // 1000 -> 100 is a factor of 10, so `int(10 / 2) = 5` triggers reduce. The result
        // must still be the right size and free of artefacts on a flat source.
        let flat = DynamicImage::ImageRgb8(RgbImage::from_pixel(1000, 500, Rgb([7, 8, 9])));
        let out = resize(&flat, 100, 50, Filter::Bicubic, Some(DEFAULT_REDUCING_GAP));
        assert_eq!((out.width(), out.height()), (100, 50));
        for pixel in out.to_rgb8().pixels() {
            assert_eq!(pixel.0, [7, 8, 9]);
        }
    }

    #[test]
    fn disabling_the_gap_skips_the_reduce_pre_pass_and_can_differ() {
        // The two paths are both valid resizes but not the same computation, which is
        // exactly why `reducing_gap` has to be reproduced rather than ignored.
        let image = gradient(800, 800);
        let with_gap = resize(&image, 50, 50, Filter::Lanczos, Some(DEFAULT_REDUCING_GAP));
        let without = resize(&image, 50, 50, Filter::Lanczos, None);
        assert_eq!((with_gap.width(), with_gap.height()), (50, 50));
        assert_ne!(
            with_gap.to_rgb8().into_raw(),
            without.to_rgb8().into_raw(),
            "the reduce pre-pass should change the result"
        );
    }

    #[test]
    fn thumbnail_preserves_aspect_ratio() {
        assert_eq!(preserve_aspect_ratio(400, 200, 100, 100), Some((100, 50)));
        assert_eq!(preserve_aspect_ratio(200, 400, 100, 100), Some((50, 100)));
        // Square fits exactly.
        assert_eq!(preserve_aspect_ratio(400, 400, 100, 100), Some((100, 100)));
    }

    #[test]
    fn thumbnail_returns_none_when_the_image_already_fits() {
        assert_eq!(preserve_aspect_ratio(10, 10, 100, 100), None);
        // Equal to the box also counts as fitting.
        assert_eq!(preserve_aspect_ratio(100, 100, 100, 100), None);
        // Only one axis over the box still resizes.
        assert!(preserve_aspect_ratio(200, 10, 100, 100).is_some());
    }

    #[test]
    fn aspect_rounding_picks_the_closer_of_floor_and_ceil() {
        // 46000x32914 into 21x22: x/y = 0.954 < aspect 1.398, so y is derived.
        // 21 / 1.3976 = 15.026 -> floor 15 is closer to the aspect than ceil 16.
        let size = preserve_aspect_ratio(46000, 32914, 21, 22).unwrap();
        assert_eq!(size, (21, 15));
    }

    #[test]
    fn a_derived_dimension_never_collapses_to_zero() {
        // An extreme aspect ratio would round to 0; Pillow clamps to 1.
        let size = preserve_aspect_ratio(1000, 3, 10, 10).unwrap();
        assert_eq!(size, (10, 1));
    }

    #[test]
    fn thumbnail_never_enlarges() {
        let small = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([1, 2, 3])));
        let out = thumbnail(&small, 500, 500, Filter::Bicubic);
        assert_eq!((out.width(), out.height()), (10, 10));
    }

    #[test]
    fn a_single_pixel_target_is_the_whole_image_averaged() {
        let flat = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([90, 90, 90])));
        let out = thumbnail(&flat, 1, 1, Filter::Lanczos);
        assert_eq!((out.width(), out.height()), (1, 1));
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [90, 90, 90]);
    }

    #[test]
    fn rgba_keeps_its_alpha_channel_through_a_resize() {
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            40,
            40,
            image::Rgba([10, 20, 30, 40]),
        ));
        let out = resize(&rgba, 9, 9, Filter::Bicubic, Some(DEFAULT_REDUCING_GAP));
        assert!(matches!(out, DynamicImage::ImageRgba8(_)));
        assert_eq!(out.to_rgba8().get_pixel(0, 0).0, [10, 20, 30, 40]);
    }

    #[test]
    fn upscaling_keeps_the_filter_at_unit_width() {
        // `filterscale` is clamped to 1.0 when scale < 1, so an upscale interpolates rather
        // than blurring. A flat image is still flat, and the size is right.
        let flat = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([200, 100, 50])));
        let out = resize(&flat, 16, 16, Filter::Bicubic, Some(DEFAULT_REDUCING_GAP));
        assert_eq!((out.width(), out.height()), (16, 16));
        assert_eq!(out.to_rgb8().get_pixel(8, 8).0, [200, 100, 50]);
    }

    #[test]
    fn a_two_tone_downscale_stays_antisymmetric() {
        // Half black, half white, down to two pixels. At this ratio the filter support
        // spans both halves, so neither output pixel is pure — but the input is
        // antisymmetric about the centre, so the outputs must be too. That is a property of
        // the algorithm rather than a number picked to fit, and it catches any asymmetry in
        // the coefficient bounds or the rounding.
        let mut image = GrayImage::new(64, 1);
        for x in 0..64 {
            image.put_pixel(x, 0, Luma([if x < 32 { 0 } else { 255 }]));
        }
        let out = resize(
            &DynamicImage::ImageLuma8(image),
            2,
            1,
            Filter::Bicubic,
            Some(DEFAULT_REDUCING_GAP),
        );
        let luma = out.to_luma8();
        let (left, right) = (luma.get_pixel(0, 0).0[0], luma.get_pixel(1, 0).0[0]);
        assert!(
            left < right,
            "the dark half must stay darker: {left} vs {right}"
        );
        assert_eq!(
            left as u16 + right as u16,
            255,
            "antisymmetric input should give antisymmetric output, got {left} and {right}"
        );
    }
}
