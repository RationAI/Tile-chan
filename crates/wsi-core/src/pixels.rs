//! The value a plugin returns from a read.
//!
//! Python plugins return one of three things and the response pipeline branches on
//! `isinstance`: a `PIL.Image`, a numpy array shaped `(C, H, W)`, or raw `bytes`. This
//! enum makes that explicit.
//!
//! The dtype tag on planar data is load-bearing: `convert_narray_to_pil_image`
//! (`wsi_service/utils/image_utils.py:65`) dispatches on it and rejects anything it does
//! not know with `400 "Array conversion not supported"`. Signed integer variants exist
//! here because the tifffile plugin can read signed pages; they hit that same 400, as in
//! Python.

use bytes::Bytes;
use image::DynamicImage;
use ndarray::Array3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    F32,
    F64,
}

impl Dtype {
    /// Bit width used as the `exp` argument of `convert_narray_uintX_to_uint8`.
    ///
    /// Python picks this by dtype: `uint16 -> 16`, `uint32`/`float32 -> 32`,
    /// `uint64`/`float64 -> 64`. `uint8` short-circuits before the conversion.
    pub fn conversion_exponent(self) -> Option<u32> {
        match self {
            Dtype::U8 => Some(8),
            Dtype::U16 => Some(16),
            Dtype::U32 | Dtype::F32 => Some(32),
            Dtype::U64 | Dtype::F64 => Some(64),
            // Python has no branch for signed dtypes; they fall through to the 400.
            Dtype::I8 | Dtype::I16 | Dtype::I32 => None,
        }
    }
}

/// Planar pixel data in numpy's `(channels, height, width)` order.
#[derive(Debug, Clone)]
pub enum PlanarData {
    U8(Array3<u8>),
    U16(Array3<u16>),
    U32(Array3<u32>),
    U64(Array3<u64>),
    I8(Array3<i8>),
    I16(Array3<i16>),
    I32(Array3<i32>),
    F32(Array3<f32>),
    F64(Array3<f64>),
}

macro_rules! for_each_variant {
    ($self:expr, $arr:ident => $body:expr) => {
        match $self {
            PlanarData::U8($arr) => $body,
            PlanarData::U16($arr) => $body,
            PlanarData::U32($arr) => $body,
            PlanarData::U64($arr) => $body,
            PlanarData::I8($arr) => $body,
            PlanarData::I16($arr) => $body,
            PlanarData::I32($arr) => $body,
            PlanarData::F32($arr) => $body,
            PlanarData::F64($arr) => $body,
        }
    };
}

impl PlanarData {
    pub fn dtype(&self) -> Dtype {
        match self {
            PlanarData::U8(_) => Dtype::U8,
            PlanarData::U16(_) => Dtype::U16,
            PlanarData::U32(_) => Dtype::U32,
            PlanarData::U64(_) => Dtype::U64,
            PlanarData::I8(_) => Dtype::I8,
            PlanarData::I16(_) => Dtype::I16,
            PlanarData::I32(_) => Dtype::I32,
            PlanarData::F32(_) => Dtype::F32,
            PlanarData::F64(_) => Dtype::F64,
        }
    }

    /// `(channels, height, width)`
    pub fn shape(&self) -> (usize, usize, usize) {
        for_each_variant!(self, a => {
            let s = a.shape();
            (s[0], s[1], s[2])
        })
    }

    pub fn channels(&self) -> usize {
        self.shape().0
    }

    pub fn height(&self) -> usize {
        self.shape().1
    }

    pub fn width(&self) -> usize {
        self.shape().2
    }

    pub fn is_empty(&self) -> bool {
        let (c, h, w) = self.shape();
        c == 0 || h == 0 || w == 0
    }

    /// Reads every sample as `f64`, in `(c, y, x)` order.
    ///
    /// The numeric pipeline works in `f64` throughout because numpy's `np.divide` promotes
    /// to float64 regardless of input dtype, and the truncating `astype(np.uint8)` at the
    /// end is applied to that float. Doing the arithmetic in any narrower type would round
    /// differently.
    // The cast is a no-op only for the `F64` arm; every other variant needs it, and the
    // macro generates one body for all nine.
    #[allow(clippy::unnecessary_cast)]
    pub fn to_f64_vec(&self) -> Vec<f64> {
        for_each_variant!(self, a => a.iter().map(|&v| v as f64).collect())
    }

    /// Value at `(channel, y, x)` as `f64`.
    #[allow(clippy::unnecessary_cast)]
    pub fn get_f64(&self, c: usize, y: usize, x: usize) -> f64 {
        for_each_variant!(self, a => a[[c, y, x]] as f64)
    }

    /// Selects channels by index, in the given order, preserving dtype.
    /// Mirrors `np.vsplit` + `np.concatenate` in `get_requested_channels_as_array`.
    pub fn select_channels(&self, indices: &[usize]) -> PlanarData {
        macro_rules! select {
            ($arr:expr, $variant:path) => {{
                let (_, h, w) = self.shape();
                let mut out = Array3::from_elem((indices.len(), h, w), Default::default());
                for (dst, &src) in indices.iter().enumerate() {
                    out.slice_mut(ndarray::s![dst, .., ..])
                        .assign(&$arr.slice(ndarray::s![src, .., ..]));
                }
                $variant(out)
            }};
        }
        match self {
            PlanarData::U8(a) => select!(a, PlanarData::U8),
            PlanarData::U16(a) => select!(a, PlanarData::U16),
            PlanarData::U32(a) => select!(a, PlanarData::U32),
            PlanarData::U64(a) => select!(a, PlanarData::U64),
            PlanarData::I8(a) => select!(a, PlanarData::I8),
            PlanarData::I16(a) => select!(a, PlanarData::I16),
            PlanarData::I32(a) => select!(a, PlanarData::I32),
            PlanarData::F32(a) => select!(a, PlanarData::F32),
            PlanarData::F64(a) => select!(a, PlanarData::F64),
        }
    }

    /// Appends `count` all-zero channels, preserving dtype.
    /// Used by `get_requested_channels_as_rgb_array`, which pads a two-channel request to
    /// three planes with `np.zeros`.
    pub fn append_zero_channels(&self, count: usize) -> PlanarData {
        macro_rules! extend {
            ($arr:expr, $variant:path) => {{
                let (c, h, w) = self.shape();
                let mut out = Array3::from_elem((c + count, h, w), Default::default());
                out.slice_mut(ndarray::s![..c, .., ..]).assign($arr);
                $variant(out)
            }};
        }
        match self {
            PlanarData::U8(a) => extend!(a, PlanarData::U8),
            PlanarData::U16(a) => extend!(a, PlanarData::U16),
            PlanarData::U32(a) => extend!(a, PlanarData::U32),
            PlanarData::U64(a) => extend!(a, PlanarData::U64),
            PlanarData::I8(a) => extend!(a, PlanarData::I8),
            PlanarData::I16(a) => extend!(a, PlanarData::I16),
            PlanarData::I32(a) => extend!(a, PlanarData::I32),
            PlanarData::F32(a) => extend!(a, PlanarData::F32),
            PlanarData::F64(a) => extend!(a, PlanarData::F64),
        }
    }

    /// Minimum and maximum over every sample, as `f64`.
    ///
    /// `process_image_region` windows explicitly-selected channels with `np.min(result)`
    /// and `np.max(result)` of the *already selected* array, so this must be computed
    /// after channel selection, never before.
    #[allow(clippy::unnecessary_cast)]
    pub fn min_max(&self) -> Option<(f64, f64)> {
        for_each_variant!(self, a => {
            let mut iter = a.iter().map(|&v| v as f64);
            let first = iter.next()?;
            let mut lo = first;
            let mut hi = first;
            for v in iter {
                if v < lo { lo = v; }
                if v > hi { hi = v; }
            }
            Some((lo, hi))
        })
    }

    /// A new array of the same dtype and channel count, filled with `value`
    /// (clamped into the dtype's range, like `__get_color_for_channel` does for signed
    /// pages), shaped `(channels, height, width)`.
    pub fn filled_like(
        &self,
        channels: usize,
        height: usize,
        width: usize,
        value: f64,
    ) -> PlanarData {
        macro_rules! filled {
            ($ty:ty, $variant:path, $lo:expr, $hi:expr) => {{
                let clamped = value.clamp($lo as f64, $hi as f64) as $ty;
                $variant(Array3::from_elem((channels, height, width), clamped))
            }};
        }
        match self {
            PlanarData::U8(_) => filled!(u8, PlanarData::U8, u8::MIN, u8::MAX),
            PlanarData::U16(_) => filled!(u16, PlanarData::U16, u16::MIN, u16::MAX),
            PlanarData::U32(_) => filled!(u32, PlanarData::U32, u32::MIN, u32::MAX),
            PlanarData::U64(_) => filled!(u64, PlanarData::U64, u64::MIN, u64::MAX),
            PlanarData::I8(_) => filled!(i8, PlanarData::I8, i8::MIN, i8::MAX),
            PlanarData::I16(_) => filled!(i16, PlanarData::I16, i16::MIN, i16::MAX),
            PlanarData::I32(_) => filled!(i32, PlanarData::I32, i32::MIN, i32::MAX),
            PlanarData::F32(_) => {
                PlanarData::F32(Array3::from_elem((channels, height, width), value as f32))
            }
            PlanarData::F64(_) => {
                PlanarData::F64(Array3::from_elem((channels, height, width), value))
            }
        }
    }

    /// A new array of the same dtype, with one fill value per channel.
    ///
    /// Reproduces numpy's `out[...] = np.asarray(padding_color).reshape(-1, 1, 1)`, which
    /// broadcasts a per-channel column across the plane. `values.len()` must equal
    /// `channels`; the caller checks that and raises the same `ValueError` message Python
    /// does when it does not.
    pub fn filled_per_channel(&self, height: usize, width: usize, values: &[f64]) -> PlanarData {
        let channels = values.len();
        macro_rules! filled {
            ($ty:ty, $variant:path, $lo:expr, $hi:expr) => {{
                let mut out =
                    Array3::<$ty>::from_elem((channels, height, width), Default::default());
                for (ci, &v) in values.iter().enumerate() {
                    let clamped = v.clamp($lo as f64, $hi as f64) as $ty;
                    out.slice_mut(ndarray::s![ci, .., ..]).fill(clamped);
                }
                $variant(out)
            }};
        }
        match self {
            PlanarData::U8(_) => filled!(u8, PlanarData::U8, u8::MIN, u8::MAX),
            PlanarData::U16(_) => filled!(u16, PlanarData::U16, u16::MIN, u16::MAX),
            PlanarData::U32(_) => filled!(u32, PlanarData::U32, u32::MIN, u32::MAX),
            PlanarData::U64(_) => filled!(u64, PlanarData::U64, u64::MIN, u64::MAX),
            PlanarData::I8(_) => filled!(i8, PlanarData::I8, i8::MIN, i8::MAX),
            PlanarData::I16(_) => filled!(i16, PlanarData::I16, i16::MIN, i16::MAX),
            PlanarData::I32(_) => filled!(i32, PlanarData::I32, i32::MIN, i32::MAX),
            PlanarData::F32(_) => {
                let mut out = Array3::<f32>::zeros((channels, height, width));
                for (ci, &v) in values.iter().enumerate() {
                    out.slice_mut(ndarray::s![ci, .., ..]).fill(v as f32);
                }
                PlanarData::F32(out)
            }
            PlanarData::F64(_) => {
                let mut out = Array3::<f64>::zeros((channels, height, width));
                for (ci, &v) in values.iter().enumerate() {
                    out.slice_mut(ndarray::s![ci, .., ..]).fill(v);
                }
                PlanarData::F64(out)
            }
        }
    }

    /// Copies `src` into `self` at offset `(dst_y, dst_x)`, for all channels.
    /// Panics if the dtypes differ — callers always produce both sides themselves.
    pub fn blit(&mut self, src: &PlanarData, dst_y: usize, dst_x: usize) {
        macro_rules! blit {
            ($dst:expr, $src:expr) => {{
                let (c, h, w) = $src.dim();
                $dst.slice_mut(ndarray::s![..c, dst_y..dst_y + h, dst_x..dst_x + w])
                    .assign($src);
            }};
        }
        match (self, src) {
            (PlanarData::U8(d), PlanarData::U8(s)) => blit!(d, s),
            (PlanarData::U16(d), PlanarData::U16(s)) => blit!(d, s),
            (PlanarData::U32(d), PlanarData::U32(s)) => blit!(d, s),
            (PlanarData::U64(d), PlanarData::U64(s)) => blit!(d, s),
            (PlanarData::I8(d), PlanarData::I8(s)) => blit!(d, s),
            (PlanarData::I16(d), PlanarData::I16(s)) => blit!(d, s),
            (PlanarData::I32(d), PlanarData::I32(s)) => blit!(d, s),
            (PlanarData::F32(d), PlanarData::F32(s)) => blit!(d, s),
            (PlanarData::F64(d), PlanarData::F64(s)) => blit!(d, s),
            _ => panic!("blit between mismatched planar dtypes"),
        }
    }

    /// Crops to the top-left `height` x `width` of every channel.
    pub fn crop_top_left(&self, height: usize, width: usize) -> PlanarData {
        macro_rules! crop {
            ($arr:expr, $variant:path) => {
                $variant($arr.slice(ndarray::s![.., ..height, ..width]).to_owned())
            };
        }
        match self {
            PlanarData::U8(a) => crop!(a, PlanarData::U8),
            PlanarData::U16(a) => crop!(a, PlanarData::U16),
            PlanarData::U32(a) => crop!(a, PlanarData::U32),
            PlanarData::U64(a) => crop!(a, PlanarData::U64),
            PlanarData::I8(a) => crop!(a, PlanarData::I8),
            PlanarData::I16(a) => crop!(a, PlanarData::I16),
            PlanarData::I32(a) => crop!(a, PlanarData::I32),
            PlanarData::F32(a) => crop!(a, PlanarData::F32),
            PlanarData::F64(a) => crop!(a, PlanarData::F64),
        }
    }
}

/// What a plugin read returns.
#[derive(Debug, Clone)]
pub enum Pixels {
    /// The `PIL.Image` path. RGB or luma.
    Image(DynamicImage),
    /// The numpy path: `(channels, height, width)`.
    Planar(PlanarData),
    /// Pre-encoded bytes (raw JPEG tiles) or a passthrough payload (MVT, GeoJSON).
    Raw(Bytes),
}

impl Pixels {
    pub fn is_raw(&self) -> bool {
        matches!(self, Pixels::Raw(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    fn sample() -> PlanarData {
        // 3 channels, 1 row, 2 columns
        PlanarData::U16(arr3(&[[[1u16, 2]], [[3, 4]], [[5, 6]]]))
    }

    #[test]
    fn shape_is_channels_height_width() {
        assert_eq!(sample().shape(), (3, 1, 2));
    }

    #[test]
    fn select_channels_reorders_and_subsets() {
        let selected = sample().select_channels(&[2, 0]);
        assert_eq!(selected.shape(), (2, 1, 2));
        assert_eq!(selected.get_f64(0, 0, 0), 5.0);
        assert_eq!(selected.get_f64(1, 0, 0), 1.0);
    }

    #[test]
    fn append_zero_channels_pads_with_zero() {
        let padded = sample().select_channels(&[0, 1]).append_zero_channels(1);
        assert_eq!(padded.shape(), (3, 1, 2));
        assert_eq!(padded.get_f64(2, 0, 0), 0.0);
        assert_eq!(padded.get_f64(2, 0, 1), 0.0);
    }

    #[test]
    fn min_max_spans_all_channels() {
        assert_eq!(sample().min_max(), Some((1.0, 6.0)));
    }

    #[test]
    fn filled_like_clamps_into_signed_range() {
        // Signed 8-bit pages cannot hold 255; Python clamps to the dtype's max.
        let signed = PlanarData::I8(arr3(&[[[0i8]]]));
        let filled = signed.filled_like(1, 1, 1, 255.0);
        assert_eq!(filled.get_f64(0, 0, 0), 127.0);
    }

    #[test]
    fn blit_places_source_at_offset() {
        let mut dst = PlanarData::U16(Array3::zeros((1, 3, 3)));
        let src = PlanarData::U16(arr3(&[[[7u16, 8]]]));
        dst.blit(&src, 1, 1);
        assert_eq!(dst.get_f64(0, 1, 1), 7.0);
        assert_eq!(dst.get_f64(0, 1, 2), 8.0);
        assert_eq!(dst.get_f64(0, 0, 0), 0.0);
    }

    #[test]
    fn crop_top_left_trims() {
        let src = PlanarData::U8(Array3::from_elem((2, 4, 4), 9u8));
        let cropped = src.crop_top_left(2, 3);
        assert_eq!(cropped.shape(), (2, 2, 3));
    }
}
