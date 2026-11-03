//! Chunk decompression — the Rust side of `page.decode(data, index, jpegtables)`.
//!
//! A chunk is one tile or one strip. Whatever the codec, the result is the chunk's samples
//! as **native-endian** bytes, so everything downstream can treat a sample as a plain
//! little- or big-endian-free value. Byte-order normalisation happens before the predictor
//! is undone, because horizontal differencing operates on sample values, not on bytes.
//!
//! The result is also always at the width of the page's container dtype
//! (`SampleDtype::from_tags`): packed samples (1-, 4-, 12-, 24-bit) are unpacked, and
//! float16 is widened to float32. The floating-point predictor is the
//! exception to the order above — it works on bytes and yields native order itself.

use std::io::Read;

use super::ifd::{ByteOrder, Result, TiffError};

/// Compression tag values this reader understands, plus the ones it names when refusing.
pub mod compression {
    pub const NONE: u16 = 1;
    pub const CCITT_RLE: u16 = 2;
    pub const CCITT_T4: u16 = 3;
    pub const CCITT_T6: u16 = 4;
    pub const LZW: u16 = 5;
    pub const OLD_JPEG: u16 = 6;
    pub const JPEG: u16 = 7;
    pub const ADOBE_DEFLATE: u16 = 8;
    pub const PACKBITS: u16 = 32773;
    pub const DEFLATE: u16 = 32946;
    pub const JPEG_2000: u16 = 33003;
    pub const JPEG_2000_APERIO: u16 = 33005;
    pub const JPEG_2000_YCBC: u16 = 34712;
    pub const LZMA: u16 = 34925;
    pub const ZSTD: u16 = 50000;
    pub const WEBP: u16 = 50001;
}

/// What one chunk of a page holds.
#[derive(Debug, Clone, Copy)]
pub struct ChunkGeometry {
    /// Pixels across — the tile width, or the image width for a strip.
    pub width: usize,
    /// Rows in this chunk. Tiles are always the full tile height, even at the edge; the
    /// last strip of a page is short.
    pub height: usize,
    /// Samples per pixel *in this chunk*: 1 when the page stores planes separately.
    pub samples: usize,
    pub bits_per_sample: usize,
    pub photometric: u16,
    /// `SampleFormat`: 1 unsigned, 2 signed, 3 float.
    pub sample_format: u16,
    /// `FillOrder`: 2 stores the bits of each byte least-significant first. Only packed
    /// samples care.
    pub fill_order: u16,
}

impl ChunkGeometry {
    /// Bytes one stored sample occupies, for byte-aligned widths.
    fn sample_bytes(&self) -> usize {
        self.bits_per_sample / 8
    }

    /// Bytes one row occupies as stored: packed rows are padded to a whole byte.
    fn stored_row_len(&self) -> usize {
        (self.width * self.samples * self.bits_per_sample).div_ceil(8)
    }

    fn stored_len(&self) -> usize {
        self.height * self.stored_row_len()
    }

    /// Bytes one sample occupies once decoded — the width of `SampleDtype::from_tags`.
    fn container_bytes(&self) -> usize {
        match self.bits_per_sample {
            0..=8 => 1,
            9..=16 if self.sample_format == 3 => 4,
            9..=16 => 2,
            17..=32 => 4,
            _ => 8,
        }
    }
}

/// Decodes one chunk into native-endian sample bytes.
pub fn decode_chunk(
    compression: u16,
    predictor: u16,
    order: ByteOrder,
    geometry: ChunkGeometry,
    data: &[u8],
    jpegtables: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if geometry.bits_per_sample == 0 || geometry.bits_per_sample > 64 {
        return Err(TiffError(format!(
            "unsupported bits per sample {}",
            geometry.bits_per_sample
        )));
    }

    if matches!(compression, compression::JPEG | compression::OLD_JPEG) {
        // The JPEG decoder produces 8-bit interleaved samples in its own order, so neither
        // byte swapping nor the predictor applies.
        return decode_jpeg(geometry, data, jpegtables);
    }
    if matches!(
        compression,
        compression::JPEG_2000 | compression::JPEG_2000_APERIO | compression::JPEG_2000_YCBC
    ) {
        // Likewise: openjpeg hands back whole sample values, already unpacked.
        return decode_jpeg2000(geometry, data);
    }

    let mut raw = match compression {
        compression::NONE => data.to_vec(),
        compression::LZW => decode_lzw(data)?,
        compression::ADOBE_DEFLATE | compression::DEFLATE => decode_deflate(data)?,
        compression::PACKBITS => decode_packbits(data),
        compression::ZSTD => decode_zstd(data)?,
        other => return Err(TiffError(format!("unsupported TIFF compression {other}"))),
    };

    // Short chunks are padded rather than rejected: a writer may omit trailing rows of an
    // edge tile, and the padding area is overwritten or cropped away by the caller.
    raw.resize(geometry.stored_len(), 0);

    if !matches!(geometry.bits_per_sample, 8 | 16 | 32 | 64) {
        // libtiff defines no predictor for packed samples, and tifffile refuses one too.
        if predictor > 1 {
            return Err(TiffError(format!(
                "unsupported TIFF predictor {predictor} for {}-bit samples",
                geometry.bits_per_sample
            )));
        }
        return Ok(unpack_bits(&raw, geometry));
    }

    if predictor == 3 {
        undo_float_predictor(&mut raw, geometry)?;
    } else {
        normalise_byte_order(&mut raw, geometry.sample_bytes(), order);
        apply_predictor(predictor, &mut raw, geometry)?;
    }
    Ok(widen(raw, geometry))
}

/// Unpacks samples narrower than a byte, or of no standard width (12-, 24-bit), into the
/// container.
///
/// TIFF packs samples as one big-endian bitstream per row — the first sample in the most
/// significant bits — and pads each row to a whole byte. That holds whatever the file's
/// byte order; `FillOrder` 2 only reverses the bits within each byte.
///
/// 24-bit integers go this way too, because that is how tifffile reads (and writes) them.
/// libtiff instead stores them as three bytes in the file's byte order, so a little-endian
/// 24-bit file written by libtiff reads differently here — exactly as it does in Python.
fn unpack_bits(raw: &[u8], geometry: ChunkGeometry) -> Vec<u8> {
    let bits = geometry.bits_per_sample;
    let size = geometry.container_bytes();
    let per_row = geometry.width * geometry.samples;
    let row_len = geometry.stored_row_len();
    let mut out = vec![0u8; geometry.height * per_row * size];

    let byte_at = |index: usize| -> u64 {
        let byte = raw[index];
        u64::from(if geometry.fill_order == 2 {
            byte.reverse_bits()
        } else {
            byte
        })
    };

    for row in 0..geometry.height {
        let row_start = row * row_len;
        for column in 0..per_row {
            let first_bit = column * bits;
            let mut value = 0u64;
            // A sample spans at most a handful of bytes; walk them most significant first.
            let mut bit = first_bit;
            while bit < first_bit + bits {
                let within = bit % 8;
                let take = (8 - within).min(first_bit + bits - bit);
                let byte = byte_at(row_start + bit / 8);
                let chunk = (byte >> (8 - within - take)) & ((1 << take) - 1);
                value = (value << take) | chunk;
                bit += take;
            }
            let at = (row * per_row + column) * size;
            write_native(&mut out[at..at + size], value);
        }
    }
    out
}

/// Writes the low bits of `value` as a native-endian integer as wide as `out`.
///
/// Truncation is the intent: a signed JPEG 2000 sample's two's-complement low bytes are
/// the container's bit pattern.
fn write_native(out: &mut [u8], value: u64) {
    match out.len() {
        1 => out[0] = value as u8,
        2 => out.copy_from_slice(&(value as u16).to_ne_bytes()),
        4 => out.copy_from_slice(&(value as u32).to_ne_bytes()),
        _ => out.copy_from_slice(&value.to_ne_bytes()),
    }
}

/// Widens float16 to its float32 container; everything else passes through.
fn widen(raw: Vec<u8>, geometry: ChunkGeometry) -> Vec<u8> {
    match (geometry.sample_format, geometry.bits_per_sample) {
        (3, 16) => raw
            .chunks_exact(2)
            .flat_map(|sample| {
                half::f16::from_ne_bytes([sample[0], sample[1]])
                    .to_f32()
                    .to_ne_bytes()
            })
            .collect(),
        _ => raw,
    }
}

/// Undoes `Predictor` 3, floating-point horizontal differencing — libtiff's `fpAcc`.
///
/// The writer split each row into byte planes, most significant byte first whatever the
/// file's byte order, then differenced the whole row byte by byte, one pixel's worth of
/// samples apart. Undoing that is a running sum over the row followed by re-interleaving
/// the planes, straight into native order.
fn undo_float_predictor(bytes: &mut [u8], geometry: ChunkGeometry) -> Result<()> {
    let sample_bytes = geometry.sample_bytes();
    if geometry.sample_format != 3 || !matches!(sample_bytes, 2 | 4 | 8) {
        return Err(TiffError(format!(
            "unsupported TIFF floating-point predictor for {}-bit samples",
            geometry.bits_per_sample
        )));
    }
    let stride = geometry.samples;
    let row_len = geometry.stored_row_len();
    let values = row_len / sample_bytes;
    let mut planes = vec![0u8; row_len];
    for row in bytes.chunks_exact_mut(row_len) {
        for index in stride..row_len {
            row[index] = row[index].wrapping_add(row[index - stride]);
        }
        planes.copy_from_slice(row);
        for value in 0..values {
            for byte in 0..sample_bytes {
                // Plane 0 holds the most significant bytes.
                let plane = if cfg!(target_endian = "little") {
                    sample_bytes - 1 - byte
                } else {
                    byte
                };
                row[value * sample_bytes + byte] = planes[plane * values + value];
            }
        }
    }
    Ok(())
}

/// Swaps each sample into native order when the file disagrees with the host.
fn normalise_byte_order(bytes: &mut [u8], sample_bytes: usize, order: ByteOrder) {
    if sample_bytes < 2 {
        return;
    }
    let file_is_little = order == ByteOrder::Little;
    if file_is_little == cfg!(target_endian = "little") {
        return;
    }
    for sample in bytes.chunks_exact_mut(sample_bytes) {
        sample.reverse();
    }
}

/// Undoes `Predictor` 2, horizontal differencing.
///
/// The accumulation runs along a row, one stride of `samples` values at a time, so a
/// contiguous RGB row differences red against red rather than against the preceding blue.
fn apply_predictor(predictor: u16, bytes: &mut [u8], geometry: ChunkGeometry) -> Result<()> {
    match predictor {
        0 | 1 => Ok(()),
        2 => {
            let stride = geometry.samples;
            let sample_bytes = geometry.sample_bytes();
            if !matches!(sample_bytes, 1 | 2 | 4 | 8) {
                return Err(TiffError(format!(
                    "unsupported TIFF predictor 2 for {}-bit samples",
                    geometry.bits_per_sample
                )));
            }
            let row_len = geometry.width * stride * sample_bytes;
            for row in bytes.chunks_mut(row_len) {
                for index in stride..(row.len() / sample_bytes) {
                    let previous = (index - stride) * sample_bytes;
                    let current = index * sample_bytes;
                    match sample_bytes {
                        1 => row[current] = row[current].wrapping_add(row[previous]),
                        2 => accumulate::<2, u16>(row, previous, current),
                        4 => accumulate::<4, u32>(row, previous, current),
                        8 => accumulate::<8, u64>(row, previous, current),
                        _ => unreachable!("checked above"),
                    }
                }
            }
            Ok(())
        }
        other => Err(TiffError(format!("unsupported TIFF predictor {other}"))),
    }
}

/// `row[current] += row[previous]`, in native order, for an `N`-byte unsigned sample.
///
/// Wrapping is correct and required: the predictor is defined modulo the sample width, and
/// signed samples differencing the same way is what the format specifies.
fn accumulate<const N: usize, T>(row: &mut [u8], previous: usize, current: usize)
where
    T: Copy + std::ops::Add<Output = T> + WrappingAdd + FromNeBytes<N> + ToNeBytes<N>,
{
    let mut left = [0u8; N];
    let mut here = [0u8; N];
    left.copy_from_slice(&row[previous..previous + N]);
    here.copy_from_slice(&row[current..current + N]);
    let sum = T::from_ne(here).wrapping_add(T::from_ne(left));
    row[current..current + N].copy_from_slice(&sum.to_ne());
}

trait WrappingAdd {
    fn wrapping_add(self, other: Self) -> Self;
}

trait FromNeBytes<const N: usize> {
    fn from_ne(bytes: [u8; N]) -> Self;
}

trait ToNeBytes<const N: usize> {
    fn to_ne(self) -> [u8; N];
}

macro_rules! ne_bytes {
    ($ty:ty, $n:literal) => {
        impl WrappingAdd for $ty {
            fn wrapping_add(self, other: Self) -> Self {
                <$ty>::wrapping_add(self, other)
            }
        }
        impl FromNeBytes<$n> for $ty {
            fn from_ne(bytes: [u8; $n]) -> Self {
                <$ty>::from_ne_bytes(bytes)
            }
        }
        impl ToNeBytes<$n> for $ty {
            fn to_ne(self) -> [u8; $n] {
                self.to_ne_bytes()
            }
        }
    };
}

ne_bytes!(u16, 2);
ne_bytes!(u32, 4);
ne_bytes!(u64, 8);

fn decode_lzw(data: &[u8]) -> Result<Vec<u8>> {
    // TIFF's LZW is MSB-first with the "early change" code-width switch libtiff writes.
    weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
        .decode(data)
        .map_err(|e| TiffError(format!("LZW: {e}")))
}

fn decode_deflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(data)
        .read_to_end(&mut out)
        .map_err(|e| TiffError(format!("Deflate: {e}")))?;
    Ok(out)
}

fn decode_zstd(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| TiffError(format!("Zstd: {e}")))?;
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| TiffError(format!("Zstd: {e}")))?;
    Ok(out)
}

/// PackBits, the TIFF variant of run-length encoding.
fn decode_packbits(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < data.len() {
        let header = data[index] as i8;
        index += 1;
        if header >= 0 {
            let run = header as usize + 1;
            let end = (index + run).min(data.len());
            out.extend_from_slice(&data[index..end]);
            index = end;
        } else if header != -128 {
            // -128 is a no-op by specification.
            let run = (1 - header as i32) as usize;
            if index < data.len() {
                out.extend(std::iter::repeat(data[index]).take(run));
                index += 1;
            }
        }
    }
    out
}

/// An `APP14` "Adobe" segment declaring that no colour transform was applied.
///
/// `FF EE`, length 14, `Adobe`, version 100, two flag words, then the transform byte: 0
/// for "the samples already are R, G and B". Without it, a three-component JPEG carrying
/// no JFIF marker is assumed to be YCbCr, and an Aperio tile — which really is RGB, as
/// its own `JPEG/RGB` comment says — comes back with 239 turned into 255.
const APP14_NO_COLOUR_TRANSFORM: [u8; 16] = [
    0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// JPEG tiles, with the page's shared `JPEGTables` spliced in when present.
///
/// An abbreviated tile stream carries no quantisation or Huffman tables of its own; the
/// page holds one copy for every tile. Concatenating the table stream without its `EOI`
/// and the tile without its `SOI` produces the complete image the decoder needs — the
/// same splice `imagecodecs` performs for tifffile.
///
/// The page's `PhotometricInterpretation` then says how to read the result:
/// `imagecodecs` is handed `colorspace=photometric`, which for `RGB` sets libjpeg's
/// `jpeg_color_space = JCS_RGB` and so skips the inverse colour transform. No decoder
/// available here exposes that knob, so the same thing is announced inside the stream.
fn decode_jpeg(geometry: ChunkGeometry, data: &[u8], jpegtables: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut merged = match jpegtables {
        Some(tables) if tables.len() > 2 && data.len() > 2 => {
            let mut merged = Vec::with_capacity(tables.len() + data.len());
            merged.extend_from_slice(&tables[..tables.len() - 2]);
            merged.extend_from_slice(&data[2..]);
            merged
        }
        _ => data.to_vec(),
    };

    // Photometric 2 is RGB. 6 is YCbCr, where the transform is wanted and the decoder's
    // default is already right.
    if geometry.photometric == 2 && geometry.samples == 3 && merged.len() >= 2 {
        let already_declared = merged.windows(5).take(256).any(|window| window == b"Adobe");
        if !already_declared {
            merged.splice(2..2, APP14_NO_COLOUR_TRANSFORM);
        }
    }

    use image::ImageDecoder;
    let decoder = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&merged))
        .map_err(|e| TiffError(format!("JPEG: {e}")))?;
    let (width, height) = decoder.dimensions();
    let colour = decoder.color_type();
    let mut pixels = vec![0u8; decoder.total_bytes() as usize];
    decoder
        .read_image(&mut pixels)
        .map_err(|e| TiffError(format!("JPEG: {e}")))?;

    let decoded_samples = match colour {
        image::ColorType::L8 => 1,
        image::ColorType::Rgb8 => 3,
        other => return Err(TiffError(format!("unsupported JPEG colour type {other:?}"))),
    };

    // A JPEG tile can be narrower or shorter than the nominal tile size at the image edge,
    // and its component count can differ from the page's when a writer stores greyscale
    // planes separately. Copy into the geometry the caller expects.
    let mut out = vec![0u8; geometry.width * geometry.height * geometry.samples];
    let copy_samples = decoded_samples.min(geometry.samples);
    for y in 0..(height as usize).min(geometry.height) {
        for x in 0..(width as usize).min(geometry.width) {
            let src = (y * width as usize + x) * decoded_samples;
            let dst = (y * geometry.width + x) * geometry.samples;
            out[dst..dst + copy_samples].copy_from_slice(&pixels[src..src + copy_samples]);
        }
    }
    Ok(out)
}

/// JPEG 2000 tiles — compressions 33003, 33005 and 34712 all hold a bare codestream.
///
/// Decoded by openjpeg, as imagecodecs does for tifffile. openjpeg undoes whatever
/// multi-component transform the codestream declares; beyond that, imagecodecs follows
/// `opj_decompress`: three components whose chroma is subsampled are taken to be sYCC,
/// upsampled and converted to RGB (`sycc_to_rgb`). That is what an Aperio 33003 tile is —
/// 4:2:0 YCbCr with no transform declared — and what this does too.
fn decode_jpeg2000(geometry: ChunkGeometry, data: &[u8]) -> Result<Vec<u8>> {
    let image =
        jpeg2k::Image::from_bytes(data).map_err(|e| TiffError(format!("JPEG 2000: {e}")))?;
    let components = image.components();
    if components.len() != geometry.samples {
        return Err(TiffError(format!(
            "JPEG 2000: {} components where the page has {} samples",
            components.len(),
            geometry.samples
        )));
    }

    // Every component at the full size of the first, chroma replicated as `sycc420_to_rgb`
    // and `sycc422_to_rgb` do.
    let (width, height) = (
        components[0].width() as usize,
        components[0].height() as usize,
    );
    let mut planes: Vec<Vec<i32>> = components
        .iter()
        .map(|component| {
            let (w, h) = (component.width() as usize, component.height() as usize);
            let values = component.data();
            if (w, h) == (width, height) {
                return values.to_vec();
            }
            let (dx, dy) = (width.div_ceil(w.max(1)), height.div_ceil(h.max(1)));
            let mut plane = Vec::with_capacity(width * height);
            for y in 0..height {
                let row = (y / dy).min(h - 1) * w;
                plane.extend((0..width).map(|x| values[row + (x / dx).min(w - 1)]));
            }
            plane
        })
        .collect();

    let subsampled_chroma = components.len() == 3 && components[1].width() as usize != width;
    let declared_sycc = matches!(image.color_space(), jpeg2k::ColorSpace::SYCC);
    if subsampled_chroma || (components.len() == 3 && declared_sycc) {
        sycc_to_rgb(&mut planes, components[0].precision());
    }

    // As with JPEG, an edge tile may be smaller than the nominal tile; the rest stays zero.
    let size = geometry.container_bytes();
    let mut out = vec![0u8; geometry.width * geometry.height * geometry.samples * size];
    for (sample, plane) in planes.iter().enumerate() {
        for y in 0..height.min(geometry.height) {
            for x in 0..width.min(geometry.width) {
                let at = ((y * geometry.width + x) * geometry.samples + sample) * size;
                write_native(&mut out[at..at + size], plane[y * width + x] as i64 as u64);
            }
        }
    }
    Ok(out)
}

/// openjpeg's `sycc_to_rgb` (`src/bin/common/color.c`), arithmetic and all: the factors
/// are doubles applied to a float, truncated toward zero, then clamped to the precision.
fn sycc_to_rgb(planes: &mut [Vec<i32>], precision: u32) {
    let offset = 1i32 << (precision.max(1) - 1);
    let upper = (1i32 << precision.max(1)) - 1;
    let [y, cb, cr] = planes else { return };
    for index in 0..y.len() {
        let luma = y[index];
        let blue = (cb[index] - offset) as f32 as f64;
        let red = (cr[index] - offset) as f32 as f64;
        y[index] = (luma + (1.402 * red) as i32).clamp(0, upper);
        cb[index] = (luma - (0.344 * blue + 0.714 * red) as i32).clamp(0, upper);
        cr[index] = (luma + (1.772 * blue) as i32).clamp(0, upper);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: usize, height: usize, samples: usize, bits: usize) -> ChunkGeometry {
        ChunkGeometry {
            width,
            height,
            samples,
            bits_per_sample: bits,
            photometric: 1,
            sample_format: 1,
            fill_order: 1,
        }
    }

    fn float_geometry(width: usize, samples: usize, bits: usize) -> ChunkGeometry {
        ChunkGeometry {
            sample_format: 3,
            ..geometry(width, 1, samples, bits)
        }
    }

    fn native_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn twelve_bit_samples_unpack_with_each_row_padded_to_a_byte() {
        // Three 12-bit samples per row is 36 bits, so each row carries 4 bits of padding.
        // Row 0: 0xABC 0x123 0xFFF; row 1: 0x001 0x800 0x7FF.
        let packed = [
            0xAB, 0xC1, 0x23, 0xFF, 0xF0, //
            0x00, 0x18, 0x00, 0x7F, 0xF0,
        ];
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Little,
            geometry(3, 2, 1, 12),
            &packed,
            None,
        )
        .unwrap();
        assert_eq!(
            native_u16(&decoded),
            vec![0xABC, 0x123, 0xFFF, 0x001, 0x800, 0x7FF]
        );
    }

    #[test]
    fn packed_samples_ignore_the_file_byte_order() {
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Big,
            geometry(2, 1, 1, 12),
            &[0xAB, 0xC1, 0x23],
            None,
        )
        .unwrap();
        assert_eq!(native_u16(&decoded), vec![0xABC, 0x123]);
    }

    #[test]
    fn one_bit_samples_unpack_to_zero_and_one() {
        // Ten pixels: 1011001110, padded to two bytes.
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Little,
            geometry(10, 1, 1, 1),
            &[0b1011_0011, 0b1000_0000],
            None,
        )
        .unwrap();
        assert_eq!(decoded, vec![1, 0, 1, 1, 0, 0, 1, 1, 1, 0]);
    }

    #[test]
    fn fill_order_two_reverses_the_bits_of_each_byte() {
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Little,
            ChunkGeometry {
                fill_order: 2,
                ..geometry(2, 1, 1, 4)
            },
            &[0x1E],
            None,
        )
        .unwrap();
        // 0x1E = 0001_1110 reversed is 0111_1000: samples 0x7 and 0x8.
        assert_eq!(decoded, vec![0x7, 0x8]);
    }

    #[test]
    fn compressed_packed_samples_unpack_after_decompression() {
        // tifffile will not write compressed packed data, so this case has no fixture.
        use flate2::write::ZlibEncoder;
        use std::io::Write;
        let mut encoder = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&[0xAB, 0xC1, 0x23]).unwrap();
        let decoded = decode_chunk(
            compression::DEFLATE,
            1,
            ByteOrder::Little,
            geometry(2, 1, 1, 12),
            &encoder.finish().unwrap(),
            None,
        )
        .unwrap();
        assert_eq!(native_u16(&decoded), vec![0xABC, 0x123]);
    }

    #[test]
    fn a_predictor_on_packed_samples_is_refused() {
        let error = decode_chunk(
            compression::NONE,
            2,
            ByteOrder::Little,
            geometry(2, 1, 1, 12),
            &[0; 3],
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported TIFF predictor 2 for 12-bit samples"
        );
    }

    #[test]
    fn twenty_four_bit_samples_are_a_big_endian_bitstream_in_either_byte_order() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let stored = [0x12, 0x34, 0x56];
            let decoded = decode_chunk(
                compression::NONE,
                1,
                order,
                geometry(1, 1, 1, 24),
                &stored,
                None,
            )
            .unwrap();
            assert_eq!(decoded, 0x123456u32.to_ne_bytes(), "{order:?}");
        }
    }

    #[test]
    fn float16_widens_to_float32() {
        let stored: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Little,
            float_geometry(2, 1, 16),
            &stored,
            None,
        )
        .unwrap();
        let values: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values, vec![1.5, -2.0]);
    }

    /// The inverse of `undo_float_predictor`, written straight from the libtiff encoder
    /// (`fpDiff`), so the test does not merely check the decoder against itself.
    fn float_predictor_encode(values: &[f32], samples: usize) -> Vec<u8> {
        let count = values.len();
        let mut planes = vec![0u8; count * 4];
        for (index, value) in values.iter().enumerate() {
            for (plane, byte) in value.to_be_bytes().iter().enumerate() {
                planes[plane * count + index] = *byte;
            }
        }
        for index in (samples..planes.len()).rev() {
            planes[index] = planes[index].wrapping_sub(planes[index - samples]);
        }
        planes
    }

    #[test]
    fn the_floating_point_predictor_restores_values() {
        let values = [0.0f32, 1.0, 3.25, -7.5, 1e-3, 65504.0];
        let decoded = decode_chunk(
            compression::NONE,
            3,
            // The planes are most significant first whatever the file says.
            ByteOrder::Big,
            float_geometry(3, 2, 32),
            &float_predictor_encode(&values, 2),
            None,
        )
        .unwrap();
        let restored: Vec<f32> = decoded
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(restored, values);
    }

    #[test]
    fn the_floating_point_predictor_refuses_integer_samples() {
        assert!(decode_chunk(
            compression::NONE,
            3,
            ByteOrder::Little,
            geometry(2, 1, 1, 16),
            &[0; 4],
            None,
        )
        .is_err());
    }

    #[test]
    fn packbits_expands_runs_and_literals() {
        // -3 => five copies of 0xAA; 1 => two literals; -128 => skipped.
        let data = [0xFDu8, 0xAA, 0x01, 0x01, 0x02, 0x80];
        assert_eq!(
            decode_packbits(&data),
            vec![0xAA, 0xAA, 0xAA, 0xAA, 0x01, 0x02]
        );
    }

    #[test]
    fn deflate_round_trips() {
        use flate2::write::ZlibEncoder;
        use std::io::Write;
        let mut encoder = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"hello tiff").unwrap();
        let compressed = encoder.finish().unwrap();
        assert_eq!(decode_deflate(&compressed).unwrap(), b"hello tiff");
    }

    #[test]
    fn the_horizontal_predictor_accumulates_per_sample() {
        // Two pixels, three samples, 8-bit: the second pixel is stored as a difference.
        let mut bytes = vec![10u8, 20, 30, 1, 2, 3];
        apply_predictor(2, &mut bytes, geometry(2, 1, 3, 8)).unwrap();
        assert_eq!(bytes, vec![10, 20, 30, 11, 22, 33]);
    }

    #[test]
    fn the_horizontal_predictor_handles_sixteen_bit_samples() {
        let stored: [u16; 4] = [1000, 5, 2000, 10];
        let mut bytes: Vec<u8> = stored.iter().flat_map(|v| v.to_ne_bytes()).collect();
        apply_predictor(2, &mut bytes, geometry(2, 1, 2, 16)).unwrap();
        let out: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(out, vec![1000, 5, 3000, 15]);
    }

    #[test]
    fn the_predictor_resets_at_each_row() {
        let mut bytes = vec![5u8, 1, 7, 2];
        apply_predictor(2, &mut bytes, geometry(2, 2, 1, 8)).unwrap();
        assert_eq!(bytes, vec![5, 6, 7, 9]);
    }

    #[test]
    fn an_unknown_codec_is_named_in_the_error() {
        let error = decode_chunk(
            compression::LZMA,
            1,
            ByteOrder::Little,
            geometry(2, 2, 1, 8),
            &[0; 4],
            None,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "unsupported TIFF compression 34925");
    }

    #[test]
    fn a_short_chunk_is_zero_padded_to_its_geometry() {
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Little,
            geometry(4, 2, 1, 8),
            &[1, 2, 3],
            None,
        )
        .unwrap();
        assert_eq!(decoded.len(), 8);
        assert_eq!(&decoded[..4], &[1, 2, 3, 0]);
    }

    #[test]
    fn big_endian_samples_are_swapped_into_native_order() {
        // 0x0102 stored big-endian must read as 258.
        let decoded = decode_chunk(
            compression::NONE,
            1,
            ByteOrder::Big,
            geometry(1, 1, 1, 16),
            &[0x01, 0x02],
            None,
        )
        .unwrap();
        assert_eq!(u16::from_ne_bytes([decoded[0], decoded[1]]), 258);
    }
}
