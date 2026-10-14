//! Chunk decompression — the Rust side of `page.decode(data, index, jpegtables)`.
//!
//! A chunk is one tile or one strip. Whatever the codec, the result is the chunk's samples
//! as **native-endian** bytes, so everything downstream can treat a sample as a plain
//! little- or big-endian-free value. Byte-order normalisation happens before the predictor
//! is undone, because horizontal differencing operates on sample values, not on bytes.

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
}

impl ChunkGeometry {
    fn sample_bytes(&self) -> usize {
        self.bits_per_sample / 8
    }

    fn expected_len(&self) -> usize {
        self.width * self.height * self.samples * self.sample_bytes()
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
    if geometry.bits_per_sample % 8 != 0 || geometry.bits_per_sample > 64 {
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
    raw.resize(geometry.expected_len(), 0);

    normalise_byte_order(&mut raw, geometry.sample_bytes(), order);
    apply_predictor(predictor, &mut raw, geometry)?;
    Ok(raw)
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
                        _ => {}
                    }
                }
            }
            Ok(())
        }
        3 => Err(TiffError(
            "unsupported TIFF floating-point predictor".into(),
        )),
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
        }
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
