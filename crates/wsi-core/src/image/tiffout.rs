//! Minimal TIFF writer for the raw-region response.
//!
//! Stands in for `tifffile.imwrite(mem, narray, photometric="minisblack",
//! planarconfig="separate", compression="DEFLATE")` in `app_utils.py:184`.
//!
//! Scope is deliberately narrow: a single IFD, one strip per plane, Adobe Deflate,
//! `BlackIsZero`, and `PlanarConfiguration=2` whenever there is more than one sample —
//! exactly the shape the Python call produces. The output is *pixel*-equivalent to
//! tifffile's, not byte-identical: tifffile emits additional descriptive tags (a `Software`
//! tag among them) and orders its strips differently. The parity harness therefore
//! compares TIFF responses by decoding them; see `docs/divergences.md`.

use std::io::Write;

use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::{ApiError, Result};
use crate::pixels::{Dtype, PlanarData};

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIGURATION: u16 = 284;
const TAG_SAMPLE_FORMAT: u16 = 339;

const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;

const COMPRESSION_ADOBE_DEFLATE: u16 = 8;
const PHOTOMETRIC_BLACK_IS_ZERO: u16 = 1;
const PLANARCONFIG_CONTIG: u16 = 1;
const PLANARCONFIG_SEPARATE: u16 = 2;

const SAMPLEFORMAT_UINT: u16 = 1;
const SAMPLEFORMAT_INT: u16 = 2;
const SAMPLEFORMAT_IEEEFP: u16 = 3;

struct SampleLayout {
    bits: u16,
    format: u16,
}

fn layout_for(dtype: Dtype) -> Result<SampleLayout> {
    Ok(match dtype {
        Dtype::U8 => SampleLayout {
            bits: 8,
            format: SAMPLEFORMAT_UINT,
        },
        Dtype::U16 => SampleLayout {
            bits: 16,
            format: SAMPLEFORMAT_UINT,
        },
        Dtype::U32 => SampleLayout {
            bits: 32,
            format: SAMPLEFORMAT_UINT,
        },
        Dtype::U64 => SampleLayout {
            bits: 64,
            format: SAMPLEFORMAT_UINT,
        },
        Dtype::I8 => SampleLayout {
            bits: 8,
            format: SAMPLEFORMAT_INT,
        },
        Dtype::I16 => SampleLayout {
            bits: 16,
            format: SAMPLEFORMAT_INT,
        },
        Dtype::I32 => SampleLayout {
            bits: 32,
            format: SAMPLEFORMAT_INT,
        },
        Dtype::F32 => SampleLayout {
            bits: 32,
            format: SAMPLEFORMAT_IEEEFP,
        },
        Dtype::F64 => SampleLayout {
            bits: 64,
            format: SAMPLEFORMAT_IEEEFP,
        },
    })
}

/// Serialises one channel plane to little-endian bytes in row-major order.
fn plane_bytes(data: &PlanarData, channel: usize) -> Vec<u8> {
    let (_, h, w) = data.shape();
    macro_rules! collect {
        ($arr:expr, $ty:ty) => {{
            let mut out = Vec::with_capacity(h * w * std::mem::size_of::<$ty>());
            for y in 0..h {
                for x in 0..w {
                    out.extend_from_slice(&$arr[[channel, y, x]].to_le_bytes());
                }
            }
            out
        }};
    }
    match data {
        PlanarData::U8(a) => collect!(a, u8),
        PlanarData::U16(a) => collect!(a, u16),
        PlanarData::U32(a) => collect!(a, u32),
        PlanarData::U64(a) => collect!(a, u64),
        PlanarData::I8(a) => collect!(a, i8),
        PlanarData::I16(a) => collect!(a, i16),
        PlanarData::I32(a) => collect!(a, i32),
        PlanarData::F32(a) => collect!(a, f32),
        PlanarData::F64(a) => collect!(a, f64),
    }
}

fn deflate(raw: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(raw)
        .map_err(|e| ApiError::bad_request(format!("Error writing tiff file: {e}")))?;
    encoder
        .finish()
        .map_err(|e| ApiError::bad_request(format!("Error writing tiff file: {e}")))
}

/// An IFD entry, held until offsets are known.
struct Entry {
    tag: u16,
    field_type: u16,
    count: u32,
    /// Inline value when it fits in four bytes, otherwise the payload written out-of-line.
    value: EntryValue,
}

enum EntryValue {
    Inline([u8; 4]),
    Offset(Vec<u8>),
}

fn short_entry(tag: u16, value: u16) -> Entry {
    let mut bytes = [0u8; 4];
    bytes[..2].copy_from_slice(&value.to_le_bytes());
    Entry {
        tag,
        field_type: TYPE_SHORT,
        count: 1,
        value: EntryValue::Inline(bytes),
    }
}

fn long_entry(tag: u16, value: u32) -> Entry {
    Entry {
        tag,
        field_type: TYPE_LONG,
        count: 1,
        value: EntryValue::Inline(value.to_le_bytes()),
    }
}

fn short_array_entry(tag: u16, values: &[u16]) -> Entry {
    if values.len() == 1 {
        return short_entry(tag, values[0]);
    }
    if values.len() == 2 {
        let mut bytes = [0u8; 4];
        bytes[..2].copy_from_slice(&values[0].to_le_bytes());
        bytes[2..].copy_from_slice(&values[1].to_le_bytes());
        return Entry {
            tag,
            field_type: TYPE_SHORT,
            count: 2,
            value: EntryValue::Inline(bytes),
        };
    }
    let mut payload = Vec::with_capacity(values.len() * 2);
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    Entry {
        tag,
        field_type: TYPE_SHORT,
        count: values.len() as u32,
        value: EntryValue::Offset(payload),
    }
}

fn long_array_entry(tag: u16, values: &[u32]) -> Entry {
    if values.len() == 1 {
        return long_entry(tag, values[0]);
    }
    let mut payload = Vec::with_capacity(values.len() * 4);
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    Entry {
        tag,
        field_type: TYPE_LONG,
        count: values.len() as u32,
        value: EntryValue::Offset(payload),
    }
}

/// Encodes a planar array as a single-IFD, Deflate-compressed, minisblack TIFF.
///
/// One strip per plane (`RowsPerStrip == height`), matching how tifffile lays out a small
/// in-memory image.
pub fn encode_planar_tiff(data: &PlanarData) -> Result<Vec<u8>> {
    let (channels, height, width) = data.shape();
    if channels == 0 || height == 0 || width == 0 {
        return Err(ApiError::bad_request(
            "Error writing tiff file: empty image region",
        ));
    }
    if height > u32::MAX as usize || width > u32::MAX as usize {
        return Err(ApiError::bad_request(
            "Error writing tiff file: image too large",
        ));
    }
    let layout = layout_for(data.dtype())?;
    let separate = channels > 1;

    // Compress each strip. With PlanarConfiguration=1 there is a single interleaved strip;
    // that case only arises for one channel, where interleaved and separate coincide.
    let strips: Vec<Vec<u8>> = (0..channels)
        .map(|c| deflate(&plane_bytes(data, c)))
        .collect::<Result<_>>()?;

    // Header (8 bytes) then strip data, then the IFD, then out-of-line tag payloads.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // IFD offset, patched below

    let mut strip_offsets: Vec<u32> = Vec::with_capacity(strips.len());
    let mut strip_byte_counts: Vec<u32> = Vec::with_capacity(strips.len());
    for strip in &strips {
        strip_offsets.push(out.len() as u32);
        strip_byte_counts.push(strip.len() as u32);
        out.extend_from_slice(strip);
    }

    let bits_per_sample = vec![layout.bits; channels];
    let sample_format = vec![layout.format; channels];

    // TIFF requires IFD entries in ascending tag order.
    let entries = vec![
        long_entry(TAG_IMAGE_WIDTH, width as u32),
        long_entry(TAG_IMAGE_LENGTH, height as u32),
        short_array_entry(TAG_BITS_PER_SAMPLE, &bits_per_sample),
        short_entry(TAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE),
        short_entry(TAG_PHOTOMETRIC, PHOTOMETRIC_BLACK_IS_ZERO),
        long_array_entry(TAG_STRIP_OFFSETS, &strip_offsets),
        short_entry(TAG_SAMPLES_PER_PIXEL, channels as u16),
        long_entry(TAG_ROWS_PER_STRIP, height as u32),
        long_array_entry(TAG_STRIP_BYTE_COUNTS, &strip_byte_counts),
        short_entry(
            TAG_PLANAR_CONFIGURATION,
            if separate {
                PLANARCONFIG_SEPARATE
            } else {
                PLANARCONFIG_CONTIG
            },
        ),
        short_array_entry(TAG_SAMPLE_FORMAT, &sample_format),
    ];

    // Pad to an even offset, as the spec requires for IFDs.
    if out.len() % 2 != 0 {
        out.push(0);
    }
    let ifd_offset = out.len() as u32;
    out[4..8].copy_from_slice(&ifd_offset.to_le_bytes());

    // entry count + 12 bytes per entry + next-IFD pointer
    let ifd_size = 2 + entries.len() * 12 + 4;
    let mut overflow_cursor = ifd_offset as usize + ifd_size;
    let mut overflow: Vec<u8> = Vec::new();

    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for entry in &entries {
        out.extend_from_slice(&entry.tag.to_le_bytes());
        out.extend_from_slice(&entry.field_type.to_le_bytes());
        out.extend_from_slice(&entry.count.to_le_bytes());
        match &entry.value {
            EntryValue::Inline(bytes) => out.extend_from_slice(bytes),
            EntryValue::Offset(payload) => {
                out.extend_from_slice(&(overflow_cursor as u32).to_le_bytes());
                overflow.extend_from_slice(payload);
                overflow_cursor += payload.len();
                if payload.len() % 2 != 0 {
                    overflow.push(0);
                    overflow_cursor += 1;
                }
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    out.extend_from_slice(&overflow);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn parse_u16(buf: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([buf[at], buf[at + 1]])
    }
    fn parse_u32(buf: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
    }

    /// Returns (tag -> (type, count, raw 4-byte value field)).
    fn read_ifd(buf: &[u8]) -> std::collections::BTreeMap<u16, (u16, u32, [u8; 4])> {
        assert_eq!(&buf[0..2], b"II");
        assert_eq!(parse_u16(buf, 2), 42);
        let ifd = parse_u32(buf, 4) as usize;
        let count = parse_u16(buf, ifd) as usize;
        let mut map = std::collections::BTreeMap::new();
        let mut last_tag = 0u16;
        for i in 0..count {
            let at = ifd + 2 + i * 12;
            let tag = parse_u16(buf, at);
            assert!(tag > last_tag, "IFD entries must be in ascending tag order");
            last_tag = tag;
            let field_type = parse_u16(buf, at + 2);
            let n = parse_u32(buf, at + 4);
            let mut value = [0u8; 4];
            value.copy_from_slice(&buf[at + 8..at + 12]);
            map.insert(tag, (field_type, n, value));
        }
        map
    }

    #[test]
    fn single_channel_is_contiguous() {
        let data = PlanarData::U8(Array3::from_elem((1, 4, 4), 7u8));
        let buf = encode_planar_tiff(&data).unwrap();
        let ifd = read_ifd(&buf);
        assert_eq!(
            parse_u16(&ifd[&TAG_PLANAR_CONFIGURATION].2, 0),
            PLANARCONFIG_CONTIG
        );
        assert_eq!(parse_u16(&ifd[&TAG_SAMPLES_PER_PIXEL].2, 0), 1);
        assert_eq!(
            parse_u16(&ifd[&TAG_COMPRESSION].2, 0),
            COMPRESSION_ADOBE_DEFLATE
        );
        assert_eq!(
            parse_u16(&ifd[&TAG_PHOTOMETRIC].2, 0),
            PHOTOMETRIC_BLACK_IS_ZERO
        );
        assert_eq!(parse_u32(&ifd[&TAG_IMAGE_WIDTH].2, 0), 4);
        assert_eq!(parse_u32(&ifd[&TAG_IMAGE_LENGTH].2, 0), 4);
        assert_eq!(parse_u32(&ifd[&TAG_ROWS_PER_STRIP].2, 0), 4);
    }

    #[test]
    fn multi_channel_is_planar_separate_with_one_strip_per_plane() {
        let data = PlanarData::U16(Array3::from_elem((3, 2, 2), 300u16));
        let buf = encode_planar_tiff(&data).unwrap();
        let ifd = read_ifd(&buf);
        assert_eq!(
            parse_u16(&ifd[&TAG_PLANAR_CONFIGURATION].2, 0),
            PLANARCONFIG_SEPARATE
        );
        assert_eq!(parse_u16(&ifd[&TAG_SAMPLES_PER_PIXEL].2, 0), 3);
        // Three strips -> the offsets/counts arrays live out-of-line.
        assert_eq!(ifd[&TAG_STRIP_OFFSETS].1, 3);
        assert_eq!(ifd[&TAG_STRIP_BYTE_COUNTS].1, 3);
        // Three 16-bit samples -> BitsPerSample is also out-of-line.
        assert_eq!(ifd[&TAG_BITS_PER_SAMPLE].1, 3);
    }

    #[test]
    fn strips_round_trip_through_deflate() {
        let mut arr = Array3::<u16>::zeros((2, 1, 3));
        arr[[0, 0, 0]] = 1;
        arr[[0, 0, 1]] = 2;
        arr[[0, 0, 2]] = 3;
        arr[[1, 0, 0]] = 4;
        arr[[1, 0, 1]] = 5;
        arr[[1, 0, 2]] = 6;
        let data = PlanarData::U16(arr);
        let buf = encode_planar_tiff(&data).unwrap();

        let ifd = read_ifd(&buf);
        let offsets_at = parse_u32(&ifd[&TAG_STRIP_OFFSETS].2, 0) as usize;
        let counts_at = parse_u32(&ifd[&TAG_STRIP_BYTE_COUNTS].2, 0) as usize;

        let mut decoded = Vec::new();
        for i in 0..2usize {
            let off = parse_u32(&buf, offsets_at + i * 4) as usize;
            let len = parse_u32(&buf, counts_at + i * 4) as usize;
            let mut plane = Vec::new();
            let mut dec = flate2::write::ZlibDecoder::new(&mut plane);
            dec.write_all(&buf[off..off + len]).unwrap();
            dec.finish().unwrap();
            decoded.extend(
                plane
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])),
            );
        }
        assert_eq!(decoded, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn float_planes_report_ieee_sample_format() {
        let data = PlanarData::F32(Array3::from_elem((1, 1, 1), 0.5f32));
        let buf = encode_planar_tiff(&data).unwrap();
        let ifd = read_ifd(&buf);
        assert_eq!(
            parse_u16(&ifd[&TAG_SAMPLE_FORMAT].2, 0),
            SAMPLEFORMAT_IEEEFP
        );
        assert_eq!(parse_u16(&ifd[&TAG_BITS_PER_SAMPLE].2, 0), 32);
    }

    #[test]
    fn empty_region_is_rejected() {
        let data = PlanarData::U8(Array3::zeros((0, 1, 1)));
        assert!(encode_planar_tiff(&data).is_err());
    }
}
