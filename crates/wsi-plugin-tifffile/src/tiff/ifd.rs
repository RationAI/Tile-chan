//! TIFF and BigTIFF directory parsing.
//!
//! The plugin needs what `tifffile` exposes as a page's tags, `dataoffsets` and
//! `databytecounts`, and it needs them for pages reached through SubIFDs as well as
//! through the main chain. Nothing here decodes pixels; that is `super::codec`.
//!
//! Only the structure is parsed, not interpreted: an [`Entry`] keeps the field type, the
//! count and the raw value bytes, and the accessors convert on demand. That keeps the
//! BigTIFF differences — 8-byte offsets, an 8-byte inline value slot, the `LONG8` field
//! types — confined to this file.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Tag numbers the plugin or the reader consults.
pub mod tag {
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    pub const SUBFILE_TYPE: u16 = 255;
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const PLANAR_CONFIGURATION: u16 = 284;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const PREDICTOR: u16 = 317;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SUB_IFDS: u16 = 330;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const JPEG_TABLES: u16 = 347;
    pub const IMAGE_DEPTH: u16 = 32997;
    pub const TILE_DEPTH: u16 = 32998;
    pub const ICC_PROFILE: u16 = 34675;
    pub const IJ_METADATA_BYTE_COUNTS: u16 = 50838;
    pub const IJ_METADATA: u16 = 50839;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

/// A parse failure. Everything the plugin surfaces as
/// `404 Failed to load tiff file. [...]` originates here.
#[derive(Debug)]
pub struct TiffError(pub String);

impl std::fmt::Display for TiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<std::io::Error> for TiffError {
    fn from(value: std::io::Error) -> Self {
        TiffError(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, TiffError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(TiffError(message.into()))
}

/// One directory entry: the field type, the element count, and the value bytes in file
/// byte order.
#[derive(Debug, Clone)]
pub struct Entry {
    pub field_type: u16,
    pub count: u64,
    /// Raw value bytes, exactly `count * size_of(field_type)` long when the type is known.
    pub bytes: Vec<u8>,
    order: ByteOrder,
}

/// Byte width of each TIFF field type, or `None` for a type this reader does not know.
fn field_size(field_type: u16) -> Option<usize> {
    Some(match field_type {
        1 | 2 | 6 | 7 => 1,              // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,                      // SHORT, SSHORT
        4 | 9 | 11 | 13 => 4,            // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 | 16 | 17 | 18 => 8, // RATIONAL, SRATIONAL, DOUBLE, LONG8, SLONG8, IFD8
        _ => return None,
    })
}

impl Entry {
    /// Unsigned integer values, widened to `u64`. Rationals yield the numerator and the
    /// denominator as consecutive entries, which is how they are stored.
    pub fn as_u64_vec(&self) -> Vec<u64> {
        let order = self.order;
        macro_rules! collect {
            ($width:literal, $convert:expr) => {
                self.bytes
                    .chunks_exact($width)
                    .map(|chunk| {
                        let mut buf = [0u8; $width];
                        buf.copy_from_slice(chunk);
                        let convert: fn([u8; $width], ByteOrder) -> u64 = $convert;
                        convert(buf, order)
                    })
                    .collect()
            };
        }
        match self.field_type {
            1 | 2 | 6 | 7 => self.bytes.iter().map(|&b| u64::from(b)).collect(),
            3 | 8 => collect!(2, |b, o| u64::from(match o {
                ByteOrder::Little => u16::from_le_bytes(b),
                ByteOrder::Big => u16::from_be_bytes(b),
            })),
            4 | 9 | 11 | 13 => collect!(4, |b, o| u64::from(match o {
                ByteOrder::Little => u32::from_le_bytes(b),
                ByteOrder::Big => u32::from_be_bytes(b),
            })),
            5 | 10 => {
                // Split each 8-byte rational into its two 4-byte halves.
                self.bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(chunk);
                        u64::from(match order {
                            ByteOrder::Little => u32::from_le_bytes(buf),
                            ByteOrder::Big => u32::from_be_bytes(buf),
                        })
                    })
                    .collect()
            }
            12 | 16 | 17 | 18 => collect!(8, |b, o| match o {
                ByteOrder::Little => u64::from_le_bytes(b),
                ByteOrder::Big => u64::from_be_bytes(b),
            }),
            _ => Vec::new(),
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_u64_vec().first().copied()
    }

    /// A single rational as a float, the way `__rational_to_float` reads `XResolution`.
    pub fn as_rational(&self) -> Option<f64> {
        match self.field_type {
            5 | 10 => {
                let parts = self.as_u64_vec();
                let (num, den) = (*parts.first()?, *parts.get(1)?);
                if den == 0 {
                    // Python's `__rational_to_float` returns 0.0, which the caller then
                    // rejects as "zero resolution".
                    return Some(0.0);
                }
                Some(num as f64 / den as f64)
            }
            11 => {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(self.bytes.get(..4)?);
                Some(f64::from(match self.order {
                    ByteOrder::Little => f32::from_le_bytes(buf),
                    ByteOrder::Big => f32::from_be_bytes(buf),
                }))
            }
            12 => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(self.bytes.get(..8)?);
                Some(match self.order {
                    ByteOrder::Little => f64::from_le_bytes(buf),
                    ByteOrder::Big => f64::from_be_bytes(buf),
                })
            }
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// ASCII value with the trailing NUL stripped, as `tifffile` reports a tag's `value`.
    pub fn as_string(&self) -> String {
        let text = String::from_utf8_lossy(&self.bytes);
        text.trim_end_matches('\0').to_owned()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One image file directory: its tags and the offset of the next directory in the chain.
#[derive(Debug, Clone, Default)]
pub struct Ifd {
    pub entries: BTreeMap<u16, Entry>,
    pub next: Option<u64>,
    /// File offset this directory was read from — `page.offset` in tifffile.
    pub offset: u64,
}

impl Ifd {
    pub fn get(&self, tag: u16) -> Option<&Entry> {
        self.entries.get(&tag)
    }

    pub fn u64_or(&self, tag: u16, default: u64) -> u64 {
        self.get(tag).and_then(Entry::as_u64).unwrap_or(default)
    }

    pub fn u64_vec(&self, tag: u16) -> Vec<u64> {
        self.get(tag).map(Entry::as_u64_vec).unwrap_or_default()
    }

    pub fn string(&self, tag: u16) -> Option<String> {
        self.get(tag).map(Entry::as_string)
    }
}

/// The file, its byte order, and whether it is BigTIFF.
pub struct TiffSource {
    file: File,
    pub order: ByteOrder,
    pub big_tiff: bool,
    pub first_ifd: u64,
}

impl TiffSource {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header[..8])?;

        let order = match &header[..2] {
            b"II" => ByteOrder::Little,
            b"MM" => ByteOrder::Big,
            _ => return err("not a TIFF file"),
        };
        let read_u16 = |bytes: &[u8]| -> u16 {
            let buf = [bytes[0], bytes[1]];
            match order {
                ByteOrder::Little => u16::from_le_bytes(buf),
                ByteOrder::Big => u16::from_be_bytes(buf),
            }
        };
        let read_u32 = |bytes: &[u8]| -> u32 {
            let buf = [bytes[0], bytes[1], bytes[2], bytes[3]];
            match order {
                ByteOrder::Little => u32::from_le_bytes(buf),
                ByteOrder::Big => u32::from_be_bytes(buf),
            }
        };

        let (big_tiff, first_ifd) = match read_u16(&header[2..4]) {
            42 => (false, u64::from(read_u32(&header[4..8]))),
            43 => {
                // BigTIFF: offset size, a zero, then the 8-byte first-directory offset.
                if read_u16(&header[4..6]) != 8 {
                    return err("unsupported BigTIFF offset size");
                }
                file.read_exact(&mut header[8..16])?;
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&header[8..16]);
                let offset = match order {
                    ByteOrder::Little => u64::from_le_bytes(buf),
                    ByteOrder::Big => u64::from_be_bytes(buf),
                };
                (true, offset)
            }
            other => return err(format!("unsupported TIFF version {other}")),
        };

        Ok(Self {
            file,
            order,
            big_tiff,
            first_ifd,
        })
    }

    pub fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u16_at(&self, bytes: &[u8]) -> u16 {
        let buf = [bytes[0], bytes[1]];
        match self.order {
            ByteOrder::Little => u16::from_le_bytes(buf),
            ByteOrder::Big => u16::from_be_bytes(buf),
        }
    }

    fn u32_at(&self, bytes: &[u8]) -> u32 {
        let buf = [bytes[0], bytes[1], bytes[2], bytes[3]];
        match self.order {
            ByteOrder::Little => u32::from_le_bytes(buf),
            ByteOrder::Big => u32::from_be_bytes(buf),
        }
    }

    fn u64_at(&self, bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[..8]);
        match self.order {
            ByteOrder::Little => u64::from_le_bytes(buf),
            ByteOrder::Big => u64::from_be_bytes(buf),
        }
    }

    /// Reads the directory at `offset`, following value offsets for any entry whose value
    /// does not fit the inline slot.
    pub fn read_ifd(&mut self, offset: u64) -> Result<Ifd> {
        // Classic: 2-byte count, 12-byte entries, 4-byte next. BigTIFF: 8, 20, 8.
        let (count_size, entry_size) = if self.big_tiff {
            (8usize, 20usize)
        } else {
            (2, 12)
        };
        let count_bytes = self.read_at(offset, count_size)?;
        let count = if self.big_tiff {
            self.u64_at(&count_bytes)
        } else {
            u64::from(self.u16_at(&count_bytes))
        };
        if count > 65_536 {
            return err(format!("directory at {offset} claims {count} entries"));
        }

        let table = self.read_at(offset + count_size as u64, count as usize * entry_size)?;

        let mut entries = BTreeMap::new();
        for index in 0..count as usize {
            let raw = &table[index * entry_size..(index + 1) * entry_size];
            let tag = self.u16_at(&raw[0..2]);
            let field_type = self.u16_at(&raw[2..4]);
            let value_count = if self.big_tiff {
                self.u64_at(&raw[4..12])
            } else {
                u64::from(self.u32_at(&raw[4..8]))
            };
            let inline = if self.big_tiff {
                &raw[12..20]
            } else {
                &raw[8..12]
            };

            // An unknown field type still occupies its slot; skip it rather than guessing.
            let Some(size) = field_size(field_type) else {
                continue;
            };
            let Some(total) = value_count.checked_mul(size as u64) else {
                continue;
            };
            let bytes = if total <= inline.len() as u64 {
                inline[..total as usize].to_vec()
            } else {
                let value_offset = if self.big_tiff {
                    self.u64_at(inline)
                } else {
                    u64::from(self.u32_at(inline))
                };
                // A truncated file must not abort the whole open; tifffile leaves such a
                // tag unread and carries on.
                match self.read_at(value_offset, total as usize) {
                    Ok(bytes) => bytes,
                    Err(_) => continue,
                }
            };
            entries.insert(
                tag,
                Entry {
                    field_type,
                    count: value_count,
                    bytes,
                    order: self.order,
                },
            );
        }

        let next_at = offset + count_size as u64 + count * entry_size as u64;
        let next_size = if self.big_tiff { 8 } else { 4 };
        let next = match self.read_at(next_at, next_size) {
            Ok(bytes) => {
                let value = if self.big_tiff {
                    self.u64_at(&bytes)
                } else {
                    u64::from(self.u32_at(&bytes))
                };
                (value != 0).then_some(value)
            }
            Err(_) => None,
        };

        Ok(Ifd {
            entries,
            next,
            offset,
        })
    }

    /// The main directory chain, in file order.
    ///
    /// The walk stops if a chain revisits an offset, which a malformed file can do and
    /// which would otherwise loop forever.
    pub fn read_chain(&mut self) -> Result<Vec<Ifd>> {
        let mut out = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut next = Some(self.first_ifd);
        while let Some(offset) = next {
            if offset == 0 || !seen.insert(offset) {
                break;
            }
            let ifd = self.read_ifd(offset)?;
            next = ifd.next;
            out.push(ifd);
            if out.len() > 1 << 16 {
                break;
            }
        }
        if out.is_empty() {
            return err("no image file directory found");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn a_classic_little_endian_file_parses() {
        let mut source = TiffSource::open(&fixture("fluorescence.ome.tif")).unwrap();
        assert_eq!(source.order, ByteOrder::Little);
        assert!(!source.big_tiff);
        let chain = source.read_chain().unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].u64_or(tag::IMAGE_WIDTH, 0), 8);
        assert_eq!(chain[0].u64_or(tag::IMAGE_LENGTH, 0), 6);
        assert_eq!(chain[0].u64_or(tag::SAMPLES_PER_PIXEL, 1), 1);
        assert_eq!(chain[0].u64_vec(tag::STRIP_OFFSETS).len(), 1);
        assert!(chain[0]
            .string(tag::IMAGE_DESCRIPTION)
            .unwrap()
            .contains("OME"));
    }

    #[test]
    fn a_bigtiff_file_parses() {
        let mut source = TiffSource::open(&fixture("bigtiff.ome.tif")).unwrap();
        assert!(source.big_tiff);
        let chain = source.read_chain().unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].u64_or(tag::IMAGE_WIDTH, 0), 48);
        assert_eq!(chain[0].u64_or(tag::TILE_WIDTH, 0), 16);
        // 3x3 tiles for each of the three separate planes.
        assert_eq!(chain[0].u64_vec(tag::TILE_OFFSETS).len(), 27);
        assert_eq!(chain[0].u64_or(tag::PLANAR_CONFIGURATION, 1), 2);
    }

    #[test]
    fn a_big_endian_file_parses() {
        let mut source = TiffSource::open(&fixture("bigendian.ome.tif")).unwrap();
        assert_eq!(source.order, ByteOrder::Big);
        let chain = source.read_chain().unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].u64_or(tag::BITS_PER_SAMPLE, 0), 16);
        assert_eq!(chain[0].u64_or(tag::IMAGE_WIDTH, 0), 8);
    }

    #[test]
    fn sub_ifds_are_reachable() {
        let mut source = TiffSource::open(&fixture("pyramid.ome.tif")).unwrap();
        let chain = source.read_chain().unwrap();
        // The pyramid levels hang off the first directory, not the main chain.
        assert_eq!(chain.len(), 1);
        let sub = chain[0].u64_vec(tag::SUB_IFDS);
        assert_eq!(sub.len(), 2);
        let level1 = source.read_ifd(sub[0]).unwrap();
        let level2 = source.read_ifd(sub[1]).unwrap();
        assert_eq!(level1.u64_or(tag::IMAGE_WIDTH, 0), 80);
        assert_eq!(level2.u64_or(tag::IMAGE_WIDTH, 0), 40);
    }

    #[test]
    fn resolution_tags_read_as_rationals() {
        let mut source = TiffSource::open(&fixture("pixelsize_um.ome.tif")).unwrap();
        let chain = source.read_chain().unwrap();
        let x = chain[0].get(tag::X_RESOLUTION).unwrap();
        assert_eq!(x.field_type, 5);
        // tifffile writes the OME pixel size into the resolution tags as well.
        assert!(x.as_rational().unwrap() > 0.0);
    }

    #[test]
    fn a_non_tiff_file_is_rejected() {
        let path = std::env::temp_dir().join("wsi-tifffile-not-a-tiff.tif");
        std::fs::write(&path, b"nope").unwrap();
        let error = match TiffSource::open(&path) {
            Ok(_) => panic!("expected an error"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("TIFF") || error.to_string().contains("fill"));
    }
}
