//! A DICOM reader, narrow by design.
//!
//! `wsidicom` is built on `pydicom`, a complete DICOM implementation. Nothing like that is
//! needed here: the plugin reads whole-slide image instances, and every one of those in the
//! test collection — 23 instances across three vendors — is `TILED_FULL` with either JPEG
//! baseline or no compression at all. So this parses the file meta group, the handful of
//! elements the plugin reads, and the encapsulated pixel data, and refuses anything else
//! with a message that names what it found.
//!
//! What it deliberately does not do:
//!
//! - **Implicit VR.** The transfer syntaxes in use (`1.2.840.10008.1.2.1` and
//!   `1.2.840.10008.1.2.4.50`) both encode the dataset in explicit VR little endian.
//! - **Big endian.** Retired from the standard, and not present here.
//! - **Reading the frames eagerly.** A single instance runs to 800 MB. Only the element
//!   headers and the frame offsets are read at open time; a frame is read when it is asked
//!   for.
//! - **Per-frame functional groups.** `TILED_FULL` lays frames out implicitly in row-major
//!   order, so the position of frame *n* is arithmetic rather than a lookup.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// A DICOM tag, `(group, element)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tag(pub u16, pub u16);

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:04X},{:04X})", self.0, self.1)
    }
}

pub mod tags {
    use super::Tag;

    pub const TRANSFER_SYNTAX_UID: Tag = Tag(0x0002, 0x0010);
    pub const IMAGE_TYPE: Tag = Tag(0x0008, 0x0008);
    pub const SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x0018);
    pub const DIMENSION_ORGANIZATION_TYPE: Tag = Tag(0x0020, 0x9311);
    pub const SAMPLES_PER_PIXEL: Tag = Tag(0x0028, 0x0002);
    pub const PHOTOMETRIC_INTERPRETATION: Tag = Tag(0x0028, 0x0004);
    pub const PLANAR_CONFIGURATION: Tag = Tag(0x0028, 0x0006);
    pub const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
    pub const ROWS: Tag = Tag(0x0028, 0x0010);
    pub const COLUMNS: Tag = Tag(0x0028, 0x0011);
    pub const PIXEL_SPACING: Tag = Tag(0x0028, 0x0030);
    pub const BITS_ALLOCATED: Tag = Tag(0x0028, 0x0100);
    pub const BITS_STORED: Tag = Tag(0x0028, 0x0101);
    pub const ICC_PROFILE: Tag = Tag(0x0028, 0x2000);
    pub const PIXEL_MEASURES_SEQUENCE: Tag = Tag(0x0028, 0x9110);
    pub const TOTAL_PIXEL_MATRIX_COLUMNS: Tag = Tag(0x0048, 0x0006);
    pub const TOTAL_PIXEL_MATRIX_ROWS: Tag = Tag(0x0048, 0x0007);
    pub const OPTICAL_PATH_SEQUENCE: Tag = Tag(0x0048, 0x0105);
    pub const SHARED_FUNCTIONAL_GROUPS_SEQUENCE: Tag = Tag(0x5200, 0x9229);
    pub const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);

    /// Delimiters, which carry a length but never a value.
    pub const ITEM: Tag = Tag(0xFFFE, 0xE000);
    pub const ITEM_DELIMITATION: Tag = Tag(0xFFFE, 0xE00D);
    pub const SEQUENCE_DELIMITATION: Tag = Tag(0xFFFE, 0xE0DD);
}

pub mod transfer_syntax {
    pub const IMPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2";
    pub const EXPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2.1";
    pub const JPEG_BASELINE: &str = "1.2.840.10008.1.2.4.50";
}

/// The value representations that reserve two bytes and then carry a 32-bit length.
///
/// PS3.5 table 7.1-1. Getting this list wrong is not a graceful failure: a long-form VR
/// read as short form takes the low half of its length as the whole of it and the parse
/// lands in the middle of the value. `UC` is the one that found this — Leica writes
/// `(0008,0119)` with it — and the six beside it were missing for the same reason.
fn is_long_form(vr: &[u8; 2]) -> bool {
    matches!(
        vr,
        b"OB"
            | b"OD"
            | b"OF"
            | b"OL"
            | b"OV"
            | b"OW"
            | b"SQ"
            | b"SV"
            | b"UC"
            | b"UN"
            | b"UR"
            | b"UT"
            | b"UV"
    )
}

#[derive(Debug)]
pub struct DicomError(pub String);

impl std::fmt::Display for DicomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DicomError {}

pub type Result<T> = std::result::Result<T, DicomError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(DicomError(message.into()))
}

/// One element's value, kept as the bytes that were on disk plus its VR.
///
/// Values are small — the largest is a 13 MB ICC profile — with the single exception of
/// `PixelData`, which is never stored here; see [`PixelData`].
#[derive(Debug, Clone)]
pub struct Value {
    pub vr: [u8; 2],
    pub bytes: Vec<u8>,
}

impl Value {
    /// A string value with DICOM's padding removed. Multi-valued strings keep their
    /// backslashes; use [`Value::strings`] to split them.
    pub fn string(&self) -> String {
        String::from_utf8_lossy(&self.bytes)
            .trim_end_matches(['\0', ' '])
            .to_owned()
    }

    /// A `\`-separated string value, split. `ImageType` and `PixelSpacing` are both this.
    pub fn strings(&self) -> Vec<String> {
        self.string().split('\\').map(str::to_owned).collect()
    }

    /// An unsigned short (`US`).
    pub fn u16(&self) -> Option<u16> {
        self.bytes
            .get(..2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    }

    /// An integer however it was encoded: `US`, `UL` and `IS` all appear on the tags this
    /// reads, and `IS` is a *string* of digits.
    pub fn integer(&self) -> Option<i64> {
        match &self.vr {
            b"US" => self.u16().map(i64::from),
            b"UL" => self
                .bytes
                .get(..4)
                .map(|b| i64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            b"SS" => self
                .bytes
                .get(..2)
                .map(|b| i64::from(i16::from_le_bytes([b[0], b[1]]))),
            b"SL" => self
                .bytes
                .get(..4)
                .map(|b| i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            // `IS` and `DS` are decimal strings.
            _ => self.string().trim().parse().ok(),
        }
    }

    /// A decimal string (`DS`), which is how `PixelSpacing` is encoded.
    pub fn decimals(&self) -> Vec<f64> {
        self.strings()
            .iter()
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }
}

/// A parsed dataset: elements by tag, plus any sequences nested inside.
#[derive(Debug, Default, Clone)]
pub struct Dataset {
    elements: std::collections::BTreeMap<Tag, Value>,
    sequences: std::collections::BTreeMap<Tag, Vec<Dataset>>,
}

impl Dataset {
    pub fn get(&self, tag: Tag) -> Option<&Value> {
        self.elements.get(&tag)
    }

    pub fn sequence(&self, tag: Tag) -> Option<&[Dataset]> {
        self.sequences.get(&tag).map(Vec::as_slice)
    }

    pub fn string(&self, tag: Tag) -> Option<String> {
        self.get(tag).map(Value::string)
    }

    pub fn integer(&self, tag: Tag) -> Option<i64> {
        self.get(tag).and_then(Value::integer)
    }

    /// Walks this dataset and everything nested in it, depth first.
    ///
    /// The Python plugin's comment about the ICC profile — "Direct access does not work,
    /// can be nested, search" — describes exactly this.
    ///
    /// The depth is bounded for the same reason the parse is: this recurses, and a parsed
    /// dataset's nesting came from the file. [`MAX_SEQUENCE_DEPTH`] already caps what the
    /// parser will build, so this cannot be reached through a parsed file — it is a second
    /// guard on a function that is `pub` and could be handed a dataset built elsewhere.
    pub fn find_nested(&self, tag: Tag) -> Option<&Value> {
        self.find_nested_within(tag, MAX_SEQUENCE_DEPTH)
    }

    fn find_nested_within(&self, tag: Tag, remaining: usize) -> Option<&Value> {
        if let Some(value) = self.get(tag) {
            return Some(value);
        }
        if remaining == 0 {
            return None;
        }
        for datasets in self.sequences.values() {
            for dataset in datasets {
                if let Some(value) = dataset.find_nested_within(tag, remaining - 1) {
                    return Some(value);
                }
            }
        }
        None
    }
}

/// Where one frame is.
///
/// A frame found by walking the items is fully located at parse time. A frame found through
/// the basic offset table is not: the table gives the position of its 8-byte item header
/// and nothing else, and for an 800 MB instance that header is far past the buffered head
/// of the file. Its length is read with the header when the frame is read.
#[derive(Debug, Clone, Copy)]
pub enum FrameLocation {
    Known { offset: u64, length: usize },
    Item { header_at: u64 },
}

/// Where each frame's bytes are, without holding any of them.
#[derive(Debug, Clone)]
pub enum PixelData {
    /// Encapsulated: the value is a sequence of items, and for these files each item is one
    /// frame.
    Encapsulated { frames: Vec<FrameLocation> },
    /// Native: one contiguous block, frames of equal size laid out in order.
    Native {
        offset: u64,
        frame_bytes: usize,
        frames: usize,
    },
}

impl PixelData {
    pub fn frame_count(&self) -> usize {
        match self {
            PixelData::Encapsulated { frames } => frames.len(),
            PixelData::Native { frames, .. } => *frames,
        }
    }

    fn locate(&self, index: usize) -> Option<FrameLocation> {
        match self {
            PixelData::Encapsulated { frames } => frames.get(index).copied(),
            PixelData::Native {
                offset,
                frame_bytes,
                frames,
            } => (index < *frames).then(|| FrameLocation::Known {
                offset: offset + (index * frame_bytes) as u64,
                length: *frame_bytes,
            }),
        }
    }
}

/// One DICOM instance: its dataset, and where its frames are in the file.
#[derive(Debug)]
pub struct Instance {
    pub path: PathBuf,
    pub transfer_syntax: String,
    pub dataset: Dataset,
    pub pixel_data: Option<PixelData>,
}

impl Instance {
    /// Reads one frame's bytes. The file is reopened per call rather than held open, which
    /// keeps an open directory from costing one descriptor per instance for the lifetime of
    /// the slide — a MIRAX directory has twelve.
    pub fn read_frame(&self, index: usize) -> Result<Vec<u8>> {
        let Some(pixel_data) = &self.pixel_data else {
            return err("instance has no pixel data");
        };
        let Some(location) = pixel_data.locate(index) else {
            return err(format!(
                "frame {index} is past the end of {} frames",
                pixel_data.frame_count()
            ));
        };
        let mut file = File::open(&self.path)
            .map_err(|e| DicomError(format!("{}: {e}", self.path.display())))?;
        let file_bytes = file
            .metadata()
            .map_err(|e| DicomError(format!("{}: {e}", self.path.display())))?
            .len();

        let (offset, length) = match location {
            FrameLocation::Known { offset, length } => (offset, length),
            FrameLocation::Item { header_at } => {
                file.seek(SeekFrom::Start(header_at))
                    .map_err(|e| DicomError(format!("seek to frame {index}: {e}")))?;
                let mut header = [0u8; 8];
                file.read_exact(&mut header)
                    .map_err(|e| DicomError(format!("read frame {index} header: {e}")))?;
                let tag = Tag(
                    u16::from_le_bytes([header[0], header[1]]),
                    u16::from_le_bytes([header[2], header[3]]),
                );
                if tag != tags::ITEM {
                    return err(format!(
                        "the offset table points at {tag} for frame {index}, not an item"
                    ));
                }
                let length =
                    u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
                (header_at + 8, length)
            }
        };

        // The length is a field in the file, so it is checked before it is believed: a
        // corrupt item header must fail the request, not allocate gigabytes for it.
        if length > MAX_FRAME_BYTES {
            return err(format!(
                "frame {index} declares {length} bytes, past the {MAX_FRAME_BYTES} limit"
            ));
        }
        if offset + length as u64 > file_bytes {
            return err(format!(
                "frame {index} claims bytes {offset}..{} of a {file_bytes}-byte file",
                offset + length as u64
            ));
        }

        file.seek(SeekFrom::Start(offset))
            .map_err(|e| DicomError(format!("seek to frame {index}: {e}")))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|e| DicomError(format!("read frame {index}: {e}")))?;
        Ok(bytes)
    }
}

/// How much of a file to buffer while parsing the element headers.
///
/// Everything the plugin reads sits before `PixelData`, and `PixelData` is where the size
/// is. 16 MB covers the largest value in the collection — Leica's 13 MB ICC profile — with
/// room to spare, and means a 1 GB instance is opened with one bounded read.
const HEADER_BUDGET: usize = 16 * 1024 * 1024;

/// How deeply sequences may nest before the parse is refused.
///
/// Sequences nest legitimately — `PixelSpacing` sits two levels down — but the depth is
/// attacker-controlled in a file that is merely corrupt, and the parser recurses. Without a
/// limit a file with a few thousand nested `SQ` elements overflows the stack, which aborts
/// the process rather than failing the request: a slide directory that a client can cause
/// to be read would take the whole service down. Real files do not nest past four.
const MAX_SEQUENCE_DEPTH: usize = 16;

/// The largest single frame this will allocate for.
///
/// A frame's length comes from its own item header, which is a 32-bit field in the file, so
/// a corrupt one can ask for up to 4 GB. The largest frame the collection holds is a
/// 1463x598 overview, well under a megabyte; 256 MB is far past anything a tile needs while
/// still refusing a length that is plainly wrong. The declared length is also checked
/// against what the file actually holds, which catches the ordinary truncation case.
const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

/// The largest number of frames an offset table may declare.
///
/// The table is bounded by [`HEADER_BUDGET`], so at four bytes an entry it can name four
/// million frames and each costs sixteen bytes to record. The largest instance here has
/// 81,420. A million is both unreachable in practice and bounded in memory.
const MAX_FRAMES: usize = 1_000_000;

pub fn read_instance(path: &Path) -> Result<Instance> {
    let mut file = File::open(path).map_err(|e| DicomError(format!("{}: {e}", path.display())))?;

    let mut head = vec![0u8; HEADER_BUDGET];
    let read = read_up_to(&mut file, &mut head)?;
    head.truncate(read);

    let mut reader = Cursor::new(&head);
    reader.expect_preamble()?;
    let meta = reader.parse_file_meta()?;

    let transfer_syntax = meta
        .string(tags::TRANSFER_SYNTAX_UID)
        .ok_or_else(|| DicomError("file meta group has no TransferSyntaxUID".into()))?;

    match transfer_syntax.as_str() {
        transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN | transfer_syntax::JPEG_BASELINE => {}
        transfer_syntax::IMPLICIT_VR_LITTLE_ENDIAN => {
            return err(
                "transfer syntax 1.2.840.10008.1.2 (implicit VR) is not supported; \
                 every whole-slide instance seen uses explicit VR",
            )
        }
        other => {
            return err(format!(
                "transfer syntax {other} is not supported; this reader handles \
                 explicit VR little endian and JPEG baseline"
            ))
        }
    }

    let (dataset, pixel_data) = reader.parse_dataset(&transfer_syntax)?;

    Ok(Instance {
        path: path.to_owned(),
        transfer_syntax,
        dataset,
        pixel_data,
    })
}

/// `Read::read_exact`, but a short file is not an error — small instances exist.
fn read_up_to(file: &mut File, buffer: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match file.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return err(format!("read failed: {e}")),
        }
    }
    Ok(filled)
}

/// A position in the buffered head of the file.
///
/// Offsets recorded into [`PixelData`] are positions in the *file*, which is the same thing
/// because the buffer starts at byte zero.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return err(format!(
                "truncated at byte {}: wanted {n} more, {} available",
                self.at,
                self.remaining()
            ));
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// The 128-byte preamble and the `DICM` magic.
    fn expect_preamble(&mut self) -> Result<()> {
        if self.remaining() < 132 {
            return err("file is too short to be DICOM");
        }
        self.at = 128;
        let magic = self.take(4)?;
        if magic != b"DICM" {
            return err("missing DICM magic; not a Part 10 DICOM file");
        }
        Ok(())
    }

    /// The file meta group, which is always explicit VR little endian.
    ///
    /// Its length is in `(0002,0000)`, so the dataset's start is known without parsing the
    /// whole group — but the group is a few hundred bytes, so it is parsed anyway and the
    /// length used only to find where to stop.
    fn parse_file_meta(&mut self) -> Result<Dataset> {
        let (tag, value) = self.read_explicit_element()?;
        if tag != Tag(0x0002, 0x0000) {
            return err(format!(
                "file meta group does not start with (0002,0000), found {tag}"
            ));
        }
        let group_length = value
            .and_then(|v| v.integer())
            .ok_or_else(|| DicomError("(0002,0000) has no length".into()))?;
        let end = self.at + group_length as usize;

        let mut meta = Dataset::default();
        while self.at < end && self.remaining() > 0 {
            let (tag, value) = self.read_explicit_element()?;
            if let Some(value) = value {
                meta.elements.insert(tag, value);
            }
        }
        Ok(meta)
    }

    /// One explicit-VR element. Returns `None` for a value this reader skips over — a
    /// sequence, or something longer than the buffer.
    fn read_explicit_element(&mut self) -> Result<(Tag, Option<Value>)> {
        let group = self.u16()?;
        let element = self.u16()?;
        let tag = Tag(group, element);

        // Items and delimiters have a 32-bit length and no VR, whatever the syntax.
        if group == 0xFFFE {
            let length = self.u32()?;
            return Ok((
                tag,
                Some(Value {
                    vr: *b"na",
                    bytes: length.to_le_bytes().to_vec(),
                }),
            ));
        }

        let vr: [u8; 2] = self.take(2)?.try_into().expect("two bytes");
        let length = if is_long_form(&vr) {
            self.take(2)?;
            self.u32()?
        } else {
            u32::from(self.u16()?)
        };

        if length == 0xFFFF_FFFF {
            // Undefined length: a sequence or encapsulated pixel data. The caller decides.
            return Ok((tag, None));
        }

        let bytes = self.take(length as usize)?.to_vec();
        Ok((tag, Some(Value { vr, bytes })))
    }

    /// The main dataset, stopping at `PixelData` and recording where its frames are.
    fn parse_dataset(&mut self, transfer_syntax: &str) -> Result<(Dataset, Option<PixelData>)> {
        self.parse_dataset_at_depth(transfer_syntax, 0)
    }

    fn parse_dataset_at_depth(
        &mut self,
        transfer_syntax: &str,
        depth: usize,
    ) -> Result<(Dataset, Option<PixelData>)> {
        let mut dataset = Dataset::default();
        let mut pixel_data = None;

        while self.remaining() >= 8 {
            let before = self.at;
            let group = u16::from_le_bytes([self.bytes[self.at], self.bytes[self.at + 1]]);
            let element = u16::from_le_bytes([self.bytes[self.at + 2], self.bytes[self.at + 3]]);
            let tag = Tag(group, element);

            if tag == tags::PIXEL_DATA {
                pixel_data = Some(self.parse_pixel_data(&dataset, transfer_syntax)?);
                break;
            }

            // Peek the VR to spot a sequence before consuming it.
            let vr: [u8; 2] = [self.bytes[self.at + 4], self.bytes[self.at + 5]];
            if &vr == b"SQ" {
                self.at = before;
                let datasets = self.parse_sequence(depth + 1)?;
                dataset.sequences.insert(tag, datasets);
                continue;
            }

            self.at = before;
            match self.read_explicit_element() {
                Ok((tag, Some(value))) => {
                    dataset.elements.insert(tag, value);
                }
                // An undefined-length value with a VR other than SQ: an implicit sequence.
                // Skip it by reading to its delimiter.
                Ok((_, None)) => {
                    self.skip_undefined_length()?;
                }
                // Running off the end of the buffered head is expected for a large
                // instance whose pixel data starts beyond it, and is only an error if
                // `PixelData` was never found. The caller checks.
                Err(_) => break,
            }
        }

        Ok((dataset, pixel_data))
    }

    /// A sequence with either an explicit length or a delimiter.
    fn parse_sequence(&mut self, depth: usize) -> Result<Vec<Dataset>> {
        if depth > MAX_SEQUENCE_DEPTH {
            return err(format!(
                "sequences nest more than {MAX_SEQUENCE_DEPTH} deep; refusing to recurse \
                 further"
            ));
        }
        let _group = self.u16()?;
        let _element = self.u16()?;
        let vr = self.take(2)?;
        debug_assert_eq!(vr, b"SQ");
        self.take(2)?;
        let length = self.u32()?;

        let end = if length == 0xFFFF_FFFF {
            None
        } else {
            Some(self.at + length as usize)
        };

        let mut items = Vec::new();
        loop {
            if let Some(end) = end {
                if self.at >= end {
                    break;
                }
            }
            if self.remaining() < 8 {
                break;
            }
            let tag = Tag(self.u16()?, self.u16()?);
            let item_length = self.u32()?;
            if tag == tags::SEQUENCE_DELIMITATION {
                break;
            }
            if tag != tags::ITEM {
                return err(format!("expected an item in a sequence, found {tag}"));
            }

            let item_end = if item_length == 0xFFFF_FFFF {
                None
            } else {
                Some(self.at + item_length as usize)
            };

            let mut item = Dataset::default();
            loop {
                match item_end {
                    Some(item_end) if self.at >= item_end => break,
                    _ => {}
                }
                if self.remaining() < 8 {
                    break;
                }
                let peek_tag = Tag(
                    u16::from_le_bytes([self.bytes[self.at], self.bytes[self.at + 1]]),
                    u16::from_le_bytes([self.bytes[self.at + 2], self.bytes[self.at + 3]]),
                );
                if peek_tag == tags::ITEM_DELIMITATION {
                    self.at += 8;
                    break;
                }
                let before = self.at;
                let vr: [u8; 2] = [self.bytes[self.at + 4], self.bytes[self.at + 5]];
                if &vr == b"SQ" {
                    self.at = before;
                    let nested = self.parse_sequence(depth + 1)?;
                    item.sequences.insert(peek_tag, nested);
                    continue;
                }
                self.at = before;
                match self.read_explicit_element()? {
                    (tag, Some(value)) => {
                        item.elements.insert(tag, value);
                    }
                    (_, None) => self.skip_undefined_length()?,
                }
            }
            items.push(item);
        }
        Ok(items)
    }

    /// Walks forward to the delimiter that closes an undefined-length value.
    fn skip_undefined_length(&mut self) -> Result<()> {
        let mut depth = 1usize;
        while self.remaining() >= 8 {
            let tag = Tag(self.u16()?, self.u16()?);
            let length = self.u32()?;
            match tag {
                tags::SEQUENCE_DELIMITATION => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                tags::ITEM if length == 0xFFFF_FFFF => depth += 1,
                tags::ITEM => self.at += length as usize,
                _ if length == 0xFFFF_FFFF => depth += 1,
                _ => self.at += length as usize,
            }
        }
        err("ran off the end looking for a sequence delimiter")
    }

    /// `PixelData`, in either of its two forms.
    ///
    /// Encapsulated data opens with a basic offset table, which may be empty; the frames
    /// are the items that follow it. Native data is one block whose frames are implied by
    /// the geometry, which is why the dataset so far is passed in.
    fn parse_pixel_data(&mut self, dataset: &Dataset, transfer_syntax: &str) -> Result<PixelData> {
        let _tag = (self.u16()?, self.u16()?);
        let vr: [u8; 2] = self.take(2)?.try_into().expect("two bytes");
        let length = if is_long_form(&vr) {
            self.take(2)?;
            self.u32()?
        } else {
            u32::from(self.u16()?)
        };

        if length != 0xFFFF_FFFF {
            // Native. Frames are equal-sized and contiguous.
            let frames = dataset.integer(tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as usize;
            let frame_bytes = length as usize / frames;
            if transfer_syntax != transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN {
                return err(format!(
                    "pixel data has a defined length under transfer syntax \
                     {transfer_syntax}, which encapsulates"
                ));
            }
            return Ok(PixelData::Native {
                offset: self.at as u64,
                frame_bytes,
                frames,
            });
        }

        // Encapsulated. The first item is the basic offset table.
        let mut frames = Vec::new();
        let mut offsets_from_table: Option<Vec<u32>> = None;
        let mut first = true;
        while self.remaining() >= 8 {
            let tag = Tag(self.u16()?, self.u16()?);
            let item_length = self.u32()? as usize;
            if tag == tags::SEQUENCE_DELIMITATION {
                break;
            }
            if tag != tags::ITEM {
                return err(format!("expected a pixel-data item, found {tag}"));
            }

            if first {
                first = false;
                // The table is a list of 32-bit offsets relative to the end of itself. An
                // empty table is legal and common.
                if item_length >= 4 {
                    let table = self.take(item_length)?;
                    offsets_from_table = Some(
                        table
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    );
                } else {
                    self.take(item_length)?;
                }
                // Everything after the table is a frame, and this is where its offsets are
                // measured from.
                let base = self.at;
                if let Some(offsets) = &offsets_from_table {
                    if offsets.len() > MAX_FRAMES {
                        return err(format!(
                            "the offset table declares {} frames, past the {MAX_FRAMES} \
                             limit",
                            offsets.len()
                        ));
                    }
                    // With a table, every frame is located arithmetically and nothing more
                    // of the file is read here. That is what makes an 800 MB instance
                    // openable: its 81,420 frames are far past the buffered head, so
                    // walking the items is not an option.
                    //
                    // The table's offsets are measured from the first byte after itself,
                    // and each points at the frame's own item header rather than its data.
                    frames.extend(offsets.iter().map(|offset| FrameLocation::Item {
                        header_at: (base + *offset as usize) as u64,
                    }));
                    return Ok(PixelData::Encapsulated { frames });
                }
                continue;
            }

            frames.push(FrameLocation::Known {
                offset: self.at as u64,
                length: item_length,
            });
            self.take(item_length)?;
        }

        if frames.is_empty() {
            return err("encapsulated pixel data has no frames");
        }
        Ok(PixelData::Encapsulated { frames })
    }
}

/// Lists the `*.dcm` files of a directory, sorted so that an open is reproducible.
pub fn dicom_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(directory)
        .map_err(|e| DicomError(format!("{}: {e}", directory.display())))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("dcm"))
        })
        .collect();
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal Part 10 file: preamble, magic, file meta group, then a dataset.
    fn file_with(transfer_syntax: &str, dataset: &[u8]) -> Vec<u8> {
        let mut meta = Vec::new();
        // (0002,0010) UI TransferSyntaxUID
        let mut uid = transfer_syntax.as_bytes().to_vec();
        if uid.len() % 2 == 1 {
            uid.push(0);
        }
        meta.extend_from_slice(&[0x02, 0x00, 0x10, 0x00]);
        meta.extend_from_slice(b"UI");
        meta.extend_from_slice(&(uid.len() as u16).to_le_bytes());
        meta.extend_from_slice(&uid);

        let mut out = vec![0u8; 128];
        out.extend_from_slice(b"DICM");
        // (0002,0000) UL group length
        out.extend_from_slice(&[0x02, 0x00, 0x00, 0x00]);
        out.extend_from_slice(b"UL");
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta);
        out.extend_from_slice(dataset);
        out
    }

    fn element(tag: Tag, vr: &[u8; 2], value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&tag.0.to_le_bytes());
        out.extend_from_slice(&tag.1.to_le_bytes());
        out.extend_from_slice(vr);
        if is_long_form(vr) {
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        } else {
            out.extend_from_slice(&(value.len() as u16).to_le_bytes());
        }
        out.extend_from_slice(value);
        out
    }

    fn write(bytes: &[u8]) -> (tempdir::TempDir, PathBuf) {
        let dir = tempdir::TempDir::new();
        let path = dir.path().join("instance.dcm");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    /// A temporary directory, since the crate has no dev-dependency for one.
    mod tempdir {
        use std::path::{Path, PathBuf};

        pub struct TempDir(PathBuf);

        impl TempDir {
            pub fn new() -> Self {
                let base = std::env::temp_dir().join(format!(
                    "wsi-dicom-test-{}-{:?}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&base).unwrap();
                Self(base)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn a_native_instance_parses_and_locates_its_single_frame() {
        let mut dataset = Vec::new();
        dataset.extend(element(
            tags::IMAGE_TYPE,
            b"CS",
            b"ORIGINAL\\PRIMARY\\LABEL\\NONE",
        ));
        dataset.extend(element(tags::SAMPLES_PER_PIXEL, b"US", &3u16.to_le_bytes()));
        dataset.extend(element(tags::PHOTOMETRIC_INTERPRETATION, b"CS", b"RGB "));
        dataset.extend(element(tags::ROWS, b"US", &2u16.to_le_bytes()));
        dataset.extend(element(tags::COLUMNS, b"US", &2u16.to_le_bytes()));
        dataset.extend(element(tags::BITS_ALLOCATED, b"US", &8u16.to_le_bytes()));
        let pixels: Vec<u8> = (0..12).collect();
        dataset.extend(element(tags::PIXEL_DATA, b"OB", &pixels));

        let (_dir, path) = write(&file_with(
            transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN,
            &dataset,
        ));
        let instance = read_instance(&path).unwrap();

        assert_eq!(
            instance.dataset.get(tags::IMAGE_TYPE).unwrap().strings(),
            ["ORIGINAL", "PRIMARY", "LABEL", "NONE"]
        );
        // Trailing-space padding is DICOM's, not part of the value.
        assert_eq!(
            instance
                .dataset
                .string(tags::PHOTOMETRIC_INTERPRETATION)
                .unwrap(),
            "RGB"
        );
        assert_eq!(instance.dataset.integer(tags::ROWS), Some(2));
        assert_eq!(instance.pixel_data.as_ref().unwrap().frame_count(), 1);
        assert_eq!(instance.read_frame(0).unwrap(), pixels);
        assert!(instance.read_frame(1).is_err());
    }

    #[test]
    fn an_encapsulated_instance_walks_its_items_when_the_offset_table_is_empty() {
        let mut dataset = Vec::new();
        dataset.extend(element(tags::NUMBER_OF_FRAMES, b"IS", b"2 "));
        // (7FE0,0010) OB, undefined length, then an empty table and two frames.
        dataset.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00]);
        dataset.extend_from_slice(b"OB");
        dataset.extend_from_slice(&[0, 0]);
        dataset.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // Empty basic offset table.
        dataset.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        dataset.extend_from_slice(&0u32.to_le_bytes());
        for frame in [b"first-frame".as_slice(), b"second!!".as_slice()] {
            dataset.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
            dataset.extend_from_slice(&(frame.len() as u32).to_le_bytes());
            dataset.extend_from_slice(frame);
        }
        dataset.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0]);
        dataset.extend_from_slice(&0u32.to_le_bytes());

        let (_dir, path) = write(&file_with(transfer_syntax::JPEG_BASELINE, &dataset));
        let instance = read_instance(&path).unwrap();
        let pixel_data = instance.pixel_data.as_ref().unwrap();
        assert_eq!(pixel_data.frame_count(), 2);
        assert_eq!(instance.read_frame(0).unwrap(), b"first-frame");
        assert_eq!(instance.read_frame(1).unwrap(), b"second!!");
    }

    #[test]
    fn an_offset_table_locates_frames_without_walking_them() {
        // The table is what makes a large instance openable: its frames are past the
        // buffered head, so they are located arithmetically rather than by reading through.
        let frames: [&[u8]; 3] = [b"aaaa", b"bb", b"cccccc"];
        let mut items = Vec::new();
        let mut offsets = Vec::new();
        for frame in frames {
            offsets.push(items.len() as u32);
            items.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
            items.extend_from_slice(&(frame.len() as u32).to_le_bytes());
            items.extend_from_slice(frame);
        }

        let mut dataset = Vec::new();
        dataset.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00]);
        dataset.extend_from_slice(b"OB");
        dataset.extend_from_slice(&[0, 0]);
        dataset.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let table: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
        dataset.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        dataset.extend_from_slice(&(table.len() as u32).to_le_bytes());
        dataset.extend_from_slice(&table);
        dataset.extend_from_slice(&items);

        let (_dir, path) = write(&file_with(transfer_syntax::JPEG_BASELINE, &dataset));
        let instance = read_instance(&path).unwrap();
        assert_eq!(instance.pixel_data.as_ref().unwrap().frame_count(), 3);
        for (index, expected) in frames.iter().enumerate() {
            assert_eq!(&instance.read_frame(index).unwrap(), expected);
        }
    }

    #[test]
    fn a_nested_sequence_is_parsed_and_searchable() {
        // PixelSpacing lives two levels down, in
        // SharedFunctionalGroupsSequence > PixelMeasuresSequence.
        let inner = element(tags::PIXEL_SPACING, b"DS", b"0.000499\\0.000499");
        let mut item = Vec::new();
        item.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        item.extend_from_slice(&(inner.len() as u32).to_le_bytes());
        item.extend_from_slice(&inner);
        let measures = element(tags::PIXEL_MEASURES_SEQUENCE, b"SQ", &item);

        let mut outer_item = Vec::new();
        outer_item.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        outer_item.extend_from_slice(&(measures.len() as u32).to_le_bytes());
        outer_item.extend_from_slice(&measures);
        let shared = element(tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE, b"SQ", &outer_item);

        let (_dir, path) = write(&file_with(
            transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN,
            &shared,
        ));
        let instance = read_instance(&path).unwrap();

        let spacing = instance
            .dataset
            .find_nested(tags::PIXEL_SPACING)
            .unwrap()
            .decimals();
        assert_eq!(spacing, vec![0.000499, 0.000499]);
    }

    /// A long-form VR read as short form takes the low half of its 32-bit length as the
    /// whole of it, and the parse resumes inside the value. Leica writes `(0008,0119)` as
    /// `UC`, which is how this was found: everything after it was garbage, and the parse
    /// failed 13 MB later with a length that pointed past the end of the file.
    #[test]
    fn every_long_form_vr_is_read_with_a_32_bit_length() {
        for vr in [
            b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV",
        ] {
            // A value long enough that a 16-bit misread would land inside it, followed by
            // a tag the parser has to still find.
            let value = [b'x'; 300];
            let mut dataset = element(Tag(0x0008, 0x0119), vr, &value);
            dataset.extend(element(tags::ROWS, b"US", &7u16.to_le_bytes()));

            let (_dir, path) = write(&file_with(
                transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN,
                &dataset,
            ));
            let instance = read_instance(&path).unwrap();
            assert_eq!(
                instance.dataset.integer(tags::ROWS),
                Some(7),
                "the element after a {} value was lost",
                String::from_utf8_lossy(vr)
            );
        }
    }

    #[test]
    fn an_unsupported_transfer_syntax_is_refused_by_name() {
        let (_dir, path) = write(&file_with("1.2.840.10008.1.2.4.90", &[]));
        let error = read_instance(&path).unwrap_err();
        assert!(
            error.0.contains("1.2.840.10008.1.2.4.90"),
            "the message has to name what it found: {error}"
        );

        let (_dir, path) = write(&file_with(transfer_syntax::IMPLICIT_VR_LITTLE_ENDIAN, &[]));
        let error = read_instance(&path).unwrap_err();
        assert!(error.0.contains("implicit VR"), "{error}");
    }

    /// Deeply nested sequences must fail the parse, not overflow the stack. A stack
    /// overflow aborts the process, which would take the whole service down rather than
    /// failing one request.
    #[test]
    fn sequences_nested_past_the_limit_are_refused() {
        // Build `MAX_SEQUENCE_DEPTH + 4` sequences inside one another, innermost first.
        let mut nested = Vec::new();
        for _ in 0..MAX_SEQUENCE_DEPTH + 4 {
            let mut item = Vec::new();
            item.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
            item.extend_from_slice(&(nested.len() as u32).to_le_bytes());
            item.extend_from_slice(&nested);
            nested = element(tags::OPTICAL_PATH_SEQUENCE, b"SQ", &item);
        }

        let (_dir, path) = write(&file_with(
            transfer_syntax::EXPLICIT_VR_LITTLE_ENDIAN,
            &nested,
        ));
        let error = read_instance(&path).unwrap_err();
        assert!(error.0.contains("nest more than"), "{error}");
    }

    /// A frame length is a 32-bit field in the file, so it has to be checked against the
    /// file before it is used to size an allocation.
    #[test]
    fn a_frame_length_past_the_end_of_the_file_is_refused_rather_than_allocated() {
        let mut dataset = Vec::new();
        dataset.extend_from_slice(&[0xE0, 0x7F, 0x10, 0x00]);
        dataset.extend_from_slice(b"OB");
        dataset.extend_from_slice(&[0, 0]);
        dataset.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // An offset table naming one frame, whose own item header claims 64 MB of a file
        // that is a few hundred bytes long.
        dataset.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        dataset.extend_from_slice(&4u32.to_le_bytes());
        dataset.extend_from_slice(&0u32.to_le_bytes());
        dataset.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0]);
        dataset.extend_from_slice(&(64u32 * 1024 * 1024).to_le_bytes());
        dataset.extend_from_slice(b"only a few bytes actually follow");

        let (_dir, path) = write(&file_with(transfer_syntax::JPEG_BASELINE, &dataset));
        let instance = read_instance(&path).unwrap();
        let error = instance.read_frame(0).unwrap_err();
        assert!(error.0.contains("of a"), "{error}");
        assert!(error.0.contains("-byte file"), "{error}");
    }

    #[test]
    fn a_file_without_the_magic_is_refused() {
        let (_dir, path) = write(&[0u8; 200]);
        assert!(read_instance(&path).unwrap_err().0.contains("DICM"));
    }
}
