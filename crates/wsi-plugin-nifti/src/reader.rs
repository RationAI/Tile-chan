//! A minimal NIfTI-1/NIfTI-2 reader, standing in for `nibabel.load(..., mmap="r")`.
//!
//! Only what the plugin needs is parsed: the data shape, the voxel type, the zooms, the
//! space unit, the scaling pair and the calibration window. Everything else in the header
//! describes orientation, which the plugin deliberately ignores — it renders volumes "in
//! their stored voxel order".
//!
//! Reads are lazy and bounded, the way nibabel's `ArrayProxy` is: `_get_window` scans the
//! whole volume in blocks and a tile reads only its own rows, so neither materialises the
//! full array. NIfTI stores the first axis fastest, so a run of `x` for one `(y, z)` is
//! contiguous and costs a single read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use ndarray::Array2;

/// Why a file could not be turned into a readable volume.
///
/// The variants map onto the two messages `Slide._open` raises before it looks at the
/// shape: `nib.load` failing, and `nib.load` succeeding with something that is not a
/// single-file NIfTI-1/2 image.
#[derive(Debug)]
pub enum OpenError {
    /// `HTTPException(400, f"Unable to read NIfTI image: {exc}")`
    Unreadable(String),
    /// `HTTPException(400, "Expected a single-file NIfTI-1 or NIfTI-2 image.")` — nibabel
    /// loads a `.hdr`/`.img` pair as `Nifti1Pair`, which is not a `Nifti1Image`.
    NotSingleFile,
    /// `nibabel.spatialimages.HeaderDataError`, which no handler catches, so `_open_slide`
    /// turns it into `Plugin nifti unable to open image (...)`.
    UnknownDataType(i16),
}

/// The voxel types whose numpy dtype kind is in `"uif"` — the set `Slide._open` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NiftiDtype {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

impl NiftiDtype {
    pub fn item_size(self) -> usize {
        match self {
            NiftiDtype::U8 | NiftiDtype::I8 => 1,
            NiftiDtype::U16 | NiftiDtype::I16 => 2,
            NiftiDtype::U32 | NiftiDtype::I32 | NiftiDtype::F32 => 4,
            NiftiDtype::U64 | NiftiDtype::I64 | NiftiDtype::F64 => 8,
        }
    }

    fn decode(self, bytes: &[u8], little_endian: bool) -> f64 {
        macro_rules! read {
            ($ty:ty, $n:literal) => {{
                let mut buf = [0u8; $n];
                buf.copy_from_slice(&bytes[..$n]);
                if little_endian {
                    <$ty>::from_le_bytes(buf) as f64
                } else {
                    <$ty>::from_be_bytes(buf) as f64
                }
            }};
        }
        match self {
            NiftiDtype::U8 => bytes[0] as f64,
            NiftiDtype::I8 => bytes[0] as i8 as f64,
            NiftiDtype::U16 => read!(u16, 2),
            NiftiDtype::I16 => read!(i16, 2),
            NiftiDtype::U32 => read!(u32, 4),
            NiftiDtype::I32 => read!(i32, 4),
            NiftiDtype::U64 => read!(u64, 8),
            NiftiDtype::I64 => read!(i64, 8),
            NiftiDtype::F32 => read!(f32, 4),
            NiftiDtype::F64 => read!(f64, 8),
        }
    }
}

/// What the datatype code resolves to.
///
/// `Slide._open` rejects anything whose numpy dtype kind is not `u`, `i` or `f`, so the
/// codes nibabel understands but the plugin refuses — complex, RGB, RGBA, bool and the
/// 128-bit float — are a distinct outcome from a code nibabel does not know at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelKind {
    Numeric(NiftiDtype),
    /// A dtype whose kind is outside `"uif"`: complex, RGB/RGBA, or bool.
    Unsupported,
}

/// `dim` / `pixdim` / scaling, as much of the header as the plugin consults.
#[derive(Debug, Clone)]
pub struct NiftiHeader {
    /// `header.get_data_shape()` — `dim[1..=dim[0]]`.
    pub shape: Vec<usize>,
    pub voxel_kind: VoxelKind,
    /// `header.get_zooms()` — `pixdim[1..=dim[0]]`.
    pub zooms: Vec<f64>,
    /// `header.get_xyzt_units()[0]`, restricted to the three the plugin maps.
    pub space_unit: Option<SpaceUnit>,
    /// Already resolved the way `ArrayProxy` does: an unusable `scl_slope` becomes 1.0 and
    /// a non-finite `scl_inter` becomes 0.0.
    pub slope: f64,
    pub inter: f64,
    pub cal_min: f64,
    pub cal_max: f64,
    data_offset: u64,
    little_endian: bool,
}

/// The space units `_open` knows how to convert, in `xyzt_units`' low three bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceUnit {
    Meter,
    Millimeter,
    Micron,
}

impl SpaceUnit {
    /// `{"meter": 1e9, "mm": 1e6, "micron": 1e3}`
    pub fn nanometres(self) -> f64 {
        match self {
            SpaceUnit::Meter => 1e9,
            SpaceUnit::Millimeter => 1e6,
            SpaceUnit::Micron => 1e3,
        }
    }

    fn from_code(code: u32) -> Option<Self> {
        // `get_xyzt_units` takes `xyzt_units % 8` and looks it up in `unit_codes`, which
        // only defines 0..=3 for space. 0 is "unknown"; 4..=7 are not in the table at all.
        match code % 8 {
            1 => Some(SpaceUnit::Meter),
            2 => Some(SpaceUnit::Millimeter),
            3 => Some(SpaceUnit::Micron),
            _ => None,
        }
    }
}

const NIFTI1_HEADER_SIZE: i64 = 348;
const NIFTI1_MIN_DATA_OFFSET: u64 = 352;
const NIFTI2_HEADER_SIZE: i64 = 540;
const NIFTI2_MIN_DATA_OFFSET: u64 = 544;

/// One dimension of a read, as a Python slice: `start`, `stop` and `step`.
///
/// `_read` passes `slice(x, x + block_width)` and `slice(None, None, step)`, both of which
/// this covers; `stop` is clamped to the axis length on use, as Python's slicing does.
#[derive(Debug, Clone, Copy)]
pub struct Axis {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

impl Axis {
    pub fn range(start: usize, stop: usize) -> Self {
        Self {
            start,
            stop,
            step: 1,
        }
    }

    pub fn strided(stop: usize, step: usize) -> Self {
        Self {
            start: 0,
            stop,
            step,
        }
    }

    fn resolved(self, length: usize) -> Self {
        Self {
            start: self.start.min(length),
            stop: self.stop.min(length),
            step: self.step.max(1),
        }
    }

    fn count(self) -> usize {
        if self.stop <= self.start {
            0
        } else {
            (self.stop - self.start).div_ceil(self.step)
        }
    }
}

/// Where the voxel bytes come from.
///
/// A plain `.nii` is read through the file handle, one contiguous run per row. A gzipped
/// file has no usable random access, so it is inflated once into memory — which is what
/// nibabel does too without `indexed_gzip` installed.
enum Source {
    File(File),
    Memory(Vec<u8>),
}

pub struct VolumeReader {
    source: Source,
    header: NiftiHeader,
    /// `shape`, padded to at least three entries so the index arithmetic is uniform.
    nx: usize,
    ny: usize,
}

impl VolumeReader {
    pub fn open(path: &Path) -> Result<Self, OpenError> {
        let mut file = File::open(path).map_err(|e| OpenError::Unreadable(format!("{e}")))?;

        let mut magic = [0u8; 2];
        let gzipped = match file.read_exact(&mut magic) {
            Ok(()) => magic == [0x1f, 0x8b],
            // A file shorter than two bytes cannot hold a header either.
            Err(e) => return Err(OpenError::Unreadable(format!("{e}"))),
        };

        let mut source = if gzipped {
            file.seek(SeekFrom::Start(0))
                .map_err(|e| OpenError::Unreadable(format!("{e}")))?;
            let mut inflated = Vec::new();
            flate2::read::GzDecoder::new(file)
                .read_to_end(&mut inflated)
                .map_err(|e| OpenError::Unreadable(format!("{e}")))?;
            Source::Memory(inflated)
        } else {
            Source::File(file)
        };

        // Enough for either header variant plus the single-file data offset.
        let mut head = [0u8; NIFTI2_MIN_DATA_OFFSET as usize];
        let read = read_at(&mut source, 0, &mut head)?;
        let header = NiftiHeader::parse(&head[..read])?;

        let nx = header.shape.first().copied().unwrap_or(1);
        let ny = header.shape.get(1).copied().unwrap_or(1);
        Ok(Self {
            source,
            header,
            nx,
            ny,
        })
    }

    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    /// `np.asarray(self._image.dataobj[xs, ys, z], dtype=np.float64).T`.
    ///
    /// The transpose is folded in: the result is indexed `[y, x]`, so its shape is
    /// `(height, width)` — which is what `Image.fromarray` then needs.
    pub fn read(&mut self, xs: Axis, ys: Axis, z: usize) -> Result<Array2<f64>, OpenError> {
        let dtype = match self.header.voxel_kind {
            VoxelKind::Numeric(dtype) => dtype,
            // Guarded at open; unreachable through the plugin.
            VoxelKind::Unsupported => {
                return Err(OpenError::Unreadable("unsupported voxel type".into()))
            }
        };
        let xs = xs.resolved(self.nx);
        let ys = ys.resolved(self.ny);
        let (cols, rows) = (xs.count(), ys.count());
        let mut out = Array2::<f64>::zeros((rows, cols));
        if cols == 0 || rows == 0 {
            return Ok(out);
        }

        let item = dtype.item_size();
        // The last element this row needs, so a strided read still fetches one span.
        let span_items = (cols - 1) * xs.step + 1;
        let mut buf = vec![0u8; span_items * item];
        let plane = z * self.ny * self.nx;
        let (slope, inter) = (self.header.slope, self.header.inter);

        for (row, y) in (0..rows).map(|r| (r, ys.start + r * ys.step)) {
            let first = plane + y * self.nx + xs.start;
            let offset = self.header.data_offset + (first * item) as u64;
            let read = read_at(&mut self.source, offset, &mut buf)?;
            for col in 0..cols {
                let at = col * xs.step * item;
                // A truncated file reads as zero past its end, matching an mmap of it.
                if at + item > read {
                    break;
                }
                let raw = dtype.decode(&buf[at..at + item], self.header.little_endian);
                out[[row, col]] = raw * slope + inter;
            }
        }
        Ok(out)
    }
}

/// Reads up to `buf.len()` bytes at `offset`, returning how many were available.
fn read_at(source: &mut Source, offset: u64, buf: &mut [u8]) -> Result<usize, OpenError> {
    match source {
        Source::File(file) => {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| OpenError::Unreadable(format!("{e}")))?;
            let mut filled = 0;
            while filled < buf.len() {
                match file.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(OpenError::Unreadable(format!("{e}"))),
                }
            }
            Ok(filled)
        }
        Source::Memory(bytes) => {
            let start = (offset as usize).min(bytes.len());
            let n = buf.len().min(bytes.len() - start);
            buf[..n].copy_from_slice(&bytes[start..start + n]);
            Ok(n)
        }
    }
}

impl NiftiHeader {
    fn parse(bytes: &[u8]) -> Result<Self, OpenError> {
        let too_short = || OpenError::Unreadable("file is too short to hold a NIfTI header".into());
        if bytes.len() < 4 {
            return Err(too_short());
        }
        // `sizeof_hdr` doubles as the endianness and version marker.
        let native = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64;
        let swapped = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64;
        let (version, little_endian) = match (native, swapped) {
            (NIFTI1_HEADER_SIZE, _) => (1, true),
            (_, NIFTI1_HEADER_SIZE) => (1, false),
            (NIFTI2_HEADER_SIZE, _) => (2, true),
            (_, NIFTI2_HEADER_SIZE) => (2, false),
            _ => {
                return Err(OpenError::Unreadable(
                    "Cannot work out file type of \"\"".into(),
                ))
            }
        };

        let get = Getters {
            bytes,
            little_endian,
        };
        if version == 1 {
            if bytes.len() < NIFTI1_HEADER_SIZE as usize {
                return Err(too_short());
            }
            match &bytes[344..348] {
                b"n+1\0" => {}
                b"ni1\0" => return Err(OpenError::NotSingleFile),
                _ => {
                    return Err(OpenError::Unreadable(
                        "Cannot work out file type of \"\"".into(),
                    ))
                }
            }
            let ndim = get.i16(40).clamp(0, 7) as usize;
            let shape = (1..=ndim)
                .map(|i| get.i16(40 + 2 * i).max(0) as usize)
                .collect();
            let zooms = (1..=ndim).map(|i| get.f32(76 + 4 * i) as f64).collect();
            let (slope, inter) = resolve_scaling(get.f32(112) as f64, get.f32(116) as f64);
            Ok(Self {
                shape,
                voxel_kind: voxel_kind(get.i16(70))?,
                zooms,
                space_unit: SpaceUnit::from_code(bytes[123] as u32),
                slope,
                inter,
                // NIfTI-1 stores `cal_max` before `cal_min`.
                cal_max: get.f32(124) as f64,
                cal_min: get.f32(128) as f64,
                data_offset: (get.f32(108) as u64).max(NIFTI1_MIN_DATA_OFFSET),
                little_endian,
            })
        } else {
            if bytes.len() < NIFTI2_HEADER_SIZE as usize {
                return Err(too_short());
            }
            match &bytes[4..12] {
                b"n+2\0\r\n\x1a\n" => {}
                b"ni2\0\r\n\x1a\n" => return Err(OpenError::NotSingleFile),
                _ => {
                    return Err(OpenError::Unreadable(
                        "Cannot work out file type of \"\"".into(),
                    ))
                }
            }
            let ndim = get.i64(16).clamp(0, 7) as usize;
            let shape = (1..=ndim)
                .map(|i| get.i64(16 + 8 * i).max(0) as usize)
                .collect();
            let zooms = (1..=ndim).map(|i| get.f64(104 + 8 * i)).collect();
            let (slope, inter) = resolve_scaling(get.f64(176), get.f64(184));
            Ok(Self {
                shape,
                voxel_kind: voxel_kind(get.i16(12))?,
                zooms,
                space_unit: SpaceUnit::from_code(get.i32(500) as u32),
                slope,
                inter,
                cal_max: get.f64(192),
                cal_min: get.f64(200),
                data_offset: (get.i64(168).max(0) as u64).max(NIFTI2_MIN_DATA_OFFSET),
                little_endian,
            })
        }
    }
}

/// `ArrayProxy`'s reading of `header.get_slope_inter()`.
///
/// A zero or non-finite `scl_slope` disables scaling entirely — nibabel returns
/// `(None, None)` and the proxy substitutes `(1.0, 0.0)`. A non-finite intercept alongside
/// a usable slope becomes 0.0.
fn resolve_scaling(slope: f64, inter: f64) -> (f64, f64) {
    if slope == 0.0 || !slope.is_finite() {
        return (1.0, 0.0);
    }
    (slope, if inter.is_finite() { inter } else { 0.0 })
}

/// The NIfTI datatype codes, split by whether the numpy dtype kind is in `"uif"`.
fn voxel_kind(code: i16) -> Result<VoxelKind, OpenError> {
    Ok(match code {
        2 => VoxelKind::Numeric(NiftiDtype::U8),
        4 => VoxelKind::Numeric(NiftiDtype::I16),
        8 => VoxelKind::Numeric(NiftiDtype::I32),
        16 => VoxelKind::Numeric(NiftiDtype::F32),
        64 => VoxelKind::Numeric(NiftiDtype::F64),
        256 => VoxelKind::Numeric(NiftiDtype::I8),
        512 => VoxelKind::Numeric(NiftiDtype::U16),
        768 => VoxelKind::Numeric(NiftiDtype::U32),
        1024 => VoxelKind::Numeric(NiftiDtype::I64),
        1280 => VoxelKind::Numeric(NiftiDtype::U64),
        // bool, complex64, RGB24, float128, complex128, complex256, RGBA32 — nibabel maps
        // all of these to a dtype whose kind is outside "uif".
        1 | 32 | 128 | 1536 | 1792 | 2048 | 2304 => VoxelKind::Unsupported,
        other => return Err(OpenError::UnknownDataType(other)),
    })
}

/// Endian-aware scalar reads at a byte offset.
struct Getters<'a> {
    bytes: &'a [u8],
    little_endian: bool,
}

macro_rules! getter {
    ($name:ident, $ty:ty, $n:literal) => {
        fn $name(&self, at: usize) -> $ty {
            let mut buf = [0u8; $n];
            buf.copy_from_slice(&self.bytes[at..at + $n]);
            if self.little_endian {
                <$ty>::from_le_bytes(buf)
            } else {
                <$ty>::from_be_bytes(buf)
            }
        }
    };
}

impl Getters<'_> {
    getter!(i16, i16, 2);
    getter!(i32, i32, 4);
    getter!(i64, i64, 8);
    getter!(f32, f32, 4);
    getter!(f64, f64, 8);
}

/// Hand-built NIfTI files, so the tests need no external fixtures.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::{NIFTI1_MIN_DATA_OFFSET, NIFTI2_MIN_DATA_OFFSET};

    /// Builds a single-file NIfTI-1 volume in memory, the way `nib.save` would.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn nifti1(
        shape: &[usize],
        zooms: &[f32],
        unit: u8,
        datatype: i16,
        item_size: usize,
        scaling: Option<(f32, f32)>,
        window: Option<(f32, f32)>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = vec![0u8; NIFTI1_MIN_DATA_OFFSET as usize];
        out[0..4].copy_from_slice(&348i32.to_le_bytes());
        out[40..42].copy_from_slice(&(shape.len() as i16).to_le_bytes());
        for (i, &d) in shape.iter().enumerate() {
            out[42 + 2 * i..44 + 2 * i].copy_from_slice(&(d as i16).to_le_bytes());
        }
        // `pixdim[0]` is qfac, which the plugin never reads.
        for (i, &z) in zooms.iter().enumerate() {
            out[80 + 4 * i..84 + 4 * i].copy_from_slice(&z.to_le_bytes());
        }
        out[70..72].copy_from_slice(&datatype.to_le_bytes());
        out[72..74].copy_from_slice(&((item_size * 8) as i16).to_le_bytes());
        out[108..112].copy_from_slice(&(NIFTI1_MIN_DATA_OFFSET as f32).to_le_bytes());
        if let Some((slope, inter)) = scaling {
            out[112..116].copy_from_slice(&slope.to_le_bytes());
            out[116..120].copy_from_slice(&inter.to_le_bytes());
        }
        out[123] = unit;
        if let Some((low, high)) = window {
            out[124..128].copy_from_slice(&high.to_le_bytes());
            out[128..132].copy_from_slice(&low.to_le_bytes());
        }
        out[344..348].copy_from_slice(b"n+1\0");
        out.extend_from_slice(data);
        out
    }

    /// `nib.Nifti2Image`'s single-file layout, for the parser's second branch.
    pub(crate) fn nifti2(
        shape: &[i64],
        zooms: &[f64],
        unit: i32,
        datatype: i16,
        item_size: usize,
        window: Option<(f64, f64)>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = vec![0u8; NIFTI2_MIN_DATA_OFFSET as usize];
        out[0..4].copy_from_slice(&540i32.to_le_bytes());
        out[4..12].copy_from_slice(b"n+2\0\r\n\x1a\n");
        out[12..14].copy_from_slice(&datatype.to_le_bytes());
        out[14..16].copy_from_slice(&((item_size * 8) as i16).to_le_bytes());
        out[16..24].copy_from_slice(&(shape.len() as i64).to_le_bytes());
        for (i, &d) in shape.iter().enumerate() {
            out[24 + 8 * i..32 + 8 * i].copy_from_slice(&d.to_le_bytes());
        }
        for (i, &z) in zooms.iter().enumerate() {
            out[112 + 8 * i..120 + 8 * i].copy_from_slice(&z.to_le_bytes());
        }
        out[168..176].copy_from_slice(&(NIFTI2_MIN_DATA_OFFSET as i64).to_le_bytes());
        if let Some((low, high)) = window {
            out[192..200].copy_from_slice(&high.to_le_bytes());
            out[200..208].copy_from_slice(&low.to_le_bytes());
        }
        out[500..504].copy_from_slice(&unit.to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    pub(crate) fn i16_bytes(values: &[i16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Writes `bytes` into a per-test temporary directory and returns the path.
    pub(crate) fn write(dir: &str, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{i16_bytes, nifti1};
    use super::*;

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        super::fixtures::write("wsi-nifti-reader-tests", name, bytes)
    }

    /// `Result::unwrap_err` needs `T: Debug`, and a reader holds a whole inflated volume.
    fn open_err(path: &std::path::Path) -> OpenError {
        match VolumeReader::open(path) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        }
    }

    #[test]
    fn parses_shape_zooms_units_and_window() {
        let bytes = nifti1(
            &[4, 3, 2],
            &[0.5, 0.75, 2.0],
            2,
            4,
            2,
            None,
            Some((0.0, 23.0)),
            &i16_bytes(&(0..24).collect::<Vec<_>>()),
        );
        let reader = VolumeReader::open(&write("meta.nii", &bytes)).unwrap();
        let header = reader.header();
        assert_eq!(header.shape, vec![4, 3, 2]);
        assert_eq!(header.zooms, vec![0.5, 0.75, 2.0]);
        assert_eq!(header.space_unit, Some(SpaceUnit::Millimeter));
        assert_eq!((header.cal_min, header.cal_max), (0.0, 23.0));
        assert_eq!((header.slope, header.inter), (1.0, 0.0));
    }

    #[test]
    fn read_transposes_so_the_result_is_indexed_row_then_column() {
        // dataobj[x, y] with x fastest: value == x + 4 * y.
        let bytes = nifti1(
            &[4, 3],
            &[1.0, 1.0],
            2,
            4,
            2,
            None,
            None,
            &i16_bytes(&(0..12).collect::<Vec<_>>()),
        );
        let mut reader = VolumeReader::open(&write("transpose.nii", &bytes)).unwrap();
        let values = reader
            .read(Axis::range(1, 3), Axis::range(0, 2), 0)
            .unwrap();
        assert_eq!(values.dim(), (2, 2));
        assert_eq!(values[[0, 0]], 1.0);
        assert_eq!(values[[0, 1]], 2.0);
        assert_eq!(values[[1, 0]], 5.0);
        assert_eq!(values[[1, 1]], 6.0);
    }

    #[test]
    fn reads_the_requested_z_plane() {
        let bytes = nifti1(
            &[2, 2, 2],
            &[1.0, 1.0, 1.0],
            2,
            4,
            2,
            None,
            None,
            &i16_bytes(&(0..8).collect::<Vec<_>>()),
        );
        let mut reader = VolumeReader::open(&write("zplane.nii", &bytes)).unwrap();
        let plane = reader
            .read(Axis::range(0, 2), Axis::range(0, 2), 1)
            .unwrap();
        assert_eq!(plane[[0, 0]], 4.0);
        assert_eq!(plane[[1, 1]], 7.0);
    }

    #[test]
    fn scaling_is_applied_as_the_array_proxy_does() {
        let bytes = nifti1(
            &[2, 1],
            &[1.0, 1.0],
            2,
            4,
            2,
            Some((2.0, -5.0)),
            None,
            &i16_bytes(&[10, 20]),
        );
        let mut reader = VolumeReader::open(&write("scaled.nii", &bytes)).unwrap();
        let values = reader
            .read(Axis::range(0, 2), Axis::range(0, 1), 0)
            .unwrap();
        assert_eq!(values[[0, 0]], 15.0);
        assert_eq!(values[[0, 1]], 35.0);
    }

    #[test]
    fn a_zero_slope_disables_scaling() {
        assert_eq!(resolve_scaling(0.0, 7.0), (1.0, 0.0));
        assert_eq!(resolve_scaling(f64::NAN, 7.0), (1.0, 0.0));
        assert_eq!(resolve_scaling(2.0, f64::NAN), (2.0, 0.0));
        assert_eq!(resolve_scaling(2.0, 3.0), (2.0, 3.0));
    }

    #[test]
    fn strided_reads_subsample_both_axes() {
        let bytes = nifti1(
            &[5, 4],
            &[1.0, 1.0],
            2,
            4,
            2,
            None,
            None,
            &i16_bytes(&(0..20).collect::<Vec<_>>()),
        );
        let mut reader = VolumeReader::open(&write("strided.nii", &bytes)).unwrap();
        let values = reader
            .read(Axis::strided(5, 2), Axis::strided(4, 2), 0)
            .unwrap();
        // x in {0, 2, 4}, y in {0, 2}; value == x + 5 * y.
        assert_eq!(values.dim(), (2, 3));
        assert_eq!(values[[0, 0]], 0.0);
        assert_eq!(values[[0, 2]], 4.0);
        assert_eq!(values[[1, 0]], 10.0);
        assert_eq!(values[[1, 2]], 14.0);
    }

    #[test]
    fn a_gzipped_volume_is_inflated_transparently() {
        use flate2::write::GzEncoder;
        use std::io::Write;

        let bytes = nifti1(
            &[2, 1],
            &[1.0, 1.0],
            2,
            4,
            2,
            None,
            None,
            &i16_bytes(&[3, 4]),
        );
        let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&bytes).unwrap();
        let gz = encoder.finish().unwrap();

        let mut reader = VolumeReader::open(&write("volume.nii.gz", &gz)).unwrap();
        let values = reader
            .read(Axis::range(0, 2), Axis::range(0, 1), 0)
            .unwrap();
        assert_eq!(values[[0, 0]], 3.0);
        assert_eq!(values[[0, 1]], 4.0);
    }

    #[test]
    fn big_endian_headers_and_voxels_are_understood() {
        let mut bytes = nifti1(
            &[2, 1],
            &[1.0, 1.0],
            2,
            4,
            2,
            None,
            None,
            &[0, 5, 0, 6], // big-endian 5, 6
        );
        // Rewrite the little-endian fields the test touches as big-endian.
        bytes[0..4].copy_from_slice(&348i32.to_be_bytes());
        bytes[40..42].copy_from_slice(&2i16.to_be_bytes());
        bytes[42..44].copy_from_slice(&2i16.to_be_bytes());
        bytes[44..46].copy_from_slice(&1i16.to_be_bytes());
        bytes[70..72].copy_from_slice(&4i16.to_be_bytes());
        bytes[108..112].copy_from_slice(&(NIFTI1_MIN_DATA_OFFSET as f32).to_be_bytes());

        let mut reader = VolumeReader::open(&write("bigendian.nii", &bytes)).unwrap();
        let values = reader
            .read(Axis::range(0, 2), Axis::range(0, 1), 0)
            .unwrap();
        assert_eq!(values[[0, 0]], 5.0);
        assert_eq!(values[[0, 1]], 6.0);
    }

    #[test]
    fn a_header_pair_is_rejected_as_not_single_file() {
        let mut bytes = nifti1(&[2, 1], &[1.0, 1.0], 2, 4, 2, None, None, &[0, 0, 0, 0]);
        bytes[344..348].copy_from_slice(b"ni1\0");
        let err = open_err(&write("pair.nii", &bytes));
        assert!(matches!(err, OpenError::NotSingleFile));
    }

    #[test]
    fn a_non_nifti_file_is_unreadable() {
        let err = open_err(&write("broken.nii", b"not a nifti file"));
        assert!(matches!(err, OpenError::Unreadable(_)));
    }

    #[test]
    fn datatype_codes_split_into_numeric_unsupported_and_unknown() {
        assert_eq!(voxel_kind(2).unwrap(), VoxelKind::Numeric(NiftiDtype::U8));
        assert_eq!(
            voxel_kind(1280).unwrap(),
            VoxelKind::Numeric(NiftiDtype::U64)
        );
        assert_eq!(voxel_kind(32).unwrap(), VoxelKind::Unsupported);
        assert_eq!(voxel_kind(128).unwrap(), VoxelKind::Unsupported);
        assert!(matches!(voxel_kind(3), Err(OpenError::UnknownDataType(3))));
    }

    #[test]
    fn nifti2_headers_are_parsed() {
        let bytes = super::fixtures::nifti2(
            &[2, 1, 1],
            &[0.5, 0.75, 2.0],
            3, // micron
            4, // int16
            2,
            Some((0.0, 5.0)),
            &i16_bytes(&[1, 2]),
        );
        let mut reader = VolumeReader::open(&write("v2.nii", &bytes)).unwrap();
        {
            let header = reader.header();
            assert_eq!(header.shape, vec![2, 1, 1]);
            assert_eq!(header.zooms, vec![0.5, 0.75, 2.0]);
            assert_eq!(header.space_unit, Some(SpaceUnit::Micron));
            assert_eq!((header.cal_min, header.cal_max), (0.0, 5.0));
        }
        let values = reader
            .read(Axis::range(0, 2), Axis::range(0, 1), 0)
            .unwrap();
        assert_eq!(values[[0, 0]], 1.0);
        assert_eq!(values[[0, 1]], 2.0);
    }
}
