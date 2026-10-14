//! One TIFF page, with the properties `tifffile` exposes on a keyframe and the two reads
//! the plugin performs: a single chunk, and the whole page.

use super::codec::{decode_chunk, ChunkGeometry};
use super::ifd::{tag, ByteOrder, Ifd, Result, TiffError, TiffSource};

/// Sample type, from `SampleFormat` and `BitsPerSample` — `keyframe.dtype` in tifffile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleDtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl SampleDtype {
    pub fn size(self) -> usize {
        match self {
            SampleDtype::U8 | SampleDtype::I8 => 1,
            SampleDtype::U16 | SampleDtype::I16 => 2,
            SampleDtype::U32 | SampleDtype::I32 | SampleDtype::F32 => 4,
            SampleDtype::U64 | SampleDtype::I64 | SampleDtype::F64 => 8,
        }
    }

    /// The numpy dtype string the fixture manifest records, so tests can compare directly.
    pub fn numpy_str(self) -> &'static str {
        match self {
            SampleDtype::U8 => "|u1",
            SampleDtype::I8 => "|i1",
            SampleDtype::U16 => "<u2",
            SampleDtype::I16 => "<i2",
            SampleDtype::U32 => "<u4",
            SampleDtype::I32 => "<i4",
            SampleDtype::U64 => "<u8",
            SampleDtype::I64 => "<i8",
            SampleDtype::F32 => "<f4",
            SampleDtype::F64 => "<f8",
        }
    }

    /// Lowest and highest representable value, for the padding clamp
    /// `__get_color_for_channel` performs with `np.iinfo`.
    pub fn integer_bounds(self) -> Option<(f64, f64)> {
        Some(match self {
            SampleDtype::U8 => (0.0, 255.0),
            SampleDtype::I8 => (-128.0, 127.0),
            SampleDtype::U16 => (0.0, 65_535.0),
            SampleDtype::I16 => (-32_768.0, 32_767.0),
            SampleDtype::U32 => (0.0, 4_294_967_295.0),
            SampleDtype::I32 => (-2_147_483_648.0, 2_147_483_647.0),
            SampleDtype::U64 => (0.0, u64::MAX as f64),
            SampleDtype::I64 => (i64::MIN as f64, i64::MAX as f64),
            SampleDtype::F32 | SampleDtype::F64 => return None,
        })
    }

    fn from_tags(sample_format: u16, bits_per_sample: usize) -> Result<Self> {
        Ok(match (sample_format, bits_per_sample) {
            (1, 8) => SampleDtype::U8,
            (1, 16) => SampleDtype::U16,
            (1, 32) => SampleDtype::U32,
            (1, 64) => SampleDtype::U64,
            (2, 8) => SampleDtype::I8,
            (2, 16) => SampleDtype::I16,
            (2, 32) => SampleDtype::I32,
            (2, 64) => SampleDtype::I64,
            (3, 32) => SampleDtype::F32,
            (3, 64) => SampleDtype::F64,
            _ => {
                return Err(TiffError(format!(
                    "unsupported sample format {sample_format} at {bits_per_sample} bits"
                )))
            }
        })
    }

    /// Writes `value` into `out`, clamped and truncated the way numpy's assignment of a
    /// Python int into an array of this dtype behaves.
    pub fn fill(self, out: &mut [u8], value: f64) {
        macro_rules! fill {
            ($ty:ty) => {{
                let bytes = (value as $ty).to_ne_bytes();
                for sample in out.chunks_exact_mut(bytes.len()) {
                    sample.copy_from_slice(&bytes);
                }
            }};
        }
        match self {
            SampleDtype::U8 => out.fill(value.clamp(0.0, 255.0) as u8),
            SampleDtype::I8 => fill!(i8),
            SampleDtype::U16 => fill!(u16),
            SampleDtype::I16 => fill!(i16),
            SampleDtype::U32 => fill!(u32),
            SampleDtype::I32 => fill!(i32),
            SampleDtype::U64 => fill!(u64),
            SampleDtype::I64 => fill!(i64),
            SampleDtype::F32 => fill!(f32),
            SampleDtype::F64 => fill!(f64),
        }
    }
}

/// A decoded rectangle, samples interleaved last — numpy's `(height, width, samples)`.
#[derive(Debug, Clone)]
pub struct Raster {
    pub height: usize,
    pub width: usize,
    pub samples: usize,
    pub dtype: SampleDtype,
    /// `height * width * samples` values, row-major, in native byte order.
    pub bytes: Vec<u8>,
}

impl Raster {
    pub fn filled(
        height: usize,
        width: usize,
        samples: usize,
        dtype: SampleDtype,
        value: f64,
    ) -> Self {
        let mut bytes = vec![0u8; height * width * samples * dtype.size()];
        dtype.fill(&mut bytes, value);
        Self {
            height,
            width,
            samples,
            dtype,
            bytes,
        }
    }

    fn offset(&self, y: usize, x: usize, sample: usize) -> usize {
        ((y * self.width + x) * self.samples + sample) * self.dtype.size()
    }

    /// Copies a `(height, width)` window of `src` into `self` at `(dst_y, dst_x)`.
    ///
    /// `sample_map` says which of `self`'s samples each of `src`'s samples lands on, which
    /// is how a separate-plane tile — one sample wide — writes into a multi-sample output.
    #[allow(clippy::too_many_arguments)]
    pub fn blit(
        &mut self,
        src: &Raster,
        dst_y: usize,
        dst_x: usize,
        src_y: usize,
        src_x: usize,
        height: usize,
        width: usize,
        sample_offset: usize,
    ) {
        let size = self.dtype.size();
        let samples = src.samples.min(self.samples - sample_offset);
        for row in 0..height {
            let (sy, dy) = (src_y + row, dst_y + row);
            if sy >= src.height || dy >= self.height {
                break;
            }
            for column in 0..width {
                let (sx, dx) = (src_x + column, dst_x + column);
                if sx >= src.width || dx >= self.width {
                    break;
                }
                let from = src.offset(sy, sx, 0);
                let to = self.offset(dy, dx, sample_offset);
                self.bytes[to..to + samples * size]
                    .copy_from_slice(&src.bytes[from..from + samples * size]);
            }
        }
    }

    /// The raw sample at `(y, x, sample)`, in native byte order.
    pub fn sample_bytes(&self, y: usize, x: usize, sample: usize) -> &[u8] {
        let at = self.offset(y, x, sample);
        &self.bytes[at..at + self.dtype.size()]
    }

    /// Value at `(y, x, sample)` as `f64`, for tests and for the RGB conversion path.
    pub fn get(&self, y: usize, x: usize, sample: usize) -> f64 {
        let bytes = self.sample_bytes(y, x, sample);
        macro_rules! read {
            ($ty:ty, $n:literal) => {{
                let mut buf = [0u8; $n];
                buf.copy_from_slice(bytes);
                <$ty>::from_ne_bytes(buf) as f64
            }};
        }
        match self.dtype {
            SampleDtype::U8 => bytes[0] as f64,
            SampleDtype::I8 => bytes[0] as i8 as f64,
            SampleDtype::U16 => read!(u16, 2),
            SampleDtype::I16 => read!(i16, 2),
            SampleDtype::U32 => read!(u32, 4),
            SampleDtype::I32 => read!(i32, 4),
            SampleDtype::U64 => read!(u64, 8),
            SampleDtype::I64 => read!(i64, 8),
            SampleDtype::F32 => read!(f32, 4),
            SampleDtype::F64 => read!(f64, 8),
        }
    }
}

/// A page's geometry and tags — everything `keyframe` carries that the plugin reads.
#[derive(Debug, Clone)]
pub struct Page {
    pub ifd: Ifd,
    /// Position in the series' page list, which is what `page.index` reports.
    pub index: usize,
    pub image_width: usize,
    pub image_length: usize,
    pub image_depth: usize,
    pub bits_per_sample: usize,
    pub samples_per_pixel: usize,
    pub photometric: u16,
    pub planar_config: u16,
    pub compression: u16,
    pub predictor: u16,
    pub tile_width: usize,
    pub tile_length: usize,
    pub tile_depth: usize,
    pub rows_per_strip: usize,
    pub dtype: SampleDtype,
    pub data_offsets: Vec<u64>,
    pub data_byte_counts: Vec<u64>,
    pub jpeg_tables: Option<Vec<u8>>,
    pub description: String,
    /// Byte order of the file, which `IJMetadata` strings are decoded in.
    pub byte_order: ByteOrder,
}

impl Page {
    pub fn from_ifd(ifd: Ifd, index: usize, byte_order: ByteOrder) -> Result<Self> {
        let image_width = ifd.u64_or(tag::IMAGE_WIDTH, 0) as usize;
        let image_length = ifd.u64_or(tag::IMAGE_LENGTH, 0) as usize;
        if image_width == 0 || image_length == 0 {
            return Err(TiffError("page has no extent".into()));
        }
        // `BitsPerSample` is one value per sample; a page mixing widths is not something
        // the numeric pipeline could represent anyway.
        let bits = ifd.u64_vec(tag::BITS_PER_SAMPLE);
        let bits_per_sample = bits.first().copied().unwrap_or(1) as usize;
        if bits.iter().any(|&b| b as usize != bits_per_sample) {
            return Err(TiffError("page mixes sample widths".into()));
        }
        let sample_format = ifd.u64_or(tag::SAMPLE_FORMAT, 1) as u16;
        let dtype = SampleDtype::from_tags(sample_format, bits_per_sample)?;

        let tile_width = ifd.u64_or(tag::TILE_WIDTH, 0) as usize;
        let tile_length = ifd.u64_or(tag::TILE_LENGTH, 0) as usize;
        let tiled = tile_width > 0 && tile_length > 0;

        let (offsets_tag, counts_tag) = if tiled {
            (tag::TILE_OFFSETS, tag::TILE_BYTE_COUNTS)
        } else {
            (tag::STRIP_OFFSETS, tag::STRIP_BYTE_COUNTS)
        };

        // The TIFF default is "one strip holding everything"; tifffile reports it the same
        // way, as `imagelength`.
        let rows_per_strip = match ifd.u64_or(tag::ROWS_PER_STRIP, u64::MAX) {
            rows if rows == 0 || rows > image_length as u64 => image_length,
            rows => rows as usize,
        };

        Ok(Self {
            index,
            image_width,
            image_length,
            image_depth: ifd.u64_or(tag::IMAGE_DEPTH, 1).max(1) as usize,
            bits_per_sample,
            samples_per_pixel: ifd.u64_or(tag::SAMPLES_PER_PIXEL, 1).max(1) as usize,
            photometric: ifd.u64_or(tag::PHOTOMETRIC, 1) as u16,
            planar_config: ifd.u64_or(tag::PLANAR_CONFIGURATION, 1).max(1) as u16,
            compression: ifd.u64_or(tag::COMPRESSION, 1) as u16,
            predictor: ifd.u64_or(tag::PREDICTOR, 1) as u16,
            tile_width,
            tile_length,
            tile_depth: ifd.u64_or(tag::TILE_DEPTH, 1).max(1) as usize,
            rows_per_strip,
            dtype,
            data_offsets: ifd.u64_vec(offsets_tag),
            data_byte_counts: ifd.u64_vec(counts_tag),
            jpeg_tables: ifd.get(tag::JPEG_TABLES).map(|e| e.as_bytes().to_vec()),
            description: ifd.string(tag::IMAGE_DESCRIPTION).unwrap_or_default(),
            byte_order,
            ifd,
        })
    }

    pub fn is_tiled(&self) -> bool {
        self.tile_width > 0 && self.tile_length > 0
    }

    /// `PLANARCONFIG=SEPARATE`: each sample has its own chunk grid.
    pub fn separate_planes(&self) -> bool {
        self.planar_config == 2
    }

    /// Samples carried by one chunk: all of them, unless the planes are separate.
    pub fn chunk_samples(&self) -> usize {
        if self.separate_planes() {
            1
        } else {
            self.samples_per_pixel
        }
    }

    pub fn tiles_across(&self) -> usize {
        if self.tile_width == 0 {
            0
        } else {
            self.image_width.div_ceil(self.tile_width)
        }
    }

    pub fn tiles_down(&self) -> usize {
        if self.tile_length == 0 {
            0
        } else {
            self.image_length.div_ceil(self.tile_length)
        }
    }

    pub fn strips_per_plane(&self) -> usize {
        if self.rows_per_strip == 0 {
            0
        } else {
            self.image_length.div_ceil(self.rows_per_strip)
        }
    }

    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.ifd.get(tag::ICC_PROFILE).map(|e| e.as_bytes())
    }

    /// Decodes chunk `index` of this page. `None` when the page has no such chunk, which
    /// is the `len(page.dataoffsets) <= index` skip the Python loop performs.
    pub fn read_chunk(&self, source: &mut TiffSource, index: usize) -> Result<Option<Raster>> {
        let Some(&offset) = self.data_offsets.get(index) else {
            return Ok(None);
        };
        let byte_count = self.data_byte_counts.get(index).copied().unwrap_or(0) as usize;
        if byte_count == 0 {
            return Ok(None);
        }
        let data = source.read_at(offset, byte_count)?;

        let (width, height) = if self.is_tiled() {
            (self.tile_width, self.tile_length)
        } else {
            let strips = self.strips_per_plane().max(1);
            let row = index % strips;
            let start = row * self.rows_per_strip;
            (
                self.image_width,
                self.rows_per_strip
                    .min(self.image_length.saturating_sub(start)),
            )
        };
        let samples = self.chunk_samples();
        let geometry = ChunkGeometry {
            width,
            height,
            samples,
            bits_per_sample: self.bits_per_sample,
            photometric: self.photometric,
        };
        let bytes = decode_chunk(
            self.compression,
            self.predictor,
            source.order,
            geometry,
            &data,
            self.jpeg_tables.as_deref(),
        )?;
        Ok(Some(Raster {
            height,
            width,
            samples,
            dtype: self.dtype,
            bytes,
        }))
    }

    /// `page.asarray()` — the whole page, samples interleaved last.
    pub fn read_full(&self, source: &mut TiffSource) -> Result<Raster> {
        let mut out = Raster::filled(
            self.image_length,
            self.image_width,
            self.samples_per_pixel,
            self.dtype,
            0.0,
        );
        let planes = if self.separate_planes() {
            self.samples_per_pixel
        } else {
            1
        };

        if self.is_tiled() {
            let (across, down) = (self.tiles_across(), self.tiles_down());
            let per_plane = across * down;
            for plane in 0..planes {
                for row in 0..down {
                    for column in 0..across {
                        let index = plane * per_plane + row * across + column;
                        let Some(tile) = self.read_chunk(source, index)? else {
                            continue;
                        };
                        out.blit(
                            &tile,
                            row * self.tile_length,
                            column * self.tile_width,
                            0,
                            0,
                            self.tile_length,
                            self.tile_width,
                            plane,
                        );
                    }
                }
            }
        } else {
            let strips = self.strips_per_plane();
            for plane in 0..planes {
                for strip in 0..strips {
                    let index = plane * strips + strip;
                    let Some(chunk) = self.read_chunk(source, index)? else {
                        continue;
                    };
                    let rows = chunk.height;
                    out.blit(
                        &chunk,
                        strip * self.rows_per_strip,
                        0,
                        0,
                        0,
                        rows,
                        self.image_width,
                        plane,
                    );
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes_and_bounds_match_numpy() {
        assert_eq!(SampleDtype::U16.size(), 2);
        assert_eq!(SampleDtype::I8.integer_bounds(), Some((-128.0, 127.0)));
        assert_eq!(SampleDtype::F32.integer_bounds(), None);
        assert_eq!(SampleDtype::F64.numpy_str(), "<f8");
    }

    #[test]
    fn an_unrepresentable_sample_layout_is_rejected() {
        assert!(SampleDtype::from_tags(3, 16).is_err());
        assert!(SampleDtype::from_tags(1, 12).is_err());
    }

    #[test]
    fn filling_clamps_into_a_signed_page() {
        // 255 does not fit int8; numpy raises, Python clamps first, so must this.
        let raster = Raster::filled(1, 1, 1, SampleDtype::I8, 127.0);
        assert_eq!(raster.get(0, 0, 0), 127.0);
    }

    #[test]
    fn blit_places_a_single_sample_plane_at_its_sample_offset() {
        let mut out = Raster::filled(2, 2, 3, SampleDtype::U8, 0.0);
        let plane = Raster {
            height: 2,
            width: 2,
            samples: 1,
            dtype: SampleDtype::U8,
            bytes: vec![1, 2, 3, 4],
        };
        out.blit(&plane, 0, 0, 0, 0, 2, 2, 1);
        assert_eq!(out.get(0, 0, 1), 1.0);
        assert_eq!(out.get(1, 1, 1), 4.0);
        assert_eq!(out.get(0, 0, 0), 0.0);
        assert_eq!(out.get(0, 0, 2), 0.0);
    }

    #[test]
    fn blit_clips_at_both_edges() {
        let mut out = Raster::filled(2, 2, 1, SampleDtype::U8, 0.0);
        let src = Raster {
            height: 4,
            width: 4,
            samples: 1,
            dtype: SampleDtype::U8,
            bytes: (0..16).collect(),
        };
        out.blit(&src, 1, 1, 0, 0, 4, 4, 0);
        assert_eq!(out.get(1, 1, 0), 0.0);
        assert_eq!(out.get(0, 0, 0), 0.0);
    }
}
