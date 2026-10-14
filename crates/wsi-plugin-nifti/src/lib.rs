//! The `nifti` plugin: scalar NIfTI volumes rendered as greyscale RGB.
//!
//! Ported from `wsi_service_base_plugins/nifti/wsi_service_plugin_nifti/`.
//!
//! The volume is served in its stored voxel order — no orientation handling — as a
//! single-level pyramid whose `extent.z` is the volume depth rather than 1. Every read is
//! windowed to a single contrast range shared by all tiles and all Z layers, taken from
//! the header's `cal_min`/`cal_max` when they are usable and otherwise from one scan of
//! the whole volume.
//!
//! Two details differ from the rest of the port and are load-bearing:
//!
//! - Rendering rounds with `np.rint`, which is **round-half-to-even**, not the truncation
//!   `convert_narray_uintX_to_uint8` uses. `127.5` becomes `128` and `0.5` becomes `0`.
//! - `get_thumbnail` subsamples with a stride and then calls
//!   `Image.resize(..., LANCZOS)` — `resize`, whose `reducing_gap` defaults to `None`,
//!   not `thumbnail`, which defaults to `2.0` and would run a box-average pre-pass.

mod reader;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use image::DynamicImage;
use ndarray::Array2;
use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::IccIntent;
use wsi_core::image::{pillow_resize, Filter};
use wsi_core::models::{rgb_channel_list, SlideExtent, SlideInfo, SlideLevel, SlidePixelSizeNm};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

use reader::{Axis, OpenError, VolumeReader, VoxelKind};

/// `tile_extent = SlideExtent(x=min(width, 512), y=min(height, 512), z=1)`
const MAX_TILE_SIZE: i64 = 512;

/// `_get_window`'s block geometry — one million voxels per read at most.
const SCAN_BLOCK_WIDTH: usize = 1024;
const SCAN_BLOCK_VOXELS: usize = 1_048_576;

pub struct NiftiPlugin {
    settings: Arc<Settings>,
}

impl NiftiPlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for NiftiPlugin {
    fn name(&self) -> &'static str {
        "nifti"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// `path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))`
    fn is_supported(&self, path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let name = name.to_ascii_lowercase();
        name.ends_with(".nii") || name.ends_with(".nii.gz")
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || NiftiSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

/// The reader plus the contrast window, behind one lock.
///
/// Python holds `self._lock` across `_read` and `_get_window` alike, so the window is
/// computed at most once and every render sees the same pair. Keeping both here reproduces
/// that without a second lock.
struct State {
    volume: VolumeReader,
    /// `self._window`: `Some` from the header, or filled in by the first render.
    window: Option<(f64, f64)>,
}

pub struct NiftiSlide {
    path: PathBuf,
    /// `None` once `close()` has run — `_image is None` in Python.
    state: Mutex<Option<State>>,
    info: SlideInfo,
    settings: Arc<Settings>,
}

impl NiftiSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        let volume = VolumeReader::open(&path).map_err(|e| match e {
            OpenError::Unreadable(detail) => {
                ApiError::bad_request(format!("Unable to read NIfTI image: {detail}"))
            }
            OpenError::NotSingleFile => {
                ApiError::bad_request("Expected a single-file NIfTI-1 or NIfTI-2 image.")
            }
            // `nibabel` raises `HeaderDataError`, which no handler catches; `_open_slide`
            // catches the bare `Exception` and reports it as a 500 instead.
            OpenError::UnknownDataType(code) => {
                ApiError::internal(format!("data dtype \"{code}\" not recognized"))
            }
        })?;

        let header = volume.header().clone();
        let shape = &header.shape;
        // `len(shape) < 2 or any(size < 1 for size in shape) or any(size != 1 for size in shape[3:])`
        if shape.len() < 2
            || shape.iter().any(|&size| size < 1)
            || shape.iter().skip(3).any(|&size| size != 1)
        {
            return Err(ApiError::bad_request(
                "NIfTI supports 2D/3D scalar images only; non-singleton time or vector axes are unsupported.",
            ));
        }
        if header.voxel_kind == VoxelKind::Unsupported {
            return Err(ApiError::bad_request(
                "Complex and RGB/vector NIfTI voxel types are unsupported.",
            ));
        }

        let (width, height) = (shape[0], shape[1]);
        let depth = shape.get(2).copied().unwrap_or(1);
        let extent = SlideExtent::new(width as i64, height as i64, depth as i64);

        // `spacing(axis)`: a zoom is only usable with a known unit, a real axis, and a
        // finite positive value.
        let spacing = |axis: usize| -> f64 {
            let Some(unit) = header.space_unit else {
                return -1.0;
            };
            match header.zooms.get(axis) {
                Some(&zoom) if zoom.is_finite() && zoom > 0.0 => zoom * unit.nanometres(),
                _ => -1.0,
            }
        };

        let info = SlideInfo {
            id: String::new(),
            channels: Some(rgb_channel_list()),
            channel_depth: Some(8),
            extent,
            num_levels: 1,
            pixel_size_nm: SlidePixelSizeNm::xyz(spacing(0), spacing(1), Some(spacing(2))),
            tile_extent: SlideExtent::new(
                (width as i64).min(MAX_TILE_SIZE),
                (height as i64).min(MAX_TILE_SIZE),
                1,
            ),
            levels: vec![SlideLevel {
                extent,
                downsample_factor: 1.0,
            }],
            format: "nifti".into(),
            raw_download: false,
        };

        // `(low, high) if np.isfinite([low, high]).all() and low < high else None`
        let (low, high) = (header.cal_min, header.cal_max);
        let window = (low.is_finite() && high.is_finite() && low < high).then_some((low, high));

        Ok(Self {
            path,
            state: Mutex::new(Some(State { volume, window })),
            info,
            settings,
        })
    }

    /// `_check_icc` — the plugin carries no profile, so a strict request cannot be met.
    fn check_icc(intent: Option<IccIntent>, strict: bool) -> Result<()> {
        if intent.is_some() && strict {
            return Err(ApiError::precondition_failed("ICC Profile not available."));
        }
        Ok(())
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        let mut guard = self.state.lock();
        match guard.as_mut() {
            Some(state) => f(state),
            None => Err(ApiError::bad_request("NIfTI image is closed.")),
        }
    }
}

/// `_get_window` — the header window, or one scan of the whole volume.
///
/// The scan shares its result across every tile and Z layer, which is the point: a
/// per-tile window would make neighbouring tiles disagree about contrast.
fn window_of(state: &mut State, extent: SlideExtent) -> Result<(f64, f64)> {
    if let Some(window) = state.window {
        return Ok(window);
    }
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    let block_width = (extent.x as usize).clamp(1, SCAN_BLOCK_WIDTH);
    let block_height = (SCAN_BLOCK_VOXELS / block_width).max(1);
    for z in 0..extent.z as usize {
        for y in (0..extent.y as usize).step_by(block_height) {
            for x in (0..extent.x as usize).step_by(block_width) {
                let values = read_block(
                    state,
                    Axis::range(x, x + block_width),
                    Axis::range(y, y + block_height),
                    z,
                )?;
                for &value in values.iter().filter(|v| v.is_finite()) {
                    low = low.min(value);
                    high = high.max(value);
                }
            }
        }
    }
    // An empty or all-non-finite volume leaves the bounds infinite, which Python replaces
    // with `(0.0, 0.0)` — a window `_render` then treats as "all black".
    let window = if low.is_finite() && high.is_finite() {
        (low, high)
    } else {
        (0.0, 0.0)
    };
    state.window = Some(window);
    Ok(window)
}

fn read_block(state: &mut State, xs: Axis, ys: Axis, z: usize) -> Result<Array2<f64>> {
    state
        .volume
        .read(xs, ys, z)
        .map_err(|e| ApiError::bad_request(format!("Unable to read NIfTI image: {e:?}")))
}

/// `_render` — window, scale to `0..255`, and replicate to RGB.
fn render(values: &Array2<f64>, window: (f64, f64)) -> DynamicImage {
    let (low, high) = window;
    let (rows, cols) = values.dim();
    let mut out = image::RgbImage::new(cols as u32, rows as u32);
    if high <= low {
        // `np.zeros(values.shape, dtype=np.uint8)` — already what `RgbImage::new` gives.
        return DynamicImage::ImageRgb8(out);
    }
    let span = high - low;
    // Finite endpoints can still overflow on subtraction, so Python rescales first.
    let divisor_scale = (!span.is_finite()).then(|| low.abs().max(high.abs()));

    for (row, line) in values.rows().into_iter().enumerate() {
        for (col, &value) in line.iter().enumerate() {
            // `np.where(np.isfinite(values), scaled, 0)` — non-finite voxels render black.
            let scaled = if !value.is_finite() {
                0.0
            } else {
                let clipped = value.clamp(low, high);
                match divisor_scale {
                    None => (clipped - low) / span,
                    Some(scale) => (clipped / scale - low / scale) / (high / scale - low / scale),
                }
            };
            // `np.rint` rounds halves to even, unlike the truncating main pipeline.
            let gray = (scaled.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8;
            out.put_pixel(col as u32, row as u32, image::Rgb([gray, gray, gray]));
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// `Image.new("RGB", (size_x, size_y), color)` then `paste(rendered, at)`.
fn paste_onto_padding(
    rendered: &DynamicImage,
    size_x: u32,
    size_y: u32,
    color: (u8, u8, u8),
    at_x: u32,
    at_y: u32,
) -> DynamicImage {
    let mut canvas =
        image::RgbImage::from_pixel(size_x, size_y, image::Rgb([color.0, color.1, color.2]));
    let src = rendered.to_rgb8();
    for (x, y, pixel) in src.enumerate_pixels() {
        let (dx, dy) = (at_x + x, at_y + y);
        if dx < size_x && dy < size_y {
            canvas.put_pixel(dx, dy, *pixel);
        }
    }
    DynamicImage::ImageRgb8(canvas)
}

#[async_trait]
impl Slide for NiftiSlide {
    async fn close(&self) -> Result<()> {
        *self.state.lock() = None;
        Ok(())
    }

    async fn get_info(&self) -> Result<SlideInfo> {
        Ok(self.info.clone())
    }

    async fn get_region(
        &self,
        level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        // The order of these four checks is observable: a request that is wrong in more
        // than one way reports whichever Python would have reported.
        Self::check_icc(opts.icc_profile_intent, opts.icc_profile_strict)?;
        if level != 0 {
            return Err(ApiError::bad_request("NIfTI images have only level 0."));
        }
        if opts.z as i64 >= self.info.extent.z {
            return Err(ApiError::bad_request(format!(
                "Invalid NIfTI Z layer: {}.",
                opts.z
            )));
        }
        if size_x == 0 || size_y == 0 {
            return Err(ApiError::bad_request("Region dimensions must be positive."));
        }
        let color = opts.padding_color.unwrap_or(self.settings.padding_color);
        let z = opts.z as usize;
        let extent = self.info.extent;

        self.with_state(|state| {
            let x0 = start_x.max(0);
            let y0 = start_y.max(0);
            let x1 = (start_x + size_x as i64).min(extent.x);
            let y1 = (start_y + size_y as i64).min(extent.y);
            if x0 >= x1 || y0 >= y1 {
                // Wholly outside: the padded canvas is the whole answer.
                return Ok(Pixels::Image(DynamicImage::ImageRgb8(
                    image::RgbImage::from_pixel(
                        size_x,
                        size_y,
                        image::Rgb([color.0, color.1, color.2]),
                    ),
                )));
            }
            let values = read_block(
                state,
                Axis::range(x0 as usize, x1 as usize),
                Axis::range(y0 as usize, y1 as usize),
                z,
            )?;
            let window = window_of(state, extent)?;
            let rendered = render(&values, window);
            Ok(Pixels::Image(paste_onto_padding(
                &rendered,
                size_x,
                size_y,
                color,
                (x0 - start_x) as u32,
                (y0 - start_y) as u32,
            )))
        })
    }

    async fn get_tile(
        &self,
        level: usize,
        tile_x: i64,
        tile_y: i64,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        let tile = self.info.tile_extent;
        self.get_region(
            level,
            tile_x * tile.x,
            tile_y * tile.y,
            tile.x as u32,
            tile.y as u32,
            opts,
        )
        .await
    }

    async fn get_thumbnail(
        &self,
        max_x: u32,
        max_y: u32,
        icc_profile_intent: Option<IccIntent>,
        icc_profile_strict: bool,
    ) -> Result<Pixels> {
        Self::check_icc(icc_profile_intent, icc_profile_strict)?;
        if max_x == 0 || max_y == 0 {
            return Err(ApiError::bad_request(
                "Thumbnail dimensions must be positive.",
            ));
        }
        let extent = self.info.extent;
        self.with_state(|state| {
            let scale = (max_x as f64 / extent.x as f64)
                .min(max_y as f64 / extent.y as f64)
                .min(1.0);
            // Python's `round` is round-half-to-even, and the result is floored at 1.
            let target_x = ((extent.x as f64 * scale).round_ties_even() as u32).max(1);
            let target_y = ((extent.y as f64 * scale).round_ties_even() as u32).max(1);
            // `int(1 / scale)` truncates.
            let step = ((1.0 / scale) as usize).max(1);
            let values = read_block(
                state,
                Axis::strided(extent.x as usize, step),
                Axis::strided(extent.y as usize, step),
                (extent.z / 2) as usize,
            )?;
            let window = window_of(state, extent)?;
            let rendered = render(&values, window);
            // `Image.resize`, whose `reducing_gap` defaults to `None` — no box pre-pass.
            Ok(Pixels::Image(pillow_resize(
                &rendered,
                target_x,
                target_y,
                Filter::Lanczos,
                None,
            )))
        })
    }

    async fn get_label(&self) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image label does not exist.",
        ))
    }

    async fn get_macro(&self, _: Option<IccIntent>, _: bool) -> Result<Pixels> {
        Err(ApiError::not_found(
            "Associated image macro does not exist.",
        ))
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "nifti"
    }
}

#[cfg(test)]
mod tests {
    use super::reader::fixtures::{f32_bytes, i16_bytes, nifti1, nifti2, write};
    use super::*;

    const INT16: i16 = 4;
    const FLOAT32: i16 = 16;
    const UINT8: i16 = 2;
    const COMPLEX64: i16 = 32;
    /// `set_xyzt_units("mm")`
    const MM: u8 = 2;

    fn settings() -> Arc<Settings> {
        Arc::new(Settings::default())
    }

    fn slide_from(name: &str, bytes: &[u8]) -> NiftiSlide {
        NiftiSlide::open(write("wsi-nifti-plugin-tests", name, bytes), settings()).unwrap()
    }

    fn error_from(name: &str, bytes: &[u8]) -> ApiError {
        match NiftiSlide::open(write("wsi-nifti-plugin-tests", name, bytes), settings()) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        }
    }

    fn rgb(pixels: Pixels) -> image::RgbImage {
        match pixels {
            Pixels::Image(image) => image.to_rgb8(),
            other => panic!("expected an image, got {other:?}"),
        }
    }

    /// The `test_discovery_metadata_and_z_regions` volume: `arange(24).reshape(4, 3, 2)`
    /// with `diag([0.5, 0.75, 2, 1])` and a `(0, 23)` calibration window.
    ///
    /// NIfTI stores the first axis fastest, so the on-disk order is not the C order
    /// `reshape` produces: voxel `(x, y, z)` of that array holds `x * 6 + y * 2 + z`.
    fn arange_volume() -> Vec<u8> {
        let mut voxels = Vec::new();
        for z in 0..2i16 {
            for y in 0..3i16 {
                for x in 0..4i16 {
                    voxels.push(x * 6 + y * 2 + z);
                }
            }
        }
        nifti1(
            &[4, 3, 2],
            &[0.5, 0.75, 2.0],
            MM,
            INT16,
            2,
            None,
            Some((0.0, 23.0)),
            &i16_bytes(&voxels),
        )
    }

    #[test]
    fn supported_names_match_the_python_predicate() {
        let plugin = NiftiPlugin::new(settings());
        let path = write("wsi-nifti-plugin-tests", "supported.nii", b"x");
        let dir = path.parent().unwrap();
        assert!(plugin.is_supported(&path));
        assert!(plugin.is_supported(&write("wsi-nifti-plugin-tests", "supported.NII.GZ", b"x")));
        // A directory and a missing file are both rejected before the name is considered.
        assert!(!plugin.is_supported(dir));
        assert!(!plugin.is_supported(&dir.join("missing.nii")));
        assert!(!plugin.is_supported(&write("wsi-nifti-plugin-tests", "image.tiff", b"x")));
    }

    #[tokio::test]
    async fn info_reports_the_volume_depth_and_the_scaled_zooms() {
        let slide = slide_from("meta.nii", &arange_volume());
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.extent, SlideExtent::new(4, 3, 2));
        assert_eq!(info.num_levels, 1);
        assert_eq!(info.levels[0].extent.z, 2);
        assert_eq!(info.pixel_size_nm.x, 500_000.0);
        assert_eq!(info.pixel_size_nm.y, 750_000.0);
        assert_eq!(info.pixel_size_nm.z, Some(2_000_000.0));
        assert_eq!(info.channel_depth, Some(8));
        assert_eq!(info.channels.as_ref().unwrap().len(), 3);
        assert_eq!(info.format, "nifti");
        assert_eq!(info.tile_extent, SlideExtent::new(4, 3, 1));
    }

    #[tokio::test]
    async fn a_region_of_the_second_z_layer_matches_the_python_expectation() {
        let slide = slide_from("zregion.nii", &arange_volume());
        let region = rgb(slide
            .get_region(
                0,
                1,
                1,
                2,
                2,
                ReadOptions {
                    z: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        // `np.rint(data[1:3, 1:3, 1].T / 23 * 255)`; data[x, y, z] == x * 6 + y * 2 + z.
        let expected = [[100u8, 166], [122, 188]];
        for (row, line) in expected.iter().enumerate() {
            for (col, &value) in line.iter().enumerate() {
                assert_eq!(
                    region.get_pixel(col as u32, row as u32).0,
                    [value, value, value],
                    "at ({col}, {row})"
                );
            }
        }
    }

    #[tokio::test]
    async fn two_dimensional_and_singleton_axes_are_accepted() {
        for shape in [vec![5usize, 3], vec![5, 3, 1], vec![5, 3, 2, 1]] {
            let voxels: Vec<f32> = (0..shape.iter().product::<usize>() as i32)
                .map(|v| v as f32)
                .collect();
            let zooms = vec![1.0f32; shape.len()];
            let slide = slide_from(
                "singleton.nii",
                &nifti1(
                    &shape,
                    &zooms,
                    MM,
                    FLOAT32,
                    4,
                    None,
                    None,
                    &f32_bytes(&voxels),
                ),
            );
            let region = rgb(slide
                .get_region(0, 0, 0, 5, 3, ReadOptions::default())
                .await
                .unwrap());
            assert_eq!(region.dimensions(), (5, 3), "shape {shape:?}");
        }
    }

    #[tokio::test]
    async fn an_unknown_space_unit_reports_minus_one_for_every_axis() {
        let slide = slide_from(
            "nounit.nii",
            &nifti1(
                &[2, 3, 4],
                &[0.5, 0.75, 2.0],
                0, // unknown
                UINT8,
                1,
                None,
                None,
                &[0u8; 24],
            ),
        );
        let spacing = slide.get_info().await.unwrap().pixel_size_nm;
        assert_eq!((spacing.x, spacing.y, spacing.z), (-1.0, -1.0, Some(-1.0)));
    }

    #[tokio::test]
    async fn a_two_dimensional_volume_has_no_z_spacing() {
        let slide = slide_from(
            "flat.nii",
            &nifti1(&[2, 2], &[0.5, 0.75], MM, UINT8, 1, None, None, &[0u8; 4]),
        );
        let spacing = slide.get_info().await.unwrap().pixel_size_nm;
        assert_eq!(spacing.x, 500_000.0);
        assert_eq!(spacing.z, Some(-1.0));
    }

    #[tokio::test]
    async fn header_scaling_and_calibration_are_both_applied() {
        // `test_header_scaling_and_calibration`: int16 (2, 3) scaled by 2x - 5, window
        // (-5, 55).
        let slide = slide_from(
            "scaled.nii",
            &nifti1(
                &[2, 3],
                &[1.0, 1.0],
                MM,
                INT16,
                2,
                Some((2.0, -5.0)),
                Some((-5.0, 55.0)),
                &i16_bytes(&[-10, 20, 0, 30, 10, 40]),
            ),
        );
        let region = rgb(slide
            .get_region(0, 0, 0, 2, 3, ReadOptions::default())
            .await
            .unwrap());
        let expected = [[0u8, 170], [0, 255], [85, 255]];
        for (row, line) in expected.iter().enumerate() {
            for (col, &value) in line.iter().enumerate() {
                assert_eq!(region.get_pixel(col as u32, row as u32).0[0], value);
            }
        }
    }

    #[tokio::test]
    async fn the_fallback_window_is_shared_across_regions_and_z_layers() {
        // `test_fallback_window_is_shared_across_regions_and_slices`:
        // float32 [[[0, 100], [25, 75]], [[50, 50], [75, 25]]], no calibration — laid out
        // x-fastest, so `(x, y, z)` reads down the nesting.
        let slide = slide_from(
            "fallback.nii",
            &nifti1(
                &[2, 2, 2],
                &[1.0, 1.0, 1.0],
                MM,
                FLOAT32,
                4,
                None,
                None,
                &f32_bytes(&[0.0, 50.0, 25.0, 75.0, 100.0, 50.0, 75.0, 25.0]),
            ),
        );
        assert!(slide.state.lock().as_ref().unwrap().window.is_none());

        let left = rgb(slide
            .get_region(0, 0, 0, 1, 2, ReadOptions::default())
            .await
            .unwrap());
        assert_eq!(left.get_pixel(0, 0).0[0], 0);
        assert_eq!(left.get_pixel(0, 1).0[0], 64);

        let right = rgb(slide
            .get_region(0, 1, 0, 1, 2, ReadOptions::default())
            .await
            .unwrap());
        assert_eq!(right.get_pixel(0, 0).0[0], 128);
        assert_eq!(right.get_pixel(0, 1).0[0], 191);

        let other = rgb(slide
            .get_region(
                0,
                0,
                0,
                2,
                2,
                ReadOptions {
                    z: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert_eq!(other.get_pixel(0, 0).0[0], 255);
        assert_eq!(other.get_pixel(1, 0).0[0], 128);
        assert_eq!(other.get_pixel(0, 1).0[0], 191);
        assert_eq!(other.get_pixel(1, 1).0[0], 64);

        assert_eq!(
            slide.state.lock().as_ref().unwrap().window,
            Some((0.0, 100.0))
        );
    }

    #[tokio::test]
    async fn a_constant_or_all_non_finite_volume_renders_black() {
        for (name, voxels) in [
            ("constant.nii", vec![7.0f32; 6]),
            ("nan.nii", vec![f32::NAN; 6]),
        ] {
            let slide = slide_from(
                name,
                &nifti1(
                    &[2, 3],
                    &[1.0, 1.0],
                    MM,
                    FLOAT32,
                    4,
                    None,
                    None,
                    &f32_bytes(&voxels),
                ),
            );
            let region = rgb(slide
                .get_region(0, 0, 0, 2, 3, ReadOptions::default())
                .await
                .unwrap());
            assert!(
                region.pixels().all(|p| p.0 == [0, 0, 0]),
                "{name} should be all black"
            );
        }
    }

    #[tokio::test]
    async fn non_finite_voxels_render_black_beside_finite_ones() {
        // float32 [[nan, inf, -inf], [0, 50, 100]] — the window comes from the finite half.
        let slide = slide_from(
            "mixed.nii",
            &nifti1(
                &[2, 3],
                &[1.0, 1.0],
                MM,
                FLOAT32,
                4,
                None,
                None,
                &f32_bytes(&[f32::NAN, 0.0, f32::INFINITY, 50.0, f32::NEG_INFINITY, 100.0]),
            ),
        );
        let region = rgb(slide
            .get_region(0, 0, 0, 2, 3, ReadOptions::default())
            .await
            .unwrap());
        let expected = [[0u8, 0], [0, 128], [0, 255]];
        for (row, line) in expected.iter().enumerate() {
            for (col, &value) in line.iter().enumerate() {
                assert_eq!(
                    region.get_pixel(col as u32, row as u32).0[0],
                    value,
                    "at ({col}, {row})"
                );
            }
        }
    }

    /// The `test_tiles_padding_and_thumbnail` volume: `(514, 4, 3)` int16 with the second
    /// Z layer at 50 and the third at 100, calibrated to `(0, 100)`.
    fn wide_volume() -> Vec<u8> {
        let mut voxels = vec![0i16; 514 * 4 * 3];
        let plane = 514 * 4;
        for value in &mut voxels[plane..2 * plane] {
            *value = 50;
        }
        for value in &mut voxels[2 * plane..] {
            *value = 100;
        }
        nifti1(
            &[514, 4, 3],
            &[1.0, 1.0, 1.0],
            MM,
            INT16,
            2,
            None,
            Some((0.0, 100.0)),
            &i16_bytes(&voxels),
        )
    }

    #[tokio::test]
    async fn edge_tiles_pad_and_the_thumbnail_subsamples() {
        let slide = slide_from("wide.nii", &wide_volume());
        assert_eq!(
            slide.get_info().await.unwrap().tile_extent,
            SlideExtent::new(512, 4, 1)
        );

        let tile = rgb(slide
            .get_tile(
                0,
                1,
                0,
                ReadOptions {
                    padding_color: Some((1, 2, 3)),
                    z: 2,
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert_eq!(tile.dimensions(), (512, 4));
        assert_eq!(tile.get_pixel(1, 0).0, [255, 255, 255]);
        assert_eq!(tile.get_pixel(2, 0).0, [1, 2, 3]);

        let region = rgb(slide
            .get_region(
                0,
                -1,
                -1,
                3,
                3,
                ReadOptions {
                    padding_color: Some((1, 2, 3)),
                    z: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert_eq!(region.get_pixel(0, 0).0, [1, 2, 3]);
        assert_eq!(region.get_pixel(1, 1).0, [128, 128, 128]);

        let outside = rgb(slide
            .get_tile(
                0,
                -100,
                100,
                ReadOptions {
                    padding_color: Some((1, 2, 3)),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert!(outside.pixels().all(|p| p.0 == [1, 2, 3]));

        let thumb = rgb(slide.get_thumbnail(100, 100, None, false).await.unwrap());
        assert!(thumb.width() <= 100 && thumb.height() <= 100);
        // The middle Z layer is a constant 50 against a (0, 100) window: 127.5 rints to 128.
        assert_eq!(thumb.get_pixel(0, 0).0, [128, 128, 128]);
    }

    #[tokio::test]
    async fn unsupported_shapes_and_voxel_types_are_rejected() {
        // 1-D, a non-singleton fourth axis, and a complex dtype.
        let one_d = nifti1(&[4], &[1.0], MM, INT16, 2, None, None, &i16_bytes(&[0; 4]));
        assert_eq!(
            error_from("oned.nii", &one_d).status,
            http::StatusCode::BAD_REQUEST
        );

        let time_axis = nifti1(
            &[2, 3, 4, 2],
            &[1.0; 4],
            MM,
            FLOAT32,
            4,
            None,
            None,
            &f32_bytes(&[0.0; 48]),
        );
        assert!(error_from("time.nii", &time_axis)
            .detail_text()
            .contains("unsupported"));

        let complex = nifti1(
            &[2, 3],
            &[1.0, 1.0],
            MM,
            COMPLEX64,
            8,
            None,
            None,
            &[0u8; 48],
        );
        assert!(error_from("complex.nii", &complex)
            .detail_text()
            .contains("unsupported"));
    }

    #[test]
    fn a_corrupt_file_reports_the_python_message() {
        let err = error_from("broken.nii", b"not a nifti file");
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert!(err.detail_text().starts_with("Unable to read NIfTI"));
    }

    #[tokio::test]
    async fn invalid_requests_and_missing_associated_images() {
        let slide = slide_from(
            "invalid.nii",
            &nifti1(&[3, 4, 2], &[1.0; 3], MM, UINT8, 1, None, None, &[0u8; 24]),
        );
        let bad = [
            (1usize, 0u32, 2u32),
            (0, 2, 2), // z past the last layer
            (0, 0, 0), // zero-sized
        ];
        for (level, z, size) in bad {
            let err = slide
                .get_region(
                    level,
                    0,
                    0,
                    size,
                    2,
                    ReadOptions {
                        z,
                        ..Default::default()
                    },
                )
                .await
                .unwrap_err();
            assert_eq!(
                err.status,
                http::StatusCode::BAD_REQUEST,
                "{level}/{z}/{size}"
            );
        }

        assert_eq!(
            slide.get_label().await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );
        assert_eq!(
            slide.get_macro(None, false).await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );
        assert_eq!(
            slide.get_icc_profile().await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );

        // A strict ICC request cannot be satisfied; a lenient one is ignored.
        let strict = slide
            .get_region(
                0,
                0,
                0,
                1,
                1,
                ReadOptions {
                    icc_profile_intent: Some(IccIntent::Perceptual),
                    icc_profile_strict: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert_eq!(strict.status, http::StatusCode::PRECONDITION_FAILED);
        slide
            .get_region(
                0,
                0,
                0,
                1,
                1,
                ReadOptions {
                    icc_profile_intent: Some(IccIntent::Perceptual),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        slide.close().await.unwrap();
        let closed = slide
            .get_region(0, 0, 0, 1, 1, ReadOptions::default())
            .await
            .unwrap_err();
        assert!(closed.detail_text().contains("closed"));
    }

    #[tokio::test]
    async fn nifti2_volumes_open_through_the_same_path() {
        let slide = slide_from(
            "v2.nii",
            &nifti2(
                &[2, 2, 1],
                &[0.5, 0.75, 2.0],
                3, // micron
                INT16,
                2,
                Some((0.0, 3.0)),
                &i16_bytes(&[0, 1, 2, 3]),
            ),
        );
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.extent, SlideExtent::new(2, 2, 1));
        assert_eq!(info.pixel_size_nm.x, 500.0);
        let region = rgb(slide
            .get_region(0, 0, 0, 2, 2, ReadOptions::default())
            .await
            .unwrap());
        assert_eq!(region.get_pixel(0, 0).0[0], 0);
        assert_eq!(region.get_pixel(1, 1).0[0], 255);
    }
}
