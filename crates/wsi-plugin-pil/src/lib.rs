//! The `pil` plugin: plain raster images.
//!
//! Ported from `wsi_service_base_plugins/pil/wsi_service_plugin_pil/`.
//!
//! Support is declared through `supported_file_extensions = [".png", ".jpg", ".jpeg"]`,
//! which Python matches against `pathlib.Path(filepath).suffix` — a single suffix, so
//! `image.tar.png` matches and `image.png.gz` does not.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use image::DynamicImage;
use parking_lot::Mutex;
use std::sync::Arc;

use wsi_core::error::{ApiError, Result};
use wsi_core::icc::{IccIntent, IccProfileCache};
use wsi_core::models::{rgb_channel_list, SlideExtent, SlideInfo, SlideLevel, SlidePixelSizeNm};
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};

const SUPPORTED_FILE_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];

/// Above this size the plugin serves tiles instead of the whole image in one go.
/// `slide.py:30` uses a strict `<` on both dimensions.
const SINGLE_TILE_MAX_DIMENSION: u32 = 5000;
const FALLBACK_TILE_SIZE: i64 = 1024;

pub struct PilPlugin {
    settings: Arc<Settings>,
}

impl PilPlugin {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl WsiPlugin for PilPlugin {
    fn name(&self) -> &'static str {
        "pil"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn is_supported(&self, path: &Path) -> bool {
        // `pathlib.Path(filepath).suffix in supported_file_extensions`. Python's `suffix`
        // is empty for a directory-like name and includes the leading dot; comparing the
        // lowercase extension is equivalent for this list.
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| SUPPORTED_FILE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    }

    async fn open(&self, path: &Path) -> Result<Arc<dyn Slide>> {
        let path = path.to_path_buf();
        let settings = Arc::clone(&self.settings);
        let slide = tokio::task::spawn_blocking(move || PilSlide::open(path, settings))
            .await
            .map_err(|e| ApiError::internal(format!("Failed to open image: {e}")))??;
        Ok(Arc::new(slide))
    }
}

pub struct PilSlide {
    path: PathBuf,
    /// `self.slide_image` — the whole image, already converted to RGB at open time.
    image: Mutex<Option<DynamicImage>>,
    icc_profile: Option<Vec<u8>>,
    icc: IccProfileCache,
    info: SlideInfo,
    settings: Arc<Settings>,
}

impl PilSlide {
    fn open(path: PathBuf, settings: Arc<Settings>) -> Result<Self> {
        let reader = image::ImageReader::open(&path)
            .map_err(|_| ApiError::internal("PIL Unidentified Image Error"))?
            .with_guessed_format()
            .map_err(|_| ApiError::internal("PIL Unidentified Image Error"))?;

        let format = reader
            .format()
            .map(format_name)
            .unwrap_or("unknown")
            .to_owned();

        let icc_profile = read_icc_profile(&path);

        let decoded = reader
            .decode()
            .map_err(|_| ApiError::internal("PIL Unidentified Image Error"))?;

        // `image.convert("RGB")` — resolves palettes and alpha before anything else runs.
        let image = DynamicImage::ImageRgb8(decoded.to_rgb8());
        let (width, height) = (image.width(), image.height());

        // `tile_extent` is the whole image for small files, else 1024x1024.
        let tile_extent = if width < SINGLE_TILE_MAX_DIMENSION && height < SINGLE_TILE_MAX_DIMENSION
        {
            SlideExtent::new(width as i64, height as i64, 1)
        } else {
            SlideExtent::new(FALLBACK_TILE_SIZE, FALLBACK_TILE_SIZE, 1)
        };

        let extent = SlideExtent::new(width as i64, height as i64, 1);
        let info = SlideInfo {
            id: String::new(),
            channels: Some(rgb_channel_list()),
            channel_depth: Some(8),
            extent,
            num_levels: 1,
            // Pixel size is unknown for a plain raster image.
            pixel_size_nm: SlidePixelSizeNm::xy(-1.0, -1.0),
            tile_extent,
            levels: vec![SlideLevel {
                extent,
                downsample_factor: 1.0,
            }],
            format,
            raw_download: false,
        };

        Ok(Self {
            path,
            image: Mutex::new(Some(image)),
            icc_profile,
            icc: IccProfileCache::new(),
            info,
            settings,
        })
    }

    fn with_image<T>(&self, f: impl FnOnce(&DynamicImage) -> Result<T>) -> Result<T> {
        let guard = self.image.lock();
        match guard.as_ref() {
            Some(image) => f(image),
            None => Err(ApiError::internal("Image has been closed")),
        }
    }

    /// `self.slide_image.crop((start_x, start_y, start_x + size_x, start_y + size_y))`.
    ///
    /// Pillow's `crop` accepts a box outside the image and fills the outside with zeros
    /// (black), rather than clamping. That behaviour is reproduced so an edge tile looks
    /// the same as it does under Python.
    fn crop(
        image: &DynamicImage,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
    ) -> DynamicImage {
        let mut out = image::RgbImage::from_pixel(size_x, size_y, image::Rgb([0, 0, 0]));
        let (w, h) = (image.width() as i64, image.height() as i64);
        let rgb = image.to_rgb8();
        for dy in 0..size_y as i64 {
            let sy = start_y + dy;
            if sy < 0 || sy >= h {
                continue;
            }
            for dx in 0..size_x as i64 {
                let sx = start_x + dx;
                if sx < 0 || sx >= w {
                    continue;
                }
                out.put_pixel(dx as u32, dy as u32, *rgb.get_pixel(sx as u32, sy as u32));
            }
        }
        DynamicImage::ImageRgb8(out)
    }

    fn apply_icc(
        &self,
        image: DynamicImage,
        intent: Option<IccIntent>,
        strict: bool,
        cache: bool,
    ) -> Result<DynamicImage> {
        match intent {
            None => Ok(image),
            Some(intent) => self.icc.process_image_lenient(
                image,
                self.icc_profile.as_deref(),
                strict,
                intent,
                cache,
            ),
        }
    }
}

#[async_trait]
impl Slide for PilSlide {
    async fn close(&self) -> Result<()> {
        *self.image.lock() = None;
        self.icc.free_cache();
        Ok(())
    }

    async fn get_info(&self) -> Result<SlideInfo> {
        Ok(self.info.clone())
    }

    async fn get_region(
        &self,
        _level: usize,
        start_x: i64,
        start_y: i64,
        size_x: u32,
        size_y: u32,
        opts: ReadOptions,
    ) -> Result<Pixels> {
        // `padding_color` is resolved but unused: Pillow's crop pads with black, and the
        // Python plugin never consults the colour either.
        let _ = opts.padding_color.unwrap_or(self.settings.padding_color);
        let region =
            self.with_image(|image| Ok(Self::crop(image, start_x, start_y, size_x, size_y)))?;
        let region = self.apply_icc(
            region,
            opts.icc_profile_intent,
            opts.icc_profile_strict,
            true,
        )?;
        Ok(Pixels::Image(region))
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
        let thumbnail = self.with_image(|image| Ok(pil_thumbnail(image, max_x, max_y)))?;
        let thumbnail = self.apply_icc(thumbnail, icc_profile_intent, icc_profile_strict, true)?;
        Ok(Pixels::Image(thumbnail))
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

    async fn get_icc_profile(&self) -> Result<Bytes> {
        IccProfileCache::payload(self.icc_profile.as_deref())
    }

    fn filepath(&self) -> &Path {
        &self.path
    }

    fn plugin_name(&self) -> &str {
        "pil"
    }
}

/// `Image.thumbnail((max_x, max_y))` — no resample argument, so Pillow's default
/// `BICUBIC` applies, not Lanczos.
pub fn pil_thumbnail(image: &DynamicImage, max_x: u32, max_y: u32) -> DynamicImage {
    wsi_core::image::pillow_thumbnail(image, max_x, max_y, wsi_core::image::Filter::Bicubic)
}

fn format_name(format: image::ImageFormat) -> &'static str {
    // `image.format.lower()` in Python. Only the formats this plugin accepts matter.
    match format {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        _ => "unknown",
    }
}

/// Extracts an embedded ICC profile.
///
/// The `image` crate exposes ICC profiles through `ImageDecoder::icc_profile`, which needs
/// a decoder rather than a `DynamicImage`, so the file is opened a second time. The cost
/// is paid once per slide open.
fn read_icc_profile(path: &Path) -> Option<Vec<u8>> {
    use image::ImageDecoder;

    let reader = image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?;
    match reader.format()? {
        image::ImageFormat::Png => {
            let mut decoder = image::codecs::png::PngDecoder::new(reader.into_inner()).ok()?;
            decoder.icc_profile().ok().flatten()
        }
        image::ImageFormat::Jpeg => {
            let mut decoder = image::codecs::jpeg::JpegDecoder::new(reader.into_inner()).ok()?;
            decoder.icc_profile().ok().flatten()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin() -> PilPlugin {
        PilPlugin::new(Arc::new(Settings::default()))
    }

    #[test]
    fn supported_extensions_match_the_python_list() {
        let p = plugin();
        assert!(p.is_supported(Path::new("a.png")));
        assert!(p.is_supported(Path::new("a.jpg")));
        assert!(p.is_supported(Path::new("a.jpeg")));
        assert!(p.is_supported(Path::new("a.JPG")));
        assert!(!p.is_supported(Path::new("a.svs")));
        assert!(!p.is_supported(Path::new("a.tiff")));
        // Python matches only the final suffix.
        assert!(p.is_supported(Path::new("a.tar.png")));
        assert!(!p.is_supported(Path::new("a.png.gz")));
        assert!(!p.is_supported(Path::new("noextension")));
    }

    #[test]
    fn crop_pads_outside_the_image_with_black() {
        let src =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30])));
        let out = PilSlide::crop(&src, 1, 1, 3, 3).to_rgb8();
        assert_eq!(out.dimensions(), (3, 3));
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30]);
        // Everything past the source is zero-filled, as Pillow's crop does.
        assert_eq!(out.get_pixel(1, 0).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(2, 2).0, [0, 0, 0]);
    }

    #[test]
    fn crop_with_negative_origin_offsets_content() {
        let src = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([1, 2, 3])));
        let out = PilSlide::crop(&src, -1, -1, 2, 2).to_rgb8();
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(1, 1).0, [1, 2, 3]);
    }

    #[test]
    fn thumbnail_preserves_aspect_ratio_and_never_enlarges() {
        let src = DynamicImage::ImageRgb8(image::RgbImage::new(400, 200));
        let out = pil_thumbnail(&src, 100, 100);
        assert_eq!((out.width(), out.height()), (100, 50));

        // Already smaller than the box: unchanged.
        let small = DynamicImage::ImageRgb8(image::RgbImage::new(10, 10));
        let out = pil_thumbnail(&small, 100, 100);
        assert_eq!((out.width(), out.height()), (10, 10));
    }

    #[tokio::test]
    async fn tile_extent_is_the_whole_image_below_the_threshold() {
        let dir = std::env::temp_dir().join("wsi-pil-plugin-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("small.png");
        image::RgbImage::from_pixel(64, 32, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();

        let slide = PilSlide::open(path.clone(), Arc::new(Settings::default())).unwrap();
        let info = slide.get_info().await.unwrap();
        assert_eq!(info.extent, SlideExtent::new(64, 32, 1));
        assert_eq!(info.tile_extent, SlideExtent::new(64, 32, 1));
        assert_eq!(info.num_levels, 1);
        assert_eq!(info.pixel_size_nm.x, -1.0);
        assert_eq!(info.format, "png");
        assert_eq!(info.channel_depth, Some(8));

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn label_and_macro_are_404() {
        let dir = std::env::temp_dir().join("wsi-pil-plugin-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("assoc.png");
        image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]))
            .save(&path)
            .unwrap();

        let slide = PilSlide::open(path.clone(), Arc::new(Settings::default())).unwrap();
        assert_eq!(
            slide.get_label().await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );
        assert_eq!(
            slide.get_macro(None, false).await.unwrap_err().status,
            http::StatusCode::NOT_FOUND
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unreadable_file_reports_the_pillow_message() {
        // `unwrap_err` needs `T: Debug`, which the slide handle cannot provide.
        let err = match PilSlide::open(
            PathBuf::from("definitely-not-an-image.png"),
            Arc::new(Settings::default()),
        ) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "PIL Unidentified Image Error");
    }
}
