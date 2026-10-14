//! Bindings to libopenslide's C API, plus a safe wrapper.
//!
//! Hand-written rather than generated: the API is twenty-odd functions and has been stable
//! since 3.4, and `openslide-python` — which the service must match — uses exactly this
//! surface. Writing it out makes the error-checking contract explicit, which matters
//! because libopenslide reports failures out-of-band through `openslide_get_error` rather
//! than through return values.
//!
//! The ICC entry points require OpenSlide 4.0 or newer; `build.rs` enforces that.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

use parking_lot::Mutex;

use wsi_core::error::{ApiError, Result};

#[allow(non_camel_case_types)]
pub type openslide_t = c_void;

#[link(name = "openslide")]
extern "C" {
    fn openslide_detect_vendor(filename: *const c_char) -> *const c_char;
    fn openslide_open(filename: *const c_char) -> *mut openslide_t;
    fn openslide_close(osr: *mut openslide_t);
    fn openslide_get_error(osr: *mut openslide_t) -> *const c_char;
    fn openslide_get_version() -> *const c_char;

    fn openslide_get_level_count(osr: *mut openslide_t) -> i32;
    fn openslide_get_level0_dimensions(osr: *mut openslide_t, w: *mut i64, h: *mut i64);
    fn openslide_get_level_dimensions(osr: *mut openslide_t, level: i32, w: *mut i64, h: *mut i64);
    fn openslide_get_level_downsample(osr: *mut openslide_t, level: i32) -> f64;

    fn openslide_read_region(
        osr: *mut openslide_t,
        dest: *mut u32,
        x: i64,
        y: i64,
        level: i32,
        w: i64,
        h: i64,
    );

    fn openslide_get_property_names(osr: *mut openslide_t) -> *const *const c_char;
    fn openslide_get_property_value(osr: *mut openslide_t, name: *const c_char) -> *const c_char;

    fn openslide_get_associated_image_names(osr: *mut openslide_t) -> *const *const c_char;
    fn openslide_get_associated_image_dimensions(
        osr: *mut openslide_t,
        name: *const c_char,
        w: *mut i64,
        h: *mut i64,
    );
    fn openslide_read_associated_image(osr: *mut openslide_t, name: *const c_char, dest: *mut u32);

    // --- OpenSlide 4.0+ ---------------------------------------------------------------
    fn openslide_get_icc_profile_size(osr: *mut openslide_t) -> i64;
    fn openslide_read_icc_profile(osr: *mut openslide_t, dest: *mut c_void);
    fn openslide_get_associated_image_icc_profile_size(
        osr: *mut openslide_t,
        name: *const c_char,
    ) -> i64;
    fn openslide_read_associated_image_icc_profile(
        osr: *mut openslide_t,
        name: *const c_char,
        dest: *mut c_void,
    );
}

/// The library version, for diagnostics.
pub fn version() -> String {
    // SAFETY: the returned pointer is a static string owned by the library.
    unsafe { cstr_to_string(openslide_get_version()) }.unwrap_or_else(|| "unknown".to_owned())
}

/// `openslide.OpenSlide.detect_format` — the vendor backend, or `None`.
pub fn detect_vendor(path: &Path) -> Option<String> {
    let filename = to_cstring(path)?;
    // SAFETY: `filename` outlives the call; the result is a static string or null.
    unsafe { cstr_to_string(openslide_detect_vendor(filename.as_ptr())) }
}

/// A pixel buffer as libopenslide produces it: ARGB32 premultiplied, host byte order.
pub struct PremultipliedArgb {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

/// An open slide handle.
///
/// libopenslide is documented as thread-safe for concurrent reads on one handle, but the
/// error state is per-handle and sticky: once set, every subsequent call on that handle
/// fails. Serialising access keeps a failed read from being misattributed to a concurrent
/// one, which is the same reason the Python tifffile plugin holds a lock.
pub struct OpenSlide {
    handle: Mutex<Handle>,
}

struct Handle(*mut openslide_t);

// SAFETY: the pointer is only dereferenced while the enclosing `Mutex` is held, and
// libopenslide keeps no thread-local state.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl OpenSlide {
    /// `openslide.OpenSlide(filename)`.
    ///
    /// A null return from `openslide_open` means the format was not recognised; a non-null
    /// handle with an error set means it was recognised but could not be read.
    /// `openslide-python` raises `OpenSlideUnsupportedFormatError` and `OpenSlideError`
    /// respectively, and the plugin turns both into the same 500.
    pub fn open(path: &Path) -> Result<Self> {
        let filename = to_cstring(path).ok_or_else(|| {
            ApiError::internal(format!(
                "OpenSlideError: path is not representable as a C string: {}",
                path.display()
            ))
        })?;

        // SAFETY: `filename` is a valid NUL-terminated string for the duration of the call.
        let raw = unsafe { openslide_open(filename.as_ptr()) };
        if raw.is_null() {
            return Err(ApiError::internal(format!(
                "OpenSlideError: Unsupported or missing image file {}",
                path.display()
            )));
        }

        let slide = Self {
            handle: Mutex::new(Handle(raw)),
        };
        // `openslide_open` can return a handle that is already in an error state.
        slide.check_error()?;
        Ok(slide)
    }

    fn with_handle<T>(&self, f: impl FnOnce(*mut openslide_t) -> T) -> T {
        let guard = self.handle.lock();
        f(guard.0)
    }

    /// libopenslide's error state is permanent once set, so this only reads it.
    fn check_error(&self) -> Result<()> {
        let message = self.with_handle(|osr| {
            if osr.is_null() {
                return Some("slide is closed".to_owned());
            }
            // SAFETY: `osr` is a live handle; the result is owned by the library.
            unsafe { cstr_to_string(openslide_get_error(osr)) }
        });
        match message {
            Some(message) => Err(ApiError::internal(format!("OpenSlideError: {message}"))),
            None => Ok(()),
        }
    }

    pub fn level_count(&self) -> Result<i32> {
        let count = self.with_handle(|osr| unsafe { openslide_get_level_count(osr) });
        self.check_error()?;
        Ok(count)
    }

    pub fn level0_dimensions(&self) -> Result<(i64, i64)> {
        let (mut w, mut h) = (0i64, 0i64);
        self.with_handle(|osr| unsafe { openslide_get_level0_dimensions(osr, &mut w, &mut h) });
        self.check_error()?;
        Ok((w, h))
    }

    pub fn level_dimensions(&self, level: i32) -> Result<(i64, i64)> {
        let (mut w, mut h) = (0i64, 0i64);
        self.with_handle(|osr| unsafe {
            openslide_get_level_dimensions(osr, level, &mut w, &mut h)
        });
        self.check_error()?;
        Ok((w, h))
    }

    pub fn level_downsample(&self, level: i32) -> Result<f64> {
        let value = self.with_handle(|osr| unsafe { openslide_get_level_downsample(osr, level) });
        self.check_error()?;
        Ok(value)
    }

    /// `openslide_read_region`. Returns premultiplied ARGB32 in host byte order.
    ///
    /// Out-of-bounds coordinates are not an error: libopenslide fills uncovered pixels with
    /// fully transparent black, which is what lets the service composite a padding colour
    /// over the gap.
    pub fn read_region(
        &self,
        x: i64,
        y: i64,
        level: i32,
        width: i64,
        height: i64,
    ) -> Result<PremultipliedArgb> {
        if width < 0 || height < 0 {
            return Err(ApiError::internal(
                "OpenSlideError: negative region dimensions".to_owned(),
            ));
        }
        let count = usize::try_from(width)
            .ok()
            .zip(usize::try_from(height).ok())
            .and_then(|(w, h)| w.checked_mul(h))
            .ok_or_else(|| {
                ApiError::internal("OpenSlideError: requested region is too large".to_owned())
            })?;

        let mut pixels = vec![0u32; count];
        if count > 0 {
            self.with_handle(|osr| unsafe {
                openslide_read_region(osr, pixels.as_mut_ptr(), x, y, level, width, height)
            });
            self.check_error()?;
        }
        Ok(PremultipliedArgb {
            width: width as u32,
            height: height as u32,
            pixels,
        })
    }

    /// One entry of `openslide.OpenSlide.properties`.
    pub fn property(&self, name: &str) -> Result<Option<String>> {
        let Ok(name) = CString::new(name) else {
            return Ok(None);
        };
        let value = self.with_handle(|osr| unsafe {
            cstr_to_string(openslide_get_property_value(osr, name.as_ptr()))
        });
        self.check_error()?;
        Ok(value)
    }

    pub fn property_names(&self) -> Result<Vec<String>> {
        let names = self.with_handle(|osr| unsafe {
            // SAFETY: the array is NULL-terminated and owned by the library.
            cstr_array_to_vec(openslide_get_property_names(osr))
        });
        self.check_error()?;
        Ok(names)
    }

    pub fn associated_image_names(&self) -> Result<Vec<String>> {
        let names = self.with_handle(|osr| unsafe {
            cstr_array_to_vec(openslide_get_associated_image_names(osr))
        });
        self.check_error()?;
        Ok(names)
    }

    pub fn associated_image(&self, name: &str) -> Result<PremultipliedArgb> {
        let c_name = CString::new(name).map_err(|_| {
            ApiError::internal(format!(
                "OpenSlideError: associated image: invalid name {name}"
            ))
        })?;

        let (mut w, mut h) = (0i64, 0i64);
        self.with_handle(|osr| unsafe {
            openslide_get_associated_image_dimensions(osr, c_name.as_ptr(), &mut w, &mut h)
        });
        self.check_error()?;

        if w <= 0 || h <= 0 {
            return Err(ApiError::internal(format!(
                "OpenSlideError: associated image: {name} has no dimensions"
            )));
        }
        let count = (w as usize).checked_mul(h as usize).ok_or_else(|| {
            ApiError::internal("OpenSlideError: associated image is too large".to_owned())
        })?;

        let mut pixels = vec![0u32; count];
        self.with_handle(|osr| unsafe {
            openslide_read_associated_image(osr, c_name.as_ptr(), pixels.as_mut_ptr())
        });
        self.check_error()?;

        Ok(PremultipliedArgb {
            width: w as u32,
            height: h as u32,
            pixels,
        })
    }

    /// `openslide.OpenSlide.color_profile` — the slide's ICC profile, if it has one.
    pub fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        let size = self.with_handle(|osr| unsafe { openslide_get_icc_profile_size(osr) });
        self.check_error()?;
        if size <= 0 {
            return Ok(None);
        }
        let mut buffer = vec![0u8; size as usize];
        self.with_handle(|osr| unsafe {
            openslide_read_icc_profile(osr, buffer.as_mut_ptr() as *mut c_void)
        });
        self.check_error()?;
        Ok(Some(buffer))
    }

    /// The ICC profile attached to an associated image, which Pillow exposes as
    /// `image.info["icc_profile"]`.
    pub fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let Ok(c_name) = CString::new(name) else {
            return Ok(None);
        };
        let size = self.with_handle(|osr| unsafe {
            openslide_get_associated_image_icc_profile_size(osr, c_name.as_ptr())
        });
        self.check_error()?;
        if size <= 0 {
            return Ok(None);
        }
        let mut buffer = vec![0u8; size as usize];
        self.with_handle(|osr| unsafe {
            openslide_read_associated_image_icc_profile(
                osr,
                c_name.as_ptr(),
                buffer.as_mut_ptr() as *mut c_void,
            )
        });
        self.check_error()?;
        Ok(Some(buffer))
    }

    /// `openslide.OpenSlide.close()`. Idempotent: the handle is nulled, so a second call,
    /// or the `Drop` impl, does nothing.
    pub fn close(&self) {
        let mut guard = self.handle.lock();
        if !guard.0.is_null() {
            // SAFETY: the handle is live and no other thread can hold it — we own the lock.
            unsafe { openslide_close(guard.0) };
            guard.0 = std::ptr::null_mut();
        }
    }

    pub fn is_closed(&self) -> bool {
        self.handle.lock().0.is_null()
    }
}

impl Drop for OpenSlide {
    fn drop(&mut self) {
        self.close();
    }
}

fn to_cstring(path: &Path) -> Option<CString> {
    CString::new(path.to_string_lossy().as_bytes()).ok()
}

/// # Safety
/// `ptr` must be null or a valid NUL-terminated string.
unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}

/// # Safety
/// `array` must be null or a NULL-terminated array of valid C strings.
unsafe fn cstr_array_to_vec(array: *const *const c_char) -> Vec<String> {
    if array.is_null() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cursor = array;
    while !(*cursor).is_null() {
        if let Some(value) = cstr_to_string(*cursor) {
            out.push(value);
        }
        cursor = cursor.add(1);
    }
    out
}

/// Converts libopenslide's premultiplied ARGB32 to straight RGBA.
///
/// This is `openslide-python`'s `_load_image`, which hands the buffer to Pillow as raw mode
/// `BGRa` — "premultiplied BGRA". Pillow's `ImagingUnpackBGRa` un-premultiplies with
/// integer arithmetic, and the exact rounding is reproduced here because it decides the low
/// bits of every partially transparent pixel:
///
/// ```text
/// alpha == 255 || alpha == 0  ->  copy the channel through
/// otherwise                   ->  (255 * value + alpha / 2) / alpha, clamped to 255
/// ```
pub fn to_rgba(buffer: &PremultipliedArgb) -> image::RgbaImage {
    let mut out = image::RgbaImage::new(buffer.width, buffer.height);
    for (pixel, packed) in out.pixels_mut().zip(buffer.pixels.iter().copied()) {
        let alpha = ((packed >> 24) & 0xFF) as u8;
        let r = ((packed >> 16) & 0xFF) as u8;
        let g = ((packed >> 8) & 0xFF) as u8;
        let b = (packed & 0xFF) as u8;
        *pixel = image::Rgba([
            unpremultiply(r, alpha),
            unpremultiply(g, alpha),
            unpremultiply(b, alpha),
            alpha,
        ]);
    }
    out
}

#[inline]
fn unpremultiply(value: u8, alpha: u8) -> u8 {
    if alpha == 255 || alpha == 0 {
        return value;
    }
    let alpha = alpha as u32;
    let half = alpha / 2;
    let scaled = (255 * value as u32 + half) / alpha;
    scaled.min(255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_and_transparent_pixels_pass_straight_through() {
        // Pillow's unpacker short-circuits both, so no rounding is involved.
        assert_eq!(unpremultiply(200, 255), 200);
        assert_eq!(unpremultiply(200, 0), 200);
    }

    #[test]
    fn partial_alpha_is_unpremultiplied_with_pillows_rounding() {
        // (255 * 64 + 64) / 128 = 128
        assert_eq!(unpremultiply(64, 128), 128);
        // (255 * 10 + 2) / 5 = 510 -> clamped
        assert_eq!(unpremultiply(10, 5), 255);
        // (255 * 1 + 63) / 127 = 2
        assert_eq!(unpremultiply(1, 127), 2);
    }

    #[test]
    fn channels_are_reordered_from_argb_to_rgba() {
        // libopenslide packs A<<24 | R<<16 | G<<8 | B.
        let buffer = PremultipliedArgb {
            width: 1,
            height: 1,
            pixels: vec![0xFF_11_22_33],
        };
        let rgba = to_rgba(&buffer);
        assert_eq!(rgba.get_pixel(0, 0).0, [0x11, 0x22, 0x33, 0xFF]);
    }

    #[test]
    fn an_uncovered_pixel_is_transparent_black() {
        // What libopenslide writes outside the slide, and what the padding colour replaces.
        let buffer = PremultipliedArgb {
            width: 1,
            height: 1,
            pixels: vec![0],
        };
        assert_eq!(to_rgba(&buffer).get_pixel(0, 0).0, [0, 0, 0, 0]);
    }
}
