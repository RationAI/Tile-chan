//! ICC profile handling, mirroring `wsi_service/utils/icc_profile.py`.
//!
//! Python uses `PIL.ImageCms`, which is a thin wrapper over Little-CMS. This module uses
//! the `lcms2` crate — the same library — so a transform built here produces the same
//! pixels as the Python one for the same profile, intent and pixel format.
//!
//! Behaviour reproduced from `ICCProfile.process_pil_image`:
//!
//! * no profile and `strict == false` -> return the image untouched;
//! * no profile and `strict == true`  -> `412 "Error: ICC Profile not available."`;
//! * a profile -> transform to sRGB with the requested intent;
//! * any failure with `strict == false` is swallowed and the original image returned;
//! * transforms are cached per intent, but only when the caller opts in (`cache=True`),
//!   which region/tile reads do and macro/thumbnail reads do not.

use std::collections::HashMap;

use image::DynamicImage;
use lcms2::{Intent, PixelFormat, Profile, Transform};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, Result};

/// `wsi_service/custom_models/queries.py:46`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IccIntent {
    Perceptual,
    RelativeColorimetric,
    Saturation,
    AbsoluteColorimetric,
}

impl IccIntent {
    fn to_lcms(self) -> Intent {
        match self {
            IccIntent::Perceptual => Intent::Perceptual,
            IccIntent::RelativeColorimetric => Intent::RelativeColorimetric,
            IccIntent::Saturation => Intent::Saturation,
            IccIntent::AbsoluteColorimetric => Intent::AbsoluteColorimetric,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            IccIntent::Perceptual => "perceptual",
            IccIntent::RelativeColorimetric => "relative_colorimetric",
            IccIntent::Saturation => "saturation",
            IccIntent::AbsoluteColorimetric => "absolute_colorimetric",
        }
    }
}

impl std::str::FromStr for IccIntent {
    type Err = ApiError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "perceptual" => Ok(IccIntent::Perceptual),
            "relative_colorimetric" => Ok(IccIntent::RelativeColorimetric),
            "saturation" => Ok(IccIntent::Saturation),
            "absolute_colorimetric" => Ok(IccIntent::AbsoluteColorimetric),
            other => Err(ApiError::unprocessable(format!(
                "Invalid icc_profile_intent {other}!"
            ))),
        }
    }
}

/// The image layouts a transform can be built for. Mirrors the `image.mode` that Python
/// passes as both input and output format to `buildTransform`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Layout {
    Rgb,
    Rgba,
    Luma,
}

impl Layout {
    fn pixel_format(self) -> PixelFormat {
        match self {
            Layout::Rgb => PixelFormat::RGB_8,
            Layout::Rgba => PixelFormat::RGBA_8,
            Layout::Luma => PixelFormat::GRAY_8,
        }
    }
}

/// Per-slide ICC state: the cached transforms.
///
/// Python subclasses `dict` and keys by intent alone. Keying by `(intent, layout)` here is
/// strictly safer — a cached RGB transform must never be applied to an RGBA buffer — and
/// cannot change output, because a given slide's region reads always have one layout.
#[derive(Default)]
pub struct IccProfileCache {
    transforms: Mutex<HashMap<(IccIntent, Layout), CachedTransform>>,
}

/// `lcms2::Transform` is not `Sync`; the cache is behind a `Mutex` and transforms are used
/// only while that lock is held, so the wrapper is sound.
struct CachedTransform(Transform<u8, u8>);

// SAFETY: the transform is only ever dereferenced while the enclosing `Mutex` is held, so
// no two threads can touch the same `Transform` concurrently.
unsafe impl Send for CachedTransform {}
unsafe impl Sync for CachedTransform {}

impl IccProfileCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops cached transforms. Called on slide close (`free_cache` / `clear` in Python).
    pub fn free_cache(&self) {
        self.transforms.lock().clear();
    }

    /// Applies `profile_data` to `image`, converting to sRGB with `intent`.
    ///
    /// `cache` mirrors the Python argument: region and tile reads pass `true` because the
    /// profile is the slide's own and never changes; macro, label and thumbnail reads pass
    /// `false` because the associated image can carry a different profile.
    pub fn process_image(
        &self,
        image: DynamicImage,
        profile_data: Option<&[u8]>,
        strict: bool,
        intent: IccIntent,
        cache: bool,
    ) -> Result<DynamicImage> {
        let Some(profile_data) = profile_data.filter(|d| !d.is_empty()) else {
            if strict {
                // `icc_profile.py:55` raises inside the `try`, so its own `except` catches
                // it and re-raises at line 61 — which prefixes "Error: ". The doubled
                // handling is Python's, and the prefix is observable.
                return Err(ApiError::precondition_failed(
                    "Error: ICC Profile not available.",
                ));
            }
            return Ok(image);
        };

        match self.transform(image, profile_data, intent, cache) {
            Ok(out) => Ok(out),
            Err(err) => {
                if strict {
                    Err(ApiError::internal(format!("Error: {err}")))
                } else {
                    // Non-strict callers get the untransformed image, as in Python where
                    // the exception handler falls through to `return image`.
                    Err(err)
                }
            }
        }
    }

    /// Like [`process_image`], but never fails when `strict` is false: on error the
    /// original image is returned. This is the exact shape of the Python control flow, and
    /// the variant plugins should call.
    pub fn process_image_lenient(
        &self,
        image: DynamicImage,
        profile_data: Option<&[u8]>,
        strict: bool,
        intent: IccIntent,
        cache: bool,
    ) -> Result<DynamicImage> {
        let fallback = image.clone();
        match self.process_image(image, profile_data, strict, intent, cache) {
            Ok(out) => Ok(out),
            Err(err) if strict => Err(err),
            Err(_) => Ok(fallback),
        }
    }

    fn transform(
        &self,
        image: DynamicImage,
        profile_data: &[u8],
        intent: IccIntent,
        cache: bool,
    ) -> Result<DynamicImage> {
        let layout = match &image {
            DynamicImage::ImageLuma8(_) => Layout::Luma,
            DynamicImage::ImageRgba8(_) => Layout::Rgba,
            _ => Layout::Rgb,
        };

        let mut guard = self.transforms.lock();
        let key = (intent, layout);
        if !cache || !guard.contains_key(&key) {
            let src = Profile::new_icc(profile_data)
                .map_err(|e| ApiError::internal(format!("Error: invalid ICC profile ({e})")))?;
            let dst = Profile::new_srgb();
            let built = Transform::new(
                &src,
                layout.pixel_format(),
                &dst,
                layout.pixel_format(),
                intent.to_lcms(),
            )
            .map_err(|e| ApiError::internal(format!("Error: {e}")))?;
            if !cache {
                // Apply without touching the cache at all.
                return Ok(apply(&built, image, layout));
            }
            guard.insert(key, CachedTransform(built));
        }
        let cached = guard.get(&key).expect("just inserted");
        Ok(apply(&cached.0, image, layout))
    }

    /// `get_for_payload` — the raw profile bytes, or a failure when there is none.
    ///
    /// The absent case is **not** a 404. `icc_profile.py` imports `HTTPException` from
    /// `http.client`, not from FastAPI, so `raise HTTPException(404, "...")` constructs a
    /// plain exception that FastAPI never sees as an HTTP error; it escapes as a
    /// plain-text 500. Reproduced — see `docs/divergences.md`.
    pub fn payload(profile_data: Option<&[u8]>) -> Result<bytes::Bytes> {
        match profile_data.filter(|d| !d.is_empty()) {
            Some(data) => Ok(bytes::Bytes::copy_from_slice(data)),
            None => Err(ApiError::unhandled(
                "HTTPException(404, 'ICC Profile not available.')",
            )),
        }
    }
}

fn apply(transform: &Transform<u8, u8>, image: DynamicImage, layout: Layout) -> DynamicImage {
    match layout {
        Layout::Luma => {
            let mut buf = image.to_luma8();
            transform.transform_in_place(buf.as_mut());
            DynamicImage::ImageLuma8(buf)
        }
        Layout::Rgba => {
            let mut buf = image.to_rgba8();
            transform.transform_in_place(buf.as_mut());
            DynamicImage::ImageRgba8(buf)
        }
        Layout::Rgb => {
            let mut buf = image.to_rgb8();
            transform.transform_in_place(buf.as_mut());
            DynamicImage::ImageRgb8(buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn intent_round_trips_through_strings() {
        for intent in [
            IccIntent::Perceptual,
            IccIntent::RelativeColorimetric,
            IccIntent::Saturation,
            IccIntent::AbsoluteColorimetric,
        ] {
            assert_eq!(intent.as_str().parse::<IccIntent>().unwrap(), intent);
        }
        assert!("nonsense".parse::<IccIntent>().is_err());
    }

    #[test]
    fn no_profile_is_a_no_op_unless_strict() {
        let cache = IccProfileCache::new();
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([1, 2, 3])));

        let out = cache
            .process_image(img.clone(), None, false, IccIntent::Perceptual, true)
            .unwrap();
        assert_eq!(out.to_rgb8().get_pixel(0, 0), &Rgb([1, 2, 3]));

        let err = cache
            .process_image(img, None, true, IccIntent::Perceptual, true)
            .unwrap_err();
        assert_eq!(err.status, http::StatusCode::PRECONDITION_FAILED);
        // The "Error: " prefix is Python's doubled handling, not a mistake here: the 412 is
        // raised inside `process_pil_image`'s own `try`, caught, and re-raised prefixed.
        assert_eq!(err.detail_text(), "Error: ICC Profile not available.");
    }

    #[test]
    fn payload_requires_a_profile_and_a_missing_one_escapes_as_a_plain_500() {
        for absent in [None, Some(&[][..])] {
            let err = match IccProfileCache::payload(absent) {
                Ok(_) => panic!("expected an error"),
                Err(err) => err,
            };
            assert_eq!(err.status, http::StatusCode::INTERNAL_SERVER_ERROR);
            assert!(err.unhandled, "http.client.HTTPException escapes FastAPI");
        }
        assert_eq!(
            IccProfileCache::payload(Some(&[1, 2, 3])).unwrap().as_ref(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn invalid_profile_is_swallowed_when_not_strict() {
        let cache = IccProfileCache::new();
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([9, 9, 9])));
        let out = cache
            .process_image_lenient(img, Some(&[0, 1, 2]), false, IccIntent::Perceptual, false)
            .unwrap();
        assert_eq!(out.to_rgb8().get_pixel(0, 0), &Rgb([9, 9, 9]));
    }
}
