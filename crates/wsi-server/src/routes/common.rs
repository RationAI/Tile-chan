//! Parameter bundles shared by the image endpoints.
//!
//! Python declares the same handful of `Query(...)` objects on every route
//! (`wsi_service/custom_models/queries.py`); this module parses them once, in the same
//! order, with the same defaults.

use std::collections::HashMap;

use http::HeaderMap;

use wsi_auth::{AuthPayload, RequestContext};
use wsi_core::error::{ApiError, Result, ValidationDetail};
use wsi_core::icc::IccIntent;
use wsi_core::slide::ReadOptions;

use crate::query::{check_path_bounds, QueryParams};
use crate::state::AppState;

/// The image-shaping query parameters every raster endpoint accepts.
pub struct ImageQuery {
    pub image_format: String,
    pub image_quality: i64,
    pub image_channels: Option<Vec<i64>>,
    pub z: i64,
    pub padding_color: Option<String>,
    pub icc_profile_intent: Option<IccIntent>,
    pub icc_profile_strict: bool,
    pub plugin: Option<String>,
}

impl ImageQuery {
    pub fn parse(params: &QueryParams) -> Result<Self> {
        let mut validation = Validation::new();
        let query = validation.image_query(params);
        validation.finish()?;
        Ok(query)
    }

    pub fn read_options(&self, padding_color: (u8, u8, u8)) -> ReadOptions {
        ReadOptions {
            padding_color: Some(padding_color),
            z: self.z as u32,
            icc_profile_intent: self.icc_profile_intent,
            icc_profile_strict: self.icc_profile_strict,
        }
    }

    /// Channel indices as `usize`.
    ///
    /// `validate_image_channels` runs before this and rejects anything at or beyond the
    /// channel count, but it does not reject negatives — Python would index from the end
    /// of the list. Negative indices are dropped here instead of wrapping, because a
    /// wrapping index would silently return a different channel's pixels.
    pub fn channels_usize(&self) -> Option<Vec<usize>> {
        self.image_channels.as_ref().map(|chs| {
            chs.iter()
                .filter(|&&c| c >= 0)
                .map(|&c| c as usize)
                .collect()
        })
    }
}

/// Collects every failing parameter instead of stopping at the first.
///
/// FastAPI validates a request in one pass and answers with *all* the errors it found:
/// `/slides/thumbnail/max_size/9999/9999` reports `max_x` and `max_y`, not just `max_x`.
/// Returning after the first failure made the `detail` array a different length, which any
/// client that enumerates it sees.
///
/// Order matters too, and it is not the handler's signature order. `solve_dependencies`
/// walks the parameter locations in turn — path, then query — so every path error precedes
/// every query error even where the query parameter is declared first, as `slide_id` is.
/// The two are therefore collected separately and concatenated by [`Validation::finish`],
/// which leaves each handler free to read its parameters in whatever order suits it.
#[derive(Default)]
pub struct Validation {
    path_errors: Vec<ValidationDetail>,
    query_errors: Vec<ValidationDetail>,
    /// A failure that is not a validation error has no place in the 422 array; it is held
    /// here and returned as-is, since FastAPI would never have reached validation.
    fatal: Option<ApiError>,
}

impl Validation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a failure and substitutes a value so parsing can continue.
    ///
    /// The substituted value is never used: `finish` is called before any of them is read,
    /// and it returns `Err` whenever anything was recorded.
    fn record<T>(&mut self, result: Result<T>, fallback: T) -> T {
        match result {
            Ok(value) => value,
            Err(err) => {
                match &err.detail {
                    wsi_core::error::Detail::Validation(entries) => {
                        // Each entry names its own location, which is what sorts it into
                        // the right bucket.
                        for entry in entries {
                            if entry.loc.first().map(String::as_str) == Some("path") {
                                self.path_errors.push(entry.clone());
                            } else {
                                self.query_errors.push(entry.clone());
                            }
                        }
                    }
                    wsi_core::error::Detail::Message(_) => {
                        if self.fatal.is_none() {
                            self.fatal = Some(err);
                        }
                    }
                }
                fallback
            }
        }
    }

    pub fn path_int(
        &mut self,
        params: &HashMap<String, String>,
        name: &str,
        ge: Option<i64>,
        le: Option<i64>,
        gt: Option<i64>,
    ) -> i64 {
        let parsed = path_int(params, name, ge, le, gt);
        self.record(parsed, 0)
    }

    pub fn required_str(&mut self, params: &QueryParams, name: &str) -> String {
        let parsed = params.required_str(name);
        self.record(parsed, String::new())
    }

    /// The shared image query parameters, in `queries.py` declaration order.
    pub fn image_query(&mut self, params: &QueryParams) -> ImageQuery {
        ImageQuery {
            image_format: params.str_or("image_format", "jpeg"),
            image_quality: self.record(params.int_or("image_quality", 90, Some(0), Some(100)), 90),
            image_channels: self.record(params.image_channels(), None),
            z: self.record(params.int_or("z", 0, Some(0), None), 0),
            padding_color: params.optional_str("padding_color"),
            icc_profile_intent: self.record(params.icc_intent(), None),
            icc_profile_strict: self.record(params.bool_or("icc_profile_strict", false), false),
            plugin: params.optional_nonempty("plugin"),
        }
    }

    /// Fails with everything that was collected, or passes.
    pub fn finish(self) -> Result<()> {
        if let Some(fatal) = self.fatal {
            return Err(fatal);
        }
        let mut errors = self.path_errors;
        errors.extend(self.query_errors);
        if errors.is_empty() {
            return Ok(());
        }
        Err(ApiError::validation(errors))
    }
}

/// Runs the configured authentication integration.
///
/// FastAPI resolves the security dependency during dependency solving, before request
/// parameters are validated, so a missing token produces 401 rather than 422 even when the
/// query string is also malformed. Calling this first reproduces that precedence.
pub async fn authenticate(
    state: &AppState,
    headers: &HeaderMap,
    path: &str,
) -> Result<AuthPayload> {
    let authorization = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    state
        .auth
        .authenticate(RequestContext {
            authorization,
            path,
        })
        .await
}

/// Reads a named path parameter as an integer, with FastAPI's 422 body on failure.
pub fn path_int(
    params: &HashMap<String, String>,
    name: &str,
    ge: Option<i64>,
    le: Option<i64>,
    gt: Option<i64>,
) -> Result<i64> {
    let raw = params.get(name).ok_or_else(|| {
        ApiError::validation(vec![ValidationDetail::new(
            "path",
            name,
            "Field required",
            "missing",
            serde_json::Value::Null,
        )])
    })?;
    let value: i64 = raw.parse().map_err(|_| {
        ApiError::validation(vec![ValidationDetail::new(
            "path",
            name,
            "Input should be a valid integer, unable to parse string as an integer",
            "int_parsing",
            raw.as_str().into(),
        )])
    })?;
    check_path_bounds(name, raw, value, ge, le, gt)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn image_query_defaults_match_python() {
        let q = ImageQuery::parse(&QueryParams::parse(None)).unwrap();
        assert_eq!(q.image_format, "jpeg");
        assert_eq!(q.image_quality, 90);
        assert_eq!(q.z, 0);
        assert!(q.image_channels.is_none());
        assert!(q.padding_color.is_none());
        assert!(q.icc_profile_intent.is_none());
        assert!(!q.icc_profile_strict);
        assert!(q.plugin.is_none());
    }

    #[test]
    fn image_query_reads_every_parameter() {
        let raw = "image_format=png&image_quality=50&image_channels=1&image_channels=0\
                   &z=2&padding_color=%23000000&icc_profile_intent=saturation\
                   &icc_profile_strict=true&plugin=openslide";
        let q = ImageQuery::parse(&QueryParams::parse(Some(raw))).unwrap();
        assert_eq!(q.image_format, "png");
        assert_eq!(q.image_quality, 50);
        assert_eq!(q.image_channels.as_deref(), Some(&[1, 0][..]));
        assert_eq!(q.z, 2);
        assert_eq!(q.padding_color.as_deref(), Some("#000000"));
        assert_eq!(q.icc_profile_intent, Some(IccIntent::Saturation));
        assert!(q.icc_profile_strict);
        assert_eq!(q.plugin.as_deref(), Some("openslide"));
    }

    #[test]
    fn negative_channels_are_dropped_rather_than_wrapping() {
        let q = ImageQuery::parse(&QueryParams::parse(Some(
            "image_channels=-1&image_channels=2",
        )))
        .unwrap();
        assert_eq!(q.channels_usize().unwrap(), vec![2]);
    }

    #[test]
    fn path_int_enforces_bounds_and_reports_the_path_location() {
        let params = params_map(&[("max_x", "0"), ("level", "abc")]);
        let err = path_int(&params, "max_x", Some(1), Some(500), None).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["loc"][0], "path");
        assert_eq!(body[0]["type"], "greater_than_equal");

        let err = path_int(&params, "level", Some(0), None, None).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["type"], "int_parsing");

        let err = path_int(&params, "missing", None, None, None).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["type"], "missing");
    }
}
