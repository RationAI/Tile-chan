//! Query-parameter parsing with FastAPI-compatible validation errors.
//!
//! Written by hand rather than derived through `serde` for three reasons:
//!
//! * `image_channels` is a repeated parameter (`?image_channels=0&image_channels=1`) that
//!   must collect into a list, which `serde_urlencoded` cannot express;
//! * the numeric constraints (`ge`, `le`, `gt`) have to produce FastAPI's 422 body, with
//!   the same `loc`, `msg` and `type` strings clients match on;
//! * missing-but-required parameters must fail the same way, not fall back to a default.

use std::collections::HashMap;

use wsi_core::error::{ApiError, Result, ValidationDetail};
use wsi_core::icc::IccIntent;

/// Parsed query string: parameter name to every value supplied for it, in order.
pub struct QueryParams {
    values: HashMap<String, Vec<String>>,
}

/// The permitted `icc_profile_intent` values, rendered the way pydantic renders an
/// enum's members. The same text appears in `msg`, after "Input should be ", and in
/// `ctx.expected`, which is why it is written once.
const ICC_INTENT_MEMBERS: &str =
    "'perceptual', 'relative_colorimetric', 'saturation' or 'absolute_colorimetric'";

/// Where a parameter came from, for the `loc` array in a validation error.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Query,
    Path,
}

impl Location {
    fn as_str(self) -> &'static str {
        match self {
            Location::Query => "query",
            Location::Path => "path",
        }
    }
}

/// Builds the single-entry 422 body FastAPI emits for one failing parameter.
///
/// `input` is the value exactly as it arrived; pass `Value::Null` for a parameter that was
/// not supplied at all, which is what pydantic reports for `missing`.
fn validation_error(
    location: Location,
    name: &str,
    msg: &str,
    kind: &str,
    input: serde_json::Value,
) -> ApiError {
    ApiError::validation(vec![ValidationDetail::new(
        location.as_str(),
        name,
        msg,
        kind,
        input,
    )])
}

/// As [`validation_error`], with the `ctx` object naming the violated constraint.
fn constraint_error(
    location: Location,
    name: &str,
    msg: &str,
    kind: &str,
    input: serde_json::Value,
    ctx: serde_json::Value,
) -> ApiError {
    ApiError::validation(vec![ValidationDetail::new(
        location.as_str(),
        name,
        msg,
        kind,
        input,
    )
    .with_ctx(ctx)])
}

impl QueryParams {
    /// Parses a raw query string. Percent-decoding follows `application/x-www-form-urlencoded`,
    /// including `+` as a space, which is what Starlette's `QueryParams` does.
    pub fn parse(raw: Option<&str>) -> Self {
        let mut values: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(raw) = raw {
            for pair in raw.split('&').filter(|p| !p.is_empty()) {
                let (key, value) = match pair.split_once('=') {
                    Some((k, v)) => (k, v),
                    None => (pair, ""),
                };
                values.entry(decode(key)).or_default().push(decode(value));
            }
        }
        Self { values }
    }

    fn first(&self, name: &str) -> Option<&str> {
        self.values
            .get(name)
            .and_then(|v| v.first())
            .map(String::as_str)
    }

    /// A required string parameter, e.g. `slide_id` or `paths`.
    pub fn required_str(&self, name: &str) -> Result<String> {
        match self.first(name) {
            Some(value) => Ok(value.to_owned()),
            None => Err(validation_error(
                Location::Query,
                name,
                "Field required",
                "missing",
                serde_json::Value::Null,
            )),
        }
    }

    /// An optional string parameter with no default.
    pub fn optional_str(&self, name: &str) -> Option<String> {
        self.first(name).map(str::to_owned)
    }

    /// An optional string parameter, treating an empty value as absent — which is how
    /// `plugin=` behaves in Python, where the empty string is falsy.
    pub fn optional_nonempty(&self, name: &str) -> Option<String> {
        self.first(name)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    }

    pub fn str_or(&self, name: &str, default: &str) -> String {
        self.first(name).unwrap_or(default).to_owned()
    }

    /// An optional integer with `ge`/`le` bounds, defaulted when absent.
    pub fn int_or(
        &self,
        name: &str,
        default: i64,
        ge: Option<i64>,
        le: Option<i64>,
    ) -> Result<i64> {
        let Some(raw) = self.first(name) else {
            return Ok(default);
        };
        let value: i64 = raw.parse().map_err(|_| {
            validation_error(
                Location::Query,
                name,
                "Input should be a valid integer, unable to parse string as an integer",
                "int_parsing",
                raw.into(),
            )
        })?;
        check_bounds(Location::Query, name, raw, value, ge, le)?;
        Ok(value)
    }

    /// A boolean with pydantic's coercion rules.
    pub fn bool_or(&self, name: &str, default: bool) -> Result<bool> {
        let Some(raw) = self.first(name) else {
            return Ok(default);
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "yes" | "y" | "on" => Ok(true),
            "0" | "false" | "f" | "no" | "n" | "off" => Ok(false),
            _ => Err(validation_error(
                Location::Query,
                name,
                "Input should be a valid boolean, unable to interpret input",
                "bool_parsing",
                raw.into(),
            )),
        }
    }

    /// `image_channels`: repeated integers, absent means "all channels".
    ///
    /// Negative values are rejected here; Python lets them through to
    /// `validate_image_channels`, whose bounds check only looks at the upper end, so a
    /// negative index would index from the end of the channel list. That is not reachable
    /// through FastAPI, which types the parameter as `List[int]` but is fed by a query
    /// string — so a negative value *is* accepted there. Reproduced: negatives pass
    /// validation and are clamped by the channel selection below.
    pub fn image_channels(&self) -> Result<Option<Vec<i64>>> {
        let Some(raw_values) = self.values.get("image_channels") else {
            return Ok(None);
        };
        if raw_values.is_empty() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(raw_values.len());
        for raw in raw_values {
            let value: i64 = raw.parse().map_err(|_| {
                validation_error(
                    Location::Query,
                    "image_channels",
                    "Input should be a valid integer, unable to parse string as an integer",
                    "int_parsing",
                    raw.as_str().into(),
                )
            })?;
            out.push(value);
        }
        Ok(Some(out))
    }

    /// `icc_profile_intent`: an optional enum. An unrecognised value is a 422 naming the
    /// permitted members, as FastAPI does for a `str, Enum` query parameter.
    pub fn icc_intent(&self) -> Result<Option<IccIntent>> {
        let Some(raw) = self.first("icc_profile_intent").filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        raw.parse::<IccIntent>().map(Some).map_err(|_| {
            constraint_error(
                Location::Query,
                "icc_profile_intent",
                &format!("Input should be {ICC_INTENT_MEMBERS}"),
                "enum",
                raw.into(),
                serde_json::json!({ "expected": ICC_INTENT_MEMBERS }),
            )
        })
    }

    /// Comma-separated integer list, used by `/batch/?levels=&xs=&ys=`.
    pub fn required_int_list(&self, name: &str) -> Result<Vec<i64>> {
        let raw = self.required_str(name)?;
        raw.split(',')
            .map(|part| {
                part.trim().parse::<i64>().map_err(|_| {
                    validation_error(
                        Location::Query,
                        name,
                        "Input should be a valid integer, unable to parse string as an integer",
                        "int_parsing",
                        raw.as_str().into(),
                    )
                })
            })
            .collect()
    }
}

/// Validates a path parameter against `ge`/`le`/`gt` constraints, producing the same 422
/// body FastAPI emits for `Path(ge=..., le=...)`.
pub fn check_path_bounds(
    name: &str,
    raw: &str,
    value: i64,
    ge: Option<i64>,
    le: Option<i64>,
    gt: Option<i64>,
) -> Result<()> {
    if let Some(gt) = gt {
        if value <= gt {
            return Err(constraint_error(
                Location::Path,
                name,
                &format!("Input should be greater than {gt}"),
                "greater_than",
                raw.into(),
                serde_json::json!({ "gt": gt }),
            ));
        }
    }
    check_bounds(Location::Path, name, raw, value, ge, le)
}

fn check_bounds(
    location: Location,
    name: &str,
    raw: &str,
    value: i64,
    ge: Option<i64>,
    le: Option<i64>,
) -> Result<()> {
    if let Some(ge) = ge {
        if value < ge {
            return Err(constraint_error(
                location,
                name,
                &format!("Input should be greater than or equal to {ge}"),
                "greater_than_equal",
                raw.into(),
                serde_json::json!({ "ge": ge }),
            ));
        }
    }
    if let Some(le) = le {
        if value > le {
            return Err(constraint_error(
                location,
                name,
                &format!("Input should be less than or equal to {le}"),
                "less_than_equal",
                raw.into(),
                serde_json::json!({ "le": le }),
            ));
        }
    }
    Ok(())
}

/// `application/x-www-form-urlencoded` decoding: `+` is a space, `%XX` is a byte.
fn decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(raw: &str) -> QueryParams {
        QueryParams::parse(Some(raw))
    }

    #[test]
    fn percent_and_plus_are_decoded() {
        let p = params("padding_color=%23FF8000&name=a+b");
        assert_eq!(p.optional_str("padding_color").unwrap(), "#FF8000");
        assert_eq!(p.optional_str("name").unwrap(), "a b");
    }

    #[test]
    fn slide_ids_containing_slashes_survive_decoding() {
        // PathsMapper IDs are paths, so they arrive percent-encoded.
        let p = params("slide_id=proj%2FcaseA%2Fone.svs");
        assert_eq!(p.required_str("slide_id").unwrap(), "proj/caseA/one.svs");
    }

    #[test]
    fn missing_required_parameter_is_a_fastapi_422() {
        let err = params("").required_str("slide_id").unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["loc"][0], "query");
        assert_eq!(body[0]["loc"][1], "slide_id");
        assert_eq!(body[0]["type"], "missing");
    }

    #[test]
    fn defaults_match_the_python_query_declarations() {
        let p = params("");
        assert_eq!(p.str_or("image_format", "jpeg"), "jpeg");
        assert_eq!(
            p.int_or("image_quality", 90, Some(0), Some(100)).unwrap(),
            90
        );
        assert_eq!(p.int_or("z", 0, Some(0), None).unwrap(), 0);
        assert!(p.optional_nonempty("plugin").is_none());
        assert!(p.image_channels().unwrap().is_none());
        assert!(p.icc_intent().unwrap().is_none());
        assert!(!p.bool_or("icc_profile_strict", false).unwrap());
    }

    #[test]
    fn empty_plugin_is_treated_as_absent() {
        let p = params("plugin=");
        assert!(p.optional_nonempty("plugin").is_none());
    }

    #[test]
    fn quality_bounds_are_enforced() {
        let p = params("image_quality=101");
        let err = p
            .int_or("image_quality", 90, Some(0), Some(100))
            .unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["type"], "less_than_equal");

        let p = params("image_quality=-1");
        let body = serde_json::to_value(
            &p.int_or("image_quality", 90, Some(0), Some(100))
                .unwrap_err()
                .detail,
        )
        .unwrap();
        assert_eq!(body[0]["type"], "greater_than_equal");
    }

    #[test]
    fn z_must_be_non_negative() {
        let p = params("z=-1");
        assert!(p.int_or("z", 0, Some(0), None).is_err());
    }

    #[test]
    fn repeated_image_channels_collect_into_a_list() {
        let p = params("image_channels=0&image_channels=2&image_channels=1");
        assert_eq!(p.image_channels().unwrap().unwrap(), vec![0, 2, 1]);
    }

    #[test]
    fn non_integer_channel_is_a_422() {
        let p = params("image_channels=x");
        assert!(p.image_channels().is_err());
    }

    #[test]
    fn icc_intent_parses_every_member() {
        for (raw, expected) in [
            ("perceptual", IccIntent::Perceptual),
            ("relative_colorimetric", IccIntent::RelativeColorimetric),
            ("saturation", IccIntent::Saturation),
            ("absolute_colorimetric", IccIntent::AbsoluteColorimetric),
        ] {
            let p = params(&format!("icc_profile_intent={raw}"));
            assert_eq!(p.icc_intent().unwrap(), Some(expected));
        }
        let p = params("icc_profile_intent=nonsense");
        assert!(p.icc_intent().is_err());
    }

    #[test]
    fn boolean_coercion_matches_pydantic() {
        assert!(params("icc_profile_strict=true")
            .bool_or("icc_profile_strict", false)
            .unwrap());
        assert!(params("icc_profile_strict=1")
            .bool_or("icc_profile_strict", false)
            .unwrap());
        assert!(!params("icc_profile_strict=off")
            .bool_or("icc_profile_strict", false)
            .unwrap());
        assert!(params("icc_profile_strict=maybe")
            .bool_or("icc_profile_strict", false)
            .is_err());
    }

    #[test]
    fn comma_separated_int_lists_parse() {
        let p = params("levels=0,5,1,2");
        assert_eq!(p.required_int_list("levels").unwrap(), vec![0, 5, 1, 2]);
        assert!(params("levels=0,x").required_int_list("levels").is_err());
        assert!(params("").required_int_list("levels").is_err());
    }

    #[test]
    fn path_bounds_produce_path_scoped_errors() {
        let err = check_path_bounds("max_x", "0", 0, Some(1), Some(500), None).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["loc"][0], "path");
        assert_eq!(body[0]["loc"][1], "max_x");
        assert_eq!(body[0]["type"], "greater_than_equal");
        // pydantic reports the raw string it was given, and the bound it checked.
        assert_eq!(body[0]["input"], "0");
        assert_eq!(body[0]["ctx"]["ge"], 1);
        assert_eq!(
            body[0]["url"],
            "https://errors.pydantic.dev/2.11/v/greater_than_equal"
        );

        let err = check_path_bounds("max_x", "501", 501, Some(1), Some(500), None).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["type"], "less_than_equal");
        assert_eq!(body[0]["ctx"]["le"], 500);
        assert!(check_path_bounds("max_x", "250", 250, Some(1), Some(500), None).is_ok());

        // size_x uses gt=0 rather than ge
        let err = check_path_bounds("size_x", "0", 0, None, None, Some(0)).unwrap_err();
        let body = serde_json::to_value(&err.detail).unwrap();
        assert_eq!(body[0]["type"], "greater_than");
        assert_eq!(body[0]["ctx"]["gt"], 0);
        assert!(check_path_bounds("size_x", "1", 1, None, None, Some(0)).is_ok());
    }
}
