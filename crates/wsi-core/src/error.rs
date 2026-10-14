//! Error type mirroring FastAPI's `HTTPException`.
//!
//! Python raises `HTTPException(status_code=..., detail=...)` and FastAPI renders it as
//! `{"detail": "..."}`. Request-validation failures instead render as a list of error
//! objects; [`ApiError::validation`] reproduces that shape so clients that inspect
//! `detail[0].loc` keep working.

use axum::response::{IntoResponse, Response};
use http::StatusCode;
use serde::Serialize;

/// The `{"detail": ...}` envelope. `detail` is a string for `HTTPException` and a list for
/// request-validation failures, exactly as FastAPI serialises them.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Detail {
    Message(String),
    Validation(Vec<ValidationDetail>),
}

/// The pydantic minor version whose documentation the `url` field points at.
///
/// pydantic builds that URL from its own version, so it moves when the Python service
/// upgrades. It is pinned here to the version `wsi_service` resolves
/// (`fastapi` -> `pydantic ^2`), and the parity harness is what notices if it drifts.
const PYDANTIC_ERROR_DOC_VERSION: &str = "2.11";

/// One entry of FastAPI's 422 `detail` array.
///
/// Field order is pydantic's own, because the body is compared verbatim in places:
/// `type`, `loc`, `msg`, `input`, `ctx` (only for the error kinds that carry one), `url`.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationDetail {
    #[serde(rename = "type")]
    pub kind: String,
    /// e.g. `["query", "image_quality"]` or `["path", "max_x"]`.
    pub loc: Vec<String>,
    pub msg: String,
    /// The value as it arrived, before coercion — a string for anything that was present,
    /// and `null` for a missing parameter. pydantic reports the *raw* input, so an
    /// out-of-range `image_quality=101` reports `"101"`, not `101`.
    pub input: serde_json::Value,
    /// The constraint that was violated: `{"le": 100}`, `{"ge": 0}`, `{"gt": 0}` or, for
    /// an enum, `{"expected": "..."}`. Absent for kinds without a constraint, such as
    /// `missing` and `int_parsing`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<serde_json::Value>,
    pub url: String,
}

impl ValidationDetail {
    /// Builds an entry, deriving `url` from `kind` the way pydantic does.
    pub fn new(
        location: &str,
        name: &str,
        msg: impl Into<String>,
        kind: &str,
        input: serde_json::Value,
    ) -> Self {
        Self {
            kind: kind.to_owned(),
            loc: vec![location.to_owned(), name.to_owned()],
            msg: msg.into(),
            input,
            ctx: None,
            url: format!("https://errors.pydantic.dev/{PYDANTIC_ERROR_DOC_VERSION}/v/{kind}"),
        }
    }

    pub fn with_ctx(mut self, ctx: serde_json::Value) -> Self {
        self.ctx = Some(ctx);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub detail: Detail,
    /// True when the Python original raises an exception FastAPI does **not** handle.
    ///
    /// `wsi_service` registers no exception handlers, so a bare `ValueError`,
    /// `AttributeError`, `IndexError`, `numpy.AxisError`, or the `http.client.HTTPException`
    /// that `icc_profile.get_for_payload` raises all fall through to Starlette's
    /// `ServerErrorMiddleware`, which answers `500` with the plain-text body
    /// `Internal Server Error` — not a JSON `{"detail": ...}` envelope. `detail` still holds
    /// the Python exception's `repr`, because the batch endpoints catch these and write that
    /// text into the archive.
    pub unhandled: bool,
}

impl ApiError {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: Detail::Message(detail.into()),
            unhandled: false,
        }
    }

    /// An exception that escapes FastAPI: `500` with Starlette's plain-text body.
    ///
    /// `python_repr` is the `repr()` of the exception Python would have raised, which is
    /// what the batch endpoints write into a `t{n}.err` entry when they catch it.
    pub fn unhandled(python_repr: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: Detail::Message(python_repr.into()),
            unhandled: true,
        }
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, detail)
    }

    pub fn unauthorized(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, detail)
    }

    pub fn forbidden(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, detail)
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, detail)
    }

    /// 412 — used by the ICC layer when a profile is required but absent.
    pub fn precondition_failed(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::PRECONDITION_FAILED, detail)
    }

    /// 416 — `get_extended_region` with `extend=false` and no overlap.
    pub fn range_not_satisfiable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::RANGE_NOT_SATISFIABLE, detail)
    }

    /// 422 with a plain string detail. Python raises `HTTPException(422, "...")` for the
    /// level/z/size validators, so the body is a string there, *not* the validation array.
    pub fn unprocessable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, detail)
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, detail)
    }

    pub fn service_unavailable(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, detail)
    }

    pub fn not_implemented(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, detail)
    }

    /// FastAPI's request-validation 422 shape.
    pub fn validation(entries: Vec<ValidationDetail>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: Detail::Validation(entries),
            unhandled: false,
        }
    }

    /// The message text, for the places Python concatenates `e.detail` (plugin selection)
    /// or embeds it in another message.
    pub fn detail_text(&self) -> String {
        match &self.detail {
            Detail::Message(m) => m.clone(),
            Detail::Validation(v) => {
                serde_json::to_string(v).unwrap_or_else(|_| "validation error".to_owned())
            }
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status.as_u16(), self.detail_text())
    }
}

impl std::error::Error for ApiError {}

#[derive(Serialize)]
struct Envelope<'a> {
    detail: &'a Detail,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.unhandled {
            // Starlette's ServerErrorMiddleware with debug off.
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                "Internal Server Error",
            )
                .into_response();
        }
        let body = serde_json::to_vec(&Envelope {
            detail: &self.detail,
        })
        .unwrap_or_else(|_| br#"{"detail":"Internal Server Error"}"#.to_vec());
        (
            self.status,
            [(http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response()
    }
}

pub type Result<T, E = ApiError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn rendered(error: ApiError) -> (StatusCode, String, String) {
        let response = error.into_response();
        let status = response.status();
        let content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            status,
            content_type,
            String::from_utf8_lossy(&body).into_owned(),
        )
    }

    #[tokio::test]
    async fn an_http_exception_renders_as_a_json_detail_envelope() {
        let (status, content_type, body) = rendered(ApiError::bad_request(
            "Provided image format parameter not supported",
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(content_type, "application/json");
        assert_eq!(
            body,
            r#"{"detail":"Provided image format parameter not supported"}"#
        );
    }

    #[tokio::test]
    async fn a_validation_error_renders_as_a_json_detail_array() {
        let (status, content_type, body) =
            rendered(ApiError::validation(vec![ValidationDetail::new(
                "query",
                "slide_id",
                "Field required",
                "missing",
                serde_json::Value::Null,
            )]))
            .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(content_type, "application/json");
        // Byte-for-byte what FastAPI emits, field order included — `ctx` is absent because
        // `missing` has no constraint, and `input` is null rather than omitted.
        assert_eq!(
            body,
            r#"{"detail":[{"type":"missing","loc":["query","slide_id"],"msg":"Field required","input":null,"url":"https://errors.pydantic.dev/2.11/v/missing"}]}"#
        );
    }

    #[tokio::test]
    async fn a_constraint_violation_carries_the_bound_in_ctx() {
        let detail = ValidationDetail::new(
            "path",
            "max_x",
            "Input should be less than or equal to 500",
            "less_than_equal",
            "9999".into(),
        )
        .with_ctx(serde_json::json!({ "le": 500 }));
        let (_, _, body) = rendered(ApiError::validation(vec![detail])).await;
        assert_eq!(
            body,
            r#"{"detail":[{"type":"less_than_equal","loc":["path","max_x"],"msg":"Input should be less than or equal to 500","input":"9999","ctx":{"le":500},"url":"https://errors.pydantic.dev/2.11/v/less_than_equal"}]}"#
        );
    }

    #[tokio::test]
    async fn an_unhandled_exception_renders_as_starlettes_plain_text_500() {
        let error = ApiError::unhandled("ValueError(\"boom\")");
        // The Python repr is kept for the batch archive's .err entries...
        assert_eq!(error.detail_text(), "ValueError(\"boom\")");
        // ...but the response body is Starlette's, and carries none of it.
        let (status, content_type, body) = rendered(error).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(content_type, "text/plain; charset=utf-8");
        assert_eq!(body, "Internal Server Error");
    }

    #[tokio::test]
    async fn a_deliberate_500_still_renders_as_json() {
        // `HTTPException(status_code=500, ...)` from the plugin loader is handled by
        // FastAPI, so it keeps the JSON envelope.
        let (status, content_type, body) =
            rendered(ApiError::internal("Plugin pil unable to open image (boom)")).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(content_type, "application/json");
        assert_eq!(
            body,
            r#"{"detail":"Plugin pil unable to open image (boom)"}"#
        );
    }
}
