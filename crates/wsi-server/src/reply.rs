//! Turning pipeline output into HTTP responses.

use axum::response::{IntoResponse, Response};
use http::header::{HeaderName, HeaderValue};

use wsi_core::image::EncodedResponse;

/// Wraps an [`EncodedResponse`], preserving the explicit `Content-Length` and
/// `Content-Encoding` headers that `make_passthrough_response` sets.
pub fn encoded(response: EncodedResponse) -> Response {
    let mut builder = http::Response::builder()
        .status(http::StatusCode::OK)
        .header(http::header::CONTENT_TYPE, response.media_type);

    for (name, value) in &response.headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            builder = builder.header(name, value);
        }
    }

    builder
        .body(axum::body::Body::from(response.body))
        .map(IntoResponse::into_response)
        .unwrap_or_else(|_| http::StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
