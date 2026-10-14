//! Authentication integrations, ported from `wsi_service/api/v3/integrations/`.
//!
//! Python resolves an integration class from `WS_API_V3_INTEGRATION` (a `module:Class`
//! descriptor) and gives it two jobs:
//!
//! * `global_depends()` — a FastAPI dependency that extracts and validates whatever the
//!   caller presented, returning an opaque "auth payload";
//! * `allow_access_slide(payload, slide_id, ...)` — a per-slide authorisation check that
//!   raises 401 or 403.
//!
//! The same split is kept here: [`AuthIntegration::authenticate`] runs once per request
//! before anything else, and [`AuthIntegration::allow_access_slide`] runs at the point
//! Python calls it — which is *not* uniform across endpoints, and the call sites reproduce
//! that ordering exactly, because it decides whether a client sees 401 or 422 when both
//! apply.

use std::sync::Arc;

use async_trait::async_trait;

use wsi_core::error::{ApiError, Result};
use wsi_core::models::SlideInfo;

pub mod disable_auth;
pub mod empaia;
pub mod jwt;
pub mod lsaai;
pub mod oauth;

pub use disable_auth::DisableAuth;
pub use empaia::EmpaiaIntegration;
pub use lsaai::{LsaaiIntegration, LsaaiMode};
pub use oauth::OAuthSettings;

/// Whatever the integration derived from the request. Opaque to the endpoints, which only
/// pass it back into `allow_access_slide`.
#[derive(Debug, Clone, Default)]
pub struct AuthPayload {
    /// Decoded JWT claims, when the integration validates a token.
    pub token: Option<serde_json::Value>,
    /// The bearer token as presented, for integrations that call a userinfo endpoint.
    pub raw_token: Option<String>,
}

/// The request context an integration may inspect.
pub struct RequestContext<'a> {
    pub authorization: Option<&'a str>,
    pub path: &'a str,
}

#[async_trait]
pub trait AuthIntegration: Send + Sync {
    /// Runs before the handler body, like the FastAPI dependency does.
    async fn authenticate(&self, ctx: RequestContext<'_>) -> Result<AuthPayload>;

    /// Per-slide authorisation. The default is a no-op, matching `Default` and
    /// `DisableAuth`, which both define it as `...`.
    ///
    /// `slide` is supplied when the caller already has the slide info, so the integration
    /// does not have to fetch it again — `/slides/info` passes it, the image endpoints do
    /// not.
    async fn allow_access_slide(
        &self,
        _payload: &AuthPayload,
        _slide_id: &str,
        _plugin: Option<&str>,
        _slide: Option<&SlideInfo>,
    ) -> Result<()> {
        Ok(())
    }

    /// Whether the OpenAPI document should advertise an OAuth2 security scheme.
    fn openapi_security(&self) -> Option<OpenApiSecurity> {
        None
    }
}

/// The OAuth2 authorization-code scheme the EMPAIA integration publishes.
#[derive(Debug, Clone)]
pub struct OpenApiSecurity {
    pub authorization_url: String,
    pub token_url: String,
}

/// `Default` — `integrations/default.py:11`.
///
/// Every request is rejected. This is what runs when `WS_API_V3_INTEGRATION` is unset, and
/// it is deliberately fail-closed: an unconfigured deployment serves nothing rather than
/// serving everything.
pub struct DefaultIntegration;

#[async_trait]
impl AuthIntegration for DefaultIntegration {
    async fn authenticate(&self, _ctx: RequestContext<'_>) -> Result<AuthPayload> {
        Err(ApiError::unauthorized("Integration not configured"))
    }
}

/// Builds an integration from a `WS_API_V3_INTEGRATION` descriptor.
///
/// The Python `module:Class` strings are accepted verbatim so existing `.env` files keep
/// working. An unrecognised descriptor is a startup error: silently falling back would
/// either lock everyone out or let everyone in, and both are worse than refusing to start.
pub fn from_descriptor(descriptor: &str) -> Result<Arc<dyn AuthIntegration>> {
    match descriptor {
        "" => Ok(Arc::new(DefaultIntegration)),
        "wsi_service.api.v3.integrations.disable_auth:DisableAuth" | "disable_auth" => {
            Ok(Arc::new(DisableAuth))
        }
        "wsi_service.api.v3.integrations.empaia:EmpaiaApiIntegration" | "empaia" => {
            Ok(Arc::new(EmpaiaIntegration::new(OAuthSettings::load())))
        }
        "wsi_service.api.v3.integrations.lsaai_groups:LSAAIIntegration" | "lsaai_groups" => {
            Ok(Arc::new(LsaaiIntegration::new(
                LsaaiMode::Groups,
                OAuthSettings::load(),
            )?))
        }
        "wsi_service.api.v3.integrations.lsaai_resources:LSAAIIntegration" | "lsaai_resources" => {
            Ok(Arc::new(LsaaiIntegration::new(
                LsaaiMode::Resources,
                OAuthSettings::load(),
            )?))
        }
        other => Err(ApiError::internal(format!(
            "Unknown WS_API_V3_INTEGRATION descriptor: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RequestContext<'static> {
        RequestContext {
            authorization: None,
            path: "/v3/slides/info",
        }
    }

    #[tokio::test]
    async fn default_integration_rejects_everything() {
        let err = match DefaultIntegration.authenticate(ctx()).await {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::UNAUTHORIZED);
        assert_eq!(err.detail_text(), "Integration not configured");
    }

    #[tokio::test]
    async fn unset_descriptor_selects_the_fail_closed_default() {
        let integration = from_descriptor("").unwrap();
        assert!(integration.authenticate(ctx()).await.is_err());
    }

    #[tokio::test]
    async fn python_disable_auth_descriptor_resolves() {
        let integration =
            from_descriptor("wsi_service.api.v3.integrations.disable_auth:DisableAuth").unwrap();
        assert!(integration.authenticate(ctx()).await.is_ok());
    }

    #[test]
    fn unknown_descriptor_refuses_to_start() {
        let err = match from_descriptor("some.module:Thing") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert!(err
            .detail_text()
            .contains("Unknown WS_API_V3_INTEGRATION descriptor"));
    }
}
