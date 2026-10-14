//! `EmpaiaApiIntegration`, ported from
//! `wsi_service/api/v3/integrations/empaia.py`.
//!
//! Validates the bearer token against the configured IdP and publishes an OAuth2
//! authorization-code scheme in the OpenAPI document. `allow_access_slide` is a no-op —
//! authentication is the whole of the policy here.

use std::sync::Arc;

use async_trait::async_trait;

use wsi_core::error::{ApiError, Result};

use crate::jwt::Auth;
use crate::oauth::OAuthSettings;
use crate::{AuthIntegration, AuthPayload, OpenApiSecurity, RequestContext};

pub struct EmpaiaIntegration {
    auth: Arc<Auth>,
    settings: OAuthSettings,
}

impl EmpaiaIntegration {
    pub fn new(settings: OAuthSettings) -> Self {
        let auth = Arc::new(Auth::new(
            settings.idp_url.trim_end_matches('/'),
            settings.refresh_interval,
            Some(settings.audience.clone()),
            Some(settings.rewrite_url_in_wellknown.clone()),
            // The EMPAIA variant has no ACR check.
            None,
            0,
        ));
        Self { auth, settings }
    }
}

#[async_trait]
impl AuthIntegration for EmpaiaIntegration {
    async fn authenticate(&self, ctx: RequestContext<'_>) -> Result<AuthPayload> {
        // FastAPI's `OAuth2` scheme rejects a request with no Authorization header before
        // the token is ever decoded.
        let Some(header) = ctx.authorization.filter(|h| !h.is_empty()) else {
            return Err(ApiError::unauthorized("Not authenticated"));
        };
        let claims = self.auth.decode_token(header).await?;
        Ok(AuthPayload {
            token: Some(claims),
            raw_token: Some(strip_bearer(header).to_owned()),
        })
    }

    fn openapi_security(&self) -> Option<OpenApiSecurity> {
        Some(OpenApiSecurity {
            authorization_url: self.settings.openapi_auth_url.clone(),
            token_url: self.settings.openapi_token_url.clone(),
        })
    }
}

pub fn strip_bearer(header: &str) -> &str {
    if header.len() >= 7 && header[..7].eq_ignore_ascii_case("bearer ") {
        &header[7..]
    } else {
        header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integration() -> EmpaiaIntegration {
        EmpaiaIntegration::new(OAuthSettings {
            idp_url: "http://127.0.0.1:1/unreachable".into(),
            audience: "org.empaia.auth.wbs".into(),
            openapi_auth_url: "https://idp.example/auth".into(),
            openapi_token_url: "https://idp.example/token".into(),
            ..OAuthSettings::default()
        })
    }

    #[test]
    fn bearer_prefix_is_stripped_case_insensitively() {
        assert_eq!(strip_bearer("Bearer abc"), "abc");
        assert_eq!(strip_bearer("bearer abc"), "abc");
        assert_eq!(strip_bearer("BEARER abc"), "abc");
        assert_eq!(strip_bearer("abc"), "abc");
    }

    #[tokio::test]
    async fn a_missing_authorization_header_is_a_401() {
        let integration = integration();
        let ctx = RequestContext {
            authorization: None,
            path: "/v3/slides/info",
        };
        let err = match integration.authenticate(ctx).await {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::UNAUTHORIZED);
        assert_eq!(err.detail_text(), "Not authenticated");
    }

    #[tokio::test]
    async fn a_token_is_rejected_when_the_provider_is_unreachable() {
        let integration = integration();
        let ctx = RequestContext {
            authorization: Some("Bearer whatever"),
            path: "/v3/slides/info",
        };
        let err = match integration.authenticate(ctx).await {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Auth not initialized on server.");
    }

    #[test]
    fn the_openapi_document_advertises_the_oauth2_flow() {
        let security = integration().openapi_security().unwrap();
        assert_eq!(security.authorization_url, "https://idp.example/auth");
        assert_eq!(security.token_url, "https://idp.example/token");
    }
}
