//! `DisableAuth`, ported from `wsi_service/api/v3/integrations/disable_auth.py`.
//!
//! Both hooks are no-ops. This is what the checked-in `.env` selects, and what a
//! standalone deployment behind its own access control uses.

use async_trait::async_trait;

use wsi_core::error::Result;

use crate::{AuthIntegration, AuthPayload, RequestContext};

pub struct DisableAuth;

#[async_trait]
impl AuthIntegration for DisableAuth {
    async fn authenticate(&self, _ctx: RequestContext<'_>) -> Result<AuthPayload> {
        Ok(AuthPayload::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn everything_is_permitted() {
        let ctx = RequestContext {
            authorization: None,
            path: "/v3/slides/info",
        };
        let payload = DisableAuth.authenticate(ctx).await.unwrap();
        assert!(payload.token.is_none());
        assert!(DisableAuth
            .allow_access_slide(&payload, "any-slide", None, None)
            .await
            .is_ok());
    }
}
