//! JWT validation against an OpenID Connect provider.
//!
//! Ported from `wsi_service/empaia_receiver_auth/__init__.py` and the near-identical
//! `wsi_service/rationai_receiver_auth/__init__.py`, which adds an `acr` claim check.
//!
//! The provider's metadata and signing keys are discovered from
//! `{idp_url}/.well-known/openid-configuration` and refreshed every `refresh_interval`
//! seconds, lazily — a fetch is attempted on the next token that arrives after the
//! interval lapses, not on a timer.
//!
//! Discovery failures are logged and swallowed, exactly as in Python: the service keeps
//! running and every token is rejected with *Auth not initialized on server.* until the
//! provider comes back. A dead IdP must not take the service down with it.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use parking_lot::RwLock;
use serde::Deserialize;

use wsi_core::error::{ApiError, Result};

/// How long discovery requests may take. Python passes `timeout=8` to both.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Debug, Deserialize)]
struct WellKnown {
    #[serde(default)]
    id_token_signing_alg_values_supported: Vec<String>,
    #[serde(default)]
    issuer: Option<String>,
    #[serde(default)]
    jwks_uri: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JwkSet {
    #[serde(default)]
    keys: Vec<serde_json::Value>,
}

/// Everything discovery produces, replaced atomically on a successful refresh.
struct Discovered {
    public_keys: HashMap<String, DecodingKey>,
    supported_algorithms: HashSet<String>,
    issuer: String,
    fetched_at: Instant,
}

pub struct Auth {
    idp_url: String,
    well_known_url: String,
    refresh_interval: Duration,
    audience: Option<String>,
    rewrite_url_in_wellknown: Option<String>,
    /// `check_acr_values` — only the RationAI variant sets it.
    acr_values: Option<String>,
    leeway: u64,
    client: reqwest::Client,
    discovered: RwLock<Option<Discovered>>,
}

impl Auth {
    pub fn new(
        idp_url: &str,
        refresh_interval: u64,
        audience: Option<String>,
        rewrite_url_in_wellknown: Option<String>,
        acr_values: Option<String>,
        leeway: u64,
    ) -> Self {
        let idp_url = idp_url.trim_end_matches('/').to_owned();
        Self {
            well_known_url: format!("{idp_url}/.well-known/openid-configuration"),
            idp_url,
            refresh_interval: Duration::from_secs(refresh_interval),
            audience: audience.filter(|a| !a.is_empty()),
            rewrite_url_in_wellknown: rewrite_url_in_wellknown.filter(|u| !u.is_empty()),
            acr_values: acr_values.filter(|a| !a.is_empty()),
            leeway,
            client: reqwest::Client::builder()
                .timeout(DISCOVERY_TIMEOUT)
                .build()
                .unwrap_or_default(),
            discovered: RwLock::new(None),
        }
    }

    /// `_rewrite_url` — rewrites a discovered URL to point back at the configured IdP.
    ///
    /// Needed when the provider advertises its public hostname but the service reaches it
    /// over an internal one.
    fn rewrite_url(&self, url: &str) -> String {
        let Some(prefix) = self.rewrite_url_in_wellknown.as_deref() else {
            return url.to_owned();
        };
        let prefix = prefix.trim_end_matches('/');
        if !url.starts_with(prefix) {
            tracing::error!(
                "Could not rewrite URL in wellknown, because URL {url} does not start with \
                 rewrite_url_in_wellknown {prefix}"
            );
            return url.to_owned();
        }
        let path = url[prefix.len()..].trim_start_matches('/');
        format!("{}/{}", self.idp_url, path)
    }

    fn needs_refresh(&self) -> bool {
        match self.discovered.read().as_ref() {
            None => true,
            Some(state) => state.fetched_at.elapsed() > self.refresh_interval,
        }
    }

    /// `_ensure_updated_internal` + `_update_internal_values`.
    ///
    /// Every failure path logs and returns without touching the cached state, so a
    /// transient provider outage leaves the previous keys in place.
    pub async fn ensure_updated(&self) {
        if !self.needs_refresh() {
            return;
        }

        let well_known: WellKnown = match self.fetch_json(&self.well_known_url).await {
            Ok(value) => match serde_json::from_value(value) {
                Ok(parsed) => parsed,
                Err(err) => {
                    tracing::error!("Well-known data is not a dict. ({err})");
                    return;
                }
            },
            Err(err) => {
                tracing::error!("Could not retrieve well-known data: {err}");
                return;
            }
        };

        if well_known.id_token_signing_alg_values_supported.is_empty() {
            tracing::error!("Well-known data does not contain supported algorithms.");
            return;
        }
        let Some(issuer) = well_known.issuer.filter(|i| !i.is_empty()) else {
            tracing::error!("Well-known data does not contain issuer.");
            return;
        };
        let Some(certs_url) = well_known.jwks_uri.filter(|u| !u.is_empty()) else {
            tracing::error!("Well-known data does not contain certs url");
            return;
        };
        let certs_url = self.rewrite_url(&certs_url);

        let certs: JwkSet = match self.fetch_json(&certs_url).await {
            Ok(value) => match serde_json::from_value(value) {
                Ok(parsed) => parsed,
                Err(err) => {
                    tracing::error!("Could not retrieve certs data: {err}");
                    return;
                }
            },
            Err(err) => {
                tracing::error!("Could not retrieve certs data: {err}");
                return;
            }
        };
        if certs.keys.is_empty() {
            tracing::error!("Certs data does not contain keys");
            return;
        }

        let mut public_keys = HashMap::new();
        for key in &certs.keys {
            let Some(kid) = key.get("kid").and_then(|k| k.as_str()) else {
                tracing::error!("Could not convert public keys from certs data: missing 'kid'");
                return;
            };
            let jwk: jsonwebtoken::jwk::Jwk = match serde_json::from_value(key.clone()) {
                Ok(jwk) => jwk,
                Err(err) => {
                    tracing::error!("Could not convert public keys from certs data: {err}");
                    return;
                }
            };
            match DecodingKey::from_jwk(&jwk) {
                Ok(decoding_key) => {
                    public_keys.insert(kid.to_owned(), decoding_key);
                }
                Err(err) => {
                    tracing::error!("Could not convert public keys from certs data: {err}");
                    return;
                }
            }
        }

        *self.discovered.write() = Some(Discovered {
            public_keys,
            supported_algorithms: well_known
                .id_token_signing_alg_values_supported
                .into_iter()
                .collect(),
            issuer,
            fetched_at: Instant::now(),
        });
    }

    async fn fetch_json(&self, url: &str) -> std::result::Result<serde_json::Value, String> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        response.json().await.map_err(|e| e.to_string())
    }

    /// `decode_token` — validates a bearer token and returns its claims.
    pub async fn decode_token(&self, token: &str) -> Result<serde_json::Value> {
        self.ensure_updated().await;

        let guard = self.discovered.read();
        let Some(state) = guard.as_ref() else {
            return Err(ApiError::unauthorized("Auth not initialized on server."));
        };

        // `if token.lower().startswith("bearer "): token = token[7:]`
        let token = if token.len() >= 7 && token[..7].eq_ignore_ascii_case("bearer ") {
            &token[7..]
        } else {
            token
        };

        let header =
            decode_header(token).map_err(|_| ApiError::unauthorized("Error decoding Token."))?;

        let algorithm_name = format!("{:?}", header.alg);
        if !state.supported_algorithms.contains(&algorithm_name) {
            return Err(ApiError::unauthorized(
                "Algorithm required for token decoding not found.",
            ));
        }
        let Some(kid) = header.kid.as_deref() else {
            return Err(ApiError::unauthorized(
                "Public key required for token decoding not found.",
            ));
        };
        let Some(key) = state.public_keys.get(kid) else {
            return Err(ApiError::unauthorized(
                "Public key required for token decoding not found.",
            ));
        };

        let mut validation = Validation::new(header.alg);
        validation.leeway = self.leeway;
        validation.set_issuer(&[state.issuer.as_str()]);
        match &self.audience {
            Some(audience) => validation.set_audience(&[audience.as_str()]),
            // `options = {"verify_aud": False}` when no audience is configured.
            None => validation.validate_aud = false,
        }

        let decoded = decode::<serde_json::Value>(token, key, &validation).map_err(|err| {
            use jsonwebtoken::errors::ErrorKind;
            match err.kind() {
                ErrorKind::ExpiredSignature => ApiError::unauthorized("Access Token expired."),
                ErrorKind::InvalidToken
                | ErrorKind::Base64(_)
                | ErrorKind::Json(_)
                | ErrorKind::Utf8(_) => ApiError::unauthorized("Error decoding Token."),
                // Any other validation failure: Python re-reads the token without
                // verifying the signature and reports a missing or mismatched audience
                // distinctly from a generic invalid token.
                _ => match audience_matches(token, self.audience.as_deref()) {
                    true => ApiError::unauthorized("Invalid Token."),
                    false => ApiError::unauthorized("API access denied."),
                },
            }
        })?;

        // RationAI only: an `acr` claim must equal the configured value.
        if let Some(expected) = &self.acr_values {
            let actual = decoded.claims.get("acr").and_then(|v| v.as_str());
            if actual != Some(expected.as_str()) {
                return Err(ApiError::unauthorized("Authentication requires valid ACR."));
            }
        }

        Ok(decoded.claims)
    }
}

/// Re-reads the token without verifying it, to tell "wrong audience" from "invalid".
fn audience_matches(token: &str, audience: Option<&str>) -> bool {
    let Some(audience) = audience else {
        return false;
    };
    let Some(claims) = decode_unverified_claims(token) else {
        return false;
    };
    match claims.get("aud") {
        Some(serde_json::Value::String(value)) => value == audience,
        // A JWT `aud` may be an array; Python's `!=` comparison against a string would be
        // false for a list, which lands on "API access denied." either way unless the list
        // contains the audience.
        Some(serde_json::Value::Array(values)) => {
            values.iter().any(|v| v.as_str() == Some(audience))
        }
        _ => false,
    }
}

fn decode_unverified_claims(token: &str) -> Option<serde_json::Value> {
    use base64::Engine;
    let payload = token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&decoded).ok()
}

/// Maps a configured algorithm name to the `jsonwebtoken` enum, for tests and for
/// validating configuration.
pub fn algorithm_from_name(name: &str) -> Option<Algorithm> {
    Some(match name {
        "HS256" => Algorithm::HS256,
        "HS384" => Algorithm::HS384,
        "HS512" => Algorithm::HS512,
        "ES256" => Algorithm::ES256,
        "ES384" => Algorithm::ES384,
        "RS256" => Algorithm::RS256,
        "RS384" => Algorithm::RS384,
        "RS512" => Algorithm::RS512,
        "PS256" => Algorithm::PS256,
        "PS384" => Algorithm::PS384,
        "PS512" => Algorithm::PS512,
        "EdDSA" => Algorithm::EdDSA,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth() -> Auth {
        Auth::new(
            "https://idp.example/auth/realms/EMPAIA/",
            300,
            Some("org.empaia.auth.wbs".into()),
            Some("https://public.example".into()),
            None,
            0,
        )
    }

    #[test]
    fn the_well_known_url_is_derived_from_the_idp_url_without_a_trailing_slash() {
        let auth = auth();
        assert_eq!(
            auth.well_known_url,
            "https://idp.example/auth/realms/EMPAIA/.well-known/openid-configuration"
        );
    }

    #[test]
    fn url_rewriting_replaces_the_advertised_prefix() {
        let auth = auth();
        assert_eq!(
            auth.rewrite_url("https://public.example/protocol/openid-connect/certs"),
            "https://idp.example/auth/realms/EMPAIA/protocol/openid-connect/certs"
        );
    }

    #[test]
    fn url_rewriting_is_skipped_when_the_prefix_does_not_match() {
        let auth = auth();
        let url = "https://elsewhere.example/certs";
        assert_eq!(auth.rewrite_url(url), url);
    }

    #[test]
    fn url_rewriting_is_disabled_when_unconfigured() {
        let auth = Auth::new("https://idp.example", 300, None, None, None, 0);
        let url = "https://public.example/certs";
        assert_eq!(auth.rewrite_url(url), url);

        let auth = Auth::new(
            "https://idp.example",
            300,
            None,
            Some(String::new()),
            None,
            0,
        );
        assert_eq!(auth.rewrite_url(url), url);
    }

    #[tokio::test]
    async fn an_undiscovered_provider_rejects_every_token() {
        // Nothing has been fetched, so `_last_time_updated is None` in Python terms.
        let auth = Auth::new("http://127.0.0.1:1/unreachable", 300, None, None, None, 0);
        let err = match auth.decode_token("whatever").await {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::UNAUTHORIZED);
        assert_eq!(err.detail_text(), "Auth not initialized on server.");
    }

    #[test]
    fn unverified_claims_are_readable_for_the_audience_fallback() {
        // header.payload.signature with payload {"aud":"a"}
        let token = "eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJhIn0.sig";
        let claims = decode_unverified_claims(token).unwrap();
        assert_eq!(claims["aud"], "a");
        assert!(audience_matches(token, Some("a")));
        assert!(!audience_matches(token, Some("b")));
        assert!(!audience_matches(token, None));
    }

    #[test]
    fn an_audience_array_is_matched_by_membership() {
        // payload {"aud":["a","b"]}
        let token = "eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOlsiYSIsImIiXX0.sig";
        assert!(audience_matches(token, Some("b")));
        assert!(!audience_matches(token, Some("c")));
    }

    #[test]
    fn algorithm_names_round_trip() {
        assert_eq!(algorithm_from_name("RS256"), Some(Algorithm::RS256));
        assert_eq!(algorithm_from_name("ES384"), Some(Algorithm::ES384));
        assert_eq!(algorithm_from_name("nonsense"), None);
        // The name used for the supported-algorithms check comes from Debug, which must
        // match the strings a provider advertises.
        assert_eq!(format!("{:?}", Algorithm::RS256), "RS256");
    }
}
