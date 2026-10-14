//! Shared OAuth plumbing, ported from `wsi_service/rationai_receiver_auth/aaa_oauth.py`.
//!
//! Settings, the user-info fetch, and the LRU-with-timeout cache that keeps the service
//! from calling the provider's userinfo endpoint on every tile request.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Deserialize;

use wsi_core::error::{ApiError, Result};
use wsi_core::settings::EnvSource;

/// `OAuthSettings` — read from the `WS_` prefix, like the rest of the service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSettings {
    pub idp_url: String,
    pub audience: String,
    pub refresh_interval: u64,
    pub openapi_token_url: String,
    pub openapi_auth_url: String,
    pub rewrite_url_in_wellknown: String,
    pub check_acr_values: Option<String>,

    // Authorization data caching
    pub user_cache_timeout: u64,
    pub user_cache_size: usize,
    /// e.g. `https://login.bbmri-eric.eu/oidc/userinfo`. Empty means the decoded token is
    /// used directly and no userinfo call is made.
    pub user_info_endpoint: String,

    /// `WSAuthSettings.lsaai_regex_prefix` — only the group-linking integration uses it.
    pub lsaai_regex_prefix: String,
}

impl Default for OAuthSettings {
    fn default() -> Self {
        Self {
            idp_url: String::new(),
            audience: String::new(),
            refresh_interval: 300,
            openapi_token_url: String::new(),
            openapi_auth_url: String::new(),
            rewrite_url_in_wellknown: String::new(),
            check_acr_values: None,
            user_cache_timeout: 1800,
            user_cache_size: 999,
            user_info_endpoint: String::new(),
            // Accepts any `urn:geant` group under any virtual organisation.
            lsaai_regex_prefix: r".*?:group:[^:]+".into(),
        }
    }
}

impl OAuthSettings {
    pub fn load() -> Self {
        Self::from_source(&EnvSource::load(".env"))
    }

    pub fn from_source(src: &EnvSource) -> Self {
        let mut settings = Self::default();
        let string = |name: &str| src.get_raw(name);

        if let Some(value) = string("idp_url") {
            settings.idp_url = value;
        }
        if let Some(value) = string("audience") {
            settings.audience = value;
        }
        if let Some(value) = string("refresh_interval").and_then(|v| v.trim().parse().ok()) {
            settings.refresh_interval = value;
        }
        if let Some(value) = string("openapi_token_url") {
            settings.openapi_token_url = value;
        }
        if let Some(value) = string("openapi_auth_url") {
            settings.openapi_auth_url = value;
        }
        if let Some(value) = string("rewrite_url_in_wellknown") {
            settings.rewrite_url_in_wellknown = value;
        }
        settings.check_acr_values = string("check_acr_values").filter(|v| !v.is_empty());
        if let Some(value) = string("user_cache_timeout").and_then(|v| v.trim().parse().ok()) {
            settings.user_cache_timeout = value;
        }
        if let Some(value) = string("user_cache_size").and_then(|v| v.trim().parse().ok()) {
            settings.user_cache_size = value;
        }
        if let Some(value) = string("user_info_endpoint") {
            settings.user_info_endpoint = value;
        }
        // The Python setting is read from `OAUTH_USER_INFO_ENDPOINT` in some deployments.
        if settings.user_info_endpoint.is_empty() {
            if let Ok(value) = std::env::var("OAUTH_USER_INFO_ENDPOINT") {
                settings.user_info_endpoint = value;
            }
        }
        if let Some(value) = string("lsaai_regex_prefix") {
            settings.lsaai_regex_prefix = value;
        }
        settings
    }
}

/// `LRUTimeoutCache` — `rationai_receiver_auth/lru_timeout_cache.py`.
///
/// Least-recently-used eviction plus a per-entry expiry that is *reset on every read*, so
/// an actively used entry never ages out. Python arms a timer per entry; checking the
/// deadline on access is equivalent and needs no background task.
pub struct LruTimeoutCache<V> {
    entries: Mutex<Vec<(String, Entry<V>)>>,
    max_size: usize,
    timeout: Duration,
}

struct Entry<V> {
    value: V,
    expires_at: Instant,
}

impl<V: Clone> LruTimeoutCache<V> {
    pub fn new(max_size: usize, timeout_seconds: u64) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            max_size,
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    pub fn get_item(&self, key: &str) -> Option<V> {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        entries.retain(|(_, entry)| entry.expires_at > now);

        let position = entries.iter().position(|(k, _)| k == key)?;
        let (key, mut entry) = entries.remove(position);
        entry.expires_at = now + self.timeout;
        let value = entry.value.clone();
        // Most recently used goes last.
        entries.push((key, entry));
        Some(value)
    }

    pub fn put_item(&self, key: &str, value: V) {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        entries.retain(|(_, entry)| entry.expires_at > now);
        entries.retain(|(k, _)| k != key);

        entries.push((
            key.to_owned(),
            Entry {
                value,
                expires_at: now + self.timeout,
            },
        ));

        while entries.len() > self.max_size {
            entries.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        entries.retain(|(_, entry)| entry.expires_at > now);
        entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The subset of the decoded token the integrations rely on.
#[derive(Debug, Deserialize)]
pub struct TokenClaims {
    pub sub: Option<String>,
}

/// `fetch_user_info` — a plain GET with the caller's bearer token forwarded.
pub async fn fetch_user_info(
    client: &reqwest::Client,
    endpoint: &str,
    raw_token: &str,
    extra_headers: &HashMap<String, String>,
) -> Result<String> {
    let authorization = if raw_token.starts_with("Bearer") {
        raw_token.to_owned()
    } else {
        format!("Bearer {raw_token}")
    };

    let mut request = client.get(endpoint).header("Authorization", authorization);
    for (name, value) in extra_headers {
        request = request.header(name.as_str(), value.as_str());
    }

    let response = request
        .send()
        .await
        .map_err(|e| ApiError::unauthorized(format!("Could not fetch user info: {e}")))?;
    response
        .text()
        .await
        .map_err(|e| ApiError::unauthorized(format!("Could not read user info: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_python_settings_class() {
        let settings = OAuthSettings::default();
        assert_eq!(settings.refresh_interval, 300);
        assert_eq!(settings.user_cache_timeout, 1800);
        assert_eq!(settings.user_cache_size, 999);
        assert!(settings.user_info_endpoint.is_empty());
        assert!(settings.check_acr_values.is_none());
        assert_eq!(settings.lsaai_regex_prefix, r".*?:group:[^:]+");
    }

    #[test]
    fn cache_returns_stored_values() {
        let cache: LruTimeoutCache<String> = LruTimeoutCache::new(2, 60);
        assert!(cache.get_item("a").is_none());
        cache.put_item("a", "one".into());
        assert_eq!(cache.get_item("a").as_deref(), Some("one"));
    }

    #[test]
    fn cache_evicts_the_least_recently_used_entry() {
        let cache: LruTimeoutCache<String> = LruTimeoutCache::new(2, 60);
        cache.put_item("a", "one".into());
        cache.put_item("b", "two".into());
        // Touching "a" makes "b" the least recently used.
        assert!(cache.get_item("a").is_some());
        cache.put_item("c", "three".into());

        assert!(cache.get_item("a").is_some());
        assert!(cache.get_item("b").is_none());
        assert!(cache.get_item("c").is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_entries_expire() {
        let cache: LruTimeoutCache<String> = LruTimeoutCache::new(4, 0);
        cache.put_item("a", "one".into());
        // A zero timeout means the entry is already stale on the next read.
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get_item("a").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn replacing_a_key_does_not_duplicate_it() {
        let cache: LruTimeoutCache<String> = LruTimeoutCache::new(4, 60);
        cache.put_item("a", "one".into());
        cache.put_item("a", "two".into());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_item("a").as_deref(), Some("two"));
    }
}
