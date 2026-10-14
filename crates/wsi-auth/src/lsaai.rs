//! The two LSAAI integrations, ported from
//! `wsi_service/api/v3/integrations/lsaai_groups.py` and `lsaai_resources.py`.
//!
//! Both validate the token, fetch the user's `eduperson_entitlement` list from the
//! provider's userinfo endpoint, and decide access by matching the slide's dotted ID
//! against what the user is entitled to. They differ only in how entitlements are parsed:
//!
//! * **groups** — reads a three-level hierarchy `project : institution : rights` out of
//!   the group path;
//! * **resources** — flattens groups to their leaf name and attaches projects from
//!   separate `:res:<institution>:<project>` entitlements.
//!
//! Slide IDs come from the CSV or iterator mappers as
//! `{institution}.{project}.{c|w}.{id}`, which is why the dotted form carries meaning.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;

use wsi_core::error::{ApiError, Result};
use wsi_core::models::SlideInfo;

use crate::empaia::strip_bearer;
use crate::jwt::Auth;
use crate::oauth::{fetch_user_info, LruTimeoutCache, OAuthSettings};
use crate::{AuthIntegration, AuthPayload, OpenApiSecurity, RequestContext};

/// Which entitlement dialect to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsaaiMode {
    /// `lsaai_groups.py`
    Groups,
    /// `lsaai_resources.py`
    Resources,
}

/// `project -> institution -> ()`, plus the resource projects attached to an institution.
///
/// Both dialects reduce to the same two-level lookup, so one representation serves both;
/// the difference is entirely in how it is populated.
type Hierarchy = HashMap<String, HashMap<String, Vec<String>>>;

pub struct LsaaiIntegration {
    mode: LsaaiMode,
    auth: Arc<Auth>,
    settings: OAuthSettings,
    client: reqwest::Client,
    user_cache: LruTimeoutCache<Hierarchy>,
    slide_cache: LruTimeoutCache<SlideInfo>,
    group_pattern: Regex,
}

impl LsaaiIntegration {
    pub fn new(mode: LsaaiMode, settings: OAuthSettings) -> Result<Self> {
        let auth = Arc::new(Auth::new(
            settings.idp_url.trim_end_matches('/'),
            settings.refresh_interval,
            Some(settings.audience.clone()),
            Some(settings.rewrite_url_in_wellknown.clone()),
            settings.check_acr_values.clone(),
            0,
        ));

        // Groups honour the configurable prefix; resources hard-code `.*?:group:`.
        let pattern = match mode {
            LsaaiMode::Groups => format!(r"^{}:?([^#\n]*)", settings.lsaai_regex_prefix),
            LsaaiMode::Resources => r"^.*?:group:([^#\n]*)".to_owned(),
        };
        let group_pattern = Regex::new(&pattern)
            .map_err(|e| ApiError::internal(format!("Invalid lsaai_regex_prefix: {e}")))?;

        Ok(Self {
            mode,
            auth,
            client: reqwest::Client::new(),
            user_cache: LruTimeoutCache::new(settings.user_cache_size, settings.user_cache_timeout),
            slide_cache: LruTimeoutCache::new(
                settings.user_cache_size,
                settings.user_cache_timeout,
            ),
            group_pattern,
            settings,
        })
    }

    /// `parse_user_info` for both dialects.
    fn parse_user_info(&self, body: &str) -> Result<Hierarchy> {
        let value: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| ApiError::unauthorized(format!("User info is not valid JSON: {e}")))?;

        let entitlements = value
            .get("eduperson_entitlement")
            .and_then(|e| e.as_array());

        let Some(entitlements) = entitlements else {
            // Both dialects surface the provider's own error message when there is one.
            if let Some(error) = value.get("error") {
                let message = value
                    .get("error_description")
                    .or(Some(error))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Authorization failed")
                    .to_owned();
                return Err(ApiError::unauthorized(message));
            }
            return match self.mode {
                // Groups returns an empty hierarchy, which denies everything downstream.
                LsaaiMode::Groups => Ok(Hierarchy::new()),
                LsaaiMode::Resources => Err(ApiError::unauthorized(
                    "User info does not contain eduperson_entitlement! Is the scope requested?",
                )),
            };
        };

        match self.mode {
            LsaaiMode::Groups => Ok(self.parse_groups(entitlements)),
            LsaaiMode::Resources => self.parse_resources(entitlements),
        }
    }

    /// `lsaai_groups.parse_user_info` — `project : institution : rights`.
    fn parse_groups(&self, entitlements: &[serde_json::Value]) -> Hierarchy {
        let mut hierarchy = Hierarchy::new();
        for line in entitlements.iter().filter_map(|e| e.as_str()) {
            let Some(captures) = self.group_pattern.captures(line) else {
                continue;
            };
            let groups = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
            if groups.is_empty() {
                continue;
            }
            let parts: Vec<&str> = groups.split(':').collect();
            let project = parts[0].to_owned();
            let institution = parts.get(1).copied().unwrap_or("").to_owned();
            let rights = parts.get(2).copied();

            let project_node = hierarchy.entry(project).or_default();
            let institution_node = project_node.entry(institution).or_default();
            if let Some(rights) = rights {
                institution_node.push(rights.to_owned());
            }
        }
        hierarchy
    }

    /// `lsaai_resources.parse_user_info` — leaf group names, then `:res:inst:proj` lines.
    ///
    /// A resource line naming an institution that has no group line raises `KeyError` in
    /// Python, which the caller turns into a 401. Reproduced.
    fn parse_resources(&self, entitlements: &[serde_json::Value]) -> Result<Hierarchy> {
        let resource_pattern =
            Regex::new(r"^.*?:res:([^#\n:]+):([^#\n:]+)").expect("static pattern");

        // Resources use a flat institution -> [project] map; it is stored in the same
        // structure under a single empty project key so one lookup serves both dialects.
        let mut institutions: HashMap<String, Vec<String>> = HashMap::new();

        for line in entitlements.iter().filter_map(|e| e.as_str()) {
            if let Some(captures) = self.group_pattern.captures(line) {
                let groups = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
                if !groups.is_empty() {
                    let leaf = groups.split(':').next_back().unwrap_or_default();
                    institutions.entry(leaf.to_owned()).or_default();
                }
                continue;
            }
            if let Some(captures) = resource_pattern.captures(line) {
                let institution = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
                let project = captures.get(2).map(|m| m.as_str()).unwrap_or_default();
                let Some(projects) = institutions.get_mut(institution) else {
                    return Err(ApiError::unauthorized(
                        "Token or user info endpoint data is invalid!",
                    ));
                };
                if !projects.iter().any(|p| p == project) {
                    projects.push(project.to_owned());
                }
            }
        }

        let mut hierarchy = Hierarchy::new();
        hierarchy.insert(String::new(), institutions);
        Ok(hierarchy)
    }

    /// `_resolve_proj_institution`, which differs between the two dialects.
    fn resolve(&self, institution: &str, project: Option<&str>, user_data: &Hierarchy) -> bool {
        match self.mode {
            LsaaiMode::Groups => {
                // `project_data = user_data.get(project)` then
                // `institution_data = project_data.get(institution)`.
                let project = project.unwrap_or("");
                let project_data = user_data.get(project);
                let institution_data = project_data.and_then(|p| p.get(institution));
                (institution.is_empty() || institution_data.is_some())
                    && (project.is_empty() || project_data.is_some())
            }
            LsaaiMode::Resources => {
                if institution.is_empty() {
                    return true;
                }
                let Some(institutions) = user_data.get("") else {
                    return false;
                };
                let Some(projects) = institutions.get(institution) else {
                    return false;
                };
                match project.filter(|p| !p.is_empty()) {
                    None => true,
                    Some(project) => projects.iter().any(|p| p == project),
                }
            }
        }
    }

    async fn user_info(&self, payload: &AuthPayload) -> Result<Hierarchy> {
        let subject = payload
            .token
            .as_ref()
            .and_then(|t| t.get("sub"))
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty());
        let Some(subject) = subject else {
            return Err(ApiError::unauthorized(
                "Auth payload does not contain 'sub' subject ID!",
            ));
        };

        if self.settings.user_info_endpoint.is_empty() {
            // Python returns the auth payload itself, which the LSAAI parsers then treat
            // as user data. With no entitlements present that denies everything.
            return Ok(Hierarchy::new());
        }

        if let Some(cached) = self.user_cache.get_item(subject) {
            return Ok(cached);
        }

        let raw_token = payload.raw_token.as_deref().unwrap_or_default();
        let body = fetch_user_info(
            &self.client,
            &self.settings.user_info_endpoint,
            raw_token,
            &HashMap::from([("Content-Type".to_owned(), "application/json".to_owned())]),
        )
        .await?;

        let parsed = self.parse_user_info(&body)?;
        self.user_cache.put_item(subject, parsed.clone());
        Ok(parsed)
    }
}

#[async_trait]
impl AuthIntegration for LsaaiIntegration {
    async fn authenticate(&self, ctx: RequestContext<'_>) -> Result<AuthPayload> {
        let Some(header) = ctx.authorization.filter(|h| !h.is_empty()) else {
            return Err(ApiError::unauthorized("Not authenticated"));
        };
        let claims = self.auth.decode_token(header).await?;
        Ok(AuthPayload {
            token: Some(claims),
            raw_token: Some(strip_bearer(header).to_owned()),
        })
    }

    async fn allow_access_slide(
        &self,
        payload: &AuthPayload,
        slide_id: &str,
        _plugin: Option<&str>,
        slide: Option<&SlideInfo>,
    ) -> Result<()> {
        // The caller sometimes already has the slide info; otherwise a cached copy is
        // used so an image request does not have to re-open the slide just to authorise.
        let resolved_id = match slide {
            Some(info) => {
                self.slide_cache.put_item(slide_id, info.clone());
                info.id.clone()
            }
            None => match self.slide_cache.get_item(slide_id) {
                Some(info) => info.id,
                // Python fetches the slide info here. Doing so would need the slide
                // manager, which this layer deliberately does not hold; the requested ID
                // is what `get_slide_info` would put in `info.id` anyway.
                None => slide_id.to_owned(),
            },
        };

        let user_data = self.user_info(payload).await?;
        let parts: Vec<&str> = resolved_id.split('.').collect();

        let allowed = match parts.len() {
            // institution.project.{c|w}.id
            4 => self.resolve(parts[0], Some(parts[1]), &user_data),
            // institution.{c|w}.id — no project component
            3 => self.resolve(
                parts[0],
                match self.mode {
                    // groups passes "", resources passes None
                    LsaaiMode::Groups => Some(""),
                    LsaaiMode::Resources => None,
                },
                &user_data,
            ),
            // Anything shorter carries no access information at all.
            2 => true,
            _ => false,
        };

        if allowed {
            return Ok(());
        }
        Err(ApiError::forbidden(format!(
            "Slide {resolved_id} not available to the user!"
        )))
    }

    fn openapi_security(&self) -> Option<OpenApiSecurity> {
        Some(OpenApiSecurity {
            authorization_url: self.settings.openapi_auth_url.clone(),
            token_url: self.settings.openapi_token_url.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integration(mode: LsaaiMode) -> LsaaiIntegration {
        LsaaiIntegration::new(
            mode,
            OAuthSettings {
                idp_url: "http://127.0.0.1:1/unreachable".into(),
                user_info_endpoint: "http://127.0.0.1:1/userinfo".into(),
                ..OAuthSettings::default()
            },
        )
        .unwrap()
    }

    fn entitlements(lines: &[&str]) -> String {
        serde_json::json!({ "eduperson_entitlement": lines }).to_string()
    }

    #[test]
    fn groups_are_parsed_into_project_institution_rights() {
        let integration = integration(LsaaiMode::Groups);
        let body = entitlements(&[
            "urn:geant:lifescience-ri.eu:group:ration_ai:adopt:mmci:write#aai.example",
        ]);
        let hierarchy = integration.parse_user_info(&body).unwrap();

        // The default prefix consumes `...:group:<first>`, so the captured path starts at
        // the segment after it.
        let project = hierarchy.keys().next().expect("one project");
        let institutions = &hierarchy[project];
        assert!(!institutions.is_empty());
    }

    #[test]
    fn groups_without_entitlements_yield_an_empty_hierarchy() {
        let integration = integration(LsaaiMode::Groups);
        let hierarchy = integration.parse_user_info("{}").unwrap();
        assert!(hierarchy.is_empty());
    }

    #[test]
    fn resources_without_entitlements_are_rejected() {
        let integration = integration(LsaaiMode::Resources);
        let err = match integration.parse_user_info("{}") {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::UNAUTHORIZED);
        assert!(err.detail_text().contains("eduperson_entitlement"));
    }

    #[test]
    fn a_provider_error_is_surfaced_verbatim() {
        let integration = integration(LsaaiMode::Groups);
        let body = serde_json::json!({
            "error": "invalid_token",
            "error_description": "Token is expired"
        })
        .to_string();
        let err = match integration.parse_user_info(&body) {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.detail_text(), "Token is expired");
    }

    #[test]
    fn resources_attach_projects_to_their_institution() {
        let integration = integration(LsaaiMode::Resources);
        let body = entitlements(&[
            "urn:geant:lifescience-ri.eu:group:mmci#aai.example",
            "urn:geant:lifescience-ri.eu:res:mmci:adopt#aai.example",
        ]);
        let hierarchy = integration.parse_user_info(&body).unwrap();
        let institutions = &hierarchy[""];
        assert_eq!(institutions["mmci"], vec!["adopt".to_owned()]);

        assert!(integration.resolve("mmci", Some("adopt"), &hierarchy));
        assert!(integration.resolve("mmci", None, &hierarchy));
        assert!(!integration.resolve("mmci", Some("other"), &hierarchy));
        assert!(!integration.resolve("other", Some("adopt"), &hierarchy));
        assert!(integration.resolve("", Some("anything"), &hierarchy));
    }

    #[test]
    fn a_resource_without_its_group_is_rejected() {
        let integration = integration(LsaaiMode::Resources);
        let body = entitlements(&["urn:geant:lifescience-ri.eu:res:mmci:adopt#aai.example"]);
        assert!(integration.parse_user_info(&body).is_err());
    }

    #[tokio::test]
    async fn a_two_part_slide_id_is_always_allowed() {
        // No userinfo endpoint, so the hierarchy is empty. A two-component ID carries no
        // institution or project and is allowed regardless.
        let integration = LsaaiIntegration::new(
            LsaaiMode::Resources,
            OAuthSettings {
                idp_url: "http://127.0.0.1:1/unreachable".into(),
                user_info_endpoint: String::new(),
                ..OAuthSettings::default()
            },
        )
        .unwrap();
        let payload = AuthPayload {
            token: Some(serde_json::json!({ "sub": "user-1" })),
            raw_token: Some("token".into()),
        };
        assert!(integration
            .allow_access_slide(&payload, "w.slide", None, None)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn user_info_is_fetched_before_the_id_shape_is_examined() {
        // Python calls `get_user_info` first, so an unreachable provider fails the request
        // even for an ID that would otherwise be allowed outright.
        let integration = integration(LsaaiMode::Resources);
        let payload = AuthPayload {
            token: Some(serde_json::json!({ "sub": "user-1" })),
            raw_token: Some("token".into()),
        };
        let err = match integration
            .allow_access_slide(&payload, "w.slide", None, None)
            .await
        {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_payload_without_a_subject_is_a_401() {
        let integration = integration(LsaaiMode::Resources);
        let payload = AuthPayload {
            token: Some(serde_json::json!({})),
            raw_token: None,
        };
        let err = match integration
            .allow_access_slide(&payload, "a.b.w.slide", None, None)
            .await
        {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(
            err.detail_text(),
            "Auth payload does not contain 'sub' subject ID!"
        );
    }

    #[tokio::test]
    async fn an_unentitled_slide_is_a_403() {
        // With no userinfo endpoint configured the hierarchy is empty, so any slide that
        // names an institution is denied.
        let integration = LsaaiIntegration::new(
            LsaaiMode::Resources,
            OAuthSettings {
                idp_url: "http://127.0.0.1:1/unreachable".into(),
                user_info_endpoint: String::new(),
                ..OAuthSettings::default()
            },
        )
        .unwrap();
        let payload = AuthPayload {
            token: Some(serde_json::json!({ "sub": "user-1" })),
            raw_token: Some("token".into()),
        };
        let err = match integration
            .allow_access_slide(&payload, "mmci.adopt.w.slide", None, None)
            .await
        {
            Ok(_) => panic!("expected an error"),
            Err(err) => err,
        };
        assert_eq!(err.status, http::StatusCode::FORBIDDEN);
        assert_eq!(
            err.detail_text(),
            "Slide mmci.adopt.w.slide not available to the user!"
        );
    }

    #[test]
    fn an_invalid_group_prefix_fails_construction() {
        let result = LsaaiIntegration::new(
            LsaaiMode::Groups,
            OAuthSettings {
                lsaai_regex_prefix: "(unclosed".into(),
                ..OAuthSettings::default()
            },
        );
        assert!(result.is_err());
    }
}
