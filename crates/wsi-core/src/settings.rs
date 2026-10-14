//! Settings, mirroring `wsi_service/settings.py`.
//!
//! Same `WS_` env prefix, same `.env` file, same defaults, same coercions. Loading is
//! hand-rolled rather than delegated to a config crate because pydantic-settings has
//! specific behaviours worth reproducing exactly:
//!
//! * booleans accept `true/false/1/0/yes/no/on/off` case-insensitively;
//! * complex types (here `Set[str]`) are parsed as **JSON** when they come from the
//!   environment, which is why `.env` carries `WS_CORS_ALLOW_ORIGINS=["*"]`;
//! * unknown keys are ignored (`extra="ignore"`);
//! * the process environment wins over the `.env` file.

use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};

/// Package version reported by `/alive` and the OpenAPI document.
/// Python reads it from installed package metadata; here it tracks the workspace version,
/// which is pinned to the Python `pyproject.toml` version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub title: String,
    pub description: String,
    pub version: String,

    // Auth / HTTP client
    pub connection_chunk_size: usize,
    pub connection_limit_per_host: usize,
    pub http_client_timeout: u64,
    pub request_timeout: u64,
    pub api_v3_integration: String,
    pub idp_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub organization_id: String,

    pub disable_openapi: bool,
    pub cors_allow_credentials: bool,
    pub cors_allow_origins: BTreeSet<String>,
    pub debug: bool,
    pub data_dir: String,
    pub mapper_address: String,
    /// `module:Class` descriptor of the local mapper, empty when disabled.
    pub local_mode: String,
    pub enable_local_routes: bool,
    pub enable_viewer_routes: bool,
    pub inactive_histo_image_timeout_seconds: u64,
    pub image_handle_cache_size: usize,
    pub max_returned_region_size: i64,
    pub max_thumbnail_size: u32,
    pub root_path: String,

    /// Default colour used to pad regions that reach outside the image extent.
    pub padding_color: (u8, u8, u8),
    pub apply_padding: bool,
    pub get_tile_apply_padding: bool,

    /// Not present in Python. When true, a 3-digit hex `padding_color` is expanded to 6
    /// digits instead of raising; see `docs/divergences.md`. Defaults to `false` so the
    /// service reproduces Python behaviour bug-for-bug unless opted in.
    pub lenient_padding_color: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            title: "WSI Service".into(),
            description: "EMPAIA WSI Service to stream whole slide images".into(),
            version: VERSION.into(),

            connection_chunk_size: 1_024_000,
            connection_limit_per_host: 100,
            http_client_timeout: 300,
            request_timeout: 300,
            api_v3_integration: String::new(),
            idp_url: String::new(),
            client_id: String::new(),
            client_secret: String::new(),
            organization_id: String::new(),

            disable_openapi: false,
            cors_allow_credentials: false,
            cors_allow_origins: BTreeSet::new(),
            debug: false,
            data_dir: "/data".into(),
            mapper_address: "http://localhost:8080/v3/slides/{slide_id}/storage".into(),
            local_mode: String::new(),
            enable_local_routes: true,
            enable_viewer_routes: true,
            inactive_histo_image_timeout_seconds: 600,
            image_handle_cache_size: 50,
            max_returned_region_size: 25_000_000,
            max_thumbnail_size: 500,
            root_path: String::new(),

            padding_color: (255, 255, 255),
            apply_padding: true,
            get_tile_apply_padding: false,

            lenient_padding_color: false,
        }
    }
}

/// A source of prefixed settings: the process environment layered over an optional `.env`
/// file.
///
/// The prefix is a parameter because not every settings class in the Python service uses
/// `ws_` — the CSV mapper uses `csws_` and the iterator mapper uses `wsit_`.
pub struct EnvSource {
    dotenv: std::collections::HashMap<String, String>,
    prefix: String,
}

impl EnvSource {
    /// Loads `.env` from `path` if it exists, for the default `ws_` prefix. A missing file
    /// is not an error — pydantic treats `env_file` as optional too.
    pub fn load(path: impl AsRef<Path>) -> Self {
        Self::load_prefixed(path, "ws_")
    }

    pub fn load_prefixed(path: impl AsRef<Path>, prefix: &str) -> Self {
        let mut dotenv = std::collections::HashMap::new();
        if let Ok(iter) = dotenvy::from_path_iter(path.as_ref()) {
            for item in iter.flatten() {
                dotenv.insert(item.0, item.1);
            }
        }
        Self {
            dotenv,
            prefix: prefix.to_owned(),
        }
    }

    pub fn empty() -> Self {
        Self {
            dotenv: std::collections::HashMap::new(),
            prefix: "ws_".to_owned(),
        }
    }

    /// Process environment first, then `.env`. Lookup is case-insensitive on the suffix,
    /// matching pydantic-settings' default `case_sensitive=False`.
    pub fn get_raw(&self, suffix: &str) -> Option<String> {
        let upper = format!(
            "{}{}",
            self.prefix.to_ascii_uppercase(),
            suffix.to_ascii_uppercase()
        );
        if let Ok(v) = env::var(&upper) {
            return Some(v);
        }
        let lower = format!("{}{}", self.prefix.to_ascii_lowercase(), suffix);
        if let Ok(v) = env::var(&lower) {
            return Some(v);
        }
        self.dotenv
            .get(&upper)
            .or_else(|| self.dotenv.get(&lower))
            .cloned()
    }

    fn get(&self, suffix: &str) -> Option<String> {
        self.get_raw(suffix)
    }
}

/// pydantic's boolean coercion. Anything else is treated as unset rather than as an error,
/// so a malformed value cannot take the service down at startup.
fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

/// `Set[str]` from the environment is JSON, per pydantic-settings. A bare comma-separated
/// string is also accepted because operators write those by hand.
fn parse_string_set(raw: &str) -> BTreeSet<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return BTreeSet::new();
    }
    if let Ok(serde_json::Value::Array(items)) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        return items
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .collect();
    }
    trimmed
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

/// `settings.py:48` — absolutise, expand `~`, and normalise separators to `/`.
///
/// The normalisation matters: `local_mode_abs_file_path_to_relative` makes storage
/// addresses relative by stripping this string as a prefix, so a backslash here would
/// silently stop matching on Windows.
pub fn resolve_data_dir(value: &str) -> String {
    let expanded = expand_user(value);
    let absolute = if expanded.is_absolute() {
        normalise_dots(&expanded)
    } else {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        normalise_dots(&cwd.join(expanded))
    };
    absolute.to_string_lossy().replace('\\', "/")
}

fn expand_user(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix('~') {
        let rest = rest.trim_start_matches(['/', '\\']);
        if let Some(home) = home_dir() {
            return if rest.is_empty() {
                home
            } else {
                home.join(rest)
            };
        }
    }
    PathBuf::from(value)
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Lexical `.`/`..` removal. `std::fs::canonicalize` is deliberately not used: the data
/// directory need not exist yet at startup (Python only logs a warning), and canonicalize
/// would fail — and on Windows would also produce a `\\?\` prefix that breaks the
/// string-prefix stripping described above.
fn normalise_dots(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

impl Settings {
    /// Loads settings from the process environment layered over `.env` in the working
    /// directory, exactly like `Settings()` in Python.
    pub fn load() -> Self {
        Self::from_source(&EnvSource::load(".env"))
    }

    pub fn from_source(src: &EnvSource) -> Self {
        let mut s = Settings::default();

        macro_rules! string_field {
            ($field:ident) => {
                if let Some(v) = src.get(stringify!($field)) {
                    s.$field = v;
                }
            };
        }
        macro_rules! parsed_field {
            ($field:ident) => {
                if let Some(v) = src.get(stringify!($field)) {
                    if let Ok(parsed) = v.trim().parse() {
                        s.$field = parsed;
                    }
                }
            };
        }
        macro_rules! bool_field {
            ($field:ident) => {
                if let Some(v) = src.get(stringify!($field)) {
                    if let Some(parsed) = parse_bool(&v) {
                        s.$field = parsed;
                    }
                }
            };
        }

        string_field!(title);
        string_field!(description);
        string_field!(version);

        parsed_field!(connection_chunk_size);
        parsed_field!(connection_limit_per_host);
        parsed_field!(http_client_timeout);
        parsed_field!(request_timeout);
        string_field!(api_v3_integration);
        string_field!(idp_url);
        string_field!(client_id);
        string_field!(client_secret);
        string_field!(organization_id);

        bool_field!(disable_openapi);
        bool_field!(cors_allow_credentials);
        if let Some(v) = src.get("cors_allow_origins") {
            s.cors_allow_origins = parse_string_set(&v);
        }
        bool_field!(debug);
        string_field!(mapper_address);
        string_field!(local_mode);
        bool_field!(enable_local_routes);
        bool_field!(enable_viewer_routes);
        parsed_field!(inactive_histo_image_timeout_seconds);
        parsed_field!(image_handle_cache_size);
        parsed_field!(max_returned_region_size);
        parsed_field!(max_thumbnail_size);
        string_field!(root_path);

        if let Some(v) = src.get("padding_color") {
            if let Some(rgb) = parse_rgb_tuple(&v) {
                s.padding_color = rgb;
            }
        }
        bool_field!(apply_padding);
        bool_field!(get_tile_apply_padding);
        bool_field!(lenient_padding_color);

        // data_dir last: the field validator runs on whatever value was resolved.
        let raw_data_dir = src.get("data_dir").unwrap_or_else(|| s.data_dir.clone());
        s.data_dir = resolve_data_dir(&raw_data_dir);

        s
    }

    /// Plugin priority override, `WS_PLUGIN_PRIORITY_<UPPERCASE_NAME>`.
    /// `wsi_service/plugins.py:85` reads this straight from `os.environ`, so it is *not*
    /// sourced from `.env` — reproduce that.
    pub fn plugin_priority_override(plugin_name: &str) -> Option<i64> {
        let key = format!("WS_PLUGIN_PRIORITY_{}", plugin_name.to_ascii_uppercase());
        env::var(key).ok()?.trim().parse().ok()
    }
}

/// Accepts `[255, 255, 255]`, `255,255,255` and `(255, 255, 255)`.
fn parse_rgb_tuple(raw: &str) -> Option<(u8, u8, u8)> {
    let cleaned: String = raw
        .chars()
        .filter(|c| !matches!(c, '[' | ']' | '(' | ')' | ' '))
        .collect();
    let mut parts = cleaned.split(',');
    let r = parts.next()?.parse().ok()?;
    let g = parts.next()?.parse().ok()?;
    let b = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let s = Settings::default();
        assert_eq!(s.max_returned_region_size, 25_000_000);
        assert_eq!(s.max_thumbnail_size, 500);
        assert_eq!(s.image_handle_cache_size, 50);
        assert_eq!(s.inactive_histo_image_timeout_seconds, 600);
        assert_eq!(s.padding_color, (255, 255, 255));
        assert!(s.apply_padding);
        assert!(!s.get_tile_apply_padding);
        assert!(s.enable_local_routes);
        assert!(s.enable_viewer_routes);
    }

    #[test]
    fn bool_coercion_matches_pydantic() {
        for truthy in ["1", "true", "TRUE", "yes", "on", "t", "y"] {
            assert_eq!(parse_bool(truthy), Some(true), "{truthy}");
        }
        for falsy in ["0", "false", "FALSE", "no", "off", "f", "n"] {
            assert_eq!(parse_bool(falsy), Some(false), "{falsy}");
        }
        assert_eq!(parse_bool("maybe"), None);
    }

    #[test]
    fn cors_origins_parse_as_json_list() {
        let set = parse_string_set(r#"["*"]"#);
        assert_eq!(set.len(), 1);
        assert!(set.contains("*"));

        let set = parse_string_set(r#"["http://a", "http://b"]"#);
        assert_eq!(set.len(), 2);

        // hand-written comma form
        let set = parse_string_set("http://a, http://b");
        assert_eq!(set.len(), 2);

        assert!(parse_string_set("").is_empty());
    }

    #[test]
    fn data_dir_is_absolute_and_slash_normalised() {
        let resolved = resolve_data_dir("/data");
        assert!(!resolved.contains('\\'));
        assert!(resolved.ends_with("/data") || resolved.ends_with(":/data"));

        // relative input resolves against the working directory
        let resolved = resolve_data_dir("./some/dir");
        assert!(!resolved.contains("/./"));
        assert!(resolved.ends_with("/some/dir"));
    }

    #[test]
    fn rgb_tuple_parses_common_spellings() {
        assert_eq!(parse_rgb_tuple("[255, 255, 255]"), Some((255, 255, 255)));
        assert_eq!(parse_rgb_tuple("(0,0,0)"), Some((0, 0, 0)));
        assert_eq!(parse_rgb_tuple("1,2,3"), Some((1, 2, 3)));
        assert_eq!(parse_rgb_tuple("1,2"), None);
        assert_eq!(parse_rgb_tuple("1,2,3,4"), None);
    }
}
