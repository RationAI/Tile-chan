//! Root-application routes, ported from `wsi_service/api/root/`.

use axum::extract::{Path, State};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::get;
use axum::Router;

use wsi_core::models::WsiServiceStatus;

use crate::state::AppState;

/// The viewer pages are compiled in, so the binary has no runtime asset dependency.
/// Both files are copied verbatim from `wsi_service/api/root/`.
const VIEWER_HTML: &str = include_str!("../../../../assets/viewer.html");
const VALIDATION_VIEWER_HTML: &str = include_str!("../../../../assets/validation_viewer.html");

/// `add_routes_root` — `api/root/__init__.py:5`.
pub fn router(state: &AppState) -> Router<AppState> {
    let mut router = Router::new().route("/alive", get(alive));

    if state.settings.enable_viewer_routes {
        router = router.route("/slides/{slide_id}/viewer", get(viewer));

        // `/validation_viewer` is registered only when a local mapper is configured.
        if !state.settings.local_mode.is_empty() {
            router = router.route("/validation_viewer", get(validation_viewer));
        }
    }

    router
}

/// `GET /alive` — `api/root/alive.py:8`.
async fn alive(State(state): State<AppState>) -> Response {
    Json(WsiServiceStatus {
        status: "ok".into(),
        version: state.settings.version.clone(),
        message: None,
        plugins: state.registry.overview(),
    })
    .into_response()
}

/// `GET /slides/{slide_id}/viewer` — `api/root/viewer.py:8`.
///
/// The placeholder substitution is a plain string replace, exactly as in Python, but the
/// value is escaped first. Python does not escape it at all, so the page reflects
/// arbitrary input straight into script source; that is a stored-and-reflected XSS and
/// reproducing it would be a poor trade for byte parity on an HTML page the parity
/// harness does not compare.
///
/// In `assets/viewer.html` the placeholder sits inside **template literals**, twice:
///
/// ```js
/// return `/v3/slides/REPLACE_SLIDE_ID/tile/level/${level}/...`
/// ```
///
/// so [`escape_js_string`] has to be safe for that context too, not just for quoted
/// strings — see its own documentation for why `$` and a backtick are the characters
/// that matter most here.
async fn viewer(Path(slide_id): Path<String>) -> Html<String> {
    Html(VIEWER_HTML.replace("REPLACE_SLIDE_ID", &escape_js_string(&slide_id)))
}

/// `GET /validation_viewer` — `api/root/viewer.py:18`.
async fn validation_viewer() -> Html<&'static str> {
    Html(VALIDATION_VIEWER_HTML)
}

/// Escapes a value for inclusion in a JavaScript string — single-quoted, double-quoted
/// **or** a template literal.
///
/// The template-literal case is the one that bites, because it does not need a quote to
/// break out of. Two characters carry the whole weight:
///
/// - a backtick closes the literal outright;
/// - `$` begins `${…}`, which evaluates an arbitrary expression — and `${alert(1)}`
///   contains no quote, no angle bracket and no backslash, so escaping only those leaves
///   it untouched. Escaping `$` itself is what closes that off, and it is safe to do
///   unconditionally: `$` is just a dollar sign in every string context.
///
/// `<`, `>` and `&` are escaped so the value cannot close the surrounding `<script>`
/// element, and U+2028/U+2029 because they terminate a line in JavaScript source even
/// though they are ordinary characters elsewhere.
fn escape_js_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\'' => out.push_str("\\'"),
            '`' => out.push_str("\\u0060"),
            '$' => out.push_str("\\u0024"),
            '<' => out.push_str("\\u003C"),
            '>' => out.push_str("\\u003E"),
            '&' => out.push_str("\\u0026"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            // Line and paragraph separators end a line in JavaScript source.
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_template_contains_the_placeholder() {
        assert!(VIEWER_HTML.contains("REPLACE_SLIDE_ID"));
    }

    #[test]
    fn slide_ids_are_escaped_before_substitution() {
        assert_eq!(escape_js_string("plain-id"), "plain-id");
        assert_eq!(escape_js_string("proj/case/a.svs"), "proj/case/a.svs");
        // Real IDs carry spaces and dots and must survive intact.
        assert_eq!(
            escape_js_string("./Fluorescence OME-Tif/a.ome.tif"),
            "./Fluorescence OME-Tif/a.ome.tif"
        );
        assert_eq!(escape_js_string("a\"b"), "a\\\"b");
        assert_eq!(escape_js_string("</script>"), "\\u003C/script\\u003E");
    }

    /// The placeholder is substituted inside a template literal, where neither a quote nor
    /// an angle bracket is needed to break out. Each of these is a working payload against
    /// an escaper that only handles quoted-string syntax.
    #[test]
    fn a_template_literal_cannot_be_broken_out_of() {
        // `${…}` evaluates an expression and contains nothing a quote-only escaper
        // touches, which is what made this reachable.
        let escaped = escape_js_string("${alert(1)}");
        assert!(!escaped.contains("${"), "{escaped}");
        assert_eq!(escaped, "\\u0024{alert(1)}");

        // A backtick would close the literal and let a second one re-open it.
        let escaped = escape_js_string("`+alert(1)+`");
        assert!(!escaped.contains('`'), "{escaped}");

        // A lone dollar is harmless but still escaped, so there is no sequence to analyse.
        assert_eq!(escape_js_string("a$b"), "a\\u0024b");
    }

    #[test]
    fn javascript_line_terminators_are_escaped() {
        // Ordinary characters in HTML, but they end a line in JavaScript source.
        assert_eq!(escape_js_string("a\u{2028}b"), "a\\u2028b");
        assert_eq!(escape_js_string("a\u{2029}b"), "a\\u2029b");
    }

    /// Guards the assumption the escaper is written against: if the template ever moves the
    /// placeholder out of a template literal and into, say, an HTML attribute, this escaper
    /// is the wrong tool and the test should fail loudly rather than the page go quiet.
    #[test]
    fn the_placeholder_is_only_ever_inside_a_template_literal() {
        for line in VIEWER_HTML
            .lines()
            .filter(|l| l.contains("REPLACE_SLIDE_ID"))
        {
            let before = line.split("REPLACE_SLIDE_ID").next().unwrap();
            assert_eq!(
                before.matches('`').count() % 2,
                1,
                "placeholder is not inside a template literal: {line}"
            );
        }
    }
}
