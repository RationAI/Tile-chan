//! The request matrix.
//!
//! For each discovered slide, this enumerates the requests worth comparing: every
//! endpoint, every level, in-bounds and edge and out-of-bounds coordinates, every output
//! format, and the parameter combinations that select different code paths in the image
//! pipeline (channel selection, z-stack, padding colour, ICC intent).
//!
//! Requests are generated from the slide's own `info` response, so a slide with four
//! levels and eight channels is exercised across all of them rather than at some fixed
//! guess.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SlideExtent {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SlideLevel {
    pub extent: SlideExtent,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SlideInfo {
    pub extent: SlideExtent,
    pub num_levels: i64,
    pub tile_extent: SlideExtent,
    pub levels: Vec<SlideLevel>,
    #[serde(default)]
    pub channels: Option<Vec<serde_json::Value>>,
}

/// One comparison: a label for the report and the path to request from both services.
#[derive(Debug, Clone)]
pub struct Request {
    pub label: String,
    pub path: String,
}

impl Request {
    fn new(label: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            path: path.into(),
        }
    }
}

/// Output formats worth exercising. `raw`/`json`/`geojson`/`mvt` only produce a payload
/// for the vector-tile plugin, but the error path is still worth comparing.
const FORMATS: [&str; 8] = ["jpeg", "png", "bmp", "tiff", "gif", "jpg", "tif", "raw"];

const QUALITIES: [i64; 4] = [0, 50, 90, 100];

const ICC_INTENTS: [&str; 4] = [
    "perceptual",
    "relative_colorimetric",
    "saturation",
    "absolute_colorimetric",
];

/// Percent-encodes a slide ID for use in a query string. Path-shaped IDs from
/// `PathsMapper` contain slashes and dots that must survive intact.
pub fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

pub fn build(slide_id: &str, info: &SlideInfo) -> Vec<Request> {
    let id = encode(slide_id);
    let mut requests = Vec::new();

    // --- metadata -------------------------------------------------------------------
    requests.push(Request::new(
        "info",
        format!("/v3/slides/info?slide_id={id}"),
    ));
    requests.push(Request::new(
        "info:unknown-plugin",
        format!("/v3/slides/info?slide_id={id}&plugin=definitely-not-a-plugin"),
    ));
    requests.push(Request::new(
        "info:missing-slide-id",
        "/v3/slides/info".to_string(),
    ));
    requests.push(Request::new(
        "icc_profile",
        format!("/v3/slides/icc_profile?slide_id={id}"),
    ));
    requests.push(Request::new(
        "download",
        format!("/v3/slides/download?slide_id={id}"),
    ));

    // --- associated images ----------------------------------------------------------
    for endpoint in ["thumbnail", "label", "macro"] {
        for (max_x, max_y) in [(1, 1), (100, 100), (37, 91)] {
            requests.push(Request::new(
                format!("{endpoint}:{max_x}x{max_y}"),
                format!("/v3/slides/{endpoint}/max_size/{max_x}/{max_y}?slide_id={id}"),
            ));
        }
        // Out of the declared bounds; only `thumbnail` constrains them.
        requests.push(Request::new(
            format!("{endpoint}:out-of-bounds"),
            format!("/v3/slides/{endpoint}/max_size/9999/9999?slide_id={id}"),
        ));
        for format in FORMATS {
            requests.push(Request::new(
                format!("{endpoint}:format={format}"),
                format!("/v3/slides/{endpoint}/max_size/64/64?slide_id={id}&image_format={format}"),
            ));
        }
    }

    // --- tiles ----------------------------------------------------------------------
    let tile_w = info.tile_extent.x.max(1);
    let tile_h = info.tile_extent.y.max(1);

    for level in 0..info.num_levels.max(1) {
        let extent = info
            .levels
            .get(level as usize)
            .map(|l| (l.extent.x, l.extent.y))
            .unwrap_or((info.extent.x, info.extent.y));

        // Interior, last, one past the last, and far outside.
        let last_x = (extent.0 - 1).max(0) / tile_w;
        let last_y = (extent.1 - 1).max(0) / tile_h;
        let coordinates = [
            (0, 0, "origin"),
            (last_x, last_y, "edge"),
            (last_x + 1, last_y + 1, "past-edge"),
            (10_000, 10_000, "far-outside"),
        ];

        for (tile_x, tile_y, kind) in coordinates {
            requests.push(Request::new(
                format!("tile:l{level}:{kind}"),
                format!("/v3/slides/tile/level/{level}/tile/{tile_x}/{tile_y}?slide_id={id}"),
            ));
        }

        // Region coordinates that exercise each branch of the padding logic.
        let regions = [
            (0, 0, 64, 64, "interior"),
            (-32, -32, 64, 64, "negative-origin"),
            (
                (extent.0 - 32).max(0),
                (extent.1 - 32).max(0),
                64,
                64,
                "overhang",
            ),
            (extent.0, extent.1, 64, 64, "exactly-outside"),
            (100_000, 100_000, 16, 16, "far-outside"),
            (0, 0, extent.0.max(1), extent.1.max(1), "whole-level"),
        ];
        for (start_x, start_y, size_x, size_y, kind) in regions {
            requests.push(Request::new(
                format!("region:l{level}:{kind}"),
                format!(
                    "/v3/slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}?slide_id={id}"
                ),
            ));
        }
    }

    // A level past the end, to compare the 422.
    requests.push(Request::new(
        "tile:level-out-of-range",
        format!(
            "/v3/slides/tile/level/{}/tile/0/0?slide_id={id}",
            info.num_levels + 5
        ),
    ));
    requests.push(Request::new(
        "region:too-large",
        format!("/v3/slides/region/level/0/start/0/0/size/60000/60000?slide_id={id}"),
    ));
    requests.push(Request::new(
        "region:zero-size",
        format!("/v3/slides/region/level/0/start/0/0/size/0/8?slide_id={id}"),
    ));

    // --- formats and quality --------------------------------------------------------
    for format in FORMATS {
        requests.push(Request::new(
            format!("tile:format={format}"),
            format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_format={format}"),
        ));
    }
    requests.push(Request::new(
        "tile:format=unsupported",
        format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_format=xcf"),
    ));
    for quality in QUALITIES {
        requests.push(Request::new(
            format!("tile:quality={quality}"),
            format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_quality={quality}"),
        ));
    }
    for quality in [-1, 101] {
        requests.push(Request::new(
            format!("tile:quality={quality}:invalid"),
            format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_quality={quality}"),
        ));
    }

    // --- channel selection ----------------------------------------------------------
    let channel_count = info.channels.as_ref().map_or(0, Vec::len);
    if channel_count > 0 {
        let selections: Vec<Vec<usize>> = vec![
            vec![0],
            vec![0, 1],
            (0..channel_count.min(3)).collect(),
            (0..channel_count).collect(),
        ];
        for selection in selections {
            if selection.is_empty() || selection.iter().any(|c| *c >= channel_count) {
                continue;
            }
            let query: String = selection
                .iter()
                .map(|c| format!("&image_channels={c}"))
                .collect();
            let label = selection
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            requests.push(Request::new(
                format!("tile:channels={label}"),
                format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}{query}"),
            ));
            requests.push(Request::new(
                format!("tile:channels={label}:tiff"),
                format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_format=tiff{query}"),
            ));
        }
        // Out of range and duplicated.
        requests.push(Request::new(
            "tile:channels=out-of-range",
            format!(
                "/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_channels={channel_count}"
            ),
        ));
        requests.push(Request::new(
            "tile:channels=duplicate",
            format!(
                "/v3/slides/tile/level/0/tile/0/0?slide_id={id}&image_channels=0&image_channels=0"
            ),
        ));
    }

    // --- z-stack --------------------------------------------------------------------
    for z in [0, 1, info.extent.z.max(1) - 1, info.extent.z.max(1)] {
        requests.push(Request::new(
            format!("tile:z={z}"),
            format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&z={z}"),
        ));
    }

    // --- padding colour -------------------------------------------------------------
    for colour in ["%23FFFFFF", "%23000000", "%23FF8000", "%23FFF", "nonsense"] {
        requests.push(Request::new(
            format!("region:padding={colour}"),
            format!(
                "/v3/slides/region/level/0/start/-16/-16/size/64/64?slide_id={id}&padding_color={colour}"
            ),
        ));
    }

    // --- ICC ------------------------------------------------------------------------
    for intent in ICC_INTENTS {
        for strict in ["false", "true"] {
            requests.push(Request::new(
                format!("tile:icc={intent}:strict={strict}"),
                format!(
                    "/v3/slides/tile/level/0/tile/0/0?slide_id={id}\
                     &icc_profile_intent={intent}&icc_profile_strict={strict}"
                ),
            ));
        }
    }
    requests.push(Request::new(
        "tile:icc=invalid",
        format!("/v3/slides/tile/level/0/tile/0/0?slide_id={id}&icc_profile_intent=nonsense"),
    ));

    // --- batch ----------------------------------------------------------------------
    for prefix in ["files", "batch"] {
        let param = if prefix == "files" { "paths" } else { "slides" };
        requests.push(Request::new(
            format!("{prefix}:info"),
            format!("/v3/{prefix}/info?{param}={id},{id}"),
        ));
        requests.push(Request::new(
            format!("{prefix}:tile"),
            format!("/v3/{prefix}/tile/level/0/tile/0/0?{param}={id},{id}"),
        ));
        requests.push(Request::new(
            format!("{prefix}:tile:png"),
            format!("/v3/{prefix}/tile/level/0/tile/0/0?{param}={id}&image_format=png"),
        ));
        requests.push(Request::new(
            format!("{prefix}:tile:failing-level"),
            format!("/v3/{prefix}/tile/level/999/tile/0/0?{param}={id}"),
        ));
        requests.push(Request::new(
            format!("{prefix}:batch"),
            format!("/v3/{prefix}/batch/?{param}={id},{id}&levels=0,0&xs=0,0&ys=0,0"),
        ));
        requests.push(Request::new(
            format!("{prefix}:thumbnail"),
            format!("/v3/{prefix}/thumbnail/max_size/64/64?{param}={id}"),
        ));
        requests.push(Request::new(
            format!("{prefix}:icc_profile"),
            // The legacy route reads `paths` on both spellings.
            format!("/v3/{prefix}/icc_profile?paths={id}"),
        ));
    }

    requests
}

/// Requests that do not depend on a particular slide.
pub fn service_level(context: Option<&str>) -> Vec<Request> {
    let mut requests = vec![
        Request::new("alive", "/alive"),
        Request::new("openapi:root", "/openapi.json"),
        Request::new("openapi:v3", "/v3/openapi.json"),
        Request::new("cases:no-context", "/v3/cases/"),
    ];
    if let Some(context) = context {
        requests.push(Request::new(
            "cases",
            format!("/v3/cases/?context={}", encode(context)),
        ));
    }
    requests.push(Request::new(
        "cases:missing-context-folder",
        "/v3/cases/?context=definitely-not-here",
    ));
    requests
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> SlideInfo {
        SlideInfo {
            extent: SlideExtent {
                x: 300,
                y: 200,
                z: 1,
            },
            num_levels: 2,
            tile_extent: SlideExtent {
                x: 256,
                y: 256,
                z: 1,
            },
            levels: vec![
                SlideLevel {
                    extent: SlideExtent {
                        x: 300,
                        y: 200,
                        z: 1,
                    },
                },
                SlideLevel {
                    extent: SlideExtent {
                        x: 150,
                        y: 100,
                        z: 1,
                    },
                },
            ],
            channels: Some(vec![
                serde_json::json!({}),
                serde_json::json!({}),
                serde_json::json!({}),
            ]),
        }
    }

    #[test]
    fn slide_ids_with_slashes_are_percent_encoded() {
        assert_eq!(encode("case/slide.png"), "case%2Fslide.png");
        assert_eq!(encode("plain-id"), "plain-id");
        assert_eq!(encode("a b"), "a%20b");
    }

    #[test]
    fn every_endpoint_is_covered() {
        let requests = build("case/slide.png", &info());
        let paths: Vec<&str> = requests.iter().map(|r| r.path.as_str()).collect();
        let joined = paths.join("\n");

        for endpoint in [
            "/v3/slides/info",
            "/v3/slides/thumbnail/",
            "/v3/slides/label/",
            "/v3/slides/macro/",
            "/v3/slides/region/",
            "/v3/slides/tile/",
            "/v3/slides/download",
            "/v3/slides/icc_profile",
            "/v3/files/info",
            "/v3/files/batch/",
            "/v3/batch/info",
        ] {
            assert!(joined.contains(endpoint), "missing {endpoint}");
        }
    }

    #[test]
    fn every_level_is_exercised() {
        let requests = build("s", &info());
        for level in 0..2 {
            assert!(
                requests
                    .iter()
                    .any(|r| r.label.contains(&format!("tile:l{level}:"))),
                "level {level} not covered"
            );
        }
    }

    #[test]
    fn out_of_bounds_coordinates_are_included() {
        let requests = build("s", &info());
        for kind in [
            "past-edge",
            "far-outside",
            "negative-origin",
            "exactly-outside",
        ] {
            assert!(
                requests.iter().any(|r| r.label.contains(kind)),
                "missing {kind}"
            );
        }
    }

    #[test]
    fn channel_selections_stay_within_the_declared_channels() {
        let requests = build("s", &info());
        // Three channels declared, so no request may name index 3 except the deliberate
        // out-of-range probe.
        let bad: Vec<&Request> = requests
            .iter()
            .filter(|r| r.path.contains("image_channels=3") && !r.label.contains("out-of-range"))
            .collect();
        assert!(bad.is_empty(), "unexpected requests: {bad:?}");
    }

    #[test]
    fn the_three_digit_padding_colour_probe_is_present() {
        let requests = build("s", &info());
        assert!(requests
            .iter()
            .any(|r| r.path.contains("padding_color=%23FFF&")
                || r.path.ends_with("padding_color=%23FFF")));
    }

    #[test]
    fn service_level_requests_include_both_openapi_documents() {
        let requests = service_level(Some("."));
        let labels: Vec<&str> = requests.iter().map(|r| r.label.as_str()).collect();
        assert!(labels.contains(&"openapi:root"));
        assert!(labels.contains(&"openapi:v3"));
        assert!(labels.contains(&"alive"));
        assert!(labels.contains(&"cases"));
    }
}
