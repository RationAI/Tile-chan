//! OpenAPI documents and the Swagger UI page.
//!
//! FastAPI mounts `/v3` as a separate application, so there are two documents: the root
//! one describing `/alive`, and the `/v3` one describing every slide route. Both are
//! suppressed when `WS_DISABLE_OPENAPI` is set — Python passes `openapi_url=""`, which
//! removes the route entirely.
//!
//! The documents are built from a route table rather than derived from handler signatures.
//! A derive-based generator would drift from the hand-written query parsing in
//! [`crate::query`], and that parsing is the thing clients actually meet.

use axum::extract::State;
use axum::response::{Html, IntoResponse, Json, Response};
use serde_json::{json, Value};

use crate::state::AppState;

/// Common query parameters, matching `wsi_service/custom_models/queries.py`.
fn param(name: &str, location: &str, required: bool, schema: Value, description: &str) -> Value {
    json!({
        "name": name,
        "in": location,
        "required": required,
        "schema": schema,
        "description": description,
    })
}

fn slide_id_param() -> Value {
    param(
        "slide_id",
        "query",
        true,
        json!({ "type": "string" }),
        "Provide id.",
    )
}

fn plugin_param() -> Value {
    param(
        "plugin",
        "query",
        false,
        json!({ "type": "string", "nullable": true }),
        "Select a specific WSI Service Plugin.",
    )
}

fn image_params(max_thumbnail_size: u32) -> Vec<Value> {
    let _ = max_thumbnail_size;
    vec![
        param(
            "image_format",
            "query",
            false,
            json!({ "type": "string", "default": "jpeg" }),
            "Output format (e.g. bmp, gif, jpeg, png, tiff, raw, json, geojson, mvt, pbf).",
        ),
        param(
            "image_quality",
            "query",
            false,
            json!({ "type": "integer", "default": 90, "minimum": 0, "maximum": 100 }),
            "Image quality (only for specific formats).",
        ),
        param(
            "icc_profile_intent",
            "query",
            false,
            json!({
                "type": "string",
                "nullable": true,
                "enum": [
                    "perceptual",
                    "relative_colorimetric",
                    "saturation",
                    "absolute_colorimetric"
                ]
            }),
            "Apply ICC profiles using the given intent. No intent means no ICC handling.",
        ),
        param(
            "icc_profile_strict",
            "query",
            false,
            json!({ "type": "boolean", "default": false }),
            "Require ICC profile application.",
        ),
    ]
}

fn channel_params() -> Vec<Value> {
    vec![
        param(
            "image_channels",
            "query",
            false,
            json!({ "type": "array", "items": { "type": "integer" }, "nullable": true }),
            "List of requested image channels. By default all channels are returned.",
        ),
        param(
            "z",
            "query",
            false,
            json!({ "type": "integer", "default": 0, "minimum": 0 }),
            "Z-Stack layer index z",
        ),
        param(
            "padding_color",
            "query",
            false,
            json!({ "type": "string", "nullable": true }),
            "Background color as 24bit-hex-string with leading #.",
        ),
    ]
}

fn path_param(name: &str, description: &str, minimum: Option<i64>, maximum: Option<i64>) -> Value {
    let mut schema = json!({ "type": "integer" });
    if let Some(min) = minimum {
        schema["minimum"] = json!(min);
    }
    if let Some(max) = maximum {
        schema["maximum"] = json!(max);
    }
    param(name, "path", true, schema, description)
}

fn image_responses() -> Value {
    json!({
        "200": {
            "description": "Successful Response",
            "content": {
                "image/*": {},
                "application/octet-stream": {},
                "application/json": {},
                "application/geo+json": {},
                "application/vnd.mapbox-vector-tile": {}
            }
        },
        "422": { "description": "Validation Error" }
    })
}

fn json_response(schema: Value) -> Value {
    json!({
        "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": schema } }
        },
        "422": { "description": "Validation Error" }
    })
}

fn operation(summary: &str, parameters: Vec<Value>, responses: Value) -> Value {
    json!({
        "get": {
            "tags": ["Main Routes"],
            "summary": summary,
            "operationId": summary.to_lowercase().replace(' ', "_"),
            "parameters": parameters,
            "responses": responses,
        }
    })
}

fn schemas() -> Value {
    json!({
        "SlideExtent": {
            "type": "object",
            "required": ["x", "y", "z"],
            "properties": {
                "x": { "type": "integer", "description": "Extent in horizontal direction" },
                "y": { "type": "integer", "description": "Extent in vertical direction" },
                "z": { "type": "integer", "description": "Number of Z-Stack layers" }
            }
        },
        "SlideColor": {
            "type": "object",
            "required": ["r", "g", "b", "a"],
            "properties": {
                "r": { "type": "integer" },
                "g": { "type": "integer" },
                "b": { "type": "integer" },
                "a": { "type": "integer" }
            }
        },
        "SlideChannel": {
            "type": "object",
            "required": ["id", "name", "color"],
            "properties": {
                "id": { "type": "integer" },
                "name": { "type": "string" },
                "color": { "$ref": "#/components/schemas/SlideColor" }
            }
        },
        "SlidePixelSizeNm": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
                "x": { "type": "number" },
                "y": { "type": "number" },
                "z": { "type": "number", "nullable": true }
            }
        },
        "SlideLevel": {
            "type": "object",
            "required": ["extent", "downsample_factor"],
            "properties": {
                "extent": { "$ref": "#/components/schemas/SlideExtent" },
                "downsample_factor": { "type": "number" }
            }
        },
        "SlideInfo": {
            "type": "object",
            "required": ["id", "extent", "num_levels", "pixel_size_nm", "tile_extent", "levels", "format"],
            "properties": {
                "id": { "type": "string" },
                "channels": {
                    "type": "array",
                    "nullable": true,
                    "items": { "$ref": "#/components/schemas/SlideChannel" }
                },
                "channel_depth": { "type": "integer", "nullable": true, "default": 8 },
                "extent": { "$ref": "#/components/schemas/SlideExtent" },
                "num_levels": { "type": "integer" },
                "pixel_size_nm": { "$ref": "#/components/schemas/SlidePixelSizeNm" },
                "tile_extent": { "$ref": "#/components/schemas/SlideExtent" },
                "levels": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/SlideLevel" }
                },
                "format": { "type": "string" },
                "raw_download": { "type": "boolean", "default": false }
            }
        },
        "StorageAddress": {
            "type": "object",
            "required": ["address", "slide_id", "main_address", "storage_address_id"],
            "properties": {
                "address": { "type": "string" },
                "slide_id": { "type": "string" },
                "main_address": { "type": "boolean" },
                "storage_address_id": { "type": "string" }
            }
        },
        "SlideStorage": {
            "type": "object",
            "required": ["slide_id", "storage_type", "storage_addresses"],
            "properties": {
                "slide_id": { "type": "string" },
                "storage_type": { "type": "string" },
                "storage_addresses": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/StorageAddress" }
                }
            }
        },
        "CaseLocalMapper": {
            "type": "object",
            "required": ["id", "local_id", "slides"],
            "properties": {
                "id": { "type": "string" },
                "local_id": { "type": "string" },
                "slides": { "type": "array", "items": { "type": "string" } }
            }
        },
        "SlideLocalMapper": {
            "type": "object",
            "required": ["id", "local_id", "slide_storage"],
            "properties": {
                "id": { "type": "string" },
                "local_id": { "type": "string" },
                "slide_storage": { "$ref": "#/components/schemas/SlideStorage" }
            }
        },
        "PluginInfo": {
            "type": "object",
            "required": ["name", "version", "priority"],
            "properties": {
                "name": { "type": "string" },
                "version": { "type": "string" },
                "priority": { "type": "integer" }
            }
        },
        "WSIServiceStatus": {
            "type": "object",
            "required": ["status", "version", "plugins"],
            "properties": {
                "status": { "type": "string", "enum": ["ok", "failure"] },
                "version": { "type": "string" },
                "message": { "type": "string", "nullable": true },
                "plugins": {
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/PluginInfo" }
                }
            }
        }
    })
}

/// The root application's document: `/alive` only, since the viewer routes are excluded
/// from the schema (`include_in_schema=False`).
pub fn root_document(state: &AppState) -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": state.settings.title,
            "description": state.settings.description,
            "version": state.settings.version,
        },
        "paths": {
            "/alive": {
                "get": {
                    "tags": ["Server"],
                    "summary": "Alive",
                    "operationId": "alive",
                    "responses": json_response(json!({ "$ref": "#/components/schemas/WSIServiceStatus" })),
                }
            }
        },
        "components": { "schemas": schemas() }
    })
}

/// The `/v3` sub-application's document.
pub fn v3_document(state: &AppState) -> Value {
    let max_thumb = state.settings.max_thumbnail_size as i64;

    let mut paths = serde_json::Map::new();

    // --- single slide -------------------------------------------------------------------
    paths.insert(
        "/slides/info".into(),
        operation(
            "Get slide info",
            vec![slide_id_param(), plugin_param()],
            json_response(json!({ "$ref": "#/components/schemas/SlideInfo" })),
        ),
    );

    for (path, name, bounded) in [
        (
            "/slides/thumbnail/max_size/{max_x}/{max_y}",
            "Get thumbnail",
            true,
        ),
        ("/slides/label/max_size/{max_x}/{max_y}", "Get label", false),
        ("/slides/macro/max_size/{max_x}/{max_y}", "Get macro", false),
    ] {
        let (min, max) = if bounded {
            (Some(1), Some(max_thumb))
        } else {
            (None, None)
        };
        let mut params = vec![
            slide_id_param(),
            path_param("max_x", "Maximum width", min, max),
            path_param("max_y", "Maximum height", min, max),
        ];
        params.extend(image_params(state.settings.max_thumbnail_size));
        params.push(plugin_param());
        paths.insert(path.into(), operation(name, params, image_responses()));
    }

    {
        let mut params = vec![
            slide_id_param(),
            path_param("level", "Pyramid level of region", Some(0), None),
            path_param("start_x", "x component of start coordinate", None, None),
            path_param("start_y", "y component of start coordinate", None, None),
            path_param("size_x", "Width of requested region", Some(1), None),
            path_param("size_y", "Height of requested region", Some(1), None),
        ];
        params.extend(channel_params());
        params.extend(image_params(state.settings.max_thumbnail_size));
        params.push(plugin_param());
        let mut responses = image_responses();
        responses["413"] = json!({ "description": "Requested region is too large" });
        paths.insert(
            "/slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}".into(),
            operation("Get region", params, responses),
        );
    }

    {
        let mut params = vec![
            slide_id_param(),
            path_param("level", "Pyramid level of tile", Some(0), None),
            path_param("tile_x", "Tile index in x", None, None),
            path_param("tile_y", "Tile index in y", None, None),
        ];
        params.extend(channel_params());
        params.extend(image_params(state.settings.max_thumbnail_size));
        params.push(plugin_param());
        paths.insert(
            "/slides/tile/level/{level}/tile/{tile_x}/{tile_y}".into(),
            operation("Get tile", params, image_responses()),
        );
    }

    paths.insert(
        "/slides/download".into(),
        operation(
            "Download raw slide data as zip",
            vec![slide_id_param(), plugin_param()],
            json!({
                "200": {
                    "description": "Successful Response",
                    "content": { "application/zip": {} }
                }
            }),
        ),
    );
    paths.insert(
        "/slides/icc_profile".into(),
        operation(
            "Download icc profile for a slide",
            vec![slide_id_param(), plugin_param()],
            json!({
                "200": {
                    "description": "Successful Response",
                    "content": { "application/octet-stream": {} }
                }
            }),
        ),
    );

    // --- multi slide --------------------------------------------------------------------
    for (prefix, id_name) in [("/files", "paths"), ("/batch", "slides")] {
        let id_param = || {
            param(
                id_name,
                "query",
                true,
                json!({ "type": "string" }),
                "Comma-separated list of slides to access simultaneously.",
            )
        };

        paths.insert(
            format!("{prefix}/info"),
            operation(
                &format!("Get info for {id_name}"),
                vec![id_param(), plugin_param()],
                json_response(json!({
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/SlideInfo" }
                })),
            ),
        );

        for (suffix, name, bounded) in [
            ("thumbnail", "thumbnails", true),
            ("label", "labels", false),
            ("macro", "macros", false),
        ] {
            let (min, max) = if bounded {
                (Some(1), Some(max_thumb))
            } else {
                (None, None)
            };
            let mut params = vec![
                id_param(),
                path_param("max_x", "Maximum width", min, max),
                path_param("max_y", "Maximum height", min, max),
            ];
            params.extend(image_params(state.settings.max_thumbnail_size));
            params.push(plugin_param());
            paths.insert(
                format!("{prefix}/{suffix}/max_size/{{max_x}}/{{max_y}}"),
                operation(&format!("Get {name}"), params, zip_responses()),
            );
        }

        {
            let mut params = vec![
                id_param(),
                path_param("level", "Pyramid level of tile", Some(0), None),
                path_param("tile_x", "Tile index in x", None, None),
                path_param("tile_y", "Tile index in y", None, None),
            ];
            params.extend(channel_params());
            params.extend(image_params(state.settings.max_thumbnail_size));
            params.push(plugin_param());
            paths.insert(
                format!("{prefix}/tile/level/{{level}}/tile/{{tile_x}}/{{tile_y}}"),
                operation("Get tiles", params, zip_responses()),
            );
        }

        {
            let mut params = vec![
                id_param(),
                param(
                    "levels",
                    "query",
                    true,
                    json!({ "type": "string" }),
                    "Comma-separated level per requested slide.",
                ),
                param(
                    "xs",
                    "query",
                    true,
                    json!({ "type": "string" }),
                    "Comma-separated tile x per requested slide.",
                ),
                param(
                    "ys",
                    "query",
                    true,
                    json!({ "type": "string" }),
                    "Comma-separated tile y per requested slide.",
                ),
            ];
            params.extend(channel_params());
            params.extend(image_params(state.settings.max_thumbnail_size));
            params.push(plugin_param());
            paths.insert(
                format!("{prefix}/batch/"),
                operation("Get arbitrary tiles", params, zip_responses()),
            );
        }

        paths.insert(
            format!("{prefix}/icc_profile"),
            operation(
                "Get icc profiles",
                vec![
                    // The legacy /batch/icc_profile route reads `paths`, not `slides`.
                    param(
                        if prefix == "/batch" { "paths" } else { id_name },
                        "query",
                        true,
                        json!({ "type": "string" }),
                        "Comma-separated list of slides.",
                    ),
                    plugin_param(),
                ],
                zip_responses(),
            ),
        );
    }

    // --- local mode ---------------------------------------------------------------------
    if state.local_mode_enabled() {
        paths.insert(
            "/cases/".into(),
            standalone_operation(
                "Get cases",
                vec![param(
                    "context",
                    "query",
                    false,
                    json!({ "type": "string", "nullable": true }),
                    "Optional context path for context-dependent mappers.",
                )],
                json_response(json!({
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/CaseLocalMapper" }
                })),
            ),
        );
        paths.insert(
            "/cases/slides/".into(),
            standalone_operation(
                "Get case slides",
                vec![param(
                    "case_id",
                    "query",
                    true,
                    json!({ "type": "string" }),
                    "Provide id.",
                )],
                json_response(json!({
                    "type": "array",
                    "items": { "$ref": "#/components/schemas/SlideLocalMapper" }
                })),
            ),
        );
        paths.insert(
            "/slides".into(),
            standalone_operation(
                "Get slide",
                vec![slide_id_param()],
                json_response(json!({ "$ref": "#/components/schemas/SlideLocalMapper" })),
            ),
        );
        paths.insert(
            "/slides/storage".into(),
            standalone_operation(
                "Get slide storage",
                vec![slide_id_param()],
                json_response(json!({ "$ref": "#/components/schemas/SlideStorage" })),
            ),
        );
        paths.insert(
            "/refresh_local_mapper".into(),
            standalone_operation("Refresh local mapper", vec![], json_response(json!({}))),
        );
    }

    let mut document = json!({
        "openapi": "3.1.0",
        "info": {
            "title": "FastAPI",
            "version": "0.1.0",
        },
        "paths": Value::Object(paths),
        "components": { "schemas": schemas() }
    });

    // The EMPAIA integration advertises an OAuth2 authorization-code flow.
    if let Some(security) = state.auth.openapi_security() {
        document["components"]["securitySchemes"] = json!({
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": security.authorization_url,
                        "tokenUrl": security.token_url,
                        "scopes": {}
                    }
                }
            }
        });
        document["security"] = json!([{ "OAuth2": [] }]);
    }

    document
}

fn zip_responses() -> Value {
    json!({
        "200": {
            "description": "Successful Response",
            "content": { "application/zip": {} }
        },
        "422": { "description": "Validation Error" }
    })
}

fn standalone_operation(summary: &str, parameters: Vec<Value>, responses: Value) -> Value {
    json!({
        "get": {
            "tags": ["Additional Routes (Standalone WSI Service)"],
            "summary": summary,
            "operationId": summary.to_lowercase().replace(' ', "_"),
            "parameters": parameters,
            "responses": responses,
        }
    })
}

pub async fn root_json(State(state): State<AppState>) -> Response {
    Json(root_document(&state)).into_response()
}

pub async fn v3_json(State(state): State<AppState>) -> Response {
    Json(v3_document(&state)).into_response()
}

/// jsDelivr serves an exact version immutably, so pinning one keeps the asset stable.
/// FastAPI pins the same way (`swagger_ui_default_parameters`).
const SWAGGER_UI_VERSION: &str = "5.17.14";

/// The Swagger UI page, matching what FastAPI serves at `/docs` — including loading the
/// bundle from the same CDN, so an air-gapped deployment behaves the same way it does
/// under Python (the page renders, the script does not load).
///
/// The `<script>` tag carries no `integrity` attribute. Adding one requires the exact
/// SHA-384 of the pinned bundle, which has to be fetched and recorded rather than guessed;
/// until then the exact-version pin is what limits exposure. Deployments that care should
/// self-host the bundle and point `SWAGGER_UI_VERSION` at it — the endpoint is a developer
/// convenience and can be removed entirely with `WS_DISABLE_OPENAPI=true`.
pub async fn docs(State(state): State<AppState>) -> Html<String> {
    let title = format!("{} - Swagger UI", state.settings.title);
    let openapi_url = format!("{}/openapi.json", state.settings.root_path);
    let version = SWAGGER_UI_VERSION;
    Html(format!(
        r#"<!DOCTYPE html>
<html>
<head>
<link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{version}/swagger-ui.css">
<title>{title}</title>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{version}/swagger-ui-bundle.js" crossorigin="anonymous"></script>
<script>
const ui = SwaggerUIBundle({{
    url: '{openapi_url}',
    dom_id: '#swagger-ui',
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    layout: "BaseLayout",
    deepLinking: true,
    showExtensions: true,
    showCommonExtensions: true
}})
</script>
</body>
</html>"#
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use wsi_core::plugins::PluginRegistry;
    use wsi_core::settings::Settings;
    use wsi_core::slide_manager::SlideManager;
    use wsi_core::storage::StorageResolver;

    fn state(local_mode: bool) -> AppState {
        let settings = Arc::new(Settings::default());
        let registry = Arc::new(PluginRegistry::new(vec![]));
        let resolver = StorageResolver::remote(reqwest::Client::new(), "http://example/{slide_id}");
        let slide_manager = Arc::new(SlideManager::new(
            resolver,
            Arc::clone(&registry),
            settings.data_dir.clone(),
            600,
            50,
        ));
        AppState {
            settings,
            registry,
            slide_manager,
            auth: Arc::new(wsi_auth::DisableAuth),
            local_mapper: if local_mode {
                Some(Arc::new(NoopMapper))
            } else {
                None
            },
        }
    }

    struct NoopMapper;
    impl wsi_core::storage::LocalMapper for NoopMapper {
        fn get_cases(
            &self,
            _: Option<&str>,
        ) -> wsi_core::error::Result<Vec<wsi_core::models::CaseLocalMapper>> {
            Ok(vec![])
        }
        fn get_slides(
            &self,
            _: &str,
        ) -> wsi_core::error::Result<Vec<wsi_core::models::SlideLocalMapper>> {
            Ok(vec![])
        }
        fn get_slide(
            &self,
            _: &str,
        ) -> wsi_core::error::Result<wsi_core::models::SlideLocalMapper> {
            Err(wsi_core::error::ApiError::not_found("none"))
        }
    }

    #[test]
    fn v3_document_lists_every_slide_route() {
        let doc = v3_document(&state(false));
        let paths = doc["paths"].as_object().unwrap();
        for expected in [
            "/slides/info",
            "/slides/thumbnail/max_size/{max_x}/{max_y}",
            "/slides/label/max_size/{max_x}/{max_y}",
            "/slides/macro/max_size/{max_x}/{max_y}",
            "/slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}",
            "/slides/tile/level/{level}/tile/{tile_x}/{tile_y}",
            "/slides/download",
            "/slides/icc_profile",
        ] {
            assert!(paths.contains_key(expected), "missing {expected}");
        }
    }

    #[test]
    fn v3_document_lists_both_batch_spellings() {
        let doc = v3_document(&state(false));
        let paths = doc["paths"].as_object().unwrap();
        for prefix in ["/files", "/batch"] {
            for suffix in [
                "/info",
                "/thumbnail/max_size/{max_x}/{max_y}",
                "/label/max_size/{max_x}/{max_y}",
                "/macro/max_size/{max_x}/{max_y}",
                "/tile/level/{level}/tile/{tile_x}/{tile_y}",
                "/batch/",
                "/icc_profile",
            ] {
                let key = format!("{prefix}{suffix}");
                assert!(paths.contains_key(&key), "missing {key}");
            }
        }
    }

    #[test]
    fn local_mode_routes_appear_only_when_a_mapper_exists() {
        let without = v3_document(&state(false));
        assert!(!without["paths"]
            .as_object()
            .unwrap()
            .contains_key("/cases/"));

        let with = v3_document(&state(true));
        let paths = with["paths"].as_object().unwrap();
        for expected in [
            "/cases/",
            "/cases/slides/",
            "/slides",
            "/slides/storage",
            "/refresh_local_mapper",
        ] {
            assert!(paths.contains_key(expected), "missing {expected}");
        }
    }

    #[test]
    fn thumbnail_max_size_is_bounded_but_label_is_not() {
        let doc = v3_document(&state(false));
        let thumb =
            &doc["paths"]["/slides/thumbnail/max_size/{max_x}/{max_y}"]["get"]["parameters"];
        let max_x = thumb
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "max_x")
            .unwrap();
        assert_eq!(max_x["schema"]["minimum"], 1);
        assert_eq!(max_x["schema"]["maximum"], 500);

        let label = &doc["paths"]["/slides/label/max_size/{max_x}/{max_y}"]["get"]["parameters"];
        let max_x = label
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "max_x")
            .unwrap();
        assert!(max_x["schema"].get("minimum").is_none());
    }

    #[test]
    fn root_document_exposes_alive_with_the_status_schema() {
        let doc = root_document(&state(false));
        assert!(doc["paths"]["/alive"]["get"].is_object());
        assert!(doc["components"]["schemas"]["WSIServiceStatus"].is_object());
    }
}
