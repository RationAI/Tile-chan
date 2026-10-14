//! End-to-end tests over the real router.
//!
//! These drive the assembled service the way a client does — routing, query parsing,
//! validation ordering, encoding — using the `pil` plugin over a temporary data directory.
//! They are the closest thing to the Python `wsi_service/tests/unit/test_api.py` that does
//! not need a real whole-slide image.

use axum::body::Body;
use axum::Router;
use http::{Request, StatusCode};
use tower::ServiceExt;

use wsi_core::settings::Settings;
use wsi_server::{build_router, build_state};

struct Fixture {
    root: std::path::PathBuf,
    router: Router,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("wsi-server-api-{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("case")).unwrap();

        // 300x200 so thumbnails and tiles have something to shrink.
        let mut image = image::RgbImage::new(300, 200);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
        }
        image.save(root.join("case/slide.png")).unwrap();
        std::fs::write(root.join("case/notes.txt"), b"not a slide").unwrap();

        let settings = Settings {
            data_dir: root.to_string_lossy().replace('\\', "/"),
            local_mode: "wsi_service.paths_mapper:PathsMapper".into(),
            api_v3_integration: "wsi_service.api.v3.integrations.disable_auth:DisableAuth".into(),
            ..Settings::default()
        };

        let state = build_state(settings).unwrap();
        Self {
            root,
            router: build_router(state),
        }
    }

    async fn get(&self, uri: &str) -> (StatusCode, http::HeaderMap, Vec<u8>) {
        let response = self
            .router
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, headers, body)
    }

    async fn get_json(&self, uri: &str) -> (StatusCode, serde_json::Value) {
        let (status, _, body) = self.get(uri).await;
        let value = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
        (status, value)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Slide IDs are paths under `PathsMapper`, so they arrive percent-encoded.
const SLIDE: &str = "case%2Fslide.png";

#[tokio::test]
async fn alive_reports_status_version_and_plugins() {
    let f = Fixture::new("alive");
    let (status, body) = f.get_json("/alive").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
    let plugins = body["plugins"].as_array().unwrap();
    assert!(plugins.iter().any(|p| p["name"] == "pil"));
    assert!(plugins[0]["priority"].is_i64());
}

#[tokio::test]
async fn slide_info_reports_extent_levels_and_format() {
    let f = Fixture::new("info");
    let (status, body) = f
        .get_json(&format!("/v3/slides/info?slide_id={SLIDE}"))
        .await;
    assert_eq!(status, StatusCode::OK);

    assert_eq!(body["id"], "case/slide.png");
    assert_eq!(body["extent"]["x"], 300);
    assert_eq!(body["extent"]["y"], 200);
    assert_eq!(body["extent"]["z"], 1);
    assert_eq!(body["num_levels"], 1);
    assert_eq!(body["levels"].as_array().unwrap().len(), 1);
    assert_eq!(body["channel_depth"], 8);
    assert_eq!(body["channels"].as_array().unwrap().len(), 3);
    // Pixel size is unknown for a plain raster image.
    assert_eq!(body["pixel_size_nm"]["x"], -1.0);
    // `_extend_slide_format_identifier`: file-<ext>-<plugin format>-<plugin name>.
    assert_eq!(body["format"], "file-png-png-pil");
    // The file is on disk, so raw download is offered.
    assert_eq!(body["raw_download"], true);
    // Optional fields are present as null rather than omitted, matching pydantic.
    assert!(body["pixel_size_nm"].as_object().unwrap().contains_key("z"));
}

#[tokio::test]
async fn a_missing_file_with_a_supported_extension_is_a_500() {
    // Support detection is extension-only for the `pil` plugin, so a non-existent
    // `.png` passes the mapper's check and fails later in `load_slide`
    // (`wsi_service/plugins.py:20`). Not a 404 — matching Python.
    let f = Fixture::new("missing-file");
    let (status, body) = f
        .get_json("/v3/slides/info?slide_id=case%2Fmissing.png")
        .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        body["detail"].as_str().unwrap().ends_with("not found."),
        "unexpected detail: {}",
        body["detail"]
    );
}

#[tokio::test]
async fn an_unsupported_file_is_not_a_slide() {
    // No plugin claims `.txt`, so the mapper itself rejects it with a 404.
    let f = Fixture::new("unsupported");
    let (status, body) = f
        .get_json("/v3/slides/info?slide_id=case%2Fnotes.txt")
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["detail"], "Slide case/notes.txt does not exist");
}

#[tokio::test]
async fn a_missing_slide_id_is_a_fastapi_422() {
    let f = Fixture::new("missing-param");
    let (status, body) = f.get_json("/v3/slides/info").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["detail"][0]["loc"][0], "query");
    assert_eq!(body["detail"][0]["loc"][1], "slide_id");
    assert_eq!(body["detail"][0]["type"], "missing");
}

#[tokio::test]
async fn tiles_default_to_jpeg() {
    let f = Fixture::new("tile-default");
    let (status, headers, body) = f
        .get(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "image/jpeg");
    assert_eq!(&body[..2], &[0xFF, 0xD8], "JPEG start-of-image marker");
}

#[tokio::test]
async fn regions_honour_the_requested_format_and_size() {
    let f = Fixture::new("region");
    let (status, headers, body) = f
        .get(&format!(
            "/v3/slides/region/level/0/start/0/0/size/64/32?slide_id={SLIDE}&image_format=png"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "image/png");

    let decoded = image::load_from_memory(&body).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (64, 32));
}

#[tokio::test]
async fn an_out_of_range_level_is_a_422_naming_the_coarsest_level() {
    let f = Fixture::new("level");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/region/level/5/start/0/0/size/8/8?slide_id={SLIDE}"
        ))
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["detail"],
        "The requested pyramid level is not available. The coarsest available level is 0."
    );
}

#[tokio::test]
async fn an_unsupported_image_format_is_a_400() {
    let f = Fixture::new("format");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&image_format=xcf"
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body["detail"],
        "Provided image format parameter not supported"
    );
}

#[tokio::test]
async fn a_region_larger_than_the_configured_maximum_is_a_422() {
    let f = Fixture::new("too-big");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/region/level/0/start/0/0/size/6000/6000?slide_id={SLIDE}"
        ))
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["detail"],
        "Requested region may not contain more than 25000000 pixels."
    );
}

#[tokio::test]
async fn a_zero_sized_region_is_rejected_by_the_path_constraint() {
    let f = Fixture::new("zero-size");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/region/level/0/start/0/0/size/0/8?slide_id={SLIDE}"
        ))
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["detail"][0]["loc"][0], "path");
    assert_eq!(body["detail"][0]["type"], "greater_than");
}

#[tokio::test]
async fn z_beyond_a_flat_image_is_a_422() {
    let f = Fixture::new("zstack");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&z=1"
        ))
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"]
        .as_str()
        .unwrap()
        .contains("does not support multiple z-layers"));
}

#[tokio::test]
async fn a_channel_beyond_the_channel_list_is_a_400() {
    let f = Fixture::new("channels");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&image_channels=7"
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["detail"]
        .as_str()
        .unwrap()
        .contains("selected: 7 max: 2"));
}

#[tokio::test]
async fn duplicate_channels_are_rejected() {
    let f = Fixture::new("dupes");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&image_channels=0&image_channels=0"
        ))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["detail"], "No duplicates allowed in channels");
}

#[tokio::test]
async fn thumbnails_fit_inside_the_requested_box() {
    let f = Fixture::new("thumbnail");
    let (status, headers, body) = f
        .get(&format!(
            "/v3/slides/thumbnail/max_size/100/100?slide_id={SLIDE}&image_format=png"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "image/png");

    let decoded = image::load_from_memory(&body).unwrap();
    // 300x200 fitted into 100x100 keeps the 3:2 ratio.
    assert_eq!((decoded.width(), decoded.height()), (100, 67));
}

#[tokio::test]
async fn thumbnail_size_is_capped_by_the_setting() {
    let f = Fixture::new("thumbnail-cap");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/thumbnail/max_size/9999/9999?slide_id={SLIDE}"
        ))
        .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["detail"][0]["loc"][1], "max_x");
    assert_eq!(body["detail"][0]["type"], "less_than_equal");
}

#[tokio::test]
async fn label_and_macro_are_404_for_plain_images() {
    let f = Fixture::new("assoc");
    for endpoint in ["label", "macro"] {
        let (status, _) = f
            .get_json(&format!(
                "/v3/slides/{endpoint}/max_size/50/50?slide_id={SLIDE}"
            ))
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{endpoint}");
    }
}

#[tokio::test]
async fn a_missing_icc_profile_escapes_as_a_plain_text_500() {
    // `icc_profile.py` imports HTTPException from `http.client`, not FastAPI, so the
    // "not available" case is an unhandled exception rather than a 404. See divergence 11.
    let f = Fixture::new("icc");
    let (status, headers, body) = f
        .get(&format!("/v3/slides/icc_profile?slide_id={SLIDE}"))
        .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(headers["content-type"], "text/plain; charset=utf-8");
    assert_eq!(String::from_utf8(body).unwrap(), "Internal Server Error");
}

#[tokio::test]
async fn download_streams_a_zip_named_after_the_slide() {
    let f = Fixture::new("download");
    let (status, headers, body) = f
        .get(&format!("/v3/slides/download?slide_id={SLIDE}"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "application/zip");
    assert_eq!(
        headers["content-disposition"],
        "attachment;filename=case/slide.png.zip"
    );

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(body)).unwrap();
    assert_eq!(archive.len(), 1);
    assert_eq!(archive.by_index(0).unwrap().name(), "slide.png");
}

#[tokio::test]
async fn batch_info_returns_one_entry_per_requested_slide() {
    let f = Fixture::new("batch-info");
    let (status, body) = f
        .get_json(&format!("/v3/files/info?paths={SLIDE},{SLIDE}"))
        .await;
    assert_eq!(status, StatusCode::OK);
    let items = body.as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["id"], "case/slide.png");
}

#[tokio::test]
async fn the_legacy_batch_spelling_uses_the_slides_parameter() {
    let f = Fixture::new("legacy-batch");
    let (status, body) = f.get_json(&format!("/v3/batch/info?slides={SLIDE}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 1);

    // The `paths` spelling is not accepted on the legacy route.
    let (status, _) = f.get_json(&format!("/v3/batch/info?paths={SLIDE}")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn batch_tiles_come_back_as_a_zip_numbered_from_one() {
    let f = Fixture::new("batch-tile");
    let (status, headers, body) = f
        .get(&format!(
            "/v3/files/tile/level/0/tile/0/0?paths={SLIDE},{SLIDE}&image_format=png"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "application/zip");

    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(body)).unwrap();
    assert_eq!(archive.len(), 2);
    assert_eq!(archive.by_index(0).unwrap().name(), "t1.png");
    assert_eq!(archive.by_index(1).unwrap().name(), "t2.png");
}

#[tokio::test]
async fn a_failing_slide_in_a_batch_becomes_a_dot_err_entry() {
    let f = Fixture::new("batch-partial");
    let (status, _, body) = f
        .get(&format!(
            "/v3/files/tile/level/9/tile/0/0?paths={SLIDE}&image_format=png"
        ))
        .await;
    // The request still succeeds; the failure is inside the archive.
    assert_eq!(status, StatusCode::OK);
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(body)).unwrap();
    assert_eq!(archive.by_index(0).unwrap().name(), "t1.err");
}

#[tokio::test]
async fn arbitrary_batch_reads_take_a_level_and_coordinate_per_slide() {
    let f = Fixture::new("batch-arbitrary");
    let (status, _, body) = f
        .get(&format!(
            "/v3/files/batch/?paths={SLIDE},{SLIDE}&levels=0,0&xs=0,0&ys=0,0&image_format=png"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
    let archive = zip::ZipArchive::new(std::io::Cursor::new(body)).unwrap();
    assert_eq!(archive.len(), 2);
}

#[tokio::test]
async fn local_mode_lists_cases_and_slides() {
    let f = Fixture::new("local-mode");

    // PathsMapper is context-dependent: no context, no listing.
    let (status, body) = f.get_json("/v3/cases/").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["detail"], "context is required");

    let (status, body) = f.get_json("/v3/cases/?context=.").await;
    assert_eq!(status, StatusCode::OK);
    let cases = body.as_array().unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0]["id"], "./case");
    assert_eq!(cases[0]["slides"][0], "./case/slide.png");

    let (status, body) = f.get_json("/v3/cases/slides/?case_id=case").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body[0]["local_id"], "slide.png");
}

#[tokio::test]
async fn local_mode_exposes_slide_storage() {
    let f = Fixture::new("storage");
    let (status, body) = f
        .get_json(&format!("/v3/slides/storage?slide_id={SLIDE}"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["storage_type"], "fs");
    assert_eq!(body["storage_addresses"][0]["address"], "case/slide.png");
    assert_eq!(body["storage_addresses"][0]["main_address"], true);
}

#[tokio::test]
async fn refreshing_the_local_mapper_succeeds() {
    let f = Fixture::new("refresh");
    let (status, body) = f.get_json("/v3/refresh_local_mapper").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["detail"], "Local mapper has been refreshed.");
}

#[tokio::test]
async fn the_viewer_page_substitutes_the_slide_id() {
    let f = Fixture::new("viewer");
    let (status, headers, body) = f.get("/slides/abc123/viewer").await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/html"));
    let html = String::from_utf8(body).unwrap();
    assert!(html.contains("abc123"));
    assert!(!html.contains("REPLACE_SLIDE_ID"));
}

#[tokio::test]
async fn both_openapi_documents_are_served() {
    let f = Fixture::new("openapi");

    let (status, body) = f.get_json("/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["paths"]["/alive"].is_object());

    let (status, body) = f.get_json("/v3/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["paths"]["/slides/info"].is_object());
    assert!(body["paths"]["/slides/tile/level/{level}/tile/{tile_x}/{tile_y}"].is_object());
}

#[tokio::test]
async fn docs_render_at_both_levels() {
    let f = Fixture::new("docs");
    for uri in ["/docs", "/v3/docs"] {
        let (status, _, body) = f.get(uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(String::from_utf8(body).unwrap().contains("swagger-ui"));
    }
}

#[tokio::test]
async fn an_unauthenticated_request_is_rejected_before_validation() {
    // With the fail-closed default integration every request is 401, even one whose query
    // string would otherwise be a 422 — matching FastAPI's dependency resolution order.
    let root = std::env::temp_dir().join("wsi-server-api-auth");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // api_v3_integration left empty -> DefaultIntegration
    let settings = Settings {
        data_dir: root.to_string_lossy().replace('\\', "/"),
        ..Settings::default()
    };
    let state = build_state(settings).unwrap();
    let router = build_router(state);

    let response = router
        .oneshot(
            Request::builder()
                .uri("/v3/slides/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["detail"], "Integration not configured");

    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn slides_are_reused_from_the_handle_cache() {
    let f = Fixture::new("cache");
    // Two requests for the same slide must both succeed; the second is served from the
    // LRU without reopening the file.
    for _ in 0..2 {
        let (status, _) = f
            .get_json(&format!("/v3/slides/info?slide_id={SLIDE}"))
            .await;
        assert_eq!(status, StatusCode::OK);
    }
}

#[tokio::test]
async fn a_pinned_plugin_that_cannot_open_the_slide_is_a_500() {
    let f = Fixture::new("plugin-pin");
    let (status, body) = f
        .get_json(&format!(
            "/v3/slides/info?slide_id={SLIDE}&plugin=openslide"
        ))
        .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body["detail"],
        "Selected plugin openslide is not available or does not support this slide. \
         Please specify another plugin."
    );
}

#[tokio::test]
async fn a_three_digit_padding_colour_reproduces_the_python_failure() {
    let f = Fixture::new("padding");
    let (status, headers, body) = f
        .get(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&padding_color=%23FFF"
        ))
        .await;
    // A bare ValueError, so Starlette answers with plain text rather than a JSON detail.
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(headers["content-type"], "text/plain; charset=utf-8");
    assert_eq!(String::from_utf8(body).unwrap(), "Internal Server Error");

    // A six-digit colour works.
    let (status, _) = f
        .get_json(&format!(
            "/v3/slides/tile/level/0/tile/0/0?slide_id={SLIDE}&padding_color=%23FF0000"
        ))
        .await;
    assert_eq!(status, StatusCode::OK);
}
