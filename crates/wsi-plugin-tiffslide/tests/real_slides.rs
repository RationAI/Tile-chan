//! The plugin against `tiffslide`'s own recorded output.
//!
//! `tests/fixtures/tiffslide.json` is what tiffslide 4.0.0 reports for three real slides:
//! the format it detects, the level pyramid it derives, the properties it synthesises, the
//! associated images it exposes, and `read_region` samples. Matching that record is direct
//! evidence of parity rather than a restatement of what this implementation produces.
//!
//! The slides are gigabytes and not in the repository, so everything here skips itself
//! unless `WSI_TEST_SLIDES` points at a directory laid out like the `cds-test-suite`
//! collection:
//!
//! ```text
//! $WSI_TEST_SLIDES/
//!   Aperio/CMU-1.svs
//!   Generic TIFF/CMU-1.tiff
//!   Hamamatsu/OS-1.ndpi
//! ```
//!
//! ```sh
//! WSI_TEST_SLIDES=/slides cargo test -p wsi-plugin-tiffslide --test real_slides -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use wsi_core::pixels::Pixels;
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};
use wsi_plugin_tiffslide::reader::TiffSlideReader;
use wsi_plugin_tiffslide::TiffSlidePlugin;

/// JPEG tiles are decoded here by the pure-Rust decoder the `image` crate bundles and in
/// tiffslide by libjpeg-turbo through `imagecodecs`; the two round the inverse DCT
/// differently. The bound is measured — see `docs/divergences.md`, divergence 28.
const JPEG_TOLERANCE: i32 = 4;

fn manifest() -> BTreeMap<String, Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiffslide.json");
    let text = std::fs::read_to_string(path).expect("tests/fixtures/tiffslide.json");
    let record: Value = serde_json::from_str(&text).expect("valid JSON");
    record["slides"]
        .as_object()
        .expect("slides object")
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Hamamatsu NDPI is refused outright; see `docs/divergences.md`, divergence 33. Every
/// other test skips it rather than restating that.
fn is_refused(expected: &Value) -> bool {
    expected["format"].as_str() == Some("hamamatsu")
}

fn slide_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("WSI_TEST_SLIDES")?);
    root.is_dir().then_some(root)
}

/// The slide path, or `None` when the collection does not have it.
fn locate(relative: &str) -> Option<PathBuf> {
    let path = slide_root()?.join(relative);
    if !path.exists() {
        eprintln!("skipping {relative}: not present");
        return None;
    }
    Some(path)
}

async fn open(relative: &str) -> Option<Arc<dyn Slide>> {
    let path = locate(relative)?;
    let plugin = TiffSlidePlugin::new(Arc::new(Settings::default()));
    assert!(
        plugin.is_supported(&path),
        "{relative} should be claimed by the tiffslide plugin"
    );
    Some(
        plugin
            .open(&path)
            .await
            .unwrap_or_else(|e| panic!("could not open {relative}: {}", e.detail_text())),
    )
}

fn as_f64(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn as_i64(value: &Value) -> i64 {
    value.as_i64().expect("integer")
}

#[tokio::test]
async fn slide_info_matches_what_tiffslide_reports() {
    for (relative, expected) in manifest() {
        if is_refused(&expected) {
            continue;
        }
        let Some(slide) = open(&relative).await else {
            continue;
        };
        let info = slide.get_info().await.unwrap();
        let at = &relative;

        assert_eq!(
            info.format,
            expected["format"].as_str().unwrap(),
            "{at}: format"
        );

        let dimensions = expected["dimensions"].as_array().unwrap();
        assert_eq!(info.extent.x, as_i64(&dimensions[0]), "{at}: extent x");
        assert_eq!(info.extent.y, as_i64(&dimensions[1]), "{at}: extent y");
        assert_eq!(info.extent.z, 1, "{at}: extent z");

        assert_eq!(
            info.num_levels,
            as_i64(&expected["level_count"]),
            "{at}: level count"
        );
        for (index, level) in expected["level_dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let dims = level.as_array().unwrap();
            assert_eq!(
                (info.levels[index].extent.x, info.levels[index].extent.y),
                (as_i64(&dims[0]), as_i64(&dims[1])),
                "{at}: level {index} extent"
            );
        }
        for (index, downsample) in expected["level_downsamples"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let want = as_f64(downsample);
            let got = info.levels[index].downsample_factor;
            assert!(
                (got - want).abs() < 1e-9,
                "{at}: level {index} downsample: got {got}, tiffslide says {want}"
            );
        }

        // `tiffslide.level[0].tile-{width,height}`
        let properties = &expected["properties"];
        assert_eq!(
            info.tile_extent.x,
            as_i64(&properties["tiffslide.level[0].tile-width"]),
            "{at}: tile width"
        );
        assert_eq!(
            info.tile_extent.y,
            as_i64(&properties["tiffslide.level[0].tile-height"]),
            "{at}: tile height"
        );

        // `pixel_size_nm = 1000 * mpp`
        for (axis, key, got) in [
            ("x", "tiffslide.mpp-x", info.pixel_size_nm.x),
            ("y", "tiffslide.mpp-y", info.pixel_size_nm.y),
        ] {
            let want = as_f64(&properties[key]) * 1000.0;
            assert!(
                (got - want).abs() < 1e-6,
                "{at}: pixel size {axis}: got {got}, tiffslide says {want}"
            );
        }

        assert_eq!(info.channel_depth, Some(8), "{at}: channel depth");
        assert_eq!(
            info.channels.as_ref().map(Vec::len),
            Some(3),
            "{at}: channel count"
        );
    }
}

#[tokio::test]
async fn the_icc_profile_matches_in_length_or_is_absent() {
    for (relative, expected) in manifest() {
        if is_refused(&expected) {
            continue;
        }
        let Some(slide) = open(&relative).await else {
            continue;
        };
        match expected["profile_length"].as_u64() {
            Some(length) => {
                let profile = slide
                    .get_icc_profile()
                    .await
                    .unwrap_or_else(|e| panic!("{relative}: {}", e.detail_text()));
                assert_eq!(profile.len() as u64, length, "{relative}: profile length");
            }
            None => {
                // No profile: `get_for_payload` raises the unhandled 404 of divergence 11.
                let error = slide.get_icc_profile().await.unwrap_err();
                assert!(error.unhandled, "{relative}: expected the plain-text 500");
            }
        }
    }
}

#[tokio::test]
async fn the_associated_images_are_the_ones_tiffslide_exposes() {
    for (relative, expected) in manifest() {
        if is_refused(&expected) {
            continue;
        }
        let Some(path) = locate(&relative) else {
            continue;
        };
        let mut reader = TiffSlideReader::open(&path).unwrap();
        let mut names: Vec<String> = reader
            .associated_names()
            .into_iter()
            .map(str::to_owned)
            .collect();
        names.sort();
        let mut want: Vec<String> = expected["associated_images"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        want.sort();
        assert_eq!(names, want, "{relative}: associated image names");

        for (name, record) in expected["associated_images"].as_object().unwrap() {
            let image = reader
                .associated_image(name)
                .unwrap_or_else(|| panic!("{relative}: {name} missing"))
                .unwrap_or_else(|e| panic!("{relative}: {name}: {}", e.detail_text()));
            let size = record["size"].as_array().unwrap();
            assert_eq!(
                (image.width() as i64, image.height() as i64),
                (as_i64(&size[0]), as_i64(&size[1])),
                "{relative}: {name} size"
            );
        }
    }
}

#[tokio::test]
async fn region_pixels_match_what_tiffslide_reads() {
    for (relative, expected) in manifest() {
        let Some(path) = locate(&relative) else {
            continue;
        };
        if is_refused(&expected) {
            continue;
        }
        let mut reader = TiffSlideReader::open(&path).unwrap();
        let mut worst = 0i32;
        for region in expected["regions"].as_array().unwrap() {
            if region.get("error").is_some() {
                continue;
            }
            let level = as_i64(&region["level"]) as usize;
            let location = region["location"].as_array().unwrap();
            let size = region["size"].as_array().unwrap();
            let (width, height) = (as_i64(&size[0]) as u32, as_i64(&size[1]) as u32);
            let image = reader
                .read_region(
                    (as_i64(&location[0]), as_i64(&location[1])),
                    level,
                    (width, height),
                )
                .unwrap_or_else(|e| panic!("{relative}: {}", e.detail_text()));
            assert_eq!(image.dimensions(), (width, height), "{relative}: size");

            for (key, want) in region["samples"].as_object().unwrap() {
                let (x, y) = key.split_once(',').expect("x,y key");
                let (x, y) = (x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap());
                let got = image.get_pixel(x, y).0;
                let want = want.as_array().unwrap();
                for channel in 0..3 {
                    let difference =
                        (i32::from(got[channel]) - as_i64(&want[channel]) as i32).abs();
                    worst = worst.max(difference);
                    assert!(
                        difference <= JPEG_TOLERANCE,
                        "{relative} level {level} at ({x}, {y}) channel {channel}: \
                         got {}, tiffslide says {} (tolerance {JPEG_TOLERANCE})",
                        got[channel],
                        want[channel]
                    );
                }
            }
        }
        eprintln!("{relative}: worst channel deviation {worst}");
    }
}

#[tokio::test]
async fn a_tile_is_shrunk_to_the_level_before_it_is_read() {
    // Unlike openslide, tiffslide's `get_tile` reduces the read to the part of the tile
    // inside the level, so an edge tile comes back smaller rather than padded.
    let Some(slide) = open("Generic TIFF/CMU-1.tiff").await else {
        return;
    };
    let info = slide.get_info().await.unwrap();
    let tile = info.tile_extent;
    let level = info.levels.len() - 1;
    let extent = info.levels[level].extent;

    let last_x = (extent.x - 1) / tile.x;
    let last_y = (extent.y - 1) / tile.y;
    let pixels = slide
        .get_tile(level, last_x, last_y, ReadOptions::default())
        .await
        .unwrap();
    let Pixels::Image(image) = pixels else {
        panic!("expected an image");
    };
    let expected_width = (extent.x - last_x * tile.x).min(tile.x);
    let expected_height = (extent.y - last_y * tile.y).min(tile.y);
    assert_eq!(
        (image.width() as i64, image.height() as i64),
        (expected_width, expected_height)
    );
}

#[tokio::test]
async fn a_label_and_a_macro_are_served_where_the_slide_has_them() {
    for (relative, expected) in manifest() {
        if is_refused(&expected) {
            continue;
        }
        let Some(slide) = open(&relative).await else {
            continue;
        };
        let associated = expected["associated_images"].as_object().unwrap();
        for (name, result) in [
            ("label", slide.get_label().await),
            ("macro", slide.get_macro(None, false).await),
        ] {
            match (associated.get(name), result) {
                (Some(record), Ok(Pixels::Image(image))) => {
                    let size = record["size"].as_array().unwrap();
                    assert_eq!(
                        (image.width() as i64, image.height() as i64),
                        (as_i64(&size[0]), as_i64(&size[1])),
                        "{relative}: {name} size"
                    );
                }
                (None, Err(error)) => {
                    assert_eq!(
                        error.status,
                        http::StatusCode::NOT_FOUND,
                        "{relative}: {name} should be a 404"
                    );
                }
                (expected, actual) => panic!(
                    "{relative}: {name}: record {} but result {}",
                    expected.is_some(),
                    actual.is_ok()
                ),
            }
        }
    }
}

#[tokio::test]
async fn an_ndpi_is_refused_with_a_reason() {
    // The pages are one restart-marker JPEG each; see divergence 33. Refusing is better
    // than serving a slide whose tile extent is zero and whose every read fails.
    let Some(path) = locate("Hamamatsu/OS-1.ndpi") else {
        return;
    };
    let plugin = TiffSlidePlugin::new(Arc::new(Settings::default()));
    // The extension is still claimed — Python's `is_supported` lists `.ndpi` — so the
    // refusal has to come from `open`, where the plugin registry can fall through to
    // another plugin.
    assert!(plugin.is_supported(&path));
    let error = match plugin.open(&path).await {
        Ok(_) => panic!("expected the NDPI to be refused"),
        Err(error) => error,
    };
    assert!(
        error.detail_text().contains("MCU-level entry"),
        "unexpected message: {}",
        error.detail_text()
    );
}
