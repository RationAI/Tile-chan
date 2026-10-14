//! Replays `tests/fixtures/wsidicom.json` against the port.
//!
//! The expectations are wsidicom 0.27.1's own output over the three real DICOM
//! directories, recorded by `tests/fixtures/generate.py`. Skipped unless `WSI_TEST_SLIDES`
//! points at the collection — the same arrangement the `openslide`, `tifffile` and
//! `tiffslide` real-slide tests use.
//!
//! Pixels are compared at the five corners-and-centre points the generator recorded, and
//! with a tolerance: `wsidicom` decodes a JPEG frame through `imagecodecs` and this decodes
//! it with its own decoder, which is divergence 28. The tolerance is the *measured*
//! deviation, and the test reports the worst it saw so a regression shows up as a number
//! rather than as a pass.

use std::path::PathBuf;

use serde_json::Value;
use wsi_plugin_tifffile::tiff::page::Raster;
use wsi_plugin_wsidicom::reader::WsiDicomReader;

/// Divergence 28's bound. The JPEG decoders disagree in the last bit of the inverse DCT.
const MAX_CHANNEL_DELTA: i32 = 4;

fn collection() -> Option<PathBuf> {
    std::env::var_os("WSI_TEST_SLIDES").map(PathBuf::from)
}

fn ground_truth() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wsidicom.json");
    let text = std::fs::read_to_string(path).expect("the recorded ground truth");
    serde_json::from_str(&text).expect("valid JSON")
}

/// Tracks the worst deviation seen, so the tolerance can be justified rather than assumed.
#[derive(Default)]
struct Deviation {
    worst: i32,
    where_worst: String,
    compared: usize,
}

impl Deviation {
    fn check(&mut self, what: &str, expected: &[i64], actual: [u8; 3]) {
        self.compared += 1;
        for (channel, want) in expected.iter().enumerate() {
            let got = i32::from(actual[channel]);
            let delta = (got - *want as i32).abs();
            if delta > self.worst {
                self.worst = delta;
                self.where_worst = format!("{what} channel {channel}: {want} vs {got}");
            }
            assert!(
                delta <= MAX_CHANNEL_DELTA,
                "{what} channel {channel}: expected {want}, got {got} \
                 (delta {delta} > {MAX_CHANNEL_DELTA})"
            );
        }
    }

    fn report(&self, slide: &str) {
        eprintln!(
            "{slide}: {} sample points compared, worst deviation {} ({})",
            self.compared,
            self.worst,
            if self.where_worst.is_empty() {
                "exact"
            } else {
                &self.where_worst
            }
        );
    }
}

fn pixel(raster: &Raster, x: usize, y: usize) -> [u8; 3] {
    let at = (y * raster.width + x) * raster.samples;
    match raster.samples {
        3 => [raster.bytes[at], raster.bytes[at + 1], raster.bytes[at + 2]],
        1 => {
            let value = raster.bytes[at];
            [value, value, value]
        }
        other => panic!("{other} samples per pixel"),
    }
}

/// Parses a `"x,y"` sample key.
fn point(key: &str) -> (usize, usize) {
    let (x, y) = key.split_once(',').expect("an \"x,y\" key");
    (x.parse().expect("x"), y.parse().expect("y"))
}

fn samples(entry: &Value) -> Option<&serde_json::Map<String, Value>> {
    entry.get("samples")?.as_object()
}

fn numbers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("a sample is an array")
        .iter()
        .map(|v| v.as_i64().expect("a sample channel is an integer"))
        .collect()
}

#[test]
fn the_pyramid_matches_what_wsidicom_reports() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };
    let truth = ground_truth();
    let slides = truth["slides"].as_object().expect("a slides object");

    for (key, expected) in slides {
        let path = root.join(key);
        if !path.is_dir() {
            eprintln!("{} is missing; skipping it", path.display());
            continue;
        }
        let reader = WsiDicomReader::open(&path).expect("the directory should open");

        let expected_levels = expected["levels"].as_array().expect("levels");
        assert_eq!(
            reader.levels.len(),
            expected_levels.len(),
            "{key}: level count"
        );
        assert_eq!(
            expected["level_count"].as_u64().unwrap() as usize,
            reader.levels.len(),
            "{key}: level_count disagrees with the levels array"
        );

        for (index, want) in expected_levels.iter().enumerate() {
            let level = &reader.levels[index];
            let field = |name: &str| want[name].as_i64().unwrap();
            // The exponent is the one that is easy to get wrong: Aperio reports 0, 2, 4
            // across list indices 0, 1, 2, and the plugin reports `2 ** exponent` as the
            // downsample.
            assert_eq!(
                i64::from(level.exponent),
                field("level"),
                "{key} level {index}: exponent"
            );
            assert_eq!(
                i64::from(level.width),
                field("width"),
                "{key} level {index}: width"
            );
            assert_eq!(
                i64::from(level.height),
                field("height"),
                "{key} level {index}: height"
            );
            assert_eq!(
                i64::from(level.tile_width),
                field("tile_width"),
                "{key} level {index}: tile width"
            );
            assert_eq!(
                i64::from(level.tile_height),
                field("tile_height"),
                "{key} level {index}: tile height"
            );

            // mpp is `PixelSpacing * 1000`, and the service multiplies by 1000 again for
            // `pixel_size_nm`, so an error here is visible in `/info`.
            let (mpp_x, mpp_y) = level.mpp.expect("every volume level has a pixel spacing");
            let want_x = want["mpp_width"].as_f64().unwrap();
            let want_y = want["mpp_height"].as_f64().unwrap();
            assert!(
                (mpp_x - want_x).abs() < 1e-9 && (mpp_y - want_y).abs() < 1e-9,
                "{key} level {index}: mpp {mpp_x}x{mpp_y}, expected {want_x}x{want_y}"
            );
        }
    }
}

#[test]
fn regions_and_tiles_match_what_wsidicom_decoded() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };
    let truth = ground_truth();

    for (key, expected) in truth["slides"].as_object().unwrap() {
        let path = root.join(key);
        if !path.is_dir() {
            continue;
        }
        let reader = WsiDicomReader::open(&path).expect("the directory should open");
        let mut deviation = Deviation::default();

        for entry in expected["regions"].as_array().unwrap() {
            // An entry the generator recorded as an error is one wsidicom itself refused;
            // the port has to refuse it too, which the out-of-bounds test covers.
            let Some(samples) = samples(entry) else {
                continue;
            };
            let level_index = entry["level_index"].as_u64().unwrap() as usize;
            let location = numbers(&entry["location"]);
            let size = numbers(&entry["size"]);
            let exponent = reader.levels[level_index].exponent;

            let raster = reader
                .read_region(
                    (location[0] as u32, location[1] as u32),
                    exponent,
                    (size[0] as u32, size[1] as u32),
                )
                .unwrap_or_else(|e| panic!("{key} region {location:?} {size:?}: {e}"));
            assert_eq!(raster.width as i64, size[0], "{key}: region width");
            assert_eq!(raster.height as i64, size[1], "{key}: region height");

            for (sample_key, want) in samples {
                let (x, y) = point(sample_key);
                deviation.check(
                    &format!("{key} region l{level_index} {location:?} at {sample_key}"),
                    &numbers(want),
                    pixel(&raster, x, y),
                );
            }
        }

        for entry in expected["tiles"].as_array().unwrap() {
            let Some(samples) = samples(entry) else {
                continue;
            };
            let level_index = entry["level_index"].as_u64().unwrap() as usize;
            let tile = numbers(&entry["tile"]);
            let size = numbers(&entry["size"]);
            let exponent = reader.levels[level_index].exponent;

            let raster = reader
                .read_tile(exponent, (tile[0] as u32, tile[1] as u32))
                .unwrap_or_else(|e| panic!("{key} tile {tile:?}: {e}"));
            // `read_tile` returns a whole frame, which TILED_FULL pads to the full tile
            // size even at the edge of the image.
            assert_eq!(raster.width as i64, size[0], "{key}: tile width");
            assert_eq!(raster.height as i64, size[1], "{key}: tile height");

            for (sample_key, want) in samples {
                let (x, y) = point(sample_key);
                deviation.check(
                    &format!("{key} tile l{level_index} {tile:?} at {sample_key}"),
                    &numbers(want),
                    pixel(&raster, x, y),
                );
            }
        }

        deviation.report(key);
    }
}

#[test]
fn the_label_and_overview_match_what_wsidicom_decoded() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };
    let truth = ground_truth();

    for (key, expected) in truth["slides"].as_object().unwrap() {
        let path = root.join(key);
        if !path.is_dir() {
            continue;
        }
        let reader = WsiDicomReader::open(&path).expect("the directory should open");
        let mut deviation = Deviation::default();

        for (name, entry) in expected["associated"].as_object().unwrap() {
            let Some(samples) = samples(entry) else {
                continue;
            };
            let raster = match name.as_str() {
                "label" => reader.read_label(),
                "overview" => reader.read_overview(),
                // `THUMBNAIL` instances are reached through the thumbnail path, not here.
                other => panic!("unexpected associated image {other}"),
            }
            .unwrap_or_else(|e| panic!("{key} {name}: {e}"));

            let size = numbers(&entry["size"]);
            assert_eq!(raster.width as i64, size[0], "{key} {name}: width");
            assert_eq!(raster.height as i64, size[1], "{key} {name}: height");

            for (sample_key, want) in samples {
                let (x, y) = point(sample_key);
                deviation.check(
                    &format!("{key} {name} at {sample_key}"),
                    &numbers(want),
                    pixel(&raster, x, y),
                );
            }
        }

        deviation.report(&format!("{key} (associated)"));
    }
}

/// The thumbnail *source* and its size, which is where the selection rule shows.
///
/// `read_thumbnail` picks the smallest image at least as wide as the box — preferring a
/// `THUMBNAIL` instance over a level — and only then shrinks it with Pillow. The recorded
/// output size is what pins the choice: 341x500 for Leica, which has a 737x1080 thumbnail
/// instance, against 500x358 for Aperio, which has none and falls to its 2875x2057 level.
#[test]
fn the_thumbnail_source_is_the_smallest_image_that_covers_the_box() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };
    let truth = ground_truth();

    for (key, expected) in truth["slides"].as_object().unwrap() {
        let path = root.join(key);
        if !path.is_dir() {
            continue;
        }
        let reader = WsiDicomReader::open(&path).expect("the directory should open");
        let thumbnail = &expected["thumbnail"];
        let Some(samples) = samples(thumbnail) else {
            continue;
        };
        let requested = numbers(&thumbnail["requested"]);
        let want_size = numbers(&thumbnail["size"]);

        let source = reader
            .thumbnail_source((requested[0] as u32, requested[1] as u32))
            .unwrap_or_else(|e| panic!("{key} thumbnail: {e}"));

        // Shrinking the source has to land on the size wsidicom reported, which is what
        // proves the *right* source was chosen.
        let image =
            image::RgbImage::from_raw(source.width as u32, source.height as u32, to_rgb(&source))
                .expect("the source raster");
        let shrunk = wsi_core::image::pillow_thumbnail(
            &image::DynamicImage::ImageRgb8(image),
            requested[0] as u32,
            requested[1] as u32,
            wsi_core::image::Filter::Bicubic,
        );
        assert_eq!(
            (i64::from(shrunk.width()), i64::from(shrunk.height())),
            (want_size[0], want_size[1]),
            "{key}: thumbnail size, which means the wrong source was selected"
        );

        let rgb = shrunk.to_rgb8();
        let mut deviation = Deviation::default();
        for (sample_key, want) in samples {
            let (x, y) = point(sample_key);
            let pixel = rgb.get_pixel(x as u32, y as u32).0;
            deviation.check(
                &format!("{key} thumbnail at {sample_key}"),
                &numbers(want),
                pixel,
            );
        }
        deviation.report(&format!("{key} (thumbnail)"));
    }
}

fn to_rgb(raster: &Raster) -> Vec<u8> {
    match raster.samples {
        3 => raster.bytes.clone(),
        1 => raster.bytes.iter().flat_map(|v| [*v, *v, *v]).collect(),
        other => panic!("{other} samples per pixel"),
    }
}

/// A region reaching outside the level is refused, not padded.
///
/// `WsiDicom.read_region` checks `valid_pixels` and raises; the Python plugin does not
/// catch it. Padding instead would be a silent difference in the pixels a client gets.
#[test]
fn a_region_outside_the_level_is_refused() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };

    for key in [
        "DICOM_Leica/f990df0c-5e8e-16e5-000f-2874470dcd3a_050136",
        "DICOM_wsidicomizer/DICOM_Aperio",
        "DICOM_wsidicomizer/DICOM_Mirax",
    ] {
        let path = root.join(key);
        if !path.is_dir() {
            continue;
        }
        let reader = WsiDicomReader::open(&path).expect("the directory should open");
        let base = &reader.levels[0];
        let exponent = base.exponent;
        let (width, height) = (base.width, base.height);

        // One pixel past the right edge.
        assert!(
            reader
                .read_region((width - 8, 0), exponent, (16, 16))
                .is_err(),
            "{key}: a region crossing the right edge should be refused"
        );
        // And past the bottom.
        assert!(
            reader
                .read_region((0, height - 8), exponent, (16, 16))
                .is_err(),
            "{key}: a region crossing the bottom edge should be refused"
        );
        // A region that fits exactly is fine, which shows the check is not off by one.
        assert!(
            reader
                .read_region((width - 16, height - 16), exponent, (16, 16))
                .is_ok(),
            "{key}: a region ending exactly at the edge should be allowed"
        );

        // A tile index past the grid is refused too, which is what the plugin turns into
        // its catch-all 500.
        let across = width.div_ceil(base.tile_width);
        assert!(
            reader.read_tile(exponent, (across, 0)).is_err(),
            "{key}: a tile past the grid should be refused"
        );
    }
}
