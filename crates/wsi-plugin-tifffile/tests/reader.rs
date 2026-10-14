//! The reader, checked against `tifffile` itself.
//!
//! `tests/fixtures/manifest.json` records, for every fixture, what tifffile reports about
//! `series[0]` and the exact bytes `page.asarray()` produces. These tests replay that
//! record, so a divergence in the series reconstruction, in a codec, or in the predictor
//! shows up as a concrete mismatch rather than as a plausible-looking image.
//!
//! Regenerate the record with `tests/fixtures/generate.py`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde_json::Value;
use wsi_plugin_tifffile::tiff::page::{Page, Raster, SampleDtype};
use wsi_plugin_tifffile::tiff::TiffFile;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn manifest() -> BTreeMap<String, Value> {
    let text = std::fs::read_to_string(data_dir().join("manifest.json"))
        .expect("tests/fixtures/manifest.json; run tests/fixtures/generate.py");
    serde_json::from_str(&text).expect("manifest is JSON")
}

fn as_usize(value: &Value, key: &str) -> usize {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} missing")) as usize
}

fn decode_base64(text: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64")
}

/// tifffile's `asarray()` layout for a page: `(H, W)`, `(H, W, S)` for chunky samples, or
/// `(S, H, W)` when the planes are stored separately.
fn expected_layout(page: &Page, shape: &[usize]) -> &'static str {
    match shape.len() {
        2 => "hw",
        3 if page.separate_planes() => "shw",
        3 => "hws",
        other => panic!("unexpected page array rank {other}"),
    }
}

/// Reads a sample out of the recorded numpy buffer, given its layout.
fn recorded(
    bytes: &[u8],
    dtype: SampleDtype,
    shape: &[usize],
    layout: &str,
    y: usize,
    x: usize,
    sample: usize,
) -> f64 {
    let index = match layout {
        "hw" => y * shape[1] + x,
        "hws" => (y * shape[1] + x) * shape[2] + sample,
        "shw" => (sample * shape[1] + y) * shape[2] + x,
        other => panic!("unknown layout {other}"),
    };
    let size = dtype.size();
    let raster = Raster {
        height: 1,
        width: 1,
        samples: 1,
        dtype,
        bytes: bytes[index * size..(index + 1) * size].to_vec(),
    };
    raster.get(0, 0, 0)
}

fn dtype_of(text: &str) -> SampleDtype {
    match text {
        "|u1" => SampleDtype::U8,
        "|i1" => SampleDtype::I8,
        "<u2" => SampleDtype::U16,
        "<i2" => SampleDtype::I16,
        "<u4" => SampleDtype::U32,
        "<i4" => SampleDtype::I32,
        "<u8" => SampleDtype::U64,
        "<i8" => SampleDtype::I64,
        "<f4" => SampleDtype::F32,
        "<f8" => SampleDtype::F64,
        other => panic!("unhandled numpy dtype {other}"),
    }
}

#[test]
fn series_axes_and_shape_match_tifffile() {
    for (name, expected) in manifest() {
        let file =
            TiffFile::open(&data_dir().join(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            file.series.axes,
            expected["series_axes"].as_str().unwrap(),
            "{name}: series axes"
        );
        let shape: Vec<usize> = expected["series_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(file.series.shape, shape, "{name}: series shape");
        assert_eq!(
            file.series.pages().len(),
            as_usize(&expected, "page_count"),
            "{name}: page count"
        );
        assert_eq!(
            file.series.levels.len(),
            expected["levels"].as_array().unwrap().len(),
            "{name}: level count"
        );
        assert_eq!(
            file.is_ome(),
            expected["is_ome"].as_bool().unwrap(),
            "{name}: is_ome"
        );
        assert_eq!(
            file.imagej().is_some(),
            expected["is_imagej"].as_bool().unwrap(),
            "{name}: is_imagej"
        );
    }
}

#[test]
fn page_geometry_matches_tifffile() {
    for (name, expected) in manifest() {
        let file = TiffFile::open(&data_dir().join(&name)).unwrap();
        for (level_index, level) in expected["levels"].as_array().unwrap().iter().enumerate() {
            let pages = level["pages"].as_array().unwrap();
            let actual = &file.series.levels[level_index].pages;
            assert_eq!(
                actual.len(),
                pages.len(),
                "{name}: level {level_index} pages"
            );
            for (page_index, page) in pages.iter().enumerate() {
                let got = &actual[page_index];
                let at = format!("{name}: level {level_index} page {page_index}");
                assert_eq!(got.image_width, as_usize(page, "imagewidth"), "{at}: width");
                assert_eq!(
                    got.image_length,
                    as_usize(page, "imagelength"),
                    "{at}: length"
                );
                assert_eq!(got.image_depth, as_usize(page, "imagedepth"), "{at}: depth");
                assert_eq!(
                    got.bits_per_sample,
                    as_usize(page, "bitspersample"),
                    "{at}: bits"
                );
                assert_eq!(
                    got.samples_per_pixel,
                    as_usize(page, "samplesperpixel"),
                    "{at}: samples"
                );
                assert_eq!(
                    u64::from(got.photometric),
                    page["photometric"].as_u64().unwrap(),
                    "{at}: photometric"
                );
                assert_eq!(
                    u64::from(got.planar_config),
                    page["planarconfig"].as_u64().unwrap(),
                    "{at}: planarconfig"
                );
                assert_eq!(
                    u64::from(got.compression),
                    page["compression"].as_u64().unwrap(),
                    "{at}: compression"
                );
                assert_eq!(
                    u64::from(got.predictor),
                    page["predictor"].as_u64().unwrap(),
                    "{at}: predictor"
                );
                assert_eq!(
                    got.is_tiled(),
                    page["is_tiled"].as_bool().unwrap(),
                    "{at}: tiled"
                );
                assert_eq!(
                    got.tile_width,
                    as_usize(page, "tilewidth"),
                    "{at}: tilewidth"
                );
                assert_eq!(
                    got.tile_length,
                    as_usize(page, "tilelength"),
                    "{at}: tilelength"
                );
                assert_eq!(
                    got.tile_depth,
                    as_usize(page, "tiledepth"),
                    "{at}: tiledepth"
                );
                assert_eq!(
                    got.dtype.numpy_str(),
                    page["dtype"].as_str().unwrap(),
                    "{at}: dtype"
                );
                let offsets: Vec<u64> = page["dataoffsets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                assert_eq!(got.data_offsets, offsets, "{at}: dataoffsets");
                let counts: Vec<u64> = page["databytecounts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                assert_eq!(got.data_byte_counts, counts, "{at}: databytecounts");
            }
        }
    }
}

#[test]
fn page_pixels_match_tifffile() {
    // Every codec must be exact, except JPEG: tifffile decodes it through libjpeg-turbo
    // and this reader through the pure-Rust decoder the `image` crate bundles, and the two
    // round the inverse DCT differently. The bound is measured, not assumed — see
    // `docs/divergences.md`, divergence 28.
    const JPEG_TOLERANCE: f64 = 4.0;

    for (name, expected) in manifest() {
        let mut file = TiffFile::open(&data_dir().join(&name)).unwrap();
        for (level_index, level) in expected["levels"].as_array().unwrap().iter().enumerate() {
            for (page_index, page) in level["pages"].as_array().unwrap().iter().enumerate() {
                let pixels = &page["pixels"];
                if pixels.get("error").is_some() {
                    continue;
                }
                let shape: Vec<usize> = pixels["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                let dtype = dtype_of(pixels["dtype"].as_str().unwrap());
                let bytes = decode_base64(pixels["bytes"].as_str().unwrap());

                let page_ref = file.series.levels[level_index].pages[page_index].clone();
                let raster = page_ref.read_full(&mut file.source).unwrap_or_else(|e| {
                    panic!("{name} level {level_index} page {page_index}: {e}")
                });
                let layout = expected_layout(&page_ref, &shape);
                assert_eq!(raster.dtype, dtype, "{name}: dtype");

                let tolerance = if page_ref.compression == 7 || page_ref.compression == 6 {
                    JPEG_TOLERANCE
                } else {
                    0.0
                };
                let at = format!("{name} level {level_index} page {page_index}");
                let mut worst = (0.0f64, (0usize, 0usize, 0usize), 0.0f64, 0.0f64);
                for y in 0..raster.height {
                    for x in 0..raster.width {
                        for sample in 0..raster.samples {
                            let got = raster.get(y, x, sample);
                            let want = recorded(&bytes, dtype, &shape, layout, y, x, sample);
                            let error = (got - want).abs();
                            if error > worst.0 {
                                worst = (error, (x, y, sample), got, want);
                            }
                        }
                    }
                }
                let (error, (x, y, sample), got, want) = worst;
                assert!(
                    error <= tolerance,
                    "{at}: worst deviation {error} at ({x}, {y}, {sample}) \
                     — got {got}, tifffile says {want} (tolerance {tolerance})"
                );
            }
        }
    }
}

#[test]
fn single_chunk_reads_agree_with_the_whole_page() {
    // The plugin's tiled path reads one chunk at a time, so a chunk has to carry the same
    // pixels the full-page read places at that position.
    for (name, _) in manifest() {
        let mut file = TiffFile::open(&data_dir().join(&name)).unwrap();
        let page = file.series.pages()[0].clone();
        if !page.is_tiled() {
            continue;
        }
        let full = page.read_full(&mut file.source).unwrap();
        let across = page.tiles_across();
        let per_plane = across * page.tiles_down();
        let planes = if page.separate_planes() {
            page.samples_per_pixel
        } else {
            1
        };
        for plane in 0..planes {
            for row in 0..page.tiles_down() {
                for column in 0..across {
                    let index = plane * per_plane + row * across + column;
                    let Some(tile) = page.read_chunk(&mut file.source, index).unwrap() else {
                        continue;
                    };
                    for y in 0..tile.height.min(full.height - row * page.tile_length) {
                        for x in 0..tile.width.min(full.width - column * page.tile_width) {
                            for sample in 0..tile.samples {
                                let got = tile.get(y, x, sample);
                                let want = full.get(
                                    row * page.tile_length + y,
                                    column * page.tile_width + x,
                                    plane + sample,
                                );
                                assert_eq!(
                                    got, want,
                                    "{name} tile {index} at ({x}, {y}, {sample})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn imagej_labels_and_luts_are_read() {
    let file = TiffFile::open(&data_dir().join("imagej_named_channels.tif")).unwrap();
    let metadata = file.imagej().expect("ImageJ metadata");
    assert_eq!(metadata.labels, vec!["DAPI", "FITC"]);
    assert_eq!(metadata.luts.len(), 2);
    assert_eq!(
        wsi_plugin_tifffile::imagej::lut_color(&metadata.luts[0]),
        Some((0, 200, 100))
    );
    assert_eq!(
        wsi_plugin_tifffile::imagej::lut_color(&metadata.luts[1]),
        Some((180, 0, 40))
    );
}

#[test]
fn imagej_spacing_is_read() {
    let file = TiffFile::open(&data_dir().join("rgb_zstack_imagej.tif")).unwrap();
    let metadata = file.imagej().expect("ImageJ metadata");
    assert_eq!(metadata.slices, 3);
    assert_eq!(
        wsi_plugin_tifffile::imagej::z_spacing_nm(metadata),
        Some(20_000.0)
    );
}

#[test]
fn ome_channels_and_pixel_sizes_are_read() {
    let file = TiffFile::open(&data_dir().join("named_channels.ome.tif")).unwrap();
    let pixels = file.ome_pixels().expect("OME pixels");
    let names: Vec<_> = pixels
        .channels
        .iter()
        .map(|c| c.name.clone().unwrap_or_default())
        .collect();
    assert_eq!(names, vec!["DAPI", "Cy5"]);
    assert_eq!(
        wsi_plugin_tifffile::ome::parse_color(pixels.channels[1].color.as_deref()),
        Some(-16_776_961)
    );

    let file = TiffFile::open(&data_dir().join("pixelsize_mm.ome.tif")).unwrap();
    let pixels = file.ome_pixels().unwrap();
    assert_eq!(
        wsi_plugin_tifffile::ome::length_to_nm(
            pixels.physical_size_x.as_deref(),
            pixels.physical_size_x_unit.as_deref()
        ),
        Some(170.0)
    );
}

#[test]
fn the_pyramid_fixture_exposes_its_sub_ifd_levels() {
    let file = TiffFile::open(&data_dir().join("pyramid.ome.tif")).unwrap();
    let widths: Vec<usize> = file
        .series
        .levels
        .iter()
        .map(|level| level.keyframe().image_width)
        .collect();
    assert_eq!(widths, vec![160, 80, 40]);
}
