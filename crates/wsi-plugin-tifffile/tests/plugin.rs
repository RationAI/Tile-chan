//! The plugin, against the same fixtures and the same expectations the Python suite uses.
//!
//! `wsi_service_plugin_tifffile/tests/test_plugin.py` builds its volumes with
//! `tifffile.imwrite` and then asserts concrete pixels; `tests/fixtures/generate.py` writes
//! those same files, so the numbers here are the Python suite's numbers rather than
//! whatever this implementation happens to produce.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use wsi_core::icc::IccIntent;
use wsi_core::models::SlideExtent;
use wsi_core::pixels::{Dtype, Pixels, PlanarData};
use wsi_core::plugins::WsiPlugin;
use wsi_core::settings::Settings;
use wsi_core::slide::{ReadOptions, Slide};
use wsi_plugin_tifffile::TiffFilePlugin;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn plugin() -> TiffFilePlugin {
    TiffFilePlugin::new(Arc::new(Settings::default()))
}

async fn open(name: &str) -> Arc<dyn Slide> {
    plugin()
        .open(&fixture(name))
        .await
        .unwrap_or_else(|e| panic!("{name}: {}", e.detail_text()))
}

fn planar(pixels: Pixels) -> PlanarData {
    match pixels {
        Pixels::Planar(data) => data,
        other => panic!("expected a channel stack, got {other:?}"),
    }
}

fn image(pixels: Pixels) -> image::RgbImage {
    match pixels {
        Pixels::Image(image) => image.to_rgb8(),
        other => panic!("expected an image, got {other:?}"),
    }
}

fn at(z: u32) -> ReadOptions {
    ReadOptions {
        z,
        ..Default::default()
    }
}

/// `_write_rgb_ome`'s reference image: `data[y][x][sample]`.
fn rgb_reference(y: usize, x: usize, sample: usize) -> u8 {
    let value = match sample {
        0 => y * 2 + x,
        1 => y + x * 3 + 40,
        _ => y * 5 + x * 7 + 90,
    };
    (value % 251) as u8
}

// --- OME: planar configuration, sparse metadata, the RGB fast path -----------------------

#[tokio::test]
async fn an_ome_rgb_region_matches_the_source_for_both_planar_configs() {
    for name in ["rgb_separate.ome.tif", "rgb_contig.ome.tif"] {
        let slide = open(name).await;
        // start_x/start_y are column/row; the reference is indexed row-first.
        let region = image(
            slide
                .get_region(0, 10, 20, 40, 24, ReadOptions::default())
                .await
                .unwrap(),
        );
        assert_eq!(region.dimensions(), (40, 24), "{name}");
        for y in 0..24usize {
            for x in 0..40usize {
                let pixel = region.get_pixel(x as u32, y as u32).0;
                for (sample, &value) in pixel.iter().enumerate() {
                    assert_eq!(
                        value,
                        rgb_reference(20 + y, 10 + x, sample),
                        "{name} at ({x}, {y}, {sample})"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn ome_separate_planes_are_not_collapsed_to_grayscale() {
    // Regression: PLANARCONFIG=SEPARATE used to read plane 0 only and broadcast it across
    // R/G/B, which yields a grey image and raises nothing.
    let slide = open("rgb_separate.ome.tif").await;
    let region = image(
        slide
            .get_region(0, 0, 0, 48, 48, ReadOptions::default())
            .await
            .unwrap(),
    );
    let mut red_equals_green = true;
    let mut green_equals_blue = true;
    for pixel in region.pixels() {
        red_equals_green &= pixel.0[0] == pixel.0[1];
        green_equals_blue &= pixel.0[1] == pixel.0[2];
    }
    assert!(!red_equals_green);
    assert!(!green_equals_blue);
}

#[tokio::test]
async fn ome_rgb_reports_rgb_channels_and_a_thumbnail_that_fits_the_box() {
    let slide = open("rgb_separate.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    let names: Vec<_> = info
        .channels
        .as_ref()
        .unwrap()
        .iter()
        .map(|channel| channel.name.clone())
        .collect();
    assert_eq!(names, vec!["Red", "Green", "Blue"]);
    assert_eq!(info.extent, SlideExtent::new(96, 64, 1));
    assert_eq!(info.tile_extent, SlideExtent::new(32, 32, 1));
    assert_eq!(info.format, "ome-tiff");

    // 96x64 into a 40x40 box: width is limiting, height follows the ratio.
    let thumbnail = image(slide.get_thumbnail(40, 40, None, false).await.unwrap());
    assert_eq!(thumbnail.dimensions(), (40, 27));
}

#[tokio::test]
async fn ome_channels_without_colour_or_name_fall_back_to_the_palette() {
    // `<Channel>` elements carrying neither Color nor Name used to raise
    // "int() argument must be ... not 'NoneType'" during open().
    let slide = open("fluorescence.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    let channels = info.channels.as_ref().unwrap();
    let names: Vec<_> = channels.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["Channel 0", "Channel 1", "Channel 2"]);
    let colours: std::collections::BTreeSet<_> = channels
        .iter()
        .map(|c| (c.color.r, c.color.g, c.color.b))
        .collect();
    assert_eq!(colours.len(), 3, "distinct palette colours, not all black");

    // data[c][y][x] == c * 10 + y * 3 + x
    let region = planar(
        slide
            .get_region(0, 1, 1, 3, 2, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(region.shape(), (3, 2, 3));
    for c in 0..3 {
        for y in 0..2 {
            for x in 0..3 {
                let expected = (c * 10 + (y + 1) * 3 + (x + 1)) as f64;
                assert_eq!(region.get_f64(c, y, x), expected, "at ({x}, {y}, {c})");
            }
        }
    }
}

#[tokio::test]
async fn an_interleaved_ome_channel_element_expands_to_the_plane_count() {
    // Bio-Formats writes RGB as a single <Channel SamplesPerPixel="3">; the reported
    // channel count must still match the planes a read returns.
    let slide = open("rgb_contig.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.channels.as_ref().unwrap().len(), 3);
}

#[tokio::test]
async fn named_ome_channels_carry_their_names_and_signed_colours() {
    let slide = open("named_channels.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    let channels = info.channels.as_ref().unwrap();
    let names: Vec<_> = channels.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["DAPI", "Cy5"]);
    // 16711935 is 0x00FF00FF; -16776961 is 0xFF0000FF.
    assert_eq!(
        (
            channels[0].color.r,
            channels[0].color.g,
            channels[0].color.b
        ),
        (0, 255, 0)
    );
    assert_eq!(
        (
            channels[1].color.r,
            channels[1].color.g,
            channels[1].color.b
        ),
        (255, 0, 0)
    );
    assert_eq!(info.channel_depth, Some(16));
}

#[tokio::test]
async fn ome_pixel_sizes_convert_through_their_units() {
    for (name, expected) in [
        ("pixelsize_mm.ome.tif", 170.0),
        ("pixelsize_um.ome.tif", 227.7),
        ("pixelsize_nm.ome.tif", 325.0),
        // The OME schema defaults the unit attribute to µm.
        ("pixelsize_default.ome.tif", 500.0),
    ] {
        let info = open(name).await.get_info().await.unwrap();
        assert!(
            (info.pixel_size_nm.x - expected).abs() < 1e-6,
            "{name}: {}",
            info.pixel_size_nm.x
        );
        assert!((info.pixel_size_nm.y - expected).abs() < 1e-6, "{name}");
    }
}

#[tokio::test]
async fn ome_without_a_physical_size_falls_back_to_one_micrometre() {
    let info = open("nopixelsize.ome.tif").await.get_info().await.unwrap();
    // tifffile writes no resolution tags without a PhysicalSize, so the generic default
    // applies.
    assert_eq!(info.pixel_size_nm.x, 1000.0);
    assert_eq!(info.pixel_size_nm.y, 1000.0);
}

#[tokio::test]
async fn padding_fits_a_signed_pixel_dtype() {
    // A region past the edge is padding-filled with settings.padding_color (255), which
    // does not fit an int8 page — numpy raises instead of wrapping, so Python clamps.
    let slide = open("signed.ome.tif").await;
    let region = planar(
        slide
            .get_region(0, 100, 100, 16, 16, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(region.shape(), (2, 16, 16));
    assert_eq!(region.dtype(), Dtype::I8);
    let (low, _) = region.min_max().unwrap();
    assert!(low >= -128.0);
}

#[tokio::test]
async fn an_untiled_ome_series_reports_a_nonzero_tile_extent_and_addresses_its_z_planes() {
    // Untiled pages report tilewidth/tilelength == 0; tile_extent must not be 0.
    let slide = open("untiled_czyx.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.tile_extent, SlideExtent::new(9, 7, 1));
    assert_eq!(info.extent.z, 3);
    assert_eq!(info.channels.as_ref().unwrap().len(), 2);

    // data[c][z][y][x] == c * 100 + z * 10 + y + x
    for z in 0..3usize {
        let tile = planar(slide.get_tile(0, 0, 0, at(z as u32)).await.unwrap());
        assert_eq!(tile.shape(), (2, 7, 9), "z {z}");
        for c in 0..2 {
            for y in 0..7 {
                for x in 0..9 {
                    let expected = (c * 100 + z * 10 + y + x) as f64;
                    assert_eq!(tile.get_f64(c, y, x), expected, "at ({x}, {y}, {c}) z {z}");
                }
            }
        }
    }
}

// --- generic and ImageJ -----------------------------------------------------------------

#[tokio::test]
async fn an_imagej_z_stack_reports_its_extent_and_reads_the_requested_z() {
    let slide = open("zstack_imagej.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.extent.z, 3);
    assert!(info.levels.iter().all(|level| level.extent.z == 3));
    assert_eq!(info.channels.as_ref().unwrap().len(), 2);
    assert_eq!(info.tile_extent, SlideExtent::new(5, 4, 1));
    assert_eq!(info.format, "tiff-multichannel");

    // data[z][c][y][x] == z * 20 + c + 1
    let region_z0 = planar(slide.get_region(0, 1, 1, 2, 2, at(0)).await.unwrap());
    let region_z2 = planar(slide.get_region(0, 1, 1, 2, 2, at(2)).await.unwrap());
    let tile_z1 = planar(slide.get_tile(0, 0, 0, at(1)).await.unwrap());
    assert_eq!(region_z0.shape(), (2, 2, 2));
    assert_eq!(region_z2.shape(), (2, 2, 2));
    assert_eq!(region_z0.get_f64(0, 0, 0), 1.0);
    assert_eq!(region_z0.get_f64(1, 0, 0), 2.0);
    assert_eq!(region_z2.get_f64(0, 0, 0), 41.0);
    assert_eq!(region_z2.get_f64(1, 0, 0), 42.0);
    assert_eq!(tile_z1.get_f64(0, 0, 0), 21.0);
    assert_eq!(tile_z1.get_f64(1, 0, 0), 22.0);
}

#[tokio::test]
async fn an_imagej_rgb_z_stack_reports_its_spacing_and_reads_the_requested_z() {
    let slide = open("rgb_zstack_imagej.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.extent.z, 3);
    assert!(info.levels.iter().all(|level| level.extent.z == 3));
    assert_eq!(info.channels.as_ref().unwrap().len(), 3);
    assert_eq!(info.pixel_size_nm.z, Some(20_000.0));
    assert_eq!(info.tile_extent, SlideExtent::new(5, 4, 1));

    // data[z][y][x][c] == z * 20 + c + 1. Three chunky samples with photometric RGB, but
    // `num_z > 1`, so this is *not* the RGB fast path — it stays a channel stack.
    let region_z0 = planar(slide.get_region(0, 1, 1, 2, 2, at(0)).await.unwrap());
    let region_z2 = planar(slide.get_region(0, 1, 1, 2, 2, at(2)).await.unwrap());
    assert_eq!(region_z0.shape(), (3, 2, 2));
    assert_eq!(region_z2.shape(), (3, 2, 2));
    for c in 0..3 {
        assert_eq!(region_z0.get_f64(c, 0, 0), (c + 1) as f64);
        assert_eq!(region_z2.get_f64(c, 0, 0), (40 + c + 1) as f64);
    }
}

#[tokio::test]
async fn a_channel_stack_without_z_still_reads_every_channel() {
    let slide = open("channels_only_imagej.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.extent.z, 1);
    assert_eq!(info.channels.as_ref().unwrap().len(), 3);
    assert_eq!(info.tile_extent, SlideExtent::new(5, 4, 1));

    // data[c][y][x] == c * 10 + 1
    let region = planar(slide.get_region(0, 0, 0, 2, 2, at(0)).await.unwrap());
    assert_eq!(region.shape(), (3, 2, 2));
    assert_eq!(region.get_f64(0, 0, 0), 1.0);
    assert_eq!(region.get_f64(1, 0, 0), 11.0);
    assert_eq!(region.get_f64(2, 0, 0), 21.0);
}

#[tokio::test]
async fn imagej_labels_name_the_channels_and_luts_colour_them() {
    let slide = open("imagej_named_channels.tif").await;
    let channels = slide.get_info().await.unwrap().channels.unwrap();
    let names: Vec<_> = channels.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["DAPI", "FITC"]);
    // The brightest entry of each LUT row.
    assert_eq!(
        (
            channels[0].color.r,
            channels[0].color.g,
            channels[0].color.b
        ),
        (0, 200, 100)
    );
    assert_eq!(
        (
            channels[1].color.r,
            channels[1].color.g,
            channels[1].color.b
        ),
        (180, 0, 40)
    );
}

#[tokio::test]
async fn a_pyramid_reports_every_level_and_its_downsample_factor() {
    let slide = open("pyramid.ome.tif").await;
    let info = slide.get_info().await.unwrap();
    assert_eq!(info.num_levels, 3);
    let extents: Vec<(i64, i64)> = info
        .levels
        .iter()
        .map(|level| (level.extent.x, level.extent.y))
        .collect();
    assert_eq!(extents, vec![(160, 128), (80, 64), (40, 32)]);
    let factors: Vec<f64> = info
        .levels
        .iter()
        .map(|level| level.downsample_factor)
        .collect();
    assert_eq!(factors, vec![1.0, 2.0, 4.0]);

    // A coarse level reads from its own SubIFD, so it must not return level-0 pixels.
    let coarse = image(
        slide
            .get_region(2, 0, 0, 8, 8, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(coarse.dimensions(), (8, 8));
    for y in 0..8usize {
        for x in 0..8usize {
            // The level was written as `data[::4, ::4]`.
            for (sample, &value) in coarse.get_pixel(x as u32, y as u32).0.iter().enumerate() {
                assert_eq!(
                    value,
                    rgb_reference(y * 4, x * 4, sample),
                    "at ({x}, {y}, {sample})"
                );
            }
        }
    }
}

#[tokio::test]
async fn every_codec_reads_the_same_channel_stack() {
    // data[c][y][x] == (c * 90 + y * 5 + x * 3) % 256, tiled 32x32 over 70x50.
    for name in [
        "codec_none.ome.tif",
        "codec_deflate.ome.tif",
        "codec_lzw.ome.tif",
        "codec_packbits.ome.tif",
        "codec_zstd.ome.tif",
    ] {
        let slide = open(name).await;
        let region = planar(
            slide
                .get_region(0, 30, 20, 8, 6, ReadOptions::default())
                .await
                .unwrap(),
        );
        assert_eq!(region.shape(), (3, 6, 8), "{name}");
        for c in 0..3 {
            for y in 0..6 {
                for x in 0..8 {
                    let expected = ((c * 90 + (y + 20) * 5 + (x + 30) * 3) % 256) as f64;
                    assert_eq!(
                        region.get_f64(c, y, x),
                        expected,
                        "{name} at ({x}, {y}, {c})"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn the_horizontal_predictor_is_undone_on_sixteen_bit_tiles() {
    // data[c][y][x] == (c * 8000 + y * 300 + x * 17) % 65536
    let slide = open("predictor_deflate.ome.tif").await;
    let region = planar(
        slide
            .get_region(0, 33, 17, 6, 4, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(region.dtype(), Dtype::U16);
    for c in 0..2 {
        for y in 0..4 {
            for x in 0..6 {
                let expected = ((c * 8000 + (y + 17) * 300 + (x + 33) * 17) % 65536) as f64;
                assert_eq!(region.get_f64(c, y, x), expected, "at ({x}, {y}, {c})");
            }
        }
    }
}

#[tokio::test]
async fn a_big_endian_page_reads_in_native_order() {
    // data[c][y][x] == c * 4096 + y * 64 + x, uint16, byteorder '>'.
    let slide = open("bigendian.ome.tif").await;
    let region = planar(
        slide
            .get_region(0, 0, 0, 8, 6, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(region.dtype(), Dtype::U16);
    for c in 0..2 {
        for y in 0..6 {
            for x in 0..8 {
                let expected = (c * 4096 + y * 64 + x) as f64;
                assert_eq!(region.get_f64(c, y, x), expected, "at ({x}, {y}, {c})");
            }
        }
    }
}

#[tokio::test]
async fn a_bigtiff_separate_rgb_page_reads_through_the_fast_path() {
    let slide = open("bigtiff.ome.tif").await;
    let region = image(
        slide
            .get_region(0, 4, 8, 16, 12, ReadOptions::default())
            .await
            .unwrap(),
    );
    assert_eq!(region.dimensions(), (16, 12));
    for y in 0..12usize {
        for x in 0..16usize {
            for (sample, &value) in region.get_pixel(x as u32, y as u32).0.iter().enumerate() {
                assert_eq!(
                    value,
                    rgb_reference(8 + y, 4 + x, sample),
                    "at ({x}, {y}, {sample})"
                );
            }
        }
    }
}

// --- the endpoints that are defects ------------------------------------------------------

#[tokio::test]
async fn the_icc_endpoint_is_always_a_500() {
    // `tags.get("ICCProfile")` hands back a TiffTag, which neither the profile endpoint nor
    // the colour transform can use. See docs/divergences.md, divergences 30 and 31.
    let slide = open("fluorescence.ome.tif").await;
    let error = slide.get_icc_profile().await.unwrap_err();
    assert_eq!(error.status, http::StatusCode::INTERNAL_SERVER_ERROR);
    assert!(error.unhandled);
}

#[tokio::test]
async fn an_icc_intent_is_ignored_unless_it_is_strict() {
    let slide = open("fluorescence.ome.tif").await;
    let lenient = ReadOptions {
        icc_profile_intent: Some(IccIntent::Perceptual),
        ..Default::default()
    };
    let plain = planar(
        slide
            .get_region(0, 0, 0, 4, 4, ReadOptions::default())
            .await
            .unwrap(),
    );
    let with_intent = planar(slide.get_region(0, 0, 0, 4, 4, lenient).await.unwrap());
    assert_eq!(plain.to_f64_vec(), with_intent.to_f64_vec());

    let strict = ReadOptions {
        icc_profile_intent: Some(IccIntent::Perceptual),
        icc_profile_strict: true,
        ..Default::default()
    };
    // This fixture has no `ICCProfile` tag, so `tags.get("ICCProfile")` is `None` and
    // `process_pil_image` takes its `elif strict` branch: a 412, not the 500 that a slide
    // *with* the tag gets when `buildTransform` raises. See divergence 31.
    let error = slide.get_region(0, 0, 0, 4, 4, strict).await.unwrap_err();
    assert_eq!(error.status, http::StatusCode::PRECONDITION_FAILED);
    assert_eq!(error.detail_text(), "Error: ICC Profile not available.");
}

#[tokio::test]
async fn label_and_macro_are_always_404() {
    let slide = open("fluorescence.ome.tif").await;
    assert_eq!(
        slide.get_label().await.unwrap_err().status,
        http::StatusCode::NOT_FOUND
    );
    assert_eq!(
        slide.get_macro(None, false).await.unwrap_err().status,
        http::StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_multichannel_thumbnail_comes_back_sixteen_bit() {
    // `img_as_uint(transform.resize(...))` — not Pillow, and always 16-bit.
    let slide = open("codec_none.ome.tif").await;
    let thumbnail = planar(slide.get_thumbnail(20, 20, None, false).await.unwrap());
    assert_eq!(thumbnail.dtype(), Dtype::U16);
    let (channels, height, width) = thumbnail.shape();
    assert_eq!(channels, 3);
    // 70x50 into a 20x20 box: width limits, so the height follows the ratio.
    assert_eq!((width, height), (20, 14));
}

#[tokio::test]
async fn a_closed_slide_reports_the_python_failure() {
    let slide = open("fluorescence.ome.tif").await;
    slide.close().await.unwrap();
    let error = slide
        .get_region(0, 0, 0, 2, 2, ReadOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.status, http::StatusCode::INTERNAL_SERVER_ERROR);
}
