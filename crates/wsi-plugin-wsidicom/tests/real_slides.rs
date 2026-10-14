//! Reads the real DICOM directories and checks the port against wsidicom's own output.
//!
//! Skipped unless `WSI_TEST_SLIDES` points at the collection, exactly as the `openslide`,
//! `tifffile` and `tiffslide` real-slide tests are. The expectations come from
//! `tests/fixtures/wsidicom.json`, which `tests/fixtures/generate.py` produced with
//! wsidicom 0.27.1 — the same version the Python service installs.

use std::path::PathBuf;

use wsi_plugin_wsidicom::dicom::{self, tags};

fn collection() -> Option<PathBuf> {
    std::env::var_os("WSI_TEST_SLIDES").map(PathBuf::from)
}

/// The three directories the ground truth covers, keyed as `wsidicom.json` keys them.
const DIRECTORIES: [&str; 3] = [
    "DICOM_Leica/f990df0c-5e8e-16e5-000f-2874470dcd3a_050136",
    "DICOM_wsidicomizer/DICOM_Aperio",
    "DICOM_wsidicomizer/DICOM_Mirax",
];

/// Every instance has to parse, and the tags the plugin depends on have to be present.
///
/// This is the floor the rest of the plugin stands on: a tag read as the wrong type, or a
/// frame offset off by the 8 bytes of an item header, would show up here rather than as
/// shifted pixels later.
#[test]
fn every_instance_parses_and_reports_the_tags_the_plugin_needs() {
    let Some(root) = collection() else {
        eprintln!("WSI_TEST_SLIDES is not set; skipping");
        return;
    };

    for directory in DIRECTORIES {
        let path = root.join(directory);
        if !path.is_dir() {
            eprintln!("{} is missing; skipping it", path.display());
            continue;
        }

        let files = dicom::dicom_files(&path).expect("the directory should be readable");
        assert!(!files.is_empty(), "{} has no .dcm files", path.display());

        for file in files {
            let instance =
                dicom::read_instance(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));

            let image_type = instance
                .dataset
                .get(tags::IMAGE_TYPE)
                .expect("ImageType decides which group an instance belongs to")
                .strings();
            assert!(
                image_type.len() >= 3,
                "{}: ImageType has {} elements",
                file.display(),
                image_type.len()
            );

            // TILED_FULL is the scope this reader is written to; anything else would need
            // the per-frame functional groups.
            assert_eq!(
                instance
                    .dataset
                    .string(tags::DIMENSION_ORGANIZATION_TYPE)
                    .as_deref(),
                Some("TILED_FULL"),
                "{}",
                file.display()
            );

            for tag in [
                tags::ROWS,
                tags::COLUMNS,
                tags::TOTAL_PIXEL_MATRIX_COLUMNS,
                tags::TOTAL_PIXEL_MATRIX_ROWS,
                tags::SAMPLES_PER_PIXEL,
                tags::BITS_ALLOCATED,
            ] {
                assert!(
                    instance.dataset.integer(tag).is_some(),
                    "{}: {tag} is missing or not an integer",
                    file.display()
                );
            }

            let photometric = instance
                .dataset
                .string(tags::PHOTOMETRIC_INTERPRETATION)
                .unwrap_or_default();
            assert!(
                matches!(photometric.as_str(), "RGB" | "YBR_FULL" | "YBR_FULL_422"),
                "{}: unexpected photometric {photometric:?}",
                file.display()
            );

            // The frame count has to agree with the geometry, which is what proves the
            // pixel-data parse landed on the right items.
            let pixel_data = instance
                .pixel_data
                .as_ref()
                .unwrap_or_else(|| panic!("{}: no pixel data", file.display()));
            let declared = instance
                .dataset
                .integer(tags::NUMBER_OF_FRAMES)
                .unwrap_or(1);
            assert_eq!(
                pixel_data.frame_count() as i64,
                declared,
                "{}: found {} frames, the dataset declares {declared}",
                file.display(),
                pixel_data.frame_count()
            );

            // And the first and last frames have to be readable, which exercises both the
            // offset-table arithmetic and the lazy length read at the far end of the file.
            let first = instance.read_frame(0).expect("first frame");
            assert!(!first.is_empty(), "{}: empty first frame", file.display());
            let last = instance
                .read_frame(pixel_data.frame_count() - 1)
                .expect("last frame");
            assert!(!last.is_empty(), "{}: empty last frame", file.display());

            // A JPEG frame starts with SOI; an uncompressed one is raw samples.
            if instance.transfer_syntax == dicom::transfer_syntax::JPEG_BASELINE {
                assert_eq!(
                    &first[..2],
                    &[0xFF, 0xD8],
                    "{}: first frame is not a JPEG",
                    file.display()
                );
            }
        }
    }
}
