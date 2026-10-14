//! Secondary storage-address collection, ported from
//! `wsi_service/utils/app_utils.py:267-320`.
//!
//! A slide is not always one file. DICOM slides are a directory of `*.dcm`; MIRAX slides
//! are a `.mrxs` file plus a sibling directory of the same stem. Both need every
//! constituent file listed so `/slides/download` can zip them.
//!
//! Paths are normalised to forward slashes throughout, matching `settings.data_dir`, which
//! `resolve_data_dir` normalises for exactly this reason: addresses are made relative by
//! stripping the data root as a *string* prefix.

use std::path::Path;

use uuid::Uuid;

use wsi_core::models::StorageAddress;

/// `local_mode_abs_file_path_to_relative` — `app_utils.py:267`.
pub fn abs_file_path_to_relative(filepath: &str, server_data_root: &str) -> String {
    let mut root = server_data_root.to_owned();
    if !root.ends_with('/') {
        root.push('/');
    }
    filepath
        .strip_prefix(&root)
        .map(|s| s.to_owned())
        .unwrap_or_else(|| filepath.to_owned())
}

pub fn normalise(path: &str) -> String {
    path.replace('\\', "/")
}

/// `local_mode_collect_secondary_files_v3` — `app_utils.py:275`.
///
/// The main address is appended **last**, after any secondaries, and carries the
/// caller-supplied `storage_address_id`; secondaries get fresh UUID4s, as in Python.
pub fn collect_secondary_files(
    main_address: &str,
    storage_address_id: &str,
    slide_id: &str,
    relative_to: &str,
) -> Vec<StorageAddress> {
    let mut root = normalise(relative_to);
    if !root.ends_with('/') {
        root.push('/');
    }
    let main_address = normalise(main_address);

    let (abspath, relpath) = if let Some(stripped) = main_address.strip_prefix(&root) {
        (main_address.clone(), stripped.to_owned())
    } else {
        (
            format!("{root}{}", main_address.trim_start_matches('/')),
            main_address.clone(),
        )
    };

    let abs = Path::new(&abspath);
    let mut result: Vec<StorageAddress> = if abs.is_dir() {
        // DICOM: every *.dcm in the folder.
        glob_sorted(&format!("{abspath}/*.dcm"))
            .into_iter()
            .map(|f| StorageAddress {
                address: abs_file_path_to_relative(&f, &root),
                slide_id: slide_id.to_owned(),
                main_address: false,
                storage_address_id: Uuid::new_v4().to_string(),
            })
            .collect()
    } else if abspath.ends_with(".mrxs") {
        // MIRAX: the sibling data directory named after the file stem.
        let parent = abs.parent().map(|p| normalise(&p.to_string_lossy()));
        let stem = abs.file_stem().map(|s| s.to_string_lossy().into_owned());
        match (parent, stem) {
            (Some(parent), Some(stem)) => glob_sorted(&format!("{parent}/{stem}/*"))
                .into_iter()
                .map(|f| StorageAddress {
                    address: abs_file_path_to_relative(&f, &root),
                    slide_id: slide_id.to_owned(),
                    main_address: false,
                    storage_address_id: Uuid::new_v4().to_string(),
                })
                .collect(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    result.push(StorageAddress {
        address: relpath,
        slide_id: slide_id.to_owned(),
        main_address: true,
        storage_address_id: storage_address_id.to_owned(),
    });
    result
}

/// `glob.glob` returns entries in an unspecified order; sorting makes the response
/// deterministic across runs and platforms.
fn glob_sorted(pattern: &str) -> Vec<String> {
    let mut matches: Vec<String> = glob::glob(pattern)
        .into_iter()
        .flatten()
        .flatten()
        .map(|p| normalise(&p.to_string_lossy()))
        .collect();
    matches.sort();
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_conversion_strips_the_data_root() {
        assert_eq!(
            abs_file_path_to_relative("/data/case/a.svs", "/data"),
            "case/a.svs"
        );
        assert_eq!(
            abs_file_path_to_relative("/data/case/a.svs", "/data/"),
            "case/a.svs"
        );
        // Not under the root: returned unchanged.
        assert_eq!(
            abs_file_path_to_relative("/elsewhere/a.svs", "/data"),
            "/elsewhere/a.svs"
        );
    }

    #[test]
    fn plain_file_gets_a_single_main_address() {
        let addresses = collect_secondary_files("/data/case/a.svs", "sid", "slide", "/data");
        assert_eq!(addresses.len(), 1);
        assert_eq!(addresses[0].address, "case/a.svs");
        assert!(addresses[0].main_address);
        assert_eq!(addresses[0].storage_address_id, "sid");
        assert_eq!(addresses[0].slide_id, "slide");
    }

    #[test]
    fn a_relative_main_address_is_kept_relative() {
        let addresses = collect_secondary_files("case/a.svs", "sid", "slide", "/data");
        assert_eq!(addresses.len(), 1);
        assert_eq!(addresses[0].address, "case/a.svs");
    }

    #[test]
    fn dicom_directory_expands_to_its_dcm_files_with_the_main_address_last() {
        let dir = std::env::temp_dir().join("wsi-secondary-dicom/case/slide");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("b.dcm"), b"x").unwrap();
        std::fs::write(dir.join("a.dcm"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();

        let root = normalise(
            &std::env::temp_dir()
                .join("wsi-secondary-dicom")
                .to_string_lossy(),
        );
        let main = format!("{root}/case/slide");
        let addresses = collect_secondary_files(&main, "sid", "slide", &root);

        assert_eq!(addresses.len(), 3, "two dcm files plus the main address");
        assert_eq!(addresses[0].address, "case/slide/a.dcm");
        assert_eq!(addresses[1].address, "case/slide/b.dcm");
        assert!(!addresses[0].main_address);
        assert!(!addresses[1].main_address);
        // Main address is appended last.
        assert!(addresses[2].main_address);
        assert_eq!(addresses[2].address, "case/slide");

        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("wsi-secondary-dicom"));
    }

    #[test]
    fn mirax_file_expands_to_its_sibling_directory() {
        let base = std::env::temp_dir().join("wsi-secondary-mrxs");
        let case = base.join("case");
        std::fs::create_dir_all(case.join("slide")).unwrap();
        std::fs::write(case.join("slide.mrxs"), b"x").unwrap();
        std::fs::write(case.join("slide/Index.dat"), b"x").unwrap();
        std::fs::write(case.join("slide/Data0000.dat"), b"x").unwrap();

        let root = normalise(&base.to_string_lossy());
        let main = format!("{root}/case/slide.mrxs");
        let addresses = collect_secondary_files(&main, "sid", "slide", &root);

        assert_eq!(addresses.len(), 3);
        assert_eq!(addresses[0].address, "case/slide/Data0000.dat");
        assert_eq!(addresses[1].address, "case/slide/Index.dat");
        assert_eq!(addresses[2].address, "case/slide.mrxs");
        assert!(addresses[2].main_address);

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn secondary_addresses_share_the_slide_id_but_get_fresh_ids() {
        let base = std::env::temp_dir().join("wsi-secondary-ids/case/slide");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("a.dcm"), b"x").unwrap();

        let root = normalise(
            &std::env::temp_dir()
                .join("wsi-secondary-ids")
                .to_string_lossy(),
        );
        let addresses =
            collect_secondary_files(&format!("{root}/case/slide"), "sid", "slide-1", &root);
        assert!(addresses.iter().all(|a| a.slide_id == "slide-1"));
        assert_ne!(addresses[0].storage_address_id, "sid");
        assert_eq!(addresses[1].storage_address_id, "sid");

        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("wsi-secondary-ids"));
    }
}
