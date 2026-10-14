//! Response models, mirroring the live pydantic models.
//!
//! Field *order* is significant: FastAPI serialises pydantic models in declaration order and
//! the parity harness compares response bodies. Every struct here therefore declares its
//! fields in the same order as the Python original, and optional fields are serialised as
//! `null` rather than skipped (pydantic's `model_dump` includes them).
//!
//! Source: `wsi_service/models/v3/slide.py`, `wsi_service/models/commons.py`,
//! `wsi_service/custom_models/{local_mapper_models,service_status}.py`,
//! `wsi_service/custom_models/old_v3/storage.py`.

use serde::{Deserialize, Serialize};

/// `wsi_service/models/v3/slide.py:16`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlideExtent {
    pub x: i64,
    pub y: i64,
    /// Number of Z-Stack layers.
    pub z: i64,
}

impl SlideExtent {
    pub fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }
}

/// `wsi_service/models/v3/slide.py:22`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlideColor {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: i32,
}

impl SlideColor {
    pub fn new(r: i32, g: i32, b: i32, a: i32) -> Self {
        Self { r, g, b, a }
    }
}

/// `wsi_service/models/v3/slide.py:29`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlidePixelSizeNm {
    pub x: f64,
    pub y: f64,
    /// Distance of layers in a Z-Stack in nm. Serialised as `null` when absent.
    pub z: Option<f64>,
}

impl SlidePixelSizeNm {
    pub fn xy(x: f64, y: f64) -> Self {
        Self { x, y, z: None }
    }

    pub fn xyz(x: f64, y: f64, z: Option<f64>) -> Self {
        Self { x, y, z }
    }
}

/// `wsi_service/models/v3/slide.py:44`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlideChannel {
    pub id: i32,
    pub name: String,
    pub color: SlideColor,
}

/// `wsi_service/models/v3/slide.py:50`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlideLevel {
    pub extent: SlideExtent,
    pub downsample_factor: f64,
}

/// `wsi_service/models/v3/slide.py:55`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlideInfo {
    pub id: String,
    pub channels: Option<Vec<SlideChannel>>,
    /// Colour depth (bitness) of each channel. Pydantic default is 8.
    pub channel_depth: Option<i32>,
    pub extent: SlideExtent,
    pub num_levels: i64,
    pub pixel_size_nm: SlidePixelSizeNm,
    pub tile_extent: SlideExtent,
    pub levels: Vec<SlideLevel>,
    pub format: String,
    pub raw_download: bool,
}

impl SlideInfo {
    /// Channel count, treating a missing channel list as zero — matching
    /// `validate_image_channels`, which indexes `len(slide_info.channels)` unconditionally.
    pub fn channel_count(&self) -> usize {
        self.channels.as_ref().map_or(0, Vec::len)
    }
}

/// The three RGB channels every brightfield plugin reports.
/// `wsi_service/utils/slide_utils.py:72` — note `a=0` for all three, as in Python.
pub fn rgb_channel_list() -> Vec<SlideChannel> {
    vec![
        SlideChannel {
            id: 0,
            name: "Red".into(),
            color: SlideColor::new(255, 0, 0, 0),
        },
        SlideChannel {
            id: 1,
            name: "Green".into(),
            color: SlideColor::new(0, 255, 0, 0),
        },
        SlideChannel {
            id: 2,
            name: "Blue".into(),
            color: SlideColor::new(0, 0, 255, 0),
        },
    ]
}

/// `wsi_service/utils/slide_utils.py:60`
pub fn original_levels(dimensions: &[(i64, i64)], downsamples: &[f64]) -> Vec<SlideLevel> {
    dimensions
        .iter()
        .zip(downsamples.iter())
        .map(|(&(w, h), &ds)| SlideLevel {
            extent: SlideExtent::new(w, h, 1),
            downsample_factor: ds,
        })
        .collect()
}

/// Effective read size for an edge tile. `wsi_service/utils/slide_utils.py:42`.
///
/// Tiles that start outside the level keep the full tile size so the caller can produce
/// padding; tiles that start inside are clamped to the remaining extent.
pub fn tile_read_size(
    level_extent: SlideExtent,
    tile_extent: SlideExtent,
    tile_x: i64,
    tile_y: i64,
) -> (i64, i64) {
    fn effective(start: i64, extent: i64, tile_size: i64) -> i64 {
        if start < 0 || start >= extent {
            tile_size
        } else {
            tile_size.min(extent - start)
        }
    }
    let start_x = tile_x * tile_extent.x;
    let start_y = tile_y * tile_extent.y;
    (
        effective(start_x, level_extent.x, tile_extent.x),
        effective(start_y, level_extent.y, tile_extent.y),
    )
}

// --- service status -------------------------------------------------------------------

/// `wsi_service/custom_models/service_status.py:8`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub priority: i64,
}

/// `wsi_service/models/commons.py:18` extended by `custom_models/service_status.py:14`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsiServiceStatus {
    pub status: String,
    pub version: String,
    pub message: Option<String>,
    pub plugins: Vec<PluginInfo>,
}

// --- storage / local mapper ------------------------------------------------------------

/// `wsi_service/custom_models/old_v3/storage.py:8`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageAddress {
    pub address: String,
    pub slide_id: String,
    pub main_address: bool,
    pub storage_address_id: String,
}

/// `wsi_service/custom_models/old_v3/storage.py:32`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlideStorage {
    pub slide_id: String,
    pub storage_type: String,
    pub storage_addresses: Vec<StorageAddress>,
}

/// `wsi_service/custom_models/local_mapper_models.py:8`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseLocalMapper {
    pub id: String,
    pub local_id: String,
    pub slides: Vec<String>,
}

/// `wsi_service/custom_models/local_mapper_models.py:14`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlideLocalMapper {
    pub id: String,
    pub local_id: String,
    pub slide_storage: SlideStorage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_fields_serialise_as_null() {
        let info = SlideInfo {
            id: "x".into(),
            channels: None,
            channel_depth: Some(8),
            extent: SlideExtent::new(1, 1, 1),
            num_levels: 1,
            pixel_size_nm: SlidePixelSizeNm::xy(-1.0, -1.0),
            tile_extent: SlideExtent::new(1, 1, 1),
            levels: vec![],
            format: "f".into(),
            raw_download: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains(r#""channels":null"#));
        assert!(json.contains(r#""z":null"#));
        // Declaration order must match pydantic's.
        assert!(json.starts_with(r#"{"id":"x","channels":null,"channel_depth":8,"extent":"#));
    }

    #[test]
    fn tile_read_size_clamps_inside_and_keeps_full_size_outside() {
        let level = SlideExtent::new(300, 300, 1);
        let tile = SlideExtent::new(256, 256, 1);
        // interior tile
        assert_eq!(tile_read_size(level, tile, 0, 0), (256, 256));
        // edge tile: clamped to the remainder
        assert_eq!(tile_read_size(level, tile, 1, 1), (44, 44));
        // fully outside: full tile size so the caller can pad
        assert_eq!(tile_read_size(level, tile, 2, 2), (256, 256));
    }
}
