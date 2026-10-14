//! The image pipeline: numeric conversion, edge padding, encoding and validation.
//!
//! Ported from `wsi_service/utils/image_utils.py` and `wsi_service/utils/app_utils.py`.

pub mod convert;
pub mod padding;
pub mod resample;
pub mod response;
pub mod tiffout;

pub use convert::{
    int_to_rgba_array, mask_rgb_channels, planar_to_image, requested_channels_as_array,
    requested_channels_as_rgb_array, rgba_array_to_int, rgba_to_rgb_with_background_color,
    scale_rgb_image_by_color, PilMode,
};
pub use padding::{
    check_complete_region_overlap, check_complete_tile_overlap, get_extended_region,
    get_extended_tile, region_intersection, tile_overlap,
};
pub use resample::{
    preserve_aspect_ratio, reduce_bands, resize as pillow_resize, thumbnail as pillow_thumbnail,
    Filter, DEFAULT_REDUCING_GAP,
};
pub use response::{
    is_passthrough_format, make_image_response, make_passthrough_response, make_response,
    make_tif_response, normalize_image_format, passthrough_media_type, process_image_region,
    process_image_region_raw, raster_media_type, validate_hex_color_string,
    validate_image_channels, validate_image_level, validate_image_request, validate_image_size,
    validate_image_z, EncodedResponse,
};
pub use tiffout::encode_planar_tiff;
