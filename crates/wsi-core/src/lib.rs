//! Core of the Rust WSI-Service: settings, models, the plugin-facing slide interface, the
//! image pipeline, the plugin registry and the slide manager.
//!
//! Every module documents the Python file it was ported from. Where the original has
//! surprising behaviour that clients can observe, the behaviour is reproduced and the
//! reason recorded — see `rust/docs/divergences.md` for the full list, including the two
//! places where the port deliberately differs.

pub mod error;
pub mod icc;
pub mod image;
pub mod models;
pub mod pixels;
pub mod plugins;
pub mod settings;
pub mod slide;
pub mod slide_manager;
pub mod storage;

pub use error::{ApiError, Result};
pub use icc::{IccIntent, IccProfileCache};
pub use models::{
    CaseLocalMapper, PluginInfo, SlideChannel, SlideColor, SlideExtent, SlideInfo, SlideLevel,
    SlideLocalMapper, SlidePixelSizeNm, SlideStorage, StorageAddress, WsiServiceStatus,
};
pub use pixels::{Dtype, Pixels, PlanarData};
pub use settings::Settings;
pub use slide::{ReadOptions, SharedSlide, Slide};
