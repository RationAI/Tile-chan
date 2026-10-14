//! A low-level TIFF reader, standing in for `tifffile.TiffFile`.
//!
//! The plugin needs per-chunk reads from arbitrary pages — including pages reached only
//! through SubIFDs — and it needs `series[0]` to be split into levels the way tifffile
//! splits it. No Rust crate offers both, so this reads directories itself
//! ([`ifd`]), decodes chunks itself ([`codec`]), and reconstructs the series
//! ([`series`]).

pub mod codec;
pub mod ifd;
pub mod page;
pub mod series;

use std::path::Path;

use crate::imagej::ImageJMetadata;
use crate::ome::OmePixels;

use ifd::{Result, TiffSource};
use series::{Flavour, Series};

/// An open TIFF file: the handle, the metadata dialect, and `series[0]`.
pub struct TiffFile {
    pub source: TiffSource,
    pub flavour: Flavour,
    pub series: Series,
}

impl TiffFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut source = TiffSource::open(path)?;
        let (flavour, series) = series::build(&mut source)?;
        Ok(Self {
            source,
            flavour,
            series,
        })
    }

    /// `tif.is_ome`
    pub fn is_ome(&self) -> bool {
        matches!(self.flavour, Flavour::Ome(_))
    }

    /// `tif.ome_metadata` — the first page's `ImageDescription`, which is where the
    /// OME-XML lives.
    pub fn ome_metadata(&self) -> Option<&str> {
        self.is_ome()
            .then(|| self.series.keyframe().description.as_str())
    }

    pub fn ome_pixels(&self) -> Option<&OmePixels> {
        match &self.flavour {
            Flavour::Ome(pixels) => Some(pixels),
            _ => None,
        }
    }

    /// `tif.imagej_metadata`
    pub fn imagej(&self) -> Option<&ImageJMetadata> {
        match &self.flavour {
            Flavour::ImageJ(metadata) => Some(metadata),
            _ => None,
        }
    }
}
