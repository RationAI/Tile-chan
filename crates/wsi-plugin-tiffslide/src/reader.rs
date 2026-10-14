//! A `tiffslide.TiffSlide` equivalent: the interface the plugin is written against.
//!
//! `tiffslide` is itself a Python reimplementation of openslide's interface over
//! `tifffile`, and the plugin reads *its* decisions — the vendor it detects, the `mpp` it
//! synthesises, how it groups pages into a pyramid, which series become associated images,
//! and what `read_region` returns. None of that follows from the TIFF specification, so the
//! expectations here come from a recording of the real thing; see
//! `tests/fixtures/generate.py`.
//!
//! **Hamamatsu NDPI is refused.** An NDPI page is a single strip holding one JPEG larger
//! than the format allows, and `tifffile` reads sub-rectangles of it by entering the
//! entropy-coded stream at a restart marker listed in the private `NDPI_MCU_STARTS` tag —
//! which needs a JPEG decoder that can be told where to start and what DC predictor to
//! start with. `OS-1.ndpi` reports 414 720 such "tiles" over one strip, and the tile
//! geometry tiffslide advertises (4096x8, halving per level) is synthesised from that same
//! tag rather than read from the file.
//!
//! So `open` refuses an NDPI rather than serving a slide whose every read would fail and
//! whose `tile_extent` would be zero. `openslide` sits at priority 10 and serves NDPI
//! pixel-exactly, so the default configuration is unaffected. See
//! `docs/divergences.md`, divergence 33.

use std::path::Path;

use wsi_core::error::{ApiError, Result};
use wsi_plugin_tifffile::tiff::ifd::{tag, Ifd, TiffSource};
use wsi_plugin_tifffile::tiff::page::{Page, Raster};

/// Private Hamamatsu tags.
mod ndpi {
    /// `NDPI_FORMAT_FLAG` — present on every page of an NDPI.
    pub const FORMAT_FLAG: u16 = 65420;
    /// `SourceLens` — the magnification, or a negative marker for a non-pyramid page.
    pub const SOURCE_LENS: u16 = 65421;
}

const TAG_MODEL: u16 = 272;
const TAG_SOFTWARE: u16 = 305;
const TAG_XMP: u16 = 700;

/// `_PropertyParser.detect_format`'s result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Aperio,
    Leica,
    Ventana,
    Hamamatsu,
    PhilipsTiff,
    GenericTiff,
}

impl Format {
    /// The string `detect_format` returns, which the plugin reports as the slide format.
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Aperio => "aperio",
            Format::Leica => "leica",
            Format::Ventana => "ventana",
            Format::Hamamatsu => "hamamatsu",
            Format::PhilipsTiff => "philips_tiff",
            Format::GenericTiff => "generic-tiff",
        }
    }

    /// The vendor `properties["tiffslide.vendor"]` ends up holding.
    ///
    /// Only Aperio, Hamamatsu, Leica and Philips set one. `parse_ventana` delegates to
    /// `parse_generic_tiff` and never sets a vendor, which is why a Ventana slide passes
    /// the plugin's `supported_vendors` check and a Leica one does not.
    ///
    /// Philips' real value is whatever `DICOM_MANUFACTURER` says in its XML; only its
    /// absence from `supported_vendors` matters, so a fixed marker stands in for it.
    pub fn vendor(self) -> Option<&'static str> {
        match self {
            Format::Aperio => Some("aperio"),
            Format::Hamamatsu => Some("hamamatsu"),
            Format::Leica => Some("leica"),
            // `parse_philips_tiff` reads DICOM_MANUFACTURER out of the XML; whatever it
            // is, it is not in `supported_vendors`.
            Format::PhilipsTiff => Some("philips"),
            Format::Ventana | Format::GenericTiff => None,
        }
    }
}

/// An associated image: one non-pyramid series, named the way tifffile names it.
#[derive(Debug, Clone)]
pub struct Associated {
    /// Lower-cased, because `_LazyAssociatedImagesDict` keys on `series.name.lower()`.
    pub name: String,
    pub page: Page,
}

/// What `properties` holds, restricted to what the plugin reads.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    pub vendor: Option<String>,
    pub mpp_x: Option<f64>,
    pub mpp_y: Option<f64>,
    pub objective_power: Option<f64>,
    pub comment: Option<String>,
    /// `tiffslide.level[n].tile-{width,height}`, one entry per level.
    pub level_tiles: Vec<(i64, i64)>,
}

pub struct TiffSlideReader {
    source: TiffSource,
    pub format: Format,
    /// One page per pyramid level, finest first.
    pub levels: Vec<Page>,
    pub associated: Vec<Associated>,
    pub properties: Properties,
    pub icc_profile: Option<Vec<u8>>,
}

impl TiffSlideReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut source = TiffSource::open(path)
            .map_err(|e| ApiError::internal(format!("TiffFileError: {e}")))?;
        let chain = source
            .read_chain()
            .map_err(|e| ApiError::internal(format!("TiffFileError: {e}")))?;
        let order = source.order;

        let mut pages = Vec::with_capacity(chain.len());
        for (index, ifd) in chain.into_iter().enumerate() {
            match Page::from_ifd(ifd, index, order) {
                Ok(page) => pages.push(page),
                // A page this reader cannot describe is not a reason to reject the file;
                // tifffile skips it too.
                Err(_) => continue,
            }
        }
        if pages.is_empty() {
            return Err(ApiError::internal(
                "TiffFileError: no readable image file directory",
            ));
        }

        let format = detect_format(&pages);
        if format == Format::Hamamatsu {
            return Err(ApiError::internal(
                "TiffFileError: Hamamatsu NDPI is not supported by this reader; \
                 each page is one restart-marker JPEG and needs MCU-level entry. \
                 Use the openslide plugin.",
            ));
        }
        let (levels, associated) = split_series(format, &pages);
        if levels.is_empty() {
            return Err(ApiError::internal(
                "TiffFileError: no pyramid level in series 0",
            ));
        }

        let properties = collect_properties(format, &pages[0], &levels);
        let icc_profile = pages[0].icc_profile().map(|bytes| bytes.to_vec());

        Ok(Self {
            source,
            format,
            levels,
            associated,
            properties,
            icc_profile,
        })
    }

    /// `slide.dimensions` — level 0's width and height.
    pub fn dimensions(&self) -> (i64, i64) {
        let level = &self.levels[0];
        (level.image_width as i64, level.image_length as i64)
    }

    /// `slide.level_dimensions`
    pub fn level_dimensions(&self) -> Vec<(i64, i64)> {
        self.levels
            .iter()
            .map(|level| (level.image_width as i64, level.image_length as i64))
            .collect()
    }

    /// `slide.level_downsamples` — **the average of the two ratios**, not the width ratio.
    ///
    /// ```python
    /// tuple(((w0 / w) + (h0 / h)) / 2.0 for w, h in self.level_dimensions)
    /// ```
    pub fn level_downsamples(&self) -> Vec<f64> {
        let (w0, h0) = self.dimensions();
        self.level_dimensions()
            .into_iter()
            .map(|(w, h)| ((w0 as f64 / w as f64) + (h0 as f64 / h as f64)) / 2.0)
            .collect()
    }

    /// `read_region(location, level, size)`.
    ///
    /// `location` is in **level 0** pixels. The rectangle is mapped into the level by
    /// integer division, clipped to it, and whatever falls outside is padded with zeros —
    /// black, not the caller's padding colour.
    pub fn read_region(
        &mut self,
        location: (i64, i64),
        level: usize,
        size: (u32, u32),
    ) -> Result<image::RgbImage> {
        let Some(page) = self.levels.get(level).cloned() else {
            return Err(ApiError::internal(format!(
                "IndexError: level={level} out of range"
            )));
        };
        let (level_w, level_h) = (page.image_width as i64, page.image_length as i64);
        let (request_w, request_h) = (size.0 as i64, size.1 as i64);

        let downsample = self.level_downsamples()[level];
        // `rx0 = int(base_x / level_ds)` — truncation, and the division is in float.
        let rx0 = (location.0 as f64 / downsample) as i64;
        let ry0 = (location.1 as f64 / downsample) as i64;
        let rx1 = rx0 + request_w;
        let ry1 = ry0 + request_h;

        let in_bounds = rx0 >= 0 && rx1 <= level_w && ry0 >= 0 && ry1 <= level_h;
        let pad_x0 = clip(-rx0, 0, request_w);
        let pad_y0 = clip(-ry0, 0, request_h);

        let (cx0, cx1) = (clip(rx0, 0, level_w), clip(rx1, 0, level_w));
        let (cy0, cy1) = (clip(ry0, 0, level_h), clip(ry1, 0, level_h));

        let mut canvas = image::RgbImage::new(size.0, size.1);
        if cx0 >= cx1 || cy0 >= cy1 {
            // Wholly outside: all zeros, which is what `np.pad` leaves behind.
            return Ok(canvas);
        }

        let raster = read_page_region(&page, &mut self.source, cx0, cy0, cx1, cy1)?;
        let (offset_x, offset_y) = if in_bounds { (0, 0) } else { (pad_x0, pad_y0) };
        for y in 0..raster.height {
            for x in 0..raster.width {
                let (dx, dy) = (offset_x as usize + x, offset_y as usize + y);
                if dx >= size.0 as usize || dy >= size.1 as usize {
                    continue;
                }
                let sample = |index: usize| -> u8 {
                    if index < raster.samples {
                        raster.get(y, x, index) as u8
                    } else {
                        // A greyscale page broadcasts its single sample, as
                        // `Image.fromarray` on a `YX` series does.
                        raster.get(y, x, 0) as u8
                    }
                };
                canvas.put_pixel(
                    dx as u32,
                    dy as u32,
                    image::Rgb([sample(0), sample(1), sample(2)]),
                );
            }
        }
        Ok(canvas)
    }

    /// One associated image, decoded whole.
    pub fn associated_image(&mut self, name: &str) -> Option<Result<image::RgbImage>> {
        let page = self
            .associated
            .iter()
            .find(|item| item.name == name)
            .map(|item| item.page.clone())?;
        Some(decode_whole_page(&page, &mut self.source))
    }

    pub fn associated_names(&self) -> Vec<&str> {
        self.associated
            .iter()
            .map(|item| item.name.as_str())
            .collect()
    }
}

fn clip(value: i64, low: i64, high: i64) -> i64 {
    value.max(low).min(high)
}

/// Reads `[y0, y1) x [x0, x1)` of a page as an interleaved raster.
fn read_page_region(
    page: &Page,
    source: &mut TiffSource,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
) -> Result<Raster> {
    let (width, height) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let mut out = Raster::filled(height, width, page.samples_per_pixel, page.dtype, 0.0);
    if !page.is_tiled() {
        // A striped page is small enough to read whole in every format this serves.
        let whole = page
            .read_full(source)
            .map_err(|e| ApiError::internal(format!("TiffFileError: {e}")))?;
        out.blit(&whole, 0, 0, y0 as usize, x0 as usize, height, width, 0);
        return Ok(out);
    }

    let (tile_w, tile_h) = (page.tile_width as i64, page.tile_length as i64);
    let across = page.tiles_across() as i64;
    let per_plane = across * page.tiles_down() as i64;
    let planes = if page.separate_planes() {
        page.samples_per_pixel
    } else {
        1
    };

    let first_row = y0.div_euclid(tile_h);
    let first_column = x0.div_euclid(tile_w);
    let last_row = (y1 - 1).div_euclid(tile_h);
    let last_column = (x1 - 1).div_euclid(tile_w);

    for plane in 0..planes {
        for row in first_row..=last_row {
            for column in first_column..=last_column {
                let index = plane as i64 * per_plane + row * across + column;
                if index < 0 {
                    continue;
                }
                let Some(tile) = page
                    .read_chunk(source, index as usize)
                    .map_err(|e| ApiError::internal(format!("TiffFileError: {e}")))?
                else {
                    continue;
                };
                // Where this tile starts inside the requested rectangle, and where inside
                // the tile the rectangle starts.
                let tile_x = column * tile_w;
                let tile_y = row * tile_h;
                let src_x = (x0 - tile_x).max(0) as usize;
                let src_y = (y0 - tile_y).max(0) as usize;
                let dst_x = (tile_x - x0).max(0) as usize;
                let dst_y = (tile_y - y0).max(0) as usize;
                out.blit(
                    &tile,
                    dst_y,
                    dst_x,
                    src_y,
                    src_x,
                    height.saturating_sub(dst_y),
                    width.saturating_sub(dst_x),
                    plane,
                );
            }
        }
    }
    Ok(out)
}

fn decode_whole_page(page: &Page, source: &mut TiffSource) -> Result<image::RgbImage> {
    let raster = page
        .read_full(source)
        .map_err(|e| ApiError::internal(format!("TiffFileError: {e}")))?;
    let mut out = image::RgbImage::new(raster.width as u32, raster.height as u32);
    for y in 0..raster.height {
        for x in 0..raster.width {
            let sample = |index: usize| -> u8 {
                let index = if index < raster.samples { index } else { 0 };
                raster.get(y, x, index) as u8
            };
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([sample(0), sample(1), sample(2)]),
            );
        }
    }
    Ok(out)
}

/// `_PropertyParser.detect_format`, using tifffile's own detectors in tifffile's order.
pub fn detect_format(pages: &[Page]) -> Format {
    let first = &pages[0];
    if first.is_tiled() && first.description.starts_with("Aperio ") {
        return Format::Aperio;
    }
    if is_scn(first) {
        return Format::Leica;
    }
    if is_bif(first) {
        return Format::Ventana;
    }
    if is_ndpi(first) {
        return Format::Hamamatsu;
    }
    if is_philips(first) {
        return Format::PhilipsTiff;
    }
    Format::GenericTiff
}

/// `TiffPage.is_scn`: page 0's description ends with `</scn>`.
fn is_scn(page: &Page) -> bool {
    page.index == 0 && page.description.trim_end().ends_with("</scn>")
}

/// `TiffPage.is_bif`: an XMP tag plus Ventana's signature in `Software`, or one of three
/// fixed descriptions.
fn is_bif(page: &Page) -> bool {
    if !page.ifd.entries.contains_key(&TAG_XMP) {
        return false;
    }
    let software = page.ifd.string(TAG_SOFTWARE).unwrap_or_default();
    software.contains("Ventana")
        || software.starts_with("ScanOutputManager")
        || matches!(
            page.description.as_str(),
            "Label Image" | "Label_Image" | "Probability_Image"
        )
}

/// `TiffPage.is_ndpi`: the format flag and a `Model` tag.
fn is_ndpi(page: &Page) -> bool {
    page.ifd.entries.contains_key(&ndpi::FORMAT_FLAG) && page.ifd.entries.contains_key(&TAG_MODEL)
}

/// `TiffPage.is_philips`: `Software` starting `Philips DP` and a `</DataObject>` description.
fn is_philips(page: &Page) -> bool {
    page.ifd
        .string(TAG_SOFTWARE)
        .is_some_and(|software| software.starts_with("Philips DP"))
        && page.description.trim_end().ends_with("</DataObject>")
}

/// Splits the pages into the pyramid and the associated images, the way tifffile's series
/// builders do for each format.
fn split_series(format: Format, pages: &[Page]) -> (Vec<Page>, Vec<Associated>) {
    match format {
        Format::Aperio => split_aperio(pages),
        Format::Hamamatsu => split_ndpi(pages),
        _ => split_by_subfile_type(pages),
    }
}

/// Aperio SVS: page 0 is the baseline, further **tiled** pages are its levels, and the
/// untiled ones are the associated images, named by the second line of their description —
/// `label …`, `macro …`, or a `W×H -> W×H` line for the thumbnail.
fn split_aperio(pages: &[Page]) -> (Vec<Page>, Vec<Associated>) {
    let mut levels = vec![pages[0].clone()];
    let mut associated = Vec::new();
    for page in &pages[1..] {
        if page.is_tiled() {
            levels.push(page.clone());
            continue;
        }
        let name = aperio_kind(&page.description).unwrap_or("thumbnail");
        associated.push(Associated {
            name: name.to_owned(),
            page: page.clone(),
        });
    }
    (levels, associated)
}

/// The marker on the second line of an Aperio description.
fn aperio_kind(description: &str) -> Option<&'static str> {
    let second = description.lines().nth(1)?.trim_start();
    if second.starts_with("label") {
        Some("label")
    } else if second.starts_with("macro") {
        Some("macro")
    } else {
        None
    }
}

/// Hamamatsu NDPI: `SourceLens` is the magnification for a pyramid level and negative for
/// the two extra images — `-1` the macro, `-2` the map.
fn split_ndpi(pages: &[Page]) -> (Vec<Page>, Vec<Associated>) {
    let mut levels = Vec::new();
    let mut associated = Vec::new();
    for page in pages {
        match source_lens(&page.ifd) {
            Some(lens) if lens < 0.0 => {
                let name = if lens <= -2.0 { "map" } else { "macro" };
                associated.push(Associated {
                    name: name.to_owned(),
                    page: page.clone(),
                });
            }
            // A positive lens, or none at all, is a pyramid level.
            _ => levels.push(page.clone()),
        }
    }
    (levels, associated)
}

fn source_lens(ifd: &Ifd) -> Option<f64> {
    ifd.get(ndpi::SOURCE_LENS)
        .and_then(|entry| entry.as_rational())
}

/// Everything else: the reduced-resolution flag marks a level, and nothing else in the
/// chain belongs to series 0.
fn split_by_subfile_type(pages: &[Page]) -> (Vec<Page>, Vec<Associated>) {
    let mut levels = vec![pages[0].clone()];
    for page in &pages[1..] {
        let reduced = page.ifd.u64_or(tag::NEW_SUBFILE_TYPE, 0) & 1 == 1
            || page.ifd.u64_or(tag::SUBFILE_TYPE, 1) == 2;
        if reduced {
            levels.push(page.clone());
        }
    }
    (levels, Vec::new())
}

/// `_PropertyParser.parse_*` plus `collect_level_info`.
fn collect_properties(format: Format, first: &Page, levels: &[Page]) -> Properties {
    let mut properties = Properties {
        vendor: format.vendor().map(str::to_owned),
        level_tiles: levels
            .iter()
            .map(|level| (level.tile_width as i64, level.tile_length as i64))
            .collect(),
        ..Default::default()
    };

    if format == Format::Aperio {
        let aperio = parse_aperio_description(&first.description);
        properties.comment = Some(first.description.clone());
        properties.mpp_x = aperio.get("MPP").and_then(|v| v.parse().ok());
        properties.mpp_y = properties.mpp_x;
        properties.objective_power = aperio.get("AppMag").and_then(|v| v.parse().ok());
    }

    if format == Format::Hamamatsu {
        // `parse_hamamatsu` takes the objective power from `SourceLens`.
        properties.objective_power = source_lens(&first.ifd);
    }

    // `if not _has_mpp(md): md.update(self.recover_mpp(series0))`
    if properties.mpp_x.is_none() || properties.mpp_y.is_none() {
        if let Some((x, y)) = recover_mpp(first) {
            properties.mpp_x = Some(x);
            properties.mpp_y = Some(y);
        }
    }
    properties
}

/// `recover_mpp` — micrometres per pixel from the resolution tags.
///
/// `ResolutionUnit` 5 (`NONE`) yields nothing; the others scale by the unit's length in
/// micrometres.
fn recover_mpp(page: &Page) -> Option<(f64, f64)> {
    let unit = page.ifd.u64_or(tag::RESOLUTION_UNIT, 2);
    let scale = match unit {
        2 => 25_400.0,
        3 => 10_000.0,
        4 => 1_000.0,
        5 => 1.0,
        _ => return None,
    };
    let x = page.ifd.get(tag::X_RESOLUTION)?.as_rational()?;
    let y = page.ifd.get(tag::Y_RESOLUTION)?.as_rational()?;
    if x == 0.0 || y == 0.0 {
        return None;
    }
    Some((scale / x, scale / y))
}

/// `svs_description_metadata`: the first `|`-separated item is the header, the rest are
/// `key = value` pairs. The values are kept as strings and parsed by the caller.
pub fn parse_aperio_description(description: &str) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut items = description.split('|');
    if let Some(header) = items.next() {
        out.insert("Header".to_owned(), header.trim().to_owned());
    }
    for item in items {
        let Some((key, value)) = item.trim().split_once('=') else {
            continue;
        };
        out.insert(key.trim().to_owned(), value.trim().to_owned());
    }
    out
}

/// `supported_vendors` membership, as `__get_pixel_size` tests it.
///
/// `None` is in the Python list, so a vendorless slide — a generic TIFF, and a Ventana one
/// too — passes.
pub fn vendor_is_supported(vendor: Option<&str>) -> bool {
    match vendor {
        None => true,
        Some(name) => matches!(name, "aperio" | "hamamatsu"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_aperio_description_splits_into_header_and_pairs() {
        let description = "Aperio Image Library v10.0.51\r\n46920x33014 [0,100 46000x32914] \
                           (256x256) JPEG/RGB Q=30|AppMag = 20|MPP = 0.4990|Left = 25.69";
        let parsed = parse_aperio_description(description);
        assert!(parsed["Header"].starts_with("Aperio Image Library"));
        assert_eq!(parsed["AppMag"], "20");
        assert_eq!(parsed["MPP"], "0.4990");
        assert_eq!(parsed["Left"], "25.69");
    }

    #[test]
    fn an_item_without_an_equals_sign_is_skipped() {
        let parsed = parse_aperio_description("Aperio Image Library|nonsense|A = 1");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["A"], "1");
    }

    #[test]
    fn the_aperio_second_line_names_the_associated_image() {
        assert_eq!(
            aperio_kind("Aperio Image Library v10.0.51\nlabel 387x463"),
            Some("label")
        );
        assert_eq!(
            aperio_kind("Aperio Image Library v10.0.51\nmacro 1280x431"),
            Some("macro")
        );
        // The thumbnail's second line is a resize arrow, which falls through to `None`.
        assert_eq!(
            aperio_kind("Aperio Image Library v10.0.51\n46000x32914 -> 1024x732 - |AppMag = 20"),
            None
        );
        assert_eq!(aperio_kind("no second line"), None);
    }

    #[test]
    fn the_vendor_is_only_set_for_the_formats_that_parse_one() {
        assert_eq!(Format::Aperio.vendor(), Some("aperio"));
        assert_eq!(Format::Hamamatsu.vendor(), Some("hamamatsu"));
        assert_eq!(Format::Leica.vendor(), Some("leica"));
        // `parse_ventana` delegates to `parse_generic_tiff`, which sets no vendor.
        assert_eq!(Format::Ventana.vendor(), None);
        assert_eq!(Format::GenericTiff.vendor(), None);
    }

    #[test]
    fn format_strings_match_tiffslide() {
        assert_eq!(Format::GenericTiff.as_str(), "generic-tiff");
        assert_eq!(Format::PhilipsTiff.as_str(), "philips_tiff");
    }

    #[test]
    fn clipping_matches_pythons_helper() {
        assert_eq!(clip(-5, 0, 10), 0);
        assert_eq!(clip(15, 0, 10), 10);
        assert_eq!(clip(3, 0, 10), 3);
    }
}

#[cfg(test)]
mod vendor_tests {
    use super::*;

    #[test]
    fn a_vendorless_slide_is_supported_and_a_named_one_usually_is_not() {
        assert!(vendor_is_supported(None));
        assert!(vendor_is_supported(Some("aperio")));
        assert!(vendor_is_supported(Some("hamamatsu")));
        assert!(!vendor_is_supported(Some("leica")));
        assert!(!vendor_is_supported(Some("philips")));
        // Ventana reports no vendor at all, so it is the third format that gets through.
        assert!(vendor_is_supported(Format::Ventana.vendor()));
    }
}
