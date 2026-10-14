//! Reconstructing `tifffile`'s `series[0]`: its `axes`, its `shape`, its pages and its
//! level pyramid.
//!
//! This is the part of the port that cannot be derived from the TIFF specification, only
//! from what tifffile decides — and the plugin's entire layout analysis (`is_supported`,
//! `__inspect_series_layout`, `__get_series_page_index`) is written against those
//! decisions. Four flavours cover what the plugin accepts:
//!
//! | flavour | axes come from |
//! |---|---|
//! | OME     | `<Pixels DimensionOrder Size*>` in the OME-XML |
//! | ImageJ  | `channels` / `slices` / `frames` in the `ImageJ=` header |
//! | shaped  | tifffile's own JSON `{"shape": [...], "axes": "..."}` |
//! | generic | one page is `YX`/`YXS`; several same-shape pages gain a leading `I` |
//!
//! In every flavour a size-1 axis is dropped, samples land on an `S` axis — appended when
//! the page is chunky, prepended when its planes are separate — and the expectations are
//! pinned against files tifffile itself wrote (`tests/fixtures`).

use crate::imagej::{self, ImageJMetadata};
use crate::ome::{self, OmePixels};

use super::ifd::{tag, ByteOrder, Ifd, Result, TiffSource};
use super::page::Page;

/// Which metadata dialect describes the file.
#[derive(Debug, Clone, PartialEq)]
pub enum Flavour {
    Ome(Box<OmePixels>),
    ImageJ(Box<ImageJMetadata>),
    /// tifffile's own `{"shape": ...}` description.
    Shaped {
        shape: Vec<usize>,
        axes: Option<String>,
    },
    Generic,
}

/// One pyramid level: the pages that make it up.
#[derive(Debug, Clone)]
pub struct Level {
    pub pages: Vec<Page>,
}

impl Level {
    /// `level.keyframe` — the first page, which carries the level's geometry.
    pub fn keyframe(&self) -> &Page {
        &self.pages[0]
    }
}

/// `series[0]`.
#[derive(Debug, Clone)]
pub struct Series {
    pub axes: String,
    pub shape: Vec<usize>,
    pub levels: Vec<Level>,
}

impl Series {
    pub fn keyframe(&self) -> &Page {
        self.levels[0].keyframe()
    }

    pub fn pages(&self) -> &[Page] {
        &self.levels[0].pages
    }
}

/// `NewSubfileType` bit 0: a reduced-resolution image, i.e. a pyramid level.
fn is_reduced(ifd: &Ifd) -> bool {
    ifd.u64_or(tag::NEW_SUBFILE_TYPE, 0) & 1 == 1
        // Some writers use the deprecated `SubfileType` instead, where 1 means full
        // resolution and 2 a reduced one.
        || ifd.u64_or(tag::SUBFILE_TYPE, 1) == 2
}

/// Reads the whole file's directory structure and builds `series[0]`.
pub fn build(source: &mut TiffSource) -> Result<(Flavour, Series)> {
    let chain = source.read_chain()?;
    let order = source.order;
    let first = Page::from_ifd(chain[0].clone(), 0, order)?;
    let flavour = classify(&first);

    // Pyramid levels declared through SubIFDs belong to their parent page, not to the
    // chain, so the chain split only has to separate full-resolution pages from
    // reduced-resolution ones written inline.
    let mut full = Vec::new();
    let mut reduced = Vec::new();
    for ifd in chain {
        if is_reduced(&ifd) {
            reduced.push(ifd);
        } else {
            full.push(ifd);
        }
    }

    let spp = first.samples_per_pixel;
    let separate = first.separate_planes();
    let (height, width) = (first.image_length, first.image_width);

    let (axes, shape, page_count) = match &flavour {
        Flavour::Ome(pixels) => ome_layout(pixels, spp, separate, height, width),
        Flavour::ImageJ(metadata) => imagej_layout(metadata, spp, separate, height, width),
        Flavour::Shaped { shape, axes } => {
            shaped_layout(shape, axes.as_deref(), spp, separate, height, width)
        }
        Flavour::Generic => generic_layout(&full, &first, spp, separate, height, width),
    };

    let page_count = page_count.clamp(1, full.len());
    let mut pages = Vec::with_capacity(page_count);
    for (index, ifd) in full.into_iter().take(page_count).enumerate() {
        pages.push(Page::from_ifd(ifd, index, order)?);
    }

    let mut levels = vec![Level { pages }];
    append_levels(source, &mut levels, &reduced, page_count, order)?;

    Ok((
        flavour,
        Series {
            axes,
            shape,
            levels,
        },
    ))
}

/// Levels from SubIFDs when the series pages declare them, otherwise from the
/// reduced-resolution pages of the main chain.
fn append_levels(
    source: &mut TiffSource,
    levels: &mut Vec<Level>,
    reduced: &[Ifd],
    page_count: usize,
    order: ByteOrder,
) -> Result<()> {
    let sub_ifds: Vec<u64> = levels[0].pages[0].ifd.u64_vec(tag::SUB_IFDS);
    if !sub_ifds.is_empty() {
        for depth in 0..sub_ifds.len() {
            let mut pages = Vec::with_capacity(page_count);
            for index in 0..page_count {
                let offsets = levels[0].pages[index].ifd.u64_vec(tag::SUB_IFDS);
                let Some(&offset) = offsets.get(depth) else {
                    break;
                };
                let ifd = source.read_ifd(offset)?;
                pages.push(Page::from_ifd(ifd, index, order)?);
            }
            if pages.len() == page_count {
                levels.push(Level { pages });
            }
        }
        return Ok(());
    }

    // Inline levels come in page_count-sized groups, coarsest last.
    for group in reduced.chunks(page_count) {
        if group.len() != page_count {
            break;
        }
        let mut pages = Vec::with_capacity(page_count);
        for (index, ifd) in group.iter().enumerate() {
            pages.push(Page::from_ifd(ifd.clone(), index, order)?);
        }
        levels.push(Level { pages });
    }
    Ok(())
}

/// Which dialect the first page's `ImageDescription` announces.
pub fn classify(first: &Page) -> Flavour {
    let description = first.description.trim();
    if description.contains("<OME") {
        if let Some(pixels) = ome::parse_pixels(description) {
            return Flavour::Ome(Box::new(pixels));
        }
    }
    if let Some(mut metadata) = imagej::parse_description(description) {
        // `Labels` and `LUTs` live in a separate tag, not in the header.
        let blob = first
            .ifd
            .get(tag::IJ_METADATA)
            .map(|e| e.as_bytes().to_vec());
        let counts = first.ifd.u64_vec(tag::IJ_METADATA_BYTE_COUNTS);
        if let Some(blob) = blob {
            let (labels, luts) = imagej::parse_ij_metadata(&blob, &counts, first.byte_order);
            metadata.labels = labels;
            metadata.luts = luts;
        }
        return Flavour::ImageJ(Box::new(metadata));
    }
    if let Some(shaped) = parse_shaped(description) {
        return shaped;
    }
    Flavour::Generic
}

/// tifffile's own description: a JSON object with a `shape` and sometimes an `axes`.
fn parse_shaped(description: &str) -> Option<Flavour> {
    let text = description.trim();
    if !text.starts_with('{') {
        return None;
    }
    // A hand-rolled read of the two keys, rather than a JSON dependency for a description
    // tifffile writes in one fixed shape.
    let shape = json_number_list(text, "\"shape\"")?;
    let axes = json_string(text, "\"axes\"");
    Some(Flavour::Shaped {
        shape: shape.into_iter().map(|v| v.max(1.0) as usize).collect(),
        axes,
    })
}

fn json_number_list(text: &str, key: &str) -> Option<Vec<f64>> {
    let at = text.find(key)? + key.len();
    let rest = &text[at..];
    let open = rest.find('[')?;
    let close = rest[open..].find(']')? + open;
    Some(
        rest[open + 1..close]
            .split(',')
            .filter_map(|part| part.trim().parse::<f64>().ok())
            .collect(),
    )
}

fn json_string(text: &str, key: &str) -> Option<String> {
    let at = text.find(key)? + key.len();
    let rest = &text[at..];
    let colon = rest.find(':')?;
    let open = rest[colon..].find('"')? + colon;
    let close = rest[open + 1..].find('"')? + open + 1;
    Some(rest[open + 1..close].to_owned())
}

/// Appends the sample axis: `S` last for a chunky page, first for separate planes.
fn with_sample_axis(
    mut axes: String,
    mut shape: Vec<usize>,
    spp: usize,
    separate: bool,
) -> (String, Vec<usize>) {
    if spp <= 1 {
        return (axes, shape);
    }
    if separate {
        axes.insert(0, 'S');
        shape.insert(0, spp);
    } else {
        axes.push('S');
        shape.push(spp);
    }
    (axes, shape)
}

fn ome_layout(
    pixels: &OmePixels,
    spp: usize,
    separate: bool,
    height: usize,
    width: usize,
) -> (String, Vec<usize>, usize) {
    // A chunky RGB page reports SizeC == samplesperpixel, and those samples are not a
    // channel axis — they are the sample axis.
    let planes_c = (pixels.size_c / spp.max(1)).max(1);
    let mut axes = String::new();
    let mut shape = Vec::new();
    let mut page_count = 1usize;

    // `DimensionOrder` is written fastest-first; the array axes are its reverse.
    for name in pixels.dimension_order.chars().rev() {
        let size = match name {
            'T' => pixels.size_t,
            'Z' => pixels.size_z,
            'C' => planes_c,
            _ => continue, // X and Y are placed below, in their fixed positions.
        };
        if size > 1 {
            axes.push(name);
            shape.push(size);
            page_count *= size;
        }
    }
    axes.push('Y');
    axes.push('X');
    shape.push(height);
    shape.push(width);
    let (axes, shape) = with_sample_axis(axes, shape, spp, separate);
    (axes, shape, page_count)
}

fn imagej_layout(
    metadata: &ImageJMetadata,
    spp: usize,
    separate: bool,
    height: usize,
    width: usize,
) -> (String, Vec<usize>, usize) {
    let mut axes = String::new();
    let mut shape = Vec::new();
    let mut page_count = 1usize;
    for (name, size) in [
        ('T', metadata.frames),
        ('Z', metadata.slices),
        ('C', metadata.channels),
    ] {
        if size > 1 {
            axes.push(name);
            shape.push(size);
            page_count *= size;
        }
    }
    axes.push('Y');
    axes.push('X');
    shape.push(height);
    shape.push(width);
    let (axes, shape) = with_sample_axis(axes, shape, spp, separate);
    (axes, shape, page_count)
}

fn shaped_layout(
    declared: &[usize],
    declared_axes: Option<&str>,
    spp: usize,
    separate: bool,
    height: usize,
    width: usize,
) -> (String, Vec<usize>, usize) {
    // The declared shape already includes the sample axis when there is one.
    let trailing = if spp > 1 { 3 } else { 2 };
    let leading = declared.len().saturating_sub(trailing);
    let page_count = declared[..leading].iter().product::<usize>().max(1);

    if let Some(axes) = declared_axes.filter(|axes| axes.len() == declared.len()) {
        return (axes.to_owned(), declared.to_vec(), page_count);
    }

    // No axes given: the leading dimensions are unnamed, which tifffile writes as `Q`.
    let mut axes = "Q".repeat(leading);
    let mut shape = declared[..leading].to_vec();
    axes.push('Y');
    axes.push('X');
    shape.push(height);
    shape.push(width);
    let (axes, shape) = with_sample_axis(axes, shape, spp, separate);
    (axes, shape, page_count)
}

fn generic_layout(
    full: &[Ifd],
    first: &Page,
    spp: usize,
    separate: bool,
    height: usize,
    width: usize,
) -> (String, Vec<usize>, usize) {
    // Pages of identical geometry form a sequence; anything else starts a new series,
    // which this plugin never looks at.
    let matching = full
        .iter()
        .take_while(|ifd| {
            ifd.u64_or(tag::IMAGE_WIDTH, 0) as usize == width
                && ifd.u64_or(tag::IMAGE_LENGTH, 0) as usize == height
                && ifd.u64_or(tag::SAMPLES_PER_PIXEL, 1).max(1) as usize == spp
                && ifd.u64_or(tag::BITS_PER_SAMPLE, 1) as usize == first.bits_per_sample
        })
        .count()
        .max(1);

    let mut axes = String::new();
    let mut shape = Vec::new();
    if matching > 1 {
        // tifffile names an unlabelled sequence axis `I`.
        axes.push('I');
        shape.push(matching);
    }
    axes.push('Y');
    axes.push('X');
    shape.push(height);
    shape.push(width);
    let (axes, shape) = with_sample_axis(axes, shape, spp, separate);
    (axes, shape, matching)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_ome(
        order: &str,
        size_c: usize,
        size_z: usize,
        size_t: usize,
        spp: usize,
        separate: bool,
    ) -> (String, Vec<usize>, usize) {
        let pixels = OmePixels {
            dimension_order: order.to_owned(),
            size_x: 96,
            size_y: 64,
            size_c,
            size_z,
            size_t,
            ..Default::default()
        };
        ome_layout(&pixels, spp, separate, 64, 96)
    }

    #[test]
    fn ome_axes_match_tifffile_for_the_fixture_shapes() {
        // rgb_contig.ome.tif
        assert_eq!(
            layout_ome("XYCZT", 3, 1, 1, 3, false),
            ("YXS".into(), vec![64, 96, 3], 1)
        );
        // rgb_separate.ome.tif
        assert_eq!(
            layout_ome("XYCZT", 3, 1, 1, 3, true),
            ("SYX".into(), vec![3, 64, 96], 1)
        );
        // fluorescence.ome.tif
        assert_eq!(
            layout_ome("XYCZT", 3, 1, 1, 1, false),
            ("CYX".into(), vec![3, 64, 96], 3)
        );
        // pixelsize_*.ome.tif — every non-spatial axis is size 1.
        assert_eq!(
            layout_ome("XYCZT", 1, 1, 1, 1, false),
            ("YX".into(), vec![64, 96], 1)
        );
        // untiled_czyx.ome.tif, whose DimensionOrder puts Z before C.
        assert_eq!(
            layout_ome("XYZCT", 2, 3, 1, 1, false),
            ("CZYX".into(), vec![2, 3, 64, 96], 6)
        );
    }

    #[test]
    fn imagej_axes_match_tifffile_for_the_fixture_shapes() {
        let metadata = |channels, slices, images| ImageJMetadata {
            images,
            channels,
            slices,
            frames: 1,
            hyperstack: true,
            ..Default::default()
        };
        // channels_only_imagej.tif
        assert_eq!(
            imagej_layout(&metadata(3, 1, 3), 1, false, 4, 5),
            ("CYX".into(), vec![3, 4, 5], 3)
        );
        // zstack_imagej.tif — slices outside channels.
        assert_eq!(
            imagej_layout(&metadata(2, 3, 6), 1, false, 4, 5),
            ("ZCYX".into(), vec![3, 2, 4, 5], 6)
        );
        // rgb_zstack_imagej.tif
        assert_eq!(
            imagej_layout(&metadata(1, 3, 3), 3, false, 4, 5),
            ("ZYXS".into(), vec![3, 4, 5, 3], 3)
        );
    }

    #[test]
    fn a_generic_sequence_gains_a_leading_axis() {
        // A single page is plain YX / YXS; several identical pages gain `I`, which the
        // plugin then refuses as an unsupported axis.
        let (axes, shape, count) = generic_layout(&[], &stub_page(4, 5, 1, 8), 1, false, 4, 5);
        assert_eq!(
            (axes.as_str(), shape.as_slice(), count),
            ("YX", &[4, 5][..], 1)
        );
    }

    #[test]
    fn a_shaped_description_is_recognised() {
        let flavour = parse_shaped(r#"{"shape": [3, 4, 5]}"#).unwrap();
        assert_eq!(
            flavour,
            Flavour::Shaped {
                shape: vec![3, 4, 5],
                axes: None
            }
        );
        let with_axes = parse_shaped(r#"{"shape": [3, 4, 5], "axes": "CYX"}"#).unwrap();
        assert_eq!(
            with_axes,
            Flavour::Shaped {
                shape: vec![3, 4, 5],
                axes: Some("CYX".into())
            }
        );
        assert!(parse_shaped("not json").is_none());
    }

    #[test]
    fn an_unlabelled_shaped_stack_uses_q_axes() {
        assert_eq!(
            shaped_layout(&[3, 4, 5], None, 1, false, 4, 5),
            ("QYX".into(), vec![3, 4, 5], 3)
        );
        assert_eq!(
            shaped_layout(&[3, 4, 5], Some("CYX"), 1, false, 4, 5),
            ("CYX".into(), vec![3, 4, 5], 3)
        );
    }

    fn stub_page(height: usize, width: usize, spp: usize, bits: usize) -> Page {
        let mut ifd = Ifd::default();
        let _ = (&mut ifd, height, width, spp, bits);
        // `generic_layout` only reads `bits_per_sample` off the page, so a minimal stub is
        // enough; the geometry is passed in separately.
        Page {
            ifd,
            index: 0,
            image_width: width,
            image_length: height,
            image_depth: 1,
            bits_per_sample: bits,
            samples_per_pixel: spp,
            photometric: 1,
            planar_config: 1,
            compression: 1,
            predictor: 1,
            tile_width: 0,
            tile_length: 0,
            tile_depth: 1,
            rows_per_strip: height,
            dtype: super::super::page::SampleDtype::U8,
            data_offsets: Vec::new(),
            data_byte_counts: Vec::new(),
            jpeg_tables: None,
            description: String::new(),
            byte_order: ByteOrder::Little,
        }
    }
}
