//! The slice of OME-XML this plugin reads.
//!
//! Python parses the whole document with `defusedxml` and then reaches for exactly two
//! things: the first `Image`'s first `Pixels` element, and that element's direct `Channel`
//! children. This does the same with a streaming parser, which also means no entity
//! expansion — `defusedxml`'s reason for existing.

use quick_xml::events::Event;
use quick_xml::Reader;

/// A `<Channel>` element, as far as the plugin cares.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OmeChannel {
    pub name: Option<String>,
    /// `Color` verbatim. OME stores a signed 32-bit RGBA integer, and the string is kept
    /// so an unparsable value falls back the same way Python's `int()` failure does.
    pub color: Option<String>,
    /// `SamplesPerPixel`, defaulting to 1; each element expands to this many entries.
    pub samples_per_pixel: usize,
}

/// The `<Pixels>` attributes, plus its channels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OmePixels {
    pub dimension_order: String,
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    pub size_c: usize,
    pub size_t: usize,
    pub interleaved: Option<bool>,
    pub physical_size_x: Option<String>,
    pub physical_size_x_unit: Option<String>,
    pub physical_size_y: Option<String>,
    pub physical_size_y_unit: Option<String>,
    pub physical_size_z: Option<String>,
    pub physical_size_z_unit: Option<String>,
    pub channels: Vec<OmeChannel>,
}

/// Parses the first `Image`'s `Pixels` element out of an OME-XML document.
///
/// Returns `None` when the document has no such element, which the caller reports as
/// `400 Could not obtain ome metadata (...)` — the same status Python's parse failure
/// produces.
pub fn parse_pixels(xml: &str) -> Option<OmePixels> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut in_image = false;
    let mut pixels: Option<OmePixels> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(start)) | Ok(Event::Empty(start)) => match local_name(&start) {
                "Image" if pixels.is_none() => in_image = true,
                "Pixels" if in_image && pixels.is_none() => {
                    pixels = Some(read_pixels(&start));
                }
                "Channel" => {
                    if let Some(pixels) = pixels.as_mut() {
                        pixels.channels.push(read_channel(&start));
                    }
                }
                _ => {}
            },
            Ok(Event::End(end)) => {
                // Stop collecting once the first Pixels element closes, so a second Image
                // cannot contribute channels to the first.
                if local_name_bytes(end.name().as_ref()) == "Pixels" && pixels.is_some() {
                    break;
                }
            }
            Ok(_) => {}
        }
    }
    pixels
}

fn local_name(start: &quick_xml::events::BytesStart<'_>) -> &'static str {
    // The element names of interest are fixed, so mapping to a static string keeps the
    // match arms simple and avoids allocating for every element in a large document.
    match local_name_bytes(start.name().as_ref()) {
        "Image" => "Image",
        "Pixels" => "Pixels",
        "Channel" => "Channel",
        _ => "",
    }
}

fn local_name_bytes(name: &[u8]) -> &str {
    let text = std::str::from_utf8(name).unwrap_or_default();
    text.rsplit(':').next().unwrap_or_default()
}

fn attribute(start: &quick_xml::events::BytesStart<'_>, wanted: &str) -> Option<String> {
    for attribute in start.attributes().with_checks(false).flatten() {
        if local_name_bytes(attribute.key.as_ref()) == wanted {
            return Some(
                attribute
                    .unescape_value()
                    .map(|value| value.into_owned())
                    .unwrap_or_else(|_| {
                        String::from_utf8_lossy(attribute.value.as_ref()).into_owned()
                    }),
            );
        }
    }
    None
}

fn read_pixels(start: &quick_xml::events::BytesStart<'_>) -> OmePixels {
    let size = |name: &str| -> usize {
        attribute(start, name)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1)
    };
    OmePixels {
        dimension_order: attribute(start, "DimensionOrder").unwrap_or_else(|| "XYCZT".into()),
        size_x: size("SizeX"),
        size_y: size("SizeY"),
        size_z: size("SizeZ"),
        size_c: size("SizeC"),
        size_t: size("SizeT"),
        interleaved: attribute(start, "Interleaved")
            .map(|value| value.trim().eq_ignore_ascii_case("true")),
        physical_size_x: attribute(start, "PhysicalSizeX"),
        physical_size_x_unit: attribute(start, "PhysicalSizeXUnit"),
        physical_size_y: attribute(start, "PhysicalSizeY"),
        physical_size_y_unit: attribute(start, "PhysicalSizeYUnit"),
        physical_size_z: attribute(start, "PhysicalSizeZ"),
        physical_size_z_unit: attribute(start, "PhysicalSizeZUnit"),
        channels: Vec::new(),
    }
}

fn read_channel(start: &quick_xml::events::BytesStart<'_>) -> OmeChannel {
    OmeChannel {
        name: attribute(start, "Name"),
        color: attribute(start, "Color"),
        samples_per_pixel: attribute(start, "SamplesPerPixel")
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1),
    }
}

/// `_OME_UNIT_TO_NM`, with `µm` as the schema default.
///
/// The lookup is on the lower-cased unit with the Greek mu folded onto the micro sign,
/// which is what Python's `.lower().replace("μ", "µ")` does.
pub fn length_to_nm(size: Option<&str>, unit: Option<&str>) -> Option<f64> {
    let size = size?;
    let unit = unit
        .unwrap_or("µm")
        .trim()
        .to_lowercase()
        .replace('\u{03bc}', "\u{b5}");
    let scale = match unit.as_str() {
        "m" => 1e9,
        "dm" => 1e8,
        "cm" => 1e7,
        "mm" => 1e6,
        "\u{b5}m" | "um" => 1e3,
        "nm" => 1.0,
        "pm" => 1e-3,
        "å" | "angstrom" => 0.1,
        "in" | "inch" => 2.54e7,
        _ => return None,
    };
    let value = size.trim().parse::<f64>().ok()?;
    (value > 0.0).then_some(value * scale)
}

/// `int(raw_color)` on a `Color` attribute, or `None` when `int()` would raise.
///
/// OME writes the colour as a *signed* 32-bit RGBA integer, so `-16776961` is valid. The
/// conversion to channels is left to `wsi_core`'s port of `convert_int_to_rgba_array`,
/// whose arithmetic shift reproduces Python's floor-shift on a negative value.
pub fn parse_color(color: Option<&str>) -> Option<i64> {
    color?.trim().parse::<i64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGB_CONTIG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
 <Image ID="Image:0" Name="Image0">
  <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8" SizeX="96" SizeY="64"
          SizeC="3" SizeZ="1" SizeT="1" Interleaved="true">
   <Channel ID="Channel:0:0" SamplesPerPixel="3"><LightPath/></Channel>
   <TiffData IFD="0" PlaneCount="1"/>
  </Pixels>
 </Image>
</OME>"#;

    #[test]
    fn pixels_attributes_and_channels_are_read() {
        let pixels = parse_pixels(RGB_CONTIG).unwrap();
        assert_eq!(pixels.dimension_order, "XYCZT");
        assert_eq!((pixels.size_x, pixels.size_y), (96, 64));
        assert_eq!((pixels.size_c, pixels.size_z, pixels.size_t), (3, 1, 1));
        assert_eq!(pixels.interleaved, Some(true));
        assert_eq!(pixels.channels.len(), 1);
        assert_eq!(pixels.channels[0].samples_per_pixel, 3);
        assert_eq!(pixels.channels[0].name, None);
    }

    #[test]
    fn named_and_coloured_channels_are_read() {
        let xml = r#"<OME><Image><Pixels SizeX="8" SizeY="6" SizeC="2">
            <Channel Name="DAPI" Color="16711935" SamplesPerPixel="1"/>
            <Channel Name="Cy5" Color="-16776961" SamplesPerPixel="1"/>
            </Pixels></Image></OME>"#;
        let pixels = parse_pixels(xml).unwrap();
        let names: Vec<_> = pixels
            .channels
            .iter()
            .map(|c| c.name.clone().unwrap())
            .collect();
        assert_eq!(names, vec!["DAPI", "Cy5"]);
        assert_eq!(parse_color(Some("16711935")), Some(16_711_935));
        // A negative colour is valid: OME stores signed 32-bit RGBA.
        assert_eq!(parse_color(Some("-16776961")), Some(-16_776_961));
        assert_eq!(parse_color(Some("not a number")), None);
        assert_eq!(parse_color(None), None);
        // The shift into channels is `wsi_core`'s, and must survive the sign.
        assert_eq!(
            wsi_core::image::int_to_rgba_array(-16_776_961),
            [255, 0, 0, 255]
        );
        assert_eq!(
            wsi_core::image::int_to_rgba_array(16_711_935),
            [0, 255, 0, 255]
        );
    }

    #[test]
    fn only_the_first_image_contributes() {
        let xml = r#"<OME>
            <Image><Pixels SizeX="1" SizeY="2" SizeC="1"><Channel Name="first"/></Pixels></Image>
            <Image><Pixels SizeX="9" SizeY="9" SizeC="1"><Channel Name="second"/></Pixels></Image>
            </OME>"#;
        let pixels = parse_pixels(xml).unwrap();
        assert_eq!((pixels.size_x, pixels.size_y), (1, 2));
        assert_eq!(pixels.channels.len(), 1);
        assert_eq!(pixels.channels[0].name.as_deref(), Some("first"));
    }

    #[test]
    fn a_document_without_pixels_yields_nothing() {
        assert_eq!(parse_pixels("<OME><Image/></OME>"), None);
        assert_eq!(parse_pixels("not xml at all"), None);
    }

    #[test]
    fn physical_sizes_convert_through_the_unit_table() {
        assert_eq!(length_to_nm(Some("1.7e-4"), Some("mm")), Some(170.0));
        assert_eq!(length_to_nm(Some("325"), Some("nm")), Some(325.0));
        // The schema default is µm.
        assert_eq!(length_to_nm(Some("0.5"), None), Some(500.0));
        // The Greek mu folds onto the micro sign.
        assert_eq!(length_to_nm(Some("1"), Some("\u{03bc}m")), Some(1000.0));
        assert_eq!(length_to_nm(Some("1"), Some("furlong")), None);
        // A non-positive size is not usable.
        assert_eq!(length_to_nm(Some("0"), Some("mm")), None);
        assert_eq!(length_to_nm(Some("-1"), Some("mm")), None);
        assert_eq!(length_to_nm(None, Some("mm")), None);
    }

    #[test]
    fn micrometre_rounds_the_way_python_does() {
        let value = length_to_nm(Some("0.2277"), Some("µm")).unwrap();
        assert!((value - 227.7).abs() < 1e-9, "{value}");
    }
}
