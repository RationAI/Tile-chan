//! ImageJ metadata: the `ImageJ=` header in `ImageDescription`, and the `IJMetadata` tag
//! that carries channel labels and lookup tables.
//!
//! `tifffile` exposes both through `TiffFile.imagej_metadata`, and the plugin reads four
//! things from it: `Labels` for channel names, `LUTs` for channel colours, and
//! `spacing`/`unit` for the Z spacing. The stack geometry (`channels`, `slices`, `frames`)
//! is what gives an ImageJ file its series axes.

use super::tiff::ifd::ByteOrder;

/// The parsed `ImageJ=` header plus the `IJMetadata` payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageJMetadata {
    pub images: usize,
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub hyperstack: bool,
    pub spacing: Option<f64>,
    pub unit: Option<String>,
    /// `Labels` — one per plane, used as channel names.
    pub labels: Vec<String>,
    /// `LUTs` — each entry is a flat `3 * 256` table.
    pub luts: Vec<Vec<u8>>,
}

/// Parses the `ImageJ=`-prefixed `ImageDescription`, or `None` when it is not one.
///
/// The header is a flat `key=value` list, one per line. Unknown keys are ignored, as
/// tifffile ignores them.
pub fn parse_description(description: &str) -> Option<ImageJMetadata> {
    if !description.starts_with("ImageJ=") {
        return None;
    }
    let mut out = ImageJMetadata {
        images: 1,
        channels: 1,
        slices: 1,
        frames: 1,
        ..Default::default()
    };
    for line in description.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "images" => out.images = value.parse().unwrap_or(1),
            "channels" => out.channels = value.parse().unwrap_or(1),
            "slices" => out.slices = value.parse().unwrap_or(1),
            "frames" => out.frames = value.parse().unwrap_or(1),
            "hyperstack" => out.hyperstack = value.eq_ignore_ascii_case("true"),
            "spacing" => out.spacing = value.parse().ok(),
            "unit" => out.unit = Some(value.to_owned()),
            _ => {}
        }
    }
    Some(out)
}

/// `_ij_metadata` — the `IJMetadata` blob, split by `IJMetadataByteCounts`.
///
/// The blob opens with a header listing `(type, count)` pairs and the data blocks follow
/// in that order. The header's own integers — the four-character type codes included —
/// are written in the file's byte order, which the magic announces: `IJIJ` for big-endian
/// and `JIJI` for little-endian, where every code reads backwards (`lbal`, `stul`). Only
/// `labl` and `luts` matter here.
pub fn parse_ij_metadata(
    blob: &[u8],
    counts: &[u64],
    order: ByteOrder,
) -> (Vec<String>, Vec<Vec<u8>>) {
    let mut labels = Vec::new();
    let mut luts = Vec::new();
    if blob.len() < 8 || counts.is_empty() {
        return (labels, luts);
    }
    let header_is_little = match &blob[..4] {
        b"IJIJ" => false,
        b"JIJI" => true,
        _ => return (labels, luts),
    };

    // The header's own length is the first byte count; the rest describe the blocks.
    let header_len = counts[0] as usize;
    if header_len > blob.len() || header_len < 8 {
        return (labels, luts);
    }
    let mut kinds: Vec<([u8; 4], usize)> = Vec::new();
    let mut at = 4;
    while at + 8 <= header_len {
        let mut kind = [0u8; 4];
        kind.copy_from_slice(&blob[at..at + 4]);
        let count_bytes = [blob[at + 4], blob[at + 5], blob[at + 6], blob[at + 7]];
        let count = if header_is_little {
            kind.reverse();
            u32::from_le_bytes(count_bytes)
        } else {
            u32::from_be_bytes(count_bytes)
        };
        kinds.push((kind, count as usize));
        at += 8;
    }

    let mut block = 1usize;
    let mut offset = header_len;
    for (kind, count) in kinds {
        for _ in 0..count {
            let Some(&length) = counts.get(block) else {
                return (labels, luts);
            };
            let length = length as usize;
            let end = offset.saturating_add(length).min(blob.len());
            let data = &blob[offset.min(blob.len())..end];
            match &kind {
                b"labl" => labels.push(decode_utf16(data, order)),
                b"luts" => luts.push(data.to_vec()),
                _ => {}
            }
            offset = end;
            block += 1;
        }
    }
    (labels, luts)
}

/// `data.decode("utf-16" + ("be" | "le"))`, which is how tifffile reads a label.
fn decode_utf16(data: &[u8], order: ByteOrder) -> String {
    let units: Vec<u16> = data
        .chunks_exact(2)
        .map(|pair| match order {
            ByteOrder::Little => u16::from_le_bytes([pair[0], pair[1]]),
            ByteOrder::Big => u16::from_be_bytes([pair[0], pair[1]]),
        })
        .collect();
    String::from_utf16_lossy(&units)
}

/// `__get_z_spacing_generic` — ImageJ's Z spacing in nanometres.
///
/// Both `spacing` and a recognised `unit` are required; either missing means the plugin
/// reports no Z spacing at all.
pub fn z_spacing_nm(metadata: &ImageJMetadata) -> Option<f64> {
    let spacing = metadata.spacing?;
    let unit = metadata.unit.as_ref()?.trim().to_lowercase();
    if unit.is_empty() {
        return None;
    }
    let unit = unit.replace('\u{03bc}', "\u{b5}");
    let scale = match unit.as_str() {
        "nm" => 1.0,
        "\u{b5}m" | "um" => 1_000.0,
        "mm" => 1_000_000.0,
        "cm" => 10_000_000.0,
        "m" => 1_000_000_000.0,
        _ => return None,
    };
    Some(spacing * scale)
}

/// `__extract_channel_colors` — the brightest entry of each LUT row.
///
/// A LUT is `(3, 256)`; the display colour is the top entry of each row. Anything that is
/// not exactly that shape falls back to the palette, which the caller supplies.
pub fn lut_color(lut: &[u8]) -> Option<(u8, u8, u8)> {
    if lut.len() != 3 * 256 {
        return None;
    }
    Some((lut[255], lut[511], lut[767]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hyperstack_header_is_parsed() {
        let description =
            "ImageJ=1.11a\nimages=6\nchannels=2\nslices=3\nhyperstack=true\nmode=composite\n";
        let metadata = parse_description(description).unwrap();
        assert_eq!(metadata.images, 6);
        assert_eq!(metadata.channels, 2);
        assert_eq!(metadata.slices, 3);
        assert_eq!(metadata.frames, 1);
        assert!(metadata.hyperstack);
    }

    #[test]
    fn spacing_and_unit_convert_to_nanometres() {
        let metadata =
            parse_description("ImageJ=1.11a\nimages=3\nslices=3\nspacing=20.0\nunit=um\n").unwrap();
        assert_eq!(metadata.spacing, Some(20.0));
        assert_eq!(z_spacing_nm(&metadata), Some(20_000.0));

        // Spacing without a unit is not usable.
        let no_unit = parse_description("ImageJ=1.11a\nspacing=20.0\n").unwrap();
        assert_eq!(z_spacing_nm(&no_unit), None);
        let odd_unit = parse_description("ImageJ=1.11a\nspacing=20.0\nunit=pixel\n").unwrap();
        assert_eq!(z_spacing_nm(&odd_unit), None);
    }

    #[test]
    fn a_non_imagej_description_is_not_claimed() {
        assert_eq!(parse_description("<?xml version=\"1.0\"?><OME/>"), None);
        assert_eq!(parse_description(""), None);
    }

    /// Builds the two-entry header the fixtures carry, in either byte order.
    fn ij_blob(little: bool, label: &[u8], lut: &[u8]) -> (Vec<u8>, Vec<u64>) {
        fn kind(blob: &mut Vec<u8>, code: &[u8; 4], little: bool) {
            let mut bytes = *code;
            if little {
                bytes.reverse();
            }
            blob.extend_from_slice(&bytes);
            blob.extend_from_slice(&if little {
                1u32.to_le_bytes()
            } else {
                1u32.to_be_bytes()
            });
        }
        let mut blob = Vec::new();
        blob.extend_from_slice(if little { b"JIJI" } else { b"IJIJ" });
        kind(&mut blob, b"labl", little);
        kind(&mut blob, b"luts", little);
        let header_len = blob.len();
        blob.extend_from_slice(label);
        blob.extend_from_slice(lut);
        (
            blob,
            vec![header_len as u64, label.len() as u64, lut.len() as u64],
        )
    }

    #[test]
    fn labels_and_luts_are_split_by_their_byte_counts() {
        // One "labl" entry and one "luts" entry, in that order.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"IJIJ");
        blob.extend_from_slice(b"labl");
        blob.extend_from_slice(&1u32.to_be_bytes());
        blob.extend_from_slice(b"luts");
        blob.extend_from_slice(&1u32.to_be_bytes());
        let header_len = blob.len();

        let label: Vec<u8> = "DAPI"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        blob.extend_from_slice(&label);
        let mut lut = vec![0u8; 768];
        lut[255] = 10;
        lut[511] = 20;
        lut[767] = 30;
        blob.extend_from_slice(&lut);

        let counts = vec![header_len as u64, label.len() as u64, 768];
        let (labels, luts) = parse_ij_metadata(&blob, &counts, ByteOrder::Little);
        assert_eq!(labels, vec!["DAPI"]);
        assert_eq!(luts.len(), 1);
        assert_eq!(lut_color(&luts[0]), Some((10, 20, 30)));
    }

    #[test]
    fn a_little_endian_header_has_its_type_codes_reversed() {
        // tifffile writes `JIJI` with `lbal`/`stul` when the file is little-endian, which
        // is what every fixture here actually contains.
        let label: Vec<u8> = "FITC"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let mut lut = vec![0u8; 768];
        lut[255] = 7;
        lut[511] = 8;
        lut[767] = 9;
        let (blob, counts) = ij_blob(true, &label, &lut);
        let (labels, luts) = parse_ij_metadata(&blob, &counts, ByteOrder::Little);
        assert_eq!(labels, vec!["FITC"]);
        assert_eq!(lut_color(&luts[0]), Some((7, 8, 9)));
    }

    #[test]
    fn a_malformed_blob_yields_nothing_rather_than_failing() {
        assert_eq!(
            parse_ij_metadata(b"nope", &[4], ByteOrder::Little),
            (Vec::new(), Vec::new())
        );
        assert_eq!(
            parse_ij_metadata(&[], &[], ByteOrder::Little),
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn a_lut_of_the_wrong_shape_has_no_colour() {
        assert_eq!(lut_color(&[0u8; 100]), None);
    }
}
