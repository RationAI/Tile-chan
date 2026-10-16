"""Regenerates the TIFF fixtures and the ground-truth dump the Rust tests read.

The Rust `tifffile` plugin has to agree with the Python one about what a TIFF *is* —
`series[0].axes`, its shape, the level pyramid, each page's geometry and, above all, the
decoded pixels. Guessing at that from the specification is how a reader ends up subtly
wrong on real slides, so the expectations are taken from tifffile itself: this script
writes the fixtures and records, for every page, the geometry and the exact bytes
`page.asarray()` produces.

Run it from anywhere with `tifffile` and `numpy` installed:

    python rust/crates/wsi-plugin-tifffile/tests/fixtures/generate.py

The fixtures are committed, so a tifffile upgrade cannot silently move the goalposts —
rerunning this is a deliberate act, and `git diff` shows exactly what changed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).parent


def rgb_planes(height: int, width: int) -> np.ndarray:
    """Three planes no two of which are equal, so a collapsed plane is obvious."""
    ys, xs = np.mgrid[0:height, 0:width]
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[..., 0] = (ys * 2 + xs) % 251
    data[..., 1] = (ys + xs * 3 + 40) % 251
    data[..., 2] = (ys * 5 + xs * 7 + 90) % 251
    return data


def fixtures() -> dict[str, callable]:
    """name -> writer. Each writer takes the destination path."""

    def plain_rgb(path):
        # `is_supported` must reject this: plain 8-bit RGB belongs to tiffslide/openslide.
        rgb = np.asarray(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8
        )
        tifffile.imwrite(path, rgb, photometric="rgb")

    def deep_rgb(path):
        # `is_supported` must accept this: openslide flattens a 16-bit RGB page to 8-bit
        # and tiffslide truncates it, so the deeper samples only survive here.
        rgb = np.asarray(
            [[[65535, 0, 0], [0, 65535, 0]], [[0, 0, 65535], [65535, 65535, 0]]],
            dtype=np.uint16,
        )
        tifffile.imwrite(path, rgb, photometric="rgb")

    def plain_grey16(path):
        # `is_supported` must accept this too: a single-channel `YX` page is a layout this
        # plugin reads at any depth, and no other plugin opens a 16-bit greyscale TIFF.
        grey = np.asarray([[0, 21845], [43690, 65535]], dtype=np.uint16)
        tifffile.imwrite(path, grey, photometric="minisblack")

    def rgb_zstack_imagej(path):
        data = np.asarray(
            [
                [[[z * 20 + c + 1 for c in range(3)] for _x in range(5)] for _y in range(4)]
                for z in range(3)
            ],
            dtype=np.uint8,
        )
        tifffile.imwrite(
            path,
            data,
            photometric="rgb",
            imagej=True,
            metadata={"axes": "ZYXS", "slices": 3, "spacing": 20.0, "unit": "um"},
        )

    def zstack_imagej(path):
        data = np.asarray(
            [
                [[[z * 20 + c + 1 for _x in range(5)] for _y in range(4)] for c in range(2)]
                for z in range(3)
            ],
            dtype=np.uint8,
        )
        tifffile.imwrite(
            path,
            data,
            imagej=True,
            metadata={
                "axes": "ZCYX",
                "channels": 2,
                "slices": 3,
                "hyperstack": True,
                "mode": "composite",
            },
        )

    def channels_only_imagej(path):
        data = np.asarray(
            [[[(c * 10) + 1 for _x in range(5)] for _y in range(4)] for c in range(3)],
            dtype=np.uint8,
        )
        tifffile.imwrite(
            path,
            data,
            imagej=True,
            metadata={"axes": "CYX", "channels": 3, "hyperstack": True, "mode": "composite"},
        )

    def imagej_named_channels(path):
        # ImageJ carries channel names in `Labels` and colours in `LUTs`.
        data = np.asarray(
            [[[c * 30 + 1 for _x in range(4)] for _y in range(3)] for c in range(2)],
            dtype=np.uint8,
        )
        luts = [
            np.stack(
                [
                    np.linspace(0, r, 256, dtype=np.uint8),
                    np.linspace(0, g, 256, dtype=np.uint8),
                    np.linspace(0, b, 256, dtype=np.uint8),
                ]
            )
            for r, g, b in ((0, 200, 100), (180, 0, 40))
        ]
        tifffile.imwrite(
            path,
            data,
            imagej=True,
            metadata={
                "axes": "CYX",
                "channels": 2,
                "hyperstack": True,
                "mode": "composite",
                "Labels": ["DAPI", "FITC"],
                "LUTs": luts,
            },
        )

    def rgb_separate_ome(path):
        tifffile.imwrite(
            path,
            np.moveaxis(rgb_planes(64, 96), -1, 0),
            photometric="rgb",
            planarconfig="separate",
            tile=(32, 32),
            ome=True,
        )

    def rgb_contig_ome(path):
        tifffile.imwrite(
            path,
            rgb_planes(64, 96),
            photometric="rgb",
            planarconfig="contig",
            tile=(32, 32),
            ome=True,
        )

    def fluorescence_ome(path):
        # <Channel> elements with neither Color nor Name: the fallback palette applies.
        data = np.asarray(
            [[[c * 10 + y * 3 + x for x in range(8)] for y in range(6)] for c in range(3)],
            dtype=np.uint8,
        )
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)

    def named_channels_ome(path):
        # Names and signed 32-bit RGBA colours, the way Bio-Formats writes them.
        data = np.zeros((2, 6, 8), dtype=np.uint16)
        data[0] = 1000
        data[1] = 2000
        tifffile.imwrite(
            path,
            data,
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": ["DAPI", "Cy5"], "Color": [16711935, -16776961]},
            },
            ome=True,
        )

    def nopixelsize_ome(path):
        tifffile.imwrite(
            path,
            np.zeros((2, 6, 8), dtype=np.uint8),
            photometric="minisblack",
            metadata={"axes": "CYX"},
            ome=True,
        )

    def signed_ome(path):
        tifffile.imwrite(
            path,
            np.zeros((2, 6, 8), dtype=np.int8),
            photometric="minisblack",
            metadata={"axes": "CYX"},
            ome=True,
        )

    def untiled_czyx_ome(path):
        data = np.asarray(
            [
                [[[c * 100 + z * 10 + y + x for x in range(9)] for y in range(7)] for z in range(3)]
                for c in range(2)
            ],
            dtype=np.uint8,
        )
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "CZYX"}, ome=True)

    def pyramid_ome(path):
        # Sub-resolutions land in SubIFDs, which is how `series[0].levels` gets more than
        # one entry — the case a reader that only walks the main IFD chain cannot see.
        data = rgb_planes(128, 160)
        with tifffile.TiffWriter(path, ome=True) as writer:
            writer.write(
                data,
                photometric="rgb",
                tile=(32, 32),
                subifds=2,
                metadata={"axes": "YXS"},
            )
            writer.write(data[::2, ::2], photometric="rgb", tile=(32, 32), subfiletype=1)
            writer.write(data[::4, ::4], photometric="rgb", tile=(32, 32), subfiletype=1)

    def bigtiff_ome(path):
        tifffile.imwrite(
            path,
            np.moveaxis(rgb_planes(48, 48), -1, 0),
            photometric="rgb",
            planarconfig="separate",
            tile=(16, 16),
            bigtiff=True,
            ome=True,
        )

    def bigendian_ome(path):
        data = np.asarray(
            [[[c * 4096 + y * 64 + x for x in range(8)] for y in range(6)] for c in range(2)],
            dtype=np.uint16,
        )
        tifffile.imwrite(
            path,
            data,
            photometric="minisblack",
            byteorder=">",
            metadata={"axes": "CYX"},
            ome=True,
        )

    def float_ome(path):
        data = np.asarray(
            [[[c + y / 4 + x / 8 for x in range(8)] for y in range(6)] for c in range(2)],
            dtype=np.float32,
        )
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)

    named = {
        "plain_rgb.tif": plain_rgb,
        "deep_rgb.tif": deep_rgb,
        "plain_grey16.tif": plain_grey16,
        "rgb_zstack_imagej.tif": rgb_zstack_imagej,
        "zstack_imagej.tif": zstack_imagej,
        "channels_only_imagej.tif": channels_only_imagej,
        "imagej_named_channels.tif": imagej_named_channels,
        "rgb_separate.ome.tif": rgb_separate_ome,
        "rgb_contig.ome.tif": rgb_contig_ome,
        "fluorescence.ome.tif": fluorescence_ome,
        "named_channels.ome.tif": named_channels_ome,
        "nopixelsize.ome.tif": nopixelsize_ome,
        "signed.ome.tif": signed_ome,
        "untiled_czyx.ome.tif": untiled_czyx_ome,
        "pyramid.ome.tif": pyramid_ome,
        "bigtiff.ome.tif": bigtiff_ome,
        "bigendian.ome.tif": bigendian_ome,
        "float.ome.tif": float_ome,
    }

    # Pixel-size units, each a separate file so the Rust test can table-drive them.
    for label, unit, size in (
        ("mm", "mm", 1.7e-4),
        ("um", "µm", 0.2277),
        ("nm", "nm", 325.0),
        ("default", None, 0.5),
    ):
        def writer(path, unit=unit, size=size):
            metadata = {"axes": "CYX", "PhysicalSizeX": size, "PhysicalSizeY": size}
            if unit is not None:
                metadata["PhysicalSizeXUnit"] = unit
                metadata["PhysicalSizeYUnit"] = unit
            tifffile.imwrite(
                path,
                np.zeros((1, 6, 8), dtype=np.uint8),
                photometric="minisblack",
                metadata=metadata,
                ome=True,
            )

        named[f"pixelsize_{label}.ome.tif"] = writer

    # One tiled multichannel image per codec, so the Rust decoders are checked against
    # tifffile's own output rather than against each other.
    for label, compression in (
        ("none", None),
        ("deflate", "deflate"),
        ("lzw", "lzw"),
        ("packbits", "packbits"),
        ("zstd", "zstd"),
    ):
        def writer(path, compression=compression):
            data = np.asarray(
                [
                    [[(c * 90 + y * 5 + x * 3) % 256 for x in range(70)] for y in range(50)]
                    for c in range(3)
                ],
                dtype=np.uint8,
            )
            tifffile.imwrite(
                path,
                data,
                photometric="minisblack",
                tile=(32, 32),
                compression=compression,
                metadata={"axes": "CYX"},
                ome=True,
            )

        named[f"codec_{label}.ome.tif"] = writer

    def codec_jpeg(path):
        # JPEG tiles share one JPEGTables table, which the decoder has to splice in.
        tifffile.imwrite(
            path,
            rgb_planes(64, 96),
            photometric="rgb",
            planarconfig="contig",
            tile=(32, 32),
            compression="jpeg",
            compressionargs={"level": 95},
            ome=True,
        )

    named["codec_jpeg.ome.tif"] = codec_jpeg

    def predictor_deflate(path):
        # Horizontal differencing on 16-bit samples: the predictor has to be undone per
        # sample, not per byte.
        data = np.asarray(
            [[[(c * 8000 + y * 300 + x * 17) % 65536 for x in range(70)] for y in range(50)]
             for c in range(2)],
            dtype=np.uint16,
        )
        tifffile.imwrite(
            path,
            data,
            photometric="minisblack",
            tile=(32, 32),
            compression="deflate",
            predictor=True,
            metadata={"axes": "CYX"},
            ome=True,
        )

    named["predictor_deflate.ome.tif"] = predictor_deflate
    return named


def tag_dump(page) -> dict:
    """The handful of tags the plugin consults, as plain JSON.

    Pages after the first are `TiffFrame`s, which carry no tags of their own and defer to
    the keyframe — exactly as the plugin does when it reads `keyframe.tags`.
    """
    tags = getattr(page, "tags", None)
    if tags is None:
        tags = page.keyframe.tags
    out = {}
    for name in ("XResolution", "YResolution", "ResolutionUnit", "ImageDescription"):
        tag = tags.get(name)
        if tag is None:
            continue
        value = tag.value
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, bytes):
            value = base64.b64encode(value).decode()
        out[name] = value
    return out


def page_dump(page) -> dict:
    keyframe = page.keyframe
    try:
        array = page.asarray()
        pixels = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode(),
        }
    except Exception as exc:  # pragma: no cover - recorded so the Rust side can skip
        pixels = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "index": page.index,
        "offset": page.offset,
        "imagewidth": int(keyframe.imagewidth),
        "imagelength": int(keyframe.imagelength),
        "imagedepth": int(keyframe.imagedepth),
        "bitspersample": int(keyframe.bitspersample),
        "samplesperpixel": int(keyframe.samplesperpixel),
        "photometric": int(keyframe.photometric),
        "planarconfig": int(keyframe.planarconfig),
        "compression": int(keyframe.compression),
        "predictor": int(getattr(keyframe, "predictor", 1) or 1),
        "is_tiled": bool(keyframe.is_tiled),
        "tilewidth": int(keyframe.tilewidth),
        "tilelength": int(keyframe.tilelength),
        "tiledepth": int(keyframe.tiledepth),
        "dtype": np.dtype(keyframe.dtype).str,
        "rowsperstrip": int(getattr(keyframe, "rowsperstrip", 0) or 0),
        "dataoffsets": [int(v) for v in page.dataoffsets],
        "databytecounts": [int(v) for v in page.databytecounts],
        "jpegtables": (
            base64.b64encode(bytes(jpegtables.value)).decode()
            if (jpegtables := getattr(page, "jpegtables", None)) is not None
            else None
        ),
        "description": keyframe.description or "",
        "tags": tag_dump(page),
        "pixels": pixels,
    }


def series_dump(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = []
        for level in series.levels:
            levels.append(
                {
                    "axes": str(level.axes),
                    "shape": [int(v) for v in level.shape],
                    "pages": [page_dump(page) for page in level.pages],
                }
            )
        imagej = tif.imagej_metadata or {}
        # LUTs and Labels are what the channel metadata is read from; the rest of the
        # ImageJ header is noise for this plugin.
        imagej_dump = {}
        if "Labels" in imagej:
            imagej_dump["Labels"] = [str(v) for v in imagej["Labels"]]
        if "LUTs" in imagej:
            imagej_dump["LUTs"] = [
                base64.b64encode(np.asarray(lut, dtype=np.uint8).tobytes()).decode()
                for lut in imagej["LUTs"]
            ]
        for key in ("spacing", "unit", "channels", "slices", "frames", "images", "hyperstack"):
            if key in imagej:
                value = imagej[key]
                imagej_dump[key] = value if isinstance(value, (int, float, str, bool)) else str(value)

        return {
            "is_ome": bool(tif.is_ome),
            "is_imagej": bool(tif.is_imagej),
            "is_bigtiff": bool(tif.is_bigtiff),
            "byteorder": tif.byteorder,
            "ome_metadata": tif.ome_metadata if tif.is_ome else None,
            "imagej_metadata": imagej_dump,
            "series_axes": str(series.axes),
            "series_shape": [int(v) for v in series.shape],
            "page_count": len(series.pages),
            "levels": levels,
        }


def main() -> None:
    manifest = {}
    for name, writer in sorted(fixtures().items()):
        path = HERE / name
        if path.exists():
            path.unlink()
        writer(path)
        manifest[name] = series_dump(path)
        print(f"{name}: {path.stat().st_size} bytes")

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"manifest.json: {(HERE / 'manifest.json').stat().st_size} bytes")


if __name__ == "__main__":
    main()
