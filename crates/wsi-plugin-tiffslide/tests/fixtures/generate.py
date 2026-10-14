"""Records what `tiffslide` reports about the real slides, as the port's expectations.

`tiffslide` is a Python reimplementation of openslide's interface on top of `tifffile`, and
the plugin is written against *its* view of a slide: the vendor it detects, the `mpp`
properties it synthesises, the level pyramid it derives, the associated images it exposes,
and the pixels `read_region` returns. None of that follows from the TIFF specification, so
guessing at it is how a port ends up subtly wrong.

The slides themselves cannot be committed — they are gigabytes — so this writes a JSON
record that can be, and `tests/real_slides.rs` reads it and skips itself when the slide
collection is not present.

Run it in a container so nothing is installed on the host:

    docker run --rm -v "<repo>/rust:/work" -v "<slides>:/slides:ro" python:3.12-slim \
        sh -c "pip install -q tiffslide && python /work/crates/wsi-plugin-tiffslide/tests/fixtures/generate.py /slides /work/crates/wsi-plugin-tiffslide/tests/fixtures/tiffslide.json"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tiffslide

# The three slides the Python plugin suite runs against.
SLIDES = [
    "Aperio/CMU-1.svs",
    "Generic TIFF/CMU-1.tiff",
    "Hamamatsu/OS-1.ndpi",
]

# `(level, (x, y), (width, height))` — `read_region` takes a *level 0* location.
REGIONS = [
    (0, (0, 0), (16, 16)),
    (0, (1000, 1000), (32, 24)),
    (1, (0, 0), (16, 16)),
    (1, (4096, 4096), (32, 24)),
]

# Where to sample an associated image or a thumbnail.
SAMPLE_POINTS = [(0, 0), (5, 5), (10, 3)]


def sample(image, points) -> list[list[int]]:
    array = np.asarray(image.convert("RGB"))
    out = []
    for x, y in points:
        if y < array.shape[0] and x < array.shape[1]:
            out.append([int(v) for v in array[y, x]])
        else:
            out.append([])
    return out


def region_dump(slide, level, location, size) -> dict:
    image = slide.read_region(location, level, size)
    array = np.asarray(image)
    return {
        "level": level,
        "location": list(location),
        "size": list(size),
        "mode": image.mode,
        "shape": list(array.shape),
        # The corners and the centre are enough to catch an origin or stride mistake.
        "samples": {
            f"{x},{y}": [int(v) for v in array[y, x]]
            for x, y in [
                (0, 0),
                (size[0] - 1, 0),
                (0, size[1] - 1),
                (size[0] - 1, size[1] - 1),
                (size[0] // 2, size[1] // 2),
            ]
        },
    }


def series_dump(path: Path) -> dict:
    """How `tifffile` splits and names the series.

    `associated_images` is `series[1:]` keyed by `series.name.lower()`, so the port has to
    reproduce tifffile's splitting — which page belongs to the pyramid, which is a label,
    which a macro — and its names. The page list is recorded alongside so the rule can be
    derived rather than guessed.
    """
    import tifffile as tf

    with tf.TiffFile(path) as file:
        pages = []
        for page in file.pages:
            keyframe = getattr(page, "keyframe", page)
            description = getattr(keyframe, "description", "") or ""
            pages.append(
                {
                    "index": int(page.index),
                    "width": int(keyframe.imagewidth),
                    "height": int(keyframe.imagelength),
                    "is_tiled": bool(keyframe.is_tiled),
                    "photometric": int(keyframe.photometric),
                    "compression": int(keyframe.compression),
                    "subfiletype": int(
                        (getattr(keyframe, "tags", None) or keyframe.keyframe.tags)
                        .get("NewSubfileType", None).value
                        if (getattr(keyframe, "tags", None) or keyframe.keyframe.tags).get(
                            "NewSubfileType", None
                        )
                        is not None
                        else 0
                    ),
                    # The descriptions are how a label or macro announces itself; they can
                    # be long, so only the leading part is recorded.
                    "description_head": description[:160],
                    # NDPI marks what a page is with `SourceLens` (65421): positive for a
                    # pyramid level, negative for the macro and the map.
                    "ndpi_tags": {
                        str(code): str(tag.value)[:80]
                        for code in (65420, 65421, 65422, 65423, 65424, 65425, 65426, 65427)
                        if (
                            tag := (
                                getattr(keyframe, "tags", None) or keyframe.keyframe.tags
                            ).get(code)
                        )
                        is not None
                    },
                    "software": str(getattr(keyframe, "software", "") or "")[:60],
                }
            )

        series = []
        for item in file.series:
            series.append(
                {
                    "name": str(item.name),
                    "axes": str(item.axes),
                    "shape": [int(v) for v in item.shape],
                    "level_count": len(item.levels),
                    "level_shapes": [[int(v) for v in lvl.shape] for lvl in item.levels],
                    "page_indices": [
                        [int(p.index) for p in lvl.pages] for lvl in item.levels
                    ],
                }
            )
        return {"is_svs": file.is_svs, "page_count": len(file.pages), "pages": pages, "series": series}


def slide_dump(path: Path) -> dict:
    with tiffslide.TiffSlide(path) as slide:
        properties = {
            str(key): (value if isinstance(value, (int, float, str, bool)) or value is None
                       else str(value))
            for key, value in slide.properties.items()
        }
        associated = {}
        for name in slide.associated_images:
            image = slide.associated_images[name]
            associated[name] = {
                "size": list(image.size),
                "mode": image.mode,
                "samples": sample(image, SAMPLE_POINTS),
            }

        profile = getattr(slide, "_profile", None)

        regions = []
        for level, location, size in REGIONS:
            if level >= slide.level_count:
                continue
            try:
                regions.append(region_dump(slide, level, location, size))
            except Exception as exc:  # pragma: no cover - recorded so Rust can skip
                regions.append(
                    {
                        "level": level,
                        "location": list(location),
                        "size": list(size),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return {
            "format": slide.detect_format(str(path)),
            "dimensions": list(slide.dimensions),
            "level_count": int(slide.level_count),
            "level_dimensions": [list(dim) for dim in slide.level_dimensions],
            "level_downsamples": [float(value) for value in slide.level_downsamples],
            "properties": properties,
            "associated_images": associated,
            "profile_length": None if profile is None else len(bytes(profile)),
            "regions": regions,
            "tifffile": series_dump(path),
        }


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    record = {"tiffslide_version": tiffslide.__version__, "slides": {}}
    for relative in SLIDES:
        path = root / relative
        if not path.exists():
            print(f"skipping {relative}: not present")
            continue
        print(f"reading {relative}")
        record["slides"][relative] = slide_dump(path)
    out.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
