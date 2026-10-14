"""Records what `wsidicom` reports about the real DICOM directories.

The plugin is written against the `WsiDicom` object: the pyramid it derives, the `mpp` it
reads, the tile size it reports, the frames `read_region`/`read_tile` return, and the
label, overview and thumbnail images. None of that is guessable from the standard alone —
which instances become levels, and in what order, is `wsidicom`'s decision — so the
expectations are taken from it directly.

The directories are hundreds of megabytes and cannot be committed; this record can, and
`tests/real_slides.rs` skips itself when the collection is absent.

Run it in a container so nothing is installed on the host:

    docker run --rm -v "<repo>/rust:/work" -v "<slides>:/slides:ro" python:3.12-slim \
        sh -c "pip install -q wsidicom && python /work/crates/wsi-plugin-wsidicom/tests/fixtures/generate.py /slides /work/crates/wsi-plugin-wsidicom/tests/fixtures/wsidicom.json"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from wsidicom import WsiDicom

# The three DICOM WSI directories in the collection.
SLIDES = [
    "DICOM_Leica/f990df0c-5e8e-16e5-000f-2874470dcd3a_050136",
    "DICOM_wsidicomizer/DICOM_Aperio",
    "DICOM_wsidicomizer/DICOM_Mirax",
]

# `(level index into slide.levels, (x, y) in level pixels, (width, height))`.
REGIONS = [
    (0, (0, 0), (16, 16)),
    (0, (1000, 1000), (32, 24)),
    (1, (0, 0), (16, 16)),
    (1, (512, 512), (32, 24)),
]

# `(level index, (tile_x, tile_y))`
TILES = [(0, (0, 0)), (0, (2, 3)), (1, (1, 1))]


def corners(image, width: int, height: int) -> dict:
    array = np.asarray(image.convert("RGB"))
    points = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (width // 2, height // 2),
    ]
    return {
        f"{x},{y}": [int(v) for v in array[y, x]]
        for x, y in points
        if y < array.shape[0] and x < array.shape[1]
    }


def slide_dump(path: Path) -> dict:
    with WsiDicom.open(path) as slide:
        levels = []
        for index, level in enumerate(slide.levels):
            levels.append(
                {
                    "index": index,
                    # `level.level` is the *exponent*: the plugin reports 2 ** it.
                    "level": int(level.level),
                    "width": int(level.size.width),
                    "height": int(level.size.height),
                    "tile_width": int(level.tile_size.width),
                    "tile_height": int(level.tile_size.height),
                    "mpp_width": float(level.mpp.width) if level.mpp else None,
                    "mpp_height": float(level.mpp.height) if level.mpp else None,
                }
            )

        regions = []
        for level_index, location, size in REGIONS:
            if level_index >= len(slide.levels):
                continue
            entry = {"level_index": level_index, "location": list(location), "size": list(size)}
            try:
                dicom_level = slide.levels[level_index].level
                image = slide.read_region(location, dicom_level, size)
                entry["mode"] = image.mode
                entry["samples"] = corners(image, size[0], size[1])
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            regions.append(entry)

        tiles = []
        for level_index, tile in TILES:
            if level_index >= len(slide.levels):
                continue
            entry = {"level_index": level_index, "tile": list(tile)}
            try:
                dicom_level = slide.levels[level_index].level
                image = slide.read_tile(dicom_level, tile)
                entry["size"] = list(image.size)
                entry["mode"] = image.mode
                entry["samples"] = corners(image, image.size[0], image.size[1])
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            tiles.append(entry)

        associated = {}
        for name, reader in (
            ("label", slide.read_label),
            ("overview", slide.read_overview),
        ):
            try:
                image = reader()
                associated[name] = {
                    "size": list(image.size),
                    "mode": image.mode,
                    "samples": corners(image, image.size[0], image.size[1]),
                }
            except Exception as exc:
                associated[name] = {"error": f"{type(exc).__name__}: {exc}"}

        thumbnail = {}
        try:
            image = slide.read_thumbnail((500, 500))
            thumbnail = {
                "requested": [500, 500],
                "size": list(image.size),
                "mode": image.mode,
                "samples": corners(image, image.size[0], image.size[1]),
            }
        except Exception as exc:
            thumbnail = {"error": f"{type(exc).__name__}: {exc}"}

        profile = None
        for optical_path in slide.metadata.optical_paths:
            if optical_path.icc_profile:
                profile = len(bytes(optical_path.icc_profile))
                break

        return {
            "level_count": len(slide.levels),
            "levels": levels,
            "regions": regions,
            "tiles": tiles,
            "associated": associated,
            "thumbnail": thumbnail,
            "profile_length": profile,
        }


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    import wsidicom as module

    record = {"wsidicom_version": getattr(module, "__version__", "unknown"), "slides": {}}
    for relative in SLIDES:
        path = root / relative
        if not path.is_dir():
            print(f"skipping {relative}: not present")
            continue
        print(f"reading {relative}")
        try:
            record["slides"][relative] = slide_dump(path)
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}")
            record["slides"][relative] = {"error": f"{type(exc).__name__}: {exc}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
