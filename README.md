# Tile-chan

Tile-chan is a Rust re-implementation of the Python `wsi_service` in this repository. Same
URLs, same parameters, same status codes, same response bodies — the goal is a drop-in
replacement, so no client needs to change.

Where the two cannot be identical, or where the Python has a defect worth preserving, it is
recorded in [`docs/divergences.md`](docs/divergences.md) rather than left to be discovered.

## Status

All five format plugins are ported, and the port is verified against the Python
service end to end.

| Area | State |
|---|---|
| Settings, models, errors | done |
| Image pipeline (conversion, padding, encoding, validation) | done |
| ICC profile handling | done |
| Plugin registry, priority and selection | done |
| Slide handle cache (LRU + inactivity expiry) | done |
| Storage resolution (local mapper and storage-mapper service) | done |
| HTTP surface — every `/v3` route, `/alive`, viewer, OpenAPI | done |
| Streaming ZIP download, batch ZIP responses | done |
| Mappers: `PathsMapper`, `SimpleMapper`, `CSVMapper`, `IteratorMapper` | done |
| Auth: `Default`, `DisableAuth`, EMPAIA, LSAAI groups, LSAAI resources | done |
| Pillow-exact resampler (thumbnail, label, macro) | done |
| Plugin: `pil` | done |
| Plugin: `openslide` — `.svs .ndpi .mrxs .scn .bif .tif .tiff .czi`, DICOM dirs | done |
| Differential parity harness | done — **run clean against the Python service** |
| Docker image, compose, CI | done |
| Plugin: `nifti` — `.nii .nii.gz`, NIfTI-1 and NIfTI-2 | done |
| Plugin: `mvt` — vector/JSON tiles, `.mbtiles`, manifests, `z/x/y` trees | done |
| Plugin: `tifffile` — OME-TIFF, ImageJ and generic multichannel/Z TIFFs | done |
| Plugin: `tiffslide` — Aperio SVS and generic pyramidal TIFF | done |
| Plugin: `wsidicom` — DICOM VL Whole Slide Microscopy directories | done |

All five format plugins are ported. 504 tests pass; clippy and rustfmt are clean.
Coverage is unit tests throughout, 34 end-to-end tests over the assembled router, 32 tests
over TIFF fixtures written by `tifffile` itself, and pixel-level tests against real
whole-slide images for `openslide`, `tiffslide` and `wsidicom` — the last replaying
wsidicom's own output over three real DICOM directories.

Those tests each compare the port against one Python *library*. The differential harness
compares the two *services*, and has been run over six slides covering every plugin, in
three configurations, all clean. It found six faults that no per-plugin test could have:

- Pillow's reduce pre-pass multiplies by a truncated single-precision reciprocal instead of
  dividing, and its resample box is narrowed to `float` before any coefficient is computed —
  together these made every thumbnail, label and macro response differ.
- Validation bodies were missing `input`, `ctx` and `url`, and reported only the first
  failing parameter where FastAPI reports them all.
- `image_quality=0` was refused where libjpeg clamps it to 1.
- One unreadable ICC profile has to fail a whole batch request rather than one entry.
- A missing ICC profile answers 412 with an `Error: ` prefix that Python's doubled exception
  handling adds, and `tifffile`'s strict path has two different failures depending on
  whether the page carries the tag at all.

It also found a defect in the Python service that the port deliberately does not reproduce:
`/slides/label` and `/slides/macro` resize **in place**, so where a plugin returns a cached
image — `wsidicom` does — every request shrinks it permanently and the response depends on
what was asked before (divergence 38).

### Verified against `tifffile`

The `tifffile` plugin needs to agree with the Python one about what a TIFF *is* — which
axes `series[0]` has, how channels and Z planes are spread over pages, where the pyramid
levels live — and that is not derivable from the TIFF specification, only from what
`tifffile` decides. So the expectations come from `tifffile`:
`crates/wsi-plugin-tifffile/tests/fixtures/generate.py` writes 27 fixtures and records, for
each, the series axes and shape, every page's geometry, and the exact bytes
`page.asarray()` returns. `tests/reader.rs` replays that record and `tests/plugin.rs`
carries the Python suite's own expected pixels on top of it.

The fixtures are committed, so a `tifffile` upgrade cannot silently change what the tests
assert; rerunning the generator is a deliberate act and `git diff` shows what moved.

### Verified against `tiffslide`

The same method, against the real slides rather than generated ones:
`crates/wsi-plugin-tiffslide/tests/fixtures/generate.py` runs `tiffslide` over
`Aperio/CMU-1.svs`, `Generic TIFF/CMU-1.tiff` and `Hamamatsu/OS-1.ndpi` and records the
format it detects, every level's extent and downsample, the properties it synthesises, the
associated images, the ICC profile length and `read_region` samples. The record is
committed; the slides are gigabytes, so the tests skip themselves without
`WSI_TEST_SLIDES`.

Measured deviation from tiffslide's own pixels: **0** on the Aperio slide and 1 on the
generic TIFF, the latter from the JPEG decoder (divergence 28).

### Verified against real slides

The `openslide` plugin is checked against the **Python suite's own expected values**, on the
same files, rather than against whatever this implementation happens to produce. Metadata
(level counts, pixel sizes, extents), region pixels, thumbnails, labels and macros all match
exactly on Aperio SVS, generic TIFF, Hamamatsu NDPI and MIRAX MRXS.

Those tests skip themselves unless a slide collection is available:

```sh
WSI_TEST_SLIDES=/path/to/cds-test-suite \
    cargo test -p wsi-plugin-openslide --test real_slides
```

## Layout

```
crates/
  wsi-core/             settings, models, errors, Slide trait, image pipeline, resampler,
                        slide cache, plugin registry
  wsi-mappers/          PathsMapper, SimpleMapper, CSVMapper, IteratorMapper
  wsi-auth/             authentication integrations, JWT/JWKS validation
  wsi-plugin-pil/       plain raster images (.png .jpg .jpeg)
  wsi-plugin-openslide/ vendor WSI formats, via libopenslide
  wsi-plugin-tifffile/  OME-TIFF, ImageJ and generic multichannel/Z TIFFs, with the
                        TIFF reader the tiffslide and wsidicom plugins also use
  wsi-plugin-tiffslide/ Aperio SVS and generic pyramidal TIFF
  wsi-plugin-wsidicom/  DICOM VL Whole Slide Microscopy directories
  wsi-plugin-nifti/     NIfTI-1 and NIfTI-2 volumes (.nii .nii.gz)
  wsi-plugin-mvt/       vector and JSON tiles, .mbtiles, manifests, z/x/y trees
  wsi-server/           axum routers, OpenAPI, download/batch archives, the binary
  wsi-parity/           the differential harness
assets/                 viewer.html and validation_viewer.html, copied verbatim
docs/divergences.md     every intentional and reproduced difference from Python
docs/followup-plan.md   how each plugin was ported, and what it cost to learn
```

## Building

The C dependencies (libjpeg-turbo, Little-CMS, the forked libopenslide) build cleanly on
Linux, which is also the deployment target, so all building and testing happens in a
container. On Windows, `dev.ps1` wraps that:

```powershell
docker build -f Dockerfile.dev -t wsi-rs-dev .
./dev.ps1 test --workspace --features wsi-server/openslide
./dev.ps1 clippy --workspace --all-targets --features wsi-server/openslide
./dev.ps1 fmt --check
```

The dev image builds libopenslide from source, so the first build takes a few minutes.

On Linux or in WSL, plain cargo works:

```sh
cargo test --workspace --features wsi-server/openslide
cargo build --release -p wsi-server --features openslide
```

### Feature flags

- `wsi-core/jpeg-turbo` *(default)* — encode JPEG through libjpeg-turbo, the same library
  Pillow links against. **Parity runs require this.** Without it a pure-Rust encoder is
  substituted so the crate still builds without a C toolchain, and JPEG output differs at
  the pixel level.
- `wsi-server/pil` *(default)* — register the `pil` plugin.
- `wsi-server/openslide` — register the `openslide` plugin. Off by default because it needs
  **libopenslide >= 4.0** at build time, which no distribution packages yet: 4.0 introduced
  the ICC profile API the Python service reaches through `openslide-python`'s
  `color_profile`. Both Docker images build it from the same fork
  (`github.com/iewchen/openslide`) the Python image uses, so the two services share a
  decoder and region pixels match by construction.

## Running

```sh
cargo run --release -p wsi-server
# or
docker compose up
```

Configuration is read from the environment and from `.env` in the working directory, with
the same `WS_` prefix, the same names and the same defaults as `wsi_service/settings.py` —
an existing `.env` works unchanged. A few notes:

- `WS_LOCAL_MODE` takes the Python `module:Class` descriptors verbatim:
  `wsi_service.paths_mapper:PathsMapper`, `wsi_service.simple_mapper:SimpleMapper`,
  `wsi_service.csv_mapper:CSVMapper`,
  `wsi_service.mapper_iterator.iterator:IteratorMapper`. An unrecognised value is a startup
  error rather than a silent fallback.
- `WS_API_V3_INTEGRATION` likewise, for `disable_auth:DisableAuth`,
  `empaia:EmpaiaApiIntegration`, `lsaai_groups:LSAAIIntegration` and
  `lsaai_resources:LSAAIIntegration`. Unset means the fail-closed `Default` integration,
  which rejects every request with 401 — the same behaviour as Python.
- `CSWS_*` configures the CSV mapper and `WSIT_*` the iterator mapper, as in Python.
- `WS_PORT` (or `PORT`) selects the listen port; the default is 8080.
- `WS_LENIENT_PADDING_COLOR` is **not** a Python setting. It opts out of reproducing the
  three-digit `padding_color` crash; see divergence 1.

## API

Unchanged from the Python service. The root application serves `/alive`, the viewer pages
and `/openapi.json`; `/v3` is a nested application with its own document at
`/v3/openapi.json`.

```
GET /alive
GET /slides/{slide_id}/viewer
GET /validation_viewer                      (local mode only)

GET /v3/slides/info                          ?slide_id=&plugin=
GET /v3/slides/thumbnail/max_size/{max_x}/{max_y}
GET /v3/slides/label/max_size/{max_x}/{max_y}
GET /v3/slides/macro/max_size/{max_x}/{max_y}
GET /v3/slides/region/level/{level}/start/{start_x}/{start_y}/size/{size_x}/{size_y}
GET /v3/slides/tile/level/{level}/tile/{tile_x}/{tile_y}
GET /v3/slides/download
GET /v3/slides/icc_profile

GET /v3/files/…                              ?paths=a,b,c
GET /v3/batch/…                              ?slides=a,b,c   (legacy spelling)

GET /v3/cases/                               ?context=       (local mode only)
GET /v3/cases/slides/                        ?case_id=
GET /v3/slides                               ?slide_id=
GET /v3/slides/storage                       ?slide_id=
GET /v3/refresh_local_mapper
```

## Verifying against Python

`wsi-parity` runs the same requests against both services and reports every difference. It
discovers slides from the services themselves, then builds a request matrix per slide:
every endpoint, every pyramid level, in-bounds/edge/out-of-bounds coordinates, every output
format and quality, channel selections, z-stack indices, padding colours and ICC intents.

```sh
docker compose --profile parity up -d      # rust on :8080, python on :8081
cargo run -p wsi-parity -- \
    --python-url http://localhost:8081 \
    --rust-url   http://localhost:8080 \
    --context .  --report parity-report.json
```

It exits non-zero on any unexplained difference. Responses are compared by kind rather than
byte-for-byte where an encoder difference would otherwise drown the signal — JSON
structurally, lossless images by pixel, planar TIFFs by their IFD tags and inflated strips,
archives entry by entry — and each relaxation is justified in `docs/divergences.md`. Known
divergences are allowlisted by label, and every allowlist entry carries the reason it exists.

Run it three times, because the right allowance depends on which decoder is in play and
both allowances default to exact — the looser configuration has to be asked for explicitly:

| pass | configuration | allowance | result |
|---|---|---|---|
| A | default, openslide serving the pyramidal slides | exact | 595 matched, 0 differed |
| B | `WS_PLUGIN_PRIORITY_OPENSLIDE=-1`, so `tiffslide` serves them | `--lossless-delta 1 --jpeg-delta 16` | 436 matched, 0 differed |
| C | a DICOM slide, served by `wsidicom` | `--lossless-delta 1 --jpeg-delta 16` | 70 matched, 0 differed |

Pass A is where the pixels are actually pinned: both services share a decoder there, so
every sample has to agree. B and C are the configurations where they do not — the matrix
never asks for a plugin by name, so disabling openslide is the only way to reach
`tiffslide`, and a DICOM slide is the only way to reach `wsidicom`. In both, the JPEG
decoders differ by a measured one per channel, which divergence 28 explains and quantifies.
Keeping the DICOM slide under its own `--context` is what lets pass A stay exact.

[`docs/followup-plan.md`](docs/followup-plan.md) records all three runs, how to
reproduce them, and the facts about the Python service that were expensive to establish.

Pointing both URLs at the same service is a useful self-check that the matrix runs clean:

```sh
cargo run -p wsi-parity -- --python-url http://localhost:8080 \
                           --rust-url http://localhost:8080 --context .
```
