# WSI-Service Rust port — implementation notes

## Context

`rust/` in this repository holds a Rust re-implementation of the Python `wsi_service`,
intended as a drop-in replacement: same URLs, parameters, status codes, response bodies, and
pixel-identical images. **It is complete**: all five format plugins are ported, and the
differential harness compares the two services clean in three configurations.

This document began as a plan and is now the record. Its most valuable part is *Knowledge
that must not be rediscovered* — facts about the Python service and the libraries under it
that were found only by comparing against real slides, each one having cost real effort.
Anyone extending the port, or explaining why it behaves as it does, should start there and
in `divergences.md`.

Where to look:

1. `rust/README.md` for the surface and how to run it; `rust/docs/divergences.md` for the
   behavioural contract, which is non-negotiable context.
2. The work lives on `feat/rust-service`, roughly one commit per plugin, with the fixes the
   parity run forced split out by cause.

---

## Current state

Clippy and rustfmt clean; 12 crates. The test counts are in `README.md`, which is the
document to keep current.

**How each plugin was verified matters more than that it was.** Each one got its
expectations from the Python library it replaces, recorded into a committed fixture rather
than asserted from whatever the port happened to produce:

| plugin | oracle | where |
|---|---|---|
| `openslide` | the Python plugin's own tests, on real slides | `wsi-plugin-openslide/tests/` |
| `tifffile` | `tifffile` itself, over 27 generated TIFFs | `wsi-plugin-tifffile/tests/fixtures/` |
| `tiffslide` | `tiffslide` itself, over three real slides | `wsi-plugin-tiffslide/tests/fixtures/` |
| `wsidicom` | `wsidicom` itself, over three real DICOM directories | `wsi-plugin-wsidicom/tests/fixtures/` |
| `nifti` | the Python plugin's own asserted pixel arrays | inline in `lib.rs` |

The generators run in a container, so nothing is installed on the host, and each records
enough that a mismatch names the pixel rather than just failing. The fixtures are committed
precisely so that upgrading one of those libraries cannot move the target silently —
regenerating is a deliberate act, and `git diff` shows what changed.

| Area | State |
|---|---|
| Settings, models, errors (incl. Starlette's plain-text 500) | done |
| Image pipeline — conversion, padding, encoding, validation | done |
| Pillow-exact resampler (`resize`, `thumbnail`, `reduce`) | done |
| ICC handling via lcms2 | done |
| Plugin registry, priority, selection | done |
| Slide handle cache (LRU + inactivity expiry) | done |
| Storage resolution (local mapper + storage-mapper service) | done |
| Every `/v3` route, `/alive`, viewer, both OpenAPI documents | done |
| Streaming ZIP download, batch ZIP responses | done |
| Mappers: Paths, Simple, CSV, Iterator | done |
| Auth: Default, DisableAuth, EMPAIA, LSAAI groups, LSAAI resources | done |
| Plugin `pil` | done |
| Plugin `openslide` — verified pixel-exact on real slides | done |
| Plugin `nifti` | done |
| Plugin `mvt` | done |
| Plugin `tifffile` — verified against tifffile's own output | done |
| Plugin `tiffslide` — verified against tiffslide's own output | done |
| Differential parity harness (`wsi-parity`) | done |
| Docker images, compose, CI | done |
| Plugin `wsidicom` — verified against wsidicom's own output | done |

### Crate layout

```
rust/crates/
  wsi-core/             settings, models, errors, Slide trait, image pipeline,
                        resampler, slide cache, plugin registry
  wsi-mappers/          Paths / Simple / CSV / Iterator mappers
  wsi-auth/             auth integrations, JWT/JWKS
  wsi-plugin-pil/       .png .jpg .jpeg
  wsi-plugin-openslide/ vendor WSI formats via libopenslide (FFI in src/ffi.rs)
  wsi-plugin-tifffile/  OME-TIFF / ImageJ / generic TIFFs, plus the TIFF reader in
                        src/tiff/ that tiffslide and wsidicom reuse
  wsi-plugin-tiffslide/ Aperio SVS, generic pyramidal TIFF
  wsi-plugin-wsidicom/  DICOM directories; src/dicom.rs is the reader, src/reader.rs
                        the WsiDicom surface the plugin uses
  wsi-plugin-nifti/     NIfTI-1 / NIfTI-2, hand-written reader for bounded lazy reads
  wsi-plugin-mvt/       vector/JSON tiles, .mbtiles via rusqlite
  wsi-server/           axum routers, OpenAPI, download/batch archives, binary
  wsi-parity/           differential harness
```

---

## Knowledge that must not be rediscovered

### 1. Pillow's resampler is ported, and must be used for every resize

`wsi-core/src/image/resample.rs`. **Never** use `image::DynamicImage::resize` for anything a
client sees. Pillow's `resize`/`thumbnail` differ from a generic Lanczos in three ways, all
reproduced: 22-bit fixed-point coefficients with its own rounding; a box-average `reduce()`
pre-pass at `reducing_gap=2.0`; and `thumbnail`'s floor-or-ceil aspect arithmetic. Using
Lanczos3 instead was off by 1–2 per channel *and* produced wrong dimensions.

Filters: `Filter::Bicubic` is `Image.thumbnail`'s default and what every plugin's own
`get_thumbnail` uses. The `label` and `macro` endpoints pass `LANCZOS` explicitly.
`tifffile` and `nifti` call `Image.resize(..., LANCZOS)` directly.

### 2. Unhandled Python exceptions are plain-text 500s

`wsi_service` registers no exception handlers, so anything that is not a
`fastapi.HTTPException` reaches Starlette and becomes `500`, `content-type: text/plain;
charset=utf-8`, body `Internal Server Error` — the detail text is lost. Use
`ApiError::unhandled(python_repr)`, which renders that way but keeps the `repr` for the batch
endpoints' `.err` entries. Divergence 11 lists all seven reachable cases. Note especially
that `utils/icc_profile.py` does `from http.client import HTTPException`, so a missing ICC
profile is a **500, not a 404**.

### 3. The Python test files are the oracle

`wsi_service_base_plugins/*/tests/test_plugin.py` carry concrete expected extents, level
counts, pixel sizes and **pixel values** for real slides. Lift those into Rust tests rather
than asserting whatever the port produces. That is how the openslide plugin was validated,
and every plugin after it followed the same rule — for the four with no usable Python test,
by generating the ground truth from the Python library itself into a committed fixture.

### 4. Real slides are available locally

`C:\Users\JiříHorák\OneDrive - BBMRI-ERIC\Desktop\Vis\scans\cds-test-suite` holds Aperio,
generic TIFF, Hamamatsu, Leica, MIRAX, Ventana, Philips, **OME-TIFF fluorescence**, and
DICOM. Tests read it through `WSI_TEST_SLIDES` and skip themselves when unset — see
`wsi-plugin-openslide/tests/real_slides.rs` for the pattern to copy.

### 5. Dead Python code — do not port

`wsi_service/models/v1/`, `models/v3/{annotation,fhir,clinical.py,job.py,examination.py}`,
`models/marketplace/` — about 5000 lines that nothing outside `models/` imports.

### 6. Build and test only inside the Linux container

The C dependencies do not build on this Windows host. `rust/dev.ps1` wraps a dev image that
builds libjpeg-turbo, Little-CMS and libopenslide 4.0.0 from source; named docker volumes
keep the target dir and crate registry warm.

```powershell
docker build -f Dockerfile.dev -t wsi-rs-dev .          # first build takes minutes
./dev.ps1 test --workspace --features wsi-server/openslide
./dev.ps1 clippy --workspace --all-targets --features wsi-server/openslide
./dev.ps1 fmt
```

### 7. Environment gotchas that already cost time

- **Git Bash mangles docker paths.** `-w /work` becomes `C:/Program Files/Git/work`. Run
  every `docker` command from **PowerShell**, not the Bash tool.
- **PowerShell treats native stderr as an error.** Never pipe `2>&1` from a native exe;
  assign to a variable and filter (`$out = & ./dev.ps1 test ...`).
- **Heredocs choke on large Rust sources.** Use the Write tool for new files; `python - <<PY`
  works well for surgical edits to existing ones.
- `openslide` is behind a cargo feature because it needs libopenslide ≥ 4.0 at build time.

---

## How each plugin was ported

### 1. `tifffile`

`crates/wsi-plugin-tifffile/`. The single most valuable thing this step produced is not the
code but **`tests/fixtures/`**: a fixture generator that drives `tifffile` itself and records,
for 27 files, its `series[0].axes`, its `shape`, every page's geometry, and the exact bytes
`page.asarray()` returns. `tests/reader.rs` replays that record. Anything about the reader
that drifts from tifffile fails as a concrete mismatch rather than as a plausible image.

Regenerate with `python crates/wsi-plugin-tifffile/tests/fixtures/generate.py` — the fixtures
are committed, so a tifffile upgrade cannot silently move the goalposts.

**The low-level reader** (`src/tiff/`) is written from scratch rather than layered on the
`tiff` crate. That crate does have `read_chunk`, and its codec coverage is fine, but
`seek_to_image` only walks the main IFD chain — there is no way to make a page reached
through a SubIFD the current image, and SubIFDs are exactly where a pyramidal OME-TIFF
keeps its levels. Reading directories directly (`ifd.rs`) also mirrors the Python path
(`fh.seek(offset); fh.read(bytecount); page.decode(...)`) closely enough to check line by
line.

- `ifd.rs` — classic TIFF and BigTIFF, both byte orders, SubIFDs, the inline-value rule.
- `codec.rs` — none, LZW (`weezl`), Deflate (`flate2`), PackBits, Zstd (`ruzstd`), JPEG
  with the page's shared `JPEGTables` spliced in; the horizontal predictor per sample.
  Everything else is named in a clean error.
- `page.rs` — keyframe properties, one chunk, and the whole page. `PLANARCONFIG=SEPARATE`
  is handled: the page holds `samplesperpixel` consecutive grids.
- `series.rs` — the series reconstruction, four dialects; see divergence 32 for what is
  *not* covered.

**The plugin**, ported verbatim:

- Five layout modes: `chunky-samples`, `page-per-channel`, `page-per-z`,
  `page-per-z-channel`, `single-channel` (`__inspect_series_layout`).
- The `is_rgb` fast path: chunky **and** exactly 3 samples **and** `num_z == 1` **and**
  photometric in `{RGB, YCBCR}` → serve as a Pillow image, not a channel stack.
- OME-XML: `_OME_UNIT_TO_NM` with `µm` default, `Channel` expansion by `SamplesPerPixel`,
  signed-32-bit colour decoding, the `_DEFAULT_PALETTE` cycle.
- Generic TIFF: resolution tags with `ResolutionUnit` 2 (inch, 2.54e7 nm) or 3 (cm, 1e7 nm),
  falling back to 1000 nm/px; ImageJ `Labels`/`LUTs`/`spacing`; Bio-Formats
  `Channel:N:Name=` regexes.
- Untiled tile extent: the whole level when `max(dim) <= 2048`, else 512.
- `get_thumbnail` uses `Image.resize(..., LANCZOS)` on the RGB path — and for planar data
  goes through `skimage.transform.resize` + `img_as_uint`, which is **not** Pillow's
  resampler. It was checked: `src/thumb.rs` reproduces the algorithm and divergence 29
  records what is and is not claimed about it.
- `get_label`/`get_macro` always 404.

`PlanarData` (`wsi-core/src/pixels.rs`) carries the channel stack, so the numeric
conversion, channel selection and windowing were already in place.

Three things worth knowing before touching this code again:

1. **The Python argument names lie.** `__read_region_of_page` is called as
   `(page, 0, start_y, start_x, size_y, size_x, ...)` but its parameters are named
   `start_x, start_y, size_x, size_y`. The swap holds all the way down, so the functions
   are row-major throughout and the behaviour is right; only the names are wrong.
   `src/region.rs` says `row`/`column` and documents the mapping.
2. **A tiled read can come back smaller than requested.** The output is assembled out of
   whole tiles and then sliced, so a region reaching past the right or bottom edge is
   truncated rather than padded, and `get_extended_region` pads it afterwards. The part of
   an edge tile that lies outside the image is whatever the writer stored, not the padding
   colour. Both are reproduced.
3. **`page.tags.get("ICCProfile")` returns a `TiffTag`, not bytes.** Everything ICC in this
   plugin is therefore broken in Python — divergences 30 and 31 — and the port reproduces
   the failures rather than fixing them.

**Verified**: 24 end-to-end tests in `tests/plugin.rs` carry the expectations from
`tifffile/.../tests/test_plugin.py` — pixel-exact regions on both planar configurations,
the palette fallback for nameless OME channels, `SamplesPerPixel` expansion, the four pixel
size units, signed-page padding, per-Z tiles of an untiled series, the ImageJ stacks, the
SubIFD pyramid, all five codecs, the predictor, big-endian and BigTIFF.

### 2. `tiffslide`

`wsi_service_base_plugins/tiffslide/.../slide.py`. Structurally openslide's twin, built on
step 1's reader. The plugin is thin; the work is in reproducing **tiffslide**, which is
itself a Python reimplementation of openslide's interface over `tifffile`, and whose
decisions the plugin is written against.

`crates/wsi-plugin-tiffslide/tests/fixtures/tiffslide.json` is the ground truth: what
tiffslide 4.0.0 reports for `Aperio/CMU-1.svs`, `Generic TIFF/CMU-1.tiff` and
`Hamamatsu/OS-1.ndpi` — format, dimensions, every level's size/downsample/tile extent,
every `tiffslide.*` and vendor property, the associated images with sampled pixels, the ICC
profile length, and `read_region` samples at four `(level, location, size)` combinations.
Regenerate with the command in `generate.py`'s docstring; it runs in a `python:3.12-slim`
container so nothing is installed on the host.

**A correction to what this document used to say.** It claimed a generic TIFF "fails on the
missing `mpp` property, giving 404". It does not: `_PropertyParser.recover_mpp` derives
`tiffslide.mpp-x` from the `XResolution`/`YResolution`/`ResolutionUnit` tags, so a generic
TIFF works. The recorded value for `Generic TIFF/CMU-1.tiff` is `0.0499001996007984`, i.e.
`pixel_size_nm = 49.9` — which is exactly the `50` the Python suite's
`test_get_slide_info_valid` expects, and ten times smaller than the `0.499` the same
physical slide reports through Aperio metadata. The file's resolution tag is what disagrees;
both tiffslide and this port follow it.

What the vendor check does gate: `supported_vendors = ["aperio", "hamamatsu", None]`, and
`__get_pixel_size`'s `else` branch is a dead `SlidePixelSizeNm()` that raises
`ValidationError`. So Leica SCN and Philips TIFF — which tiffslide labels with a vendor —
become 404 "Failed to gather slide infos", while Ventana does *not*, because
`parse_ventana` delegates to `parse_generic_tiff` and never sets a vendor.

The tiffslide behaviour to reproduce, all confirmed against its source (extract it the same
way: `pip install tiffslide` in a container, then read `tiffslide/tiffslide.py`):

- `detect_format` tries `is_svs`, `is_scn`, `is_bif`, `is_ndpi`, `is_philips` in that order
  and falls back to `generic-tiff`. The detectors are tifffile's:
  `is_svs` is `page0.is_tiled and description[:7] == "Aperio "`; `is_ndpi` is
  `65420 in tags and 271 in tags`; `is_scn` is page 0's description ending in `</scn>`;
  `is_bif` needs tag 700 plus `Ventana` in `Software`; `is_philips` needs
  `Software[:10] == "Philips DP"` and a description ending in `</DataObject>`.
- **`level_downsamples` is the average of the two ratios**: `((w0/w) + (h0/h)) / 2.0`, not
  the width ratio. That is why CMU-1.svs level 1 reports `4.000121536217793`.
- `tiffslide.level[n].tile-{width,height}` come from that level's page's `tilewidth` and
  `tilelength` — so an NDPI reports `4096x8`, halving per level, and a striped page reports
  whatever the strip geometry gives.
- `recover_mpp`: `scale / resolution` with `scale` = 25400 (inch), 10000 (cm), 1000 (mm),
  1 (µm), and no mpp at all for `RESUNIT.NONE`.
- `parse_aperio` runs tifffile's `svs_description_metadata`: split on `|`, the first item is
  `Header`, the rest are `key = value` pairs coerced to int/float where possible. `MPP`
  becomes both mpp values and `AppMag` the objective power.
- `parse_hamamatsu` is `parse_generic_tiff` plus tags 65421/65422/65423 and `Model`, with
  `SourceLens` as the objective power.
- `associated_images` are **series 1 and up**, keyed by `series.name.lower()` — so this
  needs tifffile's series *splitting and naming* beyond series 0, which `src/tiff/series.rs`
  does not do yet. CMU-1.svs gives `label`, `macro`, `thumbnail`; OS-1.ndpi gives `macro`
  and `map`.
- `read_region` takes a **level-0** location, divides by the level downsample with `int()`,
  clips the rectangle to the level, and pads the rest with **zeros** — black, not the
  padding colour. The plugin's `rgba_to_rgb_with_background_color` cannot recolour it
  because the image is already RGB.

Then the plugin on top:

- `get_tile` shrinks edge reads via `get_tile_width` (`wsi-core/src/models.rs::tile_read_size`,
  already implemented) before delegating to `get_region` — unlike openslide, which always
  reads a full tile.
- `__adapt_level_0_location` nudges the level-0 origin to match openslide (tiffslide issue #63).
- `priority = 1` (openslide is 10 via `.env`, so openslide normally wins).
- The raw-JPEG-tile fast path is commented out in Python — leave it out.
- `get_thumbnail` prefers the *embedded* `thumbnail` associated image and only falls back to
  reading a level.

**Outcome.** `crates/wsi-plugin-tiffslide/` serves Aperio SVS and generic pyramidal TIFF,
and `tests/real_slides.rs` checks it against the recording — format, dimensions, every
level's extent and downsample, tile extents, `mpp`, the associated images, the ICC profile
length, `read_region` samples, and the edge-tile shrink. Measured deviation from
tiffslide's own pixels:

| slide | worst channel deviation |
|---|---|
| `Aperio/CMU-1.svs` | **0** |
| `Generic TIFF/CMU-1.tiff` | 1 |

Getting the SVS to zero needed a real fix, not a tolerance: an Aperio JPEG tile is
photometric `RGB`, so its samples already *are* R, G and B, and a three-component JPEG with
no JFIF marker is otherwise assumed to be YCbCr — 239 came back as 255. `decode_jpeg` now
splices an `APP14` "Adobe" segment with transform 0 into such a stream, which is what
`imagecodecs` achieves by setting libjpeg's `jpeg_color_space`. That fix benefits the
`tifffile` plugin too.

**NDPI is refused**, not half-served: its pages are one restart-marker JPEG each and the
tile geometry tiffslide reports is synthesised from the `NDPI_MCU_STARTS` tag. Divergence
33 has the reasoning. `openslide` serves NDPI pixel-exactly at priority 10.

### 3. `wsidicom`

`wsi_service_base_plugins/wsidicom/.../slide.py`, 197 lines. The plugin is as thin as
`tiffslide`'s; all of the work is in reproducing the `WsiDicom` object underneath it. Read
it first — the surface it needs is small and specific:

| `WsiDicom` member | used for |
|---|---|
| `levels[n].level` | the DICOM level index, which is the *exponent*: `downsample = 2 ** level.level`, **not** a size ratio |
| `levels[n].size.{width,height}` | `extent` and each level's extent |
| `levels[n].tile_size.{width,height}` | `tile_extent` |
| `levels[0].mpp.{width,height}` | `pixel_size_nm = 1000 * mpp` |
| `read_region(location, level, size)` | `get_region`; `location` is in **level** pixels here, unlike tiffslide's level-0 ones, and no `__adapt_level_0_location` nudge is applied |
| `read_tile(level, (tile_x, tile_y))` | `get_tile` — a *tile index*, so no size arithmetic |
| `read_label()` / `read_overview()` | `get_label` / `get_macro`, each a 404 on `WsiDicomNotFoundError` |
| `read_thumbnail((max, max))` | `get_thumbnail`, then Pillow's `thumbnail` to the requested box |
| `metadata.optical_paths[].icc_profile` | the first non-empty one wins |

Notes that will save time:

- `is_supported` matches a **directory** containing `*.dcm`, never a file.
- `get_tile` wraps everything in `except Exception` and answers
  `500 Failed to extract tile from WSI.`, so no tile failure is ever specific.
- `get_thumbnail` caches on first call, so that call's `icc_profile_intent` fixes the
  colours for every later one — the same `hasattr` caching quirk `tiffslide` has.
- `__get_icc_bytes` memoises with `False` as "not yet looked", `None` as "looked, none
  there".

**Verification should follow the pattern the other two used**, which worked well: `pip
install wsidicom` in a `python:3.12-slim` container, run it over the real collection, and
record what it reports into `tests/fixtures/`. The collection has three usable directories:

```text
$WSI_TEST_SLIDES/DICOM_Leica/f990df0c-…_050136/          (Leica, ~880 MB)
$WSI_TEST_SLIDES/DICOM_wsidicomizer/DICOM_Aperio/        (~176 MB)
$WSI_TEST_SLIDES/DICOM_wsidicomizer/DICOM_Mirax/         (~205 MB)
```

**The two questions that decide the scope are already answered.** Every frame set in all
three directories was surveyed with `pydicom`:

```text
f990df0c-…_050136  THUMBNAIL 737x1080      1 frame   TILED_FULL  JPEG baseline
                   LABEL     601x610       1 frame   TILED_FULL  explicit VR (uncompressed)
                   OVERVIEW  1463x598      1 frame   TILED_FULL  JPEG baseline
                   VOLUME    60246x88230   81420     TILED_FULL  JPEG baseline  (+2 resampled)
DICOM_Aperio       VOLUME    46000x32914   23220     TILED_FULL  JPEG baseline  (+2 resampled)
                   LABEL / OVERVIEW        1 frame   TILED_FULL  JPEG baseline
DICOM_Mirax        VOLUME    94794x179005  65100     TILED_FULL  JPEG baseline  (+9 resampled)
                   LABEL / OVERVIEW        1 frame   TILED_FULL  JPEG baseline
```

1. **Transfer syntax: JPEG baseline throughout**, with one uncompressed label. No JPEG 2000
   and no JPEG-LS, so **no new C dependency is needed** — `wsi-plugin-tifffile`'s
   `codec.rs` already decodes JPEG, including the `APP14` handling that photometric-RGB
   tiles need, and that path measured *zero* deviation against libjpeg-turbo on the Aperio
   slide. The parity ceiling here is therefore much better than divergence 12 assumed.
2. **Tiling: `TILED_FULL` throughout**, so frames are laid out implicitly in row-major
   order and the per-frame functional groups never have to be read.
3. **`PhotometricInterpretation` varies, and it is what `decode_chunk` needs.** It takes
   the *TIFF* photometric number, so the mapping has to be right or every tile comes back
   with shifted colours. Surveyed per instance:

   | directory | flavour | photometric | TIFF number |
   |---|---|---|---|
   | Leica | VOLUME, THUMBNAIL | `YBR_FULL_422` | 6 (YCbCr) |
   | Leica | LABEL, OVERVIEW | `RGB` | 2 (RGB) |
   | Aperio | all | `YBR_FULL` | 6 (YCbCr) |
   | MIRAX | all | `YBR_FULL` | 6 (YCbCr) |

   The Leica `OVERVIEW` is the interesting one: photometric `RGB` *and* JPEG, which is
   exactly the case the `APP14` splice in `codec.rs` exists for (divergence 28). Its
   `LABEL` is the uncompressed instance, so photometric 2 with compression 1.

   Leica also carries an ICC profile of **13,113,264 bytes** on every instance; Aperio and
   MIRAX carry none. A 13 MB profile is worth knowing about before it is read eagerly.

What is left to build is then a focused DICOM reader rather than a general one:

- the file meta group, to get the transfer syntax, then explicit- and implicit-VR little
  endian element parsing;
- the tags the plugin needs: `ImageType`, `TotalPixelMatrixColumns`/`Rows`, `Rows`,
  `Columns`, `NumberOfFrames`, `PixelSpacing` (in
  `SharedFunctionalGroupsSequence` → `PixelMeasuresSequence`), and `ICCProfile` inside
  `OpticalPathSequence`;
- encapsulated `PixelData`: the basic offset table and the item-delimited fragments, one
  frame per item in these files;
- grouping the instances by `ImageType` — `VOLUME` into the pyramid, sorted by size, with
  `LABEL`, `OVERVIEW` and `THUMBNAIL` set aside — and deriving each level's index so that
  `2 ** level` is the downsample the plugin reports.

Scope it to `TILED_FULL` + JPEG/uncompressed and refuse anything else with a clear message,
exactly as `tiffslide` refuses NDPI. Anything richer can be added when a file needs it.

**Pixel parity cannot be assumed here** and divergence 12 already says so: `dicom-rs` is an
independent implementation. Measure the deviation and record it, as divergence 28 now does
for JPEG.

Also worth knowing: `openslide` claims `.dcm` directories at **priority 10** and the port's
openslide plugin serves them through the same libopenslide the Python service uses, so DICOM
is already pixel-exact in the default configuration. Like `tiffslide`, this plugin matters
for an explicit `plugin=wsidicom` request or a build without libopenslide.

### 4. `nifti`

`crates/wsi-plugin-nifti/`. The `nifti` crate was not used in the end: the plugin needs
bounded *lazy* reads (`_get_window` scans the whole volume in blocks; a tile reads only its
own rows), and the crate's readers materialise the array. `src/reader.rs` parses the
NIfTI-1 and NIfTI-2 single-file headers directly and reads runs of `x` for one `(y, z)` as
one contiguous read, which is what nibabel's `ArrayProxy` costs. Gzipped volumes are
inflated once into memory, as nibabel does without `indexed_gzip`.

Every item below is implemented and pinned by a test ported from
`nifti/.../tests/test_plugin.py`; the notes are kept because they are the contract:

- Guards: 2D/3D scalar only, singleton axes beyond 3, dtype kind in `uif`.
- `_read` transposes: `dataobj[x, y, z].T`, so output is `(height, width)`.
- `scl_slope`/`scl_inter` must be applied (nibabel does it through the proxy).
- Window from `cal_min`/`cal_max` when finite and increasing; otherwise **scan the whole
  volume** in blocks for min/max of finite values, shared across all tiles and z layers.
- Rendering uses `np.rint` — **round-half-to-even**, not the truncation the main pipeline
  uses. `127.5 -> 128`, `0.5 -> 0`.
- Non-finite voxels render black; a constant or all-NaN volume renders all black.
- Units `{meter: 1e9, mm: 1e6, micron: 1e3}`, else `-1`.
- `tile_extent = min(dim, 512)`; the level extent carries `z = depth`, not 1.
- ICC: intent + strict → 412; intent alone → ignored.

One trap worth recording: `test_plugin.py` builds its volumes with `np.arange(...).reshape(...)`,
which is C order, but NIfTI stores the **first** axis fastest. A fixture that writes the
reshaped bytes straight to disk therefore holds a different volume than the Python test
does, and the expected pixels silently stop meaning anything. `arange_volume` in
`lib.rs`'s tests writes `(x, y, z)` in file order for that reason.

Also note `get_thumbnail` ends with `Image.resize(..., LANCZOS)`, and `resize`'s
`reducing_gap` defaults to `None` — unlike `thumbnail`, which defaults to `2.0`. Passing
`2.0` here would add a box-average pre-pass Python never runs.

### 5. `mvt`

`crates/wsi-plugin-mvt/`. `rusqlite` with the `bundled` feature for `.mbtiles`, so SQLite
is compiled in and the runtime image needs no sqlite package; plus directory
(`z/x/y.ext`), manifest-JSON and single-file datasets; TMS/XYZ y-flip; gzip passthrough;
everything is `Pixels::Raw`. `get_region` → 501, `get_thumbnail` → 501, label/macro → 404,
`get_icc_profile` → empty bytes, which the route then rejects as divergence 11's
plain-text 500.

The Python plugin has **no tests** (`tests/test_plugin.py` is two lines and a `# TODO`),
so the port's 19 tests are the first coverage this plugin has ever had. They pin the
manifest key precedence, which is easy to get wrong: `bounds_by_zoom` and the dict form of
`levels` *overwrite* an entry, while the integer form of `levels`, `zoom_levels` and the
`min_zoom`/`max_zoom` span only fill gaps.

As decided, it is implemented *working*: Python's `_build_slide_info` omits the required
`format` field, so `SlideInfo(...)` raises `ValidationError` and the plugin returns 500 for
every dataset — it has never served a tile. The port supplies `format = "mvt"`, recorded as
divergence 26. Divergence 27 covers the case-insensitive extension match the port has
always used for `supported_file_extensions` plugins.

### 6. Wiring up

Each plugin is registered in `wsi-server/src/lib.rs::build_registry` behind a cargo feature
and enabled in `rust/Dockerfile` and `.github/workflows/rust.yml` — where the `image` job
asserts every one of the seven appears in `/alive`, which for `openslide` also proves the
library loaded.

The rule the port was held to, and which anything extending it should keep: a difference the
parity harness tolerates needs an entry in `rust/docs/divergences.md` naming the Python
source line it comes from. The allowlist is checked against that document, not against
convenience.

---

## Verification

Per-step, in the dev container:

```powershell
cd rust
./dev.ps1 test --workspace --features wsi-server/openslide
./dev.ps1 clippy --workspace --all-targets --features wsi-server/openslide
./dev.ps1 fmt --check
```

Against real slides (adjust the feature list as plugins land):

```powershell
$data = "C:\Users\JiříHorák\OneDrive - BBMRI-ERIC\Desktop\Vis\scans\cds-test-suite"
docker run --rm -v "${PWD}:/work" -v wsi-rs-target:/target `
  -v wsi-rs-registry:/usr/local/cargo/registry -v "${data}:/slides:ro" `
  -w /work -e CARGO_TARGET_DIR=/target -e WSI_TEST_SLIDES=/slides `
  wsi-rs-dev cargo test -p wsi-plugin-tifffile --test real_slides
```

End to end, through the production image:

```powershell
docker build -f Dockerfile -t wsi-rs:test .
docker run -d --name wsi-t -p 18080:8080 -v "${data}:/data:ro" `
  -e WS_LOCAL_MODE=wsi_service.paths_mapper:PathsMapper `
  -e WS_API_V3_INTEGRATION=wsi_service.api.v3.integrations.disable_auth:DisableAuth `
  wsi-rs:test
# /alive, /v3/slides/info?slide_id=..., tiles, regions
```

The real goal — differential comparison against the Python service — **has now been run, and
is clean**:

```text
running 691 comparisons
matched 595 | tolerated 93 | allowlisted 3 | differed 0 | errored 0
```

over five slides at the default allowances, where both services share a decoder and every
sample has to agree: an Aperio SVS, a generic pyramidal TIFF, a fluorescence OME-TIFF, a
generated NIfTI volume and a PNG — exercising every endpoint, level, format, channel
selection, z index, padding colour and ICC intent. Every `tolerated` entry is a
container-level difference recorded as divergence 36 or the GIF palette of divergence 19;
no comparison was skipped and no slide was. Two further passes cover the plugins this one
cannot reach; see below.

It did **not** start clean. The first run reported 60 differences and one error, and they
fell into six causes — worth recording, because four were real faults that no amount of
per-plugin testing would have found:

| cause | nature |
|---|---|
| `ImagingReduce` arithmetic | Pillow multiplies by a truncated single-precision reciprocal rather than dividing; every downscaled image was one off on about a tenth of its pixels |
| the resample box | `precompute_coeffs` takes it as `float`, so the fractional box the reduce pre-pass produces has to be narrowed before use |
| validation bodies | `input`, `ctx` and `url` were missing, and only the first failing parameter was reported |
| `image_quality=0` | libjpeg clamps 0 to 1; turbojpeg rejects it, so a documented value answered 400 |
| `/files/icc_profile` | Python's `coerce_passthrough_payload` is outside the `try`, so one unreadable profile is a 400 for the whole request |
| the harness itself | gzip bodies compared encoded, an 822 MB download buffered whole, and planar TIFFs that no general decoder can read |

### Reproducing the run

The production image is the slow path — it builds libopenslide from source — so iterate by
running the server out of the *dev* image, which already carries it:

```powershell
# An ASCII-only data directory. Docker Desktop mishandles the 8.3 short path of a
# profile directory with non-ASCII characters in it, and silently mounts an empty one.
docker run -d --name wsi-rs -p 8080:8080 -v "${PWD}:/work" `
  -v wsi-rs-target:/target -v wsi-rs-registry:/usr/local/cargo/registry `
  -v "C:\wsi-parity-data:/data:ro" -w /work --env-file parity.env `
  -e CARGO_TARGET_DIR=/target -e WS_DATA_DIR=/data `
  wsi-rs-dev cargo run --release --quiet -p wsi-server --features openslide

docker run -d --name wsi-py -p 8081:8080 -v "${repo}:/wsi-service" `
  -v "C:\wsi-parity-data:/data:ro" -w /wsi-service --env-file parity.env -e WS_DATA_DIR=/data `
  wsi-service-wsi_service_dev:latest `
  sh -c "cd /wsi-service && poetry run uvicorn wsi_service.app:app --host 0.0.0.0 --port 8080"

./parity.ps1 -Context . -LimitSlides 2      # shake out the setup first
./parity.ps1 -Context . -Report parity-report.json
```

`--release` is not optional: a debug build is slow enough on real regions to make the run
impractical. Both services must read the *same* directory, and `/alive` is the quickest
check that they agree on which plugins are present.

A second pass with `WS_PLUGIN_PRIORITY_OPENSLIDE=-1` on both services is what exercises
`tiffslide`, since the matrix never asks for a plugin by name and openslide wins at
priority 10 otherwise. It is also clean, at the allowance divergence 28 measures:

```powershell
./parity.ps1 -Context . -LosslessDelta 1 -JpegDelta 16
# matched 436 | tolerated 141 | allowlisted 3 | differed 0 | errored 0
```

Those two numbers are not interchangeable guesses. Every losslessly transported response
differs by at most one per channel, which is the decoder difference itself; the JPEG
responses differ by up to sixteen because the encoder amplifies that input, and the worst
of them was checked as BMP to confirm it is still one. Run this configuration at the
defaults and all 58 of those show up as failures.

A third pass covers `wsidicom`, which a DICOM slide is the only way to reach. It lives
under its own `--context` precisely so the first pass can stay exact — mixing it in would
mean running everything at the looser allowance:

```powershell
./parity.ps1 -Context dicomroot -LosslessDelta 1 -JpegDelta 16
# matched 70 | tolerated 43 | allowlisted 24 | differed 0 | errored 0
```

The 24 allowlisted comparisons are almost all divergence 38 — Python shrinking its own
cached label and macro — plus divergence 20 on the archive of a directory slide. Both are
scoped to DICOM in the harness's allowlist, so the same endpoints are still compared
exactly on every other slide.

### Coverage

All six plugins are now reachable from the harness:

| plugin | how it is reached |
|---|---|
| `openslide` | the Aperio SVS and generic TIFF, at priority 10 |
| `tifffile` | the fluorescence OME-TIFF, at priority 2 |
| `tiffslide` | the same pyramidal slides, with openslide disabled |
| `pil` | a PNG |
| `nifti` | a generated `.nii`, with the plugin `poetry install`ed into the Python container |
| `wsidicom` | a DICOM directory under its own context |

`mvt` is the one that cannot be compared, and not for want of a file: Python's plugin omits
the `format` field, so it has never worked at all (divergence 26). The port's `mvt` support
is covered by its own tests instead.

The `/alive` allowlist entry that used to cover the plugin *set* is gone — both services
report the same seven plugins at the same versions and priorities. What is allowlisted there
now is only their *order*, which ties at priority 0 and is divergence 24.
