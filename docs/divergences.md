# Divergences from the Python service

The Rust port aims to be behaviourally identical to `wsi_service`. This file records every
place it is not, and why. Three categories:

- **Reproduced defect** — the Python behaviour is wrong, but clients may depend on it, so it
  is reproduced. Where a fix exists it is behind a setting that defaults to Python
  behaviour.
- **Unavoidable** — a different underlying library makes byte-level parity impossible.
- **Deliberate** — the port differs on purpose, with no observable effect on correct
  clients.

Each entry names the Python source it came from so the two can be re-checked when the
upstream code changes. The list is complete as of the port's completion: the differential
harness allowlists a difference only against an entry here, so anything it tolerates is
written down below.

---

## Reproduced defects

### 1. Three-digit `padding_color` returns 500

`wsi_service/utils/app_utils.py:211`

```python
match = re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", padding_color)
if match:
    stripped_padding_color = padding_color.lstrip("#")
    int_padding_color = tuple(int(stripped_padding_color[i : i + 2], 16) for i in (0, 2, 4))
```

The regex accepts `#RGB`, but the parser always slices six characters. For `#FFF` the third
slice is the empty string and `int("", 16)` raises `ValueError`, which escapes FastAPI as an
unhandled 500.

**Port:** same failure by default — and, per divergence 11, the same plain-text body. Set
`WS_LENIENT_PADDING_COLOR=true` to expand `#RGB` to `#RRGGBB` instead.

Implemented in `wsi-core/src/image/response.rs::validate_hex_color_string`.

### 2. `/files/label` and `/files/macro` never resize

`wsi_service/api/v3/slides_batch_api_helpers.py:84,120`

```python
map(lambda l: l.thumbnail((max_x, max_y), Image.ANTIALIAS), labels)
```

Two problems in one line: `map` is lazy and its result is discarded, so the callable never
runs; and `Image.ANTIALIAS` was removed in Pillow 10, so it would raise if it did. The
`max_x`/`max_y` path parameters are therefore ignored on these two endpoints, and full-size
images are returned.

**Port:** the resize is performed, because the endpoint documents it and the single-slide
`/slides/label` and `/slides/macro` endpoints do it correctly. The parity harness
allowlists this difference.

### 3. Requesting three or more channels ignores the requested indices

`wsi_service/utils/image_utils.py:98`

```python
else:
    # three or more channels given
    temp_array = get_multi_channel_as_rgb(separate_channels)
```

`separate_channels` is every source channel, not the selected ones, so
`?image_channels=3&image_channels=4&image_channels=5` returns channels 0, 1 and 2. Only the
one- and two-channel branches honour the request.

**Port:** reproduced, with a test pinning it
(`wsi-core/src/image/convert.rs::three_or_more_requested_channels_ignore_the_requested_indices`).

### 4. `get_requested_channels_as_array` short-circuits on a matching count

`wsi_service/utils/image_utils.py:142`

```python
if narray.shape[0] == len(image_channels):
    return narray
```

Requesting `image_channels=[2,1,0]` on a three-channel image returns the array unreordered.
Only a request for *fewer* channels than the source actually selects.

**Port:** reproduced.

### 5. `?image_format=tiff` on a greyscale image returns 500

`wsi_service/utils/app_utils.py:136`

```python
narray = np.asarray(image_region)
narray = np.ascontiguousarray(narray.transpose(2, 0, 1))
```

A greyscale (`L`-mode) Pillow image becomes a 2-D array, and `transpose(2, 0, 1)` raises
`numpy.AxisError`. Reachable via `/slides/label/...?image_format=tiff` on a slide whose
label image is greyscale.

**Port:** reproduced, as the plain-text 500 of divergence 11.

### 6. Region overlap uses `<` on the far edge

`wsi_service/utils/image_utils.py:154`

```python
start_x + size_x < slide_info.levels[level].extent.x
```

A region that ends *exactly* on the level extent is not considered fully contained, so it
takes the padding path and is re-read through `get_extended_region`. Tiles use `<=` and do
not have this asymmetry.

**Port:** reproduced. It changes which code path runs, and therefore the output, for
edge-aligned requests.

### 7. `__get_best_level_for_thumbnail` can return −1

`wsi_service_base_plugins/openslide/.../slide.py:244`

When level 0 is already smaller than the requested thumbnail box, the loop returns
`best_level - 1 == -1`, which Python then uses as a negative index into `slide_info.levels`
— selecting the *coarsest* level.

**Port:** reproduced, including the negative-index semantics — `-1` selects
`levels.len() - 1`. Pinned by
`wsi-plugin-openslide/src/lib.rs::a_level_zero_that_already_fits_wraps_to_the_coarsest_level`.

### 8. `CSVMapper` loses every slide in a case but the last

`wsi_service/csv_mapper.py:155`

```python
case_id = data[settings.case_id]
case = case_map.get(case_id, None)
...
case_map[case.id] = case
```

The lookup uses the **raw** case id from the row, but the store uses `case.id`, which is
the dotted `{group_1}.{group_2}.c.{case_id}` local id. The two never match, so every row
builds a fresh case object with an empty slide list and overwrites the previous one.
`GET /v3/cases/slides/?case_id=…` therefore returns only the last row's slide for each
case. The slides themselves are still individually addressable.

**Port:** reproduced, with a test pinning it
(`wsi-mappers/src/csv.rs::a_case_ends_up_with_only_its_last_slide`).

### 9. `IteratorMapper` cannot start with its patterns unset

`wsi_service/mapper_iterator/iterator/local_id_creation.py:31`

```python
if settings.institution_pattern != "":
    institution_match = re.search(settings.institution_pattern, str(source_path))
```

`institution_pattern` and `project_pattern` default to `None`, and `None != ""` is true, so
the guard passes and `re.search(None, …)` raises `TypeError` during construction. The
mapper is only usable with both patterns explicitly set — to a real pattern, or to the
empty string to disable them.

**Port:** an unset pattern is a startup error carrying the same message. `Some("")`
disables the pattern, as it does in Python.

### 10. `iterate` returns one value where two are unpacked

`wsi_service/mapper_iterator/iterator/iterator.py:60`

```python
if not os.path.exists(source_path):
    print(f"Path does not exist: {source_path}")
    return cases          # the caller does `cases, slides = iterate(settings)`
```

A missing data directory raises `ValueError` rather than yielding an empty mapping.

**Port:** a missing data directory is a startup error. Same outcome, clearer message.

### 11. Several failures return a plain-text 500, not a JSON `detail`

`wsi_service` registers no exception handlers, so any exception that is not a
`fastapi.HTTPException` escapes to Starlette's `ServerErrorMiddleware` and becomes
`500 Internal Server Error` with `content-type: text/plain; charset=utf-8` and the literal
body `Internal Server Error`. The detail text is lost.

The reachable cases:

| Trigger | Python exception | Source |
|---|---|---|
| `/slides/icc_profile` on a slide with no profile | `http.client.HTTPException(404, ...)` — note the import | `utils/icc_profile.py:74` |
| `?padding_color=%23FFF` (three-digit hex) | `ValueError` from `int("", 16)` | `utils/app_utils.py:216` |
| `?image_format=tiff` on a greyscale image | `numpy.AxisError` | `utils/app_utils.py:140` |
| `/files/{thumbnail,label,macro}` where a slide fails to open | `AttributeError` on `None` | `slides_batch_api_helpers.py:53` |
| `/{files,batch}/batch/` with fewer coordinates than slides | `IndexError` | `slides_batch_api_helpers.py:206` |
| a storage mapper returning an empty `storage_addresses` | `IndexError` | `slide_manager.py:123` |
| `/slides/{label,macro}/max_size/0/0` | `ValueError` from Pillow | `api/v3/slides.py:125` |

The first is the most surprising: `icc_profile.py` does `from http.client import HTTPException`,
so `raise HTTPException(404, "ICC Profile not available.")` builds a plain exception that
FastAPI never recognises as an HTTP error.

**Port:** reproduced. [`ApiError::unhandled`] carries the Python exception's `repr` — which
the batch endpoints do catch and write into a `t{n}.err` entry — but renders as Starlette's
plain-text 500. The parity harness compares the content-type before the body, so a JSON
envelope appearing where Python sends plain text fails the run.

---

## Unavoidable differences

### 12. DICOM pixel output

Python uses `wsidicom`; the port uses `dicom-rs`. These are independent implementations of
DICOM WSI decoding, so pixel-level parity is measured, not assumed. The parity harness
records the observed tolerance per test slide rather than asserting equality.

### 13. GIF output

Pillow quantises RGB to a 256-entry palette with adaptive median cut
(`GifImagePlugin._normalize_mode` → `convert("P", palette=Palette.ADAPTIVE)`). No Rust
crate reproduces that algorithm, so GIF bytes and pixels both differ. GIF is not a
practical tile format; the harness compares GIF responses structurally (dimensions, colour
count) rather than by pixel.

### 14. PNG and BMP container bytes

Both are lossless, so pixels match exactly, but zlib settings and PNG filter selection
differ from Pillow's, producing different bytes. The harness decodes before comparing.

### 15. TIFF container bytes

`tifffile.imwrite` emits descriptive tags the port does not (a `Software` tag among them)
and lays strips out differently. The port writes a minimal single-IFD, Deflate,
`BlackIsZero` TIFF with `PlanarConfiguration=2` for multi-sample data — the same logical
image. The harness decodes before comparing. See `wsi-core/src/image/tiffout.rs`.

### 16. Resampling is Pillow's algorithm, ported rather than approximated

Not a divergence — a note on why it is not one. Every thumbnail, label and macro response
is resized, and a generic high-quality resize does **not** match Pillow: measured against
real slides, a plain Lanczos3 lands 1–2 per channel away, and gets the output *dimensions*
wrong too.

`wsi-core/src/image/resample.rs` ports `Image.resize` and `Image.thumbnail` exactly: 22-bit
fixed-point filter coefficients with Pillow's rounding, the `reducing_gap=2.0` box-average
pre-pass, and `thumbnail`'s floor-or-ceil aspect arithmetic. `Filter::Bicubic` is the
default every plugin's own `get_thumbnail` uses; the `label` and `macro` endpoints pass
`LANCZOS` explicitly, and both are implemented.

Verified against the Python suite's own expected pixels on real SVS, TIFF, NDPI and MIRAX
slides — see `wsi-plugin-openslide/tests/real_slides.rs`.

### 17. `openslide.level[0].tile-*` depends on the library build

`wsi_service_base_plugins/openslide/.../slide.py:187` reads the tile extent from the slide's
advertised properties, falling back to 256x256 only when either is missing. The port does
the same, so the *logic* matches — but the values come from libopenslide, not from either
service.

One observation worth recording: for `OS-1.ndpi`, libopenslide 4.0.0 advertises a tile
extent of 4096x8, while the Python suite's parametrised expectation for its NDPI case is
256x256. Both services reading the same library would report the same number, so this is a
library-version or test-fixture question rather than a parity gap. The real-slide test
asserts the derivation rule instead of a fixed value.

### 18. Little-CMS version

ICC transforms go through Little-CMS on both sides — the same library `PIL.ImageCms`
wraps — so a given profile, intent and pixel format produce the same output. The two do
not necessarily link the *same build*: Pillow's wheels bundle their own copy, while the
Rust image links the distribution's `liblcms2-2`. Little-CMS is stable across versions and
no divergence has been observed, but a mismatch would show up as small colour differences
on ICC-enabled requests, so the version is worth checking first if that ever happens.

### 19. JPEG output without the `jpeg-turbo` feature

With the default `jpeg-turbo` feature the port compresses through libjpeg-turbo, the same
library Pillow links against, at the same quality and subsampling — output should be
byte-identical. Building with `--no-default-features` substitutes a pure-Rust encoder whose
output differs at the pixel level. Parity runs must use the default feature set.

---

## Deliberate differences

### 20. Directory listing order

`os.listdir` returns entries in an unspecified, filesystem-dependent order, and
`SimpleMapper.get_cases` and `PathsMapper.get_cases` pass that order straight through. The
port sorts by name so responses are stable across runs and platforms.

The same applies to `/slides/download` of a **directory** slide, which is what a DICOM slide
is. `expand_folders` (`download_utils.py:20`) walks the directory and then de-duplicates
with `set()`, so the archive's entry order is Python's hash order — arbitrary, and not even
stable across interpreter runs. The port sorts, so its archives are byte-stable. The entries
and their contents are the same either way; only the order differs, which is why the parity
harness allowlists `download` on a DICOM slide and compares it exactly everywhere else, a
single-file slide's archive having only one entry to order.

### 21. Local mapper cache format

`SimpleMapper` pickles its index to `local_mapper.p`. The port writes JSON to
`local_mapper.json`. Pickle is Python-specific and unsafe to load from an untrusted source;
the two services also cannot share an index, because they may have different plugin sets and
therefore disagree about which files are slides. **The UUID5 slide and case IDs are
unchanged**, so client bookmarks keep working.

### 22. Local mapper locking

Python uses a cross-process `FileLock` because it runs several gunicorn workers. The Rust
service is a single multi-threaded process, so an in-process mutex is sufficient.

### 23. Region prototype cache

`_REGION_PROTO_CACHE` (`image_utils.py:163`) is a module-level dict keyed by
`id(slide_info)` — a Python object address, which is reused after garbage collection, and
which is never evicted. The port fetches the 1×1 prototype per request instead. The
prototype only exists to carry the layout and dtype of an empty canvas, so the result is
identical; only the number of reads differs, and only on fully out-of-bounds requests.

### 24. Plugin ordering for equal priorities

Python sorts supported plugins by priority with a stable sort over
`pkgutil.iter_modules()` discovery order, which is filesystem-dependent. The port sorts the
registry by plugin name at construction, so equal-priority ties are deterministic.

This is observable on `/alive`, which lists the plugins in that order. With all seven
registered on both services, `nifti` and `mvt` tie at priority 0 and the two disagree about
which comes first — Python reports `nifti`, `mvt` and the port reports `mvt`, `nifti`. It is
the only difference the parity harness finds on that endpoint: the set, and every plugin's
name, version and priority, agree. Allowlisted against this entry, which is why the
allowlist is keyed by label rather than by field — a plugin genuinely missing from one side
would still show up as a length difference.

### 25. Path separators

`settings.data_dir` is normalised to forward slashes in both implementations, but Python's
`glob` returns native separators on Windows, so the string-prefix stripping in
`local_mode_abs_file_path_to_relative` silently fails there and addresses come back
absolute. The port normalises every path to forward slashes, which matches Python's
behaviour on Linux — the deployment target.

### 26. The `mvt` plugin works

`wsi_service_base_plugins/mvt/.../slide.py:249`

```python
self.slide_info = SlideInfo(
    id="",
    channels=[],
    ...
    levels=levels,
)
```

`SlideInfo.format` has no default, so `_build_slide_info` raises `ValidationError` for
every dataset — directory tree, single tile, manifest and `.mbtiles` alike. The plugin has
never served a tile: `open()` always ends in an unhandled exception, which `_open_slide`
turns into `500 Plugin mvt unable to open image (...)`.

**Port:** `format = "mvt"`, and everything else as written. The slide manager then extends
it the usual way, so an `.mbtiles` dataset reports `file-mbtiles-mvt` and a directory
dataset `folder-mvt`. Nothing else about the plugin is changed — the TMS/XYZ flip, the
gzip passthrough, the manifest key precedence and the `_TILE_Y_RE` pattern that silently
drops a zoom whose manifest-declared suffix is outside the four standard ones are all
reproduced, defects included.

`get_region` and `get_thumbnail` stay `501`, label and macro `404`, and
`get_icc_profile` still returns `b""` — which `get_for_payload` rejects as the plain-text
500 of divergence 11.

### 27. Extension matching is case-insensitive

`wsi_service/plugins.py:69`

```python
file_extension = pathlib.Path(filepath).suffix
if file_extension in plugin.supported_file_extensions:
```

Python compares the suffix verbatim, so on a case-sensitive filesystem `slide.PNG` and
`tiles.MBTILES` are not recognised by the `pil` and `mvt` plugins, while `slide.png` is.
The port lowercases the extension before the membership test, so case never decides
whether a file is a slide.

This only ever adds recognition: every name Python accepts, the port accepts. It is a
deliberate difference because the alternative — a file that the service can obviously read
being reported as having no supporting plugin — is a defect no client depends on.

The plugins that implement `is_supported` themselves are unaffected: `openslide` and
`nifti` both lowercase in Python too.

### 28. JPEG-compressed TIFF tiles decode through a different library

`tifffile` decodes a JPEG tile through `imagecodecs`, which wraps libjpeg-turbo; the port
decodes it with the pure-Rust decoder the `image` crate bundles. Both are correct JPEG
decoders and they disagree in the last bit or two of the inverse DCT.

The difference is *measured*, not assumed, and the bound is 4:

| checked against | fixture | worst channel deviation |
|---|---|---|
| tifffile's `page.asarray()` | `codec_jpeg.ome.tif`, photometric YCbCr | 3 |
| tiffslide's `read_region()` | `Aperio/CMU-1.svs`, photometric RGB | **0** |
| tiffslide's `read_region()` | `Generic TIFF/CMU-1.tiff`, photometric RGB | 1 |

Every other codec — none, LZW, Deflate, PackBits, Zstd, and the horizontal predictor — is
compared exactly.

The differential harness confirms the bound at the service level. With openslide disabled
on both services — the only configuration in which `tiffslide` serves these slides, since
the matrix never asks for a plugin by name — every losslessly transported response differs
by **at most 1 per channel**, on between 0.004% and 0.24% of samples:

| response | differing samples | worst deviation |
|---|---|---|
| `tile/level/0/tile/0/0`, Aperio, BMP | 2 of 196,608 | 1 |
| `tile/level/1/tile/0/0`, generic TIFF, BMP | 7 of 49,152 | 1 |
| `tile/level/7/tile/0/0`, generic TIFF, BMP | 86 of 49,152 | 1 |
| `region/level/6/0/0/718/514`, generic TIFF, BMP | 2,747 of 1,108,184 | 1 |
| `region/level/3/0/0/5750/4114`, generic TIFF, BMP | 166,835 of 70,974,728 | 1 |

A JPEG response of the same region differs by more — up to 16 per channel on the largest —
because the encoder amplifies a one-per-channel input, not because any further difference
was introduced. That is why the harness takes the two allowances separately
(`--lossless-delta`, `--jpeg-delta`): the lossless formats are what guard the pixels, and
the JPEG bound only keeps those responses from drowning out the report.

The `wsidicom` plugin decodes through the same path, and its numbers say what the cost
actually is — **chroma subsampling, not the IDCT**:

| directory | photometric | worst channel deviation |
|---|---|---|
| `DICOM_wsidicomizer/DICOM_Aperio` | `YBR_FULL` | **0** |
| `DICOM_wsidicomizer/DICOM_Mirax` | `YBR_FULL` | **0** |
| `DICOM_Leica/f990df0c-…` | `YBR_FULL_422` | 2 |

Both `YBR_FULL` slides are exact across regions, tiles, labels and overviews. Only the
`YBR_FULL_422` one deviates, and that is the one whose chroma planes are subsampled: the
two decoders upsample them differently. An uncompressed frame — Leica's label — is exact,
as is every label and overview in the collection.

One part of this was a real defect rather than rounding, and is fixed. A page whose
`PhotometricInterpretation` is `RGB` holds JPEG tiles whose samples already *are* R, G and
B, and `imagecodecs` is told so — it sets libjpeg's `jpeg_color_space = JCS_RGB`, which
skips the inverse colour transform. No decoder available here exposes that, and a
three-component JPEG with no JFIF marker is assumed to be YCbCr, so an Aperio tile came
back with 239 turned into 255. `decode_jpeg` now splices an `APP14` "Adobe" segment
carrying transform 0 into such a stream, which says the same thing in-band. That is what
took the SVS deviation to zero.

JPEG *output* is unaffected; see divergence 19.

### 29. Multichannel thumbnails go through scikit-image, not Pillow

`wsi_service_base_plugins/tifffile/.../slide.py:127`

```python
if type(thumbnail_org) is np.ndarray:
    thumbnail_resized = util.img_as_uint(
        transform.resize(thumbnail_org, (thumbnail_org.shape[0], max_y, max_x))
    )
else:
    thumbnail_resized = thumbnail_org.resize((max_x, max_y), Image.LANCZOS)
```

Every other resize in the service is Pillow's, and `wsi-core/src/image/resample.rs` ports
it exactly (divergence 16). This one is not: a channel stack goes through
`skimage.transform.resize`, whose defaults are a different algorithm altogether — convert
to float by dividing by the dtype's maximum, anti-alias with a Gaussian of
`sigma = max(0, (factor - 1) / 2)` per axis, interpolate bilinearly on the half-pixel grid
`in = (out + 0.5) * factor - 0.5` with mirrored edges, clip back into the input's own
range — and `img_as_uint` then scales onto `0..65535`, so a multichannel thumbnail is
always 16-bit whatever the page held.

**Port:** `wsi-plugin-tifffile/src/thumb.rs` reproduces that pipeline step for step,
including `img_as_uint`'s refusal of a float image outside `-1..1`. Exact agreement with
scipy's separable filters is *not* claimed: the Gaussian is truncated at four standard
deviations and normalised as scipy does, but small differences in the last bit are
possible. The parity harness records the observed tolerance for this endpoint rather than
asserting equality.

The RGB fast path is unaffected — it resizes a Pillow image with `LANCZOS`, which is
ported exactly.

### 30. `tifffile`'s ICC profile endpoint is always a 500

`wsi_service_base_plugins/tifffile/.../slide.py:152`

```python
async def get_icc_profile(self):
    return self._icc.get_for_payload(
        self.tif_slide.series[0].levels[0].pages[0].tags.get("ICCProfile", None)
    )
```

`TiffTags.get` returns a `TiffTag` object, not the profile bytes — the value would be
`tags.valueof("ICCProfile")`. So:

- with no profile, `get_for_payload` raises `http.client.HTTPException(404, ...)`, which
  FastAPI does not recognise, and the response is the plain-text 500 of divergence 11;
- with a profile, `get_for_payload` returns the `TiffTag`, and Starlette's `Response`
  cannot render it either — `AttributeError: 'TiffTag' object has no attribute 'encode'`.

The endpoint has therefore never returned a profile for a `tifffile` slide.

**Port:** reproduced. Both cases are `ApiError::unhandled`, carrying the `repr` of the
exception Python would have raised so the batch endpoints write the same `.err` entry.

### 31. An `icc_profile_intent` on a `tifffile` slide is ignored, or a 500 if strict

`wsi_service_base_plugins/tifffile/.../slide.py:96`

```python
profile = tif_level.pages[0].tags.get("ICCProfile")
result = self._icc.process_pil_image(result_data, profile, icc_profile_strict, icc_profile_intent, True)
```

Same root cause as divergence 30: `profile` is a `TiffTag`, which is truthy, so
`process_pil_image` takes its "there is a profile" branch and hands the object to
`ImageCms.buildTransform`, which raises. Its `except Exception` then either swallows the
failure — returning the image *unconverted* — or, with `icc_profile_strict`, raises
`ICCProfileError(500, f"Error: {ex}")`.

No colour transform is ever applied by this plugin, whether or not the slide carries a
profile.

**Port:** reproduced. A lenient request returns the region unchanged; a strict one is a
500.

### 32. TIFF series reconstruction covers four metadata dialects, not all of tifffile's

`tifffile` decides a file's `series[0].axes` and `shape` with around twenty
format-specific readers — OME, ImageJ, its own "shaped" JSON, LSM, STK, FluoView, SIS,
MicroManager, and more. The whole layout analysis in the `tifffile` plugin
(`is_supported`, `__inspect_series_layout`, `__get_series_page_index`) is written against
those decisions, so the port has to make the same ones.

`wsi-plugin-tifffile/src/tiff/series.rs` implements four: OME, ImageJ, "shaped", and the
generic fallback where one page is `YX`/`YXS` and several same-shape pages gain a leading
`I`. Those four cover everything the plugin accepts in practice, and the expectations are
pinned against files tifffile itself wrote — `tests/fixtures/manifest.json` records its
`axes`, `shape`, page geometry and decoded pixels for 27 fixtures, and
`tests/reader.rs` replays the lot.

A file in one of the *other* dialects still opens: it falls through to the generic reader,
which will usually give it a leading `I` axis, and the plugin then answers
`400 Unsupported TIFF axes for tifffile plugin: [('I', n)]` where Python might have
recognised a channel or Z axis instead. Adding a dialect is a contained change — one more
arm in `classify` and one more `*_layout` function — and the fixture generator is the way
to get its expectations right.

### 33. The `tiffslide` plugin refuses Hamamatsu NDPI

A Hamamatsu NDPI page is a single TIFF strip holding one JPEG far larger than the format's
65 535-pixel limit. `tifffile` reads sub-rectangles of it by entering the entropy-coded
stream at a restart marker whose offset is listed in the private `NDPI_MCU_STARTS` tag
(65426), which needs a decoder that can be told where to start and what DC predictor to
start with — `imagecodecs` does it in C. `OS-1.ndpi` presents 414 720 such synthetic
"tiles" over that one strip, and the tile geometry tiffslide advertises (`4096x8`, halving
per level) is computed from the same tag rather than read from the file.

**Port:** `TiffSlideReader::open` refuses an NDPI, with a message naming the reason. The
alternative — serving the metadata and failing every read, with a `tile_extent` of zero —
is worse: `tile_extent` is a divisor in the tiling arithmetic, and a client would discover
the problem one tile at a time.

`is_supported` still claims `.ndpi`, exactly as Python's does, so the refusal happens in
`open` where `load_slide` can fall through to the next plugin. `openslide` sits at priority
10 and serves NDPI pixel-exactly, so the default configuration reaches this only if
libopenslide is absent or a client asks for `plugin=tiffslide` by name.

The Python `tiffslide` suite does not exercise NDPI either — its two parametrised slides
are the generic TIFF and the SVS.

### 34. `tiffslide` declines any file whose name ends in `ome.tif`

`wsi_service_base_plugins/tiffslide/wsi_service_plugin_tiffslide/__init__.py:10`

```python
if filename.endswith("ome.tif") or filename.endswith("ome.tiff"):
    return False
```

The test has no dot in it, so it is not "is this an OME-TIFF" but "does this name end in
those letters". `metabolome.tif`, `chromosome.tiff` and `ribosome.tif` are all declined.

**Port:** reproduced, and pinned by
`wsi-plugin-tiffslide/src/lib.rs::an_ome_tiff_is_declined_so_tifffile_can_have_it`. The
`tifffile` plugin's own OME suffixes *do* include the dot, so such a file is not treated as
an OME-TIFF there either — it is inspected as a generic TIFF, and `openslide` is left to
claim it.

### 35. `/slides/storage` performs the per-slide access check

`wsi_service/api/v3/local_mode.py:43,52`

```python
@app.get("/slides", response_model=Slide, tags=["Main Routes"])
async def _(slide_id: str = IdQuery, payload=api_integration.global_depends()):
    await api_integration.allow_access_slide(auth_payload=payload, slide_id=slide_id, manager=None, plugin=None)
    slide = localmapper.get_slide(slide_id)
    return slide

@app.get("/slides/storage", response_model=SlideStorage, tags=["Main Routes"])
async def _(slide_id: str = IdQuery, payload=api_integration.global_depends()):
    slide = localmapper.get_slide(slide_id)
    return slide.slide_storage
```

The two handlers are adjacent and take the same parameters, but only the first calls
`allow_access_slide`. `global_depends` authenticates and nothing more, so on
`/slides/storage` *any* authenticated caller can read back the storage addresses — the
absolute paths on the server's filesystem — of any slide the mapper knows, including ones
the per-slide rules forbid it from opening. That matters only under an integration that has
per-slide rules: `DisableAuth` and `EmpaiaApiIntegration` make `allow_access_slide` a
no-op, while `LsaaiAuth` derives permission from the slide ID itself.

**Port:** the check is performed, in
`wsi-server/src/routes/v3/local_mode.rs::slide_storage`. This is the one place the port
knowingly makes a request *fail* that Python answers, so it is worth being precise about
the blast radius: for `DisableAuth` and EMPAIA the added call returns `Ok` unconditionally
and behaviour is identical, so the parity harness sees no difference; for LSAAI the
response becomes the same 403 the sibling route already gives. Reproducing the omission
instead would mean shipping an authorization bypass to be byte-identical with a line its
own neighbour contradicts.

### 36. Encoded containers differ byte for byte where the pixels do not

`wsi_service/utils/app_utils.py:184`, `wsi_service/utils/app_batch_utils.py:67`

Three response envelopes are produced by a different library here than in Python, and none
of them is byte-identical even when every sample inside is:

- **TIFF.** Python calls `tifffile.imwrite(mem, narray, photometric="minisblack",
  planarconfig="separate", compression="DEFLATE")`, which writes the IFD before the data,
  tags the compression as `DEFLATE` (32946), and adds `ImageDescription`
  (`{"shape": [3, 256, 256]}`), `Software` (`tifffile.py`), `XResolution`, `YResolution`,
  `ResolutionUnit` and `ExtraSamples`. `wsi-core/src/image/tiffout.rs` writes the data
  first and the IFD last, tags the same zlib stream as `ADOBE_DEFLATE` (8), adds
  `SampleFormat`, and omits the descriptive tags. The strips also compress to different
  lengths, because the two zlib implementations make different choices at the same level.
- **PNG** and **BMP**, where chunk layout and padding differ between Pillow and the
  `image` crate.
- **ZIP**, for `/v3/slides/download` and the batch endpoints. Python's `zipfile` writes a
  classic local header carrying the file's modification time; the port's streaming writer
  emits a zip64 header with a data descriptor, the UTF-8 name flag, and a zero timestamp.
  Entry names, the stored (uncompressed) method and the payload bytes are the same.

**Port:** accepted. The samples are identical, which is the contract; the containers are
not, and chasing byte-identity would mean reimplementing three encoders' internal choices
rather than their output. The parity harness compares accordingly — `compare.rs` decodes
PNG and BMP, parses the TIFF IFD and inflates its strips, and checks the ZIP's first entry
header and payload — and reports each as tolerated rather than matched, so the
distinction stays visible.

### 37. A DICOM region outside the level is a plain-text 500

`wsi_service_base_plugins/wsidicom/.../slide.py:43`, `wsidicom/wsidicom.py:605`

```python
img = self.dicom_slide.read_region(level_location, level_dicom, (size_x, size_y))
```

```python
if not wsi_level.valid_pixels(scaled_region):
    raise WsiDicomOutOfBoundsError(
        f"Region {scaled_region}", f"level size {wsi_level.size}"
    )
```

Every other plugin pads a region that reaches past the edge of a level: `openslide` and
`tiffslide` fill the overhang with `padding_color`, which is why `?padding_color=` is a
documented parameter. `wsidicom` refuses instead, the plugin does not catch the error, and
`wsi_service` registers no handler for it — so it escapes to Starlette's
`ServerErrorMiddleware` and the response is the plain-text `Internal Server Error` of
divergence 11, with no JSON envelope and no mention of bounds.

The region endpoint's own validation does not prevent it: `validate_image_size` bounds the
*size* against `max_returned_region_size`, and nothing checks the origin, so
`/region/level/0/start/-32/-32/size/64/64` and a 64x64 region at the far corner both reach
the plugin.

**Port:** reproduced, in `wsi-plugin-wsidicom/src/lib.rs::get_region`, including for a
negative origin — which cannot even be expressed in `read_region`'s unsigned location, so
Python fails it one step earlier with the same outcome. A region that ends exactly at the
edge is allowed, and `tests/ground_truth.rs` pins both sides of that boundary on all three
real directories.

`get_tile` is different and worth not confusing with it: its `except Exception` turns an
out-of-range tile into `500 Failed to extract tile from WSI.`, a JSON body, because that
method catches what this one does not.

### 38. `/slides/label` and `/slides/macro` shrink the image they cache

`wsi_service/api/v3/slides.py:125,160`

```python
label = await slide.get_label()
label.thumbnail((max_x, max_y), Image.Resampling.LANCZOS)
```

`Image.thumbnail` resizes **in place**. Where the plugin returns a *cached* image rather
than a fresh one, that mutation outlives the request: the next caller resizes what the last
one left behind, and the image shrinks monotonically toward 1x1. The size a client gets
therefore depends on every request that came before it.

`wsidicom` is where this bites, because it caches the decoded associated image. Measured
against the Python service, requesting the Aperio DICOM label once at each size:

| request | response |
|---|---|
| `max_size/99999/99999` | 387x463 |
| `max_size/100/100` | 84x100 |
| `max_size/64/64` | 54x64 |
| `max_size/32/32` | 27x32 |
| `max_size/16/16` | 13x16 |
| `max_size/8/8` | 6x8 |
| `max_size/99999/99999` | **6x8** |

Note the 54x64, where 387x463 shrunk directly to a 64-pixel box gives 53x64: the second
number is wrong *as well as* the size, because the pixels came from the already-shrunken
84x100. And the unbounded request at the end cannot recover — the original is gone until the
slide leaves the handle cache.

`openslide` is unaffected: `associated_images` decodes on access, so each request gets its
own image and the same sequence returns 387x463 at the end. The two plugins behaving
differently under the same route is the clearest sign this is a defect rather than a design.

**Port:** **not** reproduced, deliberately. The port returns the same image for the same
request however many times it is asked, because reproducing this would make the response
depend on request history — not something a client can work with, and not something that
could be reproduced faithfully anyway, since the outcome depends on the interleaving of
concurrent requests. The parity harness allowlists `label`, `macro` and the batch
thumbnails *scoped to DICOM slides*, so the same endpoints still have to match exactly on
every slide where Python is not corrupting its own cache.
