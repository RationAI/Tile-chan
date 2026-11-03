# Changelog

## 0.1.0
- Rust TIFF reader (tifffile, tiffslide plugins) decodes more sample layouts, matching Python tifffile values:
  packed 1–32 bit samples (bilevel, 4/12/14/24-bit, FillOrder 2), float16, floating-point predictor 3,
  and JPEG 2000 compression (33003, 33005, 34712) via openjpeg.
- Rust: float16 pages are served as float32 and bilevel pages as uint8 0/1, so they render instead of
  returning 400 (divergence 42). Palette and WhiteIsZero pages serve raw stored values.

## 0.0.1
- Rewritten to mirror WSI-Service