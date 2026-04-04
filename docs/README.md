# Light L16 Modern Processing Pipeline

Rebuilding Light's Lumen software using modern ML/CV — surpassing the 2016 CIAPI.

## Pipeline Overview

```
LRI file  →  lri_extract.py  →  10 camera PNGs
                                      ↓
                            Depth Pro (per camera)
                                      ↓
                            lri_fuse_depth.py
                                      ↓
                          Fused depth map (median, 10-cam)
                                      ↓
                            lri_lumen_app.py  (UI)
                                      ↓
                          PNG / JPEG / 16-bit DNG
```

## Scripts

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `lri_extract.py` | `.lri` file | `A1–A5, B1–B5` PNG | 10-bit Bayer unpack + bilinear demosaic |
| `lri_fuse_depth.py` | frames dir + depth dir | fused depth PNG + PLY | Depth Pro maps reprojected to A1, median fused |
| `lri_lumen.py` | frames + fused dir | DNG / JPEG | Gradio web UI |
| `lri_lumen_app.py` | folder of LRI or processed sets | PNG / DNG / JPEG | Native desktop app (PySide6) |

## Docs

- [Pipeline Architecture](pipeline.md)
- [Calibration & Camera Math](calibration.md)
- [Bokeh Algorithm](bokeh.md)
- [DNG Format Notes](dng.md)
