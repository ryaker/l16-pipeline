# l16-pipeline

Modern ML/CV processing pipeline for the [Light L16](https://light.co/camera) computational camera — rebuilding the original Lumen software with Apple Depth Pro, a PySide6 desktop UI, and multi-camera depth fusion.

The Light L16 captured 10 simultaneous RAW frames (5 wide + 5 tele) and used computational photography to produce a single high-resolution image with synthetic depth-of-field. This project reverse-engineers the LRI file format and replaces the 2016 CIAPI pipeline with modern tools.

---

## What it does

```
LRI file  →  lri_extract.py  →  10 camera PNGs (A1–A5, B1–B5)
                                       ↓
                             Apple Depth Pro (per camera)
                                       ↓
                             lri_fuse_depth.py
                                       ↓
                           Fused depth map (median, multi-cam)
                                       ↓
                             lri_lumen_app.py  (desktop UI)
                                       ↓
                           PNG / JPEG / 16-bit DNG
```

The desktop app (`lri_lumen_app.py`) is a Lightroom-style editor:
- Browse a folder of LRI files with live thumbnails
- Click an LRI to run the full pipeline (extract → depth → fuse), with per-stage caching
- Click anywhere in the image to set focus distance
- Sliders update the image in real time:
  - **Tone sliders** (exposure, contrast, highlights, shadows, WB, saturation, sharpness) — instant
  - **Bokeh sliders** (focal length, f-number, focus distance) — 150ms debounce at 0.25 scale
- Export PNG, JPEG, or 16-bit DNG

---

## Requirements

**Python 3.11+** and the following packages:

```bash
pip install PySide6 numpy opencv-python pillow
```

**Apple Depth Pro** (monocular depth estimation):

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
# Download the checkpoint (~300 MB):
python -c "from depth_pro import create_model_and_transforms; create_model_and_transforms()"
# or manually: place depth_pro.pt in ml-depth-pro/checkpoints/
```

The `ml-depth-pro/` directory must live **next to** the pipeline scripts (same folder).

**Optional** (for PLY point cloud export in `lri_fuse_depth.py`):
```bash
pip install open3d
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ryaker/l16-pipeline.git
cd l16-pipeline

# 2. Install Python deps
pip install PySide6 numpy opencv-python pillow

# 3. Set up Depth Pro (see above)

# 4. Run the desktop app pointed at a folder of LRI files
python3 lri_lumen_app.py /path/to/your/LRI/folder/
```

Double-click any LRI in the browser panel to start processing. The pipeline caches results next to each LRI file (`<name>_lumen/`) so it only processes once.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `lri_extract.py` | Unpack PACKED_10BPP Bayer, bilinear demosaic → per-camera PNGs |
| `lri_calibration.py` | Parse LRI protobuf headers → camera intrinsics + extrinsics |
| `lri_fuse_depth.py` | Reproject per-camera depth maps to reference frame, median fuse |
| `lri_lumen.py` | Core algorithms: bokeh (CoC blur), tone adjustments, DNG export |
| `lri_lumen_app.py` | PySide6 desktop app — library browser + real-time editor |
| `lri_depth_mps.py` | Multi-view depth fusion on Apple Silicon (MPS) |
| `lri_depth_mlx.py` | Multi-view depth fusion using Apple MLX |
| `lri_stereo.py` | Stereo matching utilities |
| `lri_process.cpp` | C++ reference implementation of LRI parsing |

---

## LRI File Format

LRI files are sequences of `LELR`-magic blocks. Each block contains a protobuf-encoded header with camera module descriptors. Image data is `PACKED_10BPP` Bayer (4 pixels in 5 bytes, little-endian bit order).

Camera IDs map to positions:
- `A1–A5`: wide-angle modules (~35mm equivalent)
- `B1–B5`: telephoto modules (~70–150mm)

See `lri_extract.py` and `lri_calibration.py` for the full parsing logic.

---

## Pipeline Cache Layout

For each `<name>.lri`, the pipeline writes a `<name>_lumen/` directory:

```
<name>_lumen/
  frames/          # Extracted PNGs (A1.png … B5.png)
  depth/           # Depth Pro output (A1.npz = depth in metres)
  fused_10cams_median_depth.png   # 16-bit uint16, depth in mm
```

Each stage is skipped if its outputs already exist.

---

## Bokeh Algorithm

The synthetic depth-of-field uses a circle-of-confusion model:

1. Compute per-pixel CoC diameter from depth, focus distance, f-number, and 35mm-equivalent focal length
2. Blur each pixel by a disc kernel proportional to its CoC
3. Blend sharp and blurred images based on CoC magnitude

See `lri_lumen.py:apply_bokeh()` for implementation.

---

## Notes

- Raw sensor output is linear light — `lri_extract.py` outputs 8-bit PNGs with no gamma or white balance. The app applies grey-world AWB and gamma 2.2 for display.
- Depth Pro must be run from the `ml-depth-pro/` directory (checkpoint path is relative).
- The pipeline works with any LRI variant — it auto-detects which cameras are present (A/B/C series) and uses the first available as the reference frame.
