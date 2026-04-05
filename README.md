# l16-pipeline

Modern ML/CV processing pipeline for the [Light L16](https://light.co/camera) computational camera — rebuilding the original Lumen software with Apple Depth Pro, a PySide6 desktop UI, and multi-camera depth fusion.

The Light L16 captured 10 simultaneous RAW frames (5 wide + 5 tele) and used computational photography to produce a single high-resolution image with synthetic depth-of-field. This project reverse-engineers the LRI file format and replaces the 2016 CIAPI pipeline with modern tools.

---

## Working On Now

### Multi-camera image fusion overhaul ✅ (just landed)
Replaced the weighted-average accumulator in `lri_fuse_image.py` with a full multi-scale fusion pipeline:

- **Laplacian pyramid blending** — 6-level pyramid blends low/high frequencies independently per camera at each scale. Eliminates the halo artifacts that weighted-average produced at depth edges. Weight maps are decomposed into their own Gaussian pyramids (level-matched), which is the mathematically correct approach.
- **Radiometric normalization** — Before blending, each camera's gain and bias are estimated against the A1 reference using WLS (Weighted Least Squares) with inverse-gradient weighting. Edges are excluded from the gain estimate so color objects don't skew the result. Applied per-channel (R/G/B independently).
- **B-group color correction matrix** — B-group cameras use periscope mirrors that introduce spectral shifts not present in the direct-firing A-group lenses. A 3×3 CCM is estimated per B-camera (least squares over overlap pixels) and applied before blending.
- **Depth-map reprojection for B-cameras** — When a Depth Pro depth map is available (`depth/A1.npz`), B-cameras are aligned using true 3D warp (unproject via K_A⁻¹ → transform via R/t → reproject via K_B) rather than a global homography. Correctly handles depth-dependent parallax. Falls back to homography + LightGlue if depth map is absent.

### White balance estimation overhaul ✅ (just landed)
Three-tier WB in `lri_lumen.py`:
1. Read from Lumen DNG export if available (most accurate)
2. Shade-of-Grey estimator (p=6) on the actual image — robust to coloured scenes
3. Sensor median fallback (R×1.847, B×1.617)

### Live tone sliders ✅
Tone controls (exposure, contrast, highlights, shadows, WB, saturation, sharpness) update in real time — bokeh result cached separately from tone pass.

### 16-bit frame extraction ✅
`lri_extract.py` saves 16-bit PNG (10-bit × 64 → uint16) so the full tonal range is available for adjustments.

### DNG Mylio/Lightroom compatibility (in test)
IFD0 = 8-bit sRGB thumbnail + all DNG metadata tags, SubIFD = full-res 16-bit LinearRaw. Luminar Neo confirmed working; Mylio testing in progress.

---

## Still to Come

| Feature | Notes |
|---------|-------|
| Per-channel radiometric normalization | Current WLS gain uses single luminance gain. Upgrade to per-channel R/G/B gain estimation to eliminate the residual pink/green cast visible in B-group contributions. |
| Depth-map reprojection for B-cameras (with depth) | Depth map path exists and code is ready; needs Depth Pro run on A1 to produce `depth/A1.npz` before the 3D warp activates. Homography fallback is used until then. |
| A-group IBP super-resolution | Iterative Back-Projection on the 5-camera A-group stack for sub-pixel resolution gain. Decision pending on Bayer vs RGB input path. |
| Exposure fusion (Mertens style) | B4 camera shoots at ~½ the exposure of other cameras. Use for highlight recovery in blown regions. |
| BAYER_JPEG extraction | 2018-06-26 LRI files use surface format 0 (4 half-res JPEGs per Bayer channel). Unblocks a large set of images. |
| Real-ESRGAN upscale | 2× resolution post-fusion. |
| DNG SubIFD export bug | Mylio shows horizontally mirrored flat image — SubIFD layout investigation deferred. |
| Graph-Cut seam routing | Optimal seam placement at depth boundaries after depth reprojection stabilizes. |

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

## What you do NOT need

- **Lumen.app** — the original Light desktop app. This project is a complete replacement; the binary is not used anywhere.
- **lri-cpp** — a C++ LRI parser (reference only; `lri_process.cpp` is included for documentation). Not required to run the pipeline.
- **RAFT-Stereo** — an alternative stereo depth approach explored during development. Not used by the main pipeline.

---

## Requirements

**Python 3.11+** and the following packages:

### Desktop app (`lri_lumen_app.py`) — required

```bash
pip install PySide6 numpy opencv-python pillow
```

### Core algorithms (`lri_lumen.py`) — required if using the app or Gradio UI

```bash
pip install numpy opencv-python tifffile
# Only needed for the Gradio web UI (lri_lumen.py --gradio):
pip install gradio
```

### Apple Depth Pro — required for depth estimation

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
# Download the checkpoint (~300 MB):
python -c "from depth_pro import create_model_and_transforms; create_model_and_transforms()"
# or manually: place depth_pro.pt in ml-depth-pro/checkpoints/
```

The `ml-depth-pro/` directory must live **next to** the pipeline scripts (same folder).

### Optional extras

```bash
pip install open3d        # PLY point cloud export in lri_fuse_depth.py
pip install torch         # Required by lri_depth_mps.py (MPS multi-view fusion)
pip install mlx           # Required by lri_depth_mlx.py (Apple MLX fusion)
```

`lri_depth_mps.py` and `lri_depth_mlx.py` are experimental alternatives to Depth Pro for multi-view stereo depth. The main pipeline only needs Depth Pro.

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
| `lri_extract.py` | Unpack PACKED_10BPP Bayer, bilinear demosaic → per-camera 16-bit PNGs |
| `lri_calibration.py` | Parse LRI protobuf headers → camera intrinsics + extrinsics |
| `lri_fuse_image.py` | Multi-camera image fusion: radiometric normalization, CCM, Laplacian pyramid blend, depth reprojection |
| `lri_fuse_depth.py` | Reproject per-camera depth maps to reference frame, median fuse |
| `lri_lumen.py` | Core algorithms: bokeh (CoC blur), tone adjustments, adaptive WB, DNG export |
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

- Raw sensor output is linear light — `lri_extract.py` outputs **16-bit PNGs** (10-bit × 64 → uint16) with no gamma or white balance. The app applies grey-world AWB and gamma 2.2 for display.
- Depth Pro must be run from the `ml-depth-pro/` directory (checkpoint path is relative).
- The pipeline works with any LRI variant — it auto-detects which cameras are present (A/B/C series) and uses the first available as the reference frame.
