# l16-pipeline

Modern ML/CV processing pipeline for the [Light L16](https://light.co/camera) computational camera — rebuilding the original Lumen software with Apple Depth Pro, a PySide6 desktop UI, and multi-camera depth fusion.

The Light L16 captured 10 simultaneous RAW frames (5 wide + 5 tele) and used computational photography to produce a single high-resolution image with synthetic depth-of-field. This project reverse-engineers the LRI file format and replaces the 2016 CIAPI pipeline with modern tools.

---

## Working On Now

### v2 Synthetic Canvas Fusion (new)
The new pipeline in `lri_fuse_v2.py` / `lri_canvas_blend.py` builds a synthetic 81MP canvas (10449×7795) from calibration geometry:

**VirtualCamera canvas** — center at the wide-camera centroid, focal length matches B4 telephoto (fx≈8276). All cameras are remapped into this common coordinate frame.

**Confidence weights** — each camera contributes `taper * res_w^N` where `res_w = fx_src / fx_vc` (capped at 1.0). At the canvas center, B4 (res_w≈1.0) gets N=4× more weight than each wide camera (res_w≈0.408 → 0.028). Result: B4 telephoto dominates center; wide cameras fill coverage beyond B4's field of view.

**Sharpness measurements at canvas center (1600×1600px ROI)**:
| Source | Laplacian variance |
|--------|--------------------|
| A1 alone (wide, upsampled) | 159,234 |
| v3 (no CCM, flat-plane depth) | 1,823,985 |
| B4 alone (telephoto) | 2,241,225 |
| **v4 (diagonal CCM + flat-plane)** | **1,978,293** |

Canvas fusion is ~12× sharper than A1 alone at center. Active work: raising res_w exponent to bring fusion even closer to B4 alone.

**Diagonal CCM for mirror cameras** — B-group cameras use periscope mirrors that introduce per-channel spectral shifts. A full 3×3 least-squares CCM introduces cross-channel mixing that acts as a spatial low-pass filter (cross-channel averaging ≡ weighted average of adjacent colors). Instead we fit a diagonal-only 3×3 (per-channel median gain, normalized to green channel). This corrects spectral shift with zero effect on spatial frequency content.

**Forward-warp depth** — `forward_warp_depth()` in `lri_depth_loader.py` reprojects depth from source camera pixel-space into the virtual canvas. cv2.resize to canvas dimensions is wrong: it ignores that source camera pixels are not uniformly distributed in canvas space. Error reaches 20-30px for wide cameras at 1000px from optical axis.

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
| Raise res_w exponent (N=4→8) | Reduce wide-camera blur contribution at center to bring fusion closer to B4 alone. |
| Per-camera Depth Pro for all cameras | Currently only A1 has depth. Need Depth Pro inference for B1–B5 to enable forward-warp reprojection. |
| MOVABLE camera inclusion via LightGlue R refinement | B1/B2/B3/B5 mirrors may have moved since factory calibration. Use LightGlue to refine R before enabling 3D warp. |
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
| `lri_fuse_v2.py` | CLI entry point for v2 synthetic canvas fusion pipeline |
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
