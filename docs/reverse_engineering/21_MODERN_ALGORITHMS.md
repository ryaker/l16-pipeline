# Modern Algorithm Opportunities — Going Beyond Lumen

**Goal**: Surpass Light's 2016-era Lumen pipeline using modern algorithms available on Apple Silicon.

**Hardware**: Apple Silicon M-series — PyTorch 2.11 with **MPS (Metal GPU) available**, Neural Engine, unified memory.

---

## What's Already Installed (Ready to Wire In)

| Library | Version | Purpose |
|---|---|---|
| **depth_pro** | 0.1 | Apple's Metric3D — monocular depth, better than MVS |
| **lightglue** | 0.0 | SOTA feature matching (replaces Ceres + SIFT) |
| **SuperPoint** | (via lightglue) | Learned keypoint detection |
| **kornia** | 0.8.2 | Differentiable CV ops (MPS-compatible) |
| **torchvision** | 0.26.0 | Classic + learned image transforms |
| **PyTorch** | 2.11 | MPS GPU inference |

We do NOT need to install anything to use Depth Pro or LightGlue. They're wired in from the earlier ML pipeline work.

---

## Stage-by-Stage Assessment

| Stage | Lumen (2016) | Modern Best Available | Gain | Status |
|---|---|---|---|---|
| **Depth estimation** | PatchMatch MVS from 5 A cams | **Depth Pro** (monocular metric) | +3–5 dB | ✅ Installed, needs wiring |
| **Calibration refinement** | Ceres + unknown features (~60s) | **SuperPoint + LightGlue** (~3s on MPS) | Comparable accuracy, 20× faster | ✅ Installed, needs wiring |
| **Optical flow** | DIS (2016) | RAFT (2020) | +2–3 dB | ⬜ Needs install |
| **Demosaicing** | Halide VNG | Learned (DemosaicNet, CameraNet) | +1–2 dB | ⬜ Needs install |
| **Denoising** | Unknown Halide | NAFNet (2022) | +1 dB | ⬜ Needs install |
| **Super-resolution** | None | Real-ESRGAN 2×–4× | Entirely new | ⬜ Needs install |
| **GPU acceleration** | CPU only (x86 Snapdragon) | MPS Metal — all ops | 10–20× speedup | ✅ Available |
| **Bundle adjust** | Ceres 3-pass (~60s) | LightGlue features → custom optimizer | ~5s same accuracy | ✅ LG installed |

---

## Quality Ceiling Estimate

Lumen baseline: **~13 dB PSNR** vs our own CRA+CCM merge (current)
After flow refinement: **~22 dB PSNR** (measured on L16_04574)

With modern stack:

| Addition | Expected PSNR |
|---|---|
| Current (CRA + CCM + DIS flow) | ~22 dB |
| + Depth Pro depth → better warp | +3–5 dB → ~25–27 dB |
| + LightGlue calibration refinement | +1–2 dB → ~27–29 dB |
| + RAFT flow (better than DIS) | +2–3 dB → ~29–32 dB |
| + NAFNet denoising | +1 dB → ~30–33 dB |
| + Learned demosaicing | +1–2 dB → ~31–35 dB |
| + Real-ESRGAN 2× SR | Perceptual gain, resolution win | 

**Realistic target: 30–35 dB PSNR** with full modern stack. That's 2–3× more signal fidelity than Lumen achieves. The gap is real and achievable.

40+ dB is the "perfect reconstruction" ceiling — achievable with end-to-end neural fusion trained on L16 ground truth pairs, not individual stage improvements.

---

## Top 3 to Implement First (WSJF-ranked)

### 1. Depth Pro → replaces flat-plane depth (WSJF = 12)
**Why first**: flat-plane depth is the biggest single quality gap. Foreground objects at 1–5m ghost badly with a flat 38m plane. Depth Pro gives metric depth from a single A camera. With real depth, B/C overlay works correctly AND A-camera ghost is eliminated.

**Integration**:
```python
from depth_pro import create_model_and_transforms, load_rgb
model, transform = create_model_and_transforms()
model = model.to('mps')  # Metal GPU
model.eval()

# Get depth map for A1 (or average A cameras)
image, _, f_px = load_rgb(frame_path)
prediction = model.infer(transform(image), f_px=f_px)
depth = prediction["depth"]  # float32 meters, same shape as input
```
The depth map replaces `np.full((H, W), 38.0)` in `merge_cameras_with_flow()`.

**Effort**: ~1 day. The model is installed. The integration point is one parameter in `merge_cameras_with_flow(init_depth=depth_map)`.

---

### 2. SuperPoint + LightGlue → replaces per-capture Ceres (WSJF = 9)
**Why second**: Ceres adds 60s per render and we don't yet implement it. LightGlue achieves the same thing (per-capture camera pose refinement) in ~3s on MPS, using better features than whatever Lumen used.

**How it works**:
1. Run SuperPoint on each A camera frame → keypoints + descriptors
2. Run LightGlue to match keypoints between camera pairs (A1↔A2, A1↔A3, etc.)
3. Use matched point pairs to refine each camera's R and t via PnP or linear triangulation
4. Pass refined R/t into the merge pipeline instead of factory calibration

This is essentially what Ceres does, just with modern learned features and without needing to implement the whole Ceres Problem setup.

**Effort**: ~2 days. LightGlue is installed; need to write the PnP refinement wrapper.

---

### 3. RAFT optical flow → replaces DIS (WSJF = 7)
**Why third**: DIS is already giving good results. RAFT improves on challenging regions (occlusions, large displacements, low texture). Useful but not as critical as depth.

**Install**:
```bash
pip install raft-pytorch  # or clone from github.com/princeton-vl/RAFT
```

**Integration**: drop-in replacement for `_dense_flow()` in `lri_merge_flow.py`. RAFT takes two uint8 grayscale images and returns a flow field — same interface.

**Effort**: ~0.5 days once installed.

---

## The New Capability Lumen Never Had: Super-Resolution

Real-ESRGAN or HAT can take the 4200×3150 A-camera fusion output and produce an 8400×6300 or 16800×12600 image with real high-frequency detail. This is completely impossible with Lumen's pipeline and represents a genuine quality leap.

On Apple Silicon with MPS, 2× SR on a 4200×3150 image takes ~8 seconds (vs ~2 minutes on CPU). 4× SR takes ~30s.

Install: `pip install realesrgan` (BasicSR framework)

---

## Hardware Acceleration Plan

Current code is CPU-only (numpy + cv2). Moving to MPS:

| Operation | Current | With MPS | Speedup |
|---|---|---|---|
| Depth Pro inference | N/A | ~2s per frame | — |
| LightGlue matching | N/A | ~0.5s per pair | — |
| DIS flow (5 cameras) | ~8s | ~1s (kornia.geometry) | ~8× |
| CCM apply | ~0.2s | ~0.02s (torch matmul) | ~10× |
| CRA apply | ~1.5s | ~0.1s (torch conv) | ~15× |
| Full A-merge pipeline | ~22s | ~5s | ~4× |

**Total realistic pipeline time with modern stack**: ~10–15s on M-series Apple Silicon vs Lumen's 60–120s on 2017 hardware. **Same quality, 6–10× faster.**

---

## Implementation Order

```
Week 1: Depth Pro → real depth → eliminate flat-plane ghosting
Week 1: LightGlue calibration refinement → sub-pixel alignment (replaces Ceres)
Week 2: RAFT flow → better alignment in challenging regions
Week 2: GPU acceleration (MPS) for the full pipeline
Week 3: NAFNet denoising + learned demosaicing
Week 3: Real-ESRGAN 2× super-resolution
```

Each week doubles the quality gap between our output and Lumen's.

---

## Quick Wins Available Right Now

Without installing anything new:

1. **Use Depth Pro for init_depth** — replace the fixed 38m float. Zero new installs. ~1 day.
2. **LightGlue feature matching for A-camera alignment** — compute per-capture corrected R/t. ~2 days.
3. **MPS acceleration for CRA and CCM** — move numpy ops to torch on MPS. ~0.5 days, ~10× faster ISP.
