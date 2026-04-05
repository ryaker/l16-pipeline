# L16 Fusion Pipeline — Architecture Spec

## Camera Geometry (from calibration, not labels)

The L16 has 10 cameras in two optical classes. **Never hardcode camera names — determine class from calibration.**

### Wide cameras (mirror=NONE, fx≈3370)
Direct-fire, no mirror, all coplanar (Z≈0mm). Identity rotation (R_trace=3.0).
Array centroid: [7.5, 4.5, 0.0]mm. Camera closest to centroid becomes **reference**.

For L16_01337: A1 wins (8.8mm from centroid vs 25-42mm for others) — but code discovers this.

| Camera | |t| from origin | dist to centroid |
|--------|--------------|-----------------|
| A1     | 0mm          | 8.8mm → **ref** |
| A3     | 25mm         | 25mm            |
| A2     | 36mm         | 28mm            |
| A4     | 34mm         | 42mm            |
| A5     | 43mm         | 36mm            |

### Telephoto cameras (fx≈8285) — two sub-classes
Periscope mirrors fold the optical path. Class determines whether factory calibration R is usable.

**GLUED mirror (R_trace≈3.0): reliable, use 3D-warp depth path**
- B4 is the only glued-mirror tele in L16_01337

**MOVABLE mirror (R_trace far from 3.0): calibration R may be stale**
- B1, B2, B3, B5 — mirror may have moved since factory calibration
- Use image-based alignment only (LightGlue homography)
- Subject to sharpness gate AND consistency gate before blending

## Depth Map Pipeline

### Step 1: Per-camera depth (Apple Depth Pro)
Run Depth Pro on each frame → per-camera depth map (metric, in metres from Depth Pro).
Save as `depth/A1.npz`, `depth/A2.npz`, etc. (numpy float32, key='depth', metres).

Also save as `depth/A1_depthpro.png` (uint16, mm) for lri_fuse_depth.py compatibility.

### Step 2: Fuse depth maps (lri_fuse_depth.py)
Inputs: per-camera `*_depthpro.png` + calibration.json
- Reproject each wide + B4 depth map into reference frame
- Median fuse → `fused_Ncams_median_depth.png` (uint16, mm)
- Only use: wide cameras (NONE mirror) + GLUED mirror tele cameras
- Skip: MOVABLE mirror tele cameras (stale calibration → wrong reprojection)

Output: `<lumen_dir>/fused_<N>cams_median_depth.png`

### Step 3: Image fusion uses fused depth
`lri_fuse_image.py` should load depth in this priority order:
1. `<lumen_dir>/fused_*cams_median_depth.png` (uint16 PNG, mm → divide by 1000 → metres)
2. `<lumen_dir>/depth/<ref_cam>.npz` (Depth Pro native, already metres)
3. No depth → homography-only fallback for all cameras

## Image Fusion Pipeline (lri_fuse_image.py)

### Reference camera selection (dynamic)
```python
def select_reference(cameras):
    """Pick the wide (mirror=NONE) camera nearest the centroid of all wide cameras."""
    wide = {n: c for n, c in cameras.items() if c['mirror'] == 'NONE'}
    centroid = np.mean([c['t'] for c in wide.values()], axis=0)
    return min(wide, key=lambda n: np.linalg.norm(wide[n]['t'] - centroid))
```

### Camera contribution strategy per class

**Wide cameras (mirror=NONE):**
- All contribute to Laplacian pyramid blend
- Problem: blending equally-sharp co-planar cameras REDUCES sharpness (homography sub-pixel error + averaging)
- Fix: secondary wide cameras only contribute at pyramid levels where they are measurably sharper
  than reference at that scale — OR only fill where reference coverage is weak

**Glued-mirror telephoto (B4 etc.):**
- Use 3D warp via depth map (unproject A1 → world → reproject to B4)
- Sharpness gate: cam_sharp_coverage / ref_sharp_coverage ≥ 0.95
- Consistency gate: mean |warped - ref| / sigma < threshold (sigma=1000, threshold>0.40)
- CCM (3×3 color correction) to correct mirror spectral shift

**Movable-mirror telephoto:**
- Image-based alignment only (LightGlue homography — no 3D warp, calibration unreliable)
- BOTH gates apply (sharpness + consistency)
- CCM applied

### A-group sharpness regression fix
Root cause: homography has residual sub-pixel error. Blending 4 equally-sharp but slightly-offset
copies of the same scene reduces variance even when each individual frame is sharp.

Approaches (pick best experimentally):
1. **Coarse-only blend**: secondary wide cameras contribute at pyramid levels 3-5 only (low freq),
   let reference supply all high-frequency detail
2. **Strict sharpness gate per-pixel**: secondary contributes only at pixels where its local
   Laplacian is > (1 + margin) × reference local Laplacian
3. **Coverage fill only**: secondary wide cameras only contribute where reference has no valid data

## Key File Paths

| File | Description |
|------|-------------|
| `<lumen>/frames/A1.png` | Extracted 16-bit PNG |
| `<lumen>/depth/A1.npz` | Depth Pro output (float32 metres) |
| `<lumen>/depth/A1_depthpro.png` | Same as uint16 mm (for lri_fuse_depth.py) |
| `<lumen>/fused_Ncams_median_depth.png` | Fused depth from all reliable cameras |
| `<lri_dir>/<name>_cal/calibration.json` | Factory calibration (K, R, t per camera) |

## What lri_fuse_image.py currently does wrong

1. Hardcodes `ref = 'A1'` instead of computing dynamically
2. Looks only for `depth/A1.npz` — misses fused depth PNG
3. Treats all B cameras the same — doesn't split GLUED vs MOVABLE
4. Blends all wide cameras equally — causing sharpness regression

## GitHub Issues to create

- [ ] #1: Dynamic reference camera selection (geometry-based)
- [ ] #2: Load fused depth PNG (mm uint16) in addition to .npz
- [ ] #3: Split tele cameras: GLUED=3D warp, MOVABLE=homography only
- [ ] #4: Fix wide-camera sharpness regression (coarse-only blend for secondaries)

## v2 Pipeline Status (2026-04-05)

**Implemented:**
- VirtualCamera canvas (`lri_virtual_camera.py`)
- Per-camera remap with depth (`lri_camera_remap.py`)
- Confidence weights res_w^4 (`lri_confidence.py`)
- Diagonal CCM for mirror cameras (`lri_ccm.py`)
- Forward-warp depth reprojection (`lri_depth_loader.py`)
- Tiled full-resolution canvas assembly (`lri_canvas_blend.py`)
- CLI entry point (`lri_fuse_v2.py`)

**Key design decisions:**
1. **Diagonal CCM not full 3×3**: Full lstsq CCM absorbs the 3-5× exposure gap between telephoto and wide cameras as cross-channel terms. Cross-channel averaging = spatial LPF. Diagonal CCM: only 3 scalar gains, zero cross-channel mixing.
2. **res_w^4 confidence**: Gives telephoto cameras (res_w≈1.0) ~88% weight at canvas center vs 5 wide cameras combined. Simple, no heuristics, derived purely from geometry.
3. **Forward-warp not resize**: Depth Pro produces depth in source camera ray-space. Resize to canvas dimensions is wrong off-axis. Forward-warp respects camera geometry.
4. **Flat-plane fallback**: Without per-camera depth for each camera, flat-plane (3m) is better than forward-warping A1's depth through B4's geometry. A1's Depth Pro depth in B4's coordinate system introduces systematic errors.

**Not yet implemented:**
- Per-camera Depth Pro for all 10 cameras (only A1/fused available)
- MOVABLE camera inclusion (needs LightGlue R refinement)
- res_w exponent tuning (N=8 under test)
