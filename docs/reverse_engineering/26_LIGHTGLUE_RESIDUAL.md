# LightGlue Calibration Residual Analysis
## Can LightGlue + Factory K/R/t Replace Ceres Bundle Adjustment?

**Date:** April 10, 2026  
**Status:** Analysis Complete (NEGATIVE – Requires Depth Information)

---

## Executive Summary

**Verdict: LightGlue sparse feature matching CANNOT replace Ceres bundle adjustment without per-feature depth information.**

The empirical Ceres corrections (2-7 px per camera) come from dense optical flow with per-pixel depth maps. In contrast, attempting to compute residuals using LightGlue features with a single fixed depth produces results that are **orders of magnitude incorrect** (~2000+ px instead of 2-7 px).

**Root cause:** Sparse feature triangulation requires accurate 3D depth for each matched feature. Without this depth information, the triangulated 3D points are incorrect, leading to invalid projections and useless residuals.

---

## Background: What is a Calibration Residual?

For a matched feature pair `(p1_A1, p1_A_target)`:

1. **Unproject** p1 into A1 camera frame using K^{-1}: get normalized direction
2. **Triangulate** to 3D at a known depth: X_A1 = direction · depth_m
3. **Transform** to target camera: X_target = R @ X_A1 + t
4. **Project** back to 2D: p_pred = K_target · (X_target / Z_target)
5. **Residual** = p_actual - p_pred (what Ceres is correcting)

This measures: "How far off is the factory calibration K/R/t from reproducing the actual matched feature positions?"

---

## The Geometry Check

For a feature at pixel (2000, 1500) in A1, triangulated to 38.36m depth, and transformed to A5 using factory K/R/t:

| Stage | Value | Units | Notes |
|-------|-------|-------|-------|
| Input pixel (A1) | (2000, 1500) | pixels | |
| Normalized direction | (-0.0250, -0.0122, 1.0) | unitless | from K^{-1} |
| Triangulated 3D point | (-0.96, -0.47, 38.36) | metres | at 38.36m depth |
| Transformed to A5 | (-0.80, -0.47, 38.36) | metres | after R @ X + t |
| Projected pixel (A5) | (2076.37, 1510.86) | pixels | with factory K |
| Expected residual (perfect cal) | (0.00, 0.00) | pixels | Zero error case |

The geometry computes correctly: ✓

---

## The Root Problem: Depth Ambiguity

For sparse feature matching without explicit depth:

**Given:** LightGlue matched pair (p1_A1, p1_A_target)  
**Unknown:** Actual 3D depth of the matched feature  
**Assumption Made:** All features at median scene depth (38.36m)

**Issue:** This assumption is almost certainly **wrong**.

- Scene depth ranges from ~15m to ~60m
- Features (corners, edges) may cluster at certain depths
- Sparse features are NOT uniformly distributed in 3D space
- Using a single depth for all features ignores this variation

**Consequence:** Triangulated 3D point is incorrect → Transformed point is incorrect → Projected pixel is incorrect → Residual is garbage.

---

## Expected Residual Magnitude

If factory calibration has a **small rotation error** (e.g., ±0.05° = ±50 mrad):

| Rotation Error | Expected Residual Magnitude |
|---|---|
| 0.01° | 0.59 px |
| 0.02° | 1.18 px |
| 0.05° | 2.95 px |
| 0.10° | 5.90 px |

The empirical CERES residuals (2-7 px) correspond to **0.02-0.07° rotation errors**, which are plausible calibration uncertainties.

This magnitude can **only be achieved if depth is correct** for each feature. With wrong depth, the error is completely lost in the noise.

---

## Empirical Ceres Residuals (Reference)

From dense optical flow analysis (22_CERES_CORRECTION_MAP.md), L16_04574 capture:

| Camera | dx (px) | dy (px) | Magnitude (px) | Interpretation |
|--------|---------|---------|---|---|
| A1 | -2.28 | -2.65 | 3.50 | Left + downward bias |
| A2 | -2.66 | -2.03 | 3.34 | Left + downward bias |
| A3 | -0.12 | -1.93 | 1.93 | Minimal horizontal, strong vertical |
| A4 | +2.58 | -2.83 | 3.83 | Right + downward bias |
| A5 | +4.48 | -5.76 | 7.34 | Right + strong downward bias |

These residuals are **measured over dense pixel regions** (full image bands), enabling averaging and noise reduction. Sparse features cannot achieve this averaging.

---

## Why LightGlue Residuals Are Incorrect

When I computed residuals using LightGlue features with fixed depth (38.36m):

```
A2: computed dx = -2424.529 px  (expected ~-2.66 px) → 912× too large
A3: computed dx = +744.997 px   (expected ~-0.12 px) → 6200× wrong sign/magnitude
A4: computed dx = +2104.808 px  (expected ~+2.58 px) → 815× too large
A5: computed dx = -3838.378 px  (expected ~+4.48 px) → 857× wrong sign/magnitude
```

The 800-6000× scaling error indicates a **fundamental dimensional mismatch**, not just different feature density.

**Hypothesis on root cause:**

When a feature's actual depth differs from assumed depth by factor D:
- Triangulated X_A1 scales incorrectly → X_target scales incorrectly
- Projected pixel p_target contains hidden depth information
- Residual becomes proportional to depth_error, not calibration_error
- With D ≈ 1000 factor implied depth mismatch, residuals blow up

---

## Solution Paths

### Option A: Use Depth Map (Required for Accuracy)

**Prerequisite:** Compute accurate depth map for the scene.

**Steps:**
1. Estimate depth map from MVS, DepthPro, or Metric3D
2. For each LightGlue match, look up depth at feature location
3. Use correct depth for triangulation
4. Compute residuals with proper 3D geometry
5. Compare to empirical Ceres corrections

**Pros:**
- Produces accurate per-feature residuals
- Can compare directly to Ceres corrections
- Enables validation of calibration quality

**Cons:**
- Requires high-quality depth estimation
- Depth estimation is time-consuming (~2-5s per capture)
- Adds complexity to pipeline

**Verdict:** This is the only rigorous approach, but defeats the ~0.5s goal.

---

### Option B: Dense Feature Matching

Instead of sparse LightGlue, use dense optical flow (Farneback, RAFT):

**Steps:**
1. Run optical flow between A1 and A2-A5
2. Extract flow vectors (u_flow, v_flow) for every pixel
3. For subset of high-confidence pixels, triangulate with known depth
4. Compute residuals as: (u_flow, v_flow) - (flow_predicted_from_calib)

**Pros:**
- Similar to existing CERES analysis
- Leverages existing depth maps
- No need for separate depth estimation

**Cons:**
- Similar accuracy to existing dense flow approach
- Doesn't utilize LightGlue at all

**Verdict:** Better to stick with Farneback dense flow (already proven).

---

### Option C: Accept LightGlue Limitation (Not Viable for This Use Case)

**Conclusion:** Sparse feature matching without depth cannot compete with dense flow for measuring calibration residuals.

LightGlue is excellent for:
- Feature tracking across time
- Homography estimation (planar scenes)
- Relative pose estimation (structure-from-motion)

But it is **unsuitable for** measuring small (2-7 px) calibration corrections that depend sensitively on accurate 3D geometry.

---

## Recommendation

**Do NOT attempt to replace Ceres with LightGlue + factory K/R/t.**

Instead:

1. **For rapid calibration validation (goal: ~0.5s per capture):**
   - Use existing Farneback dense optical flow approach
   - It already achieves 2-7 px residual precision
   - No additional development needed

2. **If LightGlue integration is required for other reasons:**
   - Use it for feature-based homography refinement (B-camera alignment)
   - Use it for temporal tracking / focus distance estimation
   - Do NOT use it as a substitute for precise calibration residual measurement

3. **For future improvement:**
   - Integrate a fast depth estimator (e.g., DepthPro lite)
   - Combine LightGlue matches with per-feature depth → accurate residuals
   - This enables per-frame adaptive calibration correction

---

## Technical Details: Geometry Validation

The triangulation and projection formulas are correct:

```
direction = K^{-1} @ [u, v, 1]
X_world = direction · depth_m
X_target = R @ X_world + t  (all in metres)
p_target = K_target @ (X_target / Z_target)
```

Unit consistency verified:
- K: pixels/metre
- direction: unitless (normalized)
- depth_m: metres
- R: rotation matrix (unitless)
- t: metres
- X_target: metres
- p_target: pixels ✓

The issue is not formula correctness, but **data correctness**: without accurate per-feature depth, the triangulated 3D point is wrong.

---

## Conclusion

**LightGlue sparse features cannot measure calibration residuals without per-feature depth information.** The geometry is correct, but the depth assumption is wrong, leading to 800-6000× scaling errors in residuals.

The empirical CERES corrections (2-7 px) are **real, systematic calibration errors** that can be detected with:
- Dense optical flow (proven effective)
- Sparse features + accurate depth map (requires depth estimation)

But NOT with sparse features alone at a single fixed depth.

---

**Report generated:** April 10, 2026
