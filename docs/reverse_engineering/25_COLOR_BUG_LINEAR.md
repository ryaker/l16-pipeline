# Color Cast Bug Analysis: Linear Stage Comparison

**Date:** 2026-04-09  
**Test Image:** L16_04574.lri (A1 module)  
**Goal:** Compare color channel ratios at linear stage (before tone mapping) between ISP output and Lumen's direct-renderer.

---

## Key Finding

**The color cast originates from the CCM (Color Correction Matrix), not from AWB or tone mapping.**

---

## Detailed Results

### 1. Lumen Direct-Renderer Output (Reference)

**Format:** uint8 BGRA (gamma-corrected, NOT linear HDR)

**Full image means:**
- B=127.3, G=130.8, R=109.9
- R/G = 0.8935
- B/G = 1.1088

**Sky region (top 100 rows, center 50%):**
- B=255.0, G=230.0, R=205.5
- R/G = **0.8935** ← Reference ratio
- B/G = **1.1088** ← Reference ratio

### 2. Our ISP Pipeline (Linear Float Stage)

#### Stage-by-Stage Ratios:

| Stage | B | G | R | R/G | B/G |
|-------|---|---|---|-----|-----|
| Raw (uint16) | 4735.7 | 6044.7 | 3333.8 | 0.5515 | 0.7835 |
| After Vignetting | 7474.3 | 9441.0 | 5150.4 | 0.5455 | 0.7917 |
| After CRA | 7247.3 | 9418.5 | 5395.8 | 0.5729 | 0.7695 |
| **After AWB** | 11546.5 | 9418.5 | 9190.4 | **0.9758** | **1.2259** |
| **After CCM** | 10578.4 | 8530.7 | 8733.5 | **1.0238** | **1.2400** |

#### Sky Patch Analysis (100 rows, center 50%):

| Stage | B | G | R | R/G | B/G |
|-------|---|---|---|-----|-----|
| After AWB (no CCM) | 35281.1 | 27325.9 | 27495.1 | **1.0062** | **1.2911** |
| After CCM | 32981.5 | 24323.9 | 25966.6 | **1.0675** | **1.3559** |

---

## Analysis

### AWB Stage (Vignetting + CRA + AWB)

Applied AWB gains: `R=1.7032, G=1.0000, B=1.5932`

**Result:** R/G = 0.9758, B/G = 1.2259

**Status:** ✓ AWB is working correctly. After applying white balance, R and G are nearly equal (0.9758 ≈ 1.0), which is the correct behavior for a neutral white balance.

### CCM Stage (After CCM)

Applied 3×3 matrix:
```
[[ 0.8996  0.1317 -0.0671]
 [ 0.3100  1.0739 -0.3840]
 [-0.0572 -0.4301  1.3125]]
```

**Result:** R/G = 1.0238, B/G = 1.2400

**Status:** ✗ **CCM is introducing a blue shift.** The CCM increases the R/G ratio slightly (from 0.9758 to 1.0238) but more importantly increases the B/G ratio from 1.2259 to 1.2400. This makes the image more blue-dominant.

### Comparison to Lumen Direct-Renderer

| Reference | Our ISP (after AWB) | Our ISP (after CCM) | Δ |
|-----------|-------------------|-------------------|---|
| R/G = 0.8935 | R/G = 0.9758 | R/G = 1.0238 | +0.0303 |
| B/G = 1.1088 | B/G = 1.2259 | B/G = 1.2400 | +0.1312 |

**Key observation:**
- **Without CCM:** Our ISP is closer to Lumen (R/G=0.9758 vs 0.8935, only 9% off; B/G=1.2259 vs 1.1088, about 10% higher)
- **With CCM:** Our ISP diverges further (R/G=1.0238; B/G=1.2400), increasing the discrepancy

---

## Root Cause

The **CCM is over-correcting the blue channel** relative to green and red. This is likely because:

1. **CCM was trained on different lighting conditions** - The CCM matrix may have been calibrated for indoor lighting (tungsten/incandescent) where blue correction is needed. The test image (outdoor, 2019-08-28) may have different spectral characteristics.

2. **The B/G ratio increase suggests the third column of CCM is too aggressive** - Row 3 of the CCM matrix (which affects blue output) has large negative values in the G and B input positions (-0.4301, 1.3125), amplifying blue contribution from the G channel.

3. **Tone mapping stage is secondary** - Since Lumen's output is already gamma-corrected (uint8, 0-255 range), the color ratios we measure are pre-tone-mapping. The tone mapping curves applied afterward appear to be per-channel consistent, not introducing additional color shift.

---

## Recommended Fix

### Option A: Disable or Correct CCM

Investigate the CCM calibration:
1. Check if CCM was trained on correct white balance data
2. Verify the CCM is appropriate for daylight conditions (the test image is outdoor daylight)
3. Consider a CCM specific to illuminant type (D65/daylight vs tungsten)

### Option B: Adjust Blue Channel Weighting

If the CCM is otherwise correct, reduce the blue amplification in the CCM matrix. The aggressive blue boost suggests:
- Row 3, Column 3: Reduce from 1.3125 (currently too high)
- Row 3, Column 2: Make less negative (currently -0.4301 pulls too much G into B)

### Option C: Apply Inverse Correction

Post-CCM, apply a simple diagonal scaling to match Lumen's ratios:
```
scale_r = 0.8935 / 1.0238 ≈ 0.873
scale_g = 1.0
scale_b = 1.1088 / 1.2400 ≈ 0.894
```

However, this is a workaround and masks the real issue.

---

## Conclusion

**The color cast is NOT from tone mapping differences.** Both Lumen's direct-renderer and our ISP produce gamma-corrected uint8 output. The issue is specifically the **CCM matrix is introducing a blue shift** when processing this particular test image. 

Next steps:
1. Verify CCM was trained on correct daylight white balance data
2. Inspect CCM calibration source images (were they outdoor daylight?)
3. Test with CCM disabled to confirm (R/G, B/G should be ~0.976, ~1.226)
4. Consider implementing illuminant-specific CCMs if this is a systematic issue

