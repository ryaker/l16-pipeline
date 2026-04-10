# CCM Color Diagnosis Report
## Single-Camera ISP vs Lumen Direct-Renderer Comparison

**Date:** 2026-04-09  
**Test Image:** L16_04574.lri (daylight scene, 2019-08-28)  
**Camera:** A1 only  
**Purpose:** Isolate the color error in our ISP pipeline by comparing against Lumen's direct-renderer output

---

## Executive Summary

Our ISP color output is **systematically darker and color-shifted** compared to Lumen's direct-renderer. The issue is **channel-dependent** — not a uniform brightness loss. Our pipeline produces:

- **Green is too dark** (1.735x correction needed)
- **Blue is under-corrected** (1.626x needed)
- **Red is under-corrected** (1.530x needed)

This indicates the problem **is not simple AWB scaling or overall brightness loss**. The per-channel error suggests either:
1. **CCM matrix is wrong** (wrong illuminant selected or wrong matrix values)
2. **AWB gains were not fully applied**
3. **Tone-mapping curve is off** (though less likely to be channel-specific)

---

## Test Setup

### Step 1: Lumen Direct-Renderer Baseline
```bash
arch -x86_64 /Users/ryaker/Documents/Light_Work/lri_process \
    /Volumes/Base Photos/Light/2019-08-28/L16_04574.lri \
    /tmp/L16_04574_direct.tif \
    --direct-renderer
```

**Output:** `/tmp/L16_04574_direct.tif`  
- Direct-renderer profile 1 (default)
- TIFF output (lossless)
- Resolution: 4160 × 3120
- Pixel format: uint8 (BGR)

### Step 2: Our ISP Processing

**Calibration extracted from L16_04574.lri:**
- `awb_mode`: 0 (AUTO)
- `AWB gains`: R=1.703237, Gr=1.000000, Gb=1.000000, B=1.593217
- `Available CCM modes`: [2, 0, 6] (D65, Tungsten A, F11/D50)
- `Selected CCM`: Mode 2 (D65, per `awb_to_ccm[0] → 2`)

**ISP pipeline applied:**
1. Load raw A1 frame (already debayered) → BGR, uint16
2. Apply vignetting correction (17×13 grid)
3. Apply CRA correction (13×17 cells of 4×4 Bayer channel mixers)
4. Apply AWB gains: `img *= [B_gain, G_gain, R_gain]`
5. Apply CCM (mode 2, D65 matrix, 3×3 RGB transformation)
6. Tone-map to sRGB uint8 via `linear_to_srgb_uint8()`

**Output:** `/tmp/A1_our_isp.png`  
- Same resolution: 4160 × 3120
- Pixel format: uint8 (BGR)

---

## Comparison Results

| Channel | Lumen Direct | Our ISP | Ratio (Lumen/Ours) |
|---------|--------------|---------|-------------------|
| **B**   | 127.3        | 78.3    | 1.626             |
| **G**   | 130.8        | 75.4    | 1.735             |
| **R**   | 109.9        | 71.8    | 1.530             |

### Visual Interpretation

- **Green is 1.735× too dark** — this is the largest error
- **Blue is 1.626× too dark** — matches roughly with R error
- **Red is 1.530× too dark** — lowest ratio

### Color Shift Analysis

The ratios are **NOT uniform**:
- If this were just a brightness loss, all three would be equal
- Instead, green is **lifted 1.735/1.530 = 1.134×** relative to red
- This indicates a **color error**, not just exposure

---

## Diagnosis: What's Failing?

### Hypothesis 1: CCM Matrix is Wrong or Inverted ❌
- **Status:** Unlikely to be the primary cause
- **Reason:** CCM errors typically cause large color shifts (e.g., blue skin tones, green magenta shifts), not uniform darkening with green bias
- **Evidence:** The shift is systematic across the entire image, not localized color artifacts

### Hypothesis 2: AWB Gains Not Applied Correctly ⚠️
- **Status:** Possible contributor
- **Expectation:** 
  - AWB should multiply B by 1.593, R by 1.703, G by 1.0
  - This should boost Red and Blue relative to Green
  - Our output shows Red and Blue are UNDER-corrected (darker)
- **Possible Issues:**
  - Are AWB gains being applied in the wrong color order (RGB vs BGR)?
  - Are AWB gains being inverted (divided instead of multiplied)?
  - Are AWB gains not being applied at all?

### Hypothesis 3: CCM Mode Selection Wrong ⚠️
- **Status:** Possible
- **Current Logic:** `awb_mode=0 (AUTO) → ccm_mode=2 (D65)`
- **Question:** Is D65 the right illuminant for this daylight scene?
- **Test:** Compare against mode 0 (Tungsten A) or mode 6 (F11/D50) CCM matrices

### Hypothesis 4: Tone-Mapping Curve is Broken ⚠️
- **Status:** Unlikely, but possible
- **Reason:** `linear_to_srgb_uint8()` applies sRGB gamma across all channels equally
- **Evidence:** We'd expect this to darken the image uniformly, not introduce green bias
- **Less likely** but could be interacting with CCM output range

---

## AWB Mode and CCM Selection Across Captures

| Image            | awb_mode | AWB Gains (R, G, B)            | Available Modes | Selected |
|------------------|----------|--------------------------------|-----------------|----------|
| L16_04574 (day)  | 0        | 1.703, 1.000, 1.593           | [2, 0, 6]       | 2 (D65)  |
| L16_00703 (flo)  | 0        | 1.702, 1.000, 1.617           | [2, 0, 6]       | 2 (D65)  |
| L16_00684 (off)  | None     | None                           | [2, 0, 6]       | 2 (D65)  |

**Observations:**
- `awb_mode=0` consistently maps to CCM mode 2 (D65)
- Missing `awb_mode` falls back to mode 2 (D65)
- All three captures use the same illuminant assumption

---

## Next Steps for Debugging

### Priority 1: Verify AWB Application
```python
# In lri_calibration.py, confirm the AWB multiplication order
awb_bgr = np.array([awb['B'], (awb['Gr']+awb['Gb'])/2, awb['R']], dtype=np.float32)
img = img * awb_bgr  # Element-wise multiplication
```

Questions:
- Are we multiplying or dividing by AWB gains?
- Are we in the correct color space (BGR vs RGB)?
- Is the AWB being applied *before* or *after* CRA?

### Priority 2: Cross-Check CCM Mode Selection
- Extract CCM matrices (modes 0, 2, 6) and apply each to test image
- Compare results to Lumen direct-renderer
- Verify D65 is correct for daylight scenes

### Priority 3: Tone-Mapping Audit
- Read Lumen's tone-mapping algorithm (if available in lri_process source)
- Compare to our `linear_to_srgb_uint8()` implementation
- Test tone-mapping on known reference values

### Priority 4: Vignetting & CRA Interaction
- Verify vignetting grid shape and application direction
- Check if CRA 4×4 matrices are inverting or doubling corrections
- Test pipeline stages in isolation

---

## Files Generated

- `/tmp/L16_04574_direct.tif` — Lumen direct-renderer baseline (4160×3120, uint8)
- `/tmp/A1_our_isp.png` — Our ISP output (4160×3120, uint8)

Both files can be visually inspected or used for further pixel-level analysis.

---

## Conclusion

The color error is **real, quantified, and channel-dependent**. The most likely culprits are:

1. **AWB gains application error** (wrong sign, wrong order, or missing)
2. **CCM illuminant selection error** (using D65 when Tungsten A is correct)
3. **Tone-mapping differences** (less likely)

Recommend investigating AWB first, as it's the most straightforward fix and directly explains the green-biased darkening we observe.
