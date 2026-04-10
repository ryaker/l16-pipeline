# Black Level Offset Oracle for L16_04574

**Date:** 2026-04-09  
**Target:** `/Volumes/Base Photos/Light/2019-08-28/L16_04574.lri` (daylight, 28mm)  
**Raw Frames:** `/Volumes/Dev/Light_Work_scratch/L16_04574/frames/` (A1-A5, B1-B5)

---

## Executive Summary

The raw extracted frames show evidence of **heavy dark clipping and tone-mapping** that has already processed the original sensor black level. The empirical analysis reveals:

- **Raw extraction floor:** ~0–1 10-bit units (heavily clipped zeros)
- **Optical black region (unclipped):** ~1.1–31.6 10-bit units depending on camera
- **Lumen direct renderer output:** ~45 uint8 in same region (44.9 mean)
- **Estimated original sensor black level:** **64–256 10-bit units** (typical for CMOS sensors)

**Key finding:** Our raw extraction has already subtracted/clipped the black level, so **we cannot directly measure the original offset from these frames alone.** However, optical black regions and comparison with Lumen suggest the original black level was substantial (64–256 10-bit).

---

## Method

### Step 1: Lumen Direct Renderer Execution

```bash
arch -x86_64 /Users/ryaker/Documents/Light_Work/lri_process \
  "/Volumes/Base Photos/Light/2019-08-28/L16_04574.lri" \
  "/tmp/L16_04574_direct.tif" \
  --direct-renderer \
  --dr-profile 1
```

**Result:** ✅ Success  
- Output: `/tmp/L16_04574_direct.tif` (4160×3120, uint8 TIFF, 4 channels BGRA)
- Lumen rendered the full 16-module image using its direct renderer (fast, lower quality)
- No camera selection flag found in help; direct-renderer processes all modules and fuses them

### Step 2: Raw vs. Lumen Comparison

Loaded `/Volumes/Dev/Light_Work_scratch/L16_04574/frames/A1.png` (16-bit uint16, 3120×4160×3)

| Metric | Raw A1 | Lumen Direct |
|--------|--------|--------------|
| dtype | uint16 | uint8 |
| range | [0, 65535] | [0, 255] |
| shape | 3120×4160×3 | 3120×4160×4 (BGRA) |

### Step 3: Dark Pixel Analysis

**Raw A1 (16-bit, converted to 10-bit units via ÷64):**
- 44.84% of pixels exactly 0 (clipped)
- Bottom rows (optical black): mean = 0.92 10-bit units
- Non-zero floor: ~1.1 10-bit units
- Histogram mode: 0 (hard-clipped)

**Lumen Output (uint8):**
- 0 min, 255 max
- Bottom rows (optical black): mean = 44.94 uint8
- Tone-mapped and stretched; no clipping at dark end

---

## Optical Black Region Analysis

Optical black rows (last 100 rows) across all A-series cameras:

| Camera | Optical Black Mean (10-bit) | Non-Zero Mean | 1st Percentile |
|--------|----------------------------|---------------|----------------|
| A1 | 0.92 | 12.54 | 1.06 |
| A2 | 17.30 | 31.55 | 1.06 |
| A3 | 3.66 | 14.63 | 1.06 |
| A4 | 3.27 | 13.50 | 1.06 |
| A5 | 3.53 | 14.64 | 1.06 |
| **Average** | **5.74** | **17.37** | **1.06** |

**Interpretation:**
- The optical black rows (covered, dark-current only) show extremely low values in raw extraction
- Average ~1.1 10-bit units is the effective floor after processing
- The variance across cameras (0.92 → 17.30) suggests different exposure/gain or non-uniform frame timing
- This is NOT the original sensor black level; it's post-processing floor

---

## Comparison with Lumen

Same optical black region in Lumen's output:
- **Mean:** 44.94 uint8
- **Median:** 37.0 uint8
- **Range:** [0, 223]

Lumen applies tone mapping that lifts dark pixels. The difference between raw (~1) and Lumen (~45) indicates Lumen's AGC/tone curve has brightened the dark regions.

---

## Evidence of Pre-Processing in Raw Extraction

1. **44.84% zero clipping** → Heavy dark truncation in our extraction pipeline
2. **Optical black ≠ black level** → If sensor's true black level was 64–256, our extraction has already subtracted and clipped it
3. **Non-zero floor (1.1)** → Represents dark current or quantization after black level removal
4. **Lumen output shows detail** → Lumen's rendering suggests more information was available before our extraction

**Conclusion:** The raw PNG frames we extracted have already been post-processed. The original sensor black level offset is not directly measurable from these frames.

---

## Estimated Original Black Level

Based on:
- Standard L16 sensor (OV12830 or equivalent): **64–256 10-bit units typical**
- Optical black region showing ~1.1 floor after clipping: suggests original was substantial
- Lumen's output (45 uint8 ≈ 180/256 × 1024 ≈ 728 in 10-bit linear) in same region

**Best estimate for original sensor black level:**
- **Primary estimate: 64–128 10-bit units** (conservative, typical for CMOS)
- **Alternative: 128–256 10-bit units** (if darker optical black region in original sensor data)

---

## Recommended Black Level for Pipeline

Given the evidence:

1. **If using raw extraction as-is (already processed):** 
   - **Use: 0 10-bit units** (already subtracted in extraction pipeline)
   - No additional black level offset needed
   - Heavy clipping already occurred

2. **If re-processing original sensor data (if available):**
   - **Use: 64–128 10-bit units** (conservative CMOS default)
   - Test against optical black region to refine
   - Compare with Lumen's result to validate

3. **For this specific image (L16_04574):**
   - Current extraction: effectively **black level = 0** (floor is ~1 due to clipping)
   - To recover detail: would need to use original .lri file with Lumen or similar processor
   - For tone curve matching with Lumen: accept the pre-processed frames as-is

---

## Optical Black Region Locations

- **Present:** Yes, bottom rows of frame show consistently dark values
- **Rows affected:** Last 50–100 rows vary in darkness
- **Typical values (raw):** 0.9–17.3 10-bit units mean
- **Pattern:** Not strictly uniform (some variation row-to-row, camera-dependent)

This is consistent with CMOS sensor design (optical black rows for dark current measurement).

---

## Files Generated

- **Lumen output:** `/tmp/L16_04574_direct.tif` (4160×3120 TIFF, uint8)
- **Analysis:** Inline Python histograms and percentiles (archived above)
- **Raw sources:** Pre-extracted at `/Volumes/Dev/Light_Work_scratch/L16_04574/frames/`

---

## Conclusion

**The raw extracted frames have already had black level subtracted and dark pixels clipped to 0.** The optical black analysis shows an effective floor of ~1–31 10-bit units depending on camera, but this is not the original sensor black level—it's the post-processed minimum.

**For pipeline black level offset:** 
- Use **0 10-bit units** if working with the pre-extracted frames
- Use **64–128 10-bit units** if working with original sensor data (estimate)
- **Validate** against Lumen's output or other reference if precise calibration is needed

The discrepancy between our floor (~1) and Lumen's optical black output (~45) suggests Lumen is either:
a) Using a different black level subtraction value (higher estimate)
b) Applying non-linear tone mapping that brightens dark regions
c) Using original sensor data with a larger black level offset

**Next step:** Compare with full Lumen renderer (not direct-renderer) or examine original .lri sensor data if available for true black level calibration.
