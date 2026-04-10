# Open Questions

What we don't know yet, ordered roughly by importance for the Lumen-replace goal.

## Tier 1 — Must answer to match Lumen output

### 1. Ceres cost function parameter values
We have **counts and structure** of Ceres calls (3 Problems, 183 scalar params, 347 residuals, 18 cost types) but NOT the actual parameter values being optimized. Without knowing whether the parameters are pose values or intrinsics or both, we can't write an equivalent bundle adjustment.

**How to answer**: Round 3 dynamic trace via `frida` or `DYLD_INSERT_LIBRARIES` shim (zero debugger overhead) to capture `AddParameterBlock` values + `AddResidualBlock` arity (parameter blocks per residual).

### 2. Ceres cost function bodies (the 18 residual types)
We have 18 distinct cost function pointers but no idea what they COMPUTE. Lumen might use reprojection error, photometric consistency, smoothness priors, depth consistency, factory priors, or combinations. We need at least one concrete example of the cost function's input/output behavior to reimplement it.

**How to answer**:
- Static decompile of `0x3b57c0` calibration orchestrator and the 3 Ceres call sites (`0x117615`, `0x202249`, `0x20d611`) to see which cost-function virtual classes are instantiated.
- Dynamic trace the first call to each cost function with a trivial input to observe the Jacobian structure.

### 3. Multi-view stereo depth algorithm
Lumen writes a 260×195 `int32` depth map in the LRIS sidecar (negative values suggest fixed-point disparity). We know it uses some multi-view stereo approach but we don't know whether it's SGBM, PatchMatch, plane-sweep, MVS, or something else. The `StereoState` protobuf exists but its schema is unknown.

**How to answer**:
- Trace the depth map values over the course of the second Ceres call — it should refine this.
- Look for `StereoState` schema in the decompiled Android libcp.so (more symbols visible).
- Diff factory calibration with LRIS refined state to see where the depth enters.

### 4. CRA grid application algorithm
We extract the 17×13×(4×4 affine) CRA grid from FactoryModuleCalibration photometric.f1 but never apply it. Lumen definitely does (it's in the pipeline order). Without this, our per-camera images have residual geometric error that combines with sub-pixel bundle-adjust drift to produce blurry merges.

**How to answer**:
- Trace how libcp reads field 13 photometric.f1 into memory.
- Identify the CRA-apply function via pipeline mapping (near the Halide `RemoveVignettingGeneric`).
- Compare a single raw camera frame → Lumen's processed-per-camera output to see what the CRA correction actually DOES to pixel geometry.

### 5. Black level values
Referenced in the pipeline stage list ("1. black_level subtraction") but NOT found in any LRI field we've decoded so far. Options:
  - (a) Hidden in field 23 (the proto2-groups blob)
  - (b) Hidden in `FactoryModuleCalibration` photometric `f4` (the unknown int)
  - (c) Inside a sub-field of one of the ColorCalibration entries
  - (d) Hardcoded in libcp per sensor model
  - (e) Read from sensor dark pixels at capture time

**How to answer**:
- Complete LRI field 23 decode.
- Check libcp strings for hardcoded numeric tables keyed by sensor model.
- Dynamic trace which bytes libcp reads for black-level.

### 6. CCM interpolation method
We have 3 CCMs per camera at modes 0, 2, 6 with an unknown meaning, and no documented interpolation method. Likely:
- Mode 0/2/6 correspond to 3 standard illuminants (D65, A, F11, or similar)
- Lumen chooses one or blends between them based on scene color temperature (from AWB)

**How to answer**:
- Decompile the CCM-application stage in libcp (hunt for "CCM" or "ColorCorrection" symbols)
- Trace which of the 3 modes is used for a known daylight capture
- Check if the illuminant chromaticity coords (f4/f5 in ColorCalibration) reveal the identity

## Tier 2 — Important but replaceable with modern equivalents

### 7. Demosaic algorithm specifics
We know it's Halide-backed with multiple template variants (DemosaickFilter 0, 2, 3). Modern debayer is well-understood — we can use OpenCV or similar and validate against Lumen's output. The specific algorithm matters for sharpness + artifact characteristics but isn't a hard blocker.

### 8. Vignetting interpolation method
We apply the 17×13 grid bilinearly. Lumen's Halide kernel may use bicubic or spline. Diffable by comparing a single vignette-corrected frame.

### 9. Confidence blend weights formula
Lumen's multi-view blend uses some weighting scheme. We use our own guess. Result will differ in edge regions / occlusion boundaries until we match theirs. Validatable by diffing.

### 10. Tone mapping curves + exposure
Standard operations (sRGB gamma, EV, saturation, sharpening, vibrance) with specific parameter values. Can tune empirically to match `lri_process` output.

### 11. Denoising algorithm
Halide-backed. Can replace with modern learned or classical denoise.

### 12. Highlight restoration
Referenced in pipeline but no specific symbol. Can implement a standard highlight-reconstruction algorithm.

### 13. Local tone mapping
Referenced in pipeline. Can use modern local tone mapping (e.g., gradient-domain, bilateral grid).

## Tier 3 — Structural unknowns

### 14. LRI field 16 semantic meaning
32-entry LUT indexed 100..775 step 25, 8 floats per entry. The indices could be:
  - Focus distances in some unit (cm? mm? diopters×100?)
  - Color temperatures (mireds? not quite right range)
  - Wavelengths (nm? doesn't fit)
  - Lens position hall codes (doesn't fit — hall codes are 8000+)

Does it matter? Depends on whether Lumen uses it in the pipeline or if it's diagnostic/informational.

**How to answer**: Check libcp for reads of field 16 — which function pulls those 8 floats and what does it do with them?

### 15. LRI field 23 actual content
Starts with proto2 group tag (wire types 3/4) which standard parsers bail on. Earlier agent claimed it's per-focus black level LUT but the claim isn't verified.

**How to answer**: Re-decode with proto2-group-aware parser and persistent output file.

### 16. LRIS 6.5 MB middle blob
The LRIS file has 32-byte header + ~203 KB depth map + 82-byte protobuf + **6.5 MB unknown** + ~30-byte trailer. The unknown section likely contains per-camera refined calibration + possibly higher-res depth tiles or thumbnail.

**How to answer**: scan the blob for protobuf tag patterns, JPEG magic, or known structure prefixes.

### 17. Dead pixel + hot pixel map locations
Class names exist in libcp (`ltpb.DeadPixelMap`, `ltpb.HotPixelMap`) but the data is not in any LightHeader field we've decoded. Possibilities:
  - Stored in a separate LELR block (`msg_type != 0`)
  - Stored per-camera inside `FactoryModuleCalibration` under an unparsed sub-field
  - Stored in an external calibration file (`/Volumes/Base Photos/Light/.../L16_05036_cal/` dir exists but was empty)

### 18. `_cal` directory contents
The archive contains `L16_05036_cal`, `L16_05031_cal`, etc. — directories next to some LRI files. They appeared empty when checked. Are they always empty? Did they ever contain something? What did Lumen write there?

### 19. real_camera_orientation usage
`MirrorSystem` has a `real_camera_orientation` field (field 2, Matrix3x3F) that we don't use. The periscope camera architecture (front-view image confirms) has the sensor facing sideways, so there must be a real-sensor-to-virtual-camera orientation transform. We bypassed it with the V1 Rodrigues formula which happens to work — but the "correct" physical interpretation probably uses real_camera_orientation × mirror reflection.

**How to answer**: Compare V1 formula results against the true physical derivation using real_camera_orientation. If they match exactly, our shortcut is fine. If they drift, the full formula matters.

### 20. Focal scale factor (photometric.f3)
Extracted per camera, meaning unknown. Might be a small (< 0.1%) multiplier on fx/fy to account for thermal or mechanical effects.

### 21. photometric.f4 unknown int
Extracted, meaning unknown.

## Tier 4 — Nice-to-know

### 22. Where do the 3 Ceres Problems get their initial values?
Each of the 3 pyramid levels presumably starts with an initial guess. Is the coarsest level initialized from factory calibration? Is the mid level initialized from the refined coarse output? Understanding this helps our implementation warm-start correctly.

### 23. What triggers on-device `.state` file writes?
The gallery app writes state files when the user edits an image. Does it also write on first open? On rating change? Needs tracing.

### 24. Halide kernel count total
160 runtime symbols but unknown number of distinct kernels. Useful to know for completeness tracking.

### 25. RendererProfile enum values
`CIAPI::RendererProfile` enum has values (Preview, Standard, HDR inferred). Exact enum values + what they configure internally is useful for API design.

### 26. Multi-pyramid level count
`nativeGetLevelCount()` returns "number of pyramid levels" — typically 5 or so for computational photography. Exact number for L16 unknown.

### 27. What does `tempDir` in `nativePrepareRenderer(lriPath, tempDir)` get used for?
Second argument to the prep function. Presumably intermediate scratch space. Does it cache refined state between calls? Does it write anything the user might care about?

### 28. Default property values
Each `ParamFloat`, `ParamInt` etc. has a default. What are they? Needed for our replacement to match initial state.

## Closed questions (answered in prior batches)

- ✅ Does the camera write LRIS on-device? **NO** (only the desktop Lumen does, via StateFileEditor::serialize)
- ✅ Are Android libcp.so and macOS libcp.dylib the same codebase? **YES** (identical `__text` section sizes)
- ✅ Does Lumen use Ceres Solver? **YES** (libceres.dylib linked, 15 symbols imported, 3 Problem instances observed dynamically)
- ✅ Do MOVABLE B cameras point at A FOV corners? **YES** (37° off-axis confirmed; V1 Rodrigues formula works)
- ✅ Is field 16 RefinedGeomCalib? **NO** (it's a 32-entry LUT with unknown semantic)
- ✅ Does openlight-camera contain processing code? **NO** (capture app only)
