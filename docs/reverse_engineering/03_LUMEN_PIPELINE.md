# Lumen Rendering Pipeline (libcp)

Static + dynamic analysis of `libcp.dylib` (macOS x86_64) reveals the pipeline that turns a raw `.lri` capture into a final 52 MP image. Both platforms (macOS x86_64 and Android arm64) have **identical `__text` section sizes** — same compiled codebase.

## Pipeline call graph (3 levels deep)

```
CIAPI::Renderer::render(int, ROI, RenderType, bool) @ 0x390180   [PUBLIC API]
    │
    └── lt::RendererPrivate::render @ 0x3b8ba0   [INTERNAL — real pipeline lives here]
        │
        ├── Task queueing & work dispatch @ 0x3bfc40   [INFRA]
        │   │
        │   ├── Pyramid / hierarchical processing @ 0x3f0130   [GEOMETRY]
        │   │   ├── Transformation apply @ 0x3cfd80   [GEOMETRY]
        │   │   ├── Image resampling @ 0x3d0120   [ISP — likely Halide]
        │   │   └── Boundary handling @ 0x3efbb0   [ISP]
        │   │
        │   ├── Warping / remapping @ 0x3c0c70   [GEOMETRY — primary geom engine]
        │   │   ├── Synchronization @ 0x3efd90
        │   │   ├── Recursive hierarchy @ 0x3f0130
        │   │   └── SIMD utility @ 0xf540   [ISP]
        │   │
        │   └── Coordinate transformation @ 0x3c3ad0   [GEOMETRY]
        │
        ├── Work queue sync @ 0x3efd90   [INFRA]
        │
        ├── Calibration / optimization phase @ 0x3b57c0   [CALIBRATION]
        │   │
        │   └── Contains 3× Ceres::Solve calls at:
        │       • 0x117615   (hypothesized: geometric calibration, pyramid coarse)
        │       • 0x202249   (hypothesized: depth/stereo refinement, pyramid mid)
        │       • 0x20d611   (hypothesized: bundle adjustment, pyramid fine)
        │
        ├── Logging / diagnostics @ 0x7820   [INFRA]
        │
        └── Exception handler / cleanup @ 0x2840   [INFRA]
```

All addresses are from the macOS `libcp.dylib`. The Android `libcp.so` has the same code at slightly different absolute addresses but an identical relative layout (same `__text` size of `0x553ad0`).

## Pipeline stages (execution order)

### Stage 1: ISP pre-processing (per camera)
- **Black level** subtraction (data source: **unknown** — not yet located in LRI)
- **Vignetting correction** (17×13 grid from FactoryModuleCalibration photometric.f2) — **Halide-backed** via `lt::RemoveVignettingGeneric<...>` template
- **Cross-talk correction** (data + algorithm both unknown)
- **AWB** (per-channel gains from ViewPreferences.ChannelGain in LRI field 19)
- **Demosaic** — **Halide-backed** via `lt::ImageDemosaickFilter<...>` templates (multiple variants: DemosaickFilter 0, 2, 3)
- **CCM** (color correction matrix) — 3 CCMs per camera in FactoryModuleCalibration, illuminant-interpolated (method unknown)
- **Denoising** — Halide-backed, algorithm unknown
- **CRA** (chief ray angle) correction — 17×13 grid of 4×4 affine matrices from FactoryModuleCalibration photometric.f1 — **extracted but not yet applied**

### Stage 2: Geometric warping & registration (multi-camera)
- Primary: `0x3c0c70` warp/remap
- Coordinate transforms: `0x3c3ad0`
- Pyramid levels: `0x3f0130`
- Multi-view alignment using per-camera `K/R/t` from FactoryModuleCalibration
- For MOVABLE B/C cameras: virtual camera forward direction = `rodrigues((0,0,1), rotation_axis, mirror_angle)`

### Stage 3: Calibration refinement (Ceres pass 1)
- Located in `0x3b57c0` orchestrator
- Ceres `Solve` at `0x117615`
- Hypothesized: refine per-camera pose (6-DOF) against factory calibration using feature matching or photometric consistency
- Observed: ~61 scalar parameter blocks with explicit bounds
- Cost functions: photometric + geometric

### Stage 4: Multi-view stereo depth (Ceres pass 2)
- Ceres `Solve` at `0x202249`
- Hypothesized: depth/disparity refinement at the mid pyramid level
- Output: `StereoState` protobuf (written as `.lris` depth map in some form — 260×195 int32 grid)

### Stage 5: Global bundle adjustment (Ceres pass 3)
- Ceres `Solve` at `0x20d611`
- Hypothesized: final all-cameras + depth + lighting bundle adjust
- Output: RefinedGeomCalib (final per-camera refined K/R/t)

### Stage 6: Fusion / blending
- Multi-view confidence-weighted blend
- Pyramid reconstruction
- Confidence weights: unknown formula (likely depth variance + photometric consistency)

### Stage 7: Post-processing
- Highlight restoration — Halide-backed, algorithm unknown
- Local tone mapping — Halide-backed, algorithm unknown
- Global tone mapping (sRGB gamma, EV, saturation, sharpening, vibrance)
- Output format conversion (HDR float32, DNG, JPEG, PNG)

## Ceres bundle-adjustment structure (dynamically observed)

From lldb trace of `lri_process` on L16_04574.lri (no LRIS sibling → full cold-start):

| Metric | Value | Notes |
|---|---|---|
| `Problem` instances | **3** | Hypothesis: 3 pyramid levels (coarse/mid/fine) |
| Total `AddParameterBlock` | **183** | ~61 per problem |
| Parameter block size | **1 double (8 bytes)** | Per-scalar blocks, not grouped poses |
| Total `AddResidualBlock` | **347** | ~115 per problem |
| Distinct cost function pointers | **18** | Heterogeneous cost function |
| `SetParameterLowerBound` | **200** | Bounded parameters |
| `SetParameterUpperBound` | **187** | Bounded parameters |
| `Solver::Solve` calls | **~197** | Iterative re-solve pattern |
| Loss function | `ceres::CauchyLoss` | Robust (outlier-resistant) |

**Per-scalar parameter blocks + explicit bounds** = Lumen enforces physical plausibility per-parameter. Standard reasons:
- Focal-length drift bounded within small % of factory
- Principal point drift bounded within few pixels
- Rotation deltas bounded within small angle

**61 parameters per problem** matches **10 fired cameras × 6-DOF pose = 60** plus one global parameter. Hypothesis: each Ceres problem optimizes pose for all 10 firing cameras simultaneously at a different pyramid level.

**18 distinct cost functions** is more than typical bundle-adjust (which might use 2–3 residual types). Likely breakdown: per-camera-pair reprojection + photometric + factory prior + smoothness/regularization + depth-consistency residuals.

## Ceres API usage (confirmed linked)

libcp imports these Ceres symbols from `libceres.dylib`:
```
ceres::Problem::Problem()
ceres::Problem::Problem(Options)
ceres::Problem::~Problem()
ceres::Problem::AddParameterBlock(double*, int)
ceres::Problem::RemoveParameterBlock(double*)
ceres::Problem::AddResidualBlock(...)  [3-5 overloads observed]
ceres::Problem::SetParameterBlockConstant(double*)
ceres::Problem::SetParameterBlockVariable(double*)
ceres::Problem::SetParameterLowerBound(double*, int, double)
ceres::Problem::SetParameterUpperBound(double*, int, double)
ceres::Solve(Options, Problem*, Summary*)
ceres::Solver::Solve(Options, Problem*, Summary*)
ceres::Solver::Summary::Summary()
vtable for ceres::CauchyLoss
```

## Halide kernel usage

libcp contains **160+ Halide runtime symbols** (`_halide_*` helpers). Confirmed Halide-backed stages:
- Demosaicking (multiple template variants: `DemosaickFilter::E0/E2/E3`)
- Vignetting correction (`lt::RemoveVignettingGeneric<vec4x32f, bool>`)
- Probably: AWB, CCM, denoising, resampling, warp

**Strategy**: We do NOT attempt to read Halide-generated assembly. Halide IR compiles to SIMD-tiled code that's much harder to decompile than vanilla C++. Instead, we observe the **inputs and outputs** at Halide kernel boundaries via lldb/frida and match our own implementations numerically.

## Stages we have NOT yet located in the binary

- **Black level subtraction** — referenced in stage order but no symbol found
- **Dead pixel masking** — `ltpb.DeadPixelMap` class exists, code not located
- **Hot pixel masking** — `ltpb.HotPixelMap` class exists, code not located
- **Cross-talk correction** — referenced in pipeline comments, no data or code located
- **Highlight restore algorithm** — symbol not found
- **Local tone mapping algorithm** — symbol not found
- **Confidence blend formula** — not found in exported symbols
- **Stereo rectification** — no explicit epipolar/rectification symbol

These are almost all likely inside Halide kernels or hidden in template helpers with mangled inline names. Need dynamic trace or targeted decompile to locate.

## Open items

- Decompile `0x3b57c0` (calibration orchestrator) to see how the 3 Ceres problems are set up
- Decompile the 3 Ceres call sites (`0x117615`, `0x202249`, `0x20d611`) to identify which cost functions are attached to each
- Capture actual Ceres parameter values via frida or DYLD_INSERT_LIBRARIES shim (Round 3)
- Determine what's actually stored in the LRIS 6.5 MB middle blob (likely refined per-camera data)
