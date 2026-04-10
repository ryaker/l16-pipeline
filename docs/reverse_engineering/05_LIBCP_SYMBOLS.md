# libcp Key Symbols

Reference table of the most important function addresses in `libcp.dylib` (macOS x86_64) and `libcp.so` (Android arm64), which are confirmed to be the same compiled codebase.

## Verification

| Property | macOS libcp.dylib | Android libcp.so |
|---|---|---|
| Format | Mach-O 64-bit x86_64 | ELF 64-bit aarch64 |
| File size | 6.6 MB (6,930,924 B) | 6.9 MB (7,241,464 B) |
| `__text` section size | `0x553ad0` = 5,537,488 B | **`0x553ad0`** = 5,537,488 B |
| Dynamic symbols (nm -D \| c++filt) | ~494 | ~596 (more, less stripped) |
| CIAPI methods exported | ~60 | ~120 (mangled) |
| Ceres symbols imported | Identical set | Identical set |
| Build | Lumen.app timestamp Oct 2019 | `L16-RELEASE-USER 00WW-1.3.5.1` |

**Verdict**: same codebase. Android version has more visible symbols — use it as the authoritative RE target for symbol discovery.

## Public API entry points

| Symbol | macOS address | Role |
|---|---|---|
| `CIAPI::Renderer::render(int, CIAPI::ROI const&, CIAPI::RenderType, bool)` | `0x390180` | Top-level render entry; thin wrapper |
| `CIAPI::DirectRenderer::render(CIAPI::Image&)` | `0x3944f0` | Direct render path (single image output) |
| `CIAPI::Renderer::Create(RendererProfile)` | varies | Factory |
| `CIAPI::StateFileEditor::serialize(shared_ptr<ostream>)` | `0x39bc40` | Write refined state → `.lris` / `.state` |
| `CIAPI::StateFileEditor::deserialize(shared_ptr<istream>)` | (constructor) | Read state |
| `CIAPI::StateFileEditor::hasDepthEdits()` | `0x39bbb0` | |
| `CIAPI::StateFileEditor::depthType()` | `0x39bbc0` | |

## Internal pipeline (lt::RendererPrivate)

These are NOT exported (pimpl pattern hides them). Addresses resolved via static disassembly + cross-reference.

| Symbol | macOS address | Role |
|---|---|---|
| `lt::RendererPrivate::render` | `0x3b8ba0` | Real pipeline orchestrator |
| `lt::RendererPrivate::deserialize` | `0x3b6f20` | Load LRI state |
| Task queue / dispatch | `0x3bfc40` | Work dispatching |
| Pyramid / hierarchical processing | `0x3f0130` | Multi-scale processing |
| Image resampling (Halide) | `0x3d0120` | |
| Warping / remapping | `0x3c0c70` | Primary geometry engine |
| Coordinate transform | `0x3c3ad0` | |
| Transform apply | `0x3cfd80` | |
| Boundary handling | `0x3efbb0` | |
| Calibration / optimization orchestrator | `0x3b57c0` | **CONTAINS the 3 Ceres::Solve calls** |
| Work queue sync | `0x3efd90` | |
| Logging | `0x7820` | |
| Exception handler | `0x2840` | |
| SIMD utility | `0xf540` | |

## Ceres call sites in libcp.dylib

| Offset | Hypothesized role |
|---|---|
| `0x117615` | Geometric calibration refinement (pyramid coarse level) |
| `0x202249` | Depth/stereo refinement (pyramid mid level) |
| `0x20d611` | Global bundle adjustment (pyramid fine level) |

All 3 are called from `0x3b57c0` orchestrator (not yet decompiled to confirm).

## Ceres symbols imported (from libceres.dylib)

```
ceres::Problem::Problem()
ceres::Problem::Problem(ceres::Problem::Options const&)
ceres::Problem::~Problem()
ceres::Problem::AddParameterBlock(double*, int)
ceres::Problem::RemoveParameterBlock(double*)
ceres::Problem::AddResidualBlock(...)  [3-5 overloads]
ceres::Problem::SetParameterBlockConstant(double*)
ceres::Problem::SetParameterBlockVariable(double*)
ceres::Problem::SetParameterLowerBound(double*, int, double)
ceres::Problem::SetParameterUpperBound(double*, int, double)
ceres::Solve(ceres::Solver::Options const&, ceres::Problem*, ceres::Solver::Summary*)
ceres::Solver::Solve(Options, Problem*, Summary*)
ceres::Solver::Summary::Summary()
vtable for ceres::CauchyLoss
```

## Halide runtime symbols

~160 Halide-related symbols present. Key ones:
- `_halide_malloc`, `_halide_free`
- `_halide_do_task`, `_halide_do_par_for`
- `_halide_error_unaligned_host_ptr`
- `halide_device_malloc`, `halide_device_free`, `halide_device_sync`
- Various `_halide_buffer_*`

Confirmed Halide-backed templates (with multiple specializations):
- `lt::ImageDemosaickFilter<DemosaickFilter::E0/E2/E3, float, ...>`
- `lt::RemoveVignettingGeneric<vec4x32f, bool>`

## ltpb protobuf classes (from string table)

| Class | Relevance |
|---|---|
| `ltpb.LightHeader` | Top-level LRI protobuf |
| `ltpb.CameraModule` | Per-camera runtime metadata (LRI field 12) |
| `ltpb.CameraModule.Surface` | Sensor dimensions |
| `ltpb.CameraModuleHwInfo` | Hardware info |
| `ltpb.FactoryDeviceCalibration` | Device-level calibration |
| `ltpb.FactoryModuleCalibration` | Per-camera calibration (LRI field 13) |
| `ltpb.GeometricCalibration` | K/R/t bundles |
| `ltpb.MirrorSystem` | Movable mirror params |
| `ltpb.MirrorActuatorMapping` | Hall code → angle mapping |
| `ltpb.ColorCalibration` | CCMs + spectral |
| `ltpb.ColorCalibrationGold` | Golden reference CCM |
| `ltpb.Distortion` | Contains CRA + Polynomial submessages |
| `ltpb.Distortion.CRA` | Chief ray angle (per-lens-position) |
| `ltpb.Distortion.Polynomial` | Polynomial lens distortion |
| `ltpb.SensorCharacterization` | Sensor response characterization |
| `ltpb.SensorData` | Per-camera sensor data |
| `ltpb.SensorType` | Sensor model enum |
| `ltpb.DeadPixelMap` | Bad pixel list |
| `ltpb.HotPixelMap` | Hot pixel list (with HotPixelMeasurement) |
| `ltpb.RefinedGeomCalib` | **Lumen's refined calibration output** |
| `ltpb.Calibration2DPoints` | 2D feature points for calibration |
| `ltpb.Calibration3DPoints` | 3D points for calibration |
| `ltpb.RefinedGeomCalib` | Bundle-adjust output |
| `ltpb.StereoState` | Stereo depth state |
| `ltpb.DepthEditorState` | Depth editor state (with Tile, Vec2I, DataType submessages) |
| `ltpb.DepthFormat` | Depth output format enum |
| `ltpb.IMUData` | IMU sample data |
| `ltpb.IMUData.Sample` | Single IMU sample |
| `ltpb.GPSData` | GPS metadata (LRI field 24 or 26?) |
| `ltpb.FlashData` | Flash event data |
| `ltpb.FaceDataJ` | Face detection data |
| `ltpb.AFDebugInfo` | Autofocus debug info |
| `ltpb.SettingsJ` | App settings |
| `ltpb.Compatibility` | Format compatibility info |
| `ltpb.DeviceTemp` | Device temperature |
| `ltpb.ProximitySensors` | Proximity sensor data |
| `ltpb.ViewPreferences` | User view prefs incl AWB (LRI field 19) |

### Common geometry types

| Type | Size |
|---|---|
| `ltpb.Matrix3x3F` | 9 × float (36 B packed) |
| `ltpb.Matrix4x4F` | 16 × float |
| `ltpb.Point2F` | 2 × float |
| `ltpb.Point2I` | 2 × int |
| `ltpb.Point3F` | 3 × float |
| `ltpb.Range2F` | 2 × float |
| `ltpb.RectangleI` | 4 × int |

## Files in `/tmp` with symbol details

- `/tmp/libcp_android.so` — extracted Android binary (7.2 MB)
- `/tmp/libcp_android_symbols.txt` — nm -D dump
- `/tmp/libcp_macos_symbols.txt` — nm dump
- `/tmp/DETAILED_SYMBOL_DIFF_REPORT.txt` — full analysis
- `/tmp/pipeline_callgraph_final.txt` — call graph
- `/tmp/lumen_pipeline_final_report.txt` — pipeline analysis

## Halide kernel decompilation strategy

**Do NOT attempt to read Halide-generated assembly.** Instead:
1. Set breakpoints at the Halide entry points (stage boundaries)
2. Dump input/output buffers at each boundary
3. Match our own implementation numerically against those dumps

Halide's compilation output uses SIMD tiling and loop transformations that make the generated assembly dramatically harder to decompile than vanilla C++. The boundary-observation strategy sidesteps this entirely.
