# Light L16 LRI Processing — Project Handoff

## Goal

Build a working command-line LRI processor using Light's own `libcp.dylib` — their computational photography engine, extracted from the archived Lumen.app. The library is fully self-contained, has no auth/network dependencies, and exposes a clean C++ API (`CIAPI` namespace). We can call it directly under Rosetta 2 on M4.

The owner has two Light L16 cameras (currently in a storage unit) and is recovering ~1TB of LRI files from HD to SSD. This tool processes those files using Light's own algorithms, then we layer modern AI (depth estimation, super-resolution, NeRF) on top.

---

## What's in This Folder

```
~/Documents/Light Work/
├── Lumen/
│   ├── Lumen-2.3.0.606.dmg        # archived installer (from archive.org)
│   └── Lumen.app/
│       └── Contents/
│           ├── MacOS/Lumen         # x86_64 executable — freezes on M4 (auth server dead)
│           ├── Frameworks/
│           │   ├── libcp.dylib         ← THE PRIZE (6.9MB, CIAPI namespace)
│           │   ├── liblricompression.dylib  ← LRI file decoder (2.8MB)
│           │   ├── libceres.dylib      ← Google Ceres Solver, multi-view optimization
│           │   └── Qt*.framework/      ← UI frameworks (not needed for CLI)
│           └── PlugIns/            # Qt plugins (not needed)
└── LRI/
    ├── L16_00177.lri               # confirmed readable, LELR magic bytes
    └── L16_03632.lri
```

The 1TB LRI archive is being recovered to SSD separately — path TBD when available.

---

## Key Technical Findings

### Why Lumen won't run on M4
- Binary is x86_64 only (no arm64 / fat binary)
- Rosetta 2 IS available and confirmed working (`arch -x86_64 uname -m` → `x86_64`)
- Lumen **freezes at launch** because the main executable calls a dead activation/license server
- The activation is in the **executable only** — not in any of the libraries

### libcp.dylib — No Auth, Fully Usable
```
otool -L libcp.dylib shows only:
  @rpath/libcp.dylib
  @rpath/libceres.dylib
  /usr/lib/libSystem.B.dylib
  /usr/lib/libc++.1.dylib
```
Zero network dependencies. Zero auth. We can load and call it directly.

### The Full CIAPI (Computational Imaging API)

All symbols are exported and demangled cleanly. Key classes:

#### `CIAPI::Renderer` — main async renderer
```cpp
static Renderer Create(RendererProfile);
void setInputDataStream(shared_ptr<istream>);        // feed the .lri
void setOutputUpdateListener(function<void(ImagePyramid const&, ROI const&, int)>);
void setProgressUpdateListener(function<void(int)>);
void setStateChangeListener(function<void(StateChange, void*)>);
void render(int resolution, ROI const&, RenderType, bool async);
void writeImage(shared_ptr<ostream>, Point<int> const&, ExportImageFormat, function<void(int)>);
void serialize(shared_ptr<ostream>, StateType);
void deserialize(shared_ptr<istream>, StateType);
static bool IsHardwareCompatible();
void setMode(RenderingMode);
void abort();
void cancelRenderRequests();
ImagePyramid outputBuffer() const;
```

#### `CIAPI::DirectRenderer` — simpler synchronous renderer
```cpp
static DirectRenderer Create(int);
void setInputDataStream(shared_ptr<istream>);
Image render(Image&);     // single call, synchronous — SIMPLEST ENTRY POINT
```

#### `CIAPI::RendererBase` — base for both renderers
```cpp
void setInputDataStream(shared_ptr<istream>);
void setInputDataStream(void const*, size_t);   // raw bytes overload
void setProperty(ParamFloat, float);
void setProperty(ParamInt, int);
void setProperty(ParamString, string const&);
void setProperty(ParamIntArray, vector<int> const&);
void setProperty(ParamByteArray, vector<uint8_t> const&);
void setProperty(ParamFloatArray, vector<float> const&);
void transform();
void reset(ResetMode);
```

#### `CIAPI::Image`
```cpp
static Image Create(int w, int h, PixelFormat, int stride, int, void* data);
Image subImage(ROI const&);
```

#### `CIAPI::ImagePyramid`
```cpp
static ImagePyramid Create(int levels);
Image operator[](int level);
```

#### `CIAPI::DepthEditor` — post-capture depth editing
```cpp
DepthEditor(Renderer&);
float getDepthAtPoint(Point<float> const&);
void pushBrushDepthEdit(BrushDepthEditingParams const&);
void pushLassoDepthEdit(LassoDepthEditingParams const&);
void pushEdgeHealDepthEdit(HealDepthEditingParams const&);
void pushSurfaceHealDepthEdit(HealDepthEditingParams const&);
void addQuickSelectStrokes(QuickSelectDepthEditingParams const&);
void pushQuickSelectDepthEdit(float);
void enableFaceMatting(bool);
void quickSelectMask();
void clearQuickSelectMask();
void resetQuickSelect();
void undoDepthEdit();
void redoDepthEdit();
void reset();
```

#### `CIAPI::StateFileEditor` — LRIS state file (edit metadata)
```cpp
StateFileEditor();
StateFileEditor(shared_ptr<istream>);
StateFileEditor(void const*, size_t);
bool hasDepthEdits();
Image getThumbnail();
void setTransform(Transform const&);
void setUserRating(int);
void setProperty(ParamFloat, float);
void serialize(shared_ptr<ostream>);
```

#### Utility functions
```cpp
CIAPI::GetVersion()
CIAPI::StaticShutdown()
CIAPI::ApplyTuning(TuningType, RendererBase&)
CIAPI::CreateMultiStream(vector<shared_ptr<istream>> const&)  // combine module streams
CIAPI::CreateMemStream(void const*, size_t)
```

#### `ltCompress::CompressLRI` (from liblricompression.dylib)
```cpp
CompressLRI(shared_ptr<istream>, shared_ptr<ostream>, int quality, bool, int)
CompressLRI(string input_path, string output_path, int quality, bool, int)
```

---

## Build Plan for CLI Tool

### Approach
Write a C++ CLI tool compiled as **x86_64** that:
1. Accepts an `.lri` file path as argument
2. Opens the file as an `ifstream`
3. Wraps in `shared_ptr<istream>`
4. Calls `CIAPI::DirectRenderer::Create(0)` 
5. Calls `renderer.setInputDataStream(stream)`
6. Calls `renderer.render(image)`
7. Writes output as TIFF

### Compile command
```bash
clang++ -arch x86_64 \
  -std=c++17 \
  -I./include \
  -L"./Lumen/Lumen.app/Contents/Frameworks" \
  -lcp -lceres \
  -Wl,-rpath,"./Lumen/Lumen.app/Contents/Frameworks" \
  -o lri_process \
  lri_process.cpp
```

Run under Rosetta 2:
```bash
arch -x86_64 ./lri_process input.lri output.tiff
```

### Unknown enum values to discover
The following enum types are referenced but their values aren't in the symbol table — they'll need to be probed or Ghidra'd:
- `CIAPI::RendererProfile` (arg to `Renderer::Create`)
- `CIAPI::RenderType`
- `CIAPI::RenderingMode`
- `CIAPI::ResetMode`
- `CIAPI::StateType`
- `CIAPI::ExportImageFormat`
- `CIAPI::ParamFloat`, `ParamInt`, `ParamString` etc. (property keys)
- `CIAPI::TuningType`

**Strategy:** Start with `DirectRenderer` (fewer unknowns) and try integer values 0, 1, 2 for the `Create(int)` argument. If that fails, load Ghidra to find enum definitions in the binary.

---

## Ghidra for Enum Discovery (if needed)

Install: `brew install ghidra` or download from ghidra-sre.org

Load `libcp.dylib` → auto-analyze → search for functions near the known addresses:
- `CIAPI::DirectRenderer::Create` @ `0x0000000000394240`
- `CIAPI::Renderer::Create` @ `0x0000000000390540`

The enum values will be visible as immediate constants in the decompiled code.

Optional: [GhidrOllama](https://github.com/lr-m/GhidrOllama) or [OGhidra](https://github.com/llnl/OGhidra) for LLM-assisted annotation.

---

## LRI Format Reference

- Magic bytes: `LELR` (confirmed in `L16_00177.lri`)
- Format: protobuf messages wrapping Bayer JPEGs
  - Message type 0: LightHeader
  - Message type 1: ViewPreferences  
  - Message type 2: GPSData
  - Each CameraModule has a `sensor_data_surface` (Surface type)
  - Color sensors: 4 half-res JPEGs (one per Bayer position)
  - Mono sensors: 1 full-res JPEG
- Reference: https://www.eternalhorizons.org/light/lri
- Rust decoder: https://github.com/gennyble/lri-rs

---

## Community Resources

- [helloavo/Light-L16-Archive](https://github.com/helloavo/Light-L16-Archive) — archive of everything L16
- [schweizerbolzonello/light-l16-archive](https://github.com/schweizerbolzonello/light-l16-archive)
- [discuss.pixls.us — Decoding LRI raw files](https://discuss.pixls.us/t/decoding-light-l16-lri-raw-files/24219)
- [eternalhorizons.org — L16 Re-engineering](https://www.eternalhorizons.org/lighl16)
- [ookami125/openlight-camera](https://github.com/ookami125/openlight-camera) — camera APK repackaged
- [dllu/lri-rs](https://github.com/dllu/lri-rs) — alternate Rust LRI parser
- [Lumen on archive.org](https://archive.org/details/lumen-for-l16)

---

## Phase 2 — Modern AI Pipeline

Once we can extract the 16 individual sensor frames from an LRI (either via CIAPI or the Rust decoder):

| Stage | Tool |
|---|---|
| Multi-view depth | MVSNeRF, COLMAP + NeRF, 3D Gaussian Splatting |
| Super-resolution | Real-ESRGAN, SUPIR |
| Computational bokeh | DDPM lens blur, EBokeh |
| HDR fusion | HDR-Transformer, DeepHDR |
| Frame alignment | RAFT optical flow |

The L16's fixed, calibrated geometry (known baselines, known focal lengths) makes it ideal for multi-view stereo — better than ad-hoc multi-camera setups.

---

## Next Steps (in order)

1. **Write `lri_process.cpp`** — thin wrapper calling `CIAPI::DirectRenderer`
2. **Compile x86_64** with clang, link against Frameworks dir
3. **Test with `L16_00177.lri`** — confirm it renders without crashing
4. **Probe enum values** — try 0,1,2 for `DirectRenderer::Create(int)` first
5. **If blocked on enums** — load `libcp.dylib` in Ghidra, find the Create() function at offset `0x394240`
6. **Once working** — batch process the 1TB LRI archive when SSD recovery is complete
7. **Extract 16 individual module frames** via CIAPI for AI pipeline input

---

*Context from prior conversation: owner has 2 Light L16 cameras in storage, ~1TB of LRI files being recovered to SSD. Lumen archived from archive.org. All analysis done 2026-04-03.*
