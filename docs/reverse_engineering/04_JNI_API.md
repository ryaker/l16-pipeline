# JNI API = the Lumen Replacement Contract

`light_gallery.apk` on the L16 Android device exposes 32 JNI methods that fully document the public API of the Lumen pipeline. Our modern replacement should expose this same API (possibly extended, but at minimum compatible).

**Source**: `libnative-lib.so` exports extracted via `nm` from `/tmp/light_gallery.apk` (Agent E batch, 2026-04-09). The original Java/Kotlin is compiled to native AOT — no DEX bytecode — but the JNI symbol names on the native side document the exact Java method signatures.

## Library load order

```java
System.loadLibrary("c++_shared");       // LLVM C++ runtime
System.loadLibrary("ceres");            // Ceres Solver
System.loadLibrary("cp");               // libcp — main rendering pipeline
System.loadLibrary("lricompression");   // LRI file compression/decompression
System.loadLibrary("native-lib");       // JNI bridge exposing the API
```

Loaded by: `light.co.gallery.GalleryApp` (Application subclass) or `light.co.gallery.utils.LibCpRenderer` on first use.

## Class: `light.co.gallery.utils.LibCpRenderer`

31 native methods. This is the main renderer class.

### Lifecycle

| Method | Purpose |
|---|---|
| `nativeObtainRenderer() → long` | Allocate a new native `CIAPI::Renderer` instance. Returns an opaque handle. |
| `nativePrepareRenderer(String lriPath, String tempDir) → void` | Load an `.lri` file and prepare the internal state. `tempDir` is where working files may be written. |
| `nativeReleaseRenderer() → void` | Free the native renderer. |
| `nativeReset() → void` | Reset state (discard edits, keep loaded LRI). |

### Render operations

| Method | Purpose |
|---|---|
| `nativeRender(int level, ROI roi, RenderType type, boolean block) → void` | Run the pipeline for a specific pyramid level + region of interest. `block=true` waits for completion; `block=false` is async. |
| `nativeGetImage(int width, int height) → Bitmap` | Retrieve the rendered result at the given output resolution. |
| `nativeGetHistogram() → int[]` | Return histogram data for UI display. |
| `nativeAbortRenderer() → void` | Cancel an in-progress render. |

### Property (parameter) management

Six overloaded pairs — `nativeSetProperty` / `nativeGetProperty` — for different parameter types:

| Setter | Getter | Parameter type |
|---|---|---|
| `nativeSetProperty(ParamFloat, float)` | `nativeGetProperty(ParamFloat) → float` | float |
| `nativeSetProperty(ParamInt, int)` | `nativeGetProperty(ParamInt) → int` | int |
| `nativeSetProperty(ParamString, String)` | `nativeGetProperty(ParamString) → String` | string |
| `nativeSetProperty(ParamIntArray, int[])` | `nativeGetProperty(ParamIntArray) → int[]` | int array |
| `nativeSetProperty(ParamFloatArray, float[])` | `nativeGetProperty(ParamFloatArray) → float[]` | float array |
| `nativeSetProperty(ParamByteArray, byte[])` | `nativeGetProperty(ParamByteArray) → byte[]` | byte array |

Known `ParamFloat` identifiers (inferred from UI strings):
- `ViewExposure` — exposure compensation (−2.0 … +2.0)
- `ViewDofFNumber` — aperture simulation (f/1.0 … f/32.0)
- `ViewDofFocusDepth` — focus distance in metres
- `ViewColorTemperature` — white balance (3000 K … 8000 K)
- `ViewColorTint` — green ↔ magenta shift (−100 … +100)

Known `ParamInt`:
- `RenderProfile` — enum selecting render mode (Preview / Standard / HDR)
- `RenderType` — full render vs preview

Other ParamFloat values referenced in UI preset system:
- Highlights, Shadows, Saturation, Vibrance, Contrast, Sharpening

### State serialization

| Method | Purpose |
|---|---|
| `nativeSetState(String stateFilePath) → void` | Load a previously-saved state file (`.state`) and apply its settings. |
| `nativeSaveState(String outputPath) → void` | Write current editor state to a `.state` file. NOT a full LRIS. |
| `nativeGetParamFloatFromStateFile(String stateFile, String paramName) → float` | Utility to read a single param from an existing state file without loading it. |

### Transform operations

| Method | Purpose |
|---|---|
| `nativeTxReset() → void` | Reset transform (identity). |
| `nativeTxRotate(float angle) → void` | Apply rotation (degrees?). |
| `nativeTxFlipX() → void` | Mirror horizontally. |
| `nativeTxFlipY() → void` | Mirror vertically. |
| `nativeTxSetCrop(int left, int top, int right, int bottom) → void` | Set crop rectangle. |
| `nativeTxGetCrop() → int[]` | Get current crop. |
| `nativeTxGetMatrix() → float[]` | Get composed 3×3 transform matrix. |
| `nativeTxGetLriMatrix() → float[]` | Get matrix in LRI coordinate space (pre-transform). |
| `nativeTxExists() → boolean` | Whether any non-identity transform is active. |

### Depth / DOF

| Method | Purpose |
|---|---|
| `nativeSetDofDepth(float depth) → void` | Set the in-focus plane depth (metres). Drives bokeh simulation. |
| `nativeGetDepthAtPoint(float x, float y) → float` | Query per-pixel depth map at normalized image coordinates. |

### Metadata queries

| Method | Purpose |
|---|---|
| `nativeGetSize() → Size` | Image dimensions at full resolution. |
| `nativeGetLevelCount() → int` | Number of pyramid levels available. |
| `nativeGetUserRating() → int` | Current star rating. |
| `nativeSetUserRating(int) → void` | Set star rating. |
| `nativeGetLibCpVersion() → String` | libcp version string. |
| `nativeSetTraceLevels(boolean) → void` | Enable/disable verbose logging. |

### Export

| Method | Purpose |
|---|---|
| `nativeSaveImage(int width, int height, int format, String outputPath, boolean skipDepth) → void` | Render + save final image. `format` selects JPEG / PNG / HDR / DNG. `skipDepth=true` means no depth map in output (faster). |

## Class: `light.co.gallery.utils.LriCompressor`

1 native method.

| Method | Purpose |
|---|---|
| `nativeCompress(byte[] inputData, String outputPath) → boolean` | Compress LRI data to the output path. Uses `liblricompression.so`. |

## User-facing flows (reconstructed from JNI API + UI strings)

### Open + view an LRI
```
1. nativeObtainRenderer()
2. nativePrepareRenderer(lriPath, tempDir)
3. nativeRender(level=0, roi=FULL, type=PREVIEW, block=false)
4. nativeGetImage(width, height)  → preview bitmap displayed
```

### Adjust exposure/WB/DOF (live preview)
```
1. nativeSetProperty(ViewExposure, value)        // or ViewDofFocusDepth, etc.
2. nativeRender(level=0, roi=VISIBLE, type=PREVIEW, block=false)
3. nativeGetImage()  → updated preview
```

### Apply a preset
```
1. Load preset from disk (Java side)
2. for each (param, value) in preset:
     nativeSetProperty(param, value)
3. nativeRender(...)
```

### Tap-to-focus (DOF)
```
1. nativeSetDofDepth(depth)
2. nativeRender(...)
```

### Query depth at a point
```
nativeGetDepthAtPoint(x, y) → float depth
```

### Save final image
```
1. nativeSaveState(statePath)                              // persist edits
2. nativeRender(level=FULL, roi=FULL, type=FULL, block=true)
3. nativeSaveImage(width, height, JPEG, outputPath, skipDepth=false)
4. nativeReleaseRenderer()
```

## Replacement strategy

A modern Lumen replacement needs to expose these 32 methods (names can be modernized — e.g., Python bindings, Rust bindings, Metal-backed kernels, GPU depth). The SEMANTICS of each method are what matter:

| Category | Count | Criticality |
|---|---|---|
| Render pipeline entry/lifecycle | 4 | Must have |
| Render execution | 4 | Must have |
| Property get/set | 12 | Must have (at least the float params) |
| State file I/O | 3 | Should have (for re-open with saved edits) |
| Transform ops | 9 | Should have (standard photo editor features) |
| Depth query + DOF control | 2 | Must have (core L16 feature) |
| Metadata | 6 | Nice to have |
| Export | 1 | Must have |
| Compression | 1 | Nice to have |

Minimum viable product = **render pipeline + property setters + depth + export** = ~12 methods that do the heavy lifting.

## Notes on state files vs LRIS

- `.state` file = written by `nativeSaveState()` on Android and similar path on desktop. Contains user edits only (exposure, tone curve, DOF, transform, rating). Small.
- `.lris` sidecar = written by `CIAPI::StateFileEditor::serialize()` on desktop. Contains refined calibration + computed depth map + user edits. Larger. Used as Lumen's cache.
- The gallery app only produces `.state` files. It does NOT run Ceres bundle adjustment on-device (or if it does, it doesn't persist the result). Only Lumen desktop produces LRIS.

## Function entry addresses in libcp

| Public JNI-visible function | macOS libcp.dylib | Android libcp.so |
|---|---|---|
| `CIAPI::Renderer::render(...)` | `0x390180` | (offset-equivalent) |
| `CIAPI::DirectRenderer::render(Image&)` | `0x3944f0` | same |
| `CIAPI::StateFileEditor::serialize(...)` | `0x39bc40` | `0x3a2140` |
| `lt::RendererPrivate::render` | `0x3b8ba0` | same |
| `lt::RendererPrivate::deserialize` | `0x3b6f20` | same |
