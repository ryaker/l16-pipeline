# Light L16 Reverse Engineering — Index

**Goal**: Build a modern host-side replacement for Lumen (Light's discontinued desktop image processor) that turns L16 `.lri` captures into sharp 52MP images. Target audience: any L16 owner worldwide.

**Scope in**:
- Host-side LRI → final image processing
- Running on modern hardware (Apple Silicon, Linux, potentially Windows)
- Open source, self-contained, no dependency on Light's dead servers

**Scope out**:
- L16 firmware / on-device Android OS (the camera works fine — don't replace it)
- USB protocol / camera file transfer (L16 mounts as mass storage, just copy files)
- Reprocessing existing LRIs (only if our pipeline is provably better than Lumen)

## Documents in this collection

| Doc | Topic |
|---|---|
| `00_INDEX.md` | This file |
| `01_LRI_FORMAT.md` | The `.lri` file format (LELR blocks + LightHeader protobuf) |
| `02_LRIS_FORMAT.md` | The `.lris` sidecar format (Lumen's cache) |
| `03_LUMEN_PIPELINE.md` | libcp rendering pipeline + stage ordering + Ceres structure |
| `04_JNI_API.md` | The 32-method JNI contract = our replacement's public API |
| `05_LIBCP_SYMBOLS.md` | Key function addresses in libcp.dylib / libcp.so |
| `06_OPEN_QUESTIONS.md` | What we don't know yet |

## Top-level facts (as of 2026-04-09)

1. **The L16 has 16 cameras**: 5×28mm (A1–A5, direct-fire), 5×70mm (B1–B5, periscope+mirror), 6×150mm (C1–C6, periscope+mirror).
2. **Capture modes fire different subsets**: 28mm mode fires 5A+5B=10 cameras; 70mm fires 5B+6C=11; 150mm fires 6C.
3. **The Lumen engine (libcp) is the SAME codebase on macOS and Android**. Android's libcp.so and macOS's libcp.dylib have **identical `__text` section sizes** (0x553ad0 bytes). One codebase, two platforms.
4. **Lumen runs Ceres Solver** (Google's nonlinear least-squares library) at render time to refine factory calibration per-capture. Dynamically observed: **3 Ceres::Problem instances, 183 AddParameterBlock calls (size=1 double each), 347 AddResidualBlock calls, 18 distinct cost functions, ~197 Solver::Solve calls**.
5. **Halide kernels power the ISP stages** (demosaic, vignetting, etc). Halide-generated assembly is opaque — we observe stage input/output instead of reading kernel source.
6. **The camera's Android app has no processing**. `ookami125/openlight-camera` = the capture app only (writes raw LRI via LELR blocks + Square Wire protobufs). No Ceres, no calibration refinement.
7. **The gallery app (`light_gallery.apk`) DOES contain libcp.so + libceres.so + liblricompression.so** = on-device Lumen.
8. **LRIS is Lumen-desktop-only.** The on-device gallery writes `.state` files with user edits (tone curve, DOF, rating) — NOT full LRIS sidecars. LRIS is written by `CIAPI::StateFileEditor::serialize` (Lumen desktop).
9. **9,438 LRI files + 5,466 LRIS files** in the archive at `/Volumes/Base Photos/Light/`. Only ~58% of LRIs have LRIS siblings (those that have been processed through Lumen at least once).

## Critical assets

| Asset | Path |
|---|---|
| Lumen app (decomposable) | `/Users/ryaker/Documents/Light_Work/Lumen/Lumen.app/` |
| libcp.dylib (macOS x86_64, 6.6 MB, unstripped) | `Lumen.app/Contents/Frameworks/libcp.dylib` |
| libcp.so (Android arm64, 6.9 MB, stripped but more dynamic symbols) | `/tmp/libcp_android.so` (extracted from `light_gallery.apk`) |
| libceres.dylib / libceres.so | In both frameworks |
| `lri_process` binary (Lumen CLI renderer) | `/Users/ryaker/Documents/Light_Work/lri_process` (x86_64, runs under Rosetta 2) |
| Light tech docs | `/Users/ryaker/Documents/Light_Work/l16-tech-part-1-3.md` |
| LRI archive (9,438 files) | `/Volumes/Base Photos/Light/` |
| LRIS archive (5,466 files) | Same, `.lris` siblings |
| Scratch directory | `/Volumes/Dev/Light_Work_scratch/` |

## Reference repos

- `ookami125/openlight-camera` — decompiled L16 Android camera CAPTURE app (writes LRI, no processing)
- `helloavo/Light-L16-Archive` — archive of original L16 APKs (camera + gallery + framework)

## IEEE Spectrum article

"Inside the Development of Light, the Tiny Digital Camera That Outperforms DSLRs" by Rajiv Laroia (Light founder), Oct 2016. Key quote confirming the B-camera quadrant layout:

> Our camera adjusts the mirrors in front of four of those [70mm] lenses so that different modules point at each of the four quadrants of the 28-mm frame we're trying to take... The fifth 70-mm module points at the center of the 28-mm frame.
