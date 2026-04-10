# LRIS File Format

The `.lris` file is Lumen's **sidecar/cache format**. It is **NOT written by the L16 camera** — only by Lumen desktop (`CIAPI::StateFileEditor::serialize` in libcp.dylib). The on-device gallery app writes `.state` files with user edits only, not full LRIS sidecars.

When Lumen renders an LRI for the first time, it runs the full pipeline (Ceres bundle adjustment + stereo depth + etc.) and writes the results as an LRIS sibling. On subsequent opens of the same LRI, Lumen detects the LRIS and loads the pre-computed refined calibration + depth, skipping the expensive stages.

## Why LRIS matters for us

- **Not as runtime data** — the replacement app has to work on fresh LRIs without any LRIS, since a new capture will never have one.
- **As an oracle** — for every LRI that has a matching LRIS in our archive (5,466 pairs), we have Lumen's own "correct answer" for refined calibration + depth. Perfect ground truth for validating our pipeline stages.
- **As a cache design reference** — if we later want our own replacement to skip re-computation on re-opens, we can write something structurally similar.

## Top-level structure

```
┌── 32-byte header ────┐
│  magic + offsets     │
├── depth map ─────────┤
│  ~200 KB int32 LE    │   260×195 quantized depth
├── 82-byte protobuf ──┤   Single-camera calibration (not full RefinedGeomCalib)
├── ~6.5 MB unknown ───┤   Likely repeated per-camera calibrations
├── trailer protobuf ──┤   DepthEditorState (small, ~30 bytes)
└──────────────────────┘
```

Total size: ~6.7 MB for a typical LRIS (vs ~85 MB for its matching LRI).

## Header (32 bytes)

| Offset | Size | Type | Value (L16_05036.lris) | Meaning |
|---|---|---|---|---|
| 0 | 4 | u32 LE | `0x12345678` (`78 56 34 12`) | Magic number |
| 4 | 8 | u64 LE | `202,922` | Total proto end: 202,840 + 82 |
| 12 | 8 | u64 LE | `202,840` (`0x31858`) | Offset of main protobuf |
| 20 | 4 | u32 LE | `82` | Size of main protobuf |
| 24 | 8 | u64 LE | `0x00007fff6832f76b` | **Looks like a leaked writer pointer** — may be accidental, or part of another structure |
| 32 | 4 | u32 LE | `260` | Depth map width (in grid cells) |
| 36 | 4 | u32 LE | `195` | Depth map height (in grid cells) |

## Depth/disparity map (offset 0x28 → 0x31858)

- **Size**: 202,800 bytes
- **Format**: `int32 LE` array, 260 × 195 cells (fits byte count: 260 × 195 × 4 = 202,800)
- **Values**: mostly negative integers around `-7237` (i.e., `0xffffe3bb`) with small variations
- **Interpretation**: Quantized disparity or fixed-point inverse-depth. Negative values may mean invalid/occluded (standard stereo-vision convention).
- **Resolution**: 260 × 195 ≈ 50,700 depth samples. That's FAR less than image pixels (~50 MP). This is a coarse depth grid, likely used as input to a per-pixel upsample stage inside the renderer.

## Main protobuf at offset 0x31858 (82 bytes)

Raw bytes:
```
0a 06 08 00 10 1a 18 03 1a 48
0d 00 00 00 00 15 d1 db 98 45 1d bb d4 8d c0
25 00 00 00 00 2d 00 00 00 00 35 00 00 00 00
3d 00 00 00 00 45 00 00 00 00 4d 00 00 00 00
55 00 00 00 00 5d 00 00 00 00
62 0a 0d 43 be 73 41 15 00 00 80 bf 6d 00 00 00 00
```

Decoded protobuf tree:
- **f1**: length-delimited, 6 bytes = nested `{f1=0, f2=26, f3=3}` — likely camera ID + mode flags
- **f3**: length-delimited, 72 bytes = nested CalibrationData
  - 11 flat `fixed32` floats: `{0.0, 4891.477, −4.432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}` — first two non-zero suggest focal length and a distortion coefficient
  - 1 nested message `f12 = {f1: 15.234, f2: -1.0}` — possibly pose angles
  - 1 trailing zero float

**IMPORTANT**: This does NOT match the inferred `RefinedGeomCalib { module_id, k_mat, r_mat, t_vec }` schema that Agent 3's string-table heuristic suggested. The actual structure is flatter and includes what looks like a single focal length + distortion coefficient + pose angles — more like `Distortion` than `RefinedGeomCalib`. The schema reconstruction needs to be re-done from actual wire evidence, not string ordering.

## Unknown middle blob (offset 0x31894 → 0x6b791c)

- **Size**: ~6.5 MB (6,840,456 bytes)
- **Magic detection**: no JPEG (0xFFD8), no PNG (0x89504E47)
- **Hypotheses**:
  - Repeated per-camera calibrations (10 cameras × ~650 KB each — doesn't divide cleanly)
  - Compressed depth tiles (LZ4, zlib) — opaque binary
  - Full-resolution depth map at higher precision than the 260×195 grid
  - Depth editor state (tile data for user edits)
  - JPEG thumbnail embedded somewhere (no magic though)

Needs dedicated decoder pass. Not critical-path for the "replace Lumen" product since LRIS is write-only for oracle purposes.

## Trailer protobuf (~0x6b791c, ~30 bytes)

```
0a 14 08 9a cf 84 cb f9 e9 bd 80 54 10 da ec ed c3 ad bd d0 a3 11 10 00
```

Decoded:
- **f1**: length-delimited, 17 bytes = nested metadata (large varint values)
- **f2**: varint = 0 (flag/terminator)

Looks like `DepthEditorState` or similar lightweight metadata — possibly timestamps, edit history, or version.

## Writer function

- **`CIAPI::StateFileEditor::serialize(shared_ptr<ostream>)`**
  - macOS libcp.dylib: `0x39bc40`
  - Android libcp.so: `0x3a2140`
- Only high-level API for writing refined state. Hooking this function via lldb/frida would capture the complete refined-state contents at the moment Lumen writes them.

## Auto-detection at read time

`lri_process` (and Lumen proper) automatically reads an LRIS sibling when one exists next to an LRI, printing:
```
LRIS auto-detected: /path/to/file.lris
LRIS state loaded.
```
Observed on L16_05036.lri → L16_05036.lris in `/Volumes/Base Photos/Light/2019-11-27/`.

When no LRIS exists, the full pipeline runs cold-start (observed on L16_04574 which has no LRIS sibling — Round 1 lldb trace captured 3 Problem instances and 183 AddParameterBlock calls, the cold-start signature).

## Related: on-device `.state` files

The L16's on-device gallery app writes `.state` files via `LibCpRenderer.nativeSaveState(path)`. These contain user adjustments only (exposure, DOF focus depth, F-number, color temperature, tint, transforms, crop, user rating). They are **NOT full LRIS sidecars** — no refined calibration, no depth map.
