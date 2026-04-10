# LRI File Format

The `.lri` file is Light's raw multi-camera capture format. It's a sequence of **LELR blocks**, each containing a 32-byte header and a protobuf payload.

## Top-level structure

```
┌── LELR block 1 ──────┐
│  32-byte header      │
│  protobuf payload    │  → LightHeader, SensorData, or other
└──────────────────────┘
┌── LELR block 2 ──────┐
│  32-byte header      │
│  protobuf payload    │
└──────────────────────┘
  ... (many blocks)
```

Typical L16_04574.lri (85 MB) has **11 LELR blocks**, of which **8 are LightHeader** (`msg_type=0`). Some blocks are small (~33 KB calibration blocks), some huge (~85 MB containing image frames).

## LELR block header (32 bytes)

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | bytes | Magic `LELR` (0x4C 0x45 0x4C 0x52) |
| 4 | 8 | uint64 LE | `block_len` — total block size (header + payload) |
| 12 | 8 | uint64 LE | `msg_offset` — offset from block start to protobuf payload |
| 20 | 4 | uint32 LE | `msg_len` — protobuf payload length in bytes |
| 24 | 1 | uint8 | `msg_type` — 0 = LightHeader, others = image/calibration blocks |
| 25 | 7 | bytes | Padding/reserved |

## LightHeader protobuf (msg_type = 0)

The LightHeader is the metadata + calibration container. The full field enumeration (as of 2026-04-09):

| Field | Wire | Count | Size | Parsed? | Semantic |
|---|---|---|---|---|---|
| 1 | varint | 2 | 8 | no | Timestamp-like (u64) |
| 2 | varint | 2 | 8 | no | UUID/session-like (u64) |
| 3 | len-delim | 2 | 42 | no | Hash-like blobs (16B each) |
| **4** | varint | 2 | 8 | **YES** | `image_focal_length_mm` (u64) |
| 5 | varint | 1 | 4 | no | Small flag |
| 6 | varint | 1 | 4 | no | Small int |
| 7 | varint | 1 | 4 | no | Small int |
| 8 | len-delim | 1 | 8 | no | Tiny blob |
| 9 | len-delim | 2 | 44 | no | Medium blob |
| 11 | len-delim | 1 | 7 | no | Tiny blob |
| **12** | len-delim | 10 | 1096 | **YES** | `CameraModule[]` — per-camera runtime metadata |
| **13** | len-delim | 74 | 331166 | **YES** | `FactoryModuleCalibration[]` — per-camera factory data |
| 14 | len-delim | 1 | 37 | no | Small blob |
| **16** | len-delim | 1 | 1782 | **PARTIAL** | LUT: 32 entries indexed 100..775 step 25, 8 floats each — semantic unknown |
| 17 | 32-bit | 1 | 8 | no | Single float |
| 18 | len-delim | 3 | 133 | no | Likely `IMUData` (timestamp + accel + gyro) |
| **19** | len-delim | 1 | 56 | **YES** | `ViewPreferences` — contains AWB gains |
| 23 | len-delim | 1 | 885 | no | Starts with proto2 group tag (wire type 3); possibly per-focus black level LUT (unverified) |
| 24 | len-delim | 2 | 22 | no | GPS-like (lat/lon?) |
| 26 | len-delim | 2 | 24 | no | GPS-like (altitude/heading?) |
| 27 | len-delim | 1 | 9 | no | Small |

### Field 12 — CameraModule

Per-camera runtime metadata for the fired cameras. Fields:
- `f2`: camera_id (0..15, A1=0, B1=5, C1=10)
- `f4`: `mirror_position_hall` (for MOVABLE cameras — feeds `hall_code_to_mirror_angle`)
- `f5`: `lens_position_hall` (actual lens focus position at capture time)
- `f7`: `analog_gain` (float)
- `f8`: `exposure_ns` (uint64)
- `f9`: `sensor_surface { w, h, format, bayer }`
- `f11`: `flip_h` (bool)
- `f12`: `flip_v` (bool)

### Field 13 — FactoryModuleCalibration

Per-camera factory data. One entry per calibrated focus/exposure point (74 total entries across 16 cameras = multiple calibration bundles per camera). Structure:

- `f1`: `camera_id`
- `f3`: `GeometricCalibration`
  - `f1`: `mirror_type` (0=NONE, 1=GLUED, 2=MOVABLE)
  - `f2`: `CalibrationFocusBundle[]` — per-focus calibration bundles
    - `f1`: `focus_distance` (float, metres, e.g. 0.818, 1.500)
    - `f2`: `Intrinsics { f1: Matrix3x3F k_mat }` (fx, fy, cx, cy in pixels)
    - `f3`: `Extrinsics`
      - `f1`: `canonical { f1: rotation R, f2: translation t }` — for NONE/GLUED cameras
      - `f2`: `movable_mirror { f1: MirrorSystem, f2: MirrorActuatorMapping }` — for MOVABLE
    - `f6`: `focus_hall_code` (float, optional — correlates with lens_position)
- `f4`: photometric calibration
  - `f1`: CRA grid — 17×13 cells of 4×4 affine matrices (14,144 B blob) — **extracted but NOT applied by our code**
  - `f2`: vignetting — 17×13 flat-gain grid (884 B blob) — **applied**
  - `f3`: focal scale factor (float, meaning unclear)
  - `f4`: unknown int
- `f2`: `ColorCalibration[]` — 3 modes per camera (mode values 0, 2, 6 — illuminants unknown)
  - `f1`: mode (uint64: 0, 2, or 6)
  - `f2`: forward CCM (Matrix3x3F) — **extracted but NOT applied**
  - `f3`: inverse CCM (Matrix3x3F)
  - `f4, f5`: illuminant chromaticity pair
  - `f8`: spectral data
    - `f2`: 3 curves (R, G, B)
      - `f1`: wavelength start (380)
      - `f2`: wavelength end (755)
      - `f3`: 76 float32 samples (5nm step)
- `f7`: calibration date

### Field 19 — ViewPreferences

Contains the AWB (auto white balance) gains applied for the capture:
- `f15`: `ChannelGain`
  - `f1`: R gain (float)
  - `f2`: Gr gain (float)
  - `f3`: Gb gain (float)
  - `f4`: B gain (float)

## MovableMirrorFormat submessages

### MirrorSystem
- `f1`: `real_camera_location` (Point3F, mm, A1 = origin)
- `f2`: `real_camera_orientation` (Matrix3x3F, world→camera — **NOT yet used**)
- `f3`: `rotation_axis` (Point3F, unit vector of the actuator axis)
- `f4`: `point_on_rotation_axis` (Point3F, mm)
- `f5`: `distance_mirror_plane_to_point_on_axis` (float, mm)
- `f6`: `mirror_normal_at_zero_degrees` (Point3F, unit vector) — **empirically not the true mirror normal**; the actual virtual camera forward direction is computed by rotating (0,0,1) around `rotation_axis` by the mirror angle
- `f7`: `flip_img_around_x` (bool)

### MirrorActuatorMapping
- `f2`: `actuator_length_offset` (float)
- `f3`: `actuator_length_scale` (float)
- `f4`: `mirror_angle_offset` (float, degrees)
- `f5`: `mirror_angle_scale` (float, degrees per normalized unit)

Formula (MEAN_STD_NORMALIZE):
```
normalized = (hall_code - actuator_length_offset) / actuator_length_scale
angle_deg  = mirror_angle_offset + normalized * mirror_angle_scale
```

### Virtual camera forward direction (for MOVABLE cameras)

**Empirically correct formula** (confirmed on L16_04574, 2026-04-09):
```
forward = rodrigues((0, 0, 1), rotation_axis, angle_deg)
```
This gives ~37° off-axis for MOVABLE B cameras, matching the 4 corners of the 28mm A FOV (IEEE Spectrum article confirms: "mirrors ... point at each of the four quadrants").

The intuitive approach of using `-n` (negative mirror normal) gives ~53°, which is wrong by approximately the complement (cos↔sin).

## What we're still missing in the LRI

- **Semantic meaning of field 16** — 32-entry LUT indexed 100..775 step 25, 8 floats each. Could be: per-focus distortion coefficients, per-temperature CCM interpolation, tone curve, spectral response. Unknown.
- **Field 23 decoded contents** — starts with proto2 group tag; may contain per-focus black level correction LUT (unverified).
- **Black level values location** — not found in an unambiguous place. May be in field 23, or within `FactoryModuleCalibration` photometric sub-field `f4` (unknown int), or hardcoded per sensor model.
- **Dead pixel map** — referenced by class name in libcp (`ltpb.DeadPixelMap`) but not located in LRI.
- **Hot pixel map** — referenced by class name in libcp (`ltpb.HotPixelMap`) but not located in LRI.
- **Cross-talk correction** — referenced in pipeline order, no data located.
- **Polynomial distortion** — `ltpb.Distortion.Polynomial` protobuf exists in libcp with schema (distortion_center, normalization, coeffs, fit_cost, valid_roi) but not located in LRI.

## Reference code

- Current parser: `/Users/ryaker/Documents/Light_Work/lri_calibration.py`
  - `parse_lri(path)` — walks LELR blocks, parses LightHeader fields 4/12/13/19
  - `extract_sensor_calibration(path)` — extracts vignetting + CCMs + AWB + spectral response
  - `ProtoReader` class — minimal protobuf wire-format parser
  - `compute_movable_mirror_pose` — fixed 2026-04-09 to use V1 formula (rotate (0,0,1) by angle)

## Related format: LRI_FIELD_16

Field 16 structure (decoded but not yet interpreted):
```
message (field 16):
  f1: varint = 2
  f2: submessage:
    f1, f2, f3: fixed32 (3 leading floats)
    f4: repeated submessage × 32:
      f1: varint (100, 125, 150, 175, ..., 775 — step 25)
      f2: fixed32 (float)
      f3: fixed32 (float)
      f4..f7: 4 × {f1: fixed32, f2: fixed32}  (4 × Point2F-like pairs)
```

Each of the 32 entries holds 8 floats + one index. The indices walk linearly from 100 to 775 in steps of 25 — likely a discretized parameter (focus distance? wavelength? temperature?).
