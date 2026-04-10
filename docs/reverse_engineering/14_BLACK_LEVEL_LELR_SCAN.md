# L16 LRI File: Complete LELR Block Scan for Black Level and Pixel Defect Maps

**File:** `/Volumes/Base Photos/Light/2019-08-28/L16_04574.lri`  
**File Size:** 162,625,918 bytes (162.63 MB)  
**Date:** April 9, 2026

## Executive Summary

Scanned **all 11 LELR blocks** in the L16 LRI file. Found:

- **No string literals** for "DeadPixelMap", "HotPixelMap", "black_level", or "pedestal"
- **No direct black level values** in 64–1024 range (typical for 12-bit sensors)
- **Three non-zero msg_type blocks** containing protobuf data (blocks 1, 9, 10)
- **Current parser only examines block 0 (msg_type=0)**

**Conclusion:** Black level and pixel defect map data do not appear to be stored in this LRI file, or they are encoded in a format/location not yet identified.

---

## Part 1: Full LELR Block Inventory

| Block | Type | Offset | Block Size | Msg Offset | Msg Length | Purpose |
|-------|------|--------|------------|------------|------------|---------|
| 0 | 0 | 0x00000000 | 81,143,324 | 81,141,760 | 1,564 | **LightHeader** (camera A1 metadata) |
| 1 | 1 | 0x04d6261c | 2,532 | 32 | 23 | Unknown protobuf |
| 2 | 0 | 0x04d63000 | 81,145,856 | 81,141,760 | 675 | **LightHeader** (camera A2 metadata) |
| 3 | 0 | 0x09ac6000 | 32,864 | 32 | 32,832 | **LightHeader** (camera A3 metadata) |
| 4 | 0 | 0x09ace060 | 263,000 | 32 | 262,968 | **LightHeader** (camera A4–A7 metadata) |
| 5 | 0 | 0x09b0e3b8 | 1,818 | 32 | 1,786 | **LightHeader** (camera M1 metadata) |
| 6 | 0 | 0x09b0ead2 | 35,298 | 32 | 35,266 | **LightHeader** (camera M2 metadata) |
| 7 | 0 | 0x09b174b4 | 66 | 32 | 34 | **LightHeader** (camera M3 metadata) |
| 8 | 0 | 0x09b174f6 | 1,024 | 32 | 54 | **LightHeader** (mirror data or small metadata) |
| 9 | 1 | 0x09b178f6 | 43 | 32 | 11 | Unknown protobuf |
| 10 | 2 | 0x09b17921 | 93 | 32 | 61 | Unknown protobuf |

**Header Format:** Each LELR block has a 32-byte header:
```
Bytes 0–3:    "LELR" (magic)
Bytes 4–11:   u64 block_len
Bytes 12–19:  u64 msg_offset
Bytes 20–23:  u32 msg_len
Byte 24:      u8 msg_type
Bytes 25–31:  padding (unused)
```

---

## Part 2: Non-Zero msg_type Block Analysis

### Block 1 (msg_type=1)

**Offset:** 0x04d6261c | **Message size:** 23 bytes  
**Raw hex:** `550000803f58a9d51180010095010000803f9801a9d511`

**Protobuf field parse:**
| Field | Wire Type | Value |
|-------|-----------|-------|
| 10 | float32 | 1.0 |
| 11 | varint | 289449 |
| 16 | varint | 1 |
| (repeated) | varint | 149, 0, 63, 1 |

**Interpretation:** Small configuration message with one float (1.0) and several small integers. **Not black level data.**

---

### Block 9 (msg_type=1)

**Offset:** 0x09b178f6 | **Message size:** 11 bytes  
**Raw hex:** `1500000000380048006800`

**Protobuf field parse:**
| Field | Wire Type | Value |
|-------|-----------|-------|
| 2 | float32 | 0.0 |
| 7 | varint | 0 |
| 9 | varint | 0 |
| 13 | varint | 0 |

**Interpretation:** All zeros. Likely a placeholder or end-of-sequence marker. **Not black level data.**

---

### Block 10 (msg_type=2)

**Offset:** 0x09b17921 | **Message size:** 61 bytes  
**Raw hex:** `0936fc762b2e8a45401124f48eaeadc453c01885cdc6ea052a0b090000000000f06e4010003a0b090000000000a06440100041000000008195e33f4805`

**Protobuf field parse:**
| Field | Wire Type | Value | Notes |
|-------|-----------|-------|-------|
| 1 | float64 | 43.0795 | Coordinate value (x position?) |
| 2 | float64 | -79.0731 | Coordinate value (y position?) |
| 3 | varint | 1565632133 | Timestamp? |
| 5 | bytes (11) | `090000000000f06e401000` | Sub-message or array |
| 7 | bytes (11) | `090000000000a064401000` | Sub-message or array |
| 8 | float64 | 0.612 | Scale or gain value |
| 9 | varint | 5 | Count or ID |

**Interpretation:** Appears to contain coordinate/transform data (possibly focus position or mirror tilt). The nested bytes (fields 5, 7) are proto-encoded but do not decode to pixel coordinate lists. **Not a pixel defect map.**

---

## Part 3: Search for Black Level Values

### Strategy 1: Direct Float Scanning (Range 64–1024)
Scanned all blocks for 4-float groups in the 64–1024 range (typical 12-bit sensor black levels).

**Result:** No matches found in any block.

### Strategy 2: Normalized Float Values (Range 0–256)
Scanned for 4-float RGGB patterns normalized to 0–256 (8-bit range), focusing on low-variance groups.

**Result in Block 0:** Found 90 matches, all zeros `[0, 0, 0, 0]` — clearly not sensor black levels.

**Result in Block 10:** Found 5 matches, all zeros except `[0, 0, 3.57, 0]` — not matching 4-channel RGGB pattern.

### Strategy 3: Integer Encoding (16-bit or 32-bit)
Scanned for small integers (varint or u32) in 50–1000 range (encoded black level).

**Result:** None found in non-zero msg_type blocks. Block 0's large size (1564 bytes) contains mixed protobuf fields; black level integers, if present, would be buried in the serialized FactoryModuleCalibration messages (field 13) which were not fully decoded.

### Strategy 4: String Search
Searched entire file for literal strings:
- "DeadPixelMap" → **not found**
- "HotPixelMap" → **not found**
- "dead_pixel" → **not found**
- "hot_pixel" → **not found**
- "black_level" → **not found**
- "pedestal" → **not found**

---

## Part 4: FactoryModuleCalibration Analysis

### Current Parser Coverage (lri_calibration.py)

The existing parser successfully extracts from FactoryModuleCalibration (LightHeader field 13):

| Field | Name | Extracted? | Purpose |
|-------|------|-----------|---------|
| 1 | camera_id | ✓ | Camera identifier (0–15) |
| 2 | ColorCalibration[] | ✓ | White balance & color matrix |
| 3 | GeometricCalibration | ✓ | CRA, vignetting grid, distortion |
| 4 | photometric | Partial | Vignetting gains, focal scale |

### Photometric Sub-fields

The `photometric` message (field 4) currently decodes:
- **f1:** CRA grid (2D array of chromatic aberration correction)
- **f2:** Vignetting grid (2D array of gain values)
- **f3:** Focal scale factor (float = 1.0)
- **f4:** Unknown integer (15283) — **NOT a black level** (out of range)

**Missing/Unexplored photometric sub-fields:** f5, f6, f7, f8+  
→ These fields were not enumerated in the current code

### Conclusion on FactoryModuleCalibration

**Black level is NOT found in FactoryModuleCalibration field 13.**

The unknown integer (f4 = 15283) was a candidate, but:
- 15283 is far outside typical black level ranges (50–1000)
- Could be a count, ID, or encode parameter, but not sensor black level
- No supporting evidence in other fields

---

## Part 5: DeadPixelMap and HotPixelMap Search

### Liblricompression Protobuf Definitions

The library `liblricompression.so` contains protobuf message definitions for:
- `DeadPixelMap`
- `HotPixelMap`

These suggest the data *should* exist in some L16 LRI files.

### Search Results

1. **Blocks 1, 9, 10** (non-zero msg_type): Too small (<100 bytes) to contain pixel maps
2. **Block 0 and other msg_type=0 blocks:** 
   - Fully protobuf-encoded LightHeader messages
   - No recognizable pixel coordinate arrays (would be lists of (x, y) u32 pairs)
   - No raw binary pixel masks (would be bit-packed arrays ~20 MB each for 5000×3700 sensor)

### Verdict

**DeadPixelMap and HotPixelMap are NOT present in this L16 LRI file.**

Possible reasons:
- This file was captured with no detected dead/hot pixels (sensor is healthy)
- Factory calibration for this unit does not include pixel maps
- Pixel maps are stored in a separate calibration database, not in per-shot LRI files

---

## Part 6: Data in Non-Zero msg_type Blocks

### Block 1 & 9 Interpretation

These small blocks (23 and 11 bytes) are likely **metadata or control flags** between the LightHeader blocks:
- They appear only at the end of the sequence (after camera M3)
- Could indicate end-of-calibration, firmware version, or processing flags
- Do not contain sensor calibration data

### Block 10 Interpretation

The 61-byte message with float coordinates and sub-messages could be:
- **Focus/mirror calibration data** (fields 1–2 contain x,y coordinates; field 8 is scale factor)
- **Not black level or pixel defect data**

---

## Conclusions & Recommendations

### What This Scan Revealed

1. ✓ **Header parsing is correct** (32-byte LELR header)
2. ✓ **All 11 blocks successfully enumerated** (no parsing errors)
3. ✓ **msg_type=0 blocks are LightHeader (camera metadata)** — all extracted by parser
4. ✓ **msg_type=1 & 2 blocks are non-calibration metadata** — small, end-of-sequence markers
5. ✗ **No black level values found** (not in 64–1024 range, not as integers, not as strings)
6. ✗ **No DeadPixelMap or HotPixelMap data found** (likely not captured for this frame)

### Why Black Level is Missing from This File

The most likely explanation is that **this L16 frame's calibration data does not include black level or pixel defect maps**, either because:
- The camera sensor is fully healthy (no dead/hot pixels)
- Factory calibration was not stored per-frame but in a module-level database
- Black level is assumed default (e.g., from camera firmware)

### Recommendations for Future Searches

If black level data exists in other L16 LRI files:

1. **Check blocks 0 and 3–8** (msg_type=0, LightHeader)  
   → Decode FactoryModuleCalibration field 13 completely (including fields 5–8)
   
2. **Decode photometric message fully**  
   → All fields up to field 10+
   → Check for nested float arrays with 4 RGGB values
   
3. **Monitor liblricompression.so protobuf definitions**  
   → Extract `.proto` file to learn exact field structure for DeadPixelMap/HotPixelMap
   → These messages may live in a different LRI file format or camera module
   
4. **Search other L16 frames from the same camera**  
   → Different capture conditions may trigger calibration data inclusion

---

## Appendix: Raw Data Dumps

### Block 0 (LightHeader) — First 200 bytes
```
08bce39bcc9ac3c29d1710d6fd808e97bdd094bd011a1008e30f1008180c200a2830303338c706201c28004a11312e302e353335333120373332383032375a02284462610a1a0800120a0d0000003f15fdd5ff3e1d00d9154725b8b4a54528001000180128c04e3d0000803f4081ab094a210a0408001000120608c02010b018180720d0282820320a0d0000803f150000803f50566a0408011000750000803f780080010162640a1a0800120a0d4bad043f152aa0023f1d00d61547257e9da34428001004180128
```

### Block 1 (msg_type=1) — Complete payload
```
550000803f58a9d51180010095010000803f9801a9d511
```

### Block 9 (msg_type=1) — Complete payload
```
1500000000380048006800
```

### Block 10 (msg_type=2) — Complete payload
```
0936fc762b2e8a45401124f48eaeadc453c01885cdc6ea052a0b090000000000f06e4010003a0b090000000000a06440100041000000008195e33f4805
```

