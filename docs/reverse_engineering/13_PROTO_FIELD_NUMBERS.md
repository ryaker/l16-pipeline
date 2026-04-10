# Ground-Truth Protobuf Field Numbers (from openlight-camera smali)

Source: decompiled Square Wire bytecode in `/tmp/openlight-camera/light_camera/smali/`
Extraction method: smali constant analysis (Square Wire embeds field numbers as literal integers)

## ViewPreferences (LightHeader field 19)

| Field | Tag | Name | Type |
|---|---|---|---|
| 1 | 0x1 | `f_number` | float |
| 2 | 0x2 | `ev_offset` | float |
| 3 | 0x3 | `disable_cropping` | bool |
| 4 | 0x4 | `hdr_mode` | ViewPreferences.HDRMode enum |
| 5 | 0x5 | `view_preset` | ViewPreferences.ViewPresets enum |
| 6 | 0x6 | `scene_mode` | ViewPreferences.SceneMode enum |
| 7 | 0x7 | `awb_mode` | ViewPreferences.AWBMode enum |
| 9 | 0x9 | `orientation` | ViewPreferences.Orientation enum |
| 10 | 0xa | `image_gain` | float |
| 11 | 0xb | `image_exposure` | uint64 |
| 12 | 0xc | `user_rating` | uint32 |
| 13 | 0xd | `aspect_ratio` | ViewPreferences.AspectRatio enum |
| 14 | 0xe | `crop` | ViewPreferences.Crop message |
| **15** | **0xf** | **`awb_gains`** | **ViewPreferences.ChannelGain message ← our current parse** |
| 16 | 0x10 | `is_on_tripod` | bool |
| 17 | 0x11 | `qc_lux_index` | float |

### ViewPreferences.ChannelGain (field 15 submessage)
| Field | Name | Type |
|---|---|---|
| 1 | `r` | float |
| 2 | `g_r` | float |
| 3 | `g_b` | float |
| 4 | `b` | float |

### ViewPreferences.Crop (field 14 submessage)
| Field | Name | Type |
|---|---|---|
| 1 | `start` | Point2F |
| 2 | `size` | Point2F |

### ViewPreferences.AWBMode enum
| Value | Name |
|---|---|
| 0 | AUTO |
| 1 | DAYLIGHT |
| 2 | SHADE |
| 3 | CLOUDY |
| 4 | TUNGSTEN |
| 5 | FLUORESCENT |
| 6 | FLASH |
| 7 | CUSTOM |
| 8 | KELVIN |

**How to use**: AWBMode in field 7 tells us which CCM mode to use:
- AUTO → interpolate based on AWB gains
- DAYLIGHT → CCM mode 2 (D65)
- TUNGSTEN → CCM mode 0 (Illuminant A)
- FLUORESCENT → CCM mode 6 (F11/D50)
- SHADE/CLOUDY → CCM mode 2 or extrapolate higher CCT

### ViewPreferences.HDRMode enum
| Value | Name |
|---|---|
| 0 | NONE |
| 1 | DEFAULT |
| 2 | NATURAL |
| 3 | SURREAL |

### ViewPreferences.ViewPresets enum
| Value | Name |
|---|---|
| 0 | NONE |
| 1 | NATURAL |
| 2 | FAITHFUL |
| 3 | LANDSCAPE |
| 4 | PORTRAIT |

### ViewPreferences.SceneMode enum
| Value | Name |
|---|---|
| 0 | PORTRAIT |
| 1 | LANDSCAPE |
| 2 | SPORT |
| 3 | MACRO |
| 4 | NIGHT |
| 5 | NONE |

### ViewPreferences.Orientation enum
| Value | Name |
|---|---|
| 0 | NORMAL |
| 1 | ROT90_CW |
| 2 | ROT90_CCW |
| 3 | ROT90_CW_VFLIP |
| 4 | ROT90_CCW_VFLIP |
| 5 | VFLIP |
| 6 | HFLIP |
| 7 | ROT180 |

---

## GPSData (LightHeader fields 24 and/or 26)

| Field | Tag | Name | Type |
|---|---|---|---|
| 1 | 0x1 | `latitude` | double |
| 2 | 0x2 | `longitude` | double |
| 3 | 0x3 | `timestamp` | uint64 |
| 4 | 0x4 | `dop` | double (dilution of precision) |
| 5 | 0x5 | `track` | GPSData.Track message |
| 6 | 0x6 | `heading` | GPSData.Heading message |
| 7 | 0x7 | `altitude` | GPSData.Altitude message |
| 8 | 0x8 | `speed` | double |
| 9 | 0x9 | `processing_method` | GPSData.ProcessingMethod enum |

### GPSData.Track / GPSData.Heading (same structure)
| Field | Name | Type |
|---|---|---|
| 1 | `value` | double |
| 2 | `ref` | ReferenceNorth enum (MAGNETIC=0, TRUE=1) |

### GPSData.Altitude
| Field | Name | Type |
|---|---|---|
| 1 | `value` | double |
| 2 | `ref` | ReferenceAltitude enum |

### GPSData.ProcessingMethod enum
| Value | Name |
|---|---|
| 0 | UNKNOWN |
| 1 | GPS |
| 2 | CELLID |
| 3 | WLAN |
| 4 | MANUAL |
| 5 | FUSED |

---

## LightHeader sidecar type constants

From `LightHeader.smali`:
- `TYPE_VIEW_PREFS = 0x1` — sidecar LELR block type for ViewPreferences
- `TYPE_GPS_DATA = 0x2` — sidecar LELR block type for GPSData
- `MAGIC_ID = [0x4C, 0x45, 0x4C, 0x52]` = "LELR" (confirmed)
- `HEADER_LENGTH = 0x1c` = 28 bytes minimum (agent's calc; our parser uses 32 — may vary by firmware version)

---

## Point2F (used in Crop, and broadly across geometry)
| Field | Name | Type |
|---|---|---|
| 1 | `x` | float |
| 2 | `y` | float |

---

## Notes / Gaps

- Only ViewPreferences and GPSData classes were extracted from the smali. The CameraModule, FactoryModuleCalibration, GeometricCalibration, MirrorSystem, and MirrorActuatorMapping smali classes weren't enumerated in this pass.
- The `openlight-camera` repo has the capture-app protos; the gallery/libcp protos (which include RefinedGeomCalib, StereoState, DepthEditorState) are in libcp, not exposed in smali.
- CCM-related field numbers (ColorCalibration fields 4 and 5 = illuminant chromaticity f4/f5) were confirmed separately in `12_CCM_ILLUMINANTS.md`.
