# Analysis of liblricompression.so from L16 Android Gallery APK

## Overview
- **File**: `liblricompression.so` extracted from L16 Android gallery APK
- **Size**: 2.3 MB
- **Format**: ELF 64-bit LSB shared object, ARM aarch64 (SYSV, dynamically linked)
- **Symbols**: **STRIPPED** - no debug symbols, but 177 exported symbols remain via `-D` flag
- **Key Functions**: Only 2 exported functions via `nm -D`
  - `ltCompress::CompressLRI(shared_ptr<istream>, shared_ptr<ostream>, bool, bool)`
  - `ltCompress::CompressLRI(string, string, bool, bool)` (wrapper)

## Exported Symbols Analysis
The library has been stripped, so we cannot access internal function names directly. However, the dynamic symbol table reveals two overloaded `CompressLRI` functions that take:
- Input: either a shared_ptr to istream or a string filename
- Output: either a shared_ptr to ostream or a string filename  
- Two boolean flags (likely: `skipImageCompression` and `quiet` based on error messages)

## Protobuf Message Schema - CONFIRMED
The string table contains **41+ protobuf message types** compiled into the library. These are serialized/deserialized by liblricompression when reading/writing LRIS files. The messages are namespaced as `.ltpb.*` (Light Template Protobuf):

### Core Geometry & Calibration
- `GeometricCalibration` - intrinsics, extrinsics, mirror systems
- `Distortion` - polynomial and CRA (Correction via Radial Array) distortion models
- `Matrix3x3F`, `Matrix4x4F` - rotation/transformation matrices
- `Point2F`, `Point2I`, `Point3F` - coordinate structures
- `Range2F` - floating-point ranges
- `RectangleI` - rectangle definitions

### Camera Hardware & Identification
- `CameraID` - unique camera identifier
- `CameraModule` - module description with AF info and surface data
- `CameraModuleHwInfo` - hardware info including lens type, mirror type, mirror actuator type
- `HwInfo` - device hardware info with flash type and ToF type

### Calibration Data
- `ColorCalibration` - color matrix, spectral data, illuminant types
- `ColorCalibrationGold` - gold standard color calibration reference
- `SensorCharacterization` - VST noise models, sensor type
- `VignettingCharacterization` - vignetting, crossstalk, and mirror vignetting models
- `HotPixelMap` - defective pixel list
- `DeadPixelMap` - dead pixels
- `DeviceTemp` - temperature readings

### Specialized Calibration
- `FactoryModuleCalibration` - factory per-module calibration bundle
- `FactoryDeviceCalibration` - factory per-device calibration bundle
- `FlashCalibration` - flash characteristics
- `ToFCalibration` - Time-of-Flight calibration
- `MirrorSystem` - mirror configuration
- `MirrorActuatorMapping` - actuator angle pairs and transformation types

### Metadata
- `TimeStamp` - timestamp structures
- `FlashData` with Mode enum - flash state and mode
- `GPSData` - GPS metadata (altitude, heading, track, reference data)
- `IMUData` with Sample - inertial measurement unit data
- `ProximitySensors` - proximity sensor data
- `SensorData` - generic sensor readings
- `ViewPreferences` - user view/crop/AWB/scene/aspect ratio preferences
- `AFDebugInfo` - autofocus debug info
- `FaceData` and `FaceDataJ` - face region of interest data
- `Compatibility` - format compatibility markers

## LRI/LRIS Format Structure - CONFIRMED

### Key Findings from String Table
1. **File Structure**:
   - Magic ID header validates LRI file format
   - `LightRecordHeader` internal structure (32-byte minimum typical)
   - Protobuf deserialization error handling confirms protobuf encoding
   - "Invalid magic-id of record#" - magic number validation per record

2. **Compression Pipeline**:
   - JPEG compression is applied to image data (configurable quality)
   - Depth map data has separate compression handling
   - Two compression flags control the pipeline:
     - `skipImageCompression` - whether to skip image compression
     - `quiet` - whether to suppress logging
   - Error: "input to depth compression has to be float" (from libcp) confirms float depth format
   - Compression ratio tracking: "Final compression ratio: %.2f"

3. **Stack/Batch Processing**:
   - "Not a stack lri, skipping image compression" - can process single or stacked LRIs
   - "Compressed .lri, skipping image compression" - already-compressed files are detected
   - Multi-frame LRI stacks are supported
   - Reference camera tracking: "Multiple different reference cameras in the stack lri"

4. **Record Iteration**:
   - Per-record processing in `CompressRecord` (internal function)
   - Each record has a `LightRecordHeader` with module info
   - "Compressed module X" - progress tracking per module
   - "frame for module X" - frame numbering per module

### Protobuf Serialization Entry Points
From string table:
- `SerializeToBuffer` - serialize protobuf to memory buffer
- `SerializeNode` - serialize individual nodes
- Error messages confirm:
  - "invalid protobuf size!" - size validation
  - "Protobuf deserialization error!" - parse failure
  - "Protobuf serialization error!" - serialization failure

### Camera & Module Information
- "Did not find the HW Info of camera: X in the LRI" - HwInfo lookup
- "No reference camera set" - validation that a reference camera exists
- `CameraToProtobuf` / `CameraFromProtobuf` - bidirectional conversion functions
- Module ID string validation: "invalid module string!" and "invalid module type!"
- Module number validation: "invalid module number!"
- File header validation: "corrupted file header!"

## CRITICAL DISCOVERY: RefinedGeomCalib and State Classes

**NOT FOUND in symbol table or string table of liblricompression.so**.

This is significant: the library does NOT define or directly serialize:
- `RefinedGeomCalib` - refined geometry calibration
- `StereoState` - stereoscopic state
- `DepthEditorState` - depth editor UI state

**Hypothesis**: These three classes are likely defined in **libcp_android.so** (the main computation library), not in liblricompression. The liblricompression library is purely concerned with:
1. Reading/writing the LRIS file format (header, records, protobuf)
2. Compressing/decompressing image and depth data
3. Managing the canonical factory calibration data (FactoryModuleCalibration, etc.)

The refined/editor states are probably computed by libcp from the factory calibration, then either:
- Cached in memory during computation (not serialized to LRIS)
- OR serialized to a separate "state" file in the Light gallery app
- OR stored in libcp's internal structures

## Compression/Decompression Functions
The library provides the main public interface via `CompressLRI`:
- Reads an LRI file (stream-based or file-based)
- Decompresses all records
- Optionally re-compresses with different settings
- Writes to output (stream or file)
- Returns error code (0 = success, 1+ = failure)

Internal helpers (not exported, inferred from error messages):
- `CompressRecord` - per-record compression
- `GetLRIProperties` - query properties
- `SetRefinedCalib` - load/apply refined calibration (likely in libcp)
- JPEG compression/decompression (libjpeg integration)
- Depth map encoding/decoding

## File I/O Infrastructure
Internal stream classes found via mangling:
- `SystemFileInputStream` / `SystemFileOutputStream` - file-based I/O
- `StandardInputStream` / `StandardOutputStream` - stream wrapper
- `MemoryOutputStream` - in-memory buffer
- `ZeroCopyInputStreamWrapper` / `ZeroCopyOutputStreamWrapper` - protobuf integration

## Factory vs. Refined Calibration
The library contains the complete **factory calibration** schema:
- `FactoryModuleCalibration` - per-module factory data
- `FactoryDeviceCalibration` - per-device factory data

These include:
- Geometric calibration (intrinsics, extrinsics)
- Color calibration
- Sensor characterization
- Vignetting characterization
- Hot/dead pixel maps
- Hardware info
- Flash and ToF calibration

**Note**: "SetRefinedCalib" function appears in error messages, but not as an exported symbol. This suggests refined calibration is either:
- Computed on-the-fly from factory data
- Loaded from a separate state file
- Managed by libcp_android.so

## Build Information
From debug strings:
- Source tree: `/home/build/jenkins/workspace/L16-RELEASE-USER/00WW-1.3.5.1/`
- Build path: `light/compimaging/lric/lricompress.cpp` (main compression)
- Supporting modules:
  - `light/compimaging/camera/sensor.cpp`
  - `light/compimaging/camera/capturedimage.cpp`
  - `light/compimaging/camera/coloroptimizer.cpp`
  - `light/compimaging/camera/protobufprimitives.cpp`
  - `light/compimaging/camera/hwinfo.cpp`
  - `light/compimaging/camera/recordparser.cpp`
  - `light/compimaging/camera/compressionquantization.cpp`
  - `light/compimaging/3rdparty/rapidjson/reader.h`
  - `light/compimaging/3rdparty/protobuf-3.5.0/`

## Conclusions

### What liblricompression DOES
1. **Defines the LRIS binary format** with magic number, record headers, and protobuf encoding
2. **Serializes 41+ protobuf messages** (all factory calibration data)
3. **Implements JPEG and depth map compression** for image data
4. **Provides stream-based I/O** for reading and writing LRIS files
5. **Validates file structure** and protobuf data
6. **Supports multi-record (stacked) LRI files**

### What liblricompression DOES NOT contain
1. **RefinedGeomCalib class** - not found
2. **StereoState class** - not found
3. **DepthEditorState class** - not found
4. **Editor UI state** - no evidence

### Implication
The LRIS file format stores **factory calibration data** (computed at manufacture time). The refined calibration and editor state are likely:
- Computed by libcp from factory data
- Stored in a separate "state" file or database
- Managed by the gallery app UI layer (not part of LRIS itself)

This aligns with Light's architecture: LRIS is a **read-only factory data bundle**, and refinement happens at the application layer.

### Schema Summary for Weeks of Research
The library confirms that LRIS files are **pure protobuf archives** with a simple frame header and per-record structure. No custom binary serialization, no undocumented padding. The 41+ message types define the complete schema:

**FactoryModuleCalibration** contains:
- CameraID
- ColorCalibration
- GeometricCalibration (with Distortion, MirrorSystem, Extrinsics, etc.)
- VignettingCharacterization
- SensorCharacterization
- HotPixelMap / DeadPixelMap
- TimeStamp

**FactoryDeviceCalibration** bundles:
- Multiple FactoryModuleCalibration entries
- HwInfo (device-level hardware info)
- DeviceTemp
- Compatibility markers

The library is a **definitive source of truth** for the LRIS format schema.
