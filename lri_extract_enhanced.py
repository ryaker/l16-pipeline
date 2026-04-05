#!/usr/bin/env python3
"""
Enhanced LRI extraction with capture metadata.

This script extends lri_extract.py to also capture and return metadata
that was previously ignored:
  - Focus distance (from af_info and focus hall codes)
  - Exposure time and analog gain (ISO-equivalent)
  - White balance (channel gains, AWB mode)
  - Focal length preset (image_focal_length in mm)
  - Reference camera ID
  - Device metadata and timestamp

The extended extract_modules() function now returns:
  (module_images: Dict, metadata: Dict)
"""

import struct
from typing import Dict, List, Optional, Tuple, Any


class ProtoReader:
    def __init__(self, data: bytes, offset: int = 0, length: int = -1):
        self.data   = data
        self.pos    = offset
        self.end    = offset + (length if length >= 0 else len(data) - offset)
        self.fields: Dict[int, List] = {}

    def _read_varint(self) -> int:
        result, shift = 0, 0
        while True:
            b = self.data[self.pos]; self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80): return result
            shift += 7

    def parse(self) -> 'ProtoReader':
        while self.pos < self.end:
            key       = self._read_varint()
            field_num = key >> 3
            wire_type = key & 0x7
            if wire_type == 0:
                val = self._read_varint()
            elif wire_type == 1:
                val = struct.unpack_from('<Q', self.data, self.pos)[0]; self.pos += 8
            elif wire_type == 2:
                length = self._read_varint()
                val    = self.data[self.pos : self.pos + length]; self.pos += length
            elif wire_type == 5:
                val = struct.unpack_from('<f', self.data, self.pos)[0]; self.pos += 4
            else:
                raise ValueError(f"Unknown wire type {wire_type}")
            self.fields.setdefault(field_num, []).append(val)
        return self

    def get_uint64(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field); return int(v[0]) if v else default

    def get_int64(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        u = int(v[0]); return u - (1 << 64) if u >= (1 << 63) else u

    def get_int32(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        u = int(v[0]) & 0xFFFFFFFF
        return u - (1 << 32) if u >= (1 << 31) else u

    def get_float(self, field: int, default: float = 0.0) -> float:
        v = self.fields.get(field); return float(v[0]) if v else default

    def get_string(self, field: int, default: str = "") -> str:
        v = self.fields.get(field)
        if not v or not isinstance(v[0], bytes): return default
        return v[0].decode('utf-8', errors='replace')

    def get_bool(self, field: int, default: bool = False) -> bool:
        v = self.fields.get(field)
        if not v: return default
        return int(v[0]) != 0

    def get_message(self, field: int) -> Optional['ProtoReader']:
        v = self.fields.get(field)
        if not v or not isinstance(v[0], bytes): return None
        return ProtoReader(v[0]).parse()

    def get_message_array(self, field: int) -> List['ProtoReader']:
        return [ProtoReader(v).parse() for v in self.fields.get(field, []) if isinstance(v, bytes)]


CAMERA_ID_NAMES = {
    0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5',
    5: 'B1', 6: 'B2', 7: 'B3', 8: 'B4', 9: 'B5',
    10: 'C1', 11: 'C2', 12: 'C3', 13: 'C4', 14: 'C5', 15: 'C6',
}


def parse_lri_metadata(lri_path: str) -> Dict[str, Any]:
    """
    Parse LRI file and extract all metadata (ignoring image blocks).
    
    Returns dict with:
      - image_metadata: top-level image properties
      - module_metadata: per-module capture-time settings
      - module_calibration: per-module focus calibration bundles
      - device_metadata: device info and temperature
    """
    with open(lri_path, 'rb') as f:
        data = f.read()

    LELR_MAGIC = b'LELR'
    HEADER_SIZE = 32
    MSG_TYPE_LIGHT = 0

    result = {
        'image_metadata': {},
        'module_metadata': {},
        'module_calibration': {},
        'device_metadata': {},
    }

    offset = 0
    while offset + HEADER_SIZE <= len(data):
        if data[offset:offset+4] != LELR_MAGIC:
            break
        block_len, msg_offset, msg_len, msg_type = struct.unpack_from('<QQIB', data, offset + 4)
        
        if msg_type == MSG_TYPE_LIGHT and msg_len < 5_000_000:
            hdr = ProtoReader(data, offset + msg_offset, msg_len).parse()
            
            # Image-level metadata (fields 1-11, 24)
            result['image_metadata'] = {
                'unique_id_low': hdr.get_uint64(1),
                'unique_id_high': hdr.get_uint64(2),
                'focal_length_mm': hdr.get_int32(4),  # field 4: determines resolution mode
                'reference_camera': CAMERA_ID_NAMES.get(hdr.get_uint64(5), 'unknown'),  # field 5
                'device_id_low': hdr.get_uint64(6),
                'device_id_high': hdr.get_uint64(7),
                'device_model': hdr.get_string(8),
                'device_fw_version': hdr.get_string(9),
                'device_asic_fw_version': hdr.get_string(10),
            }
            
            # Device temperature (field 11)
            temp_msg = hdr.get_message(11)
            if temp_msg:
                result['device_metadata']['temperature_c'] = temp_msg.get_float(1)
            
            # Per-module capture-time metadata (field 12)
            for mod_msg in hdr.get_message_array(12):
                cam_id = mod_msg.get_uint64(2)
                cam_name = CAMERA_ID_NAMES.get(cam_id, f'UNK{cam_id}')
                
                # AF info (field 1 in CameraModule)
                af_info = mod_msg.get_message(1)
                focus_distance = None
                if af_info:
                    # Try disparity_focus_distance (field 3) or contrast_focus_distance (field 4)
                    focus_distance = af_info.get_float(3) or af_info.get_float(4)
                
                result['module_metadata'][cam_name] = {
                    'camera_id': cam_id,
                    'enabled': mod_msg.get_bool(3),
                    'lens_position_hall_code': mod_msg.get_int32(5),  # field 5
                    'mirror_position_hall_code': mod_msg.get_int32(4),  # field 4
                    'sensor_analog_gain': mod_msg.get_float(7),  # field 7: ISO-equivalent
                    'sensor_exposure_ns': mod_msg.get_uint64(8),  # field 8: nanoseconds
                    'sensor_is_horizontal_flip': mod_msg.get_bool(11),
                    'sensor_is_vertical_flip': mod_msg.get_bool(12),
                    'sensor_digital_gain': mod_msg.get_float(14),
                    'frame_index': mod_msg.get_uint64(15),
                    'focus_distance_m': focus_distance,  # meters
                }
            
            # Module calibration (field 13) — includes per-focus bundles
            for calib_msg in hdr.get_message_array(13):
                cam_id = calib_msg.get_uint64(1)
                cam_name = CAMERA_ID_NAMES.get(cam_id, f'UNK{cam_id}')
                
                geo = calib_msg.get_message(3)  # field 3: GeometricCalibration
                if geo:
                    focus_bundles = []
                    for bundle in geo.get_message_array(2):  # field 2: per_focus_calibration
                        focus_dist = bundle.get_float(1)  # field 1: focus_distance
                        hall_code = bundle.get_float(6)   # field 6: focus_hall_code
                        focus_bundles.append({
                            'focus_distance_m': focus_dist,
                            'hall_code': hall_code,
                        })
                    
                    result['module_calibration'][cam_name] = {
                        'camera_id': cam_id,
                        'focus_bundles': focus_bundles,
                    }
            
            # View preferences (field 19) if present
            for vp in hdr.get_message_array(19):
                # Field 7: AWB mode
                awb_mode = vp.get_int64(7)
                # Field 15: AWB gains (ChannelGain message)
                gains_msg = vp.get_message(15)
                if gains_msg:
                    result['image_metadata']['white_balance'] = {
                        'awb_mode': awb_mode,
                        'gain_r': gains_msg.get_float(1),
                        'gain_g_r': gains_msg.get_float(2),
                        'gain_g_b': gains_msg.get_float(3),
                        'gain_b': gains_msg.get_float(4),
                    }
        
        offset += block_len

    return result


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 lri_extract_enhanced.py <input.lri> [output_json]")
        sys.exit(1)
    
    lri_file = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else 'metadata.json'
    
    print(f"Parsing metadata from {lri_file}...")
    metadata = parse_lri_metadata(lri_file)
    
    print("\n=== IMAGE-LEVEL METADATA ===")
    print(json.dumps(metadata['image_metadata'], indent=2))
    
    print("\n=== MODULE CAPTURE SETTINGS ===")
    print(json.dumps(metadata['module_metadata'], indent=2))
    
    print("\n=== MODULE CALIBRATION (Focus Bundles) ===")
    print(json.dumps(metadata['module_calibration'], indent=2))
    
    print("\n=== DEVICE METADATA ===")
    print(json.dumps(metadata['device_metadata'], indent=2))
    
    # Write to JSON file
    with open(out_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {out_json}")
