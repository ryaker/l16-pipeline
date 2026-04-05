#!/usr/bin/env python3
"""
lri_extract_v2.py — Extract per-module images from Light L16 LRI files WITH metadata.

Enhanced version of lri_extract.py that also captures and returns:
  - Image-level metadata: focal_length, reference_camera, device info
  - Per-module capture settings: lens_position, exposure_time, analog_gain, digital_gain, 
    focus_distance, sensor_flips
  - White balance settings: AWB mode and per-channel gains

Output: Image files (A1.png, etc.) + optional metadata.json

Usage:
  python3 lri_extract_v2.py input.lri output_dir [--half] [--raw] [--scale N] [--metadata]

Options:
  --half       Output half-resolution (debayer by subsampling, fast)
  --raw        Output 16-bit grayscale Bayer (no debayer, for custom ISP)
  --scale N    Downsample by factor N after debayer (default: 1 = full res)
  --format FMT Output format: png (default), tiff
  --metadata   Save metadata as JSON alongside images
"""

import sys
import os
import struct
import math
import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

# ── LELR block constants ──────────────────────────────────────────────────────
LELR_MAGIC     = b'LELR'
HEADER_SIZE    = 32
MSG_TYPE_LIGHT = 0

CAMERA_ID_NAMES = {
    0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5',
    5: 'B1', 6: 'B2', 7: 'B3', 8: 'B4', 9: 'B5',
    10: 'C1', 11: 'C2', 12: 'C3', 13: 'C4', 14: 'C5', 15: 'C6',
}

FORMAT_BAYER_JPEG   = 0
FORMAT_PACKED_10BPP = 7
FORMAT_PACKED_12BPP = 8
FORMAT_PACKED_14BPP = 9

_RGGB_R_POS = {
    0: (0, 0),  # RGGB
    1: (0, 1),  # GRBG
    2: (1, 1),  # BGGR
    3: (1, 0),  # GBRG
}


# ── Protobuf parser ───────────────────────────────────────────────────────────

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

    def get_uint32(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        return int(v[0]) & 0xFFFFFFFF

    def get_int64(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        u = int(v[0]); return u - (1 << 64) if u >= (1 << 63) else u

    def get_int32(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        if isinstance(v[0], bytes): return default
        u = int(v[0]) & 0xFFFFFFFF
        return u - (1 << 32) if u >= (1 << 31) else u

    def get_float(self, field: int, default: float = 0.0) -> float:
        v = self.fields.get(field)
        if not v: return default
        if isinstance(v[0], float): return v[0]
        return default

    def get_bool(self, field: int, default: bool = False) -> bool:
        v = self.fields.get(field)
        if not v: return default
        return bool(int(v[0]))

    def get_string(self, field: int, default: str = "") -> str:
        v = self.fields.get(field)
        if not v or not isinstance(v[0], bytes): return default
        try:
            return v[0].decode('utf-8')
        except:
            return default

    def get_message(self, field: int) -> Optional['ProtoReader']:
        v = self.fields.get(field)
        if not v or not isinstance(v[0], bytes): return None
        return ProtoReader(v[0]).parse()

    def get_message_array(self, field: int) -> List['ProtoReader']:
        return [ProtoReader(v).parse() for v in self.fields.get(field, []) if isinstance(v, bytes)]


# ── Metadata dataclasses ──────────────────────────────────────────────────────

@dataclass
class ModuleCapture:
    """Per-module capture settings."""
    camera_id: str = ""
    enabled: bool = True
    lens_position_hall: int = 0
    mirror_position_hall: Optional[int] = None
    sensor_analog_gain: float = 1.0
    sensor_exposure_ns: int = 0
    sensor_digital_gain: Optional[float] = None
    focus_distance_m: Optional[float] = None
    sensor_flip_h: bool = False
    sensor_flip_v: bool = False

@dataclass
class CaptureMetadata:
    """Complete capture metadata for an LRI image."""
    image_id_low: int = 0
    image_id_high: int = 0
    focal_length_mm: int = 0
    reference_camera: str = ""
    device_model: str = ""
    device_fw_version: str = ""
    device_temperature_c: Optional[float] = None
    awb_mode: Optional[int] = None
    awb_gains: Optional[Dict[str, float]] = None
    modules: Dict[str, ModuleCapture] = field(default_factory=dict)


# ── 10-bit unpacking ──────────────────────────────────────────────────────────

def unpack_10bpp(data: bytes, abs_offset: int, width: int, height: int, stride: int) -> np.ndarray:
    """
    Unpack PACKED_10BPP data → uint16 array of shape (height, width).
    Every 4 consecutive pixels occupy 5 bytes (little-endian bit order).
    """
    raw = np.frombuffer(data, dtype=np.uint8,
                        offset=abs_offset,
                        count=stride * height).reshape(height, stride)

    groups_per_row = width // 4
    g = raw[:, :groups_per_row * 5].reshape(height, groups_per_row, 5).astype(np.uint16)

    p0 = g[:, :, 0]        | ((g[:, :, 1] & 0x03) << 8)
    p1 = (g[:, :, 1] >> 2) | ((g[:, :, 2] & 0x0F) << 6)
    p2 = (g[:, :, 2] >> 4) | ((g[:, :, 3] & 0x3F) << 4)
    p3 = (g[:, :, 3] >> 6) |  (g[:, :, 4]         << 2)

    return np.stack([p0, p1, p2, p3], axis=2).reshape(height, width)


# ── BAYER_JPEG decoder ───────────────────────────────────────────────────────

def decode_bayer_jpeg(data: bytes, offset: int, width: int, height: int, pattern: int = 0) -> np.ndarray:
    """Decode BAYER_JPEG format (JPEG of single Bayer CFA plane)."""
    jpeg_end = offset
    while jpeg_end < len(data) and data[jpeg_end:jpeg_end+2] != b'\xFF\xD9':
        jpeg_end += 1
    jpeg_end += 2
    
    jpeg_data = data[offset:jpeg_end]
    from io import BytesIO
    pil = PILImage.open(BytesIO(jpeg_data))
    bayer = np.array(pil, dtype=np.uint16)
    if bayer.ndim == 3:
        bayer = bayer[:, :, 0]
    return bayer


# ── Debayering ────────────────────────────────────────────────────────────────

def debayer_bilinear(bayer: np.ndarray, pattern: int = 0) -> np.ndarray:
    """Bilinear debayer to (height, width, 3) float32 RGB [0..1023]."""
    H, W = bayer.shape
    r_row, r_col = _RGGB_R_POS[pattern]
    
    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)
    
    bayer_f = bayer.astype(np.float32)
    
    for row in range(H):
        for col in range(W):
            if row % 2 == r_row and col % 2 == r_col:
                R[row, col] = bayer_f[row, col]
                if row > 0 and row < H-1:
                    G[row, col] = (bayer_f[row-1, col] + bayer_f[row+1, col]) / 2
                else:
                    G[row, col] = bayer_f[row, col]
                if col > 0 and col < W-1:
                    B[row, col] = (bayer_f[row, col-1] + bayer_f[row, col+1]) / 2
                else:
                    B[row, col] = bayer_f[row, col]
            elif row % 2 == (1 - r_row) and col % 2 == (1 - r_col):
                B[row, col] = bayer_f[row, col]
                if row > 0 and row < H-1:
                    G[row, col] = (bayer_f[row-1, col] + bayer_f[row+1, col]) / 2
                else:
                    G[row, col] = bayer_f[row, col]
                if col > 0 and col < W-1:
                    R[row, col] = (bayer_f[row, col-1] + bayer_f[row, col+1]) / 2
                else:
                    R[row, col] = bayer_f[row, col]
            else:
                G[row, col] = bayer_f[row, col]
                if row % 2 == r_row:
                    if col > 0 and col < W-1:
                        R[row, col] = (bayer_f[row, col-1] + bayer_f[row, col+1]) / 2
                    else:
                        R[row, col] = bayer_f[row, col]
                    if row > 0 and row < H-1:
                        B[row, col] = (bayer_f[row-1, col] + bayer_f[row+1, col]) / 2
                    else:
                        B[row, col] = bayer_f[row, col]
                else:
                    if row > 0 and row < H-1:
                        R[row, col] = (bayer_f[row-1, col] + bayer_f[row+1, col]) / 2
                    else:
                        R[row, col] = bayer_f[row, col]
                    if col > 0 and col < W-1:
                        B[row, col] = (bayer_f[row, col-1] + bayer_f[row, col+1]) / 2
                    else:
                        B[row, col] = bayer_f[row, col]
    
    return np.stack([R, G, B], axis=2)


def debayer_half(bayer: np.ndarray, pattern: int = 0) -> np.ndarray:
    """Half-resolution demosaic: fast subsampling."""
    r_row, r_col = _RGGB_R_POS[pattern]
    b_row, b_col = 1 - r_row, 1 - r_col

    R  = bayer[r_row::2, r_col::2].astype(np.uint16)
    B  = bayer[b_row::2, b_col::2].astype(np.uint16)
    G1 = bayer[r_row::2, b_col::2].astype(np.uint16)
    G2 = bayer[b_row::2, r_col::2].astype(np.uint16)
    G  = ((G1.astype(np.uint32) + G2.astype(np.uint32)) >> 1).astype(np.uint16)

    return np.stack([R, G, B], axis=2)


# ── Metadata extraction ───────────────────────────────────────────────────────

def extract_metadata_from_header(hdr: ProtoReader) -> CaptureMetadata:
    """Extract image-level and device-level metadata from LightHeader proto."""
    meta = CaptureMetadata()
    
    meta.image_id_low = hdr.get_uint64(1)
    meta.image_id_high = hdr.get_uint64(2)
    meta.focal_length_mm = hdr.get_int32(4)
    
    ref_cam_id = hdr.get_uint32(5)
    meta.reference_camera = CAMERA_ID_NAMES.get(ref_cam_id, f"UNK{ref_cam_id}")
    
    meta.device_model = hdr.get_string(8)
    meta.device_fw_version = hdr.get_string(9)
    
    dev_temp_msg = hdr.get_message(11)
    if dev_temp_msg:
        meta.device_temperature_c = dev_temp_msg.get_float(1)
    
    vp_arr = hdr.get_message_array(19)
    if vp_arr:
        vp = vp_arr[0]
        meta.awb_mode = vp.get_int64(7)
        
        awb_gains_arr = vp.get_message_array(15)
        if awb_gains_arr:
            ag = awb_gains_arr[0]
            meta.awb_gains = {
                'r': ag.get_float(1),
                'g_r': ag.get_float(2),
                'g_b': ag.get_float(3),
                'b': ag.get_float(4),
            }
    
    return meta


# ── Main extraction with metadata ─────────────────────────────────────────────

def extract_modules(
    lri_path:  str,
    out_dir:   str,
    half_res:  bool = False,
    raw_bayer: bool = False,
    scale:     int  = 1,
    fmt:       str  = 'png',
) -> Tuple[List[str], CaptureMetadata]:
    """
    Extract all active camera modules from an LRI file.
    Returns: (list of written file paths, CaptureMetadata with per-module details)
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(lri_path, 'rb') as f:
        data = f.read()

    metadata = CaptureMetadata()
    modules = {}

    offset = 0
    block_count = 0
    while offset + HEADER_SIZE <= len(data):
        if data[offset:offset+4] != LELR_MAGIC:
            break
        block_len, msg_offset, msg_len, msg_type = struct.unpack_from('<QQIB', data, offset + 4)
        if msg_type == MSG_TYPE_LIGHT and msg_len < 5_000_000:
            hdr = ProtoReader(data, offset + msg_offset, msg_len).parse()
            
            # Extract image-level metadata (update from first block only)
            if block_count == 0:
                metadata = extract_metadata_from_header(hdr)
            block_count += 1
            
            # Extract per-module metadata (field 12: CameraModule array)
            for mod_msg in hdr.get_message_array(12):
                cam_id = mod_msg.get_uint64(2)
                cam_name = CAMERA_ID_NAMES.get(cam_id, f"UNK{cam_id}")
                
                mod_cap = ModuleCapture(
                    camera_id=cam_name,
                    enabled=mod_msg.get_bool(3),
                    lens_position_hall=mod_msg.get_int32(5),
                    mirror_position_hall=mod_msg.get_int32(4) if mod_msg.fields.get(4) else None,
                    sensor_analog_gain=mod_msg.get_float(7),
                    sensor_exposure_ns=mod_msg.get_uint64(8),
                    sensor_digital_gain=mod_msg.get_float(14),
                    sensor_flip_h=mod_msg.get_bool(11),
                    sensor_flip_v=mod_msg.get_bool(12),
                )
                
                # Extract focus distance from AF info (field 1)
                af_info = mod_msg.get_message(1)
                if af_info:
                    focus_dist = af_info.get_float(3)
                    if focus_dist <= 0:
                        focus_dist = af_info.get_float(4)
                    if focus_dist > 0:
                        mod_cap.focus_distance_m = focus_dist
                metadata.modules[cam_name] = mod_cap
                
                # Extract surface data for image decoding
                surf = mod_msg.get_message(9)
                if surf is None:
                    continue

                size_msg = surf.get_message(2)
                width    = size_msg.get_int64(1) if size_msg else 0
                height   = size_msg.get_int64(2) if size_msg else 0
                fmt_int  = surf.get_uint64(3)
                stride   = surf.get_uint64(4)
                data_off = surf.get_uint64(5)

                bayer_pattern = -1
                bov_arr = mod_msg.get_message_array(13)
                if bov_arr:
                    bov = bov_arr[0]
                    bx  = bov.get_int64(1); by = bov.get_int64(2)
                    bayer_pattern = int((bx + 2) % 2) | (int((by + 2) % 2) << 1)

                if fmt_int in (FORMAT_PACKED_10BPP, FORMAT_BAYER_JPEG):
                    abs_data = int(offset + data_off)
                    if cam_id not in modules or (
                        fmt_int == FORMAT_BAYER_JPEG and abs_data < modules[cam_id]['abs_data']
                    ):
                        modules[cam_id] = {
                            'name':          cam_name,
                            'width':         int(width),
                            'height':        int(height),
                            'stride':        int(stride),
                            'abs_data':      abs_data,
                            'format':        fmt_int,
                            'bayer_pattern': bayer_pattern,
                        }
        offset += block_len

    written = []
    for cam_id in sorted(modules):
        m    = modules[cam_id]
        name = m['name']
        W, H = m['width'], m['height']

        print(f"  Extracting {name} ({W}×{H}) fmt={m['format']} ...", end=' ', flush=True)

        if m['format'] == FORMAT_BAYER_JPEG:
            bayer = decode_bayer_jpeg(data, m['abs_data'], W, H,
                                      m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
        else:
            bayer = unpack_10bpp(data, m['abs_data'], W, H, m['stride'])

        _BLACK = 64
        _WHITE = 1023
        bayer = np.clip(bayer.astype(np.int32) - _BLACK, 0, _WHITE - _BLACK).astype(np.uint16)

        if raw_bayer:
            ext  = '.tiff' if fmt == 'tiff' else '.png'
            path = os.path.join(out_dir, f"{name}_raw{ext}")
            pil  = PILImage.fromarray(bayer.astype(np.uint16), mode='I;16')
            pil.save(path)
        elif half_res:
            rgb = debayer_half(bayer, m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
            path = _save_rgb(rgb, out_dir, name, scale, fmt)
        else:
            rgb_f = debayer_bilinear(bayer, m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
            rgb = np.clip(rgb_f, 0, 1023).astype(np.uint16)
            path = _save_rgb(rgb, out_dir, name, scale, fmt)

        written.append(path)
        print(f"→ {os.path.basename(path)}")

    return written, metadata


def _save_rgb(rgb: np.ndarray, out_dir: str, name: str, scale: int, fmt: str) -> str:
    if scale > 1:
        H, W = rgb.shape[:2]
        nH, nW = H // scale, W // scale
        rgb = rgb[:nH*scale, :nW*scale].reshape(nH, scale, nW, scale, 3).mean(axis=(1, 3)).astype(np.uint16)

    ext = '.tiff' if fmt == 'tiff' else '.png'
    path = os.path.join(out_dir, f"{name}{ext}")

    if fmt == 'tiff':
        pil = PILImage.fromarray(rgb.astype(np.uint16))
        pil.save(path)
    else:
        rgb16 = (rgb.astype(np.float32) * (65535.0 / 959.0)).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))

    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract per-module images and metadata from Light L16 LRI files.'
    )
    parser.add_argument('lri_file',   help='Input .lri file')
    parser.add_argument('output_dir', nargs='?', default='.',
                        help='Output directory [default: current dir]')
    parser.add_argument('--half',     action='store_true',
                        help='Half-resolution debayer (fast subsampling)')
    parser.add_argument('--raw',      action='store_true',
                        help='Output 16-bit Bayer grayscale (no debayer)')
    parser.add_argument('--scale',    type=int, default=1,
                        help='Downsample factor after debayer (default: 1)')
    parser.add_argument('--format',   choices=['png', 'tiff'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--metadata', action='store_true',
                        help='Save metadata as JSON alongside images')
    args = parser.parse_args()

    print(f"Extracting: {args.lri_file}")
    paths, metadata = extract_modules(
        lri_path  = args.lri_file,
        out_dir   = args.output_dir,
        half_res  = args.half,
        raw_bayer = args.raw,
        scale     = args.scale,
        fmt       = args.format,
    )
    print(f"\nExtracted {len(paths)} module images → {args.output_dir}")
    
    if args.metadata:
        meta_path = os.path.join(args.output_dir, "metadata.json")
        
        meta_dict = {
            'image_id_low': metadata.image_id_low,
            'image_id_high': metadata.image_id_high,
            'focal_length_mm': metadata.focal_length_mm,
            'reference_camera': metadata.reference_camera,
            'device_model': metadata.device_model,
            'device_fw_version': metadata.device_fw_version,
            'device_temperature_c': metadata.device_temperature_c,
            'awb_mode': metadata.awb_mode,
            'awb_gains': metadata.awb_gains,
            'modules': {
                name: asdict(mod) for name, mod in metadata.modules.items()
            },
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        print(f"Metadata saved → {meta_path}")


if __name__ == '__main__':
    main()
