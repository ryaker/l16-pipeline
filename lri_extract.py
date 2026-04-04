#!/usr/bin/env python3
"""
lri_extract.py — Extract per-module images from Light L16 LRI files.

Reads PACKED_10BPP raw Bayer data from each camera module, debayers it
with bilinear interpolation, and saves the results as named PNG files.
Output filenames match what lri_calibration.py expects in images.txt
(e.g., A1.png, A2.png, B1.png, ...).

Usage:
  python3 lri_extract.py input.lri output_dir [--half] [--raw] [--scale N]

Options:
  --half          Output half-resolution (debayer by subsampling, fast)
  --raw           Output 16-bit grayscale Bayer (no debayer, for custom ISP)
  --scale N       Downsample by factor N after debayer (default: 1 = full res)
  --format FMT    Output format: png (default), tiff
"""

import sys
import os
import struct
import math
import argparse
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


# ── Minimal protobuf parser (subset of lri_calibration.py) ───────────────────

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

    def get_message(self, field: int) -> Optional['ProtoReader']:
        v = self.fields.get(field)
        if not v or not isinstance(v[0], bytes): return None
        return ProtoReader(v[0]).parse()

    def get_message_array(self, field: int) -> List['ProtoReader']:
        return [ProtoReader(v).parse() for v in self.fields.get(field, []) if isinstance(v, bytes)]


# ── 10-bit unpacking ──────────────────────────────────────────────────────────

def unpack_10bpp(data: bytes, abs_offset: int, width: int, height: int, stride: int) -> np.ndarray:
    """
    Unpack PACKED_10BPP data → uint16 array of shape (height, width).

    Layout: every 4 consecutive pixels occupy 5 bytes (little-endian bit order).
      byte_group = [b0, b1, b2, b3, b4]
      pixel0 = b0        | ((b1 & 0x03) << 8)
      pixel1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
      pixel2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
      pixel3 = (b3 >> 6) |  (b4         << 2)
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

def decode_bayer_jpeg(data: bytes, abs_offset: int, width: int, height: int,
                      bayer_pattern: int) -> np.ndarray:
    """
    Decode a BAYER_JPEG block → uint16 Bayer array (height, width).

    Structure at abs_offset (from lri-rs/block.rs):
      bytes  0– 3: unknown
      bytes  4– 7: format  (u32 LE)  0=colour/4 JPEGs, 1=mono/1 JPEG
      bytes  8–11: jpeg0_len (u32 LE)
      bytes 12–15: jpeg1_len (u32 LE)
      bytes 16–19: jpeg2_len (u32 LE)
      bytes 20–23: jpeg3_len (u32 LE)
      [1576 bytes total header]
      jpeg0 data  (half-res, one Bayer channel)
      jpeg1 data
      jpeg2 data
      jpeg3 data

    Each JPEG is (height//2, width//2) — one of the 4 Bayer sub-channels.
    We decode all 4 and interleave them into the full (height, width) Bayer grid
    using the CFA pattern to assign R/G1/G2/B positions.
    """
    import io
    hdr = data[abs_offset : abs_offset + 1576]
    colour_fmt  = struct.unpack_from('<I', hdr, 4)[0]   # 0=colour, 1=mono
    jpeg_lens   = struct.unpack_from('<4I', hdr, 8)     # lengths of 4 JPEGs

    pos = abs_offset + 1576
    jpegs = []
    n_jpegs = 1 if colour_fmt == 1 else 4
    for i in range(n_jpegs):
        jpegs.append(data[pos : pos + jpeg_lens[i]])
        pos += jpeg_lens[i]

    # Decode each JPEG → numpy (h2, w2) grayscale
    # Each JPEG is a half-res single-channel image for one Bayer position
    channels = []
    for jpg in jpegs:
        img = PILImage.open(io.BytesIO(jpg)).convert('L')
        arr = np.array(img, dtype=np.uint16)
        # JPEGs are 8-bit; scale to 10-bit range to match PACKED_10BPP files
        arr = (arr.astype(np.uint32) * 4).clip(0, 1023).astype(np.uint16)
        channels.append(arr)

    H2, W2 = channels[0].shape

    if colour_fmt == 1:
        # Mono: single full-res JPEG (H×W) — return directly as flat Bayer.
        # Debayer will produce a grey image; that's correct for a mono sensor.
        return channels[0]

    # Colour: 4 half-res JPEGs → interleave into full-res Bayer grid (H2*2, W2*2)
    # JPEG order from lri-rs: jpeg0=R, jpeg1=G1, jpeg2=G2, jpeg3=B.
    # Bayer pattern (from sensor_bayer_red_override) tells us the CFA layout.
    # Pattern: 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR
    bayer_out = np.zeros((H2 * 2, W2 * 2), dtype=np.uint16)
    r_row, r_col = _RGGB_R_POS.get(bayer_pattern, (0, 0))
    b_row, b_col = 1 - r_row, 1 - r_col
    g1_row, g1_col = r_row,  b_col
    g2_row, g2_col = b_row,  r_col

    bayer_out[r_row::2,  r_col::2]  = channels[0]   # R
    bayer_out[g1_row::2, g1_col::2] = channels[1]   # G1
    bayer_out[g2_row::2, g2_col::2] = channels[2]   # G2
    bayer_out[b_row::2,  b_col::2]  = channels[3]   # B
    return bayer_out


# ── Bayer debayering ──────────────────────────────────────────────────────────

# Bayer pattern index: 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR
# (row_offset, col_offset) for R location within 2×2 Bayer tile
_RGGB_R_POS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

def debayer_bilinear(bayer: np.ndarray, pattern: int = 0) -> np.ndarray:
    """
    Bilinear demosaic of Bayer-pattern image.

    Parameters
    ----------
    bayer   : uint16 (H, W) — raw Bayer data (10-bit values in [0..1023])
    pattern : int — Bayer CFA pattern (0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR)

    Returns
    -------
    np.ndarray, float32, shape (H, W, 3), values in [0..1023]
    """
    H, W = bayer.shape
    bayer = bayer.astype(np.float32)

    # Row/col offsets for R, G1, G2, B within the 2×2 tile
    # RGGB:  R@(0,0) G1@(0,1) G2@(1,0) B@(1,1)
    # GRBG:  G1@(0,0) R@(0,1) B@(1,0) G2@(1,1)
    # GBRG:  G1@(0,0) B@(0,1) R@(1,0) G2@(1,1)
    # BGGR:  B@(0,0) G1@(0,1) G2@(1,0) R@(1,1)
    r_row, r_col   = _RGGB_R_POS[pattern]
    b_row, b_col   = 1 - r_row, 1 - r_col
    g1_row, g1_col = r_row,     b_col
    g2_row, g2_col = b_row,     r_col

    # Helper: pad-replicate then apply simple 2D kernel
    def conv2(img, kernel):
        """2D convolution with edge replication."""
        ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
        out = np.zeros_like(img)
        for ki in range(kernel.shape[0]):
            for kj in range(kernel.shape[1]):
                out += padded[ki:ki+H, kj:kj+W] * kernel[ki, kj]
        return out

    # ── Red channel ──────────────────────────────────────────────────────────
    R = np.zeros((H, W), np.float32)
    R[r_row::2, r_col::2] = bayer[r_row::2, r_col::2]   # known R
    # Horizontal interpolation at same-row G positions
    R[r_row::2, g1_col::2] = 0.5 * (
        np.roll(R[r_row::2, :], +1, axis=1)[:, g1_col::2] +
        R[r_row::2, r_col::2]
    )
    # Vertical interpolation at opposite-row G positions
    R[g2_row::2, r_col::2] = 0.5 * (
        np.roll(R[:, r_col::2], +1, axis=0)[g2_row::2, :] +
        R[r_row::2, r_col::2]
    )
    # Bilinear at B positions (average of 4 diagonal R neighbors)
    R[b_row::2, b_col::2] = 0.25 * (
        R[r_row::2, r_col::2] +
        np.roll(R[r_row::2, r_col::2], -1, axis=1) +
        np.roll(R[r_row::2, r_col::2], -1, axis=0) +
        np.roll(np.roll(R[r_row::2, r_col::2], -1, axis=0), -1, axis=1)
    )

    # ── Blue channel ─────────────────────────────────────────────────────────
    B = np.zeros((H, W), np.float32)
    B[b_row::2, b_col::2] = bayer[b_row::2, b_col::2]   # known B
    # Horizontal at G cols
    B[b_row::2, g2_col::2] = 0.5 * (
        np.roll(B[b_row::2, :], +1, axis=1)[:, g2_col::2] +
        B[b_row::2, b_col::2]
    )
    # Vertical at G rows
    B[g1_row::2, b_col::2] = 0.5 * (
        np.roll(B[:, b_col::2], +1, axis=0)[g1_row::2, :] +
        B[b_row::2, b_col::2]
    )
    # Bilinear at R positions
    B[r_row::2, r_col::2] = 0.25 * (
        B[b_row::2, b_col::2] +
        np.roll(B[b_row::2, b_col::2], -1, axis=1) +
        np.roll(B[b_row::2, b_col::2], -1, axis=0) +
        np.roll(np.roll(B[b_row::2, b_col::2], -1, axis=0), -1, axis=1)
    )

    # ── Green channel ────────────────────────────────────────────────────────
    G = np.zeros((H, W), np.float32)
    G[g1_row::2, g1_col::2] = bayer[g1_row::2, g1_col::2]
    G[g2_row::2, g2_col::2] = bayer[g2_row::2, g2_col::2]
    # At R and B positions: average of 4 cross-shaped neighbors
    for (pr, pc) in [(r_row, r_col), (b_row, b_col)]:
        sub = bayer[pr::2, pc::2]
        # Neighbors: up, down, left, right in the G subgrid
        g_rows_above = (pr - 1) % 2; g_rows_below = (pr + 1) % 2
        # Simple average of the 4 immediate same-channel neighbors in the bayer grid
        # Using roll in the full image space is easiest:
        G[pr::2, pc::2] = 0.25 * (
            np.roll(G, -1, axis=0)[pr::2, pc::2] +
            np.roll(G, +1, axis=0)[pr::2, pc::2] +
            np.roll(G, -1, axis=1)[pr::2, pc::2] +
            np.roll(G, +1, axis=1)[pr::2, pc::2]
        )

    return np.stack([R, G, B], axis=2)


def debayer_half(bayer: np.ndarray, pattern: int = 0) -> np.ndarray:
    """
    Half-resolution demosaic: split Bayer into RGGB sub-channels and combine.
    Fast; output is (H/2, W/2, 3) uint16.
    """
    r_row, r_col = _RGGB_R_POS[pattern]
    b_row, b_col = 1 - r_row, 1 - r_col

    R  = bayer[r_row::2, r_col::2].astype(np.uint16)
    B  = bayer[b_row::2, b_col::2].astype(np.uint16)
    G1 = bayer[r_row::2, b_col::2].astype(np.uint16)
    G2 = bayer[b_row::2, r_col::2].astype(np.uint16)
    G  = ((G1.astype(np.uint32) + G2.astype(np.uint32)) >> 1).astype(np.uint16)

    return np.stack([R, G, B], axis=2)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_modules(
    lri_path:  str,
    out_dir:   str,
    half_res:  bool = False,
    raw_bayer: bool = False,
    scale:     int  = 1,
    fmt:       str  = 'png',
) -> List[str]:
    """
    Extract all active camera modules from an LRI file.

    Returns list of written file paths (named A1.png, B2.png, etc.).
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(lri_path, 'rb') as f:
        data = f.read()

    # Walk LELR blocks and collect module descriptors
    modules = {}   # camera_id → {name, width, height, stride, abs_data_offset, format, bayer}
    offset = 0
    while offset + HEADER_SIZE <= len(data):
        if data[offset:offset+4] != LELR_MAGIC:
            break
        block_len, msg_offset, msg_len, msg_type = struct.unpack_from('<QQIB', data, offset + 4)
        if msg_type == MSG_TYPE_LIGHT and msg_len < 5_000_000:
            hdr = ProtoReader(data, offset + msg_offset, msg_len).parse()
            for mod_msg in hdr.get_message_array(12):
                cam_id = mod_msg.get_uint64(2)
                surf   = mod_msg.get_message(9)
                if surf is None:
                    continue

                size_msg = surf.get_message(2)
                width    = size_msg.get_int64(1) if size_msg else 0
                height   = size_msg.get_int64(2) if size_msg else 0
                fmt_int  = surf.get_uint64(3)
                stride   = surf.get_uint64(4)
                data_off = surf.get_uint64(5)   # relative to block start

                # Bayer pattern from sensor_bayer_red_override (field 13)
                bayer_pattern = -1
                bov_arr = mod_msg.get_message_array(13)
                if bov_arr:
                    bov = bov_arr[0]
                    bx  = bov.get_int64(1); by = bov.get_int64(2)
                    bayer_pattern = int((bx + 2) % 2) | (int((by + 2) % 2) << 1)

                if fmt_int in (FORMAT_PACKED_10BPP, FORMAT_BAYER_JPEG):
                    # For BAYER_JPEG: keep only the first exposure per camera
                    # (lowest data_off = exposure index 0 = primary/shortest exposure)
                    abs_data = int(offset + data_off)
                    if cam_id not in modules or (
                        fmt_int == FORMAT_BAYER_JPEG and abs_data < modules[cam_id]['abs_data']
                    ):
                        modules[cam_id] = {
                            'name':          CAMERA_ID_NAMES.get(cam_id, f'UNK{cam_id}'),
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

        # Decode raw Bayer data — format-dependent
        if m['format'] == FORMAT_BAYER_JPEG:
            bayer = decode_bayer_jpeg(data, m['abs_data'], W, H,
                                      m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
        else:
            bayer = unpack_10bpp(data, m['abs_data'], W, H, m['stride'])

        if raw_bayer:
            # Save as 16-bit grayscale (no demosaic)
            ext  = '.tiff' if fmt == 'tiff' else '.png'
            path = os.path.join(out_dir, f"{name}_raw{ext}")
            pil  = PILImage.fromarray(bayer.astype(np.uint16), mode='I;16')
            pil.save(path)
        elif half_res:
            rgb = debayer_half(bayer, m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
            path = _save_rgb(rgb, out_dir, name, scale, fmt)
        else:
            # Full-resolution bilinear debayer
            rgb_f = debayer_bilinear(bayer, m['bayer_pattern'] if m['bayer_pattern'] >= 0 else 0)
            # Clamp and convert to uint16
            rgb = np.clip(rgb_f, 0, 1023).astype(np.uint16)
            path = _save_rgb(rgb, out_dir, name, scale, fmt)

        written.append(path)
        print(f"→ {os.path.basename(path)}")

    return written


def _save_rgb(
    rgb:     np.ndarray,   # (H, W, 3) uint16
    out_dir: str,
    name:    str,
    scale:   int,
    fmt:     str,
) -> str:
    if scale > 1:
        H, W = rgb.shape[:2]
        nH, nW = H // scale, W // scale
        # Box-filter downsample
        rgb = rgb[:nH*scale, :nW*scale].reshape(nH, scale, nW, scale, 3).mean(axis=(1, 3)).astype(np.uint16)

    ext = '.tiff' if fmt == 'tiff' else '.png'
    path = os.path.join(out_dir, f"{name}{ext}")

    from PIL import Image as PILImage
    if fmt == 'tiff':
        # 16-bit TIFF (all 3 channels at 10-bit values in uint16)
        pil = PILImage.fromarray(rgb.astype(np.uint16))
        pil.save(path)
    else:
        # 16-bit PNG: scale 10-bit [0..1023] → 16-bit [0..65535]
        rgb16 = (rgb.astype(np.uint32) * 64).clip(0, 65535).astype(np.uint16)
        # cv2 expects BGR; imwrite with uint16 writes proper 16-bit PNG
        cv2.imwrite(path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))

    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract per-module images from Light L16 LRI files.'
    )
    parser.add_argument('lri_file',   help='Input .lri file')
    parser.add_argument('output_dir', nargs='?', default='.',
                        help='Output directory [default: current dir]')
    parser.add_argument('--half',   action='store_true',
                        help='Half-resolution debayer (fast subsampling)')
    parser.add_argument('--raw',    action='store_true',
                        help='Output 16-bit Bayer grayscale (no debayer)')
    parser.add_argument('--scale',  type=int, default=1,
                        help='Downsample factor after debayer (default: 1)')
    parser.add_argument('--format', choices=['png', 'tiff'], default='png',
                        help='Output format (default: png)')
    args = parser.parse_args()

    print(f"Extracting: {args.lri_file}")
    paths = extract_modules(
        lri_path  = args.lri_file,
        out_dir   = args.output_dir,
        half_res  = args.half,
        raw_bayer = args.raw,
        scale     = args.scale,
        fmt       = args.format,
    )
    print(f"\nExtracted {len(paths)} module images → {args.output_dir}")


if __name__ == '__main__':
    main()
