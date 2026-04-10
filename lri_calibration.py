#!/usr/bin/env python3
"""
lri_calibration.py — Extract per-module calibration from Light L16 LRI files.

Parses the binary LELR block format and protobuf-encoded LightHeader to extract
per-module camera calibration: intrinsics (fx, fy, cx, cy), extrinsics (R, t),
and image dimensions.

Outputs:
  <outdir>/calibration.json  — Full calibration for all modules
  <outdir>/cameras.txt       — COLMAP PINHOLE cameras (one per module)
  <outdir>/images.txt        — COLMAP image poses (rotation + translation)
  <outdir>/frame_info.txt    — Per-frame sensor info (gain, exposure, format)

Usage:
  python3 lri_calibration.py input.lri [output_dir]
  python3 lri_calibration.py input.lri [output_dir] --json-only

Field map (from lri-cpp protobuf headers):
  LightHeader.modules          field 12 → CameraModule[]
  LightHeader.module_calibration field 13 → FactoryModuleCalibration[]

  CameraModule:
    id                    field 2  (uint64 → CameraID 0-15)
    lens_position         field 5  (int64 — actual focus hall code at shot time)
    sensor_analog_gain    field 7  (float)
    sensor_exposure       field 8  (uint64 ns)
    sensor_data_surface   field 9  (Surface message)
    sensor_bayer_red_override field 13 (Point2I optional)
    sensor_is_horizontal_flip field 11 (bool)
    sensor_is_vertical_flip   field 12 (bool)

  Surface:
    size       field 2  (Point2I: x=width, y=height)
    format     field 3  (uint64 → FormatType: 0=BAYER_JPEG, 7=PACKED_10BPP)
    row_stride field 4  (uint64)
    data_offset field 5 (uint64)

  FactoryModuleCalibration:
    camera_id  field 1  (uint64)
    geometry   field 3  (GeometricCalibration optional)

  GeometricCalibration:
    per_focus_calibration  field 2  (CalibrationFocusBundle[])

  CalibrationFocusBundle:
    focus_distance  field 1  (float)
    intrinsics      field 2  (Intrinsics optional)
    extrinsics      field 3  (Extrinsics optional)
    focus_hall_code field 6  (float optional — correlates to lens_position)

  Intrinsics:
    k_mat  field 1  (Matrix3x3F)

  Matrix3x3F:
    x00 field 1 = fx    x01 field 2 = 0    x02 field 3 = cx
    x10 field 4 = 0     x11 field 5 = fy   x12 field 6 = cy
    x20 field 7 = 0     x21 field 8 = 0    x22 field 9 = 1

  GeometricCalibration:
    mirror_type            field 1  (uint64: 0=NONE, 1=GLUED, 2=MOVABLE)
    per_focus_calibration  field 2  (CalibrationFocusBundle[])

  Extrinsics:
    canonical        field 1  (CanonicalFormat optional — NONE/GLUED cameras)
    moveable_mirror  field 2  (MovableMirrorFormat optional — MOVABLE cameras)

  CanonicalFormat:
    rotation     field 1  (Matrix3x3F — 3x3 rotation matrix)
    translation  field 2  (Point3F — x, y, z)

  MovableMirrorFormat:
    mirror_system           field 1  (MirrorSystem)
    mirror_actuator_mapping field 2  (MirrorActuatorMapping)

  MirrorSystem:
    real_camera_location                      field 1  (Point3F — mm, A1 = origin)
    real_camera_orientation                   field 2  (Matrix3x3F — R world→cam)
    rotation_axis                             field 3  (Point3F — unit vector)
    point_on_rotation_axis                    field 4  (Point3F — mm)
    distance_mirror_plane_to_point_on_axis    field 5  (float — mm)
    mirror_normal_at_zero_degrees             field 6  (Point3F — unit vector)
    flip_img_around_x                         field 7  (bool)

  MirrorActuatorMapping:
    actuator_length_offset  field 2  (float)
    actuator_length_scale   field 3  (float)
    mirror_angle_offset     field 4  (float — degrees)
    mirror_angle_scale      field 5  (float — degrees / normalized unit)

  Point3F:
    x field 1, y field 2, z field 3  (float)
"""

import sys
import os
import struct
import json
import math
from typing import Optional, List, Dict, Any, Tuple

# ── LELR block header (32 bytes, little-endian, packed) ──────────────────────
LELR_MAGIC       = b'LELR'
HEADER_SIZE      = 32
MSG_TYPE_LIGHT   = 0   # LightHeader protobuf

CAMERA_ID_NAMES = {
    0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5',
    5: 'B1', 6: 'B2', 7: 'B3', 8: 'B4', 9: 'B5',
    10: 'C1', 11: 'C2', 12: 'C3', 13: 'C4', 14: 'C5', 15: 'C6',
}

FORMAT_NAMES = {
    0: 'BAYER_JPEG', 7: 'PACKED_10BPP', 8: 'PACKED_12BPP', 9: 'PACKED_14BPP',
}

# ── Minimal protobuf wire-format parser ──────────────────────────────────────
# Wire types: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit

class ProtoReader:
    """Parse protobuf binary into a flat dict of {field_number: [values]}."""

    def __init__(self, data: bytes, offset: int = 0, length: int = -1):
        self.data   = data
        self.pos    = offset
        self.end    = offset + (length if length >= 0 else len(data) - offset)
        self.fields: Dict[int, List[Any]] = {}

    def _read_varint(self) -> int:
        result, shift = 0, 0
        while True:
            b = self.data[self.pos]; self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def parse(self) -> 'ProtoReader':
        while self.pos < self.end:
            key       = self._read_varint()
            field_num = key >> 3
            wire_type = key & 0x7

            if wire_type == 0:   # varint
                val = self._read_varint()
            elif wire_type == 1: # 64-bit
                val = struct.unpack_from('<Q', self.data, self.pos)[0]
                self.pos += 8
            elif wire_type == 2: # length-delimited (bytes / nested message)
                length = self._read_varint()
                val    = self.data[self.pos : self.pos + length]
                self.pos += length
            elif wire_type == 5: # 32-bit
                val = struct.unpack_from('<f', self.data, self.pos)[0]
                self.pos += 4
            else:
                raise ValueError(f"Unknown wire type {wire_type} at pos {self.pos}")

            self.fields.setdefault(field_num, []).append(val)
        return self

    def get_uint64(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        return int(v[0]) if v else default

    def get_int64(self, field: int, default: int = 0) -> int:
        v = self.fields.get(field)
        if not v: return default
        u = int(v[0])
        # Convert unsigned 64-bit varint to signed
        return u - (1 << 64) if u >= (1 << 63) else u

    def get_float(self, field: int, default: float = 0.0) -> float:
        v = self.fields.get(field)
        return float(v[0]) if v else default

    def get_bytes(self, field: int) -> Optional[bytes]:
        v = self.fields.get(field)
        return v[0] if v and isinstance(v[0], bytes) else None

    def get_message(self, field: int) -> Optional['ProtoReader']:
        b = self.get_bytes(field)
        if b is None: return None
        return ProtoReader(b).parse()

    def get_message_array(self, field: int) -> List['ProtoReader']:
        vals = self.fields.get(field, [])
        result = []
        for v in vals:
            if isinstance(v, bytes):
                result.append(ProtoReader(v).parse())
        return result

    def get_float_array(self, field: int) -> List[float]:
        return [float(v) for v in self.fields.get(field, [])]

# ── Calibration extraction helpers ───────────────────────────────────────────

def parse_matrix3x3(msg: ProtoReader) -> List[List[float]]:
    """Return 3×3 matrix as [[row0], [row1], [row2]]."""
    return [
        [msg.get_float(1), msg.get_float(2), msg.get_float(3)],
        [msg.get_float(4), msg.get_float(5), msg.get_float(6)],
        [msg.get_float(7), msg.get_float(8), msg.get_float(9)],
    ]

def parse_point3f(msg: ProtoReader) -> List[float]:
    return [msg.get_float(1), msg.get_float(2), msg.get_float(3)]

def intrinsics_from_kmat(mat: List[List[float]]) -> Dict[str, float]:
    """Extract fx, fy, cx, cy from the camera intrinsic matrix."""
    return {
        'fx': mat[0][0],
        'fy': mat[1][1],
        'cx': mat[0][2],
        'cy': mat[1][2],
        'skew': mat[0][1],
    }

def rotation_matrix_to_colmap_quat(R: List[List[float]]) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    m = R
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return (w, x, y, z)

def pick_focus_bundle(bundles: List[ProtoReader], lens_position: int) -> Optional[ProtoReader]:
    """
    Select the CalibrationFocusBundle whose focus_hall_code is closest
    to the actual lens_position used at capture time.
    Falls back to the middle bundle if no focus_hall_code is present.
    """
    if not bundles:
        return None

    candidates = []
    for b in bundles:
        hall = b.get_float(6, default=None)
        if hall is not None:
            candidates.append((abs(hall - lens_position), b))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # No hall codes — pick middle bundle (reasonable for daylight photography)
    return bundles[len(bundles) // 2]


# ── Movable-mirror virtual camera math ───────────────────────────────────────

def _mat3_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _dot3(a: List[float], b: List[float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _rodrigues(v: List[float], axis: List[float], angle_deg: float) -> List[float]:
    """Rotate vector v by angle_deg degrees around unit axis."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    k = axis
    kxv = [k[1]*v[2]-k[2]*v[1], k[2]*v[0]-k[0]*v[2], k[0]*v[1]-k[1]*v[0]]
    kdv = _dot3(k, v)
    return [v[i]*c + kxv[i]*s + k[i]*kdv*(1-c) for i in range(3)]

def _householder(n: List[float]) -> List[List[float]]:
    """3×3 reflection matrix across plane with unit normal n.  det = -1."""
    return [[float(i==j) - 2*n[i]*n[j] for j in range(3)] for i in range(3)]


def hall_code_to_mirror_angle(mam_msg: ProtoReader, hall_code: int) -> float:
    """
    Convert actuator hall code to mirror angle (degrees) using
    MEAN_STD_NORMALIZE transform from MirrorActuatorMapping.

    Formula:
      normalized = (hall_code - actuator_length_offset) / actuator_length_scale
      angle_deg  = mirror_angle_offset + normalized * mirror_angle_scale
    """
    a_off   = mam_msg.get_float(2)   # actuator_length_offset
    a_scale = mam_msg.get_float(3)   # actuator_length_scale
    m_off   = mam_msg.get_float(4)   # mirror_angle_offset (degrees)
    m_scale = mam_msg.get_float(5)   # mirror_angle_scale  (degrees / normalized unit)
    if a_scale == 0:
        return m_off
    normalized = (hall_code - a_off) / a_scale
    return m_off + normalized * m_scale


def compute_movable_mirror_pose(
    ms_msg:  ProtoReader,
    mam_msg: ProtoReader,
    mirror_hall_code: int,
) -> Tuple[Optional[List[List[float]]], Optional[List[float]]]:
    """
    Derive virtual camera extrinsics (R, t) for a movable-mirror module.

    The periscope camera's physical sensor faces inward toward a flat mirror;
    the mirror redirects the optical path toward the scene.  The virtual
    camera (what we pretend is looking directly at the scene) has:
      - Position  = reflection of the real sensor position across the mirror plane
      - Viewing direction = (0,0,1) rotated around the actuator axis by the
                             mirror angle.  Empirically verified on L16_04574:
                             this produces ~37° off-axis for B1/B2/B3/B5 matching
                             the 28mm A-camera FOV corners.  (The proto's
                             `mirror_normal_at_zero_degrees` field is NOT the
                             physical mirror normal in a way that directly gives
                             the virtual camera forward — −n comes out at ~53°.)

    We construct R_virt via look-at: forward is the rotated sensor axis,
    up is world +y (A-camera convention), right = up × forward.
    R_virt rows = [right, up, forward] — world→cam, det = +1.

    Returns (R_virt 3×3, t_virt 3-vector) or (None, None) if data is missing.
    """
    loc_msg  = ms_msg.get_message(1)   # real_camera_location  (Point3F)
    axis_msg = ms_msg.get_message(3)   # rotation_axis          (Point3F)
    pont_msg = ms_msg.get_message(4)   # point_on_rotation_axis (Point3F)
    dist     = ms_msg.get_float(5)     # dist mirror-plane → axis point (mm)
    norm_msg = ms_msg.get_message(6)   # mirror_normal_at_zero_degrees (Point3F)

    if not all([loc_msg, axis_msg, pont_msg, norm_msg]):
        return None, None

    C_real = parse_point3f(loc_msg)        # real camera centre in world (mm)
    axis   = parse_point3f(axis_msg)       # rotation axis (unit vector)
    pont   = parse_point3f(pont_msg)       # point on rotation axis
    n0     = parse_point3f(norm_msg)       # mirror normal at 0 °

    # Normalize axis (should already be unit, but be safe)
    ax_len = math.sqrt(_dot3(axis, axis))
    if ax_len < 1e-9: return None, None
    axis = [v/ax_len for v in axis]

    # Mirror normal at actual angle (rotate n0 around the actuator axis)
    angle_deg = hall_code_to_mirror_angle(mam_msg, mirror_hall_code)
    n = _rodrigues(n0, axis, angle_deg)
    n_len = math.sqrt(_dot3(n, n))
    if n_len < 1e-9: return None, None
    n = [v/n_len for v in n]

    # Mirror plane anchor point:  P = pont − dist * n
    P = [pont[i] - dist * n[i] for i in range(3)]

    # Virtual camera centre = reflect real centre across mirror plane
    d      = _dot3(n, [C_real[i] - P[i] for i in range(3)])
    C_virt = [C_real[i] - 2*d*n[i] for i in range(3)]

    # ── Build R_virt via look-at from rotated sensor axis ─────────────────
    # Virtual forward = (0,0,1) rotated around the actuator axis by angle_deg.
    # Empirical: this gives ~37° off-axis for L16_04574 B1/B2/B3/B5, matching
    # the four A-FOV corners.  Using −n here gives ~53°, wrong by the
    # complement (cos→sin swap).
    fwd = _rodrigues([0.0, 0.0, 1.0], axis, angle_deg)
    fl = math.sqrt(_dot3(fwd, fwd))
    if fl < 1e-9: return None, None
    fwd = [v/fl for v in fwd]

    # World up convention (matches A-camera identity R in the L16 calibration)
    up = [0.0, 1.0, 0.0]
    # Guard: if forward and up are nearly parallel, use an alternate up axis
    if abs(_dot3(up, fwd)) > 0.999:
        up = [0.0, 0.0, 1.0] if abs(fwd[1]) < 0.9 else [1.0, 0.0, 0.0]

    # right = up × forward
    right = [up[1]*fwd[2] - up[2]*fwd[1],
             up[2]*fwd[0] - up[0]*fwd[2],
             up[0]*fwd[1] - up[1]*fwd[0]]
    rl = math.sqrt(_dot3(right, right))
    if rl < 1e-9: return None, None
    right = [v/rl for v in right]

    # up' = forward × right (make orthogonal)
    new_up = [fwd[1]*right[2] - fwd[2]*right[1],
              fwd[2]*right[0] - fwd[0]*right[2],
              fwd[0]*right[1] - fwd[1]*right[0]]
    ul = math.sqrt(_dot3(new_up, new_up))
    new_up = [v/ul for v in new_up]

    # R_virt rows = [right, up, forward]  (world→cam, det = +1, trace in [-1, 3])
    R_virt = [right, new_up, fwd]

    # COLMAP translation:  t = −R · C_virt
    t_virt = [-sum(R_virt[i][j]*C_virt[j] for j in range(3)) for i in range(3)]

    return R_virt, t_virt


# ── Main extraction ───────────────────────────────────────────────────────────

def parse_lri(path: str) -> Dict[str, Any]:
    """
    Read an LRI file and return per-module calibration + frame info.

    Returns a dict:
      {
        'lri_path': str,
        'image_focal_length_mm': int or None,
        'modules': [
          {
            'camera_id': int,
            'camera_name': str,      # 'A1' .. 'C6'
            'width': int,
            'height': int,
            'format': str,           # 'BAYER_JPEG' etc.
            'lens_position': int,
            'analog_gain': float,
            'exposure_ns': int,
            'is_mono': bool,         # True if no bayer_override (monochrome module)
            'flip_h': bool,
            'flip_v': bool,
            'bayer_pattern': int,    # 0=RGGB 1=GRBG 2=GBRG 3=BGGR, -1=mono
            'calibration': {
              'focus_distance': float,
              'fx': float, 'fy': float,
              'cx': float, 'cy': float,
              'rotation': [[3x3]],    # R: world→camera
              'translation': [x,y,z], # t: world→camera (mm scale)
            } or None
          }
        ]
      }
    """
    with open(path, 'rb') as f:
        data = f.read()

    # ── Walk ALL LELR blocks ──────────────────────────────────────────────────
    # The LRI format splits data across multiple blocks:
    #   - Image blocks (type 0, large ~85 MB): contain modules with image data
    #   - Calibration blocks (type 0, small ~33 KB): contain module_calibration
    # We must accumulate from every LightHeader block before merging.
    all_header_msgs: List[ProtoReader] = []
    offset = 0
    while offset + HEADER_SIZE <= len(data):
        sig = data[offset:offset+4]
        if sig != LELR_MAGIC:
            break
        block_len, msg_offset, msg_len, msg_type = struct.unpack_from(
            '<QQIB', data, offset + 4)
        if msg_type == MSG_TYPE_LIGHT:
            abs_msg_start = offset + msg_offset
            all_header_msgs.append(
                ProtoReader(data, abs_msg_start, msg_len).parse())
        offset += block_len

    if not all_header_msgs:
        raise ValueError(f"No LightHeader found in {path}")

    # Global image focal length from the first header that has it
    image_focal_length = None
    for hdr in all_header_msgs:
        fl = hdr.get_uint64(4)
        if fl:
            image_focal_length = fl
            break

    # ── Parse modules (field 12) — accumulate from all blocks ────────────────
    module_info: Dict[int, Dict] = {}   # camera_id → info

    for light_header_msg in all_header_msgs:
        for mod_msg in light_header_msg.get_message_array(12):
            cam_id      = mod_msg.get_uint64(2)
            lens_pos    = mod_msg.get_int64(5)
            mirror_pos  = mod_msg.get_int64(4)    # CameraModule field 4: mirror_position
            analog_gain = mod_msg.get_float(7)
            exposure_ns = mod_msg.get_uint64(8)
            flip_h      = bool(mod_msg.get_uint64(11))
            flip_v      = bool(mod_msg.get_uint64(12))

            # sensor_data_surface (field 9)
            surf_msg = mod_msg.get_message(9)
            width = height = 0
            fmt_int = 0
            if surf_msg:
                size_msg = surf_msg.get_message(2)
                if size_msg:
                    width  = size_msg.get_int64(1)
                    height = size_msg.get_int64(2)
                fmt_int = surf_msg.get_uint64(3)

            # Bayer pattern from sensor_bayer_red_override (field 13)
            bayer_override_msgs = mod_msg.get_message_array(13)
            bayer_pattern = -1   # -1 = monochrome
            if bayer_override_msgs:
                bov = bayer_override_msgs[0]
                bx = bov.get_int64(1)
                by = bov.get_int64(2)
                bayer_pattern = int((bx + 2) % 2) | (int((by + 2) % 2) << 1)

            module_info[cam_id] = {
                'camera_id':      cam_id,
                'camera_name':    CAMERA_ID_NAMES.get(cam_id, f'UNK{cam_id}'),
                'width':          int(width),
                'height':         int(height),
                'format':         FORMAT_NAMES.get(fmt_int, f'FMT{fmt_int}'),
                'lens_position':  int(lens_pos),
                'mirror_position': int(mirror_pos),   # hall code at shot time
                'analog_gain':    analog_gain,
                'exposure_ns':    int(exposure_ns),
                'is_mono':        bayer_pattern == -1,
                'flip_h':         flip_h,
                'flip_v':         flip_v,
                'bayer_pattern':  bayer_pattern,
                'calibration':    None,
            }

    # ── Parse factory calibration (field 13) — accumulate from all blocks ────
    calib_raw: List[ProtoReader] = []
    for light_header_msg in all_header_msgs:
        calib_raw.extend(light_header_msg.get_message_array(13))

    for cal_msg in calib_raw:
        cam_id = cal_msg.get_uint64(1)

        # geometry field 3 → GeometricCalibration
        geo_msg = cal_msg.get_message(3)
        if geo_msg is None:
            continue

        # per_focus_calibration field 2 → CalibrationFocusBundle[]
        bundles = geo_msg.get_message_array(2)
        if not bundles:
            continue

        # Intrinsics and extrinsics live in SEPARATE CalibrationFocusBundle entries.
        # Intrinsics bundles have a focus_hall_code (≠ 0) and field 2 (Intrinsics).
        # The extrinsics bundle has hall_code = 0 and field 3 (Extrinsics).
        #
        # Extrinsics come in two flavours:
        #   NONE / GLUED cameras → Extrinsics.canonical  (field 1) = direct R, t
        #   MOVABLE cameras      → Extrinsics.moveable_mirror (field 2) = mirror system
        #
        # For MOVABLE cameras we compute the virtual camera pose via mirror reflection.

        # mirror_type from GeometricCalibration field 1  (0=NONE, 1=GLUED, 2=MOVABLE)
        mirror_type_int = geo_msg.fields.get(1, [0])[0]
        MIRROR_TYPES = {0: 'NONE', 1: 'GLUED', 2: 'MOVABLE'}
        mirror_type_str = MIRROR_TYPES.get(mirror_type_int, f'UNK{mirror_type_int}')

        mod = module_info.get(cam_id, {})
        lens_pos      = mod.get('lens_position', 0)
        mirror_pos    = mod.get('mirror_position', 0)   # actual mirror hall code at shot
        intrinsics_data  = None
        rotation_data    = None
        translation_data = None
        focus_dist       = 0.0

        # Find best intrinsics bundle (closest focus_hall_code to actual lens_position)
        intr_bundle = pick_focus_bundle(
            [b for b in bundles if b.get_message(2) is not None],
            lens_pos)
        if intr_bundle:
            focus_dist = intr_bundle.get_float(1)
            intr_msg   = intr_bundle.get_message(2)
            kmat_msg   = intr_msg.get_message(1) if intr_msg else None
            if kmat_msg:
                mat = parse_matrix3x3(kmat_msg)
                intrinsics_data = intrinsics_from_kmat(mat)
                intrinsics_data['k_mat'] = mat

        # Find extrinsics bundle (hall_code = 0, field 3 present)
        camera_location = None   # 3D world position of real lens centre (mm)
        for b in bundles:
            extr_msg = b.get_message(3)
            if extr_msg is None:
                continue

            # ── canonical (NONE / GLUED) ──────────────────────────────────────
            canonical_msgs = extr_msg.get_message_array(1)
            if canonical_msgs:
                canon   = canonical_msgs[0]
                rot_msg = canon.get_message(1)
                tra_msg = canon.get_message(2)
                if rot_msg:
                    rotation_data = parse_matrix3x3(rot_msg)
                if tra_msg:
                    translation_data = parse_point3f(tra_msg)
                # Derive world position: C = -R^T @ t
                if rotation_data and translation_data:
                    R = rotation_data
                    t = translation_data
                    camera_location = [
                        -sum(R[j][i] * t[j] for j in range(3))
                        for i in range(3)
                    ]
                break

            # ── movable mirror ────────────────────────────────────────────────
            mov_msgs = extr_msg.get_message_array(2)
            if mov_msgs:
                mov     = mov_msgs[0]
                ms_msg  = mov.get_message(1)   # MirrorSystem
                mam_msg = mov.get_message(2)   # MirrorActuatorMapping
                if ms_msg and mam_msg:
                    rotation_data, translation_data = compute_movable_mirror_pose(
                        ms_msg, mam_msg, mirror_pos)
                    # Extract real_camera_location directly (physical lens position)
                    loc_msg = ms_msg.get_message(1)
                    if loc_msg:
                        camera_location = parse_point3f(loc_msg)
                break

        if cam_id in module_info:
            module_info[cam_id]['calibration'] = {
                'mirror_type':    mirror_type_str,
                'focus_distance': focus_dist,
                'focus_bundles':  len(bundles),
                'intrinsics':     intrinsics_data,
                'rotation':       rotation_data,
                'translation':    translation_data,
                'camera_location': camera_location,  # 3D world pos of real lens (mm)
            }

    return {
        'lri_path':             os.path.abspath(path),
        'image_focal_length_mm': int(image_focal_length) if image_focal_length else None,
        'modules':              list(module_info.values()),
    }


# ── COLMAP output ─────────────────────────────────────────────────────────────

def write_colmap(result: Dict, outdir: str):
    """
    Write COLMAP cameras.txt and images.txt with known-calibration poses.

    COLMAP cameras.txt format:
      CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
      Model PINHOLE: fx fy cx cy

    COLMAP images.txt format:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      (blank line — no 2D keypoints for now)
    """
    cameras_path = os.path.join(outdir, 'cameras.txt')
    images_path  = os.path.join(outdir, 'images.txt')

    cam_lines  = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] (fx, fy, cx, cy for PINHOLE)",
        "# Generated by lri_calibration.py from Light L16 LRI factory calibration",
        "",
    ]
    img_lines  = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# (second line empty — no 2D keypoints; add with lri_extractor output)",
        "# Generated by lri_calibration.py from Light L16 LRI factory calibration",
        "",
    ]

    cam_id_col = 1  # COLMAP camera ID counter
    img_id_col = 1  # COLMAP image ID counter

    for mod in result['modules']:
        cal = mod['calibration']
        if cal is None or cal['intrinsics'] is None:
            continue
        intr = cal['intrinsics']
        w, h = mod['width'], mod['height']
        if w == 0 or h == 0:
            continue

        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']

        cam_lines.append(
            f"{cam_id_col} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
            f"  # {mod['camera_name']} ({'MONO' if mod['is_mono'] else 'COLOR'})"
        )

        if cal['rotation'] is not None and cal['translation'] is not None:
            qw, qx, qy, qz = rotation_matrix_to_colmap_quat(cal['rotation'])
            tx, ty, tz = cal['translation']
            img_lines.append(
                f"{img_id_col} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{tx:.6f} {ty:.6f} {tz:.6f} {cam_id_col} "
                f"{mod['camera_name']}.png"
            )
            img_lines.append("")  # empty keypoints line

        cam_id_col += 1
        img_id_col += 1

    with open(cameras_path, 'w') as f:
        f.write('\n'.join(cam_lines) + '\n')
    with open(images_path, 'w') as f:
        f.write('\n'.join(img_lines) + '\n')

    return cameras_path, images_path


def write_frame_info(result: Dict, outdir: str):
    """Write a human-readable per-frame summary."""
    path = os.path.join(outdir, 'frame_info.txt')
    lines = [
        f"LRI: {result['lri_path']}",
        f"Image focal length (35mm equiv): {result['image_focal_length_mm']} mm",
        f"Modules captured: {len(result['modules'])}",
        "",
        f"{'ID':<4} {'Name':<4} {'W':>5} {'H':>5} {'Format':<12} "
        f"{'Gain':>6} {'Exp(µs)':>9} {'Mono':>5} {'Cal':>4} {'FocusBundles':>13}",
        "-" * 75,
    ]
    for mod in result['modules']:
        cal = mod['calibration']
        lines.append(
            f"{mod['camera_id']:<4} {mod['camera_name']:<4} "
            f"{mod['width']:>5} {mod['height']:>5} "
            f"{mod['format']:<12} "
            f"{mod['analog_gain']:>6.2f} "
            f"{mod['exposure_ns']//1000:>9} "
            f"{'yes' if mod['is_mono'] else 'no':>5} "
            f"{'yes' if cal else 'no':>4} "
            f"{cal['focus_bundles'] if cal else '-':>13}"
        )
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# ── Sensor calibration: vignetting + CCM + AWB ──────────────────────────────

def extract_sensor_calibration(path: str) -> Dict:
    """
    Extract per-module sensor calibration from an LRI file.

    Returns a dict with:
      'awb': {'R': float, 'Gr': float, 'Gb': float, 'B': float}
      'modules': {
          'A1': {
              'vignetting': np.ndarray (H_grid, W_grid) float32 gain multipliers,
              'ccm': [{'mode': int, 'forward': 3x3, 'inverse': 3x3}, ...],
          }, ...
      }

    The vignetting grid is a 17×13 flat-field correction: multiply raw pixel
    values by the bilinear-interpolated grid to remove lens/sensor vignetting.
    Grid values are normalised so the brightest point = 1.0 and corners > 1.

    CCMs are 3×3 color correction matrices at different illuminant modes.
    """
    import numpy as np

    with open(path, 'rb') as f:
        data = f.read()

    all_header_msgs: List[ProtoReader] = []
    offset = 0
    while offset + HEADER_SIZE <= len(data):
        sig = data[offset:offset+4]
        if sig != LELR_MAGIC:
            break
        block_len, msg_offset, msg_len, msg_type = struct.unpack_from(
            '<QQIB', data, offset + 4)
        if msg_type == MSG_TYPE_LIGHT:
            abs_msg_start = offset + msg_offset
            all_header_msgs.append(
                ProtoReader(data, abs_msg_start, msg_len).parse())
        offset += block_len

    result: Dict[str, Any] = {'awb': None, 'modules': {}}

    for hdr in all_header_msgs:
        # ── AWB from ViewPreferences (field 19) ─────────────────────────────
        vp_msgs = hdr.get_message_array(19)
        if vp_msgs and result['awb'] is None:
            vp = vp_msgs[0]
            gain_msg = vp.get_message(15)   # ChannelGain: f1=R f2=Gr f3=Gb f4=B
            if gain_msg:
                result['awb'] = {
                    'R':  gain_msg.get_float(1),
                    'Gr': gain_msg.get_float(2),
                    'Gb': gain_msg.get_float(3),
                    'B':  gain_msg.get_float(4),
                }

        # ── Vignetting from FactoryModuleCalibration field 4 ────────────────
        for cal_msg in hdr.get_message_array(13):
            cam_id = cal_msg.get_uint64(1)
            cam_name = CAMERA_ID_NAMES.get(cam_id, f'cam{cam_id}')
            if cam_name not in result['modules']:
                result['modules'][cam_name] = {'vignetting': None, 'ccm': [], 'cra': None}

            f4_msgs = cal_msg.get_message_array(4)
            if f4_msgs:
                f4 = f4_msgs[0]
                f2_msgs = f4.get_message_array(2)
                if f2_msgs:
                    f2 = f2_msgs[0]
                    sub = f2.get_message_array(2)
                    if sub:
                        s = sub[0]
                        gw = s.get_uint64(1)
                        gh = s.get_uint64(2)
                        blob = s.fields.get(3, [b''])[0]
                        if isinstance(blob, bytes) and len(blob) >= gw * gh * 4:
                            vig = np.frombuffer(blob, dtype=np.float32)[:gw * gh]
                            result['modules'][cam_name]['vignetting'] = \
                                vig.reshape(int(gh), int(gw)).copy()

            # ── CRA grid from photometric field 1 ────────────────────────────
            # Structure: photometric.f1 → {sf1=width(17), sf2=height(13), sf4=blob}
            # blob = width × height × 16 float32 values = 17×13 matrices of 4×4
            # The 4×4 matrix is a per-cell Bayer channel mixing correction:
            #   channels = (R=0, Gr=1, Gb=2, B=3)
            #   output_RGGB = M @ input_RGGB  (corrects cross-channel leakage from CRA)
            if f4_msgs:
                f4 = f4_msgs[0]
                f1_raw_list = f4.fields.get(1, [])
                if f1_raw_list:
                    f1_raw = f1_raw_list[0]
                    if isinstance(f1_raw, (bytes, bytearray)):
                        try:
                            cra_sub = ProtoReader(f1_raw, 0, len(f1_raw)).parse()
                            cw = int(cra_sub.fields.get(1, [17])[0])
                            ch = int(cra_sub.fields.get(2, [13])[0])
                            cra_blob = cra_sub.fields.get(4, [b''])[0]
                            if (isinstance(cra_blob, (bytes, bytearray))
                                    and len(cra_blob) == cw * ch * 16 * 4):
                                cra_4x4 = np.frombuffer(cra_blob, dtype=np.float32)
                                result['modules'][cam_name]['cra'] = \
                                    cra_4x4.reshape(ch, cw, 4, 4).copy()
                        except Exception:
                            pass

            # ── CCM from ColorCalibration (field 2 inside field 13) ─────────
            f2_color = cal_msg.get_message_array(2)
            if f2_color:
                for cc in f2_color:
                    mode = cc.get_uint64(1)
                    fwd_msgs = cc.get_message_array(2)
                    inv_msgs = cc.get_message_array(3)
                    fwd = None
                    inv = None
                    if fwd_msgs:
                        m = fwd_msgs[0]
                        fwd = np.array([
                            m.fields.get(j, [0.0])[0] for j in range(1, 10)
                        ], dtype=np.float32).reshape(3, 3)
                    if inv_msgs:
                        m = inv_msgs[0]
                        inv = np.array([
                            m.fields.get(j, [0.0])[0] for j in range(1, 10)
                        ], dtype=np.float32).reshape(3, 3)
                    if fwd is not None:
                        result['modules'][cam_name]['ccm'].append({
                            'mode': int(mode),
                            'forward': fwd,
                            'inverse': inv,
                        })

    return result


def apply_vignetting_correction(
    img: 'np.ndarray',
    vig_grid: 'np.ndarray',
) -> 'np.ndarray':
    """
    Apply factory vignetting (flat-field) correction to a raw image.

    Parameters
    ----------
    img : np.ndarray
        Raw debayered image, uint16 or float32 (H, W, 3).
    vig_grid : np.ndarray
        Factory vignetting grid (H_grid, W_grid) float32.
        Values are gain multipliers: multiply by these to flatten response.

    Returns
    -------
    np.ndarray
        float32 (H, W, 3) corrected image.
    """
    import numpy as np
    import cv2

    H, W = img.shape[:2]
    # Bilinear interpolate the small grid (e.g. 13×17) to full image size
    vig_full = cv2.resize(vig_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32)
    # Apply as per-pixel gain (same correction for all 3 channels)
    corrected = img_f * vig_full[:, :, None]
    return corrected


def apply_cra_correction(
    img: 'np.ndarray',
    cra_grid: 'np.ndarray',
) -> 'np.ndarray':
    """
    Apply factory CRA (Chief Ray Angle) correction to a debayered BGR image.

    The CRA grid corrects for cross-channel color leakage caused by off-axis
    light rays hitting the Bayer sensor at an angle.  The factory stores a
    17×13 grid of 4×4 mixing matrices in (R, Gr, Gb, B) channel order.  We
    convert each cell to a 3×3 RGB matrix by averaging the two green rows/cols,
    then bilinearly interpolate the grid to full image resolution and apply it
    as a spatially-varying local color-correction.

    Correction is negligible at the image center (~0.3% maximum deviation) and
    grows to ~5% at the corners — exactly the expected CRA roll-off profile.

    Parameters
    ----------
    img : np.ndarray
        Debayered BGR image, float32 (H, W, 3).  Vignetting should already
        be applied before calling this.
    cra_grid : np.ndarray
        Factory CRA grid (H_grid=13, W_grid=17, 4, 4) float32, from
        extract_sensor_calibration()['modules'][cam_name]['cra'].

    Returns
    -------
    np.ndarray
        float32 (H, W, 3) CRA-corrected BGR image.
    """
    import numpy as np
    import cv2

    H, W = img.shape[:2]
    GH, GW = cra_grid.shape[:2]  # 13, 17

    # Convert 4×4 (R, Gr, Gb, B) matrix to 3×3 (R, G, B).
    # Post-demosaic G ≈ (Gr + Gb) / 2, so Gr ≈ Gb ≈ G.
    # Derivation (output = M4 @ [R, Gr, Gb, B]^T, substituting Gr=Gb=G):
    #
    #   R_out  = M[0,0]*R + (M[0,1]+M[0,2])*G + M[0,3]*B
    #   G_out  = ((M[1,0]+M[2,0])/2)*R + ((M[1,1]+M[1,2]+M[2,1]+M[2,2])/2)*G
    #                                   + ((M[1,3]+M[2,3])/2)*B
    #   B_out  = M[3,0]*R + (M[3,1]+M[3,2])*G + M[3,3]*B
    #
    # Key: R/B input G-column = SUM of Gr+Gb cols (not average);
    #      G output row        = AVERAGE of Gr+Gb rows.
    M4 = cra_grid  # (GH, GW, 4, 4)
    M3_rgb = np.empty((*M4.shape[:2], 3, 3), dtype=np.float32)

    # R output row
    M3_rgb[..., 0, 0] = M4[..., 0, 0]
    M3_rgb[..., 0, 1] = M4[..., 0, 1] + M4[..., 0, 2]          # sum Gr+Gb cols
    M3_rgb[..., 0, 2] = M4[..., 0, 3]

    # G output row (average of Gr row and Gb row)
    M3_rgb[..., 1, 0] = (M4[..., 1, 0] + M4[..., 2, 0]) * 0.5
    M3_rgb[..., 1, 1] = (M4[..., 1, 1] + M4[..., 1, 2]
                        + M4[..., 2, 1] + M4[..., 2, 2]) * 0.5
    M3_rgb[..., 1, 2] = (M4[..., 1, 3] + M4[..., 2, 3]) * 0.5

    # B output row
    M3_rgb[..., 2, 0] = M4[..., 3, 0]
    M3_rgb[..., 2, 1] = M4[..., 3, 1] + M4[..., 3, 2]          # sum Gr+Gb cols
    M3_rgb[..., 2, 2] = M4[..., 3, 3]

    # Upsample each of the 9 coefficient planes to full image size and apply.
    # Input/output channel order: img is BGR so:
    #   img channel 0 = B → RGB index 2
    #   img channel 1 = G → RGB index 1
    #   img channel 2 = R → RGB index 0
    out = np.zeros_like(img)
    for out_bgr, out_rgb in enumerate([2, 1, 0]):  # out channel BGR order
        for in_bgr, in_rgb in enumerate([2, 1, 0]):  # in  channel BGR order
            coeff = M3_rgb[:, :, out_rgb, in_rgb]   # (GH, GW) one coefficient plane
            coeff_full = cv2.resize(coeff, (W, H), interpolation=cv2.INTER_LINEAR)
            out[:, :, out_bgr] += coeff_full * img[:, :, in_bgr]

    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    lri_path   = sys.argv[1]
    outdir     = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') \
                 else os.path.splitext(lri_path)[0] + '_cal'
    json_only  = '--json-only' in sys.argv

    os.makedirs(outdir, exist_ok=True)

    print(f"Parsing: {lri_path}")
    result = parse_lri(lri_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    mods = result['modules']
    cal_count  = sum(1 for m in mods if m['calibration'] is not None)
    intr_count = sum(1 for m in mods
                     if m['calibration'] and m['calibration']['intrinsics'])
    extr_count = sum(1 for m in mods
                     if m['calibration'] and m['calibration']['rotation'])
    mono_count = sum(1 for m in mods if m['is_mono'])

    print(f"  Modules found    : {len(mods)}")
    print(f"  Color / Mono     : {len(mods)-mono_count} / {mono_count}")
    print(f"  With calibration : {cal_count}")
    print(f"  With intrinsics  : {intr_count}")
    print(f"  With extrinsics  : {extr_count}")
    print(f"  35mm focal length: {result['image_focal_length_mm']} mm")

    for mod in mods:
        cal = mod['calibration']
        intr = cal['intrinsics'] if cal else None
        status = (f"fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                  f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}") if intr else "no calibration"
        print(f"    {mod['camera_name']:>3} {'MONO' if mod['is_mono'] else 'RGB ':4} "
              f"{mod['width']:>5}×{mod['height']:<5} {status}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    json_path = os.path.join(outdir, 'calibration.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  calibration.json → {json_path}")

    info_path = write_frame_info(result, outdir)
    print(f"  frame_info.txt   → {info_path}")

    if not json_only:
        cam_path, img_path = write_colmap(result, outdir)
        print(f"  cameras.txt      → {cam_path}")
        print(f"  images.txt       → {img_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
