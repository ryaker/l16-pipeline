#!/usr/bin/env python3
"""
lri_fuse_image.py — Multi-camera image fusion for the Light L16.

Aligns and fuses all 10 camera frames into a single 16-bit linear PNG
(fused_image_16bit.png) at A1 resolution (4160×3120).

Algorithm:
  A-group (A1–A5): RAFT optical flow on Apple MPS
    - Same ~35mm focal length, small baselines → dense sub-pixel flow
    - More accurate than depth-based warping, no depth errors
  B-group (B1–B5): calibration homography + LightGlue feature refinement
    - Different focal lengths (70–150mm), large baselines → flow unreliable
    - Initial H from factory K/R/t → refine with SuperPoint+LightGlue features
  Fusion: per-pixel sharpness-weighted mean over all aligned frames

Usage:
    python3 lri_fuse_image.py <frames_dir> <calibration.json> [--output OUTPUT]

    frames_dir      Directory with A1.png … B5.png (from lri_extract.py)
    calibration.json  Factory calibration (from lri_calibration.py)
    --output        Output path [default: fused_image_16bit.png in frames_dir parent]
    --no-cache      Re-compute even if fused_image_16bit.png already exists
"""

import argparse
import json
import os
import sys
import warnings

import cv2
import numpy as np
import torch
import torchvision.models.optical_flow as tof
from PIL import Image as PILImage

# Suppress verbose warnings from RAFT / torchvision
warnings.filterwarnings('ignore', category=UserWarning)

# ── device ───────────────────────────────────────────────────────────────────

def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = _device()


# ── calibration ──────────────────────────────────────────────────────────────

def load_cameras(cal_path: str) -> dict:
    """Load camera intrinsics + extrinsics from calibration.json."""
    data = json.load(open(cal_path))
    cameras = {}
    for mod in data['modules']:
        name = mod['camera_name']
        cal  = mod.get('calibration')
        if not cal or 'intrinsics' not in cal or cal['intrinsics'] is None:
            continue
        intr = cal['intrinsics']
        K = np.array([[intr['fx'], 0,          intr['cx']],
                      [0,          intr['fy'],  intr['cy']],
                      [0,          0,           1        ]], dtype=np.float64)
        cameras[name] = dict(
            K=K,
            R=np.array(cal['rotation'],    dtype=np.float64) if cal.get('rotation') else None,
            t=np.array(cal['translation'], dtype=np.float64) if cal.get('translation') else None,
            W=mod['width'], H=mod['height'],
        )
    return cameras


# ── image I/O ────────────────────────────────────────────────────────────────

def load_frame_uint16(path: str) -> np.ndarray:
    """Load PNG → float32 in [0, 65535]. Supports 8-bit and 16-bit PNGs."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        return img.astype(np.float32) * 257.0   # 0–255 → 0–65535
    return img.astype(np.float32)               # already uint16-range


def save_16bit_png(path: str, img: np.ndarray) -> None:
    """Save float32 RGB [0, 65535] → 16-bit PNG via cv2."""
    arr = np.clip(img, 0, 65535).astype(np.uint16)
    # cv2 uses BGR; convert from RGB
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


# ── sharpness weight ─────────────────────────────────────────────────────────

def sharpness_weight(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Per-pixel sharpness weight = Laplacian response magnitude, normalised.

    img : float32 (H, W, 3) in any range
    Returns float32 (H, W) in [0, 1].
    """
    mx = img.max()
    grey = cv2.cvtColor(((img / mx * 255) if mx > 0 else img).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap  = np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=ksize))
    # Smooth so single-pixel noise doesn't dominate
    lap  = cv2.GaussianBlur(lap, (ksize, ksize), 0)
    mx   = lap.max()
    if mx > 0:
        lap /= mx
    return lap


# ── RAFT optical flow (A-group) ──────────────────────────────────────────────

_raft_model = None

def _get_raft():
    global _raft_model
    if _raft_model is None:
        print("  Loading RAFT-Large on", DEVICE, "...", flush=True)
        weights = tof.Raft_Large_Weights.DEFAULT
        _raft_model = tof.raft_large(weights=weights).to(DEVICE).eval()
    return _raft_model


def _img_to_raft_tensor(img_float32: np.ndarray) -> torch.Tensor:
    """
    float32 (H, W, 3) [0..65535] → float32 tensor (1, 3, H, W) [0..255] for RAFT.
    RAFT expects float32 images in [0, 255].
    """
    arr = (img_float32 / 257.0).clip(0.0, 255.0).astype(np.float32)
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return t


def warp_flow(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Apply optical flow field to warp src image.

    src  : float32 (H, W, 3)
    flow : float32 (H, W, 2) — flow[y, x] = (dx, dy) displacement at (x, y)
    Returns warped float32 (H, W, 3), zero-filled outside.
    """
    H, W = src.shape[:2]
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


RAFT_SCALE = 4   # Downsample factor before running RAFT (memory guard)


def align_raft(ref: np.ndarray, src: np.ndarray) -> tuple:
    """
    Align src to ref using RAFT optical flow.

    ref, src : float32 (H, W, 3) [0..65535]
    RAFT runs at 1/RAFT_SCALE resolution to fit in GPU memory;
    the resulting flow is bilinearly upsampled back to full resolution.
    Returns (warped, mask) where mask is float32 (H, W) in [0, 1].
    """
    H, W = ref.shape[:2]
    # RAFT requires dimensions divisible by 8
    H_s  = (H // RAFT_SCALE // 8) * 8
    W_s  = (W // RAFT_SCALE // 8) * 8

    ref_s = cv2.resize(ref, (W_s, H_s), interpolation=cv2.INTER_AREA)
    src_s = cv2.resize(src, (W_s, H_s), interpolation=cv2.INTER_AREA)

    raft  = _get_raft()
    t_ref = _img_to_raft_tensor(ref_s)
    t_src = _img_to_raft_tensor(src_s)

    with torch.no_grad():
        flow_list = raft(t_ref, t_src)
    flow_s = flow_list[-1][0].permute(1, 2, 0).cpu().numpy()  # (H_s, W_s, 2)

    # Upscale flow and rescale magnitudes to match full resolution
    flow = cv2.resize(flow_s, (W, H), interpolation=cv2.INTER_LINEAR) * RAFT_SCALE

    warped = warp_flow(src, flow)
    mask   = (warped.sum(axis=2) > 0).astype(np.float32)
    return warped, mask


# ── homography alignment (B-group) ───────────────────────────────────────────

def homography_from_calibration(cam_ref: dict, cam_src: dict) -> np.ndarray:
    """
    Compute planar homography from calibration (assuming a far-away scene).
    H maps src pixel → ref pixel coordinates.
    Uses the rotation component only (valid for distant scene / planar approx).
    """
    K1 = cam_ref['K']
    K2 = cam_src['K']
    R1 = cam_ref['R']   # world→cam1
    R2 = cam_src['R']   # world→cam2

    if R1 is None or R2 is None:
        return None

    # Relative rotation: cam2 → cam1
    R_rel = R1 @ R2.T
    H = K1 @ R_rel @ np.linalg.inv(K2)
    H /= H[2, 2]
    return H


def align_b_camera(ref: np.ndarray, src: np.ndarray,
                   cam_ref: dict, cam_src: dict) -> tuple:
    """
    Align a B-group src frame to A1 ref using:
      1. Calibration homography (initial estimate)
      2. SuperPoint + LightGlue feature refinement (improves accuracy)
      3. Perspective warp to A1 canvas

    Returns (warped, mask) float32.
    """
    H, W = ref.shape[:2]

    # ── Step 1: calibration homography ───────────────────────────────────────
    H_cal = homography_from_calibration(cam_ref, cam_src)

    # ── Step 2: warp src to ref space with initial homography ────────────────
    if H_cal is not None:
        src_pre = cv2.warpPerspective(src, H_cal, (W, H),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        # No calibration — try direct feature matching (fallback)
        src_pre = src.copy()
        H_cal   = np.eye(3)

    # ── Step 3: LightGlue refinement ─────────────────────────────────────────
    try:
        H_refined = _lightglue_refine(ref, src_pre)
        if H_refined is not None:
            warped = cv2.warpPerspective(src_pre, H_refined, (W, H),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            warped = src_pre
    except Exception as e:
        print(f"    LightGlue failed ({e}), using calibration-only warp")
        warped = src_pre

    mask = (warped.sum(axis=2) > 0).astype(np.float32)
    return warped, mask


_lg_extractor = None
_lg_matcher   = None

def _get_lightglue():
    global _lg_extractor, _lg_matcher
    if _lg_extractor is None:
        from lightglue import LightGlue, SuperPoint
        print("  Loading SuperPoint+LightGlue on", DEVICE, "...", flush=True)
        _lg_extractor = SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
        _lg_matcher   = LightGlue(features='superpoint').eval().to(DEVICE)
    return _lg_extractor, _lg_matcher


def _lightglue_refine(ref: np.ndarray, src: np.ndarray) -> np.ndarray | None:
    """
    Estimate a refinement homography between pre-warped src and ref
    using SuperPoint + LightGlue.

    Returns 3×3 homography or None if not enough matches.
    """
    from lightglue.utils import load_image, rbd

    extractor, matcher = _get_lightglue()

    def to_lg_tensor(img_f32):
        # LightGlue expects float32 (1, 3, H, W) in [0, 1]
        arr = (img_f32 / 65535.0).clip(0, 1).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    t_ref = to_lg_tensor(ref)
    t_src = to_lg_tensor(src)

    with torch.no_grad():
        feat_ref = extractor.extract(t_ref)
        feat_src = extractor.extract(t_src)
        matches  = matcher({'image0': feat_ref, 'image1': feat_src})

    # Remove batch dimension
    feat_ref, feat_src, matches = [rbd(x) for x in [feat_ref, feat_src, matches]]

    kp_ref = feat_ref['keypoints'].cpu().numpy()   # (N, 2)
    kp_src = feat_src['keypoints'].cpu().numpy()   # (M, 2)
    m      = matches['matches'].cpu().numpy()       # (K, 2) index pairs

    if len(m) < 8:
        return None

    pts_ref = kp_ref[m[:, 0]]
    pts_src = kp_src[m[:, 1]]

    H_ref, inliers = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    if H_ref is None or (inliers is not None and inliers.sum() < 8):
        return None

    return H_ref


# ── core fusion ──────────────────────────────────────────────────────────────

A_CAMS = ['A1', 'A2', 'A3', 'A4', 'A5']
B_CAMS = ['B1', 'B2', 'B3', 'B4', 'B5']


def fuse_frames(frames_dir: str, cal_path: str, output_path: str) -> str:
    """
    Main fusion entry point.

    frames_dir  : directory with A1.png … B5.png
    cal_path    : calibration.json from lri_calibration.py
    output_path : where to write fused_image_16bit.png
    Returns output_path.
    """
    print(f"\n=== Multi-camera fusion ===")
    print(f"  frames : {frames_dir}")
    print(f"  cal    : {cal_path}")
    print(f"  output : {output_path}")
    print(f"  device : {DEVICE}\n")

    cameras = load_cameras(cal_path)

    # ── Load A1 reference ─────────────────────────────────────────────────────
    a1_path = os.path.join(frames_dir, 'A1.png')
    ref = load_frame_uint16(a1_path)
    H_ref, W_ref = ref.shape[:2]
    print(f"  A1 reference: {W_ref}×{H_ref}")

    accum  = np.zeros((H_ref, W_ref, 3), dtype=np.float64)
    weight = np.zeros((H_ref, W_ref),    dtype=np.float64)

    # Seed with A1 itself
    sharp_a1 = sharpness_weight(ref).astype(np.float64)
    accum  += ref.astype(np.float64) * sharp_a1[..., np.newaxis]
    weight += sharp_a1

    # ── A-group: RAFT flow alignment ─────────────────────────────────────────
    print("\n--- A-group (RAFT optical flow) ---")
    for cam in ['A2', 'A3', 'A4', 'A5']:
        path = os.path.join(frames_dir, f'{cam}.png')
        if not os.path.exists(path):
            print(f"  {cam}: missing, skip")
            continue
        print(f"  {cam} ...", end=' ', flush=True)
        src     = load_frame_uint16(path)
        warped, mask = align_raft(ref, src)
        sharp        = sharpness_weight(warped).astype(np.float64)
        w            = sharp * mask.astype(np.float64)
        accum  += warped.astype(np.float64) * w[..., np.newaxis]
        weight += w
        print("done")

    # ── B-group: calibration + LightGlue alignment ───────────────────────────
    print("\n--- B-group (homography + LightGlue) ---")
    cam_a1 = cameras.get('A1')
    for cam in B_CAMS:
        path = os.path.join(frames_dir, f'{cam}.png')
        if not os.path.exists(path):
            print(f"  {cam}: missing, skip")
            continue
        if cam not in cameras:
            print(f"  {cam}: no calibration, skip")
            continue
        print(f"  {cam} ...", end=' ', flush=True)
        src = load_frame_uint16(path)
        cam_b = cameras[cam]
        warped, mask = align_b_camera(ref, src, cam_a1, cam_b)
        sharp        = sharpness_weight(warped).astype(np.float64)
        w            = sharp * mask.astype(np.float64)
        accum  += warped.astype(np.float64) * w[..., np.newaxis]
        weight += w
        print("done")

    # ── Weighted average ──────────────────────────────────────────────────────
    print("\nMerging ...", flush=True)
    safe_w = np.where(weight > 0, weight, 1.0)
    fused  = (accum / safe_w[..., np.newaxis]).clip(0, 65535).astype(np.float32)

    # Where no camera contributed, fall back to A1
    no_data = (weight == 0)
    fused[no_data] = ref[no_data]

    save_16bit_png(output_path, fused)
    print(f"Fused image → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-camera image fusion for the Light L16.'
    )
    parser.add_argument('frames_dir',      help='Directory with A1.png … B5.png')
    parser.add_argument('calibration_json', help='calibration.json from lri_calibration.py')
    parser.add_argument('--output', default=None,
                        help='Output path (default: fused_image_16bit.png in frames_dir parent)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Re-run even if output already exists')
    args = parser.parse_args()

    if args.output:
        out_path = args.output
    else:
        parent   = os.path.dirname(os.path.normpath(args.frames_dir))
        out_path = os.path.join(parent, 'fused_image_16bit.png')

    if not args.no_cache and os.path.exists(out_path):
        print(f"Already exists: {out_path}  (use --no-cache to re-run)")
        return

    fuse_frames(args.frames_dir, args.calibration_json, out_path)


if __name__ == '__main__':
    main()
