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
    """Load camera intrinsics + extrinsics (and exposure/WB metadata) from calibration.json.

    In addition to optical parameters (K, R, t), each camera entry now also
    carries radiometric metadata needed for white-balance and exposure
    normalization:

      ``analog_gain``   — ISP analogue gain factor at capture time (float).
      ``exposure_ns``   — Sensor integration time in nanoseconds (int).
      ``bayer_pattern`` — Bayer CFA layout: 0=RGGB, 1=GRBG, 2=GBRG, 3=BGGR,
                          -1=monochrome.

    These fields are present in calibration.json when it was produced by
    lri_calibration.py.  If absent (e.g. legacy JSON) the keys are omitted from
    the camera dict so callers can use .get() safely.
    """
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
        mt = cal.get('mirror_type', 'NONE')
        entry = dict(
            K=K,
            R=np.array(cal['rotation'],    dtype=np.float64) if cal.get('rotation') else None,
            t=np.array(cal['translation'], dtype=np.float64) if cal.get('translation') else None,
            W=mod['width'], H=mod['height'],
            mirror_type=mt,
            # MOVABLE virtual pose includes diag(-1,1,1) x-flip; compute_remap must
            # undo this to get physical sensor pixel coordinates.
            virt_mirror_x=(mt == 'MOVABLE'),
        )
        # Radiometric metadata — present when calibration.json was produced by
        # lri_calibration.py; silently absent for older JSON files.
        if 'analog_gain' in mod:
            entry['analog_gain'] = float(mod['analog_gain'])
        if 'exposure_ns' in mod:
            entry['exposure_ns'] = int(mod['exposure_ns'])
        if 'bayer_pattern' in mod:
            entry['bayer_pattern'] = int(mod['bayer_pattern'])
        cameras[name] = entry
    return cameras


# ── reference camera selection ───────────────────────────────────────────────

def select_reference(cameras: dict, frames_dir: str) -> str:
    """
    Pick the reference camera from first principles — not by name.

    Rules (in order):
      1. Only consider direct-fire cameras (mirror_type='NONE') — widest FOV,
         no mirror, identity rotation, reliable calibration.
      2. Among those, pick the one geometrically closest to the centroid of
         all direct-fire camera positions.  This minimises the average warp
         distance to every other wide camera, reducing interpolation error.
      3. If t is missing for any camera, fall back to comparing |t| from origin.
      4. Frame must actually exist on disk.

    Returns the camera name (e.g. 'A1').
    """
    import glob as _glob

    wide = {n: c for n, c in cameras.items()
            if c.get('mirror_type', 'NONE') == 'NONE'
            and c.get('t') is not None
            and os.path.exists(os.path.join(frames_dir, f'{n}.png'))}

    if not wide:
        # Fallback: first camera with an existing frame
        for n in cameras:
            if os.path.exists(os.path.join(frames_dir, f'{n}.png')):
                return n
        raise RuntimeError('No camera frames found in ' + frames_dir)

    positions = np.array([c['t'] for c in wide.values()])   # (N, 3)
    centroid  = positions.mean(axis=0)
    ref_name  = min(wide, key=lambda n: np.linalg.norm(wide[n]['t'] - centroid))
    return ref_name


def load_depth_map(lumen_dir: str, ref_name: str,
                   target_hw: tuple) -> np.ndarray | None:
    """
    Load the best available depth map for the reference camera.

    Priority:
      1. depth/mvs_a_cameras.npz       — PatchMatch MVS (metric, virtual A-cam frame)
      2. fused_*cams_median_depth.png  — multi-camera median fusion (uint16, mm)
      3. depth/<ref_name>.npz          — single-camera Depth Pro output (float32, m)

    MVS depth is in the virtual A-camera frame (centroid of A1-A5).  For B-camera
    depth_reproject_warp this is close enough to A1's frame — the A-group centroid
    is within ~35mm of A1, sub-pixel parallax at any scene depth > 5m.

    Returns float32 (H, W) depth in METRES, or None if nothing found.
    target_hw : (H, W) to resize depth map to if shape differs.
    """
    import glob as _glob

    H_ref, W_ref = target_hw

    # Priority 1: PatchMatch MVS depth (metric, geometrically consistent)
    mvs_path = os.path.join(lumen_dir, 'depth', 'mvs_a_cameras.npz')
    if os.path.exists(mvs_path):
        d = np.load(mvs_path)['depth'].astype(np.float32)
        if d.shape != (H_ref, W_ref):
            d = cv2.resize(d, (W_ref, H_ref), interpolation=cv2.INTER_LINEAR)
        valid = d > 0
        print(f"  Depth map: depth/mvs_a_cameras.npz "
              f"({d[valid].min():.1f}–{d[valid].max():.1f}m, "
              f"{100*valid.mean():.0f}% valid)")
        return d

    # Priority 2: fused multi-camera depth PNG (mm uint16)
    fused_candidates = sorted(_glob.glob(
        os.path.join(lumen_dir, 'fused_*cams_median_depth.png')))
    if fused_candidates:
        path = fused_candidates[-1]   # highest cam count if multiple
        d = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if d is not None:
            d = d.astype(np.float32) / 1000.0     # mm → metres
            if d.shape != (H_ref, W_ref):
                d = cv2.resize(d, (W_ref, H_ref), interpolation=cv2.INTER_LINEAR)
            print(f"  Depth map: {os.path.basename(path)} "
                  f"({d[d>0].min():.1f}–{d[d>0].max():.1f}m, "
                  f"{100*(d>0).mean():.0f}% valid)")
            return d

    # Priority 3: single-camera Depth Pro .npz
    npz_path = os.path.join(lumen_dir, 'depth', f'{ref_name}.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path)['depth'].astype(np.float32)
        if d.shape != (H_ref, W_ref):
            d = cv2.resize(d, (W_ref, H_ref), interpolation=cv2.INTER_LINEAR)
        print(f"  Depth map: depth/{ref_name}.npz "
              f"({d[d>0].min():.1f}–{d[d>0].max():.1f}m)")
        return d

    print('  No depth map found — homography fallback for telephoto cameras')
    return None


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


# ── coverage mask ────────────────────────────────────────────────────────────

def coverage_mask(H: np.ndarray,
                  src_hw: tuple, dst_hw: tuple,
                  feather_px: int = 60) -> np.ndarray:
    """
    Return float32 (dst_H, dst_W) in [0,1] indicating which dst pixels have
    valid content when warping src → dst through homography H.

    Warps a fully-ones src mask through H.  Pixels that fall outside the src
    sensor bounds become 0; the boundary is softened with a Gaussian feather
    so blending is smooth rather than a hard cut.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    src_ones = np.ones((src_h, src_w), dtype=np.float32)
    dst_mask = cv2.warpPerspective(src_ones, H, (dst_w, dst_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0.0)
    if feather_px > 0:
        k = feather_px * 2 + 1
        dst_mask = cv2.GaussianBlur(dst_mask, (k, k), feather_px / 3.0)
        dst_mask = np.clip(dst_mask, 0.0, 1.0)
    return dst_mask


# ── consistency weight ───────────────────────────────────────────────────────

def consistency_weight(ref: np.ndarray, warped: np.ndarray,
                       sigma: float = 5000.0) -> np.ndarray:
    """
    Per-pixel weight: how well does `warped` match `ref`?

    Returns float32 (H, W) in [0, 1].  Near 1 where warped ≈ ref (good
    alignment), near 0 where they differ (flow failure / overexposed ceiling /
    misregistration).  When a pixel's weight is low the accumulator falls back
    to the A1 reference which already has full weight there.

    sigma controls the tolerance: 5000 ≈ 7.6% of the 16-bit range.  A telephoto
    camera warped via depth map typically has ~3000-3500 mean diff from sub-pixel
    depth errors (≈5% of 16-bit); sigma=5000 gives consist≈0.50 for those.

    Overexposure suppression: pixels where ref is near-saturated (>85% of
    65535) get progressively zeroed out.  Overexposed glass/sky regions have
    no reliable gradient for RAFT to track, so any secondary-camera
    contribution there is likely misaligned.  Falling back to A1 alone is
    cleaner than a ghost blend of badly-matched frames.
    """
    # ── alignment consistency ─────────────────────────────────────────────────
    # Zero-out diff where warped has no content (black-filled outside coverage).
    # Without this, the huge diff at zero-padding edges bleeds through the
    # Gaussian blur into the coverage area, making consistency look near-zero
    # even when the alignment inside coverage is actually good.
    valid_warp = (warped.astype(np.float32).max(axis=2) > 0).astype(np.float32)
    diff = np.abs(warped.astype(np.float32) - ref.astype(np.float32)).mean(axis=2)
    diff = diff * valid_warp   # zero outside coverage before blurring
    diff = cv2.GaussianBlur(diff, (31, 31), 10.0)
    w = np.exp(-diff / sigma)
    # Also zero consistency outside coverage so the gate tests only real content
    w = w * valid_warp

    # ── overexposure mask ─────────────────────────────────────────────────────
    # ramp: 1.0 at ≤85% sat, 0.0 at ≥95% sat  (linear ramp over 10% window)
    ref_f  = ref.astype(np.float32)
    ref_mx = ref_f.max(axis=2) / 65535.0          # brightest channel per pixel
    ovr    = np.clip((ref_mx - 0.85) / 0.10, 0.0, 1.0)   # 0..1 overexposure
    sat_mask = 1.0 - ovr                           # 1=good, 0=blown-out

    return (w * sat_mask).astype(np.float32)


# ── sharpness weight ─────────────────────────────────────────────────────────

def laplacian_pyramid_blend(frames: list, weights: list, levels: int = 6,
                            first_levels: list | None = None) -> np.ndarray:
    """
    Multi-scale Laplacian pyramid blend of N aligned frames.

    frames       : list of float32 (H, W, 3) arrays in [0..65535] range
    weights      : list of float32 (H, W) weight maps (raw sharpness×mask×consistency)
    levels       : number of pyramid levels (6 = ~64px resolution at base)
    first_levels : optional list of ints (one per frame).  Frame i only
                   contributes at pyramid level >= first_levels[i].
                   Use first_levels[i] = COARSE_FIRST (=3) for secondary wide
                   cameras: they provide colour/exposure at low frequencies but
                   their sub-pixel warp error blurs fine detail.  Fine levels
                   (0, 1, 2) fall through to the reference camera only.

    Returns float32 (H, W, 3) blended image.
    """
    N = len(frames)
    assert N == len(weights) and N > 0
    if first_levels is None:
        first_levels = [0] * N

    H, W = frames[0].shape[:2]

    # ── Build per-frame Laplacian pyramids + normalised weight pyramids ────────
    lp_pyramids = []   # lp_pyramids[i][l] = Laplacian level l for frame i
    gp_weights  = []   # gp_weights[i][l]  = Gaussian weight level l for frame i

    for i in range(N):
        # Gaussian pyramid for the image
        gp_img = [frames[i].astype(np.float32)]
        for _ in range(levels - 1):
            gp_img.append(cv2.pyrDown(gp_img[-1]))

        # Laplacian pyramid: difference between consecutive Gaussian levels
        lp = []
        for l in range(levels - 1):
            up = cv2.pyrUp(gp_img[l + 1], dstsize=(gp_img[l].shape[1], gp_img[l].shape[0]))
            lp.append(gp_img[l] - up)
        lp.append(gp_img[-1])  # coarsest level is kept as-is
        lp_pyramids.append(lp)

        # Gaussian pyramid for the weight map
        w = weights[i].astype(np.float32)
        gp_w = [w]
        for _ in range(levels - 1):
            gp_w.append(cv2.pyrDown(gp_w[-1]))
        gp_weights.append(gp_w)

    # ── Blend each pyramid level ───────────────────────────────────────────────
    blended_pyr = []
    for l in range(levels):
        # Stack weight maps at this level; zero out frames that don't contribute
        # at this pyramid level (first_levels gating for secondary wide cameras).
        w_stack = np.stack([
            gp_weights[i][l] if l >= first_levels[i]
            else np.zeros_like(gp_weights[i][l])
            for i in range(N)
        ], axis=0)   # (N, H_l, W_l)
        w_sum   = w_stack.sum(axis=0, keepdims=True).clip(min=1e-6)
        w_norm  = w_stack / w_sum                                            # (N, H_l, W_l)

        # Weighted blend of Laplacian coefficients at this level
        blended = np.zeros_like(lp_pyramids[0][l])
        for i in range(N):
            blended += lp_pyramids[i][l] * w_norm[i, ..., np.newaxis]
        blended_pyr.append(blended)

    # ── Collapse the blended pyramid back to full resolution ─────────────────
    result = blended_pyr[-1]
    for l in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyr[l].shape[1], blended_pyr[l].shape[0]))
        result = result + blended_pyr[l]

    return result.clip(0, 65535).astype(np.float32)


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


def _img_to_raft_tensor(img_float32: np.ndarray,
                        scale: float | None = None) -> tuple:
    """
    float32 (H, W, 3) [0..65535] → (tensor (1, 3, H, W) [0..255], scale_factor).
    RAFT expects float32 images in [0, 255], trained on normally-exposed photos.
    Raw L16 frames are very dark (mean ~5000/65535 = 7.6% brightness), so a naive
    /257 rescale gives mean ~19/255 — too dark for reliable optical flow.
    Stretch to bring the 99th-percentile brightness to ~200/255.
    Pass scale from the reference image when converting the source so both images
    use the same linear stretch — RAFT needs relative brightness preserved.
    Returns (tensor, scale_used).
    """
    arr = img_float32 / 257.0
    if scale is None:
        p99 = float(np.percentile(arr, 99))
        scale = 200.0 / max(p99, 1.0)
    arr = np.clip(arr * scale, 0.0, 255.0)
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return t, scale


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
    t_ref, scale = _img_to_raft_tensor(ref_s)           # compute scale from ref
    t_src, _     = _img_to_raft_tensor(src_s, scale)    # apply same scale to src

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


def depth_reproject_warp(ref: np.ndarray, src: np.ndarray,
                        cam_a1: dict, cam_b: dict,
                        depth_map: np.ndarray) -> tuple:
    """
    Warp src (B-camera frame) into A1 reference space using a depth map.

    Instead of a planar homography, each A1 pixel is unprojected to 3D using
    its measured depth, transformed to the B-camera frame via R/t, and
    reprojected onto the B image plane.  This correctly handles parallax for
    non-planar scenes.

    ref       : float32 (H, W, 3) A1 reference frame (used for shape only)
    src       : float32 (H, W, 3) B-camera frame at native resolution
    cam_a1    : dict with key 'K' — 3x3 intrinsic for A1 (R/t assumed identity)
    cam_b     : dict with keys 'K' (3x3 intrinsic), 'R' (3x3 rotation),
                't' (3,) translation — world->B-cam extrinsics
    depth_map : float32 (H, W) depth in metres, co-registered with ref

    Returns (warped, coverage) float32 arrays of shape (H, W, 3) and (H, W).
    """
    H, W = ref.shape[:2]
    H_src, W_src = src.shape[:2]

    K_A = cam_a1['K']
    K_B = cam_b['K']
    R   = cam_b['R']
    # Calibration t is in millimetres; depth_map is in metres — convert to metres
    t   = cam_b['t'].reshape(3) / 1000.0

    # Build pixel coordinate grid for A1
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64),
                             np.arange(H, dtype=np.float64))
    ones = np.ones((H, W), dtype=np.float64)

    # Unproject A1 pixels to 3D using depth
    K_A_inv = np.linalg.inv(K_A)
    pts_h   = np.stack([cols, rows, ones], axis=-1)          # (H, W, 3)
    pts_cam = (K_A_inv @ pts_h.reshape(-1, 3).T).T.reshape(H, W, 3)
    pts_3d  = pts_cam * depth_map[:, :, np.newaxis]          # (H, W, 3)

    # Transform to B-camera frame:  X_B = R @ X_A1 + t
    pts_b = (R @ pts_3d.reshape(-1, 3).T + t.reshape(3, 1)).T.reshape(H, W, 3)

    # Project onto B image plane
    z_b       = pts_b[:, :, 2:3].clip(min=1e-6)
    pts_b_nrm = pts_b / z_b                                  # (H, W, 3)
    pts_b_px  = (K_B @ pts_b_nrm.reshape(-1, 3).T).T.reshape(H, W, 3)

    map_x = pts_b_px[:, :, 0].astype(np.float32)
    map_y = pts_b_px[:, :, 1].astype(np.float32)

    # Sample B-frame at the computed coordinates
    warped = cv2.remap(src, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Coverage mask: valid where projected coords land inside B-frame bounds
    # and depth is positive
    valid = ((map_x >= 0) & (map_x < W_src) &
             (map_y >= 0) & (map_y < H_src) &
             (depth_map > 0))
    cov = valid.astype(np.float32)
    # Feather edges for smooth blending (matches coverage_mask feather style)
    cov = cv2.GaussianBlur(cov, (0, 0), sigmaX=30).clip(0, 1)

    return warped.astype(np.float32), cov.astype(np.float32)


def _homography_maps_within(H: np.ndarray, src_hw: tuple, dst_hw: tuple,
                            margin: float = 0.5) -> bool:
    """
    Return True if H maps the center of src to within the dst frame
    (±margin × dst_dimension tolerance).

    Used to detect when a calibration homography has gone wildly off-screen,
    which happens for MOVABLE mirror B cameras whose rotation matrices don't
    directly feed a simple rotation-only homography.
    """
    sh, sw = src_hw
    dh, dw = dst_hw
    pt = H @ np.array([sw / 2.0, sh / 2.0, 1.0])
    if abs(pt[2]) < 1e-8:
        return False
    x, y = pt[0] / pt[2], pt[1] / pt[2]
    return ((-margin * dw) < x < ((1 + margin) * dw) and
            (-margin * dh) < y < ((1 + margin) * dh))


def align_b_camera(ref: np.ndarray, src: np.ndarray,
                   cam_ref: dict, cam_src: dict,
                   depth_map: np.ndarray | None = None) -> tuple:
    """
    Align a B-group src frame to A1 ref using:
      1. Calibration homography (initial estimate) — used only when it maps
         src pixels into a sensible region of ref (i.e., GLUED mirror cameras).
         For MOVABLE mirror cameras the calibration rotation maps completely
         off-screen, so we skip the pre-warp and go straight to step 2.
      2. SuperPoint + LightGlue feature matching on the (possibly pre-warped)
         src image.  When the calibration pre-warp is skipped, LightGlue
         matches directly from raw src space to ref space and returns the full
         homography.
      3. Perspective warp to A1 canvas.

    If depth_map is provided (float32 H×W, metres, co-registered with ref),
    uses depth_reproject_warp() instead for a true 3D warp that correctly
    handles parallax from the physical B-camera baseline.

    Returns (warped, mask) float32.
    mask is a coverage mask: 1.0 where src sensor pixels land in ref space,
    0.0 outside (feathered).  This prevents B-camera content from bleeding
    into the wide-angle corners/ceiling where it has no valid data.
    """
    if depth_map is not None:
        return depth_reproject_warp(ref, src, cam_ref, cam_src, depth_map)

    H_ref, W_ref = ref.shape[:2]
    H_src, W_src = src.shape[:2]

    # ── Step 1: calibration homography ───────────────────────────────────────
    H_cal = homography_from_calibration(cam_ref, cam_src)

    # Check whether H_cal actually maps src content into ref's neighbourhood.
    # GLUED mirror B cameras (e.g. B4) have near-identity rotations → valid.
    # MOVABLE mirror B cameras have large optical-axis offsets → H_cal maps
    # the src center thousands of pixels outside ref → useless pre-warp.
    cal_valid = (H_cal is not None and
                 _homography_maps_within(H_cal, (H_src, W_src), (H_ref, W_ref),
                                         margin=0.5))

    # ── Step 2 + 3: align via LightGlue ──────────────────────────────────────
    try:
        if cal_valid:
            # Pre-warp with calibration, then refine residual with LightGlue
            src_pre = cv2.warpPerspective(src, H_cal, (W_ref, H_ref),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            H_total = H_cal.copy()
            H_refined = _lightglue_refine(ref, src_pre)
            if H_refined is not None:
                warped  = cv2.warpPerspective(src_pre, H_refined, (W_ref, H_ref),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                H_total = H_refined @ H_total
            else:
                print("    (LightGlue returned no matches, using cal-only warp)")
                warped = src_pre
        else:
            # Calibration pre-warp is invalid (MOVABLE mirror camera).
            # The B camera has a much higher focal length than A1 (typically 2-3×),
            # so direct full-resolution matching fails because features appear at
            # very different scales.  Strategy:
            #   1. Try LightGlue on raw src (works if scale difference is small).
            #   2. If that fails, resize src to A1's effective scale (fx_ref/fx_src)
            #      so both images show the same feature sizes.  The homography
            #      H_scaled (scaled_src → ref) is then adjusted back to raw src
            #      space: H_total = H_scaled @ S  where S = diag(scale, scale, 1).
            reason = "no cal" if H_cal is None else "cal maps off-screen"
            print(f"    ({reason} — LightGlue on raw src)", end=' ', flush=True)
            H_lg = _lightglue_refine(ref, src)
            if H_lg is not None:
                warped  = cv2.warpPerspective(src, H_lg, (W_ref, H_ref),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                H_total = H_lg
            else:
                # Raw-src match failed — try scale-normalised matching.
                # Rescale src to approximately A1's resolution of the same scene area.
                fx_ref = cam_ref['K'][0, 0] if cam_ref is not None else None
                fx_src = cam_src['K'][0, 0] if cam_src is not None else None
                scale  = (fx_ref / fx_src) if (fx_ref and fx_src and fx_src > fx_ref) else None
                if scale is not None and scale < 0.95:
                    sw_scaled = max(64, int(W_src * scale))
                    sh_scaled = max(64, int(H_src * scale))
                    src_scaled = cv2.resize(src, (sw_scaled, sh_scaled),
                                            interpolation=cv2.INTER_AREA)
                    print(f"    (scale-norm {scale:.3f} → {sw_scaled}×{sh_scaled})", end=' ', flush=True)
                    H_scaled = _lightglue_refine(ref, src_scaled)
                    if H_scaled is not None:
                        # Map: raw_src →[S]→ scaled_src →[H_scaled]→ ref
                        S = np.diag([scale, scale, 1.0])
                        H_total = H_scaled @ S
                        warped  = cv2.warpPerspective(src, H_total, (W_ref, H_ref),
                                                      flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=0)
                    else:
                        print("    LightGlue found no matches — camera skipped")
                        blank = np.zeros_like(ref, dtype=np.float32)
                        return blank, np.zeros((H_ref, W_ref), dtype=np.float32)
                else:
                    print("    LightGlue found no matches — camera skipped")
                    blank = np.zeros_like(ref, dtype=np.float32)
                    return blank, np.zeros((H_ref, W_ref), dtype=np.float32)

    except Exception as e:
        print(f"    LightGlue failed ({e}), skipping camera")
        blank = np.zeros_like(ref, dtype=np.float32)
        return blank, np.zeros((H_ref, W_ref), dtype=np.float32)

    # Coverage mask: which output pixels came from valid src sensor coordinates.
    cov = coverage_mask(H_total, (H_src, W_src), (H_ref, W_ref), feather_px=150)
    return warped, cov


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


def estimate_b_to_ref_homography(
    ref: np.ndarray,
    src: np.ndarray,
    cam_ref: dict | None = None,
    cam_src: dict | None = None,
) -> np.ndarray | None:
    """
    Estimate 3×3 homography mapping src (B-camera) pixels → ref (A1) pixels.

    Uses SuperPoint + LightGlue.  When the B camera has a longer focal length
    than the A1 reference (typical: fx_b ≈ 2–3× fx_a1) a scale-normalised
    fallback is attempted: src is resized to approximately A1's angular scale
    before matching, and the homography is adjusted back to raw-src space.

    Parameters
    ----------
    ref, src   : float32 (H, W, 3) in [0..65535]
    cam_ref    : camera dict with key 'K' (3×3 intrinsic) for the ref camera
    cam_src    : camera dict with key 'K' (3×3 intrinsic) for the src camera

    Returns
    -------
    3×3 float64 homography (src_pixel → ref_pixel), or None on failure.
    """
    # Try direct full-resolution match
    H = _lightglue_refine(ref, src)
    if H is not None:
        return H

    # Scale-normalised fallback — resize src so scene features appear the same
    # angular size as in ref.  Needed when fx_src ≫ fx_ref.
    fx_ref = cam_ref['K'][0, 0] if cam_ref is not None else None
    fx_src = cam_src['K'][0, 0] if cam_src is not None else None
    scale = (fx_ref / fx_src) if (fx_ref and fx_src and fx_src > fx_ref) else None
    if scale is not None and scale < 0.95:
        H_h, W_w = src.shape[:2]
        sw_s = max(64, int(W_w * scale))
        sh_s = max(64, int(H_h * scale))

        # For very high zoom cameras (C cameras, scale ≈ 0.18), downscaling src to
        # ~762×571 px is too small for reliable LightGlue matching.  Instead, crop
        # the corresponding central region of ref and upscale it to src resolution,
        # then match at full resolution.  The homography is mapped back to full-ref
        # coordinates after matching.
        if sw_s < 1024 and cam_ref is not None:
            H_ref_img, W_ref_img = ref.shape[:2]
            cx_ref = cam_ref['K'][0, 2]
            cy_ref = cam_ref['K'][1, 2]
            # Crop size in ref pixels = src_size * scale, plus 30% padding
            pad_w = max(60, int(sw_s * 0.30))
            pad_h = max(60, int(sh_s * 0.30))
            crop_w = sw_s + 2 * pad_w
            crop_h = sh_s + 2 * pad_h
            u0 = max(0, int(cx_ref - crop_w / 2))
            v0 = max(0, int(cy_ref - crop_h / 2))
            u1 = min(W_ref_img, u0 + crop_w)
            v1 = min(H_ref_img, v0 + crop_h)
            ref_crop = ref[v0:v1, u0:u1]
            # Upscale ref crop to src resolution for matched-scale matching
            ref_crop_up = cv2.resize(ref_crop, (W_w, H_h), interpolation=cv2.INTER_LINEAR)
            H_src_to_up = _lightglue_refine(ref_crop_up, src)
            if H_src_to_up is not None:
                # Convert crop_upscaled coords → full ref coords:
                #   crop_up → crop_original: multiply by ((u1-u0)/W_w, (v1-v0)/H_h)
                #   crop_original → full ref: add (u0, v0)
                sx = (u1 - u0) / W_w
                sy = (v1 - v0) / H_h
                S_down = np.array([[sx, 0., 0.], [0., sy, 0.], [0., 0., 1.]])
                T_off  = np.array([[1., 0., u0], [0., 1., v0], [0., 0., 1.]])
                return T_off @ S_down @ H_src_to_up

        src_scaled = cv2.resize(src, (sw_s, sh_s), interpolation=cv2.INTER_AREA)
        H_scaled = _lightglue_refine(ref, src_scaled)
        if H_scaled is not None:
            # Map: raw_src →[S]→ scaled_src →[H_scaled]→ ref
            S = np.diag([scale, scale, 1.0])
            return H_scaled @ S

    return None


# ── radiometric normalization ────────────────────────────────────────────────

def radiometric_normalize(src: np.ndarray, ref: np.ndarray,
                          coverage: np.ndarray) -> np.ndarray:
    """
    Per-channel WLS gain+bias normalization of src to match ref.
    Weight = 1/(|grad_ref| + 1e-3) to exclude edges from estimation.
    src, ref: float32 (H,W,3) in [0..65535]
    coverage: float32 (H,W) mask of valid overlap pixels
    Returns: normalized src, float32 (H,W,3)
    """
    result = src.copy()
    # Gradient magnitude of ref (grayscale)
    ref_gray = cv2.cvtColor((ref / 65535.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    w = (1.0 / (grad_mag + 1e-3)) * (coverage > 0).astype(np.float32)
    w_flat = w.ravel()

    for c in range(3):
        s = src[:, :, c].ravel().astype(np.float64)
        r = ref[:, :, c].ravel().astype(np.float64)
        wf = w_flat.astype(np.float64)
        # WLS: minimize sum w*(g*s + b - r)^2
        # Normal equations: [[sum(w*s^2), sum(w*s)], [sum(w*s), sum(w)]] * [g,b] = [sum(w*s*r), sum(w*r)]
        ws2 = np.sum(wf * s * s)
        ws  = np.sum(wf * s)
        ww  = np.sum(wf)
        wsr = np.sum(wf * s * r)
        wr  = np.sum(wf * r)
        if ww < 100:  # insufficient overlap
            continue
        A = np.array([[ws2, ws], [ws, ww]])
        b_vec = np.array([wsr, wr])
        try:
            g, b = np.linalg.solve(A, b_vec)
            # Sanity clamp: gain in valid range; bias limited to ±20% of 16-bit range
            # (A cameras with 3.5× exposure difference can have large negative bias if
            # saturated highlights skew the regression — clamping bias prevents dark-level clipping)
            g = float(np.clip(g, 0.05, 20.0))
            b = float(np.clip(b, -13107.0, 13107.0))   # ±20% of 65535
            result[:, :, c] = np.clip(g * src[:, :, c] + b, 0, 65535)
        except np.linalg.LinAlgError:
            pass  # singular — skip normalization for this channel
    return result


# ── color correction matrix (B-group only) ───────────────────────────────────

def estimate_ccm(src: np.ndarray, ref: np.ndarray,
                 coverage: np.ndarray) -> np.ndarray | None:
    """
    Estimate 3x3 CCM mapping src RGB -> ref RGB in overlap region.
    src, ref: float32 (H,W,3)
    coverage: float32 (H,W) overlap mask
    Returns 3x3 float64 CCM, or None if insufficient overlap.
    """
    mask = coverage > 0.5
    if mask.sum() < 500:
        return None
    s = src[mask].astype(np.float64)   # (N, 3)
    r = ref[mask].astype(np.float64)   # (N, 3)
    # Solve s @ CCM ≈ r  →  CCM = lstsq(s, r)
    ccm, _, _, _ = np.linalg.lstsq(s, r, rcond=None)
    return ccm  # shape (3, 3)


def apply_ccm(src: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply 3x3 CCM to float32 (H,W,3) image. Returns float32 (H,W,3)."""
    H, W = src.shape[:2]
    result = (src.reshape(-1, 3).astype(np.float64) @ ccm).reshape(H, W, 3)
    return result.clip(0, 65535).astype(np.float32)


# ── core fusion ──────────────────────────────────────────────────────────────

# Finest pyramid level at which secondary wide cameras contribute.
# Levels 0-(COARSE_FIRST-1) are fine detail — secondary wide cameras are
# excluded because sub-pixel warp error blurs them.  Levels COARSE_FIRST+ are
# colour/exposure — all cameras contribute.
COARSE_FIRST = 3


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

    # ── Select reference camera from geometry, not label ─────────────────────
    ref_name = select_reference(cameras, frames_dir)
    ref_path = os.path.join(frames_dir, f'{ref_name}.png')
    ref = load_frame_uint16(ref_path)
    H_ref, W_ref = ref.shape[:2]
    cam_ref = cameras[ref_name]
    print(f"  Reference: {ref_name}  ({W_ref}×{H_ref}  mirror={cam_ref['mirror_type']})")

    # ── Load depth map ────────────────────────────────────────────────────────
    lumen_dir  = os.path.dirname(frames_dir.rstrip('/'))
    depth_map  = load_depth_map(lumen_dir, ref_name, (H_ref, W_ref))

    ref_sharp     = sharpness_weight(ref).astype(np.float32)
    frames_list   = [ref.astype(np.float32)]
    weights_list  = [ref_sharp]
    coverage_list = [np.ones((H_ref, W_ref), dtype=np.float32)]
    first_levels  = [0]   # reference contributes at all pyramid levels

    # ── Group cameras by optical type ─────────────────────────────────────────
    # wide        = direct-fire, no mirror → near-coplanar, reliable calibration
    # tele_fixed  = glued periscope mirror → reliable calibration R,t
    # tele_movable= movable periscope mirror → R computed from hall code at shot
    #               time via compute_movable_mirror_pose() in lri_calibration.py;
    #               det(R)=+1 confirmed, treat same as GLUED for depth warping
    wide_cams         = [n for n, c in cameras.items()
                         if c.get('mirror_type') == 'NONE' and n != ref_name]
    tele_fixed_cams   = [n for n, c in cameras.items()
                         if c.get('mirror_type') == 'GLUED']
    tele_movable_cams = [n for n, c in cameras.items()
                         if c.get('mirror_type') == 'MOVABLE']

    print(f"  Wide (secondary):    {sorted(wide_cams)}")
    print(f"  Tele GLUED (3D warp):{sorted(tele_fixed_cams)}")
    print(f"  Tele MOVABLE (hall R):{sorted(tele_movable_cams)}")

    # ── Wide cameras (secondary): coplanar, near-identity rotation ───────────
    # LightGlue alignment (exposure-invariant).  Secondary wide cameras blend
    # at coarse pyramid levels only (first_level=COARSE_FIRST) — their fine-
    # scale detail is identical to the reference but shifted by sub-pixel warp
    # error, causing blur if blended at full resolution.
    print(f"\n--- Wide cameras (secondary, coarse-blend only above level {COARSE_FIRST}) ---")
    for cam in sorted(wide_cams):
        path = os.path.join(frames_dir, f'{cam}.png')
        if not os.path.exists(path):
            print(f"  {cam}: missing, skip")
            continue
        print(f"  {cam} ...", end=' ', flush=True)
        src = load_frame_uint16(path)
        if (src >= 65530).mean() > 0.40:
            print(f"skip (>40% saturated)")
            continue
        cam_x = cameras[cam]
        warped, mask = align_b_camera(ref, src, cam_ref, cam_x)
        warped   = radiometric_normalize(warped, ref.astype(np.float32), mask)
        consist  = consistency_weight(ref, warped).astype(np.float32)
        cov_px   = mask > 0.5
        if cov_px.sum() > 0 and consist[cov_px].mean() < 0.40:
            print(f"skip (consist={consist[cov_px].mean():.2f} < 0.40)")
            continue
        sharp = sharpness_weight(warped).astype(np.float32)
        w     = sharp * mask * consist
        frames_list.append(warped)
        weights_list.append(w)
        coverage_list.append((mask * consist).clip(0, 1))
        first_levels.append(COARSE_FIRST)   # coarse-only blend
        print(f"ok (consist={consist[cov_px].mean():.2f})")

    # ── Telephoto cameras ─────────────────────────────────────────────────────
    # GLUED mirror: calibration R from factory → 3D-warp via depth map when available.
    # MOVABLE mirror: calibration R computed from hall code at shot time via
    #   compute_movable_mirror_pose() → det(R)=+1, same treatment as GLUED.
    # Both are gated on sharpness and consistency.
    def _fuse_tele(cam, use_depth):
        path = os.path.join(frames_dir, f'{cam}.png')
        if not os.path.exists(path):
            print(f"  {cam}: missing, skip")
            return
        print(f"  {cam} ({'3D-warp' if use_depth else 'homog'}) ...", end=' ', flush=True)
        src   = load_frame_uint16(path)
        cam_b = cameras[cam]
        dm    = depth_map if use_depth else None
        warped, mask = align_b_camera(ref, src, cam_ref, cam_b, depth_map=dm)
        warped = radiometric_normalize(warped, ref.astype(np.float32), mask)
        ccm    = estimate_ccm(warped, ref.astype(np.float32), mask)
        if ccm is not None:
            warped = apply_ccm(warped, ccm)
        sharp   = sharpness_weight(warped).astype(np.float32)
        consist = consistency_weight(ref, warped).astype(np.float32)
        cov_px  = mask > 0.5
        if cov_px.sum() == 0:
            print("skip (no coverage)")
            return
        # Consistency gate
        mean_consist = consist[cov_px].mean()
        if mean_consist < 0.40:
            print(f"skip (consist={mean_consist:.2f} < 0.40)")
            return
        # Sharpness gate:
        # - 3D-warp path (use_depth=True, GLUED and MOVABLE w/ depth map):
        #   depth_reproject_warp downsamples the tele camera by (fx_tele/fx_wide)
        #   into the reference frame, so warped image always appears blurrier.
        #   Accept if not extremely blurry (ratio ≥ 0.10).
        # - Homography path (use_depth=False, no depth map available):
        #   telephoto must be ≥ 0.95× reference sharpness — same resolution
        #   class as the tele reference, strict gate filters out-of-focus frames.
        ref_sharp_cov = ref_sharp[cov_px].mean()
        cam_sharp_cov = sharp[cov_px].mean()
        sharp_ratio   = cam_sharp_cov / max(ref_sharp_cov, 1e-9)
        if use_depth:
            # 3D-warp path: accept if not extremely blurry (ratio ≥ 0.10)
            if sharp_ratio < 0.10:
                print(f"skip (sharp_ratio={sharp_ratio:.2f} — too blurry even for coarse blend)")
                return
        else:
            # Homography path: strict gate — same-resolution class camera
            if sharp_ratio < 0.95:
                print(f"skip (sharp_ratio={sharp_ratio:.2f} — blurrier than reference)")
                return
        # Without depth map, block movable-mirror cameras covering the upper
        # frame where parallax artifacts are worst
        if dm is None:
            ys = np.where(mask > 0.3)[0]
            if len(ys) > 0 and ys.min() / H_ref < 0.40:
                print(f"skip (coverage top={ys.min()/H_ref:.2f} — upper parallax, no depth)")
                return
        w = sharp * mask * consist
        frames_list.append(warped)
        weights_list.append(w)
        coverage_list.append((mask * consist).clip(0, 1))
        # Both GLUED and MOVABLE telephoto cameras contribute at coarse pyramid
        # levels only (≥ COARSE_FIRST = level 3, i.e. 8px+ features).
        # depth_reproject_warp downsamples the tele into the wide reference frame,
        # so warped tele detail is always coarser than the reference — fine-level
        # blending adds blur rather than sharpness.  Coarse-level contribution
        # still improves colour, tone, and coverage at large scales.
        first_levels.append(COARSE_FIRST)
        print(f"ok (consist={mean_consist:.2f}, sharp_ratio={sharp_ratio:.2f})")

    print("\n--- Telephoto GLUED (3D warp via depth map) ---")
    for cam in sorted(tele_fixed_cams):
        _fuse_tele(cam, use_depth=(depth_map is not None))

    print("\n--- Telephoto MOVABLE (hall-code R — same path as GLUED) ---")
    for cam in sorted(tele_movable_cams):
        _fuse_tele(cam, use_depth=(depth_map is not None))

    # ── Smooth-fill secondary frames with reference before pyramid blend ──────
    # Zero-padded regions in secondary frames create hard edges that spike into
    # coarse pyramid levels as visible seams.  Fill with reference content
    # outside coverage — the blend weight is still gated by coverage so the
    # fill region contributes near-zero weight.
    ref_f32 = frames_list[0]
    for i in range(1, len(frames_list)):
        cov_3c = coverage_list[i][:, :, np.newaxis]
        frames_list[i] = frames_list[i] * cov_3c + ref_f32 * (1.0 - cov_3c)

    # ── Laplacian pyramid blend ───────────────────────────────────────────────
    print(f"\nBlending {len(frames_list)} frames with Laplacian pyramid ...", flush=True)
    fused = laplacian_pyramid_blend(frames_list, weights_list,
                                    first_levels=first_levels)

    weight_sum = sum(w for w in weights_list)
    no_data = (weight_sum == 0)
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
