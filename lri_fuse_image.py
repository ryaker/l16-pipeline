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
                       sigma: float = 2000.0) -> np.ndarray:
    """
    Per-pixel weight: how well does `warped` match `ref`?

    Returns float32 (H, W) in [0, 1].  Near 1 where warped ≈ ref (good
    alignment), near 0 where they differ (flow failure / overexposed ceiling /
    misregistration).  When a pixel's weight is low the accumulator falls back
    to the A1 reference which already has full weight there.

    sigma controls the tolerance: 2000 ≈ 3% of the 16-bit range (strict).

    Overexposure suppression: pixels where ref is near-saturated (>85% of
    65535) get progressively zeroed out.  Overexposed glass/sky regions have
    no reliable gradient for RAFT to track, so any secondary-camera
    contribution there is likely misaligned.  Falling back to A1 alone is
    cleaner than a ghost blend of badly-matched frames.
    """
    # ── alignment consistency ─────────────────────────────────────────────────
    diff = np.abs(warped.astype(np.float32) - ref.astype(np.float32)).mean(axis=2)
    diff = cv2.GaussianBlur(diff, (31, 31), 10.0)
    w = np.exp(-diff / sigma)

    # ── overexposure mask ─────────────────────────────────────────────────────
    # ramp: 1.0 at ≤85% sat, 0.0 at ≥95% sat  (linear ramp over 10% window)
    ref_f  = ref.astype(np.float32)
    ref_mx = ref_f.max(axis=2) / 65535.0          # brightest channel per pixel
    ovr    = np.clip((ref_mx - 0.85) / 0.10, 0.0, 1.0)   # 0..1 overexposure
    sat_mask = 1.0 - ovr                           # 1=good, 0=blown-out

    return (w * sat_mask).astype(np.float32)


# ── sharpness weight ─────────────────────────────────────────────────────────

def laplacian_pyramid_blend(frames: list, weights: list, levels: int = 6) -> np.ndarray:
    """
    Multi-scale Laplacian pyramid blend of N aligned frames.

    frames  : list of float32 (H, W, 3) arrays in [0..65535] range
    weights : list of float32 (H, W) weight maps (not normalised — raw sharpness×mask×consistency)
    levels  : number of pyramid levels (6 = ~64px resolution at base)

    Returns float32 (H, W, 3) blended image.

    Why this fixes halos vs weighted-average:
      Weighted average mixes pixel VALUES across all frames at once.  At a depth
      edge, a sharp-foreground pixel from cam A and a blurred-background pixel
      from cam B get averaged — producing a half-sharp, half-blurred ghost.
      Laplacian pyramid blending separates low-frequency content (base colour,
      gradual tone) from high-frequency content (edges, texture) and blends them
      independently at each spatial scale.  The edge detail from the sharpest
      camera dominates at fine scales; colour and brightness average naturally
      at coarse scales.  No halos.
    """
    N = len(frames)
    assert N == len(weights) and N > 0

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
        # Stack weight maps at this level and normalise so they sum to 1
        w_stack = np.stack([gp_weights[i][l] for i in range(N)], axis=0)   # (N, H_l, W_l)  # gp_weights[i][l] — level-matched weight pyramid (correct)
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
    t   = cam_b['t'].reshape(3)

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


def align_b_camera(ref: np.ndarray, src: np.ndarray,
                   cam_ref: dict, cam_src: dict,
                   depth_map: np.ndarray | None = None) -> tuple:
    """
    Align a B-group src frame to A1 ref using:
      1. Calibration homography (initial estimate)
      2. SuperPoint + LightGlue feature refinement (improves accuracy)
      3. Perspective warp to A1 canvas

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

    # ── Step 2: warp src to ref space with initial homography ────────────────
    if H_cal is not None:
        src_pre = cv2.warpPerspective(src, H_cal, (W_ref, H_ref),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        H_total = H_cal.copy()
    else:
        src_pre = src.copy()
        H_total = np.eye(3, dtype=np.float64)

    # ── Step 3: LightGlue refinement ─────────────────────────────────────────
    try:
        H_refined = _lightglue_refine(ref, src_pre)
        if H_refined is not None:
            warped  = cv2.warpPerspective(src_pre, H_refined, (W_ref, H_ref),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            H_total = H_refined @ H_total
        else:
            warped = src_pre
    except Exception as e:
        print(f"    LightGlue failed ({e}), using calibration-only warp")
        warped = src_pre

    # Coverage mask: which output pixels came from valid src sensor coordinates.
    # Warping a src-shaped ones-mask through H_total gives exactly this region.
    # A telephoto B-camera only covers the center of A1; the mask goes to zero
    # outside that region, preventing ceiling/corner artifacts.
    cov = coverage_mask(H_total, (H_src, W_src), (H_ref, W_ref), feather_px=60)
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
            # Sanity clamp: gain should be near 1.0
            g = float(np.clip(g, 0.5, 2.0))
            b = float(np.clip(b, -5000, 5000))
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

    frames_list  = [ref.astype(np.float32)]
    weights_list = [sharpness_weight(ref).astype(np.float32)]

    # ── A-group: RAFT flow alignment ─────────────────────────────────────────
    print("\n--- A-group (RAFT optical flow) ---")
    for cam in ['A2', 'A3', 'A4', 'A5']:
        path = os.path.join(frames_dir, f'{cam}.png')
        if not os.path.exists(path):
            print(f"  {cam}: missing, skip")
            continue
        print(f"  {cam} ...", end=' ', flush=True)
        src          = load_frame_uint16(path)
        warped, mask = align_raft(ref, src)
        warped = radiometric_normalize(warped.astype(np.float32), ref.astype(np.float32), mask)
        sharp        = sharpness_weight(warped).astype(np.float32)
        consist      = consistency_weight(ref, warped).astype(np.float32)
        w            = sharp * mask.astype(np.float32) * consist
        frames_list.append(warped.astype(np.float32))
        weights_list.append(w)
        print("done")

    # ── B-group: depth-map reprojection (or homography fallback) ────────────
    # Try to load A1 depth map for B-camera reprojection.
    # Expected path: <lumen_dir>/depth/A1.npz  (produced by Depth Pro).
    depth_map_path = os.path.join(os.path.dirname(frames_dir.rstrip('/')),
                                  'depth', 'A1.npz')
    depth_map_a1 = None
    if os.path.exists(depth_map_path):
        depth_map_a1 = np.load(depth_map_path)['depth'].astype(np.float32)
        # Resize to match ref frame if needed
        if depth_map_a1.shape != (H_ref, W_ref):
            depth_map_a1 = cv2.resize(depth_map_a1, (W_ref, H_ref),
                                      interpolation=cv2.INTER_LINEAR)
        print(f"  Depth map loaded: {depth_map_path}")
    else:
        print(f"  No depth map found, using homography fallback for B-group")

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
        src   = load_frame_uint16(path)
        cam_b = cameras[cam]
        warped, mask = align_b_camera(ref, src, cam_a1, cam_b,
                                      depth_map=depth_map_a1)
        warped = radiometric_normalize(warped.astype(np.float32), ref.astype(np.float32), mask)
        # B-group CCM: correct spectral shift from periscope mirrors
        ccm = estimate_ccm(warped, ref.astype(np.float32), mask)
        if ccm is not None:
            warped = apply_ccm(warped, ccm)
        sharp        = sharpness_weight(warped).astype(np.float32)
        consist      = consistency_weight(ref, warped).astype(np.float32)
        w            = sharp * mask.astype(np.float32) * consist
        frames_list.append(warped.astype(np.float32))
        weights_list.append(w)
        print("done")

    # ── Laplacian pyramid blend ───────────────────────────────────────────────
    print(f"\nBlending {len(frames_list)} frames with Laplacian pyramid ...", flush=True)
    fused = laplacian_pyramid_blend(frames_list, weights_list)

    # Where no camera contributed any weight, fall back to A1
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
