#!/usr/bin/env python3
"""
lri_mvs_depth.py — PatchMatch Multi-View Stereo for L16 A-cameras.

Uses all 5 A-cameras simultaneously on Apple M4 MPS (or CPU).
Produces a metric depth map in the virtual camera frame.

Usage:
    python lri_mvs_depth.py <frames_dir> <calibration_json> [options]

Options:
    --cameras A1 A2 A3 A4 A5   cameras to use (default: all A-group)
    --output-dir DIR            output directory (default: frames_dir/depth/)
    --iterations N              PatchMatch iterations (default: 3)
    --patch-size N              NCC patch half-width (default: 3 → 7×7)
    --scale N                   downsample factor for speed (default: 8)
    --depth-min M               minimum depth in metres
    --depth-max M               maximum depth in metres
    --device cpu|mps            compute device (default: mps)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Add project root to path so we can import load_cameras ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lri_fuse_image import load_cameras  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────
A_CAMERAS = ('A1', 'A2', 'A3', 'A4', 'A5')
_WIDE_FX_THRESHOLD = 5000.0   # A-cameras have fx≈3372; B-cameras fx≈8276


# ── sRGB gamma ────────────────────────────────────────────────────────────────
def apply_srgb_gamma(img_float: np.ndarray) -> np.ndarray:
    """Convert linear [0,1] float to sRGB gamma-encoded [0,1] float."""
    mask = img_float <= 0.0031308
    out = np.where(
        mask,
        12.92 * img_float,
        1.055 * (np.clip(img_float, 1e-10, None) ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(out, 0.0, 1.0)


# ── Image loading ─────────────────────────────────────────────────────────────
def load_image_float(path: str) -> np.ndarray:
    """
    Load PNG → float32 (H, W, 3) in [0, 1] with sRGB gamma applied.

    Handles both uint8 (8-bit) and uint16 (16-bit) PNGs.
    sRGB gamma is applied because NCC is computed on appearance, and
    perceptual encoding is more uniform for patch matching.
    """
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img)
    if arr.dtype == np.uint16:
        f = arr.astype(np.float32) / 65535.0
    else:
        f = arr.astype(np.float32) / 255.0
    return apply_srgb_gamma(f)   # (H, W, 3) float32, [0,1]


# ── Virtual camera ────────────────────────────────────────────────────────────
def build_virtual_camera(cameras: dict, cam_names: list[str]) -> dict:
    """
    Build the virtual centroid camera from the selected A-cameras.

    Returns a dict with keys: K (3×3), R (3×3), t (3,), W, H — all numpy.
    K and output size are computed from the A-camera FOV union at A-camera
    pixel density (no B-cameras involved in the depth pass).
    R = SVD re-orthogonalised mean of A-camera rotations (Kabsch on SO(3)).
    t = arithmetic mean of A-camera translation vectors (mm).
    """
    sel = {n: cameras[n] for n in cam_names if n in cameras}
    if not sel:
        raise ValueError(f"None of {cam_names} found in calibration.")

    # ── Intrinsics: A-camera FOV union at A-camera pixel density ────────────
    half_fov_h_list, half_fov_v_list = [], []
    fx_list, fy_list = [], []
    for cam in sel.values():
        K = cam['K']
        W, H = cam['W'], cam['H']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        half_fov_h_list.append(max(np.arctan(cx / fx), np.arctan((W - cx) / fx)))
        half_fov_v_list.append(max(np.arctan(cy / fy), np.arctan((H - cy) / fy)))
        fx_list.append(fx)
        fy_list.append(fy)

    half_fov_h = max(half_fov_h_list)
    half_fov_v = max(half_fov_v_list)
    output_fx = float(np.median(fx_list))
    output_fy = float(np.median(fy_list))

    W_out = int(round(2.0 * np.tan(half_fov_h) * output_fx))
    H_out = int(round(2.0 * np.tan(half_fov_v) * output_fy))
    cx_out = W_out / 2.0
    cy_out = H_out / 2.0

    K_virt = np.array([
        [output_fx, 0.0,       cx_out],
        [0.0,       output_fy, cy_out],
        [0.0,       0.0,       1.0  ],
    ], dtype=np.float64)

    # ── Rotation: SVD re-orthogonalised mean (Kabsch on SO(3)) ───────────────
    R_mats = [c['R'] for c in sel.values() if c.get('R') is not None]
    if not R_mats:
        R_virt = np.eye(3, dtype=np.float64)
    else:
        R_stack = np.stack(R_mats, axis=0)       # (N, 3, 3)
        R_mean = R_stack.mean(axis=0)            # (3, 3) — not yet in SO(3)
        U, _, Vt = np.linalg.svd(R_mean)
        R_virt = (U @ Vt).astype(np.float64)    # nearest rotation matrix

    # ── Translation: centroid of A-camera translations (mm) ──────────────────
    ts = [c['t'] for c in sel.values() if c.get('t') is not None]
    t_virt = np.mean(np.stack(ts, axis=0), axis=0).astype(np.float64) if ts else np.zeros(3)

    return dict(K=K_virt, R=R_virt, t=t_virt, W=W_out, H=H_out)


# ── Per-camera reprojection grid ──────────────────────────────────────────────
def build_reprojection_grid(
    depth: torch.Tensor,          # (H_v, W_v)
    K_virt_inv: torch.Tensor,     # (3, 3)
    R_virt: torch.Tensor,         # (3, 3)
    t_virt: torch.Tensor,         # (3,)
    K_src: torch.Tensor,          # (3, 3)
    R_src: torch.Tensor,          # (3, 3)
    t_src: torch.Tensor,          # (3,)
    H_src: int,
    W_src: int,
) -> torch.Tensor:
    """
    For every virtual camera pixel at given depth hypothesis, compute the
    normalised grid coordinates (in [-1, 1]) in source camera i.

    Returns: (1, H_v, W_v, 2) grid suitable for F.grid_sample.

    Projection convention (standard pinhole):
        p = K [R | t] X_world   →  X_world = R^T (K^{-1} p * d - t)
    """
    H_v, W_v = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Pixel grid for virtual camera (H_v, W_v)
    v_idx = torch.arange(H_v, device=device, dtype=dtype)   # rows
    u_idx = torch.arange(W_v, device=device, dtype=dtype)   # cols
    vv, uu = torch.meshgrid(v_idx, u_idx, indexing='ij')    # (H_v, W_v)

    # Homogeneous pixel coords → normalised camera rays in virtual frame
    # p_hom: (3, H_v*W_v)
    ones = torch.ones_like(uu)
    p_hom = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)  # (3, N)
    rays = K_virt_inv @ p_hom                                    # (3, N)

    # Scale rays by depth → 3D points in virtual camera frame
    d_flat = depth.reshape(1, -1)           # (1, N)
    pts_cam = rays * d_flat                 # (3, N)  X_cam = ray * depth

    # Transform to world: X_world = R_virt^T (X_cam - t_virt)
    # Note: if t_virt is the camera centre in world coords,
    # then: X_cam = R_virt @ X_world + t_virt
    #   →  X_world = R_virt^T @ (X_cam - t_virt)
    pts_world = R_virt.T @ (pts_cam - t_virt.unsqueeze(1))  # (3, N)

    # Project into source camera frame
    pts_src_cam = R_src @ pts_world + t_src.unsqueeze(1)    # (3, N) in src cam frame

    # Depth in source cam frame (z component before K)
    z_cam = pts_src_cam[2:3, :]                             # (1, N)
    valid = z_cam > 1e-3
    z_safe = torch.where(valid, z_cam, torch.ones_like(z_cam))

    # Perspective project to pixel coords: (u,v) = K @ (X/Z, Y/Z, 1)
    pts_src_px = K_src @ pts_src_cam                        # (3, N) — homogeneous pixel
    # Divide by z (which equals z_cam after K applied since K[2]=[0,0,1])
    uv_src = pts_src_px[:2, :] / z_safe                    # (2, N) src pixel coords

    # Convert pixel coords to normalised [-1, 1] for grid_sample
    # grid_sample convention: x = col (u), y = row (v)
    u_norm = (uv_src[0, :] / (W_src - 1)) * 2.0 - 1.0     # (N,)
    v_norm = (uv_src[1, :] / (H_src - 1)) * 2.0 - 1.0     # (N,)

    # Mark behind-camera as invalid (push to far outside [-1,1])
    invalid_mask = (~valid.squeeze(0))
    u_norm = torch.where(invalid_mask, torch.full_like(u_norm, 2.0), u_norm)
    v_norm = torch.where(invalid_mask, torch.full_like(v_norm, 2.0), v_norm)

    grid = torch.stack([u_norm, v_norm], dim=-1)            # (N, 2)
    grid = grid.reshape(1, H_v, W_v, 2)                     # (1, H_v, W_v, 2)
    return grid


# ── NCC patch cost ────────────────────────────────────────────────────────────
def compute_ncc_cost(
    img_virt: torch.Tensor,    # (1, C, H_v, W_v) — virtual camera image
    img_src: torch.Tensor,     # (1, C, H_s, W_s) — source camera image
    grid: torch.Tensor,        # (1, H_v, W_v, 2)
    patch_half: int,           # half-width of NCC patch
) -> torch.Tensor:
    """
    Compute per-pixel NCC cost between virtual camera patches and reprojected
    source camera patches.  Returns (H_v, W_v) cost in [0, 1]; 0=perfect match,
    1=worst.  Pixels where std < 1e-4 are penalised with cost=1.
    """
    device = img_virt.device
    dtype = img_virt.dtype
    _, C, H_v, W_v = img_virt.shape

    # Sample source image at reprojected coordinates
    # grid_sample bilinear, zero-padding for out-of-bounds
    sampled = F.grid_sample(
        img_src, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )  # (1, C, H_v, W_v)

    # Extract patches via unfold — works on CPU and MPS
    pw = 2 * patch_half + 1   # patch width

    # Pad then unfold to get (1, C, H_v, W_v, pw, pw) patch windows
    pad = patch_half
    virt_padded = F.pad(img_virt, (pad, pad, pad, pad), mode='reflect')
    src_padded = F.pad(sampled, (pad, pad, pad, pad), mode='reflect')

    # unfold: (1, C, H_v, W_v, pw*pw)
    def extract_patches(x):
        # x: (1, C, H, W) → (1, C, H_v, pw, W_v, pw) → (1, C, H_v, W_v, pw*pw)
        B, Cv, Hx, Wx = x.shape
        x_uf = x.unfold(2, pw, 1).unfold(3, pw, 1)  # (B, C, H_v, W_v, pw, pw)
        return x_uf.reshape(B, Cv, H_v, W_v, pw * pw)

    p_virt = extract_patches(virt_padded)   # (1, C, H_v, W_v, pw²)
    p_src  = extract_patches(src_padded)    # (1, C, H_v, W_v, pw²)

    # Flatten channel and patch dims → (1, C*pw², H_v, W_v)
    # then compute NCC across the combined axis
    p_v = p_virt.reshape(1, C * pw * pw, H_v, W_v)   # (1, P, H_v, W_v)
    p_s = p_src.reshape(1, C * pw * pw, H_v, W_v)

    # Zero-mean and normalise across P axis
    mean_v = p_v.mean(dim=1, keepdim=True)
    mean_s = p_s.mean(dim=1, keepdim=True)
    pv0 = p_v - mean_v
    ps0 = p_s - mean_s

    std_v = (pv0.pow(2).mean(dim=1)).sqrt()       # (1, H_v, W_v)
    std_s = (ps0.pow(2).mean(dim=1)).sqrt()       # (1, H_v, W_v)

    num = (pv0 * ps0).mean(dim=1)                 # (1, H_v, W_v)
    denom = std_v * std_s                         # (1, H_v, W_v)

    # Low-texture / out-of-bounds → cost=1
    valid = (denom > 1e-4).squeeze(0)             # (H_v, W_v)
    ncc = torch.zeros(H_v, W_v, device=device, dtype=dtype)
    ncc[valid] = (num.squeeze(0)[valid] / denom.squeeze(0)[valid]).clamp(-1.0, 1.0)

    # Convert NCC ∈ [-1,1] to cost ∈ [0,1]: cost = (1 - NCC) / 2
    cost = (1.0 - ncc) * 0.5
    # Pixels with low texture → maximum cost
    cost[~valid] = 1.0
    return cost  # (H_v, W_v)


# ── Aggregate NCC over all source cameras ────────────────────────────────────
def aggregate_cost(
    depth: torch.Tensor,          # (H_v, W_v)
    img_virt: torch.Tensor,       # (1, 3, H_v, W_v)
    src_imgs: list[torch.Tensor], # list of (1, 3, H_s, W_s)
    K_virt_inv: torch.Tensor,
    R_virt: torch.Tensor,
    t_virt: torch.Tensor,
    src_cams: list[dict],         # list of {K (scaled), R, t, W (scaled), H (scaled)}
    patch_half: int,
    scale: float = 1.0,           # unused — kept for call-site compatibility
) -> torch.Tensor:
    """Compute mean NCC cost over all source cameras. Returns (H_v, W_v)."""
    total_cost = torch.zeros_like(depth)
    n_valid = 0

    for img_s, cam in zip(src_imgs, src_cams):
        # cam['K'] is already scaled to the downsampled resolution by the caller
        K_s = torch.tensor(cam['K'], dtype=depth.dtype, device=depth.device)
        R_s = torch.tensor(cam['R'], dtype=depth.dtype, device=depth.device)
        t_s = torch.tensor(cam['t'], dtype=depth.dtype, device=depth.device)
        H_s, W_s = img_s.shape[2], img_s.shape[3]

        grid = build_reprojection_grid(
            depth, K_virt_inv, R_virt, t_virt,
            K_s, R_s, t_s, H_s, W_s,
        )
        cost = compute_ncc_cost(img_virt, img_s, grid, patch_half)
        total_cost += cost
        n_valid += 1

    if n_valid > 0:
        total_cost /= n_valid
    return total_cost


# ── PatchMatch ────────────────────────────────────────────────────────────────
def run_patchmatch(
    img_virt: torch.Tensor,       # (1, 3, H_v, W_v)
    src_imgs: list[torch.Tensor], # each (1, 3, H_s, W_s) — scaled
    K_virt: np.ndarray,           # 3×3 — at SCALED resolution
    R_virt: np.ndarray,           # 3×3
    t_virt: np.ndarray,           # (3,)
    src_cams_scaled: list[dict],  # calibration dicts with scaled K
    depth_min: float,
    depth_max: float,
    n_iterations: int,
    patch_half: int,
    scale: float,
    device: torch.device,
    depth_init: float | None = None,  # if given, initialise here instead of random
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PatchMatch MVS. Returns (depth, cost_map) at scaled resolution.
    """
    dtype = torch.float32
    _, _, H_v, W_v = img_virt.shape
    print(f"[mvs] virtual canvas: {W_v}×{H_v}  depth range: {depth_min/1000:.1f}–{depth_max/1000:.1f} m  ({depth_min:.0f}–{depth_max:.0f} mm)")

    K_virt_t = torch.tensor(K_virt, dtype=dtype, device=device)
    K_virt_inv = torch.linalg.inv(K_virt_t)
    R_virt_t = torch.tensor(R_virt, dtype=dtype, device=device)
    t_virt_t = torch.tensor(t_virt, dtype=dtype, device=device)

    # ── Stage 1: initialisation ──────────────────────────────────────────────
    if depth_init is not None:
        # Initialise at focus distance + small random perturbation (2% of range).
        # Far-field scenes have sub-pixel disparity; tight init prevents noise-driven
        # propagation from spreading incorrect depth hypotheses.
        print(f"[mvs] Stage 1: initialising at focus depth {depth_init/1000:.1f} m ± 2% range")
        jitter = (depth_max - depth_min) * 0.02
        rng = torch.Generator(device=device)
        noise = (torch.rand(H_v, W_v, generator=rng, device=device, dtype=dtype) * 2.0 - 1.0)
        depth = (depth_init + noise * jitter).clamp(depth_min, depth_max)
    else:
        print("[mvs] Stage 1: initialising random depth hypotheses...")
        rng = torch.Generator(device=device)
        depth = torch.rand(H_v, W_v, generator=rng, device=device, dtype=dtype)
        depth = depth * (depth_max - depth_min) + depth_min  # (H_v, W_v)

    # Fronto-parallel normals (z > 0); correct for far-field/planar scenes
    normals = torch.zeros(H_v, W_v, 3, device=device, dtype=dtype)
    normals[..., 2] = 1.0   # [0, 0, 1]

    # Evaluate initial cost
    cost = aggregate_cost(
        depth, img_virt, src_imgs,
        K_virt_inv, R_virt_t, t_virt_t,
        src_cams_scaled, patch_half, scale,
    )
    print(f"[mvs] initial cost: mean={cost.mean():.4f}  median={cost.median():.4f}")

    # ── Stage 2: PatchMatch iterations ───────────────────────────────────────
    # When initialized at focus distance, start delta at 10% of focus depth.
    # This prevents far-field noise from randomly walking depth far from the prior.
    if depth_init is not None:
        depth_delta = depth_init * 0.1   # 10% of focus depth
    else:
        depth_delta = (depth_max - depth_min) * 0.5

    for it in range(n_iterations):
        t0 = time.time()
        print(f"[mvs] Iteration {it + 1}/{n_iterations}  delta={depth_delta/1000:.2f} m")

        # Checkerboard propagation: red pass (even i+j), then black pass (odd)
        for parity in (0, 1):
            depth, cost = propagation_pass(
                depth, cost, img_virt, src_imgs,
                K_virt_inv, R_virt_t, t_virt_t,
                src_cams_scaled, patch_half, scale, parity,
            )

        # Random refinement
        depth, cost = random_refinement(
            depth, cost, img_virt, src_imgs,
            K_virt_inv, R_virt_t, t_virt_t,
            src_cams_scaled, patch_half, scale, depth_delta, depth_min, depth_max,
        )
        depth_delta *= 0.5
        print(f"[mvs] it{it+1} done in {time.time()-t0:.1f}s  cost mean={cost.mean():.4f}  "
              f"depth median={depth.median()/1000:.1f}m  std={depth.std()/1000:.2f}m")

    return depth, cost


def propagation_pass(
    depth: torch.Tensor, cost: torch.Tensor,
    img_virt, src_imgs, K_virt_inv, R_virt_t, t_virt_t,
    src_cams_scaled, patch_half, scale, parity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Checkerboard propagation: pixels of given parity adopt best neighbour depth.
    parity=0 → even pixels check up/left neighbours.
    parity=1 → odd pixels check down/right neighbours.
    """
    H_v, W_v = depth.shape
    device = depth.device

    if parity == 0:
        offsets = [(-1, 0), (0, -1)]   # up, left
    else:
        offsets = [(1, 0), (0, 1)]     # down, right

    # Checkerboard mask: (i+j) % 2 == parity
    row_idx = torch.arange(H_v, device=device)
    col_idx = torch.arange(W_v, device=device)
    vv, uu = torch.meshgrid(row_idx, col_idx, indexing='ij')
    mask = ((vv + uu) % 2 == parity)   # (H_v, W_v) bool

    best_depth = depth.clone()
    best_cost = cost.clone()

    for dr, dc in offsets:
        # Neighbour depth (with boundary clamping)
        neigh_depth = torch.roll(depth, shifts=(-dr, -dc), dims=(0, 1))
        # Clamp boundary (rolled values are invalid)
        if dr == -1:
            neigh_depth[0, :] = depth[0, :]
        elif dr == 1:
            neigh_depth[-1, :] = depth[-1, :]
        if dc == -1:
            neigh_depth[:, 0] = depth[:, 0]
        elif dc == 1:
            neigh_depth[:, -1] = depth[:, -1]

        # Only evaluate on checkerboard pixels to save compute
        hyp_depth = depth.clone()
        hyp_depth[mask] = neigh_depth[mask]

        hyp_cost = aggregate_cost(
            hyp_depth, img_virt, src_imgs,
            K_virt_inv, R_virt_t, t_virt_t,
            src_cams_scaled, patch_half, scale,
        )

        improve = mask & (hyp_cost < best_cost)
        best_depth[improve] = hyp_depth[improve]
        best_cost[improve] = hyp_cost[improve]

    return best_depth, best_cost


def random_refinement(
    depth: torch.Tensor, cost: torch.Tensor,
    img_virt, src_imgs, K_virt_inv, R_virt_t, t_virt_t,
    src_cams_scaled, patch_half, scale,
    delta: float, depth_min: float, depth_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb depth by ±delta, keep if cost improves."""
    H_v, W_v = depth.shape
    device = depth.device

    perturb = (torch.rand(H_v, W_v, device=device) * 2.0 - 1.0) * delta
    hyp = (depth + perturb).clamp(depth_min, depth_max)

    hyp_cost = aggregate_cost(
        hyp, img_virt, src_imgs,
        K_virt_inv, R_virt_t, t_virt_t,
        src_cams_scaled, patch_half, scale,
    )

    improve = hyp_cost < cost
    new_depth = torch.where(improve, hyp, depth)
    new_cost = torch.where(improve, hyp_cost, cost)
    return new_depth, new_cost


# ── Geometric consistency ─────────────────────────────────────────────────────
def geometric_consistency_mask(
    depth: torch.Tensor,          # (H_v, W_v)
    K_virt_inv: torch.Tensor,
    K_virt: torch.Tensor,
    R_virt: torch.Tensor,
    t_virt: torch.Tensor,
    src_cams_scaled: list[dict],
    threshold_px: float = 2.0,
    min_agree: int = 3,
) -> torch.Tensor:
    """
    For each pixel, reproject into each source camera and back to virtual.
    Count cameras that agree within threshold_px. Return bool mask.
    """
    H_v, W_v = depth.shape
    device = depth.device
    dtype = depth.dtype

    agree_count = torch.zeros(H_v, W_v, device=device, dtype=torch.int32)

    # Build pixel grid once
    v_idx = torch.arange(H_v, device=device, dtype=dtype)
    u_idx = torch.arange(W_v, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v_idx, u_idx, indexing='ij')

    for cam in src_cams_scaled:
        K_s = torch.tensor(cam['K'], dtype=dtype, device=device)
        R_s = torch.tensor(cam['R'], dtype=dtype, device=device)
        t_s = torch.tensor(cam['t'], dtype=dtype, device=device)
        H_s, W_s = cam['H'], cam['W']

        # Project virtual pixel → world → source
        ones = torch.ones_like(uu)
        p_hom = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)  # (3, N)
        rays = K_virt_inv @ p_hom                                   # (3, N)
        d_flat = depth.reshape(1, -1)
        pts_cam = rays * d_flat
        pts_world = R_virt.T @ (pts_cam - t_virt.unsqueeze(1))     # (3, N)

        # Forward: virtual → world → source cam frame
        pts_src_cam_f = R_s @ pts_world + t_s.unsqueeze(1)             # (3, N)
        z_src_cam = pts_src_cam_f[2:3, :]                              # (1, N)
        valid_z = (z_src_cam > 1e-3).squeeze(0)                        # (N,)
        z_src_safe = torch.where(z_src_cam > 1e-3, z_src_cam, torch.ones_like(z_src_cam))

        # src pixel coords
        pts_src_px = K_s @ pts_src_cam_f                               # (3, N)
        uv_src_px = pts_src_px[:2, :] / z_src_safe                    # (2, N)

        # Back-project: src pixel at z_src_cam depth → world → virtual
        # X_src_cam = (K_s^-1 @ [u,v,1]) * z_src_cam
        K_s_inv = torch.linalg.inv(K_s)
        uv_src_h = torch.cat([uv_src_px, torch.ones(1, H_v * W_v, device=device, dtype=dtype)], dim=0)
        rays_s = K_s_inv @ uv_src_h                                    # normalised dirs
        pts_s_cam2 = rays_s * z_src_safe                               # (3, N) in src cam
        pts_world2 = R_s.T @ (pts_s_cam2 - t_s.unsqueeze(1))          # (3, N) world

        # Project back into virtual camera
        pts_virt2 = R_virt @ pts_world2 + t_virt.unsqueeze(1)         # (3, N) virt cam
        z_v2 = pts_virt2[2:3, :]
        valid_zv = (z_v2 > 1e-3).squeeze(0)
        z_v2_safe = torch.where(z_v2 > 1e-3, z_v2, torch.ones_like(z_v2))
        uv_back_h = K_virt @ pts_virt2                                 # (3, N)
        uv_back = uv_back_h[:2, :] / z_v2_safe                        # (2, N) virt pixels

        # Compare reprojected pixel vs original pixel
        du = uv_back[0, :] - uu.reshape(-1)
        dv = uv_back[1, :] - vv.reshape(-1)
        dist = (du.pow(2) + dv.pow(2)).sqrt()                      # (N,)

        consistent = valid_z & valid_zv & (dist <= threshold_px)
        agree_count += consistent.reshape(H_v, W_v).to(torch.int32)

    return agree_count >= min_agree   # (H_v, W_v) bool


# ── Depth range from calibration ──────────────────────────────────────────────
def depth_range_from_calibration(cal_path: str, cam_names: list[str]) -> tuple[float, float, float | None]:
    """
    Extract per-shot focus distance from calibration.json.
    Returns (depth_min, depth_max, focus_distance_m) in metres.
    focus_distance_m may be None if not in calibration.
    """
    data = json.load(open(cal_path))
    focus_distances = []
    for mod in data['modules']:
        if mod['camera_name'] in cam_names:
            fd = mod.get('calibration', {}).get('focus_distance')
            if fd and fd > 0:
                focus_distances.append(float(fd))

    if focus_distances:
        fd_m = float(np.median(focus_distances))
        # focus_distance in calibration appears to be in metres (1500 = 1500m)
        # Sanity check: values < 1 are likely in mm (e.g. 0.5 = 500mm = 0.5m)
        if fd_m < 1.0:
            fd_m = fd_m * 1000.0
        print(f"[mvs] focus_distance from calibration: {fd_m:.1f} m")
        d_min = max(0.5, fd_m * 0.5)
        d_max = min(50000.0, fd_m * 10.0)
        return d_min, d_max, fd_m
    else:
        print("[mvs] WARNING: no focus_distance in calibration, using fallback range 1–10000 m")
        return 1.0, 10000.0, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PatchMatch MVS depth map from L16 A-cameras."
    )
    parser.add_argument('frames_dir', help="Directory containing A1.png … A5.png")
    parser.add_argument('calibration_json', help="Path to calibration.json")
    parser.add_argument('--cameras', nargs='+', default=list(A_CAMERAS),
                        metavar='CAM', help="Cameras to use (default: A1-A5)")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory (default: frames_dir/depth/)")
    parser.add_argument('--iterations', type=int, default=3,
                        help="PatchMatch iterations (default: 3)")
    parser.add_argument('--patch-size', type=int, default=3,
                        help="NCC patch half-width (default: 3 → 7×7)")
    parser.add_argument('--scale', type=int, default=8,
                        help="Downsample factor for speed (default: 8)")
    parser.add_argument('--depth-min', type=float, default=None,
                        help="Minimum depth in metres")
    parser.add_argument('--depth-max', type=float, default=None,
                        help="Maximum depth in metres")
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'],
                        help="Compute device (default: mps)")
    args = parser.parse_args()

    # ── Device setup ──────────────────────────────────────────────────────────
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("[mvs] WARNING: MPS not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"[mvs] device: {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(args.frames_dir, 'depth')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load calibration ──────────────────────────────────────────────────────
    print(f"[mvs] loading calibration: {args.calibration_json}")
    cameras = load_cameras(args.calibration_json)

    # Filter to requested cameras that exist in calibration
    cam_names = [n for n in args.cameras if n in cameras]
    if len(cam_names) < 2:
        print(f"ERROR: need at least 2 cameras, found: {cam_names}")
        sys.exit(1)
    print(f"[mvs] using cameras: {cam_names}")

    # ── Depth range ───────────────────────────────────────────────────────────
    d_min, d_max, focus_dist_m = depth_range_from_calibration(args.calibration_json, cam_names)
    if args.depth_min is not None:
        d_min = args.depth_min
    if args.depth_max is not None:
        d_max = args.depth_max
    print(f"[mvs] depth range: {d_min:.2f} – {d_max:.2f} m")

    # ── Build virtual camera ──────────────────────────────────────────────────
    print("[mvs] building virtual camera...")
    vc = build_virtual_camera(cameras, cam_names)
    scale = args.scale
    W_v_full, H_v_full = vc['W'], vc['H']
    W_v = W_v_full // scale
    H_v = H_v_full // scale
    print(f"[mvs] virtual camera: {W_v_full}×{H_v_full} → {W_v}×{H_v} at 1/{scale}")

    # Scale intrinsics for virtual camera
    K_virt_scaled = vc['K'].copy()
    K_virt_scaled[0, :] /= scale   # fx, cx
    K_virt_scaled[1, :] /= scale   # fy, cy
    R_virt = vc['R']
    t_virt = vc['t']

    # ── Load source images ────────────────────────────────────────────────────
    print("[mvs] loading A-camera images...")
    src_imgs_list = []
    src_cams_scaled = []
    missing = []

    for cam_name in cam_names:
        img_path = os.path.join(args.frames_dir, f'{cam_name}.png')
        if not os.path.exists(img_path):
            missing.append(cam_name)
            continue
        img_float = load_image_float(img_path)   # (H_s, W_s, 3) float32 [0,1]
        H_s_full, W_s_full = img_float.shape[:2]

        # Downsample source image by scale
        img_t = torch.tensor(img_float).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        if scale > 1:
            img_t = F.interpolate(
                img_t, size=(H_s_full // scale, W_s_full // scale),
                mode='bilinear', align_corners=False,
            )
        img_t = img_t.to(device=device, dtype=torch.float32)
        src_imgs_list.append(img_t)

        # Scale source camera intrinsics
        K_s_scaled = cameras[cam_name]['K'].copy()
        K_s_scaled[0, :] /= scale
        K_s_scaled[1, :] /= scale
        H_s_scaled = H_s_full // scale
        W_s_scaled = W_s_full // scale
        src_cams_scaled.append({
            'K': K_s_scaled,
            'R': cameras[cam_name]['R'],
            't': cameras[cam_name]['t'],
            'W': W_s_scaled,
            'H': H_s_scaled,
        })
        print(f"[mvs]   {cam_name}: {W_s_full}×{H_s_full} → {W_s_scaled}×{H_s_scaled}")

    if missing:
        print(f"[mvs] WARNING: missing frames for {missing}")
    if len(src_imgs_list) < 2:
        print("ERROR: need at least 2 source images")
        sys.exit(1)

    # ── Build virtual camera image (for NCC reference) ────────────────────────
    # Use A1 warped to virtual frame as the reference, or average of all A-cams.
    # Simplest approach: use the first available camera (A1) after warping.
    # For NCC we need the virtual camera's view — we approximate by warping A1.
    # Since the virtual cam is the centroid, A1 is a decent approximation.
    print("[mvs] building virtual camera image (warp A1 to virtual frame)...")
    ref_name = cam_names[0]
    ref_img_t = src_imgs_list[0]   # (1, 3, H_s_scaled, W_s_scaled)

    K_ref_s = torch.tensor(src_cams_scaled[0]['K'], dtype=torch.float32, device=device)
    R_ref = torch.tensor(src_cams_scaled[0]['R'], dtype=torch.float32, device=device)
    t_ref = torch.tensor(src_cams_scaled[0]['t'], dtype=torch.float32, device=device)
    H_s_s = src_cams_scaled[0]['H']
    W_s_s = src_cams_scaled[0]['W']

    K_virt_t = torch.tensor(K_virt_scaled, dtype=torch.float32, device=device)
    K_virt_inv = torch.linalg.inv(K_virt_t)
    R_virt_t = torch.tensor(R_virt, dtype=torch.float32, device=device)
    t_virt_t = torch.tensor(t_virt, dtype=torch.float32, device=device)

    # ── Unit convention: calibration t vectors are in mm.
    # All internal depth calculations must be in mm for consistency.
    # d_min/d_max are in metres (from CLI / calibration); convert to mm here.
    d_min_mm = d_min * 1000.0
    d_max_mm = d_max * 1000.0

    # Warp ref image to virtual frame using focus distance (or midpoint if unknown)
    median_depth_mm = float(focus_dist_m * 1000.0 if focus_dist_m is not None
                            else (d_min_mm + d_max_mm) / 2)
    depth_flat = torch.full((H_v, W_v), median_depth_mm,
                            dtype=torch.float32, device=device)

    # Build grid: virtual → A1
    # For forward warp (virtual → src), we need: for virtual pixel p_v at depth d,
    # find where it maps in A1.
    # We use build_reprojection_grid with VIRTUAL as target and A1 as source,
    # then grid_sample A1 image at those coords → virtual image.
    grid_v2ref = build_reprojection_grid(
        depth_flat, K_virt_inv, R_virt_t, t_virt_t,
        K_ref_s, R_ref, t_ref, H_s_s, W_s_s,
    )  # (1, H_v, W_v, 2)
    img_virt = F.grid_sample(
        ref_img_t, grid_v2ref, mode='bilinear', padding_mode='zeros', align_corners=True,
    )  # (1, 3, H_v, W_v)
    print(f"[mvs] virtual image: {img_virt.shape[3]}×{img_virt.shape[2]}")

    # ── Run PatchMatch (depths in mm internally) ──────────────────────────────
    # Initialise at focus distance if available (at far-field, NCC is flat so
    # random init can't converge — use focus distance as the strong prior).
    depth_init_mm = focus_dist_m * 1000.0 if focus_dist_m is not None else None

    t_mvs_start = time.time()
    depth_map_mm, cost_map = run_patchmatch(
        img_virt=img_virt,
        src_imgs=src_imgs_list,
        K_virt=K_virt_scaled,
        R_virt=R_virt,
        t_virt=t_virt,
        src_cams_scaled=src_cams_scaled,
        depth_min=d_min_mm,
        depth_max=d_max_mm,
        n_iterations=args.iterations,
        patch_half=args.patch_size,
        scale=float(scale),
        device=device,
        depth_init=depth_init_mm,
    )
    print(f"[mvs] PatchMatch done in {time.time()-t_mvs_start:.1f}s")

    # Convert depth map from mm → metres for output
    depth_map = depth_map_mm / 1000.0

    # ── Stage 3: Geometric consistency (pass depth in mm, consistent with t) ──
    print("[mvs] Stage 3: geometric consistency filtering...")
    geo_mask = geometric_consistency_mask(
        depth=depth_map_mm,
        K_virt_inv=K_virt_inv,
        K_virt=K_virt_t,
        R_virt=R_virt_t,
        t_virt=t_virt_t,
        src_cams_scaled=src_cams_scaled,
        threshold_px=2.0,
        min_agree=3,
    )
    n_total = H_v * W_v
    n_valid = int(geo_mask.sum().item())
    print(f"[mvs] geometric consistency: {n_valid}/{n_total} pixels pass "
          f"({100.0 * n_valid / n_total:.1f}%)")

    # Apply mask (set invalid pixels to 0)
    depth_masked = depth_map.clone()
    depth_masked[~geo_mask] = 0.0

    # ── Statistics ────────────────────────────────────────────────────────────
    valid_depths = depth_map[geo_mask]
    if valid_depths.numel() > 0:
        print(f"[mvs] depth stats (valid pixels only):")
        print(f"      min={valid_depths.min():.2f}m  max={valid_depths.max():.2f}m  "
              f"mean={valid_depths.mean():.2f}m  median={valid_depths.median():.2f}m  "
              f"std={valid_depths.std():.2f}m")
    else:
        print("[mvs] WARNING: no valid pixels after geometric consistency filtering")
        print(f"[mvs] unmasked depth stats:")
        print(f"      min={depth_map.min():.2f}m  max={depth_map.max():.2f}m  "
              f"mean={depth_map.mean():.2f}m  median={depth_map.median():.2f}m  "
              f"std={depth_map.std():.2f}m")

    # ── Save outputs ──────────────────────────────────────────────────────────
    depth_np = depth_masked.cpu().numpy().astype(np.float32)
    out_npz = os.path.join(out_dir, 'mvs_a_cameras.npz')
    np.savez_compressed(out_npz, depth=depth_np)
    print(f"[mvs] saved depth NPZ: {out_npz}  shape={depth_np.shape}")

    # Normalised uint8 PNG for visual inspection
    out_png = os.path.join(out_dir, 'mvs_a_cameras.png')
    d_vis = depth_map.cpu().numpy()
    d_min_vis, d_max_vis = float(d_vis.min()), float(d_vis.max())
    if d_max_vis > d_min_vis:
        d_norm = ((d_vis - d_min_vis) / (d_max_vis - d_min_vis) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d_vis, dtype=np.uint8)
    Image.fromarray(d_norm).save(out_png)
    print(f"[mvs] saved depth PNG: {out_png}")

    # Also save the masked version
    out_png_masked = os.path.join(out_dir, 'mvs_a_cameras_masked.png')
    d_masked_vis = depth_masked.cpu().numpy()
    if d_max_vis > d_min_vis:
        d_norm_m = ((d_masked_vis - d_min_vis) / (d_max_vis - d_min_vis) * 255).astype(np.uint8)
    else:
        d_norm_m = np.zeros_like(d_masked_vis, dtype=np.uint8)
    Image.fromarray(d_norm_m).save(out_png_masked)
    print(f"[mvs] saved masked depth PNG: {out_png_masked}")

    print("\n[mvs] Done.")


if __name__ == '__main__':
    main()
