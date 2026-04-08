#!/usr/bin/env python3
"""
lri_camera_remap.py — Dense remap arrays for L16 multi-camera fusion.

For each source camera, computes (map_x, map_y, coverage_mask) arrays that map
every output canvas pixel to its corresponding location in that camera's image.

Public API
----------
compute_remap(virtual_cam, source_cam, depth_map=None, tile_size=512)
    -> (map_x, map_y, coverage_mask)

cache_remap(remap_dir, cam_name, map_x, map_y, coverage_mask)
load_remap_cache(remap_dir, cam_name) -> tuple | None

apply_remap(src_image, map_x, map_y) -> np.ndarray
"""

import os
from pathlib import Path

import cv2
import numpy as np

# ── Default depth for flat-plane fallback ────────────────────────────────────
_DEFAULT_DEPTH_MM = 3000.0   # 3 metres in mm


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_R(cam: dict) -> np.ndarray:
    """Return rotation matrix, defaulting to identity if None."""
    R = cam.get('R')
    if R is None:
        return np.eye(3, dtype=np.float64)
    return np.asarray(R, dtype=np.float64)


def _safe_t(cam: dict) -> np.ndarray:
    """Return translation vector, defaulting to zeros if None."""
    t = cam.get('t')
    if t is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(t, dtype=np.float64)


def _project_points_src(P_src: np.ndarray, K_src: np.ndarray):
    """
    Project 3-D points (N,3) in source camera space into source pixel coords.

    Returns
    -------
    u_src, v_src : np.ndarray, shape (N,)
    valid        : bool array — points in front of camera (z > 0)
    """
    z = P_src[:, 2]
    valid = z > 1e-6
    # Avoid division by zero for points behind the camera
    z_safe = np.where(valid, z, 1.0)

    fx = K_src[0, 0]
    fy = K_src[1, 1]
    cx = K_src[0, 2]
    cy = K_src[1, 2]

    u_src = fx * (P_src[:, 0] / z_safe) + cx
    v_src = fy * (P_src[:, 1] / z_safe) + cy
    return u_src, v_src, valid


def _process_tile(
    u_out_flat: np.ndarray,
    v_out_flat: np.ndarray,
    K_virt_inv: np.ndarray,
    R_virt_T: np.ndarray,
    t_virt: np.ndarray,
    R_src: np.ndarray,
    t_src: np.ndarray,
    K_src: np.ndarray,
    depth_vals,           # float or np.ndarray (N,) — depth in mm
    use_depth: bool,
):
    """
    Core per-pixel computation for a flat array of (u_out, v_out) pairs.

    Uses Apple MPS (Metal) when available via PyTorch, falling back to NumPy
    on CPU.  M4's unified memory means no CPU↔GPU copy cost.

    Returns u_src, v_src, front_of_camera (all shape (N,)).
    """
    global _MPS_AVAILABLE
    if _MPS_AVAILABLE is None:
        try:
            import torch
            _MPS_AVAILABLE = torch.backends.mps.is_available()
        except ImportError:
            _MPS_AVAILABLE = False

    if _MPS_AVAILABLE:
        return _process_tile_mps(
            u_out_flat, v_out_flat, K_virt_inv, R_virt_T, t_virt,
            R_src, t_src, K_src, depth_vals, use_depth,
        )

    return _process_tile_numpy(
        u_out_flat, v_out_flat, K_virt_inv, R_virt_T, t_virt,
        R_src, t_src, K_src, depth_vals, use_depth,
    )


_MPS_AVAILABLE = None   # lazily detected


def _process_tile_numpy(
    u_out_flat, v_out_flat, K_virt_inv, R_virt_T, t_virt,
    R_src, t_src, K_src, depth_vals, use_depth,
):
    """NumPy CPU fallback implementation of _process_tile."""
    N = u_out_flat.shape[0]

    # --- Step 1: unproject output pixels to rays in virtual camera space ---
    uvw = np.stack([u_out_flat, v_out_flat, np.ones(N, dtype=np.float64)], axis=1)  # (N,3)
    rays = (K_virt_inv @ uvw.T).T    # (N,3)  — unnormalised; z=1 for rectilinear

    # --- Step 2: lift to 3-D points ---
    if use_depth:
        depth_m = depth_vals.astype(np.float64)
        scale = depth_m / rays[:, 2]
        P_virt = rays * scale[:, np.newaxis]
    else:
        scale = _DEFAULT_DEPTH_MM / rays[:, 2]
        P_virt = rays * scale[:, np.newaxis]

    # --- Step 3: virtual camera space → world space ---
    P_world = P_virt - t_virt[np.newaxis, :]

    # --- Step 4: world space → source camera space ---
    P_src = (R_src @ P_world.T).T + t_src[np.newaxis, :]

    # --- Step 5: project into source image ---
    u_src, v_src, front = _project_points_src(P_src, K_src)
    return u_src, v_src, front


def _process_tile_mps(
    u_out_flat, v_out_flat, K_virt_inv, R_virt_T, t_virt,
    R_src, t_src, K_src, depth_vals, use_depth,
):
    """
    Apple Metal (MPS) implementation of _process_tile.

    Uses float32 throughout (MPS native precision).  The precision loss vs
    float64 is negligible for pixel-coordinate arithmetic at L16 sensor
    resolutions (~4K).  On M4 unified memory there is no CPU↔GPU copy cost.
    """
    import torch
    device = torch.device('mps')
    f32 = torch.float32

    # Pre-computed camera matrices as float32 MPS tensors
    Ki = torch.from_numpy(K_virt_inv.astype(np.float32)).to(device)   # (3,3)
    R  = torch.from_numpy(R_src.astype(np.float32)).to(device)         # (3,3)
    tv = torch.from_numpy(t_virt.astype(np.float32)).to(device)        # (3,)
    ts = torch.from_numpy(t_src.astype(np.float32)).to(device)         # (3,)

    N = u_out_flat.shape[0]
    u = torch.from_numpy(u_out_flat.astype(np.float32)).to(device)
    v = torch.from_numpy(v_out_flat.astype(np.float32)).to(device)
    ones = torch.ones(N, dtype=f32, device=device)

    # Step 1: unproject
    uvw = torch.stack([u, v, ones], dim=1)          # (N, 3)
    rays = (Ki @ uvw.T).T                           # (N, 3)

    # Step 2: lift to 3D
    if use_depth:
        d = torch.from_numpy(depth_vals.astype(np.float32)).to(device)
        scale = d / rays[:, 2]
    else:
        scale = torch.full((N,), _DEFAULT_DEPTH_MM, dtype=f32, device=device) / rays[:, 2]
    P_virt = rays * scale.unsqueeze(1)              # (N, 3)

    # Step 3: virtual → world
    P_world = P_virt - tv.unsqueeze(0)              # (N, 3)

    # Step 4: world → source camera
    P_src = (R @ P_world.T).T + ts.unsqueeze(0)    # (N, 3)

    # Step 5: project
    z = P_src[:, 2]
    valid = z > 1e-6
    z_safe = torch.where(valid, z, torch.ones_like(z))

    fx = float(K_src[0, 0]); fy = float(K_src[1, 1])
    cx = float(K_src[0, 2]); cy = float(K_src[1, 2])

    u_src = fx * (P_src[:, 0] / z_safe) + cx
    v_src = fy * (P_src[:, 1] / z_safe) + cy

    # Return as numpy (zero-copy via MPS shared memory on Apple Silicon)
    return (
        u_src.cpu().numpy().astype(np.float64),
        v_src.cpu().numpy().astype(np.float64),
        valid.cpu().numpy(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def _camera_canvas_roi(virtual_cam, source_cam: dict, margin_px: int = 200):
    """
    Approximate canvas-space bounding box (x0, y0, x1, y1) of source_cam's FOV.

    Projects the source sensor corners + centre through the camera geometry at
    the default flat-plane depth to find where they land on the virtual canvas.
    Returns a rectangle with extra margin_px padding on each side, clamped to
    the canvas bounds.  Used to skip canvas tiles that can't possibly contain
    valid pixels for a given source camera.
    """
    W_src = int(source_cam['W'])
    H_src = int(source_cam['H'])
    K_src = source_cam['K'].astype(np.float64)
    R_src = _safe_R(source_cam)
    t_src = _safe_t(source_cam)
    t_virt = np.asarray(
        virtual_cam.t if virtual_cam.t is not None else np.zeros(3), dtype=np.float64
    )
    K_virt = virtual_cam.K.astype(np.float64)
    H_out = virtual_cam.H
    W_out = virtual_cam.W

    # Sample sensor corners + centre + edge midpoints (9 points)
    us = [0.0, W_src / 2, float(W_src)]
    vs = [0.0, H_src / 2, float(H_src)]
    pts = np.array([[u, v] for u in us for v in vs], dtype=np.float64)  # (9, 2)

    # Optionally apply the x-flip for MOVABLE cameras
    if source_cam.get('virt_mirror_x'):
        pts[:, 0] = (W_src - 1.0) - pts[:, 0]

    # Unproject sensor points to 3D rays in source camera space at default depth
    K_src_inv = np.linalg.inv(K_src)
    uvw = np.column_stack([pts, np.ones(len(pts))])        # (9, 3)
    rays = (K_src_inv @ uvw.T).T                           # (9, 3)
    depth = _DEFAULT_DEPTH_MM / rays[:, 2]
    P_src = rays * depth[:, np.newaxis]                    # (9, 3) in source cam space

    # Source cam space → world space
    P_world = (R_src.T @ P_src.T).T - (R_src.T @ t_src)   # (9, 3)

    # World space → virtual cam space (R_virt = I)
    P_vc = P_world - t_virt[np.newaxis, :]                 # (9, 3)

    # Project onto virtual canvas
    valid = P_vc[:, 2] > 1e-6
    if not np.any(valid):
        return 0, 0, W_out, H_out  # can't determine — process everything

    P_vc_v = P_vc[valid]
    px = K_virt[0, 0] * P_vc_v[:, 0] / P_vc_v[:, 2] + K_virt[0, 2]
    py = K_virt[1, 1] * P_vc_v[:, 1] / P_vc_v[:, 2] + K_virt[1, 2]

    x0 = int(np.floor(px.min())) - margin_px
    x1 = int(np.ceil(px.max()))  + margin_px
    y0 = int(np.floor(py.min())) - margin_px
    y1 = int(np.ceil(py.max()))  + margin_px

    x0 = max(x0, 0); y0 = max(y0, 0)
    x1 = min(x1, W_out); y1 = min(y1, H_out)
    return x0, y0, x1, y1


def compute_remap(
    virtual_cam,
    source_cam: dict,
    depth_map=None,
    tile_size: int = 512,
):
    """
    Compute dense remap arrays mapping output canvas pixels → source pixels.

    Parameters
    ----------
    virtual_cam : VirtualCamera
        The synthetic output canvas (from lri_virtual_camera.py).
    source_cam : dict
        Camera dict with keys K, R, t, W, H (R and t may be None).
    depth_map : np.ndarray or None
        Float32 array (H_out, W_out) in metres, or None for flat-plane fallback.
    tile_size : int
        Process in tiles of this many rows to limit peak RAM usage.

    Returns
    -------
    map_x : np.ndarray, float32, shape (H_out, W_out)
        Source column for each output pixel.
    map_y : np.ndarray, float32, shape (H_out, W_out)
        Source row for each output pixel.
    coverage_mask : np.ndarray, bool, shape (H_out, W_out)
        True where the mapped source pixel is within sensor bounds (2px margin).
    """
    H_out = virtual_cam.H
    W_out = virtual_cam.W

    K_virt = virtual_cam.K.astype(np.float64)
    K_virt_inv = np.linalg.inv(K_virt)

    # Virtual camera: R = identity, t = centroid of wide cameras (or zeros)
    t_virt = np.asarray(virtual_cam.t if virtual_cam.t is not None else np.zeros(3),
                        dtype=np.float64)
    R_virt_T = np.eye(3, dtype=np.float64)   # R_virt = I, so R_virt^T = I

    # Source camera parameters
    K_src = source_cam['K'].astype(np.float64)
    R_src = _safe_R(source_cam)
    t_src = _safe_t(source_cam)
    W_src = int(source_cam['W'])
    H_src = int(source_cam['H'])

    use_depth = depth_map is not None

    # Approximate canvas bounding box for this camera — skip tiles that can't
    # possibly contain valid pixels.  Provides ~10-30× speedup for telephoto
    # cameras (C cameras cover ~3% of the 400 MP wide canvas).
    _rx0, _ry0, _rx1, _ry1 = _camera_canvas_roi(virtual_cam, source_cam)

    # Output arrays (float32 to keep memory reasonable for ~81 MP canvas)
    map_x = np.full((H_out, W_out), -1.0, dtype=np.float32)
    map_y = np.full((H_out, W_out), -1.0, dtype=np.float32)

    # Tile over rows — only within the ROI row range
    for row_start in range(_ry0, _ry1, tile_size):
        row_end = min(row_start + tile_size, _ry1)
        rows = row_end - row_start

        # Build pixel grid for this tile (only the ROI columns)
        us = np.arange(_rx0, _rx1, dtype=np.float64)
        vs = np.arange(row_start, row_end, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)           # (rows, W_roi)
        u_flat = uu.ravel()                    # (rows*W_roi,)
        v_flat = vv.ravel()
        W_roi = _rx1 - _rx0

        if use_depth:
            depth_tile = depth_map[row_start:row_end, _rx0:_rx1].astype(np.float32)
            # Convert metres → mm to be consistent with camera translation units
            depth_vals = (depth_tile.ravel() * 1000.0).astype(np.float64)
        else:
            depth_vals = None

        u_src, v_src, _ = _process_tile(
            u_flat, v_flat,
            K_virt_inv,
            R_virt_T,
            t_virt,
            R_src, t_src, K_src,
            depth_vals,
            use_depth,
        )

        map_x[row_start:row_end, _rx0:_rx1] = u_src.reshape(rows, W_roi).astype(np.float32)
        map_y[row_start:row_end, _rx0:_rx1] = v_src.reshape(rows, W_roi).astype(np.float32)

    # MOVABLE cameras: virtual pose R_virt includes a diag(-1,1,1) x-flip so that
    # det(R_virt) = +1.  This means the projected x-coordinate is in "flipped" sensor
    # space.  The physical sensor pixel is:  u_physical = (W_src - 1) - u_projected.
    # The flag 'virt_mirror_x' is set by load_cameras for all MOVABLE cameras.
    if source_cam.get('virt_mirror_x'):
        map_x = (W_src - 1.0) - map_x

    # Coverage mask: source pixel within sensor with 2px margin
    coverage_mask = (
        (map_x >= 2.0) & (map_x <= W_src - 2) &
        (map_y >= 2.0) & (map_y <= H_src - 2)
    )

    return map_x, map_y, coverage_mask


def cache_remap(
    remap_dir: str,
    cam_name: str,
    map_x: np.ndarray,
    map_y: np.ndarray,
    coverage_mask: np.ndarray,
):
    """
    Save remap arrays to <remap_dir>/<cam_name>_map.npz (compressed).

    Parameters
    ----------
    remap_dir : str
        Directory where cache files are stored (created if absent).
    cam_name : str
        Camera identifier, e.g. 'A1', 'B4'.
    map_x, map_y : np.ndarray, float32
        Remap arrays returned by compute_remap().
    coverage_mask : np.ndarray, bool
        Coverage mask returned by compute_remap().
    """
    os.makedirs(remap_dir, exist_ok=True)
    path = os.path.join(remap_dir, f'{cam_name}_map.npz')
    np.savez_compressed(path, map_x=map_x, map_y=map_y, coverage=coverage_mask)


def load_remap_cache(remap_dir: str, cam_name: str):
    """
    Load remap arrays from cache, or return None on cache miss.

    Parameters
    ----------
    remap_dir : str
        Directory containing cache files.
    cam_name : str
        Camera identifier, e.g. 'A1', 'B4'.

    Returns
    -------
    (map_x, map_y, coverage_mask) or None
    """
    path = os.path.join(remap_dir, f'{cam_name}_map.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data['map_x'], data['map_y'], data['coverage'].astype(bool)


def apply_remap(
    src_image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Warp src_image into the output canvas coordinate frame.

    Parameters
    ----------
    src_image : np.ndarray
        Source camera image (H_src, W_src[, C]).
    map_x : np.ndarray, float32, shape (H_out, W_out)
        Source column for each output pixel.
    map_y : np.ndarray, float32, shape (H_out, W_out)
        Source row for each output pixel.
    interpolation : int
        OpenCV interpolation flag.  Default INTER_LINEAR.
        Use INTER_NEAREST for cameras mapped at ~1:1 scale (same focal length
        as the virtual canvas) to preserve full source sharpness.

    Returns
    -------
    np.ndarray, same dtype as src_image, shape (H_out, W_out[, C])
        Warped image; out-of-bounds pixels filled with 0.
    """
    return cv2.remap(
        src_image, map_x, map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from lri_fuse_image import load_cameras
    from lri_virtual_camera import VirtualCamera

    cal_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_cal/calibration.json/calibration.json'

    cameras = load_cameras(cal_path)
    vc = VirtualCamera(cameras)
    print(f"Canvas: {vc.W}×{vc.H}")

    # Test remap for A1 (wide, should be nearly identity in center)
    cam = cameras['A1']
    map_x, map_y, mask = compute_remap(vc, cam)
    print(f"A1 remap: coverage={mask.mean()*100:.1f}%")
    print(f"  center map_x[H//2, W//2] = {map_x[vc.H//2, vc.W//2]:.1f}  (expect ~{cam['W']//2})")
    print(f"  center map_y[H//2, W//2] = {map_y[vc.H//2, vc.W//2]:.1f}  (expect ~{cam['H']//2})")

    # Test remap for B4 (telephoto, should only cover center ~40%)
    cam_b4 = cameras.get('B4')
    if cam_b4:
        map_x4, map_y4, mask4 = compute_remap(vc, cam_b4)
        print(f"B4 remap: coverage={mask4.mean()*100:.1f}%  (expect ~15-25%)")
