#!/usr/bin/env python3
"""
lri_depth_loader.py — Load and reproject depth maps for the L16 fusion pipeline.

Provides depth maps in the VirtualCamera coordinate system for use during
per-camera remap to correctly handle parallax between cameras separated by
up to 50mm.

Functions
---------
load_depth_for_canvas(lumen_dir, virtual_cam, source_cam_name, source_cam)
    Load the best available depth and reproject to the VirtualCamera canvas.

flat_plane_depth(virtual_cam, depth_m=3.0)
    Return a constant depth map at depth_m metres.

depth_stats(depth_map)
    Return summary statistics for a depth map.
"""

import glob
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Baseline threshold below which reprojection shift is sub-pixel.
# Wide cameras are ≤ ~35 mm from the virtual-camera centroid; 60 mm gives
# a comfortable margin before telephoto territory.
_WIDE_BASELINE_MM = 60.0


def load_depth_for_canvas(
    lumen_dir: str,
    virtual_cam,
    source_cam_name: str,
    source_cam: dict,
) -> "np.ndarray | None":
    """Load the best available depth map and return it on the VirtualCamera canvas.

    Depth source priority:
      1. ``<lumen_dir>/fused_*cams_median_depth.png``  — uint16 PNG, values in mm
      2. ``<lumen_dir>/depth/<source_cam_name>.npz``   — float32 array, values in
         metres (key='depth')
      3. Returns None if nothing is found.

    After loading, the depth is:
      - Converted to float32 metres.
      - Resized to match the VirtualCamera canvas (H_out, W_out) using
        cv2.INTER_LINEAR.
      - Returned directly for wide cameras (baseline < 60 mm from virtual_cam.t)
        because the reprojection shift is sub-pixel at most scene depths.
      - Returned with a warning for telephoto cameras (this case should not arise
        in practice because telephoto cameras never contribute to the fused depth).

    Parameters
    ----------
    lumen_dir : str
        Path to the lumen directory for this capture.
    virtual_cam : VirtualCamera
        Output canvas definition.
    source_cam_name : str
        Name of the source camera (e.g. 'A1').
    source_cam : dict
        Camera dict with keys K, R, t, W, H, mirror_type.

    Returns
    -------
    np.ndarray or None
        float32 ndarray of shape (H_out, W_out), depth in metres, or None if no
        depth file is available.
    """
    H_out = int(virtual_cam.H)
    W_out = int(virtual_cam.W)

    depth_m = _load_raw_depth(lumen_dir, source_cam_name)
    if depth_m is None:
        return None

    # Resize to VirtualCamera canvas.
    if depth_m.shape != (H_out, W_out):
        depth_m = cv2.resize(depth_m, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

    # Determine baseline from VirtualCamera origin.
    baseline_mm = _baseline_mm(virtual_cam, source_cam)

    if baseline_mm < _WIDE_BASELINE_MM:
        # Wide camera — sub-pixel reprojection shift, return as-is.
        return depth_m

    # Telephoto camera — large parallax would require full reprojection, but this
    # case should not arise because only wide / A cameras contribute to the fused
    # depth.  Return the resized depth with a warning.
    logger.warning(
        "load_depth_for_canvas: source camera '%s' has a large baseline of %.1f mm "
        "from the virtual camera origin.  Full reprojection is not implemented; "
        "returning resized depth.  This may introduce parallax errors.",
        source_cam_name,
        baseline_mm,
    )
    return depth_m


# ── private helpers ────────────────────────────────────────────────────────────


def _load_raw_depth(lumen_dir: str, source_cam_name: str) -> "np.ndarray | None":
    """Try each depth source in priority order and return float32 metres or None."""

    # Priority 1 — fused depth PNG (uint16, mm).
    # Matches both legacy "fused_*cams_median_depth.png" and newer "fused_*cams_*depth.png"
    # produced by different pipeline versions (e.g. depthpro, median, etc.).
    pattern = os.path.join(lumen_dir, "fused_*cams_*depth.png")
    matches = sorted(glob.glob(pattern))
    if matches:
        png_path = matches[0]
        if len(matches) > 1:
            logger.debug(
                "_load_raw_depth: multiple fused depth PNGs found, using %s", png_path
            )
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            logger.debug("_load_raw_depth: loaded fused depth PNG %s", png_path)
            return img.astype(np.float32) / 1000.0  # mm → metres

        logger.warning("_load_raw_depth: failed to read %s", png_path)

    # Priority 2 — per-camera NPZ (float32, metres).
    npz_path = os.path.join(lumen_dir, "depth", f"{source_cam_name}.npz")
    if os.path.isfile(npz_path):
        try:
            data = np.load(npz_path)
            depth = data["depth"].astype(np.float32)
            logger.debug("_load_raw_depth: loaded depth NPZ %s", npz_path)
            return depth
        except Exception as exc:  # noqa: BLE001
            logger.warning("_load_raw_depth: failed to load %s — %s", npz_path, exc)

    return None


def _baseline_mm(virtual_cam, source_cam: dict) -> float:
    """Return Euclidean distance (mm) between virtual_cam.t and source_cam['t'].

    Returns 0.0 if either translation vector is None or missing.
    """
    vc_t = getattr(virtual_cam, "t", None)
    sc_t = source_cam.get("t", None)

    if vc_t is None or sc_t is None:
        return 0.0

    return float(np.linalg.norm(np.asarray(vc_t, dtype=np.float64)
                                - np.asarray(sc_t, dtype=np.float64)))


# ── public utilities ───────────────────────────────────────────────────────────


def forward_warp_depth(
    depth_src_m: np.ndarray,
    src_cam: dict,
    virtual_cam,
) -> np.ndarray:
    """Reproject a source-camera depth map into the virtual camera canvas frame.

    The input depth is in the source camera's own ray/pixel space (as produced by
    monocular depth estimators such as Depth Pro running on the source camera image).
    A simple cv2.resize to canvas dimensions is wrong because the source camera's
    pixel grid is not the same as the virtual camera's ray grid — the discrepancy
    grows with distance from the optical axis and can reach 20-30 px for wide cameras.

    This function forward-warps: for every source pixel (u_s, v_s) the 3-D point is
    reconstructed using the source camera's intrinsics and extrinsics, then projected
    into the virtual camera's coordinate frame to get its canvas position and depth.
    Holes left by the forward warp are filled with nearest-neighbour in-painting.

    Parameters
    ----------
    depth_src_m : np.ndarray
        float32 (H_src, W_src) depth in metres as seen by the source camera.
    src_cam : dict
        Camera dict with keys K (3×3), R (3×3 or None), t (3, or None), W, H.
    virtual_cam : VirtualCamera
        Target canvas (H_out, W_out, K).

    Returns
    -------
    np.ndarray
        float32 (H_out, W_out) depth in metres in the virtual camera's ray space.
        Pixels with no coverage are 0.0.
    """
    H_src, W_src = depth_src_m.shape
    H_out = int(virtual_cam.H)
    W_out = int(virtual_cam.W)

    # Source camera parameters
    K_src = np.asarray(src_cam['K'], dtype=np.float64)
    R_src = np.asarray(src_cam['R'], dtype=np.float64) if src_cam.get('R') is not None else np.eye(3)
    t_src = np.asarray(src_cam['t'], dtype=np.float64).ravel() if src_cam.get('t') is not None else np.zeros(3)

    # Virtual camera parameters
    K_vc = np.asarray(virtual_cam.K, dtype=np.float64)
    t_vc = np.asarray(virtual_cam.t if virtual_cam.t is not None else np.zeros(3), dtype=np.float64).ravel()
    # R_vc = I (virtual camera is axis-aligned)

    fx_s, fy_s = K_src[0, 0], K_src[1, 1]
    cx_s, cy_s = K_src[0, 2], K_src[1, 2]
    fx_v, fy_v = K_vc[0, 0], K_vc[1, 1]
    cx_v, cy_v = K_vc[0, 2], K_vc[1, 2]

    # Build pixel grid for source camera (all pixels at once)
    us = np.arange(W_src, dtype=np.float64)
    vs = np.arange(H_src, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)             # (H_src, W_src)
    d_mm = depth_src_m.astype(np.float64) * 1000.0  # metres → mm (same units as t)

    # Skip zero/invalid depth pixels
    valid = d_mm > 0.0

    u_flat = uu[valid]
    v_flat = vv[valid]
    d_flat = d_mm[valid]

    # Unproject to source camera 3-D (mm)
    X_s = (u_flat - cx_s) / fx_s * d_flat
    Y_s = (v_flat - cy_s) / fy_s * d_flat
    Z_s = d_flat                              # (N,)

    P_src = np.stack([X_s, Y_s, Z_s], axis=1)  # (N, 3)

    # Source camera → world:  P_world = R_src^T @ (P_src - t_src)
    P_world = (R_src.T @ (P_src - t_src[np.newaxis, :]).T).T  # (N, 3)

    # World → virtual camera:  P_vc = P_world - t_vc  (R_vc = I)
    P_vc = P_world - t_vc[np.newaxis, :]  # (N, 3)

    Z_vc = P_vc[:, 2]
    front = Z_vc > 1e-6

    u_vc = (fx_v * P_vc[front, 0] / Z_vc[front] + cx_v)
    v_vc = (fy_v * P_vc[front, 1] / Z_vc[front] + cy_v)
    d_vc_m = Z_vc[front] / 1000.0   # mm → metres

    # Splat into output canvas (integer coordinates, keep closest depth if multiple hit)
    u_vc_i = np.round(u_vc).astype(np.int32)
    v_vc_i = np.round(v_vc).astype(np.int32)

    in_bounds = (u_vc_i >= 0) & (u_vc_i < W_out) & (v_vc_i >= 0) & (v_vc_i < H_out)

    canvas_depth = np.zeros((H_out, W_out), dtype=np.float32)
    # Accumulate with mean (sum + count, then divide)
    canvas_sum = np.zeros((H_out, W_out), dtype=np.float64)
    canvas_cnt = np.zeros((H_out, W_out), dtype=np.int32)

    np.add.at(canvas_sum, (v_vc_i[in_bounds], u_vc_i[in_bounds]), d_vc_m[in_bounds])
    np.add.at(canvas_cnt, (v_vc_i[in_bounds], u_vc_i[in_bounds]), 1)

    filled = canvas_cnt > 0
    canvas_depth[filled] = (canvas_sum[filled] / canvas_cnt[filled]).astype(np.float32)

    # Fill holes with nearest-neighbour in-painting
    if not np.all(filled):
        hole_mask = (~filled).astype(np.uint8) * 255
        # Scale depth to uint16 range for cv2.inpaint (which handles 8u and 16u only)
        max_d = float(canvas_depth.max()) if canvas_depth.max() > 0 else 1.0
        canvas_u16 = np.clip(canvas_depth / max_d * 65535.0, 0, 65535).astype(np.uint16)
        canvas_u16 = cv2.inpaint(canvas_u16, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        canvas_depth = canvas_u16.astype(np.float32) / 65535.0 * max_d

    return canvas_depth


def flat_plane_depth(virtual_cam, depth_m: float = 3.0) -> np.ndarray:
    """Return a constant depth map at *depth_m* metres.

    Used as a fallback when no depth file is available.

    Parameters
    ----------
    virtual_cam : VirtualCamera
        Output canvas whose H and W define the array shape.
    depth_m : float
        Scene depth in metres (default 3.0).

    Returns
    -------
    np.ndarray
        float32 array of shape (H_out, W_out) filled with *depth_m*.
    """
    return np.full((int(virtual_cam.H), int(virtual_cam.W)), depth_m, dtype=np.float32)


def depth_stats(depth_map: np.ndarray) -> dict:
    """Return summary statistics for *depth_map*.

    Parameters
    ----------
    depth_map : np.ndarray
        2-D float array, depth in metres.

    Returns
    -------
    dict
        Keys: min, max, median, valid_fraction (fraction of pixels > 0).
    """
    valid = depth_map[depth_map > 0]
    valid_fraction = float(valid.size) / float(depth_map.size) if depth_map.size else 0.0

    if valid.size:
        return {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "median": float(np.median(valid)),
            "valid_fraction": valid_fraction,
        }

    return {
        "min": 0.0,
        "max": 0.0,
        "median": 0.0,
        "valid_fraction": 0.0,
    }


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from lri_fuse_image import load_cameras
    from lri_virtual_camera import VirtualCamera

    lumen_dir = "/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_lumen"
    cal_path  = "/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_cal/calibration.json/calibration.json"

    cameras = load_cameras(cal_path)
    vc = VirtualCamera(cameras)
    print(f"Canvas: {vc.W}×{vc.H}")

    depth = load_depth_for_canvas(lumen_dir, vc, "A1", cameras["A1"])
    if depth is not None:
        stats = depth_stats(depth)
        print(f"Depth loaded: shape={depth.shape}, {stats}")
    else:
        print("No depth found — flat plane fallback")
        depth = flat_plane_depth(vc, depth_m=3.0)
        print(f"Flat plane: shape={depth.shape}, depth={depth[0,0]:.1f}m")
