"""
lri_canvas_blend.py — Core canvas assembly step for the L16 fusion pipeline.

Warps every source camera frame onto the virtual output canvas, blends them
using per-pixel confidence weights, and returns a float32 (H_out, W_out, 3)
image in the same uint16 intensity scale as the inputs.

Public API
----------
load_image(frames_dir, cam_name)  -> np.ndarray  uint16 (H, W, 3)
save_canvas(canvas_f32, path)     -> None
assemble_canvas(...)              -> np.ndarray  float32 (H_out, W_out, 3)
"""

from __future__ import annotations

import os
import numpy as np
import cv2

from lri_camera_remap import compute_remap, apply_remap, load_remap_cache, cache_remap
from lri_confidence import compute_confidence
from lri_depth_loader import forward_warp_depth, _load_raw_depth
from lri_confidence import resolution_weight
from lri_wb import apply_wb_exposure
from lri_fuse_image import estimate_b_to_ref_homography


# ---------------------------------------------------------------------------
# Geometry-grounded color correction for B/C cameras
# ---------------------------------------------------------------------------

def _compute_b_to_a_color_correction(
    b_img: np.ndarray,
    a_img: np.ndarray,
    b_cam: dict,
    a_cam: dict,
    depth_m: float,
    n_grid: int = 50,
) -> tuple[float, float, float]:
    """
    Compute per-channel color correction factors by projecting a regular grid
    of B-camera pixels into the A-reference camera using K/R/t geometry at
    a flat depth plane.

    Parameters
    ----------
    b_img : np.ndarray  float32 (H, W, 3)   B camera source image
    a_img : np.ndarray  uint16 or float32    A reference camera source image
    b_cam : dict        B camera calibration (K, R, t in metres)
    a_cam : dict        A reference camera calibration
    depth_m : float     flat-plane depth in metres (focus distance)
    n_grid : int        grid resolution per axis (n_grid × n_grid samples)

    Returns
    -------
    (r_scale, g_scale, b_scale) : float
        Multiply B image by these to match A1's colors in the overlap region.
        Returns (1.0, 1.0, 1.0) if fewer than 100 valid overlap samples found.
    """
    H_b, W_b = b_img.shape[:2]
    H_a, W_a = a_img.shape[:2]

    K_b = b_cam['K'].astype(np.float64)
    R_b = b_cam['R'].astype(np.float64)
    t_b = b_cam['t'].astype(np.float64).ravel()
    K_a = a_cam['K'].astype(np.float64)
    R_a = a_cam['R'].astype(np.float64)
    t_a = a_cam['t'].astype(np.float64).ravel()

    # Sample a central grid of B pixels (avoid 15% border to skip lens vignette)
    margin = 0.15
    u_b = np.linspace(W_b * margin, W_b * (1 - margin), n_grid)
    v_b = np.linspace(H_b * margin, H_b * (1 - margin), n_grid)
    uu, vv = np.meshgrid(u_b, v_b)
    u_flat = uu.ravel()
    v_flat = vv.ravel()

    # Unproject B pixels to 3-D at flat focus depth.
    # Use mm throughout — camera t vectors are in mm (calibration.json convention).
    depth_mm = depth_m * 1000.0

    K_b_inv = np.linalg.inv(K_b)
    ones = np.ones(len(u_flat))
    uv1 = np.stack([u_flat, v_flat, ones], axis=-1)       # (N, 3)
    ray_c = (K_b_inv @ uv1.T).T                           # (N, 3) camera-frame rays
    P_c = depth_mm * ray_c / ray_c[:, 2:3]                # (N, 3) in mm

    # B camera frame → world frame  (P_world = R_b^T * (P_c - t_b), units: mm)
    P_world = (R_b.T @ (P_c - t_b).T).T                  # (N, 3)

    # World → A camera frame → A pixel coords
    P_a = (R_a @ P_world.T + t_a[:, None]).T              # (N, 3)
    valid = P_a[:, 2] > 0.01
    u_a = K_a[0, 0] * P_a[:, 0] / np.where(valid, P_a[:, 2], 1.0) + K_a[0, 2]
    v_a = K_a[1, 1] * P_a[:, 1] / np.where(valid, P_a[:, 2], 1.0) + K_a[1, 2]

    # Keep only samples inside both image bounds
    pad = 8
    valid &= (u_a >= pad) & (u_a < W_a - pad) & (v_a >= pad) & (v_a < H_a - pad)

    if valid.sum() < 100:
        return 1.0, 1.0, 1.0

    ui_b = u_flat[valid].astype(np.int32)
    vi_b = v_flat[valid].astype(np.int32)
    ui_a = np.clip(u_a[valid].astype(np.int32), 0, W_a - 1)
    vi_a = np.clip(v_a[valid].astype(np.int32), 0, H_a - 1)

    b_pix = b_img[vi_b, ui_b].astype(np.float64)          # (N, 3)
    a_pix = a_img.astype(np.float64)[vi_a, ui_a]          # (N, 3)

    # Exclude clipped / black pixels in either image
    b_lum = b_pix.mean(axis=1)
    a_lum = a_pix.mean(axis=1)
    good = (b_lum > 512) & (b_lum < 62000) & (a_lum > 512) & (a_lum < 62000)

    if good.sum() < 50:
        return 1.0, 1.0, 1.0

    b_g = b_pix[good]
    a_g = a_pix[good]

    # Median per-channel ratio (robust to outlier pixel pairs)
    r_scale = float(np.clip(np.median(a_g[:, 0] / np.maximum(b_g[:, 0], 1.0)),
                            0.1, 10.0))
    g_scale = float(np.clip(np.median(a_g[:, 1] / np.maximum(b_g[:, 1], 1.0)),
                            0.1, 10.0))
    b_scale = float(np.clip(np.median(a_g[:, 2] / np.maximum(b_g[:, 2], 1.0)),
                            0.1, 10.0))

    return r_scale, g_scale, b_scale


def _compute_b_to_canvas_color_correction(
    b_img: np.ndarray,
    b_cam: dict,
    virtual_cam,
    reference_canvas: np.ndarray,
    depth_m: float,
    n_grid: int = 50,
) -> tuple[float, float, float]:
    """
    Compute per-channel color correction using the pre-computed reference canvas
    (e.g. the all-A canvas) rather than the A1 source image.

    Projects B camera pixels forward to virtual canvas space, samples
    ``reference_canvas`` at those positions, computes median per-channel ratio.

    This is more accurate than ``_compute_b_to_a_color_correction`` when the
    reference canvas is an all-A blend (5 cameras), because the 5-camera blend
    gives a more complete and representative reference than A1 alone.

    Parameters
    ----------
    b_img : np.ndarray  float32 (H, W, 3)   B camera source image
    b_cam : dict        B camera calibration (K, R, t in mm)
    virtual_cam         VirtualCamera with K, R, t, W, H attributes
    reference_canvas : np.ndarray  float32 (H_vc, W_vc, 3)   all-A canvas
    depth_m : float     flat-plane depth in metres
    n_grid : int        grid resolution per axis

    Returns
    -------
    (r_scale, g_scale, b_scale) : float
        Multiply B image by these to match the reference canvas colors.
    """
    H_b, W_b = b_img.shape[:2]
    H_vc, W_vc = reference_canvas.shape[:2]

    K_b = b_cam['K'].astype(np.float64)
    R_b = b_cam['R'].astype(np.float64)
    t_b = b_cam['t'].astype(np.float64).ravel()

    K_vc = virtual_cam.K.astype(np.float64)
    # VirtualCamera has R=I (identity) by convention — no self.R attribute.
    R_vc = np.eye(3, dtype=np.float64)
    t_vc = (np.asarray(virtual_cam.t, dtype=np.float64).ravel()
            if virtual_cam.t is not None else np.zeros(3))

    depth_mm = depth_m * 1000.0

    # Sample central grid of B pixels
    margin = 0.15
    u_b = np.linspace(W_b * margin, W_b * (1 - margin), n_grid)
    v_b = np.linspace(H_b * margin, H_b * (1 - margin), n_grid)
    uu, vv = np.meshgrid(u_b, v_b)
    u_flat = uu.ravel()
    v_flat = vv.ravel()

    # B source pixel → 3-D at focus depth (mm)
    K_b_inv = np.linalg.inv(K_b)
    ones = np.ones(len(u_flat))
    uv1 = np.stack([u_flat, v_flat, ones], axis=-1)
    ray_c = (K_b_inv @ uv1.T).T
    P_c = depth_mm * ray_c / ray_c[:, 2:3]

    # B camera frame → world frame
    P_world = (R_b.T @ (P_c - t_b).T).T

    # World → virtual camera frame → canvas pixel
    P_vc = (R_vc @ P_world.T + t_vc[:, None]).T
    valid = P_vc[:, 2] > 0.01
    u_vc = K_vc[0, 0] * P_vc[:, 0] / np.where(valid, P_vc[:, 2], 1.0) + K_vc[0, 2]
    v_vc = K_vc[1, 1] * P_vc[:, 1] / np.where(valid, P_vc[:, 2], 1.0) + K_vc[1, 2]

    pad = 8
    valid &= (u_vc >= pad) & (u_vc < W_vc - pad) & (v_vc >= pad) & (v_vc < H_vc - pad)

    if valid.sum() < 100:
        return 1.0, 1.0, 1.0

    ui_b = u_flat[valid].astype(np.int32)
    vi_b = v_flat[valid].astype(np.int32)
    ui_vc = np.clip(u_vc[valid].astype(np.int32), 0, W_vc - 1)
    vi_vc = np.clip(v_vc[valid].astype(np.int32), 0, H_vc - 1)

    b_pix = b_img[vi_b, ui_b].astype(np.float64)
    ref_pix = reference_canvas[vi_vc, ui_vc].astype(np.float64)

    # Exclude black/clipped/zero-weight pixels in the reference canvas
    b_lum = b_pix.mean(axis=1)
    ref_lum = ref_pix.mean(axis=1)
    good = (b_lum > 512) & (b_lum < 62000) & (ref_lum > 512) & (ref_lum < 62000)

    if good.sum() < 50:
        return 1.0, 1.0, 1.0

    b_g = b_pix[good]
    ref_g = ref_pix[good]

    r_scale = float(np.clip(np.median(ref_g[:, 0] / np.maximum(b_g[:, 0], 1.0)),
                            0.1, 10.0))
    g_scale = float(np.clip(np.median(ref_g[:, 1] / np.maximum(b_g[:, 1], 1.0)),
                            0.1, 10.0))
    b_scale = float(np.clip(np.median(ref_g[:, 2] / np.maximum(b_g[:, 2], 1.0)),
                            0.1, 10.0))

    return r_scale, g_scale, b_scale


def _pick_interp(cam: dict, virtual_cam) -> int:
    """
    Choose cv2 interpolation flag for warping ``cam`` onto ``virtual_cam``.

    When the source camera has the same angular resolution as the canvas
    (resolution_weight ≥ 0.95, i.e. same focal length) the remap is
    essentially 1:1 pixel-for-pixel.  INTER_NEAREST preserves full source
    sharpness with no bilinear blur.  For lower-resolution cameras (wide
    A-cameras warped onto a tele canvas) INTER_LINEAR avoids aliasing.
    """
    rw = resolution_weight(cam, virtual_cam)
    return cv2.INTER_NEAREST if rw >= 0.95 else cv2.INTER_LINEAR


# ---------------------------------------------------------------------------
# MVS depth loader (A-cameras)
# ---------------------------------------------------------------------------

def _load_metric3d_depth(depth_dir: str, cam_name: str, virtual_cam) -> np.ndarray | None:
    """
    Load pre-computed Metric3D V2 depth for a single B-camera and resize to
    the virtual canvas dimensions if necessary.

    Looks for ``<depth_dir>/metric3d_<cam_name>.npz`` (the file written by
    ``lri_run_metric3d.py``).  The depth map is in native source-camera pixel
    space (H_src × W_src), float32, metres.  This function only resizes the
    map to the virtual canvas size; forward-warping into the virtual frame is
    handled by the caller (``_resolve_depth_for_camera`` → ``forward_warp_depth``).

    Returns
    -------
    np.ndarray or None
        float32 (H_src, W_src) depth in metres, or None if the file is absent.
    """
    if depth_dir is None:
        return None
    path = os.path.join(depth_dir, f'metric3d_{cam_name}.npz')
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path)
        return data['depth'].astype(np.float32)
    except Exception as exc:
        import warnings
        warnings.warn(f"Could not load Metric3D depth from {path}: {exc}")
        return None


def _load_mvs_depth(lumen_dir: str, virtual_cam) -> np.ndarray | None:
    """
    Load pre-computed PatchMatch MVS depth map for the A-camera group.

    The file ``<lumen_dir>/depth/mvs_a_cameras.npz`` contains a float32
    (H_mvs, W_mvs) depth map expressed in metres in the virtual camera's
    coordinate frame.  If the map was generated at a reduced scale (e.g.
    --scale 8 → 1/8 resolution) it is upsampled to the virtual canvas size
    using bilinear interpolation before being returned.

    Returns
    -------
    np.ndarray or None
        float32 (H_out, W_out) depth in metres, or None if the file is absent.
    """
    if lumen_dir is None:
        return None
    mvs_path = os.path.join(lumen_dir, 'depth', 'mvs_a_cameras.npz')
    if not os.path.isfile(mvs_path):
        return None
    try:
        data = np.load(mvs_path)
        depth = data['depth'].astype(np.float32)  # (H_mvs, W_mvs)
    except Exception as exc:
        import warnings
        warnings.warn(f"Could not load MVS depth from {mvs_path}: {exc}")
        return None

    H_out, W_out = virtual_cam.H, virtual_cam.W
    if depth.shape != (H_out, W_out):
        # Scale mismatch (typical when MVS was run at --scale > 1).
        # Bilinear resize: nearest-neighbour would introduce aliasing in the
        # remap lookup; bilinear preserves smooth metric depth gradients.
        depth = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_LINEAR)

    return depth


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(frames_dir: str, cam_name: str) -> np.ndarray:
    """
    Load a uint16 PNG frame for *cam_name*.

    Parameters
    ----------
    frames_dir : str
        Directory containing per-camera PNG files named ``<cam_name>.png``.
    cam_name : str
        Camera identifier (e.g. ``'A1'``, ``'B5'``).

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype uint16, BGR channel order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cv2 cannot open it.
    """
    path = os.path.join(frames_dir, f'{cam_name}.png')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"Could not load frame for camera '{cam_name}' at path: {path}"
        )
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img  # uint16 (H, W, 3)


def save_canvas(canvas_f32: np.ndarray, output_path: str) -> None:
    """
    Save a float32 canvas as a 16-bit PNG.

    Parameters
    ----------
    canvas_f32 : np.ndarray
        Shape (H, W, 3) float32 with values in [0, 65535].
    output_path : str
        Destination file path (parent directory must exist).
    """
    canvas_u16 = np.clip(canvas_f32, 0, 65535).astype(np.uint16)
    cv2.imwrite(output_path, canvas_u16)


# ---------------------------------------------------------------------------
# Reference camera selection
# ---------------------------------------------------------------------------

def _select_ref_camera(cameras: dict) -> str:
    """
    Return the name of the wide (mirror_type='NONE') camera closest to the
    centroid of all wide-camera translation vectors.

    Falls back to the first wide camera if only one exists or translations
    are missing.

    If no mirror_type='NONE' cameras exist (e.g. B+C-only L16 shots), falls
    back to the camera with the smallest focal length (widest FOV).
    """
    wide = {n: c for n, c in cameras.items() if c.get('mirror_type', 'NONE') == 'NONE'}
    if not wide:
        # Focal-length fallback: treat smallest-fx cameras as the wide group.
        all_fx = sorted(c['K'][0, 0] for c in cameras.values())
        median_fx = float(np.median(all_fx))
        wide = {n: c for n, c in cameras.items() if c['K'][0, 0] < median_fx}
    if not wide:
        raise ValueError("No wide cameras (mirror_type='NONE') found in cameras dict.")
    if len(wide) == 1:
        return next(iter(wide))

    # Compute centroid of translations
    ts = []
    for c in wide.values():
        t = c.get('t')
        if t is not None:
            ts.append(np.asarray(t, dtype=np.float64).ravel()[:3])

    if not ts:
        return next(iter(wide))  # no translations available

    centroid = np.mean(ts, axis=0)
    best_name = min(
        wide.keys(),
        key=lambda n: np.linalg.norm(
            np.asarray(wide[n].get('t', centroid), dtype=np.float64).ravel()[:3]
            - centroid
        )
    )
    return best_name


# ---------------------------------------------------------------------------
# Per-camera depth resolution
# ---------------------------------------------------------------------------

def _resolve_depth_for_camera(
    cam_name: str,
    cam: dict,
    virtual_cam,
    lumen_dir: str,
    fallback_canvas_depth,
    fixed_fallback_m: float = None,
) -> np.ndarray:
    """
    Return the best available canvas-space depth (H_out, W_out) float32 metres for
    one source camera, using this priority:

      1. Per-camera depth NPZ at <lumen_dir>/depth/<cam_name>.npz, forward-warped
         to virtual camera canvas space using THIS camera's own calibration.
      2. fallback_canvas_depth (pre-loaded canvas-space depth passed in by the caller).
      3. If fixed_fallback_m is given and steps 1–2 produce nothing, return a constant
         depth plane at fixed_fallback_m metres.  Used for B cameras in wide mode
         when no per-camera depth file exists.

    IMPORTANT: The fused PNG (fused_*cams_*depth.png) is in the source reference
    camera's pixel space (typically A1's 4160×3120 sensor grid) — NOT virtual camera
    canvas space.  It must NOT be used here for cameras other than the one it was
    derived from, because forward-warping A1's depth with B4's calibration gives
    nonsense.  The fused PNG is handled separately in load_depth_for_canvas (called
    by the top-level pipeline) and passed in as fallback_canvas_depth.

    Forward-warping is correct: it reprojects each source pixel's 3-D point from the
    source camera's frame into the virtual camera frame, so off-axis pixels get the
    right depth rather than the wrong depth from a naive pixel-grid resize.
    """
    if lumen_dir is None:
        if fixed_fallback_m is not None and fallback_canvas_depth is None:
            from lri_depth_loader import flat_plane_depth
            return flat_plane_depth(virtual_cam, fixed_fallback_m)
        return fallback_canvas_depth

    # Only use the per-camera NPZ — NOT the fused PNG (which is in another camera's space)
    import os
    npz_path = os.path.join(lumen_dir, 'depth', f'{cam_name}.npz')
    if not os.path.isfile(npz_path):
        # For B-cameras: try Metric3D V2 depth before falling back to a flat plane.
        # metric3d_{cam}.npz lives next to the Depth Pro NPZ in the same depth dir.
        if cam_name.startswith('B'):
            m3d = _load_metric3d_depth(os.path.join(lumen_dir, 'depth'), cam_name, virtual_cam)
            if m3d is not None:
                return forward_warp_depth(m3d, cam, virtual_cam)
        if fixed_fallback_m is not None and fallback_canvas_depth is None:
            from lri_depth_loader import flat_plane_depth
            return flat_plane_depth(virtual_cam, fixed_fallback_m)
        return fallback_canvas_depth

    try:
        data = np.load(npz_path)
        raw_depth = data['depth'].astype(np.float32)
    except Exception:
        if fixed_fallback_m is not None and fallback_canvas_depth is None:
            from lri_depth_loader import flat_plane_depth
            return flat_plane_depth(virtual_cam, fixed_fallback_m)
        return fallback_canvas_depth

    return forward_warp_depth(raw_depth, cam, virtual_cam)


# ---------------------------------------------------------------------------
# Homography-based remap helper (MOVABLE B cameras)
# ---------------------------------------------------------------------------

def _remap_from_homography(
    H_vc_to_b: np.ndarray,
    row_start: int,
    H_tile: int,
    W_out: int,
    cam: dict,
) -> tuple:
    """
    Generate remap arrays from a virtual-canvas → B-camera homography.

    Parameters
    ----------
    H_vc_to_b : 3×3 float64 — maps vc pixel (u_global, v_global) → B pixel
    row_start  : first row of this tile in global virtual-canvas coordinates
    H_tile     : height of tile in rows
    W_out      : width of virtual canvas
    cam        : camera dict (needs 'W', 'H')

    Returns
    -------
    map_x, map_y : float32 (H_tile, W_out) — B-camera pixel coordinates
    mask         : bool   (H_tile, W_out) — True where within B-sensor bounds
    """
    us = np.arange(W_out, dtype=np.float64)
    vs = np.arange(row_start, row_start + H_tile, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)                             # (H_tile, W_out)
    N = H_tile * W_out
    pts = np.stack([uu.ravel(), vv.ravel(), np.ones(N)], axis=1).T  # (3, N)
    pts_b = H_vc_to_b @ pts
    # Perspective divide (guard against near-zero w)
    w = pts_b[2:3, :]
    w = np.where(np.abs(w) > 1e-8, w, 1e-8)
    pts_b = pts_b / w
    map_x = pts_b[0].reshape(H_tile, W_out).astype(np.float32)
    map_y = pts_b[1].reshape(H_tile, W_out).astype(np.float32)
    mask = (
        (map_x >= 2.0) & (map_x <= cam['W'] - 2) &
        (map_y >= 2.0) & (map_y <= cam['H'] - 2)
    )
    return map_x, map_y, mask


def _compute_movable_homographies(
    cameras: dict,
    source_images: dict,          # {cam_name: uint16 ndarray} — already WB-corrected
    ref_name: str,
    virtual_cam,
) -> dict:
    """
    For every MOVABLE B camera present in source_images, run LightGlue against
    the reference A camera to obtain a virtual-canvas → B-sensor homography.

    Returns
    -------
    dict mapping cam_name → H_vc_to_b (3×3 float64), or empty dict on failure.
    """
    import warnings

    result = {}
    ref_img_f32 = source_images.get(ref_name)
    if ref_img_f32 is None:
        warnings.warn(f"Reference camera {ref_name} not in source_images; "
                      "cannot compute MOVABLE homographies.")
        return result
    ref_img_f32 = ref_img_f32.astype(np.float32)

    K_ref = cameras[ref_name]['K'].astype(np.float64)
    K_vc  = virtual_cam.K.astype(np.float64)
    # S maps ref-camera pixels to virtual-canvas pixels
    S = K_vc @ np.linalg.inv(K_ref)

    for cam_name, cam in cameras.items():
        if cam.get('mirror_type') != 'MOVABLE':
            continue
        if cam_name not in source_images:
            continue
        print(f"  LightGlue homography for {cam_name}...", end=' ', flush=True)
        try:
            b_img_f32 = source_images[cam_name].astype(np.float32)
            H_b_to_ref = estimate_b_to_ref_homography(
                ref_img_f32, b_img_f32,
                cam_ref=cameras[ref_name], cam_src=cam,
            )
            if H_b_to_ref is None:
                print("FAILED (no matches)", flush=True)
                continue
            # H_b_to_vc = S @ H_b_to_ref maps B pixel → vc pixel
            H_b_to_vc = S @ H_b_to_ref
            H_vc_to_b = np.linalg.inv(H_b_to_vc)
            H_vc_to_b /= H_vc_to_b[2, 2]
            result[cam_name] = H_vc_to_b
            print("OK", flush=True)
        except Exception as exc:
            print(f"ERROR ({exc})", flush=True)

    return result


# ---------------------------------------------------------------------------
# Core assembly — single tile
# ---------------------------------------------------------------------------

def _assemble_tile(
    row_start: int,
    row_end: int,
    source_images: dict,        # preloaded {cam_name: uint16 ndarray}
    cameras: dict,
    virtual_cam,
    per_cam_depth: dict,        # {cam_name: float32 (H_out, W_out)} or empty
    remap_cache_dir,
) -> np.ndarray:
    """
    Assemble rows [row_start, row_end) of the output canvas.

    per_cam_depth: per-camera canvas-space depth maps (H_out, W_out) float32 metres.
    Each camera uses its own forward-warped depth when available (correct coordinate
    system), falling back to None (flat-plane) otherwise.

    Returns float32 (row_end-row_start, W_out, 3).
    """
    H_tile = row_end - row_start
    W_out = virtual_cam.W

    canvas = np.zeros((H_tile, W_out, 3), dtype=np.float64)
    weight_sum = np.zeros((H_tile, W_out), dtype=np.float64)

    # Build a lightweight virtual-cam-like object for this tile so that
    # compute_remap generates maps of shape (H_tile, W_out).
    class _TileVCam:
        pass

    tvc = _TileVCam()
    tvc.K = virtual_cam.K.copy()
    tvc.K[1, 2] -= row_start          # shift principal point by tile offset
    tvc.W = W_out
    tvc.H = H_tile
    tvc.R = getattr(virtual_cam, 'R', np.eye(3))
    tvc.t = getattr(virtual_cam, 't', np.zeros(3))

    vc_mode = getattr(virtual_cam, 'mode', 'tele')

    for cam_name, cam in cameras.items():
        if cam_name not in source_images:
            continue

        # Quick ROI pre-screen: skip this camera entirely if its canvas-space
        # bounding box doesn't overlap the current tile row range.  This is
        # especially valuable for telephoto cameras (C cameras cover ~4% of the
        # wide canvas) which only appear in a few of the 18 tiles.
        from lri_camera_remap import _camera_canvas_roi
        _, roi_y0, _, roi_y1 = _camera_canvas_roi(virtual_cam, cam)
        if roi_y0 >= row_end or roi_y1 <= row_start:
            continue  # This camera has no coverage in this tile row range

        # 1. Source image already loaded and WB-corrected
        src_img = source_images[cam_name]

        # 2. Remap generation — all cameras use calibration-based compute_remap.
        #    MOVABLE cameras have correct virtual poses after the convention fix
        #    in compute_movable_mirror_pose (R_proto transposed to COLMAP).
        # 2a. Slice this camera's depth to the current tile (or None)
        cam_depth = per_cam_depth.get(cam_name)
        depth_tile = cam_depth[row_start:row_end, :] if cam_depth is not None else None

        # 2b. Load or compute remap for this tile
        tile_suffix = f'{cam_name}_tile_{row_start}_{row_end}'
        map_x, map_y, mask = None, None, None
        if remap_cache_dir:
            cached = load_remap_cache(remap_cache_dir, tile_suffix)
            if cached is not None:
                map_x, map_y, mask = cached

        if map_x is None:
            map_x, map_y, mask = compute_remap(tvc, cam, depth_tile)
            if remap_cache_dir:
                os.makedirs(remap_cache_dir, exist_ok=True)
                cache_remap(remap_cache_dir, tile_suffix, map_x, map_y, mask)

        # 3. Compute confidence map (H_tile, W_out)
        conf = compute_confidence(src_img, cam, tvc, mask, map_x, map_y)

        # 4. Warp source image to tile canvas space
        # Use INTER_NEAREST for same-resolution cameras (tele on tele canvas) to
        # preserve full source sharpness; INTER_LINEAR otherwise to avoid aliasing.
        warped = apply_remap(src_img.astype(np.float32), map_x, map_y,
                             interpolation=_pick_interp(cam, tvc))  # (H_tile, W_out, 3)

        # 6. Accumulate weighted sum
        canvas += warped.astype(np.float64) * conf[:, :, None]
        weight_sum += conf.astype(np.float64)

    # 7. Normalise
    weight_sum = np.maximum(weight_sum, 1e-8)
    tile_result = (canvas / weight_sum[:, :, None]).astype(np.float32)
    return tile_result


# ---------------------------------------------------------------------------
# Full-frame assembly — public entry point
# ---------------------------------------------------------------------------

def assemble_canvas(
    frames_dir: str,
    cameras: dict,
    virtual_cam,             # VirtualCamera
    depth_map,               # float32 (H_out, W_out) metres, or None (legacy fallback)
    remap_cache_dir=None,    # path to cache remap arrays, or None
    apply_wb_flag: bool = True,
    apply_ccm_flag: bool | None = None,  # deprecated alias for apply_wb_flag
    tile_rows=None,          # int or None; None = full-frame (single pass)
    lumen_dir: str = None,   # lumen directory for per-camera depth maps
    focus_distance_m: float | None = None,  # calibrated focus plane (metres); when set,
                             # all cameras use a flat depth plane at this distance —
                             # pure geometry merge, no MVS / DepthPro required.
    reference_canvas: np.ndarray | None = None,  # pre-computed all-A canvas for
                             # B-camera color correction; when provided, B cameras are
                             # color-matched to this canvas rather than to A1 source.
) -> np.ndarray:             # float32 (H_out, W_out, 3)
    """
    Assemble the virtual output canvas by warping and blending all cameras.

    Parameters
    ----------
    frames_dir : str
        Directory containing ``<cam_name>.png`` files.
    cameras : dict
        Camera dict (K, R, t, W, H, mirror_type, analog_gain, exposure_ns)
        keyed by camera name.  Exposure/gain fields are produced by
        ``load_cameras()`` from lri_fuse_image.py.
    virtual_cam : VirtualCamera
        Target projection (defines H_out, W_out, K).
    depth_map : np.ndarray or None
        Legacy single-camera canvas-space depth (H_out, W_out) float32 metres.
        Used only when per-camera depth is unavailable.  Pass ``None`` to fall
        back to flat-plane inside compute_remap.
    remap_cache_dir : str or None
        Directory for caching remap arrays (.npz).  ``None`` disables caching.
    apply_wb_flag : bool
        Apply per-camera white-balance and exposure normalization before
        warping.  Uses gray-world WB gains relative to the reference camera
        and exposure_ns * analog_gain to compute EV offsets.
        No cross-channel mixing — purely three independent scale factors.
    apply_ccm_flag : bool or None
        Deprecated.  When set, overrides ``apply_wb_flag`` for backward
        compatibility with callers that used the old CCM interface.  The CCM
        has been replaced by the simpler per-channel WB+exposure normalization.
    tile_rows : int or None
        Number of output rows per processing tile.  ``None`` processes the
        entire canvas in one pass (~1.8 GB RAM for 10449×7795).  Use e.g.
        ``tile_rows=500`` on 16 GB machines to cap working-set to ~900 MB.
    lumen_dir : str or None
        Lumen directory.  When provided, per-camera depth maps are loaded and
        forward-warped into the virtual camera canvas frame before being used
        in compute_remap.  This replaces the legacy resize-and-share approach
        that used A1's depth with A1's coordinate system for all cameras.
    focus_distance_m : float or None
        Calibrated focus plane distance in metres (read from LRI metadata,
        converting from mm: ``focus_distance_m = hall_focus_mm / 1000``).
        When provided, ALL cameras use a flat depth plane at this distance —
        the merge becomes a pure geometry operation driven only by K, R, t
        and the physical focus plane.  MVS and per-camera DepthPro maps are
        NOT loaded.  This is the correct Light.co architecture: depth (scene
        estimation) is used only for post-processing bokeh / re-focus, not
        for the multi-camera pixel merge itself.

    Returns
    -------
    np.ndarray
        Float32 (H_out, W_out, 3) fused canvas in uint16 intensity scale.
    """
    # Backward-compat: apply_ccm_flag was the old parameter name
    if apply_ccm_flag is not None:
        import warnings
        warnings.warn(
            "apply_ccm_flag is deprecated; use apply_wb_flag instead. "
            "The CCM has been replaced by per-channel WB+exposure normalization.",
            DeprecationWarning, stacklevel=2,
        )
        apply_wb_flag = bool(apply_ccm_flag)

    H_out = virtual_cam.H
    W_out = virtual_cam.W

    # Identify reference wide camera for WB normalization
    ref_name = _select_ref_camera(cameras)
    ref_img = load_image(frames_dir, ref_name)  # uint16

    vc_mode = getattr(virtual_cam, 'mode', 'tele')

    if tile_rows is None:
        # ── Full-frame path (simple, single pass) ─────────────────────────
        canvas = np.zeros((H_out, W_out, 3), dtype=np.float64)
        weight_sum = np.zeros((H_out, W_out), dtype=np.float64)

        # Depth strategy (full-frame path):
        #   focus_distance_m set  → flat plane at focus distance for ALL cameras
        #                           (pure geometry merge; no MVS / DepthPro)
        #   otherwise             → MVS map when available, else per-camera DepthPro
        if focus_distance_m is not None:
            from lri_depth_loader import flat_plane_depth
            mvs_depth = None
            _focus_plane = flat_plane_depth(virtual_cam, focus_distance_m)
            print(f"  geometry merge: flat plane at {focus_distance_m:.3f}m "
                  f"(no MVS / DepthPro)", flush=True)
        else:
            _focus_plane = None
            mvs_depth = _load_mvs_depth(lumen_dir, virtual_cam)
            if mvs_depth is not None:
                print(f"  MVS depth detected — using mvs_a_cameras.npz for A cameras "
                      f"(shape {mvs_depth.shape}, range "
                      f"{mvs_depth[mvs_depth > 0].min():.1f}–{mvs_depth.max():.1f}m)",
                      flush=True)

        # Preload all source images for homography computation and the main loop.
        print("  preloading source images (full-frame)...", flush=True)
        source_images_ff = {}
        for cam_name, cam in cameras.items():
            frame_path = os.path.join(frames_dir, f'{cam_name}.png')
            if not os.path.exists(frame_path):
                continue
            src_img = load_image(frames_dir, cam_name)
            if apply_wb_flag and cam_name != ref_name:
                is_b_cam = cam.get('mirror_type', 'NONE') != 'NONE'
                if is_b_cam and focus_distance_m is not None:
                    # Geometry-grounded overlap color correction.
                    # r_s/g_s/b_s capture BOTH EV and sensor-color differences —
                    # do NOT also apply exposure_ev_scale (double-counts EV).
                    if reference_canvas is not None:
                        # Two-pass mode: use the pre-computed all-A canvas as reference.
                        # This accounts for all A cameras, not just A1's source image.
                        r_s, g_s, b_s = _compute_b_to_canvas_color_correction(
                            src_img.astype(np.float32), cam,
                            virtual_cam, reference_canvas, focus_distance_m)
                        method = 'canvas-ref'
                    else:
                        r_s, g_s, b_s = _compute_b_to_a_color_correction(
                            src_img.astype(np.float32), ref_img,
                            cam, cameras[ref_name], focus_distance_m)
                        method = 'a1-ref'
                    scale = np.array([r_s, g_s, b_s], dtype=np.float32)
                    src_img = np.clip(src_img.astype(np.float32) * scale[None, None, :],
                                      0.0, 65535.0).astype(np.float32)
                    print(f"    {cam_name} overlap-correction ({method}): "
                          f"R×{r_s:.3f} G×{g_s:.3f} B×{b_s:.3f}",
                          flush=True)
                else:
                    # A cameras: EV-only (same sensor/scene — gray-world adds noise).
                    src_img = apply_wb_exposure(src_img, cam, cameras[ref_name],
                                                ref_img=ref_img,
                                                use_gray_world=False)
            source_images_ff[cam_name] = src_img

        for cam_name, cam in cameras.items():
            if cam_name not in source_images_ff:
                continue
            src_img = source_images_ff[cam_name]

            # Resolve per-camera canvas-space depth.
            if _focus_plane is not None:
                # Pure geometry merge: all cameras use the calibrated focus plane.
                cam_depth = _focus_plane
            elif cam_name[:1] == 'A' and mvs_depth is not None:
                cam_depth = mvs_depth
            else:
                _b_fixed_depth = (
                    5.0
                    if (vc_mode == 'wide' and cam_name[:1] == 'B')
                    else None
                )
                cam_depth = _resolve_depth_for_camera(
                    cam_name, cam, virtual_cam, lumen_dir, depth_map,
                    fixed_fallback_m=_b_fixed_depth,
                )

            map_x, map_y, mask = None, None, None
            if remap_cache_dir:
                cached = load_remap_cache(remap_cache_dir, cam_name)
                if cached is not None:
                    map_x, map_y, mask = cached
            if map_x is None:
                map_x, map_y, mask = compute_remap(virtual_cam, cam, cam_depth)
                if remap_cache_dir:
                    os.makedirs(remap_cache_dir, exist_ok=True)
                    cache_remap(remap_cache_dir, cam_name, map_x, map_y, mask)

            # Confidence map
            conf = compute_confidence(src_img, cam, virtual_cam, mask, map_x, map_y)

            # Warp source image to output canvas
            warped = apply_remap(src_img.astype(np.float32), map_x, map_y,
                                 interpolation=_pick_interp(cam, virtual_cam))

            # 7. Accumulate weighted sum
            canvas += warped.astype(np.float64) * conf[:, :, None]
            weight_sum += conf.astype(np.float64)

        # 8. Normalise
        weight_sum = np.maximum(weight_sum, 1e-8)
        result = (canvas / weight_sum[:, :, None]).astype(np.float32)
        return result

    else:
        # ── Tiled path — cap peak RAM usage ───────────────────────────────
        # Preload all source images once (avoid 260 disk reads for 26 tiles × 10 cameras)
        print("  preloading source images...", flush=True)
        source_images = {}
        for cam_name, cam in cameras.items():
            frame_path = os.path.join(frames_dir, f'{cam_name}.png')
            if not os.path.exists(frame_path):
                continue
            src_img = load_image(frames_dir, cam_name)
            # Apply per-channel WB + exposure normalization (replaces CCM).
            # Skip for the reference camera itself.
            # A cameras: EV-only (same sensor/scene — gray-world adds noise).
            # B/C cameras: geometry-grounded overlap correction when focus_distance_m
            #   is set; otherwise fall back to gray-world + EV.
            if apply_wb_flag and cam_name != ref_name:
                is_b_cam = cam.get('mirror_type', 'NONE') != 'NONE'
                if is_b_cam and focus_distance_m is not None:
                    # r_s/g_s/b_s capture EV + sensor-color differences together.
                    # Do NOT also apply exposure_ev_scale — that would double-count EV.
                    if reference_canvas is not None:
                        r_s, g_s, b_s = _compute_b_to_canvas_color_correction(
                            src_img.astype(np.float32), cam,
                            virtual_cam, reference_canvas, focus_distance_m)
                        method = 'canvas-ref'
                    else:
                        r_s, g_s, b_s = _compute_b_to_a_color_correction(
                            src_img.astype(np.float32), ref_img,
                            cam, cameras[ref_name], focus_distance_m)
                        method = 'a1-ref'
                    scale = np.array([r_s, g_s, b_s], dtype=np.float32)
                    src_img = np.clip(src_img.astype(np.float32) * scale[None, None, :],
                                      0.0, 65535.0).astype(np.float32)
                    print(f"    {cam_name} overlap-correction ({method}): "
                          f"R×{r_s:.3f} G×{g_s:.3f} B×{b_s:.3f}",
                          flush=True)
                else:
                    src_img = apply_wb_exposure(src_img, cam, cameras[ref_name],
                                                ref_img=ref_img,
                                                use_gray_world=is_b_cam)
            source_images[cam_name] = src_img
        active_cams = list(source_images.keys())
        print(f"  loaded {len(source_images)} cameras: {active_cams}", flush=True)

        # Pre-compute per-camera canvas-space depth maps.
        # Done once before the tile loop — each tile just slices the relevant rows.
        # A cameras: use MVS depth (single map, already in virtual camera frame) when
        # mvs_a_cameras.npz is present, bypassing per-camera Depth Pro NPZ forward-warp.
        # B cameras: unchanged — per-camera Depth Pro NPZ or fixed 5.0m plane.
        print("  resolving per-camera depth maps...", flush=True)
        # Depth strategy (tiled path):
        #   focus_distance_m set  → flat plane for ALL cameras (pure geometry merge)
        #   otherwise             → MVS map when available, else per-camera DepthPro
        if focus_distance_m is not None:
            from lri_depth_loader import flat_plane_depth
            _focus_plane = flat_plane_depth(virtual_cam, focus_distance_m)
            print(f"  geometry merge: flat plane at {focus_distance_m:.3f}m "
                  f"(no MVS / DepthPro)", flush=True)
            per_cam_depth = {n: _focus_plane for n in source_images}
        else:
            _focus_plane = None
            mvs_depth = _load_mvs_depth(lumen_dir, virtual_cam)
            if mvs_depth is not None:
                valid_mvs = mvs_depth[mvs_depth > 0]
                print(
                    f"  MVS depth detected — using mvs_a_cameras.npz for A cameras "
                    f"(shape {mvs_depth.shape}, range "
                    f"{valid_mvs.min():.1f}–{mvs_depth.max():.1f}m)",
                    flush=True,
                )
            per_cam_depth = {}
            for cam_name, cam in cameras.items():
                if cam_name not in source_images:
                    continue
                if cam_name[:1] == 'A' and mvs_depth is not None:
                    # Use single MVS map directly for all A cameras — no forward-warp needed.
                    per_cam_depth[cam_name] = mvs_depth
                else:
                    # B cameras (or A cameras when MVS unavailable): existing per-camera path.
                    # In wide mode, B cameras without a depth NPZ fall back to 5.0m rather
                    # than A1's depth (which is in a different coordinate system).
                    _b_fixed_depth = (
                        5.0
                        if (vc_mode == 'wide' and cam_name[:1] == 'B')
                        else None
                    )
                    cam_depth = _resolve_depth_for_camera(
                        cam_name, cam, virtual_cam, lumen_dir, depth_map,
                        fixed_fallback_m=_b_fixed_depth,
                    )
                    per_cam_depth[cam_name] = cam_depth  # None → flat-plane; array → per-cam
        n_with_depth = sum(1 for d in per_cam_depth.values() if d is not None)
        print(f"  depth: {n_with_depth}/{len(per_cam_depth)} cameras have per-camera depth",
              flush=True)

        # Build tile ranges and submit in parallel using threads.
        # When MPS (Metal GPU) is active, the GPU executes each remap serially
        # inside compute_remap — sending work from multiple threads just contends
        # on the single MPS command queue and is slower than serial dispatch.
        # Use n_workers=1 when MPS is available; multi-thread only for pure NumPy.
        import concurrent.futures, os as _os
        from lri_camera_remap import _MPS_AVAILABLE as _mps_flag
        import lri_camera_remap as _remap_mod
        if _remap_mod._MPS_AVAILABLE is None:
            try:
                import torch
                _remap_mod._MPS_AVAILABLE = torch.backends.mps.is_available()
            except ImportError:
                _remap_mod._MPS_AVAILABLE = False
        _using_mps = _remap_mod._MPS_AVAILABLE
        if _using_mps:
            n_workers = int(_os.environ.get('LRI_BLEND_WORKERS', '1'))
        else:
            n_workers = min(
                int(_os.environ.get('LRI_BLEND_WORKERS', '0')) or _os.cpu_count() or 4,
                10,  # cap at 10 — beyond this, memory bandwidth becomes the limit
            )

        tile_ranges = []
        row = 0
        while row < H_out:
            tile_ranges.append((row, min(row + tile_rows, H_out)))
            row += tile_rows

        print(f"  assembling {len(tile_ranges)} tiles on {n_workers} threads", flush=True)
        strips_by_start: dict = {}

        def _run_tile(row_start_row_end):
            rs, re = row_start_row_end
            return rs, _assemble_tile(
                row_start=rs,
                row_end=re,
                source_images=source_images,
                cameras=cameras,
                virtual_cam=virtual_cam,
                per_cam_depth=per_cam_depth,
                remap_cache_dir=remap_cache_dir,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_tile, rng): rng for rng in tile_ranges}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                rs, strip = fut.result()
                strips_by_start[rs] = strip
                done += 1
                print(f"  [{done}/{len(tile_ranges)}] tile {rs}–{rs+strip.shape[0]} done",
                      flush=True)

        strips = [strips_by_start[rs] for rs, _ in tile_ranges]
        return np.concatenate(strips, axis=0)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from lri_fuse_image import load_cameras
    from lri_virtual_camera import VirtualCamera
    from lri_depth_loader import load_depth_for_canvas, flat_plane_depth

    parser = argparse.ArgumentParser(
        description='Assemble L16 virtual canvas from calibrated frames.'
    )
    parser.add_argument(
        '--cal',
        default='/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_cal'
                '/calibration.json/calibration.json',
        help='Path to calibration JSON',
    )
    parser.add_argument(
        '--frames',
        default='/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_lumen/frames',
        help='Directory containing per-camera PNG frames',
    )
    parser.add_argument(
        '--lumen',
        default='/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_lumen',
        help='Lumen directory (for depth map discovery)',
    )
    parser.add_argument(
        '--output',
        default='/Volumes/Dev/Light_Work_scratch/canvas_test.png',
        help='Output PNG path',
    )
    parser.add_argument(
        '--cache',
        default=None,
        help='Remap cache directory (optional)',
    )
    parser.add_argument(
        '--tile-rows',
        type=int,
        default=None,
        help='Process canvas in horizontal tiles of this many rows (RAM saver)',
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='Use 1/4-scale virtual camera for a quick smoke test',
    )
    args = parser.parse_args()

    print("Loading cameras …")
    cameras = load_cameras(args.cal)

    print("Building virtual camera …")
    vc = VirtualCamera(cameras)

    if args.small:
        # Shrink virtual camera to 1/4 scale for a quick smoke test
        vc.K = vc.K.copy()
        vc.K[0, 0] /= 4
        vc.K[1, 1] /= 4
        vc.K[0, 2] /= 4
        vc.K[1, 2] /= 4
        vc.W = vc.W // 4
        vc.H = vc.H // 4
        print(f"Small mode: {vc.W}×{vc.H}")

    print("Loading depth map …")
    ref_name = _select_ref_camera(cameras)
    depth = load_depth_for_canvas(args.lumen, vc, ref_name, cameras[ref_name])
    if depth is None:
        print("  No depth found — using flat-plane fallback.")
        # Pass None so compute_remap uses its internal _DEFAULT_DEPTH_MM constant
        # rather than a pre-allocated flat array.  This avoids loading a
        # 1.5 GB float32 flat-plane depth into every camera's use_depth=True path.
        depth = None

    print(f"Assembling canvas {vc.W}×{vc.H} …")
    result = assemble_canvas(
        frames_dir=args.frames,
        cameras=cameras,
        virtual_cam=vc,
        depth_map=depth,
        remap_cache_dir=args.cache,
        tile_rows=args.tile_rows,
        lumen_dir=args.lumen,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_canvas(result, args.output)
    print(f"Saved: {args.output}  shape={result.shape}  max={result.max():.0f}")
