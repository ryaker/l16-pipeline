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
from lri_ccm import estimate_ccm_from_cameras, apply_ccm
from lri_depth_loader import forward_warp_depth, _load_raw_depth


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
    """
    wide = {n: c for n, c in cameras.items() if c.get('mirror_type', 'NONE') == 'NONE'}
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

        # Skip movable-mirror cameras in tele mode: their factory calibration R
        # is stale (mirror shifted post-calibration), so compute_remap gives
        # wrong coords.  These cameras need LightGlue image-based alignment.
        # In wide mode we include them (after a one-time warning) because the
        # B cameras contribute additional resolution; depth-guided warping
        # compensates for coarse alignment errors.
        if cam.get('mirror_type', 'NONE') == 'MOVABLE':
            if vc_mode != 'wide':
                continue
            # Wide mode: emit one-time warning per camera then proceed.
            import warnings
            warnings.warn(
                f"Camera {cam_name} has a MOVABLE mirror — factory R calibration "
                "may be stale. Alignment in wide mode may be imperfect.",
                stacklevel=2,
            )

        # 1. Source image already loaded and CCM-corrected
        src_img = source_images[cam_name]

        # 2. Slice this camera's depth to the current tile (or None for flat-plane)
        cam_depth = per_cam_depth.get(cam_name)
        depth_tile = cam_depth[row_start:row_end, :] if cam_depth is not None else None

        # 3. Load or compute remap for this tile
        # Cache key encodes tile extents so different tiles don't collide.
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

        # 4. Compute confidence map (H_tile, W_out)
        conf = compute_confidence(src_img, cam, tvc, mask, map_x, map_y)

        # 5. Warp source image to tile canvas space
        warped = apply_remap(src_img.astype(np.float32), map_x, map_y)  # (H_tile, W_out, 3)

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
    apply_ccm_flag: bool = True,
    tile_rows=None,          # int or None; None = full-frame (single pass)
    lumen_dir: str = None,   # lumen directory for per-camera depth maps
) -> np.ndarray:             # float32 (H_out, W_out, 3)
    """
    Assemble the virtual output canvas by warping and blending all cameras.

    Parameters
    ----------
    frames_dir : str
        Directory containing ``<cam_name>.png`` files.
    cameras : dict
        Camera dict (K, R, t, W, H, mirror_type) keyed by camera name.
    virtual_cam : VirtualCamera
        Target projection (defines H_out, W_out, K).
    depth_map : np.ndarray or None
        Legacy single-camera canvas-space depth (H_out, W_out) float32 metres.
        Used only when per-camera depth is unavailable.  Pass ``None`` to fall
        back to flat-plane inside compute_remap.
    remap_cache_dir : str or None
        Directory for caching remap arrays (.npz).  ``None`` disables caching.
    apply_ccm_flag : bool
        Apply Colour-Correction Matrix to mirror cameras (GLUED / MOVABLE).
        The CCM is now exposure-neutral (chromaticity-only) so this is safe to
        leave enabled — it will no longer reduce B4's sharpness.
    tile_rows : int or None
        Number of output rows per processing tile.  ``None`` processes the
        entire canvas in one pass (~1.8 GB RAM for 10449×7795).  Use e.g.
        ``tile_rows=500`` on 16 GB machines to cap working-set to ~900 MB.
    lumen_dir : str or None
        Lumen directory.  When provided, per-camera depth maps are loaded and
        forward-warped into the virtual camera canvas frame before being used
        in compute_remap.  This replaces the legacy resize-and-share approach
        that used A1's depth with A1's coordinate system for all cameras.

    Returns
    -------
    np.ndarray
        Float32 (H_out, W_out, 3) fused canvas in uint16 intensity scale.
    """
    H_out = virtual_cam.H
    W_out = virtual_cam.W

    # Identify reference wide camera for CCM estimation
    ref_name = _select_ref_camera(cameras)
    ref_img = load_image(frames_dir, ref_name)  # uint16

    vc_mode = getattr(virtual_cam, 'mode', 'tele')

    if tile_rows is None:
        # ── Full-frame path (simple, single pass) ─────────────────────────
        canvas = np.zeros((H_out, W_out, 3), dtype=np.float64)
        weight_sum = np.zeros((H_out, W_out), dtype=np.float64)

        for cam_name, cam in cameras.items():
            # Skip movable-mirror cameras in tele mode.
            # In wide mode, include them with a warning (stale R calibration).
            if cam.get('mirror_type', 'NONE') == 'MOVABLE':
                if vc_mode != 'wide':
                    continue
                import warnings
                warnings.warn(
                    f"Camera {cam_name} has a MOVABLE mirror — factory R calibration "
                    "may be stale. Alignment in wide mode may be imperfect.",
                    stacklevel=2,
                )
            frame_path = os.path.join(frames_dir, f'{cam_name}.png')
            if not os.path.exists(frame_path):
                continue

            # 1. Load source image
            src_img = load_image(frames_dir, cam_name)  # uint16

            # 2. Apply CCM to GLUED mirror cameras only (exposure-neutral chromaticity fix)
            if apply_ccm_flag and cam.get('mirror_type', 'NONE') not in ('NONE', 'MOVABLE'):
                ccm = estimate_ccm_from_cameras(src_img, ref_img, cam, cameras[ref_name])
                src_img = apply_ccm(src_img, ccm)

            # 3. Resolve per-camera canvas-space depth (forward-warped from source space).
            # In wide mode, B cameras without a depth NPZ fall back to a fixed 5.0m plane
            # rather than A1's depth map (which is in a different coordinate system).
            _b_fixed_depth = (
                5.0
                if (vc_mode == 'wide' and cam_name[:1] == 'B')
                else None
            )
            cam_depth = _resolve_depth_for_camera(
                cam_name, cam, virtual_cam, lumen_dir, depth_map,
                fixed_fallback_m=_b_fixed_depth,
            )

            # 4. Load or compute remap
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

            # 5. Compute confidence map
            conf = compute_confidence(src_img, cam, virtual_cam, mask, map_x, map_y)

            # 6. Warp source image to output canvas
            warped = apply_remap(src_img.astype(np.float32), map_x, map_y)

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
        _movable_warned: set = set()
        for cam_name, cam in cameras.items():
            # Skip movable-mirror cameras in tele mode.
            # In wide mode, include them with a per-camera warning.
            if cam.get('mirror_type', 'NONE') == 'MOVABLE':
                if vc_mode != 'wide':
                    continue
                if cam_name not in _movable_warned:
                    import warnings
                    warnings.warn(
                        f"Camera {cam_name} has a MOVABLE mirror — factory R calibration "
                        "may be stale. Alignment in wide mode may be imperfect.",
                        stacklevel=2,
                    )
                    _movable_warned.add(cam_name)
            frame_path = os.path.join(frames_dir, f'{cam_name}.png')
            if not os.path.exists(frame_path):
                continue
            src_img = load_image(frames_dir, cam_name)
            # Apply CCM to GLUED mirror cameras (exposure-neutral chromaticity fix)
            if apply_ccm_flag and cam.get('mirror_type', 'NONE') not in ('NONE', 'MOVABLE'):
                ccm = estimate_ccm_from_cameras(src_img, ref_img, cam, cameras[ref_name])
                src_img = apply_ccm(src_img, ccm)
            source_images[cam_name] = src_img
        active_cams = list(source_images.keys())
        print(f"  loaded {len(source_images)} cameras: {active_cams}", flush=True)

        # Pre-compute per-camera canvas-space depth maps (forward-warped from source space).
        # Done once before the tile loop — each tile just slices the relevant rows.
        print("  resolving per-camera depth maps...", flush=True)
        per_cam_depth = {}
        for cam_name, cam in cameras.items():
            if cam_name not in source_images:
                continue
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

        strips = []
        row = 0
        while row < H_out:
            row_end = min(row + tile_rows, H_out)
            print(f"  tile rows {row}–{row_end} / {H_out}", flush=True)
            strip = _assemble_tile(
                row_start=row,
                row_end=row_end,
                source_images=source_images,
                cameras=cameras,
                virtual_cam=virtual_cam,
                per_cam_depth=per_cam_depth,
                remap_cache_dir=remap_cache_dir,
            )
            strips.append(strip)
            row = row_end

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
        depth = flat_plane_depth(vc)

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
