"""
lri_merge.py — Symmetric multi-camera merge for the Light L16.

Merges a group of same-focal-length cameras into a single image at their
native pixel density, centered on the group centroid.  No camera is picked
as a reference.  All participating cameras contribute equally through
confidence-weighted blending.

Pipeline per camera (independent):
  1. Load raw frame
  2. Factory vignetting correction (17x13 grid from LRI)
  3. Factory AWB (R, Gr, Gb, B — same gains for all cameras)
  4. Absolute EV normalization to a target exposure

Fusion (all cameras simultaneously):
  5. Warp to a virtual camera at the group centroid (flat depth plane)
  6. Confidence-weighted blend (resolution match × edge taper)

No reference camera.  No empirical correction.  No camera-by-name logic.
Intended as Step 1 of the Light L16 pipeline: merge the same-focal-length
group that fired.  The output is a single high-resolution image at that
focal length, which other (higher-focal-length) camera groups can later
be overlaid onto using depth.
"""

from __future__ import annotations

import os
import numpy as np
import cv2

from lri_calibration import (apply_vignetting_correction, apply_cra_correction,
                              apply_ccm_correction, select_ccm)
from lri_camera_remap import compute_remap, apply_remap
from lri_confidence import compute_confidence


# ---------------------------------------------------------------------------
# Group virtual camera (symmetric, no reference)
# ---------------------------------------------------------------------------

class GroupVirtualCamera:
    """
    Virtual output camera representing the collective view of a same-focal-
    length camera group.

    Position: centroid of the group's translations.
    Orientation: identity (all L16 cameras point along +Z).
    Intrinsics: focal length (median by default, configurable via fx_mode),
                FOV = union of group FOVs.

    No camera in the group is singled out.  Order of cameras passed in is
    irrelevant; the output is permutation-invariant.
    """

    def __init__(self, cameras: dict, fx_mode: str | float = 'median'):
        """
        Parameters
        ----------
        cameras : dict
            Subset of cameras to merge.  Each entry must have keys
            K, R, t, W, H.  All cameras in the group should share the
            same focal length (within a few percent).
        fx_mode : str or float, optional
            Method for selecting the canvas focal length:
            - 'median' (default): use median focal length across all cameras.
              Good for same-focal-length groups.
            - 'max': use the highest focal length in the group. Best for
              mixed-focal-length merges (e.g., 5 wide + 5 tele cameras):
              canvas is at tele resolution, wide content upsampled once.
            - 'min': use the lowest focal length in the group.
            - float: explicit focal length value (in pixels).
        """
        if not cameras:
            raise ValueError("cameras dict is empty")

        # Centroid of translations (in mm, matching calibration convention)
        ts = [np.asarray(c['t'], dtype=np.float64).ravel() for c in cameras.values()]
        self.t = np.mean(np.stack(ts, axis=0), axis=0)

        # Identity rotation (L16 cameras all point +Z)
        self.R = np.eye(3, dtype=np.float64)

        # Focal length selection based on fx_mode
        fxs = [float(c['K'][0, 0]) for c in cameras.values()]
        fys = [float(c['K'][1, 1]) for c in cameras.values()]
        
        if isinstance(fx_mode, (int, float)):
            output_fx = float(fx_mode)
            output_fy = float(fx_mode)
        elif fx_mode == 'median':
            output_fx = float(np.median(fxs))
            output_fy = float(np.median(fys))
        elif fx_mode == 'max':
            output_fx = float(np.max(fxs))
            output_fy = float(np.max(fys))
        elif fx_mode == 'min':
            output_fx = float(np.min(fxs))
            output_fy = float(np.min(fys))
        else:
            raise ValueError(f"fx_mode='{fx_mode}' not recognized; "
                           "use 'median', 'max', 'min', or a numeric value")

        # FOV = union across all participating cameras
        half_fov_h_list = []
        half_fov_v_list = []
        for c in cameras.values():
            K = c['K']
            W = c['W']
            H = c['H']
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            half_fov_h_list.append(max(np.arctan(cx / fx),
                                       np.arctan((W - cx) / fx)))
            half_fov_v_list.append(max(np.arctan(cy / fy),
                                       np.arctan((H - cy) / fy)))
        half_fov_h = max(half_fov_h_list)
        half_fov_v = max(half_fov_v_list)

        W_out = int(round(2.0 * np.tan(half_fov_h) * output_fx))
        H_out = int(round(2.0 * np.tan(half_fov_v) * output_fy))

        self.K = np.array([
            [output_fx, 0.0,       W_out / 2.0],
            [0.0,       output_fy, H_out / 2.0],
            [0.0,       0.0,       1.0        ],
        ], dtype=np.float64)
        self.W = W_out
        self.H = H_out

        # Mode hint for compute_confidence() (wide vs tele) — our group merge
        # is at the group's native resolution, so mark as 'wide' so the
        # confidence function uses res_w^0 (no cross-resolution penalty).
        # When all cameras share the same fx, res_w=1.0 and this is moot.
        self.mode = 'wide'

    def output_shape(self) -> tuple[int, int]:
        return (self.H, self.W)


# ---------------------------------------------------------------------------
# Per-camera ISP (symmetric — identical for every camera)
# ---------------------------------------------------------------------------

def _apply_factory_isp(
    raw_img: np.ndarray,           # (H, W, 3) uint16 raw debayered BGR
    cam: dict,                     # camera dict with analog_gain, exposure_ns
    module_cal: dict | None,       # sensor_cal['modules'][cam_name] or None
    awb: dict | None,              # sensor_cal['awb'] or None
    target_effective_exposure: float,
    awb_mode: int | None = None,   # sensor_cal['awb_mode'] — selects CCM
) -> np.ndarray:
    """
    Apply the factory ISP stages to a single raw frame.

    Stages applied (in order):
      1. Factory vignetting (17x13 flat-field grid)
      2. Factory CRA correction (17x13 grid of 4x4 Bayer channel mixing matrices)
      3. Factory AWB (per-channel gains, same for all cameras)
      3b. CCM (3x3 Color Correction Matrix — converts camera linear → sRGB linear)
      4. Absolute EV normalization

    Returns float32 BGR (H, W, 3) in the uint16 intensity scale [0, 65535].
    """
    # 1. Vignetting
    if module_cal is not None and module_cal.get('vignetting') is not None:
        img = apply_vignetting_correction(raw_img, module_cal['vignetting'])
    else:
        img = raw_img.astype(np.float32)

    # 2. CRA (Chief Ray Angle) correction — spatially-varying per-cell color mixing
    if module_cal is not None and module_cal.get('cra') is not None:
        img = apply_cra_correction(img, module_cal['cra'])

    # 3. AWB — applied identically to every camera
    if awb is not None:
        awb_bgr = np.array([
            awb['B'],
            (awb['Gr'] + awb['Gb']) / 2.0,
            awb['R'],
        ], dtype=np.float32)
        img = img * awb_bgr[None, None, :]

    # 3b. CCM — converts camera linear RGB to sRGB-linear color space
    if module_cal is not None and module_cal.get('ccm'):
        ccm = select_ccm(awb_mode, module_cal['ccm'])
        if ccm is not None:
            img = apply_ccm_correction(img, ccm)

    # 4. Absolute EV normalization
    # target_effective_exposure is analog_gain * exposure_ns for the reference
    # exposure (chosen externally, e.g. median of firing cameras).
    # This is NOT a reference camera — it's a scalar exposure target.
    cam_eff = float(cam.get('analog_gain', 1.0)) * float(cam.get('exposure_ns', 1))
    cam_eff = max(cam_eff, 1.0)
    ev_scale = target_effective_exposure / cam_eff
    img = img * ev_scale

    return np.clip(img, 0.0, 65535.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def merge_cameras(
    frames_dir: str,
    cameras: dict,
    sensor_cal: dict,
    virtual_cam: GroupVirtualCamera | None = None,
    depth: float | np.ndarray = 18.0,
    target_effective_exposure: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric merge of all cameras in ``cameras`` dict into a single canvas.

    Parameters
    ----------
    frames_dir : str
        Directory containing ``<cam_name>.png`` raw debayered frames.
    cameras : dict
        The cameras to merge.  Every camera in the dict is treated equally.
        Each entry has K, R, t, W, H, analog_gain, exposure_ns.
    sensor_cal : dict
        Output of ``lri_calibration.extract_sensor_calibration(lri_path)``.
        Provides per-camera factory vignetting + scene AWB.
    virtual_cam : GroupVirtualCamera, optional
        Output canvas.  If None, one is built from the group's centroid and
        median focal length.  For best quality with mixed-focal-length groups
        (e.g., 5 wide + 5 tele cameras), pass
        ``virtual_cam=GroupVirtualCamera(cameras, fx_mode='max')`` to canvas
        at the tele (highest-fx) resolution and upsample wide content once.
    depth : float or np.ndarray
        Scene depth in metres.  Either a flat plane (single float) or a
        per-pixel depth map (H_out, W_out) in the virtual camera's frame.
    target_effective_exposure : float, optional
        Reference value for ``analog_gain * exposure_ns`` used for EV
        normalization.  If None, uses the median across all cameras in the
        group.  This is NOT a reference camera — it's a scalar target.

    Returns
    -------
    merged : np.ndarray (H_out, W_out, 3) float32
        Fused canvas in the [0, 65535] intensity scale.
    weight_sum : np.ndarray (H_out, W_out) float32
        Accumulated confidence per pixel (for debugging / coverage masks).
    """
    if not cameras:
        raise ValueError("cameras dict is empty")

    if virtual_cam is None:
        virtual_cam = GroupVirtualCamera(cameras)

    if target_effective_exposure is None:
        effs = []
        for cam in cameras.values():
            eff = float(cam.get('analog_gain', 1.0)) * float(cam.get('exposure_ns', 1))
            effs.append(eff)
        target_effective_exposure = float(np.median(effs))

    H_out = virtual_cam.H
    W_out = virtual_cam.W

    # Accumulators
    canvas = np.zeros((H_out, W_out, 3), dtype=np.float64)
    weight_sum = np.zeros((H_out, W_out), dtype=np.float64)

    # Depth map in canvas space
    if isinstance(depth, (int, float)):
        depth_map = np.full((H_out, W_out), float(depth), dtype=np.float32)
        depth_label = f"flat {float(depth):.2f}m"
    else:
        depth_map = np.asarray(depth, dtype=np.float32)
        if depth_map.shape != (H_out, W_out):
            raise ValueError(
                f"depth shape {depth_map.shape} != canvas {(H_out, W_out)}"
            )
        depth_label = f"per-pixel {depth_map.mean():.2f}m mean"

    print(f"[merge] canvas: {W_out}×{H_out} = {W_out*H_out/1e6:.1f}MP")
    print(f"[merge] depth:  {depth_label}")
    print(f"[merge] target effective exposure: {target_effective_exposure:.0f}")
    print(f"[merge] cameras: {sorted(cameras.keys())}")

    for cam_name in sorted(cameras.keys()):
        cam = cameras[cam_name]
        frame_path = os.path.join(frames_dir, f'{cam_name}.png')
        if not os.path.exists(frame_path):
            print(f"  {cam_name}: MISSING frame at {frame_path}")
            continue

        raw = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"  {cam_name}: failed to read frame")
            continue
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        module_cal = sensor_cal.get('modules', {}).get(cam_name) if sensor_cal else None
        awb = sensor_cal.get('awb') if sensor_cal else None
        awb_mode = sensor_cal.get('awb_mode') if sensor_cal else None

        img = _apply_factory_isp(raw, cam, module_cal, awb, target_effective_exposure,
                                  awb_mode=awb_mode)

        map_x, map_y, mask = compute_remap(virtual_cam, cam, depth_map)
        warped = apply_remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        conf = compute_confidence(img, cam, virtual_cam, mask, map_x, map_y)

        canvas += warped.astype(np.float64) * conf[:, :, None].astype(np.float64)
        weight_sum += conf.astype(np.float64)

        n_covered = int((conf > 0).sum())
        print(f"  {cam_name}: coverage={100*n_covered/(H_out*W_out):5.1f}%  "
              f"mean_conf={conf.mean():.4f}  mean_val={img.mean():.0f}")

    weight_sum_safe = np.maximum(weight_sum, 1e-8)
    merged = (canvas / weight_sum_safe[:, :, None]).astype(np.float32)
    return merged, weight_sum.astype(np.float32)


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def linear_to_srgb_uint8(
    img_f32: np.ndarray,
    percentile_white: float = 99.5,
) -> np.ndarray:
    """Convert linear [0, 65535+] float32 to viewable sRGB uint8."""
    white = max(float(np.percentile(img_f32, percentile_white)), 1.0)
    norm = np.clip(img_f32 / white, 0.0, 1.0)
    mask = norm <= 0.0031308
    srgb = np.where(
        mask,
        12.92 * norm,
        1.055 * np.power(np.maximum(norm, 1e-12), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
