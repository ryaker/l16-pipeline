"""
lri_merge_flow.py — Merge same-focal-length cameras with dense-flow refinement.

Approach (classical CV, fast on 2017-era CPU):
  1. Warp every camera to a virtual canvas at a flat depth plane (K/R/t + one depth).
  2. Compute the mean of all warped images — this is the initial "consensus."
  3. For each camera, compute dense optical flow from its warped image TO the consensus.
  4. Apply that flow to bring every camera into local per-pixel alignment.
  5. Compute the aligned mean (or weighted blend) — the refined consensus.
  6. Repeat 3–5 for N iterations until it converges.

The consensus is the mean across all cameras at each step — no single camera is
a reference. After convergence, every camera is warped to match the consensus,
so the final blend is aligned per-pixel without any explicit depth estimation.

This is what classical photo-stitching and super-resolution tools did in the
2000s–2010s, and it runs in seconds on desktop CPU for 5 wide-camera frames.
"""

from __future__ import annotations
import os
import time
import numpy as np
import cv2

from lri_calibration import apply_vignetting_correction
from lri_camera_remap import compute_remap, apply_remap
from lri_confidence import compute_confidence
from lri_merge import GroupVirtualCamera, _apply_factory_isp, linear_to_srgb_uint8


def _warp_by_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Apply an optical-flow field to an image.
    flow: (H, W, 2) float32 — flow[y, x] = (dx, dy) telling where pixel (x, y)
          in the SOURCE should move to in the TARGET.
    We want the reverse: for each target pixel, sample the source at (x - dx, y - dy).
    """
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = xx - flow[..., 0]
    map_y = yy - flow[..., 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _dense_flow(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Dense optical flow from src → dst (both (H, W) uint8 grayscale).
    Uses OpenCV's DIS flow (2016, fast and accurate).
    Returns (H, W, 2) float32.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(src, dst, None)


def _downsample_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Downsample image by scale factor (e.g., 0.5 means half size)."""
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _upsample_flow(flow: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Upsample flow to target (H, W) shape."""
    h, w = target_shape
    flow_up = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_up[..., 0] *= w / float(flow.shape[1])
    flow_up[..., 1] *= h / float(flow.shape[0])
    return flow_up


def merge_cameras_with_flow(
    frames_dir: str,
    cameras: dict,
    sensor_cal: dict,
    virtual_cam: GroupVirtualCamera | None = None,
    init_depth: float = 38.0,
    n_iterations: int = 2,
    target_effective_exposure: float | None = None,
    camera_weights: dict[str, float] | None = None,
    coarse_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge cameras using flat-plane warp + iterative optical-flow refinement.

    Parameters
    ----------
    frames_dir : str
        Directory with <cam>.png frames.
    cameras : dict
        Cameras to merge (symmetric, no single camera is special).
    sensor_cal : dict
        Output of extract_sensor_calibration().
    virtual_cam : GroupVirtualCamera
        Output canvas (built from the group centroid if None).
    init_depth : float
        Flat-plane depth in metres for the initial warp.  A sensible default
        is the capture focus distance, but the exact value is not critical
        because the flow refinement corrects the residual.
    n_iterations : int
        Number of flow refinement passes (2–3 is typical).
    target_effective_exposure : float, optional
        Absolute EV target; defaults to median across firing cameras.
    camera_weights : dict[str, float] | None
        Per-camera calibration weights (default 1.0). Lower drift → higher weight.
        E.g., {'A5': 0.7, 'A4': 0.85, 'A1': 1.0}. Multiplies confidence in consensus.
    coarse_scale : float
        If < 1.0 and n_iterations >= 2, run one consensus + flow pass at this
        scale first, then refine at full resolution (catches large offsets).

    Returns
    -------
    merged : float32 (H, W, 3) in [0, 65535] linear
    weight_sum : float32 (H, W)
    """
    if not cameras:
        raise ValueError("cameras dict is empty")

    if virtual_cam is None:
        virtual_cam = GroupVirtualCamera(cameras)

    if target_effective_exposure is None:
        effs = [float(c.get('analog_gain', 1.0)) * float(c.get('exposure_ns', 1))
                for c in cameras.values()]
        target_effective_exposure = float(np.median(effs))

    H_out = virtual_cam.H
    W_out = virtual_cam.W

    # Initial flat-plane depth map
    depth_map = np.full((H_out, W_out), float(init_depth), dtype=np.float32)

    print(f"[merge-flow] canvas: {W_out}×{H_out}")
    print(f"[merge-flow] init depth: {init_depth}m")
    print(f"[merge-flow] iterations: {n_iterations}")
    print(f"[merge-flow] camera_weights: {camera_weights or 'default (1.0 for all)'}")
    print(f"[merge-flow] coarse_scale: {coarse_scale}")
    print(f"[merge-flow] cameras: {sorted(cameras.keys())}")

    # ── Stage 1: factory ISP + flat-plane warp for every camera ───────────
    warped_list = []    # list of (name, warped float32 BGR, conf, cal_weight)
    cam_names = sorted(cameras.keys())

    for cam_name in cam_names:
        cam = cameras[cam_name]
        frame_path = os.path.join(frames_dir, f'{cam_name}.png')
        if not os.path.exists(frame_path):
            print(f"  {cam_name}: MISSING")
            continue
        raw = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
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

        # Get calibration weight for this camera (default 1.0)
        cal_weight = float((camera_weights or {}).get(cam_name, 1.0))
        warped_list.append((cam_name, warped.astype(np.float32), conf.astype(np.float32), cal_weight))
        print(f"  {cam_name}: warped at flat plane, mean_conf={conf.mean():.3f}, cal_weight={cal_weight:.2f}")

    if len(warped_list) < 2:
        raise RuntimeError(f"Need at least 2 cameras, got {len(warped_list)}")

    # ── Stage 2: iterative optical-flow refinement ─────────────────────────
    # Each iteration:
    #   a) Compute consensus = confidence-weighted mean of all current warps
    #      (weighted by both confidence and per-camera calibration weight)
    #   b) For each camera, compute dense flow from its warped image → consensus
    #   c) Apply the flow to update that camera's warp
    # The consensus in step (a) is symmetric across all cameras; no single one
    # is special.  The flow correction converges in 2–3 iterations.

    def consensus(warped_list):
        """Compute confidence-weighted mean, including per-camera calibration weights."""
        sum_w = np.zeros((H_out, W_out, 3), dtype=np.float64)
        sum_c = np.zeros((H_out, W_out), dtype=np.float64)
        for _, warped, conf, cal_weight in warped_list:
            weighted_conf = conf * cal_weight
            sum_w += warped.astype(np.float64) * weighted_conf[:, :, None]
            sum_c += weighted_conf
        return (sum_w / np.maximum(sum_c[:, :, None], 1e-6)).astype(np.float32), sum_c.astype(np.float32)

    # Determine if we should do a coarse pre-pass
    # NOTE: Coarse pyramid pass disabled for now - causes shape mismatch in consensus
    # TODO: Need to refactor consensus() to handle variable-size warped_list
    do_coarse_pass = False  # (coarse_scale < 1.0) and (n_iterations >= 2)

    if do_coarse_pass:
        print(f"[merge-flow] coarse pre-pass at scale {coarse_scale}")
        t0 = time.time()

        # Downsample all warped images for coarse pass
        warped_coarse = []
        for name, warped, conf, cal_weight in warped_list:
            w_coarse = _downsample_image(warped, coarse_scale)
            c_coarse = _downsample_image(conf, coarse_scale)
            warped_coarse.append((name, w_coarse, c_coarse, cal_weight))

        # Coarse consensus + flow pass
        cons_coarse, _ = consensus(warped_coarse)
        cons_gray_coarse = cv2.cvtColor(
            linear_to_srgb_uint8(cons_coarse, percentile_white=99.0),
            cv2.COLOR_BGR2GRAY,
        )

        warped_coarse_refined = []
        for name, warped_c, conf_c, cal_weight in warped_coarse:
            warped_gray_c = cv2.cvtColor(
                linear_to_srgb_uint8(warped_c, percentile_white=99.0),
                cv2.COLOR_BGR2GRAY,
            )
            flow_coarse = _dense_flow(warped_gray_c, cons_gray_coarse)
            new_warped_c = _warp_by_flow(warped_c, flow_coarse)
            warped_coarse_refined.append((name, new_warped_c, conf_c, cal_weight))

        # Upsample the coarse-refined images and flows back to full resolution
        warped_list_new = []
        for (name, warped_orig, conf_orig, cal_weight), (_, warped_c_refined, _, _) in \
                zip(warped_list, warped_coarse_refined):
            # Upsample the refined coarse warp
            warped_up = cv2.resize(warped_c_refined, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
            warped_list_new.append((name, warped_up.astype(np.float32), conf_orig, cal_weight))

        warped_list = warped_list_new
        print(f"  coarse pre-pass done in {time.time()-t0:.1f}s")

    # Main refinement iterations (at full resolution)
    main_iterations = (n_iterations - 1) if do_coarse_pass else n_iterations
    for iteration in range(main_iterations):
        t0 = time.time()
        cons, _ = consensus(warped_list)

        # Grayscale for flow (DIS works on 8-bit gray)
        cons_gray = cv2.cvtColor(
            linear_to_srgb_uint8(cons, percentile_white=99.0),
            cv2.COLOR_BGR2GRAY,
        )

        refined = []
        for name, warped, conf, cal_weight in warped_list:
            warped_gray = cv2.cvtColor(
                linear_to_srgb_uint8(warped, percentile_white=99.0),
                cv2.COLOR_BGR2GRAY,
            )
            flow = _dense_flow(warped_gray, cons_gray)   # (H, W, 2)
            flow_mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            new_warped = _warp_by_flow(warped, flow)
            refined.append((name, new_warped.astype(np.float32), conf, cal_weight))
            print(f"  iter{iteration+1} {name}: mean flow = {flow_mag:.2f}px")

        warped_list = refined
        print(f"  iter{iteration+1} done in {time.time()-t0:.1f}s")

    # ── Stage 3: final blend ───────────────────────────────────────────────
    final, weight_sum = consensus(warped_list)
    return final, weight_sum
