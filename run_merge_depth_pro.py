"""
Run the A-camera merge using Depth Pro for per-pixel depth instead of flat plane.
Test on L16_00684 (2017-12-07) — the indoor office scene that kills the flat-plane merge.

Since init_depth in merge_cameras_with_flow forces a flat plane, we replicate
the entire merge pipeline here with per-pixel depth support.
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, '/Users/ryaker/Documents/Light_Work')

from lri_calibration import extract_sensor_calibration, parse_lri
from lri_merge import (
    GroupVirtualCamera,
    linear_to_srgb_uint8,
    _apply_factory_isp,
)
from lri_camera_remap import compute_remap, apply_remap
from lri_confidence import compute_confidence
from lri_merge_flow import _dense_flow, _warp_by_flow
from lri_depth_pro import estimate_depth


def load_lri_cameras(lri_path: str) -> dict:
    """Load camera calibration from LRI file."""
    lri = parse_lri(lri_path)
    
    cams = {}
    for m in lri['modules']:
        cal = m.get('calibration')
        if not cal or not cal.get('rotation'):
            continue
        
        # Get intrinsics - can be either k_mat or individual fx/fy/cx/cy
        intr = cal.get('intrinsics') or {}
        if 'k_mat' in intr:
            K = np.array(intr['k_mat'], dtype=np.float64)
        else:
            K = np.array([
                [intr['fx'], 0, intr['cx']],
                [0, intr['fy'], intr['cy']],
                [0, 0, 1]
            ], dtype=np.float64)
        
        R = np.array(cal['rotation'], dtype=np.float64)
        t = np.array(cal['translation'], dtype=np.float64)
        cams[m['camera_name']] = {
            'K': K,
            'R': R,
            't': t,
            'W': m['width'],
            'H': m['height'],
            'analog_gain': m['analog_gain'],
            'exposure_ns': m['exposure_ns'],
        }
    return cams


def merge_cameras_with_depth_pro(
    frames_dir: str,
    cameras: dict,
    sensor_cal: dict,
    virtual_cam: GroupVirtualCamera,
    depth_map: np.ndarray,
    n_iterations: int = 2,
    target_effective_exposure: float | None = None,
    camera_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge cameras using per-pixel depth warp + iterative optical-flow refinement.
    
    This is the same as merge_cameras_with_flow, except it uses a per-pixel
    depth_map instead of a flat plane.
    
    Parameters
    ----------
    frames_dir : str
        Directory with <cam>.png frames.
    cameras : dict
        Cameras to merge.
    sensor_cal : dict
        Output of extract_sensor_calibration().
    virtual_cam : GroupVirtualCamera
        Output canvas.
    depth_map : np.ndarray
        float32 (H_canvas, W_canvas) depth map in metres.
    n_iterations : int
        Number of flow refinement passes.
    target_effective_exposure : float, optional
        Absolute EV target.
    camera_weights : dict[str, float] | None
        Per-camera calibration weights.
    
    Returns
    -------
    merged : float32 (H, W, 3) in [0, 65535] linear
    weight_sum : float32 (H, W)
    """
    if not cameras:
        raise ValueError("cameras dict is empty")

    if target_effective_exposure is None:
        effs = [float(c.get('analog_gain', 1.0)) * float(c.get('exposure_ns', 1))
                for c in cameras.values()]
        target_effective_exposure = float(np.median(effs))

    H_out = virtual_cam.H
    W_out = virtual_cam.W

    print(f"[merge-depth-pro] canvas: {W_out}×{H_out}")
    print(f"[merge-depth-pro] depth_map: min={depth_map.min():.2f}m  max={depth_map.max():.2f}m")
    print(f"[merge-depth-pro] iterations: {n_iterations}")
    print(f"[merge-depth-pro] camera_weights: {camera_weights or 'default (1.0 for all)'}")
    print(f"[merge-depth-pro] cameras: {sorted(cameras.keys())}")

    # ── Stage 1: factory ISP + per-pixel-depth warp for every camera ───────
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

        # Factory ISP
        module_cal = sensor_cal.get('modules', {}).get(cam_name) if sensor_cal else None
        awb = sensor_cal.get('awb') if sensor_cal else None
        awb_mode = sensor_cal.get('awb_mode') if sensor_cal else None
        img = _apply_factory_isp(raw, cam, module_cal, awb, target_effective_exposure,
                                  awb_mode=awb_mode)

        # Warp using per-pixel depth
        map_x, map_y, mask = compute_remap(virtual_cam, cam, depth_map)
        warped = apply_remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        conf = compute_confidence(img, cam, virtual_cam, mask, map_x, map_y)

        # Get calibration weight for this camera
        cal_weight = float((camera_weights or {}).get(cam_name, 1.0))
        warped_list.append((cam_name, warped.astype(np.float32), conf.astype(np.float32), cal_weight))
        print(f"  {cam_name}: warped, mean_conf={conf.mean():.3f}, cal_weight={cal_weight:.2f}")

    if len(warped_list) < 2:
        raise RuntimeError(f"Need at least 2 cameras, got {len(warped_list)}")

    # ── Stage 2: iterative optical-flow refinement ─────────────────────────
    def consensus(warped_list):
        """Compute confidence-weighted mean."""
        sum_w = np.zeros((H_out, W_out, 3), dtype=np.float64)
        sum_c = np.zeros((H_out, W_out), dtype=np.float64)
        for _, warped, conf, cal_weight in warped_list:
            weighted_conf = conf * cal_weight
            sum_w += warped.astype(np.float64) * weighted_conf[:, :, None]
            sum_c += weighted_conf
        return (sum_w / np.maximum(sum_c[:, :, None], 1e-6)).astype(np.float32), sum_c.astype(np.float32)

    # Main refinement iterations (at full resolution)
    for iteration in range(n_iterations):
        t0 = time.time()
        cons, _ = consensus(warped_list)

        # Grayscale for flow
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
            flow = _dense_flow(warped_gray, cons_gray)
            flow_mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            new_warped = _warp_by_flow(warped, flow)
            refined.append((name, new_warped.astype(np.float32), conf, cal_weight))
            print(f"  iter{iteration+1} {name}: mean flow = {flow_mag:.2f}px")

        warped_list = refined
        print(f"  iter{iteration+1} done in {time.time()-t0:.1f}s")

    # ── Stage 3: final blend ───────────────────────────────────────────────
    final, weight_sum = consensus(warped_list)
    return final, weight_sum


def main():
    LRI = '/Volumes/Base Photos/Light/2017-12-07/L16_00684.lri'
    FRAMES = '/Volumes/Dev/Light_Work_scratch/L16_00684/2017-12-07/frames'
    OUT_DIR = '/Volumes/Dev/Light_Work_scratch/L16_00684/2017-12-07'

    print('[run] Loading LRI and building cameras...')
    all_cams = load_lri_cameras(LRI)
    a_cams = {k: v for k, v in all_cams.items() if k.startswith('A')}
    print(f'[run] A cameras: {sorted(a_cams.keys())}')

    # Sensor calibration
    sc = extract_sensor_calibration(LRI)

    # Virtual camera
    vcam = GroupVirtualCamera(a_cams)
    print(f'[run] Canvas: {vcam.W}×{vcam.H}')

    # ─── Depth Pro pass ───
    print('\n[run] Running Depth Pro on A1...')
    t0 = time.time()
    depth_map = estimate_depth(
        frame_path=os.path.join(FRAMES, 'A1.png'),
        canvas_hw=(vcam.H, vcam.W),
        focal_px=float(a_cams['A1']['K'][0, 0]),
        device='mps',
    )
    t_depth = time.time() - t0
    print(f'[run] Depth Pro done in {t_depth:.1f}s')
    print(f'[run] Depth map: min={depth_map.min():.2f}m  median={np.median(depth_map):.2f}m  max={depth_map.max():.2f}m')

    # Save depth visualization
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_PLASMA)
    depth_vis_path = os.path.join(OUT_DIR, 'depth_pro_vis.png')
    cv2.imwrite(depth_vis_path, depth_vis)
    print(f'[run] Saved depth visualization to {depth_vis_path}')

    # ─── Merge using real depth ───
    print('\n[run] Merging with Depth Pro depth map...')
    t0 = time.time()
    final, wsum = merge_cameras_with_depth_pro(
        frames_dir=FRAMES,
        cameras=a_cams,
        sensor_cal=sc,
        virtual_cam=vcam,
        depth_map=depth_map,
        n_iterations=2,
        camera_weights={'A5': 0.7, 'A4': 0.85},
    )
    t_merge = time.time() - t0
    print(f'[run] Merge done in {t_merge:.1f}s')

    # Save outputs
    out_16 = os.path.join(OUT_DIR, 'merged_a_depth_pro_16bit.png')
    out_pv = os.path.join(OUT_DIR, 'merged_a_depth_pro_preview.png')

    final16 = np.clip(final, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_16, final16)
    cv2.imwrite(out_pv, linear_to_srgb_uint8(final, percentile_white=99.0))

    print(f'\n[run] ✓ Saved {out_16}')
    print(f'[run] ✓ Saved {out_pv}')
    print(f'[run] Weight-sum stats: min={wsum.min():.3f} max={wsum.max():.3f} mean={wsum.mean():.3f}')
    print(f'\n[run] Total time: {t_depth + t_merge:.1f}s (Depth Pro: {t_depth:.1f}s, Merge: {t_merge:.1f}s)')
    print('[run] Done!')


if __name__ == '__main__':
    main()
