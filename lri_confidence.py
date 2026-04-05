"""
lri_confidence.py — Per-camera confidence weight map for L16 synthetic canvas assembly.

Each pixel of the weight map says how much to trust that camera's contribution
at that output location. Used for weighted blending in canvas assembly.
"""

import numpy as np
import cv2


def resolution_weight(source_cam: dict, virtual_cam) -> float:
    """
    Scalar weight for how well this camera's angular resolution matches the output canvas.

    Wide cameras (fx≈3370) against virtual cam (fx≈8275) → weight ≈ 0.41
    Telephoto cameras (fx≈8285) → weight ≈ 1.0
    """
    fx_src = source_cam['K'][0, 0]
    fx_virt = virtual_cam.K[0, 0]
    return min(fx_src / fx_virt, 1.0)


def sharpness_map(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Compute local sharpness (Laplacian variance) at source resolution.

    image: (H, W) or (H, W, 3) uint16 or float32
    Returns: (H, W) float32 sharpness map, values in [0, 1] range
    """
    # Convert to grayscale float32 if colour
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float32)

    # Apply Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    # Compute local variance: E[x^2] - E[x]^2
    k = (kernel_size, kernel_size)
    lap_sq_mean = cv2.blur(lap ** 2, k)
    lap_mean_sq = cv2.blur(lap, k) ** 2
    variance = lap_sq_mean - lap_mean_sq

    # Clip negatives (numerical noise) and normalise to [0, 1].
    # Use p99 rather than global max so that a single bright LED or chrome
    # reflection doesn't compress the sharpness scale for the rest of the image.
    variance = np.maximum(variance, 0.0)
    p99 = float(np.percentile(variance, 99))
    sharp = np.clip(variance / (p99 + 1e-6), 0.0, 1.0)

    return sharp.astype(np.float32)


def coverage_taper(coverage_mask: np.ndarray, taper_px: int = 50) -> np.ndarray:
    """
    Smooth falloff near the edge of a camera's coverage region.

    coverage_mask: bool (H, W)
    Returns: float32 (H, W), 1.0 inside, tapers to 0 at boundary
    """
    # distanceTransform requires uint8 input; 255 = foreground
    mask_u8 = coverage_mask.astype(np.uint8) * 255

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Clip to [0, taper_px] and normalise → [0, 1]
    taper = np.clip(dist, 0.0, float(taper_px)) / float(taper_px)

    return taper.astype(np.float32)


def compute_confidence(
    source_img: np.ndarray,      # (H_src, W_src, 3) uint16
    source_cam: dict,
    virtual_cam,                  # VirtualCamera instance
    coverage_mask: np.ndarray,   # (H_out, W_out) bool — from compute_remap
    map_x: np.ndarray,           # (H_out, W_out) float32 — from compute_remap
    map_y: np.ndarray,           # (H_out, W_out) float32 — from compute_remap
) -> np.ndarray:                  # (H_out, W_out) float32
    """
    Compute full confidence map for one camera.

    Tele mode (virtual fx≈8276):
        Confidence = taper * res_w^8

        res_w^8 gives B4 (fx≈8276, res_w=1.0) a weight of 1.0 vs each A
        camera (fx≈3370, res_w≈0.41) at ~0.0007, so B4 captures ~99.6% of
        the blended weight where it has coverage, eliminating wide-camera
        parallax ghosting at the canvas centre.

    Wide mode (virtual fx≈3370):
        In wide mode both A cameras (fx≈3370) and B cameras (fx≈8276) have
        res_w = min(fx/fx_virt, 1.0) = 1.0 — the resolution scalar no longer
        differentiates them.  Using res_w^8 would give every camera identical
        weight, erasing useful depth-based alignment information that B cameras
        contribute when correctly warped.

        Instead, wide mode uses taper-only confidence (equal scalar weight = 1)
        so that the depth warp is the sole arbiter of each camera's contribution.
        Pixels near coverage boundaries still fade smoothly via the taper.
        B cameras with stale MOVABLE mirror calibration may produce misaligned
        contributions — these are handled at the fusion layer (warning + skip or
        fixed-depth fallback), not here.

    Parameters
    ----------
    source_img : np.ndarray
        Raw source frame (used only if sharpness_map is needed in future).
    source_cam : dict
        Camera calibration dict (K, R, t, W, H, mirror_type).
    virtual_cam : VirtualCamera
        Target canvas.  ``virtual_cam.mode`` selects the confidence formula.
    coverage_mask : np.ndarray
        Boolean (H_out, W_out) indicating where this camera has valid pixels.
    map_x, map_y : np.ndarray
        Remap arrays from compute_remap (unused directly, reserved for future
        sharpness-at-canvas-location weighting).
    """
    vc_mode = getattr(virtual_cam, 'mode', 'tele')

    # Smooth boundary falloff (used in both modes)
    taper = coverage_taper(coverage_mask)

    if vc_mode == 'wide':
        # Wide mode: taper only — equal weight for all cameras, let depth warp
        # determine spatial contributions.
        confidence = taper.copy()
    else:
        # Tele mode: res_w^8 strongly favours telephoto cameras.
        res_w = resolution_weight(source_cam, virtual_cam)
        confidence = taper * (res_w ** 8)

    # Zero out pixels outside coverage
    confidence[~coverage_mask] = 0.0

    return confidence.astype(np.float32)


if __name__ == '__main__':
    import sys
    import cv2
    from lri_fuse_image import load_cameras
    from lri_virtual_camera import VirtualCamera
    from lri_camera_remap import compute_remap

    cal_path = '/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_cal/calibration.json/calibration.json'
    frames_dir = '/Volumes/L16IMAGES/Light/2018-10-12/L16_02532_lumen/frames'

    print("Loading cameras...")
    cameras = load_cameras(cal_path)
    vc = VirtualCamera(cameras)

    for cam_name in ['A1', 'B4']:
        img = cv2.imread(f'{frames_dir}/{cam_name}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"{cam_name}: frame not found, skipping")
            continue
        cam = cameras[cam_name]
        print(f"{cam_name}: computing remap (81MP canvas — may take a few minutes)...")
        map_x, map_y, mask = compute_remap(vc, cam)
        print(f"{cam_name}: computing confidence...")
        conf = compute_confidence(img, cam, vc, mask, map_x, map_y)
        print(f"{cam_name}: coverage={mask.mean()*100:.1f}%  "
              f"conf_mean={conf[mask].mean():.4f}  "
              f"res_weight={resolution_weight(cam, vc):.3f}")
