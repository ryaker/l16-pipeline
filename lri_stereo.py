#!/usr/bin/env python3
"""
lri_stereo.py — Stereo depth estimation from extracted L16 images + calibration.

Uses known factory calibration to rectify stereo pairs and run dense stereo matching.
Outputs depth maps and a fused point cloud.

Usage:
    python3 lri_stereo.py <frames_dir> <calibration.json> [output_dir]
    python3 lri_stereo.py /tmp/L16_00001_frames /tmp/L16_00001_cal/calibration.json /tmp/L16_00001_depth

Stereo pairs used:
    A cameras (28mm): A1-A5 (43mm baseline), A2-A4, A2-A3, A3-A4  — wide FOV
    B cameras (70mm): B1-B4, B2-B4, B1-B3                        — telephoto
"""

import argparse, json, os, sys
import numpy as np
import cv2

# ── camera pair configuration ────────────────────────────────────────────────
# (left_cam, right_cam, description)
STEREO_PAIRS = [
    ('A1', 'A5', 'A1-A5_wide_baseline'),  # 43mm horizontal — primary A pair
    ('A2', 'A4', 'A2-A4_diagonal'),
    ('A3', 'A5', 'A3-A5_diagonal'),
    ('A1', 'A4', 'A1-A4_diagonal'),
    ('B1', 'B4', 'B1-B4_telephoto'),
    ('B2', 'B4', 'B2-B4_telephoto'),
    ('B1', 'B3', 'B1-B3_telephoto'),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_calibration(cal_path):
    """Load calibration JSON → dict keyed by camera name."""
    data = json.load(open(cal_path))
    cameras = {}
    for mod in data['modules']:
        name = mod['camera_name']
        cal  = mod['calibration']
        if 'rotation' not in cal:
            continue
        intr = cal['intrinsics']
        cameras[name] = {
            'fx': intr['fx'], 'fy': intr['fy'],
            'cx': intr['cx'], 'cy': intr['cy'],
            'R':  np.array(cal['rotation'],    dtype=np.float64),
            't':  np.array(cal['translation'], dtype=np.float64),
            'W':  mod['width'],
            'H':  mod['height'],
            'mirror_type': cal.get('mirror_type', 'NONE'),
        }
    return cameras


def K_mat(cam):
    """Return 3×3 intrinsic matrix for camera dict."""
    return np.array([[cam['fx'], 0, cam['cx']],
                     [0, cam['fy'], cam['cy']],
                     [0, 0, 1]], dtype=np.float64)


def relative_pose(cam_l, cam_r):
    """
    Compute relative rotation R_rel and translation t_rel such that:
        P_r = R_rel @ P_l + t_rel
    In COLMAP convention: P_cam = R @ P_world + t
        P_l = R_l @ P_w + t_l  →  P_w = R_l^T @ (P_l - t_l)
        P_r = R_r @ P_w + t_r  →  P_r = R_r @ R_l^T @ (P_l - t_l) + t_r
    So:
        R_rel = R_r @ R_l^T
        t_rel = t_r - R_rel @ t_l
    """
    R_rel = cam_r['R'] @ cam_l['R'].T
    t_rel = cam_r['t'] - R_rel @ cam_l['t']
    return R_rel, t_rel


def rectify_pair(cam_l, cam_r, size=None):
    """
    Stereo rectification using OpenCV stereoRectify.
    Returns (R1, R2, P1, P2, Q, roi1, roi2, valid_roi).
    Distortion is zero — PINHOLE with no distortion coefficients.
    """
    W = cam_l['W']
    H = cam_l['H']
    if size is None:
        size = (W, H)

    K1 = K_mat(cam_l)
    K2 = K_mat(cam_r)
    dist = np.zeros(5, dtype=np.float64)  # PINHOLE, no distortion

    R_rel, t_rel = relative_pose(cam_l, cam_r)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist, K2, dist,
        size, R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    return R1, R2, P1, P2, Q, roi1, roi2


def build_remap(cam, R_rect, P_rect, size):
    """Build undistort-rectify remap for one camera."""
    K    = K_mat(cam)
    dist = np.zeros(5, dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R_rect, P_rect, size, cv2.CV_32FC1
    )
    return map1, map2


def apply_remap(img, map1, map2):
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


def run_sgbm(img_l_rect, img_r_rect, num_disparities=256, block_size=7):
    """
    Run Semi-Global Block Matching (SGBM) on rectified grayscale images.
    Returns float32 disparity map (pixels, positive = left shift).
    """
    gray_l = cv2.cvtColor(img_l_rect, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(img_r_rect, cv2.COLOR_RGB2GRAY)

    # SGBM parameters tuned for multi-megapixel images
    P1 = 8  * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,  # must be divisible by 16
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp16 = sgbm.compute(gray_l, gray_r)  # int16 * 16 (fixed-point)
    disp   = disp16.astype(np.float32) / 16.0
    disp[disp < 0] = 0  # mask invalid
    return disp


def disparity_to_depth(disp, Q):
    """
    Use Q matrix from stereoRectify to reproject disparity → 3D.
    Returns (X,Y,Z,W) with W=0 for invalid pixels, or None slices.
    """
    points4d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    Z = points4d[:, :, 2]
    # Invalid disparities → huge Z; mask them
    Z[disp <= 0] = np.nan
    Z[Z > 1e4]  = np.nan
    Z[Z < 0]    = np.nan
    return points4d, Z


def depth_to_pointcloud(points4d, img_rgb, valid_mask):
    """Convert 3D + color image → Nx6 point cloud (X,Y,Z,R,G,B)."""
    pts = points4d[valid_mask]   # N×4
    rgb = img_rgb[valid_mask]    # N×3
    xyz = pts[:, :3]
    return np.hstack([xyz, rgb.astype(np.float32)])


def save_ply(pc, path):
    """Save Nx6 (X,Y,Z,R,G,B) array as binary PLY."""
    n = len(pc)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                   ('r','u1'),('g','u1'),('b','u1')])
    rec = np.empty(n, dtype=dt)
    rec['x'] = pc[:, 0]
    rec['y'] = pc[:, 1]
    rec['z'] = pc[:, 2]
    rec['r'] = np.clip(pc[:, 3], 0, 255).astype(np.uint8)
    rec['g'] = np.clip(pc[:, 4], 0, 255).astype(np.uint8)
    rec['b'] = np.clip(pc[:, 5], 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rec.tobytes())
    return path


def save_depth_png(Z, path):
    """Save depth as 16-bit PNG (millimetres, max ~65m)."""
    Z_mm = np.nan_to_num(Z, nan=0.0)
    Z_mm = np.clip(Z_mm, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, Z_mm)


def process_pair(name_l, name_r, label, cameras, frames_dir, out_dir,
                 scale=1, num_disparities=256, verbose=True):
    """Process one stereo pair → depth map + partial point cloud."""

    if name_l not in cameras or name_r not in cameras:
        if verbose: print(f"  [skip] {label}: camera not in calibration")
        return None

    cam_l = cameras[name_l]
    cam_r = cameras[name_r]

    # Load images
    img_l_path = os.path.join(frames_dir, f"{name_l}.png")
    img_r_path = os.path.join(frames_dir, f"{name_r}.png")
    if not os.path.exists(img_l_path) or not os.path.exists(img_r_path):
        if verbose: print(f"  [skip] {label}: images not found")
        return None

    if verbose: print(f"  [{label}] loading images ...", end='', flush=True)
    img_l = cv2.cvtColor(cv2.imread(img_l_path), cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(cv2.imread(img_r_path), cv2.COLOR_BGR2RGB)

    W, H = cam_l['W'], cam_l['H']
    if scale != 1:
        sW, sH = W // scale, H // scale
        img_l = cv2.resize(img_l, (sW, sH))
        img_r = cv2.resize(img_r, (sW, sH))
        # Scale intrinsics
        for cam in (cam_l, cam_r):
            cam = dict(cam)
        cam_l = {**cam_l, 'fx': cam_l['fx']/scale, 'fy': cam_l['fy']/scale,
                 'cx': cam_l['cx']/scale, 'cy': cam_l['cy']/scale,
                 'W': sW, 'H': sH}
        cam_r = {**cam_r, 'fx': cam_r['fx']/scale, 'fy': cam_r['fy']/scale,
                 'cx': cam_r['cx']/scale, 'cy': cam_r['cy']/scale,
                 'W': sW, 'H': sH}
        W, H = sW, sH

    if verbose: print(f" rectifying ...", end='', flush=True)
    size = (W, H)
    R1, R2, P1, P2, Q, roi1, roi2 = rectify_pair(cam_l, cam_r, size)

    map1_l, map2_l = build_remap(cam_l, R1, P1, size)
    map1_r, map2_r = build_remap(cam_r, R2, P2, size)

    rect_l = apply_remap(img_l, map1_l, map2_l)
    rect_r = apply_remap(img_r, map1_r, map2_r)

    # Save rectified images for inspection
    cv2.imwrite(os.path.join(out_dir, f"{label}_rect_L.jpg"),
                cv2.cvtColor(rect_l, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(os.path.join(out_dir, f"{label}_rect_R.jpg"),
                cv2.cvtColor(rect_r, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Draw epipolar lines on a composite for verification
    composite = np.hstack([rect_l, rect_r])
    step = H // 20
    for y in range(0, H, step):
        cv2.line(composite, (0, y), (W*2, y), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, f"{label}_epipolar.jpg"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])

    if verbose: print(f" SGBM ...", end='', flush=True)
    disp = run_sgbm(rect_l, rect_r, num_disparities=num_disparities)

    if verbose: print(f" depth ...", end='', flush=True)
    points4d, Z = disparity_to_depth(disp, Q)

    # Save depth map (16-bit PNG in mm)
    depth_path = os.path.join(out_dir, f"{label}_depth.png")
    save_depth_png(Z, depth_path)

    # Save disparity visualization
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, f"{label}_disparity.jpg"), disp_color,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Build point cloud from valid depth
    valid = np.isfinite(Z) & (Z > 0)
    pc = depth_to_pointcloud(points4d, rect_l, valid)
    ply_path = os.path.join(out_dir, f"{label}_cloud.ply")
    save_ply(pc, ply_path)

    valid_count = valid.sum()
    Z_valid = Z[valid]
    if verbose:
        print(f" done. depth range [{Z_valid.min():.0f}, {Z_valid.max():.0f}]mm  "
              f"{valid_count/1e6:.2f}M points → {label}_depth.png")

    return {
        'label': label, 'depth_path': depth_path, 'ply_path': ply_path,
        'Z_mean': float(Z_valid.mean()), 'Z_median': float(np.median(Z_valid)),
        'valid_pixels': int(valid_count),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='L16 stereo depth estimation')
    ap.add_argument('frames_dir',       help='Directory with A1.png, A2.png, ...')
    ap.add_argument('calibration_json', help='calibration.json from lri_calibration.py')
    ap.add_argument('output_dir',       nargs='?', default='.',
                    help='Output directory [default: current dir]')
    ap.add_argument('--pairs',  nargs='+', default=None,
                    help='Camera pairs to process, e.g. A1 A5 B1 B4 (must be even count)')
    ap.add_argument('--scale',  type=int, default=2,
                    help='Downsample factor for speed [default: 2]')
    ap.add_argument('--ndisp',  type=int, default=256,
                    help='Number of disparity levels [default: 256]')
    ap.add_argument('--all-pairs', action='store_true',
                    help='Process all configured stereo pairs (slow)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cameras = load_calibration(args.calibration_json)
    print(f"Loaded calibration for: {sorted(cameras)}")

    # Determine which pairs to run
    if args.pairs:
        if len(args.pairs) % 2 != 0:
            print("ERROR: --pairs must list camera names in pairs (even count)", file=sys.stderr)
            sys.exit(1)
        pairs = [(args.pairs[i], args.pairs[i+1],
                  f"{args.pairs[i]}-{args.pairs[i+1]}")
                 for i in range(0, len(args.pairs), 2)]
    elif args.all_pairs:
        pairs = STEREO_PAIRS
    else:
        # Default: just the primary wide-baseline A pair + primary B pair
        pairs = [STEREO_PAIRS[0], STEREO_PAIRS[4]]

    print(f"Processing {len(pairs)} stereo pair(s) at 1/{args.scale} scale ...\n")

    results = []
    for name_l, name_r, label in pairs:
        result = process_pair(
            name_l, name_r, label,
            cameras, args.frames_dir, args.output_dir,
            scale=args.scale, num_disparities=args.ndisp,
        )
        if result:
            results.append(result)

    # Summary
    print(f"\nDone. {len(results)} depth maps saved to {args.output_dir}")
    for r in results:
        print(f"  {r['label']}: median depth {r['Z_median']:.0f}mm, "
              f"{r['valid_pixels']//1000}K valid pixels")


if __name__ == '__main__':
    main()
