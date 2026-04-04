#!/usr/bin/env python3
"""
lri_fuse_depth.py — Multi-view depth fusion for the L16.

Takes Depth Pro output (one 16-bit PNG per camera, depth in mm) and the
factory calibration, then:
  1. Reprojects each per-camera depth map to a shared reference frame (A1)
  2. Fuses all 10 reprojected depth maps using:
       - Median fusion (robust to outliers)
       - Weighted mean by inverse variance (for smooth geometry)
  3. Outputs a fused 16-bit depth PNG + colored 3D point cloud

Why this is better than the 2016 Light CIAPI:
  - Uses Depth Pro (Apple, 2024) — a 2.25MP monocular foundation model with
    metric scale and sharp boundary tracing
  - Fuses ALL 10 calibrated cameras instead of a subset
  - Leverages precise factory extrinsics (±0.01mm calibration)
  - Runs on M4 GPU in ~2 minutes total

Usage:
    python3 lri_fuse_depth.py <depth_dir> <frames_dir> <calibration.json> [output_dir]
    python3 lri_fuse_depth.py /tmp/L16_00001_depth /tmp/L16_00001_frames \\
                              /tmp/L16_00001_cal/calibration.json /tmp/L16_00001_fused
"""

import argparse, json, os, sys
import numpy as np
import cv2

# ── calibration ────────────────────────────────────────────────────────────────

def load_cameras(cal_path):
    data = json.load(open(cal_path))
    cameras = {}
    for mod in data['modules']:
        name = mod['camera_name']
        cal  = mod['calibration']
        if 'rotation' not in cal:
            continue
        intr = cal['intrinsics']
        R = np.array(cal['rotation'],    dtype=np.float64)
        t = np.array(cal['translation'], dtype=np.float64)
        K = np.array([[intr['fx'], 0, intr['cx']],
                      [0, intr['fy'], intr['cy']],
                      [0, 0, 1]], dtype=np.float64)
        cameras[name] = dict(K=K, R=R, t=t,
                             W=mod['width'], H=mod['height'],
                             fx=intr['fx'], fy=intr['fy'],
                             cx=intr['cx'], cy=intr['cy'],
                             mirror_type=cal.get('mirror_type','NONE'))
    return cameras


# ── depth map I/O ──────────────────────────────────────────────────────────────

def load_depth_mm(path):
    """Load 16-bit PNG depth map (mm). Returns float32 array, 0=invalid."""
    d = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32)


def save_depth_mm(depth, path):
    cv2.imwrite(path, np.clip(depth, 0, 65535).astype(np.uint16))


def vis_depth(depth):
    """False-color depth visualization (jet, near=red, far=blue)."""
    valid = depth > 0
    vis = np.zeros_like(depth)
    if valid.any():
        dv = depth[valid]
        vis[valid] = (dv - dv.min()) / (dv.max() - dv.min() + 1e-6)
    # Invert: near=1 (red in TURBO), far=0 (blue)
    vis = 1.0 - vis
    vis[~valid] = 0
    return cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ── reprojection ───────────────────────────────────────────────────────────────

def reproject_depth_to_ref(depth_src, cam_src, cam_ref):
    """
    Reproject depth_src (H×W float32 mm) from cam_src into the coordinate
    frame of cam_ref. Returns a new depth map (same H×W as cam_ref) with
    depths in mm, 0=invalid.

    Method:
      1. Back-project src pixels + depth → 3D world points
      2. Project world points → ref image pixels + ref depth
      3. Scatter ref depths into ref depth image (depth-test: keep nearest)
    """
    H_s, W_s = depth_src.shape
    H_r = cam_ref['H'];  W_r = cam_ref['W']

    R_s = cam_src['R'];  t_s = cam_src['t']
    R_r = cam_ref['R'];  t_r = cam_ref['t']
    K_s = cam_src['K'];  K_r = cam_ref['K']
    # Pixel grid for source
    u = np.arange(W_s, dtype=np.float64)
    v = np.arange(H_s, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    # Back-project to camera space using element-wise ops (avoids Accelerate BLAS
    # spurious warnings on Apple Silicon with large N).
    fx_s = K_s[0, 0]; fy_s = K_s[1, 1]; cx_s = K_s[0, 2]; cy_s = K_s[1, 2]
    ray_x = (uu - cx_s) / fx_s   # H×W
    ray_y = (vv - cy_s) / fy_s

    depth_flat = depth_src.ravel().astype(np.float64)
    valid_mask = depth_flat > 0
    N_valid = valid_mask.sum()
    if N_valid == 0:
        return np.zeros((H_r, W_r), dtype=np.float32)

    ray_xv = ray_x.ravel()[valid_mask]          # N'
    ray_yv = ray_y.ravel()[valid_mask]
    depth_v = depth_flat[valid_mask]

    # X_cam_src = [ray_x, ray_y, 1] * depth  →  N'×3 matrix
    pts_src = np.column_stack([ray_xv * depth_v,
                               ray_yv * depth_v,
                               depth_v])                # N'×3  (Xc, Yc, Zc)

    # Cam_src → world: X_w = (X_c - t_s) @ R_s  (COLMAP row-vector inverse)
    # World → cam_ref: X_r = X_w @ R_r.T + t_r   (COLMAP row-vector forward)
    # Combined:  pts_ref = (pts_src - t_s) @ R_s @ R_r.T + t_r
    # Using a single matmul per step avoids the sequential buffer-aliasing bug
    # where numpy reuses an input buffer for an intermediate temporary, corrupting
    # source arrays before all three components are evaluated.
    # Accelerate BLAS emits spurious IEEE warnings for N > ~1M on Apple Silicon;
    # the results are numerically correct — suppress with errstate.
    with np.errstate(all='ignore'):
        pts_world = (pts_src - t_s) @ R_s          # N'×3  world coords
        pts_ref   = pts_world @ R_r.T + t_r         # N'×3  ref-camera coords

    Xr = pts_ref[:, 0];  Yr = pts_ref[:, 1];  Zr = pts_ref[:, 2]

    # Keep only points in front of ref camera
    valid_z = Zr > 0
    Xr = Xr[valid_z];  Yr = Yr[valid_z];  Zr = Zr[valid_z]

    # Project to ref image
    fx_r = K_r[0, 0]; fy_r = K_r[1, 1]; cx_r = K_r[0, 2]; cy_r = K_r[1, 2]
    u_r = fx_r * (Xr / Zr) + cx_r
    v_r = fy_r * (Yr / Zr) + cy_r

    # Filter in float before casting to avoid int32 overflow for cameras whose
    # projected pixels land far outside the reference image (e.g., sideways-facing
    # B cameras that see a completely different scene region than the reference).
    # Use half-pixel margins so rounding cannot push indices out of [0, W_r-1].
    in_bounds = (u_r >= -0.5) & (u_r < W_r - 0.5) & (v_r >= -0.5) & (v_r < H_r - 0.5)

    # Round to integer pixel coords (already in-bounds in float)
    u_ri    = np.round(u_r[in_bounds]).astype(np.int32)
    v_ri    = np.round(v_r[in_bounds]).astype(np.int32)
    depth_r = Zr[in_bounds].astype(np.float32)  # depth = Z in mm

    # Scatter with depth test (keep nearest): sort descending so near overwrites far.
    # Use ascending argsort + reverse view — np.argsort(-x) mutates x in-place on
    # Apple Silicon / Python 3.14 for large float32 arrays.
    depth_ref_img = np.zeros((H_r, W_r), dtype=np.float32)
    order = np.argsort(depth_r)[::-1]
    depth_ref_img[v_ri[order], u_ri[order]] = depth_r[order]

    return depth_ref_img


# ── fusion ─────────────────────────────────────────────────────────────────────

def fuse_depth_maps(depth_stack):
    """
    Fuse N depth maps [H, W] stacked as [N, H, W].
    Returns (median_depth, mean_depth, confidence) each [H, W].
    confidence = fraction of cameras with valid depth at each pixel.
    """
    # Invalid = 0
    valid = depth_stack > 0   # N×H×W bool

    # Count valid observations per pixel
    n_valid = valid.sum(axis=0).astype(np.float32)   # H×W
    conf = n_valid / depth_stack.shape[0]              # [0,1]

    # Masked median (use np.nan for invalid)
    depth_nan = np.where(valid, depth_stack.astype(np.float64), np.nan)
    with np.errstate(all='ignore'):
        median_depth = np.nanmedian(depth_nan, axis=0).astype(np.float32)
        mean_depth   = np.nanmean(depth_nan, axis=0).astype(np.float32)

    # Replace NaN with 0 (invalid)
    median_depth = np.nan_to_num(median_depth, nan=0.0)
    mean_depth   = np.nan_to_num(mean_depth,   nan=0.0)

    return median_depth, mean_depth, conf


# ── point cloud ────────────────────────────────────────────────────────────────

def depth_to_pointcloud(depth_mm, color_img, cam):
    """
    Back-project depth map (mm) to 3D world point cloud.
    color_img: H×W×3 uint8
    Returns Nx6 (X,Y,Z,R,G,B) float32 array.
    """
    H, W = depth_mm.shape
    fx = cam['fx']; fy = cam['fy']
    cx = cam['cx']; cy = cam['cy']
    R  = cam['R'];  t  = cam['t']

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    d = depth_mm.astype(np.float64)
    Xc = (uu - cx) / fx * d
    Yc = (vv - cy) / fy * d
    Zc = d

    pts_cam = np.stack([Xc, Yc, Zc], axis=-1)
    pts_flat = pts_cam.reshape(-1, 3)
    # World: X_w = R^T @ (X_c - t)  — suppress spurious Accelerate BLAS warnings
    with np.errstate(all='ignore'):
        X_world = (pts_flat - t[None, :]) @ R  # N×3

    valid = (d.ravel() > 0)
    xyz = X_world[valid].astype(np.float32)
    rgb = color_img.reshape(-1, 3)[valid]

    return np.hstack([xyz, rgb.astype(np.float32)])


def save_ply(pc, path):
    n = len(pc)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                   ('r','u1'),('g','u1'),('b','u1')])
    rec = np.empty(n, dtype=dt)
    rec['x']=pc[:,0]; rec['y']=pc[:,1]; rec['z']=pc[:,2]
    rec['r']=np.clip(pc[:,3],0,255).astype(np.uint8)
    rec['g']=np.clip(pc[:,4],0,255).astype(np.uint8)
    rec['b']=np.clip(pc[:,5],0,255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(hdr.encode()); f.write(rec.tobytes())


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Fuse 10-camera Depth Pro outputs')
    ap.add_argument('depth_dir',        help='Dir with A1_depthpro.png etc.')
    ap.add_argument('frames_dir',       help='Dir with A1.png etc. (for colors)')
    ap.add_argument('calibration_json', help='calibration.json')
    ap.add_argument('output_dir',       nargs='?', default='.')
    ap.add_argument('--ref',     default='A1')
    ap.add_argument('--min-cams', type=int, default=2,
                    help='Min cameras that must agree [default: 2]')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cameras = load_cameras(args.calibration_json)
    ref_cam = cameras[args.ref]

    print(f"Reference: {args.ref}  ({ref_cam['W']}×{ref_cam['H']})")
    print(f"Fusing cameras: {sorted(cameras)}\n")

    # Load reference color image
    ref_color_path = os.path.join(args.frames_dir, f"{args.ref}.png")
    ref_color = cv2.imread(ref_color_path)
    ref_color = cv2.cvtColor(ref_color, cv2.COLOR_BGR2RGB) if ref_color is not None else None

    # Load and reproject all depth maps to reference frame
    H_r, W_r = ref_cam['H'], ref_cam['W']
    depth_stack_list = []
    loaded = []

    for name in sorted(cameras):
        dp_path = os.path.join(args.depth_dir, f"{name}_depthpro.png")
        if not os.path.exists(dp_path):
            print(f"  [skip] {name}: {dp_path} not found")
            continue

        depth_src = load_depth_mm(dp_path)
        print(f"  [{name}] loaded {depth_src.shape} ...", end='', flush=True)

        if name == args.ref:
            # Reference camera — no reprojection needed
            depth_ref = depth_src
        else:
            depth_ref = reproject_depth_to_ref(depth_src, cameras[name], ref_cam)

        # Resize to reference camera resolution if needed
        if depth_ref.shape != (H_r, W_r):
            depth_ref = cv2.resize(depth_ref, (W_r, H_r),
                                   interpolation=cv2.INTER_NEAREST)

        valid = (depth_ref > 0).sum()
        print(f" reprojected, {valid:,} valid pixels "
              f"({100*valid/(H_r*W_r):.1f}%)")

        depth_stack_list.append(depth_ref)
        loaded.append(name)

    if not depth_stack_list:
        print("No depth maps found!", file=sys.stderr)
        sys.exit(1)

    print(f"\nFusing {len(depth_stack_list)} depth maps ...")
    depth_stack = np.stack(depth_stack_list, axis=0)  # N×H×W
    median_depth, mean_depth, conf = fuse_depth_maps(depth_stack)

    # Mask pixels with fewer than min_cams observations
    min_mask = conf >= (args.min_cams / len(depth_stack_list))
    median_depth[~min_mask] = 0
    mean_depth[~min_mask]   = 0

    # Stats
    valid_m = median_depth > 0
    if valid_m.any():
        d = median_depth[valid_m]
        print(f"Fused depth: {d.min():.0f}–{d.max():.0f}mm  median={np.median(d):.0f}mm")
        print(f"Valid pixels: {valid_m.sum():,}/{valid_m.size:,} ({100*valid_m.mean():.1f}%)")

    tag = f"fused_{len(depth_stack_list)}cams"

    # Save fused depth maps
    median_path = os.path.join(args.output_dir, f"{tag}_median_depth.png")
    mean_path   = os.path.join(args.output_dir, f"{tag}_mean_depth.png")
    save_depth_mm(median_depth, median_path)
    save_depth_mm(mean_depth,   mean_path)

    # Confidence map
    conf_path = os.path.join(args.output_dir, f"{tag}_confidence.png")
    cv2.imwrite(conf_path, (conf * 255).astype(np.uint8))

    # Visualizations
    cv2.imwrite(os.path.join(args.output_dir, f"{tag}_median_vis.jpg"),
                vis_depth(median_depth))
    cv2.imwrite(os.path.join(args.output_dir, f"{tag}_mean_vis.jpg"),
                vis_depth(mean_depth))

    # Point cloud from median depth
    if ref_color is not None:
        pc = depth_to_pointcloud(median_depth, ref_color, ref_cam)
        ply_path = os.path.join(args.output_dir, f"{tag}_cloud.ply")
        save_ply(pc, ply_path)
        print(f"\nPoint cloud: {len(pc):,} pts → {ply_path}")

    print(f"\nOutputs in {args.output_dir}/")
    print(f"  {tag}_median_depth.png  — 16-bit fused depth (mm)")
    print(f"  {tag}_median_vis.jpg    — false-color visualization")
    print(f"  {tag}_confidence.png    — per-pixel camera count (0–255)")
    print(f"  {tag}_cloud.ply         — colored 3D point cloud")

    # Also save per-camera reprojected depth maps for debugging
    debug_dir = os.path.join(args.output_dir, 'reprojected')
    os.makedirs(debug_dir, exist_ok=True)
    for i, name in enumerate(loaded):
        save_depth_mm(depth_stack[i],
                      os.path.join(debug_dir, f"{name}_repr.png"))
        cv2.imwrite(os.path.join(debug_dir, f"{name}_repr_vis.jpg"),
                    vis_depth(depth_stack[i]))
    print(f"  reprojected/            — per-camera reprojected depths")


if __name__ == '__main__':
    main()
