#!/usr/bin/env python3
"""
lri_depth_mps.py — Multi-view plane-sweep depth estimation for the L16.

Uses all 10 calibrated cameras simultaneously with PyTorch MPS (Apple M4 GPU).
Plane-sweep algorithm:
  1. For each depth hypothesis d in [d_min, d_max]:
       Warp all source images to the reference view at depth d
       Compute NCC-based photometric consistency
  2. WTA or semi-global selection
  3. Sub-pixel refinement via parabola fitting
  4. Confidence-weighted fusion across camera groups

This is fundamentally better than pairwise stereo because it uses ALL cameras,
giving redundant evidence and eliminating occlusion holes.

Usage:
    python3 lri_depth_mps.py <frames_dir> <calibration.json> [output_dir]
    python3 lri_depth_mps.py /tmp/L16_00001_frames /tmp/L16_00001_cal/calibration.json /tmp/L16_00001_depth
"""

import argparse, json, os, sys, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ── device selection ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── calibration loading ───────────────────────────────────────────────────────

def load_cameras(cal_path):
    """Load calibration → dict keyed by camera name."""
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
        center = -R.T @ t  # camera center in world coords
        cameras[name] = dict(K=K, R=R, t=t, center=center,
                             W=mod['width'], H=mod['height'],
                             fx=intr['fx'], fy=intr['fy'],
                             cx=intr['cx'], cy=intr['cy'],
                             mirror_type=cal.get('mirror_type', 'NONE'))
    return cameras


# ── image loading ─────────────────────────────────────────────────────────────

def load_image(path, scale=1):
    """Load RGB image, optionally downsample, return float32 [0,1] HxWx3."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if scale > 1:
        H, W = img.shape[:2]
        img = cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    return img


# ── plane-sweep MVS ───────────────────────────────────────────────────────────

class PlaneSweepMVS:
    """
    Multi-view plane-sweep stereo using all calibrated cameras.

    For each reference pixel (u,v) and depth hypothesis d:
      - Back-project to 3D: X = d * K_ref^-1 * [u, v, 1]^T in camera coords
      - Convert to world: X_w = R_ref^T @ (X - t_ref) ... actually:
        X_w = R_ref^T @ X - R_ref^T @ t_ref = R_ref^T @ (d * K^-1 @ [u,v,1]) + C_ref
      - Project into source camera s: x_s = K_s @ (R_s @ X_w + t_s)
      - Sample color from source image at x_s
      - Compute consistency across all sources

    Uses normalized cross-correlation (NCC) in local patches.
    """

    def __init__(self, cameras, ref_name, src_names, device,
                 depth_min=300.0, depth_max=15000.0, num_depths=192,
                 patch_radius=4, scale=4):
        self.cameras   = cameras
        self.ref_name  = ref_name
        self.src_names = [n for n in src_names if n in cameras and n != ref_name]
        self.device    = device
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depths = num_depths
        self.patch_radius = patch_radius
        self.scale     = scale

        ref = cameras[ref_name]
        self.W = ref['W'] // scale
        self.H = ref['H'] // scale

        # Pre-compute scaled intrinsics for reference
        s = 1.0 / scale
        K_ref = ref['K'].copy()
        K_ref[0] *= s; K_ref[1] *= s
        self.K_ref    = torch.tensor(K_ref, dtype=torch.float32, device=device)
        self.K_ref_inv = torch.linalg.inv(self.K_ref)
        self.R_ref    = torch.tensor(ref['R'], dtype=torch.float32, device=device)
        self.t_ref    = torch.tensor(ref['t'], dtype=torch.float32, device=device)

        # Pre-compute source camera matrices
        self.src_K  = []
        self.src_Rt = []  # [R | t] 3×4
        for n in self.src_names:
            sc = cameras[n]
            Ks = sc['K'].copy()
            Ks[0] *= s; Ks[1] *= s
            Rs = sc['R']
            ts = sc['t']
            K_Rt = Ks @ np.hstack([Rs, ts.reshape(3,1)])
            self.src_K.append(torch.tensor(Ks,  dtype=torch.float32, device=device))
            self.src_Rt.append(torch.tensor(K_Rt, dtype=torch.float32, device=device))

        # Pixel grid for reference image
        u = torch.arange(self.W, dtype=torch.float32, device=device)
        v = torch.arange(self.H, dtype=torch.float32, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # H×W
        ones = torch.ones_like(uu)
        # [3, H, W] homogeneous pixels
        self.pix_h = torch.stack([uu, vv, ones], dim=0)

        # Back-project rays in world coords for each pixel
        # ray = R_ref^T @ K_ref^-1 @ pix (direction in world)
        pix_flat = self.pix_h.reshape(3, -1)  # [3, H*W]
        ray_cam  = self.K_ref_inv @ pix_flat   # [3, H*W]  (normalized, z=1)
        # World coords of a point at depth d: X_w = R_ref^T @ (d * ray_cam) + C_ref
        # = d * R_ref^T @ ray_cam + C_ref
        self.ray_world = self.R_ref.T @ ray_cam  # [3, H*W]
        C_ref = -(self.R_ref.T @ self.t_ref)      # [3]
        self.C_ref = C_ref

    def warp_source_to_ref_depth(self, src_img_t, src_K_Rt, depth):
        """
        For a given depth hypothesis, compute the projected coords of all
        reference pixels in the source image and sample.

        src_img_t: [1, C, H, W] float32
        src_K_Rt:  [3, 4] float32 — K @ [R | t] for source cam
        depth:     scalar float

        Returns [1, C, H, W] warped source image.
        """
        HW = self.H * self.W

        # World points at this depth: X_w = d * ray_world + C_ref
        X_w = depth * self.ray_world + self.C_ref.unsqueeze(1)  # [3, HW]

        # Project to source: h_s = K_s @ R_s @ X_w + K_s @ t_s
        X_w_h = torch.cat([X_w, torch.ones(1, HW, device=self.device)], dim=0)  # [4, HW]
        x_s = src_K_Rt @ X_w_h  # [3, HW]

        # Normalize homogeneous
        z_s = x_s[2:3, :].clamp(min=1e-6)
        u_s = x_s[0, :] / z_s[0, :]  # [HW]
        v_s = x_s[1, :] / z_s[0, :]  # [HW]

        # Normalize to [-1, 1] for grid_sample
        u_n = (u_s / (self.W - 1)) * 2 - 1
        v_n = (v_s / (self.H - 1)) * 2 - 1

        grid = torch.stack([u_n, v_n], dim=1).reshape(1, self.H, self.W, 2)
        warped = F.grid_sample(src_img_t, grid,
                               mode='bilinear', padding_mode='zeros',
                               align_corners=True)
        # Zero-out behind-camera projections
        valid = (z_s[0] > 0).reshape(1, 1, self.H, self.W)
        warped = warped * valid.float()
        return warped

    def ncc_cost(self, ref_patch, src_patches, patch_r):
        """
        Compute NCC between reference patch and each source patch.
        ref_patch:  [1, C, H, W]
        src_patches: list of [1, C, H, W]
        Returns mean NCC cost [H, W] in [0, 1] (1=best match).
        """
        r = patch_r
        ks = 2 * r + 1

        def local_stats(x):
            # x: [1, C, H, W]
            mu = F.avg_pool2d(x, ks, stride=1, padding=r)
            mu2 = F.avg_pool2d(x**2, ks, stride=1, padding=r)
            var = (mu2 - mu**2).clamp(min=1e-6)
            return mu, var

        ref_mu, ref_var = local_stats(ref_patch)
        scores = []
        for src_p in src_patches:
            src_mu, src_var = local_stats(src_p)
            cov = F.avg_pool2d(ref_patch * src_p, ks, stride=1, padding=r) - ref_mu * src_mu
            ncc = cov / (ref_var.sqrt() * src_var.sqrt() + 1e-6)
            ncc = ncc.mean(dim=1, keepdim=True)  # avg over channels
            scores.append(ncc.clamp(-1, 1))
        if scores:
            return torch.stack(scores, dim=0).mean(dim=0).squeeze()  # [H, W]
        return torch.zeros(self.H, self.W, device=self.device)

    def run(self, ref_img, src_imgs, verbose=True):
        """
        Run full plane-sweep MVS.

        ref_img:  [H, W, 3] numpy float32
        src_imgs: list of [H, W, 3] numpy float32

        Returns depth_map [H, W] float32 (in mm, 0=invalid),
                confidence [H, W] float32.
        """
        t0 = time.time()
        H, W = self.H, self.W

        # Move to device
        def to_t(img):
            return torch.tensor(img, device=self.device, dtype=torch.float32
                                ).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

        ref_t  = to_t(ref_img)
        src_ts = [to_t(s) for s in src_imgs]

        # Inverse depth sweep (uniform in 1/d, better for near objects)
        inv_min = 1.0 / self.depth_max
        inv_max = 1.0 / self.depth_min
        inv_depths = torch.linspace(inv_min, inv_max, self.num_depths,
                                    device=self.device)
        depths = 1.0 / inv_depths  # [D]

        # Accumulate cost volume [D, H, W]
        cost_vol = torch.zeros(self.num_depths, H, W, device=self.device)

        if verbose:
            print(f"  Sweeping {self.num_depths} depth planes "
                  f"[{self.depth_min:.0f}–{self.depth_max:.0f}mm] "
                  f"with {len(src_ts)} source cams ...", flush=True)

        for di, d in enumerate(depths):
            d_val = d.item()
            # Warp all sources to reference at depth d
            warped = [self.warp_source_to_ref_depth(st, KRt, d_val)
                      for st, KRt in zip(src_ts, self.src_Rt)]
            cost = self.ncc_cost(ref_t, warped, self.patch_radius)
            cost_vol[di] = cost

            if verbose and (di % 32 == 0 or di == self.num_depths - 1):
                pct = 100 * (di + 1) / self.num_depths
                elapsed = time.time() - t0
                eta = elapsed / (di + 1) * (self.num_depths - di - 1)
                print(f"    [{pct:5.1f}%] d={d_val:.0f}mm  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

        # Winner-takes-all depth selection
        best_di  = cost_vol.argmax(dim=0)  # [H, W]
        best_ncc = cost_vol.max(dim=0).values  # [H, W]

        # Sub-pixel refinement via parabola fitting
        # d_sub = d_idx - (cost[d+1] - cost[d-1]) / (2*(cost[d+1] - 2*cost[d] + cost[d-1]))
        d_clamped = best_di.clamp(1, self.num_depths - 2)
        idx   = d_clamped.long()
        h_idx = torch.arange(H, device=self.device).unsqueeze(1).expand(H, W)
        w_idx = torch.arange(W, device=self.device).unsqueeze(0).expand(H, W)

        c0 = cost_vol[idx - 1, h_idx, w_idx]
        c1 = cost_vol[idx,     h_idx, w_idx]
        c2 = cost_vol[idx + 1, h_idx, w_idx]
        denom = (2 * c1 - c0 - c2).clamp(min=1e-6)
        sub   = (c0 - c2) / (2 * denom)
        sub_di = d_clamped.float() + sub  # [H, W] subpixel depth index

        # Convert index to depth (inverse depth interpolation)
        sub_inv_d = inv_min + sub_di / (self.num_depths - 1) * (inv_max - inv_min)
        depth_map = (1.0 / sub_inv_d.clamp(min=1e-9))  # [H, W]

        # Confidence mask: reject low NCC scores
        conf = (best_ncc + 1.0) / 2.0  # [0,1]
        depth_map = torch.where(conf > 0.3, depth_map, torch.zeros_like(depth_map))

        return depth_map.cpu().numpy(), conf.cpu().numpy()


# ── point cloud export ────────────────────────────────────────────────────────

def depth_to_pointcloud(depth_map, conf, ref_img, cameras, ref_name, scale):
    """Unproject depth map to 3D point cloud in world coordinates."""
    cam = cameras[ref_name]
    H, W = depth_map.shape
    s = 1.0 / scale

    fx = cam['fx'] * s
    fy = cam['fy'] * s
    cx = cam['cx'] * s
    cy = cam['cy'] * s
    R  = cam['R']
    t  = cam['t']

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Back-project to camera space at depth d
    Xc = (uu - cx) / fx * depth_map
    Yc = (vv - cy) / fy * depth_map
    Zc = depth_map

    # Camera to world
    # X_w = R^T @ (X_c - t)
    pts_cam = np.stack([Xc, Yc, Zc], axis=-1)  # H×W×3
    pts_flat = pts_cam.reshape(-1, 3)
    pts_world = (R.T @ (pts_flat - t).T).T        # N×3

    valid = (depth_map.ravel() > 0) & (conf.ravel() > 0.3)
    pts_world = pts_world[valid]
    colors = (ref_img.reshape(-1, 3)[valid] * 255).astype(np.uint8)

    return np.hstack([pts_world, colors])


def save_ply(pc, path):
    """Save Nx6 (X,Y,Z,R,G,B) as binary PLY."""
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
    rec['x'] = pc[:,0]; rec['y'] = pc[:,1]; rec['z'] = pc[:,2]
    rec['r'] = pc[:,3]; rec['g'] = pc[:,4]; rec['b'] = pc[:,5]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(rec.tobytes())


def save_depth_png(depth_map, path):
    """Save depth as 16-bit PNG (mm, 0=invalid)."""
    d = np.clip(depth_map, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, d)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='L16 multi-view plane-sweep depth on Apple M4 MPS GPU'
    )
    ap.add_argument('frames_dir',       help='Directory with A1.png, A2.png, ...')
    ap.add_argument('calibration_json', help='calibration.json from lri_calibration.py')
    ap.add_argument('output_dir',       nargs='?', default='.',
                    help='Output directory [default: current dir]')
    ap.add_argument('--ref',     default='A1',   help='Reference camera [default: A1]')
    ap.add_argument('--group',   default='A',
                    choices=['A', 'B', 'AB'],
                    help='Camera group to use: A (28mm), B (70mm), AB (all) [default: A]')
    ap.add_argument('--scale',   type=int, default=4,
                    help='Downsample factor [default: 4 → 1040×780]')
    ap.add_argument('--depths',  type=int, default=192,
                    help='Number of depth planes [default: 192]')
    ap.add_argument('--dmin',    type=float, default=300.0,
                    help='Min depth mm [default: 300]')
    ap.add_argument('--dmax',    type=float, default=15000.0,
                    help='Max depth mm [default: 15000]')
    ap.add_argument('--patch',   type=int, default=4,
                    help='NCC patch radius [default: 4]')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    cameras = load_cameras(args.calibration_json)
    print(f"Cameras: {sorted(cameras.keys())}")

    # Select camera group
    all_names = sorted(cameras.keys())
    if args.group == 'A':
        group = [n for n in all_names if n.startswith('A')]
    elif args.group == 'B':
        group = [n for n in all_names if n.startswith('B')]
    else:
        group = all_names

    if args.ref not in cameras:
        print(f"ERROR: Reference camera {args.ref} not in calibration", file=sys.stderr)
        sys.exit(1)

    src_names = [n for n in group if n != args.ref]
    print(f"Reference: {args.ref}  Sources: {src_names}")

    # Load images
    print("Loading images ...", flush=True)
    ref_img  = load_image(os.path.join(args.frames_dir, f"{args.ref}.png"), args.scale)
    src_imgs = []
    for n in src_names:
        p = os.path.join(args.frames_dir, f"{n}.png")
        if os.path.exists(p):
            src_imgs.append(load_image(p, args.scale))
        else:
            print(f"  [skip] {n}: image not found")
            src_names.remove(n)

    H, W = ref_img.shape[:2]
    print(f"Image size: {W}×{H}  ({len(src_imgs)} sources)")

    # Build MVS
    mvs = PlaneSweepMVS(
        cameras=cameras,
        ref_name=args.ref,
        src_names=src_names,
        device=device,
        depth_min=args.dmin,
        depth_max=args.dmax,
        num_depths=args.depths,
        patch_radius=args.patch,
        scale=args.scale,
    )

    # Run
    t0 = time.time()
    depth_map, conf = mvs.run(ref_img, src_imgs, verbose=True)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Stats
    valid = depth_map > 0
    if valid.any():
        d_valid = depth_map[valid]
        print(f"Depth range: {d_valid.min():.0f} – {d_valid.max():.0f} mm  "
              f"median {np.median(d_valid):.0f} mm")
        print(f"Valid pixels: {valid.sum()} / {valid.size} "
              f"({100*valid.mean():.1f}%)")

    # Save outputs
    tag = f"{args.ref}_group{args.group}_d{args.depths}_s{args.scale}"

    depth_path = os.path.join(args.output_dir, f"{tag}_depth.png")
    save_depth_png(depth_map, depth_path)
    print(f"Depth PNG: {depth_path}")

    # Depth visualization
    vis = depth_map.copy()
    vis_valid = vis > 0
    if vis_valid.any():
        vis[vis_valid] = (1.0 - (vis[vis_valid] - vis[vis_valid].min()) /
                          (vis[vis_valid].max() - vis[vis_valid].min() + 1e-6))
    vis8 = (vis * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(args.output_dir, f"{tag}_depth_vis.jpg"), vis_color)

    # Confidence visualization
    conf8 = (conf * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, f"{tag}_conf.jpg"), conf8)

    # Point cloud
    pc = depth_to_pointcloud(depth_map, conf, ref_img, cameras, args.ref, args.scale)
    ply_path = os.path.join(args.output_dir, f"{tag}_cloud.ply")
    save_ply(pc, ply_path)
    print(f"Point cloud ({len(pc):,} pts): {ply_path}")

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
