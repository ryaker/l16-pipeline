#!/usr/bin/env python3
"""
Run Metric3D V2 ViT-Small on L16 camera frames with calibrated intrinsics.

Uses canonical-space depth normalisation: depth_metric = depth_canonical * (fx_scaled / 1000.0)
where 1000.0 is the focal length of Metric3D's canonical camera and fx_scaled is the
camera's calibrated fx rescaled to match the model input resolution.

This is architecturally correct for L16 B-cameras (fx ≈ 8276–8300 px, 70 mm equivalent),
where assuming a typical focal length would give badly wrong scale.

Usage:
  lri_run_metric3d.py <frames_dir> <calibration_json> [options]

Examples:
  # Single camera, explicit output dir
  lri_run_metric3d.py /path/to/frames calibration.json --cameras B4 --output-dir /path/to/out

  # All cameras with MPS device
  lri_run_metric3d.py /path/to/frames calibration.json --device mps

  # CPU fallback (slower but no MPS quirks)
  lri_run_metric3d.py /path/to/frames calibration.json --device cpu
"""

# ── MPS patch applied to hub cache ───────────────────────────────────────────
# The Metric3D V2 decoder hard-codes device="cuda" in get_bins().
# Patched file: ~/.cache/torch/hub/yvanyin_metric3d_main/mono/model/decode_heads/
#               RAFTDepthNormalDPTDecoder5.py  (get_bins method)
# Fix: derive device from next(self.parameters()).device instead of "cuda".
# This is a one-time edit to the hub cache; if the cache is cleared, re-apply.
# ─────────────────────────────────────────────────────────────────────────────

import os
# MUST be set before torch import so MPS-unsupported ops fall back to CPU
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
import json
import time
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

# ── Metric3D hub path ─────────────────────────────────────────────────────────
METRIC3D_HUB = os.path.expanduser('~/.cache/torch/hub/yvanyin_metric3d_main')

# ViT-Small input resolution (H, W) as required by the model
VIT_INPUT_SIZE = (616, 1064)

# ImageNet mean/std used by Metric3D V2 preprocessing
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD  = np.array([58.395,  57.12,  57.375],  dtype=np.float32)

# Metric3D canonical focal length (model was trained with 1000 px canonical fx)
CANONICAL_FX = 1000.0


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Metric3D V2 ViT-Small on L16 camera frames.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('frames_dir',
        help='Directory containing <camera>.png frames')
    parser.add_argument('calibration_json',
        help='Path to calibration.json (factory intrinsics)')
    parser.add_argument('--cameras', nargs='*', default=None,
        help='Camera names to process (default: all in calibration)')
    parser.add_argument('--output-dir', default=None,
        help='Output directory for depth/*.npz and PNG visualisations '
             '(default: <frames_dir>/../depth_metric3d)')
    parser.add_argument('--device', default=None,
        help='PyTorch device: mps | cpu | cuda (default: mps if available, else cpu)')
    parser.add_argument('--overwrite', action='store_true',
        help='Re-run even if output already exists')
    return parser.parse_args()


# ── calibration ───────────────────────────────────────────────────────────────

def load_cameras(cal_path: str) -> dict:
    """Load K matrices from calibration.json using lri_fuse_image.load_cameras format."""
    with open(cal_path) as f:
        data = json.load(f)
    cameras = {}
    for mod in data['modules']:
        name = mod['camera_name']
        cal  = mod.get('calibration')
        if not cal or 'intrinsics' not in cal or cal['intrinsics'] is None:
            continue
        intr = cal['intrinsics']
        cameras[name] = dict(
            fx=float(intr['fx']),
            fy=float(intr['fy']),
            cx=float(intr['cx']),
            cy=float(intr['cy']),
            W=int(mod.get('width',  4160)),
            H=int(mod.get('height', 3120)),
        )
    return cameras


# ── image preprocessing ───────────────────────────────────────────────────────

def apply_srgb_gamma(img_uint8: np.ndarray) -> np.ndarray:
    """
    Apply sRGB gamma to a linear-light uint8 image.

    L16 camera images are linear light; Metric3D (like Depth Pro) was trained
    on sRGB-encoded images.  Skipping this step makes the model see a very dark,
    flat-contrast image and degrades depth predictions.
    """
    img_f = img_uint8.astype(np.float32) / 255.0
    mask = img_f <= 0.0031308
    srgb = np.where(mask, 12.92 * img_f, 1.055 * (img_f ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def preprocess_image(img_bgr: np.ndarray, intrinsics: dict) -> tuple:
    """
    Resize image to VIT_INPUT_SIZE with keep-ratio + padding, scale intrinsics.

    Returns:
        rgb_tensor  : float32 tensor (1, 3, H_in, W_in) normalised with ImageNet stats
        pad_info    : (pad_top, pad_bot, pad_left, pad_right) applied padding
        scale       : scale factor applied to image (to rescale fx/fy)
    """
    h_orig, w_orig = img_bgr.shape[:2]
    rgb_origin = img_bgr[:, :, ::-1]  # BGR → RGB

    # Keep-ratio resize
    scale = min(VIT_INPUT_SIZE[0] / h_orig, VIT_INPUT_SIZE[1] / w_orig)
    new_h = int(h_orig * scale)
    new_w = int(w_orig * scale)
    rgb = cv2.resize(rgb_origin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Constant-value padding to exact model input size
    pad_h     = VIT_INPUT_SIZE[0] - new_h
    pad_w     = VIT_INPUT_SIZE[1] - new_w
    pad_top   = pad_h // 2
    pad_bot   = pad_h - pad_top
    pad_left  = pad_w // 2
    pad_right = pad_w - pad_left
    padding   = [123.675, 116.28, 103.53]  # ImageNet mean as fill
    rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bot, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=padding)

    # ImageNet normalisation
    rgb_f = rgb.astype(np.float32)
    rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD  # (H, W, 3)
    tensor = torch.from_numpy(rgb_f.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, 3, H, W)

    return tensor, (pad_top, pad_bot, pad_left, pad_right), scale


# ── depth postprocessing ──────────────────────────────────────────────────────

def postprocess_depth(pred_depth: torch.Tensor,
                      pad_info: tuple,
                      orig_h: int, orig_w: int,
                      fx_orig: float,
                      scale: float) -> np.ndarray:
    """
    Remove padding, upsample to original size, apply canonical→metric conversion.

    canonical_to_real_scale = fx_scaled / CANONICAL_FX
    where fx_scaled = fx_orig * scale  (fx rescaled to match model input resolution)
    """
    depth = pred_depth.squeeze()  # (H_in, W_in)
    pad_top, pad_bot, pad_left, pad_right = pad_info

    # Un-pad
    h_in, w_in = depth.shape
    depth = depth[
        pad_top : h_in - pad_bot if pad_bot > 0 else h_in,
        pad_left : w_in - pad_right if pad_right > 0 else w_in,
    ]

    # Upsample to original image size
    depth = torch.nn.functional.interpolate(
        depth[None, None, :, :],
        (orig_h, orig_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze()

    # Canonical → metric  (fx scaled to match the resized input)
    fx_scaled = fx_orig * scale
    canonical_to_real = fx_scaled / CANONICAL_FX
    depth = depth * canonical_to_real

    # Clamp to sane range (0–3000 m; B-cameras look at infinity)
    depth = torch.clamp(depth, 0.0, 3000.0)

    return depth.cpu().numpy().astype(np.float32)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> torch.nn.Module:
    """Load Metric3D V2 ViT-Small from torch hub (cached after first download)."""
    print("Loading Metric3D V2 ViT-Small...")
    t0 = time.time()
    model = torch.hub.load(
        'yvanyin/metric3d',
        'metric3d_vit_small',
        pretrain=True,
        trust_repo=True,
    )
    model = model.to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s  |  "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  |  "
          f"device: {next(model.parameters()).device}")
    return model


# ── per-camera inference ──────────────────────────────────────────────────────

def run_camera(model: torch.nn.Module,
               cam_name: str,
               img_path: str,
               intrinsics: dict,
               out_npz: str,
               out_png: str,
               device: torch.device,
               overwrite: bool = False) -> dict | None:
    """
    Run Metric3D V2 on one camera frame.

    Returns a result dict {min, max, median, time} or None if skipped.
    """
    if not os.path.exists(img_path):
        print(f"\n[SKIP] {cam_name}: image not found at {img_path}")
        return None

    if not overwrite and os.path.exists(out_npz):
        print(f"\n[SKIP] {cam_name}: output already exists at {out_npz}")
        return None

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    print(f"\n[{cam_name}]  fx={fx:.1f}  fy={fy:.1f}  image={img_path}")

    t_start = time.time()

    # ── load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(f"  ERROR: failed to read image")
        return None

    # Handle uint16 (L16 native): take high byte as uint8 approximation
    if img_bgr.dtype == np.uint16:
        img_bgr = (img_bgr >> 8).astype(np.uint8)

    # Handle grayscale
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    orig_h, orig_w = img_bgr.shape[:2]

    # sRGB gamma (L16 is linear light; model trained on sRGB)
    img_bgr_lin = img_bgr.copy()
    img_bgr = apply_srgb_gamma(img_bgr)
    print(f"  shape={img_bgr.shape}  "
          f"p50_before={np.median(img_bgr_lin):.0f}  "
          f"p50_after={np.median(img_bgr):.0f}")

    # ── preprocess ────────────────────────────────────────────────────────────
    rgb_tensor, pad_info, scale = preprocess_image(img_bgr, intrinsics)
    rgb_tensor = rgb_tensor.to(device)

    print(f"  resize scale={scale:.4f}  "
          f"fx_scaled={fx * scale:.1f}  "
          f"canonical_scale={fx * scale / CANONICAL_FX:.4f}")

    # ── inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb_tensor})

    # ── postprocess ───────────────────────────────────────────────────────────
    depth = postprocess_depth(pred_depth, pad_info, orig_h, orig_w, fx, scale)

    elapsed = time.time() - t_start

    # ── save outputs ──────────────────────────────────────────────────────────
    np.savez_compressed(out_npz, depth=depth)

    # PNG visualisation: uint16 in millimetres (same convention as depthpro)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_png, depth_mm)

    d_min    = float(depth.min())
    d_max    = float(depth.max())
    d_median = float(np.median(depth))

    print(f"  depth: min={d_min:.3f}m  max={d_max:.3f}m  median={d_median:.3f}m  "
          f"({elapsed:.1f}s)")
    print(f"  saved: {out_npz}")
    print(f"  saved: {out_png}")

    return dict(min=d_min, max=d_max, median=d_median, time=elapsed)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    frames_dir = args.frames_dir
    cal_path   = args.calibration_json

    if not os.path.isdir(frames_dir):
        print(f"ERROR: frames_dir '{frames_dir}' not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(cal_path):
        print(f"ERROR: calibration_json '{cal_path}' not found", file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(frames_dir.rstrip('/')), 'depth_metric3d')
    os.makedirs(out_dir, exist_ok=True)

    # Camera intrinsics
    cameras_cal = load_cameras(cal_path)
    if not cameras_cal:
        print("ERROR: no calibrated cameras found in calibration.json", file=sys.stderr)
        sys.exit(1)

    cameras = args.cameras if args.cameras else sorted(cameras_cal.keys())

    missing = [c for c in cameras if c not in cameras_cal]
    if missing:
        print(f"ERROR: cameras not in calibration: {missing}", file=sys.stderr)
        sys.exit(1)

    print("Cameras and intrinsics:")
    for cam in cameras:
        c = cameras_cal[cam]
        print(f"  {cam}: fx={c['fx']:.1f}  fy={c['fy']:.1f}  "
              f"cx={c['cx']:.1f}  cy={c['cy']:.1f}  ({c['W']}×{c['H']})")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # Load model (downloads ~300 MB checkpoint on first run)
    model = load_model(device)

    # Run per camera
    results = {}
    for cam_name in cameras:
        img_path = os.path.join(frames_dir, f"{cam_name}.png")
        out_npz  = os.path.join(out_dir, f"metric3d_{cam_name}.npz")
        out_png  = os.path.join(out_dir, f"metric3d_{cam_name}.png")

        r = run_camera(
            model, cam_name, img_path,
            cameras_cal[cam_name],
            out_npz, out_png,
            device,
            overwrite=args.overwrite,
        )
        if r is not None:
            results[cam_name] = r

    print("\n=== Summary ===")
    for cam, r in results.items():
        print(f"  {cam}: {r['min']:.3f}–{r['max']:.3f}m  "
              f"median={r['median']:.3f}m  time={r['time']:.1f}s")
    if not results:
        print("  (no new depth maps processed)")
    print("\nDone.")


if __name__ == '__main__':
    main()
