#!/usr/bin/env python3
"""
Run Depth Anything V2 metric (VKITTI ViT-L) on L16 B-camera frames.

Produces metric depth maps (absolute metres) for B-cameras, which have
no stereo baseline — monocular depth is the only option for these 70mm
cameras with movable mirrors.

Usage:
  lri_run_dav2.py <frames_dir> <calibration_json> [--cameras B1 B2 ...]
                  [--output-dir DIR] [--device mps|cpu]

Example:
  lri_run_dav2.py /Volumes/Dev/Light_Work_scratch/frames \
                  /Volumes/Dev/Light_Work_scratch/calibration.json \
                  --cameras B4 \
                  --output-dir /Volumes/Dev/Light_Work_scratch/depth_dav2/ \
                  --device mps
"""
import sys
import os
import argparse
import time
import numpy as np
import cv2

# DA-V2 paths
DAV2_ROOT   = '/Users/ryaker/Documents/ml-depth-anything-v2'
METRIC_ROOT = os.path.join(DAV2_ROOT, 'metric_depth')
CKPT_PATH   = os.path.join(METRIC_ROOT, 'checkpoints',
                            'depth_anything_v2_metric_vkitti_vitl.pth')

sys.path.insert(0, DAV2_ROOT)
sys.path.insert(0, METRIC_ROOT)

# L16 pipeline helpers
sys.path.insert(0, '/Users/ryaker/Documents/Light_Work')

import torch
from depth_anything_v2.dpt import DepthAnythingV2


# B-cameras: 70mm, movable mirrors — no stereo baseline available
DEFAULT_B_CAMERAS = ['B1', 'B2', 'B3', 'B4', 'B5']

# ViT-L config for metric depth
MODEL_CONFIG = {
    'encoder': 'vitl',
    'features': 256,
    'out_channels': [256, 512, 1024, 1024],
}

# VKITTI outdoor depth ceiling (metres)
MAX_DEPTH = 80


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Depth Anything V2 metric on L16 B-camera frames.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('frames_dir',
                        help='Directory containing <camera>.png frames')
    parser.add_argument('calibration_json',
                        help='Path to calibration.json')
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='Camera names to process. Default: B1-B5.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for .npz/.png files. '
                             'Default: <frames_dir>/../depth_dav2/')
    parser.add_argument('--device', default=None,
                        choices=['mps', 'cpu'],
                        help='Compute device. Default: mps if available, else cpu.')
    return parser.parse_args()


def load_cameras(cal_path: str) -> dict:
    """Load camera metadata via lri_fuse_image.load_cameras (returns K, R, t, W, H)."""
    from lri_fuse_image import load_cameras as _load_cameras
    return _load_cameras(cal_path)


def apply_srgb_gamma(image_uint8: np.ndarray) -> np.ndarray:
    """
    Convert linear-light uint8 image to sRGB-gamma-encoded uint8.

    L16 frames are stored as linear light (linear raw → uint16 high-byte → uint8).
    DA-V2 (like Depth Pro) was trained on sRGB-gamma images, so we must apply
    the IEC 61966-2-1 sRGB transfer function before inference.
    """
    image_f = image_uint8.astype(np.float32) / 255.0
    mask = image_f <= 0.0031308
    image_srgb = np.where(
        mask,
        12.92 * image_f,
        1.055 * (image_f ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(image_srgb * 255.0, 0, 255).astype(np.uint8)


def save_depth_png(depth_m: np.ndarray, path: str) -> None:
    """Save depth map as uint16 PNG in millimetres (OpenCV compatible)."""
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_mm)


def main():
    args = parse_args()

    frames_dir = args.frames_dir
    cal_path   = args.calibration_json

    if not os.path.isdir(frames_dir):
        print(f"Error: frames_dir '{frames_dir}' not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(cal_path):
        print(f"Error: calibration.json '{cal_path}' not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(CKPT_PATH):
        print(f"Error: checkpoint not found at {CKPT_PATH}", file=sys.stderr)
        print("  Run: python3 -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download(repo_id='depth-anything/Depth-Anything-V2-Metric-VKITTI-Large', "
              "filename='depth_anything_v2_metric_vkitti_vitl.pth', "
              "local_dir='/Users/ryaker/Documents/ml-depth-anything-v2/metric_depth/checkpoints')\"",
              file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(frames_dir.rstrip('/')), 'depth_dav2')
    os.makedirs(output_dir, exist_ok=True)

    # Camera list
    try:
        cameras_meta = load_cameras(cal_path)
    except Exception as e:
        print(f"Error loading calibration: {e}", file=sys.stderr)
        sys.exit(1)

    if args.cameras:
        cameras = args.cameras
    else:
        # Default to B-cameras present in calibration
        cameras = [c for c in DEFAULT_B_CAMERAS if c in cameras_meta]
        if not cameras:
            print("No B-cameras found in calibration. Pass --cameras explicitly.",
                  file=sys.stderr)
            sys.exit(1)

    missing_cal = [c for c in cameras if c not in cameras_meta]
    if missing_cal:
        print(f"Warning: cameras not in calibration (no intrinsics): {missing_cal}",
              file=sys.stderr)
        cameras = [c for c in cameras if c in cameras_meta]

    print(f"Cameras to process: {cameras}")
    print(f"Output dir: {output_dir}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading DA-V2 metric ViT-L (max_depth={MAX_DEPTH}m)...")
    t0 = time.time()
    model = DepthAnythingV2(**MODEL_CONFIG, max_depth=MAX_DEPTH)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    )
    model = model.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Inference loop
    results = {}
    for cam_name in cameras:
        img_path  = os.path.join(frames_dir, f"{cam_name}.png")
        out_npz   = os.path.join(output_dir, f"dav2_{cam_name}.npz")
        out_png   = os.path.join(output_dir, f"dav2_{cam_name}.png")

        if os.path.exists(out_npz):
            print(f"\n[SKIP] {cam_name}: already exists at {out_npz}")
            continue

        if not os.path.exists(img_path):
            print(f"\n[SKIP] {cam_name}: image not found at {img_path}")
            continue

        meta = cameras_meta[cam_name]
        fx   = float(meta['K'][0, 0])
        print(f"\n[{cam_name}] f_px={fx:.2f}  image={img_path}")

        t_start = time.time()

        # Load image (BGR uint8, as OpenCV returns)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  Error: could not read {img_path}", file=sys.stderr)
            continue

        # L16 images are linear light — apply sRGB gamma before inference.
        # infer_image expects uint8 BGR sRGB (same as a normal camera JPEG).
        image_bgr = apply_srgb_gamma(image_bgr)

        print(f"  shape={image_bgr.shape}  dtype={image_bgr.dtype}  "
              f"p50={np.median(image_bgr):.0f}  p99={np.percentile(image_bgr, 99):.0f}")

        # infer_image handles resizing, normalisation, and device transfer internally.
        # Returns float32 numpy array (H, W) in metres.
        with torch.no_grad():
            depth = model.infer_image(image_bgr, input_size=518)

        depth = depth.astype(np.float32)

        # Save NPZ (float32, metres)
        np.savez_compressed(out_npz, depth=depth)

        # Save uint16 PNG (millimetres)
        save_depth_png(depth, out_png)

        elapsed = time.time() - t_start
        stats = {
            'min':    float(depth.min()),
            'max':    float(depth.max()),
            'median': float(np.median(depth)),
            'time':   elapsed,
        }
        results[cam_name] = stats
        print(f"  depth range: {stats['min']:.2f}–{stats['max']:.2f}m  "
              f"median: {stats['median']:.2f}m  ({elapsed:.1f}s)")
        print(f"  saved: {out_npz}")
        print(f"  saved: {out_png}")

    # Summary
    print("\n=== Summary ===")
    for cam, r in results.items():
        print(f"  {cam}: {r['min']:.2f}–{r['max']:.2f}m  "
              f"median={r['median']:.2f}m  time={r['time']:.1f}s")

    if not results:
        print("  (no new depth maps processed)")

    print("\nDone.")


if __name__ == '__main__':
    main()
