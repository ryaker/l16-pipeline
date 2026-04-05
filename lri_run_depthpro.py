#!/usr/bin/env python3
"""
Generalized script to run Apple Depth Pro on L16 camera frames.

Produces metric depth maps using calibration focal lengths.
Skips cameras that already have depth/<name>.npz.

Usage:
  lri_run_depthpro.py <lumen_dir> <calibration_json> [--cameras A1 A2 ...]

Example:
  lri_run_depthpro.py /Volumes/L16IMAGES/Light/2018-10-12/L16_02532_lumen \
                      /Volumes/L16IMAGES/Light/2018-10-12/L16_02532_cal/calibration.json \
                      --cameras A2 A3 A4 A5 B4
"""
import sys
import json
import time
import os
import argparse
import numpy as np
import cv2

# Must change to ml-depth-pro dir so relative checkpoint path resolves
ML_DEPTH_PRO_PATH = '/Users/ryaker/Documents/Light_Work/ml-depth-pro'
sys.path.insert(0, ML_DEPTH_PRO_PATH)
os.chdir(ML_DEPTH_PRO_PATH)

import torch
from depth_pro import create_model_and_transforms, load_rgb


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Apple Depth Pro on L16 camera frames.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'lumen_dir',
        help='Path to lumen directory (must contain frames/ subdirectory)'
    )
    parser.add_argument(
        'calibration_json',
        help='Path to calibration.json file'
    )
    parser.add_argument(
        '--cameras',
        nargs='*',
        default=None,
        help='Camera names to process (e.g., A1 A2 B4). If not specified, auto-detect from calibration.'
    )
    return parser.parse_args()


def load_calibration(cal_path):
    """Load calibration JSON and extract focal lengths for each camera."""
    with open(cal_path) as f:
        cal = json.load(f)
    
    focal_lengths = {}
    for m in cal['modules']:
        name = m['camera_name']
        fx = m['calibration']['intrinsics']['fx']
        focal_lengths[name] = fx
    
    return focal_lengths


def main():
    args = parse_args()
    
    lumen_dir = args.lumen_dir
    cal_path = args.calibration_json
    frames_dir = os.path.join(lumen_dir, 'frames')
    depth_dir = os.path.join(lumen_dir, 'depth')
    
    # Validate inputs
    if not os.path.isdir(lumen_dir):
        print(f"Error: lumen_dir '{lumen_dir}' not found", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(frames_dir):
        print(f"Error: frames directory not found at '{frames_dir}'", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(cal_path):
        print(f"Error: calibration file not found at '{cal_path}'", file=sys.stderr)
        sys.exit(1)
    
    # Load calibration focal lengths
    focal_lengths = load_calibration(cal_path)
    
    # Determine which cameras to process
    if args.cameras:
        cameras = args.cameras
    else:
        # Auto-detect: use all cameras in calibration
        cameras = sorted(focal_lengths.keys())
    
    # Validate cameras exist in calibration
    missing = [cam for cam in cameras if cam not in focal_lengths]
    if missing:
        print(f"Error: cameras not found in calibration: {missing}", file=sys.stderr)
        sys.exit(1)
    
    print("Focal lengths loaded:")
    for cam in cameras:
        print(f"  {cam}: fx={focal_lengths[cam]:.2f}px")
    
    # Device setup
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("Loading Depth Pro model...")
    t0 = time.time()
    model, transform = create_model_and_transforms(
        device=device,
        precision=torch.half
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # Ensure output directory exists
    os.makedirs(depth_dir, exist_ok=True)
    
    # Run inference
    results = {}
    for cam_name in cameras:
        img_path = os.path.join(frames_dir, f"{cam_name}.png")
        out_npz = os.path.join(depth_dir, f"{cam_name}.npz")
        out_png = os.path.join(depth_dir, f"{cam_name}_depthpro.png")
        
        # Skip if already processed
        if os.path.exists(out_npz):
            print(f"\n[SKIP] {cam_name}: depth already exists at {out_npz}")
            continue
        
        if not os.path.exists(img_path):
            print(f"\n[SKIP] {cam_name}: image not found at {img_path}")
            continue
        
        f_px = focal_lengths[cam_name]
        print(f"\n[{cam_name}] f_px={f_px:.2f}  image={img_path}")
        
        t_start = time.time()
        image, _, _ = load_rgb(img_path)

        # L16 images are linear light (uint16 → uint8 high-byte = still linear).
        # Depth Pro was trained on sRGB-gamma-encoded images. Apply sRGB transfer function.
        image_f = image.astype(np.float32) / 255.0
        mask = image_f <= 0.0031308
        image_srgb = np.where(
            mask,
            12.92 * image_f,
            1.055 * (image_f ** (1.0 / 2.4)) - 0.055
        )
        image = np.clip(image_srgb * 255.0, 0, 255).astype(np.uint8)

        print(f"[depth-pro] {cam_name}: image shape={image.shape} dtype={image.dtype} p50={np.median(image):.0f} p99={np.percentile(image, 99):.0f}")

        with torch.no_grad():
            # CRITICAL: f_px must be a torch tensor, not a plain float
            f_px_tensor = torch.tensor(f_px, dtype=torch.float32, device=device)
            prediction = model.infer(transform(image), f_px=f_px_tensor)
        
        depth = prediction['depth'].cpu().numpy().astype(np.float32)  # metres, (H, W)
        
        # Save NPZ (float32, metres)
        np.savez_compressed(out_npz, depth=depth)
        
        # Save uint16 PNG in millimetres
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(out_png, depth_mm)
        
        elapsed = time.time() - t_start
        results[cam_name] = {
            'min': float(depth.min()),
            'max': float(depth.max()),
            'median': float(np.median(depth)),
            'time': elapsed,
        }
        print(f"  depth range: {depth.min():.2f}–{depth.max():.2f}m  median: {np.median(depth):.2f}m  ({elapsed:.1f}s)")
        print(f"  saved: {out_npz}")
        print(f"  saved: {out_png}")
    
    print("\n=== Summary ===")
    for cam, r in results.items():
        print(f"  {cam}: {r['min']:.2f}–{r['max']:.2f}m  median={r['median']:.2f}m  time={r['time']:.1f}s")
    
    if not results:
        print("  (no new depth maps processed)")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
