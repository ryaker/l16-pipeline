#!/usr/bin/env python3
"""
lri_fuse_v2.py — L16 Synthetic Canvas Fusion v2

Clean CLI entry point that wires together:
  lri_virtual_camera, lri_depth_loader, lri_canvas_blend, lri_fuse_image

Usage:
    python3 lri_fuse_v2.py <frames_dir> <calibration_json> [options]

See --help for full option list.
"""

import argparse
import os
import sys

import numpy as np

from lri_fuse_image import load_cameras
from lri_virtual_camera import VirtualCamera
from lri_depth_loader import load_depth_for_canvas, flat_plane_depth, depth_stats
from lri_canvas_blend import assemble_canvas, save_canvas


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def fuse_v2(frames_dir: str, cal_path: str, output_path: str, **opts) -> str:
    """
    Run the v2 synthetic-canvas fusion pipeline.

    Parameters
    ----------
    frames_dir : str
        Directory containing A1.png … B5.png.
    cal_path : str
        Factory calibration JSON path.
    output_path : str
        Destination PNG path.
    **opts : keyword arguments
        depth, no_ccm, flat_plane, tile_rows, remap_cache, no_cache, small

    Returns
    -------
    str
        Absolute path to the saved output PNG.
    """
    print("=== L16 Synthetic Canvas Fusion v2 ===")

    # 1. Load calibration
    cameras = load_cameras(cal_path)
    print(f"  Cameras: {list(cameras.keys())}")

    # 2. Build virtual camera
    vc = VirtualCamera(cameras)
    if opts.get('small'):
        vc.K[0, 0] /= 4
        vc.K[1, 1] /= 4
        vc.K[0, 2] /= 4
        vc.K[1, 2] /= 4
        vc.W //= 4
        vc.H //= 4
    print(f"  Canvas: {vc.W}×{vc.H} = {vc.W * vc.H / 1e6:.1f}MP")

    # 3. Load depth
    lumen_dir = os.path.dirname(frames_dir.rstrip('/\\'))

    if opts.get('flat_plane'):
        depth = flat_plane_depth(vc)
        print("  Depth: flat-plane (3.0m)")
    else:
        depth_override = opts.get('depth')
        if depth_override:
            ext = os.path.splitext(depth_override)[1].lower()
            if ext == '.npz':
                # float32 array in metres, stored under key 'depth'
                data = np.load(depth_override)
                depth = data['depth'].astype(np.float32)
                print(f"  Depth: loaded from .npz override ({depth_override})")
            elif ext == '.png':
                import cv2
                raw = cv2.imread(depth_override, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    print(f"  WARNING: could not load depth PNG {depth_override}, using flat-plane")
                    depth = flat_plane_depth(vc)
                else:
                    depth = raw.astype(np.float32) / 1000.0   # uint16 mm → float32 m
                    print(f"  Depth: loaded from .png override ({depth_override})")
            else:
                print(f"  WARNING: unrecognised depth file extension '{ext}', using flat-plane")
                depth = flat_plane_depth(vc)
        else:
            # Find the reference wide camera (closest to origin)
            wide_cams = {
                n: c for n, c in cameras.items()
                if c.get('mirror_type', 'NONE') == 'NONE'
            }
            ref_name = min(
                wide_cams,
                key=lambda n: (
                    np.linalg.norm(wide_cams[n]['t'])
                    if wide_cams[n].get('t') is not None
                    else float('inf')
                ),
            )
            depth = load_depth_for_canvas(lumen_dir, vc, ref_name, cameras[ref_name])

        if depth is None:
            print("  Depth: not found, using flat-plane fallback")
            depth = flat_plane_depth(vc)
        else:
            stats = depth_stats(depth)
            print(
                f"  Depth: {stats['min']:.1f}–{stats['max']:.1f}m, "
                f"valid={stats['valid_fraction'] * 100:.0f}%"
            )

    # 4. Remap cache dir
    remap_cache_dir = opts.get('remap_cache')
    if opts.get('no_cache'):
        remap_cache_dir = None

    # 5. Assemble canvas
    print("  Assembling canvas...")
    result = assemble_canvas(
        frames_dir,
        cameras,
        vc,
        depth,
        remap_cache_dir=remap_cache_dir,
        apply_ccm_flag=not opts.get('no_ccm', False),
        tile_rows=opts.get('tile_rows', 500),
        lumen_dir=lumen_dir,
    )

    # 6. Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_canvas(result, output_path)
    print(
        f"  Output: {output_path}  "
        f"({result.shape[1]}×{result.shape[0]}, max={result.max():.0f})"
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='L16 synthetic canvas fusion — v2 pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('frames_dir',        help='Directory containing A1.png … B5.png')
    parser.add_argument('calibration_json',  help='Factory calibration JSON path')
    parser.add_argument('--output',          default=None,
                        help='Output PNG path [default: <lumen_dir>/fused_v2.png]')
    parser.add_argument('--depth',           default=None,
                        help='Override depth map path (uint16 mm PNG or float32 .npz)')
    parser.add_argument('--no-ccm',          action='store_true',
                        help='Skip CCM colour-correction for mirror cameras')
    parser.add_argument('--flat-plane',      action='store_true',
                        help='Use flat-plane depth assumption (fast, no parallax correction)')
    parser.add_argument('--tile-rows',       type=int, default=500,
                        help='Process canvas in N-row tiles to reduce peak RAM')
    parser.add_argument('--remap-cache',     default=None,
                        help='Cache remap arrays here to speed up repeat runs')
    parser.add_argument('--no-cache',        action='store_true',
                        help='Ignore existing remap cache')
    parser.add_argument('--small',           action='store_true',
                        help='Use 1/4-scale canvas (quick test mode)')
    args = parser.parse_args()

    if args.output is None:
        lumen_dir = os.path.dirname(args.frames_dir.rstrip('/\\'))
        suffix = '_small' if args.small else ''
        args.output = os.path.join(lumen_dir, f'fused_v2{suffix}.png')

    fuse_v2(
        args.frames_dir,
        args.calibration_json,
        args.output,
        depth=args.depth,
        no_ccm=args.no_ccm,
        flat_plane=args.flat_plane,
        tile_rows=args.tile_rows,
        remap_cache=args.remap_cache,
        no_cache=args.no_cache,
        small=args.small,
    )


if __name__ == '__main__':
    main()
