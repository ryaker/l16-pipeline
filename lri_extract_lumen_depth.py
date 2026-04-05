#!/usr/bin/env python3
"""
lri_extract_lumen_depth.py — Extract Lumen's depth map from an LRI file.

Uses lri_process (compiled CIAPI binary) to render a DNG with embedded
GDepth XMP, then decodes the RangeInverse depth map.

Usage:
  python lri_extract_lumen_depth.py <lri_file> [--output-dir DIR] [--lris PATH]

Output (in output_dir):
  lumen_depth.npz        — float32 depth in metres, shape (H, W)
  lumen_depth.png        — 16-bit grayscale visualization (uint16, 0=near, 65535=far)
  lumen_depth_colour.png — 8-bit viridis-style colormap visualization (uint8 RGB)
  metadata.json          — near/far/units, image shape, depth stats

Notes:
  - Format 4 (DPC/DNG) embeds depth even without a .lris state file.
  - If a .lris file is available alongside the .lri, it is passed automatically
    via --lris to load Lumen's refined depth state.
  - GDepth:Format is always "RangeInverse" with units "mm".
  - Decode: depth_mm = 1 / (1/Far + (1/Near - 1/Far) * pixel_normalized)
  - pixel_normalized = pixel_uint8 / 255.0
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# ── Decode ────────────────────────────────────────────────────────────────────

def decode_rangeinverse(pixel_arr: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    """
    Decode a GDepth RangeInverse depth image to depth in metres.

    Args:
        pixel_arr: float32 array of pixel values normalized to [0, 1].
                   pixel=1.0  → nearest depth (near_mm)
                   pixel=0.0  → farthest depth (far_mm)
        near_mm:   Near range in mm (from XMP GDepth:Near).
        far_mm:    Far range in mm (from XMP GDepth:Far).

    Returns:
        depth_m: float32 array, same shape as pixel_arr, depth in metres.
    """
    # Avoid division by zero for pure-black pixels (pixel=0 → exactly far)
    denom = 1.0 / far_mm + (1.0 / near_mm - 1.0 / far_mm) * pixel_arr
    # Clamp denominator to small positive to avoid inf
    denom = np.clip(denom, 1e-12, None)
    depth_mm = 1.0 / denom
    return (depth_mm / 1000.0).astype(np.float32)


# ── XMP parsing ───────────────────────────────────────────────────────────────

def extract_xmp_metadata(dng_path: str) -> dict:
    """
    Use exiftool to extract GDepth XMP fields from the DNG.

    Returns dict with keys: format, near_mm, far_mm, units, measure_type, manufacturer.
    """
    result = subprocess.run(
        ["exiftool", "-XMP:all", dng_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed: {result.stderr.strip()}")

    meta = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()
        if key == "format":
            meta["format"] = val
        elif key == "near":
            meta["near_mm"] = float(val)
        elif key == "far":
            meta["far_mm"] = float(val)
        elif key == "units":
            meta["units"] = val
        elif key == "measure_type":
            meta["measure_type"] = val
        elif key == "manufacturer":
            meta["manufacturer"] = val
        elif key == "mime":
            meta["mime"] = val

    required = ["format", "near_mm", "far_mm"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise RuntimeError(f"Missing required XMP fields: {missing}\n"
                           f"exiftool output:\n{result.stdout[:800]}")

    if meta.get("format") != "RangeInverse":
        raise RuntimeError(f"Unexpected GDepth:Format = {meta['format']!r} "
                           f"(only 'RangeInverse' is supported)")

    units = meta.get("units", "mm")
    if units.lower() != "mm":
        raise RuntimeError(f"Unexpected GDepth:Units = {units!r} (expected 'mm')")

    return meta


def extract_depth_image(dng_path: str) -> np.ndarray:
    """
    Extract the GDepth:Data JPEG blob via exiftool -b and decode to float32 [0,1].
    """
    result = subprocess.run(
        ["exiftool", "-b", "-XMP:DepthImage", dng_path],
        capture_output=True
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"exiftool failed to extract depth image blob.\n"
            f"stderr: {result.stderr.decode(errors='replace').strip()}"
        )

    blob = result.stdout
    # The blob should start with JFIF/JPEG magic (0xFF 0xD8)
    if blob[:2] != b"\xff\xd8":
        raise RuntimeError(
            f"Depth blob does not start with JPEG magic "
            f"(got {blob[:4].hex()!r}). "
            f"Blob size: {len(blob)} bytes."
        )

    # Save to temp file and open with PIL
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(blob)
        tmp_path = tf.name

    try:
        img = Image.open(tmp_path)
        arr = np.array(img)
    finally:
        os.unlink(tmp_path)

    if arr.ndim == 3:
        # RGB/RGBA → take first channel (should be uniform for greyscale depth)
        arr = arr[..., 0]

    # Normalize uint8 → float32 in [0, 1]
    return arr.astype(np.float32) / 255.0


# ── Visualization ─────────────────────────────────────────────────────────────

def depth_to_png16(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """
    Convert depth in metres to 16-bit grayscale uint16 (near=65535, far=0).
    Values outside [near_m, far_m] are clamped.
    """
    norm = np.clip((depth_m - near_m) / (far_m - near_m + 1e-12), 0.0, 1.0)
    # Invert so near = bright (high value)
    return ((1.0 - norm) * 65535).astype(np.uint16)


def depth_to_colour_png(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """
    Apply a simple viridis-style 8-bit RGB colormap for visualization.
    Uses matplotlib if available, falls back to a simple blue→red gradient.
    """
    norm = np.clip((depth_m - near_m) / (far_m - near_m + 1e-12), 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        rgba = cm.viridis(norm)
        return (rgba[..., :3] * 255).astype(np.uint8)
    except ImportError:
        # Simple blue → red gradient fallback
        r = (norm * 255).astype(np.uint8)
        g = np.zeros_like(r)
        b = ((1.0 - norm) * 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_lri_process(lri_path: str, dng_path: str, lris_path: str | None) -> None:
    """
    Run arch -x86_64 ./lri_process <lri> <dng> [--lris <lris>]
    from the Light_Work directory.
    """
    script_dir = Path(__file__).parent.resolve()
    binary = str(script_dir / "lri_process")

    if not Path(binary).exists():
        raise FileNotFoundError(f"lri_process binary not found at: {binary}")

    cmd = ["arch", "-x86_64", binary, lri_path, dng_path]
    if lris_path:
        cmd += ["--lris", lris_path]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)  # let stdout/stderr flow through
    if result.returncode != 0:
        raise RuntimeError(f"lri_process exited with code {result.returncode}")


def extract_lumen_depth(
    lri_path: str,
    output_dir: str,
    lris_path: str | None = None,
) -> dict:
    """
    Full pipeline: LRI → DNG → depth npz + PNG.

    Returns dict with metadata and file paths.
    """
    lri_path = str(Path(lri_path).resolve())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(lri_path).stem  # e.g. "L16_00177"

    # Auto-detect .lris alongside .lri if not specified
    if lris_path is None:
        candidate = Path(lri_path).with_suffix(".lris")
        if candidate.exists():
            lris_path = str(candidate)
            print(f"  LRIS auto-detected: {lris_path}")

    # ── Step 1: Render DNG ───────────────────────────────────────
    dng_path = str(output_dir / f"{stem}_lumen.dng")
    print(f"\n[1/4] Rendering DNG: {lri_path} → {dng_path}")
    run_lri_process(lri_path, dng_path, lris_path)

    if not Path(dng_path).exists():
        raise RuntimeError(f"DNG not produced: {dng_path}")
    print(f"  DNG size: {Path(dng_path).stat().st_size / 1024:.0f} KB")

    # ── Step 2: Extract XMP metadata ────────────────────────────
    print("\n[2/4] Extracting GDepth XMP metadata...")
    xmp = extract_xmp_metadata(dng_path)
    near_mm = xmp["near_mm"]
    far_mm = xmp["far_mm"]
    near_m = near_mm / 1000.0
    far_m = far_mm / 1000.0
    print(f"  Format : {xmp['format']}")
    print(f"  Near   : {near_mm:.1f} mm  ({near_m:.3f} m)")
    print(f"  Far    : {far_mm:.1f} mm  ({far_m:.3f} m)")
    print(f"  Units  : {xmp.get('units', '?')}")
    print(f"  Source : {xmp.get('manufacturer', '?')}")

    # ── Step 3: Extract and decode depth image ───────────────────
    print("\n[3/4] Extracting and decoding depth image...")
    pixel_norm = extract_depth_image(dng_path)
    depth_m = decode_rangeinverse(pixel_norm, near_mm, far_mm)
    H, W = depth_m.shape
    print(f"  Shape  : {depth_m.shape}  (H×W)")
    print(f"  Min    : {depth_m.min():.3f} m")
    print(f"  Max    : {depth_m.max():.3f} m")
    print(f"  Median : {np.median(depth_m):.3f} m")
    print(f"  Mean   : {depth_m.mean():.3f} m")
    print(f"  P5/P95 : {np.percentile(depth_m,5):.3f} m / {np.percentile(depth_m,95):.3f} m")

    # ── Step 4: Save outputs ──────────────────────────────────────
    print("\n[4/4] Saving outputs...")

    npz_path = str(output_dir / "lumen_depth.npz")
    np.savez_compressed(npz_path, depth_m=depth_m)
    print(f"  npz  → {npz_path}  ({Path(npz_path).stat().st_size/1024:.0f} KB)")

    png16_path = str(output_dir / "lumen_depth.png")
    arr16 = depth_to_png16(depth_m, near_m, far_m)
    # Save 16-bit PNG: PIL expects mode "I" (int32) for 16-bit PNG saves
    Image.fromarray(arr16.astype(np.int32), mode="I").save(png16_path)
    print(f"  png16→ {png16_path}")

    colour_path = str(output_dir / "lumen_depth_colour.png")
    arr_colour = depth_to_colour_png(depth_m, near_m, far_m)
    Image.fromarray(arr_colour, mode="RGB").save(colour_path)
    print(f"  colour→ {colour_path}")

    stats = {
        "min_m": float(depth_m.min()),
        "max_m": float(depth_m.max()),
        "mean_m": float(depth_m.mean()),
        "median_m": float(np.median(depth_m)),
        "p5_m": float(np.percentile(depth_m, 5)),
        "p95_m": float(np.percentile(depth_m, 95)),
        "std_m": float(depth_m.std()),
    }
    meta_out = {
        "lri_file": lri_path,
        "lris_file": lris_path,
        "dng_file": dng_path,
        "gdepth_format": xmp["format"],
        "near_mm": near_mm,
        "far_mm": far_mm,
        "near_m": near_m,
        "far_m": far_m,
        "units": xmp.get("units", "mm"),
        "manufacturer": xmp.get("manufacturer", ""),
        "image_height": H,
        "image_width": W,
        "depth_stats": stats,
        "outputs": {
            "npz": npz_path,
            "png16": png16_path,
            "colour_png": colour_path,
        },
    }

    meta_path = str(output_dir / "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  meta → {meta_path}")

    return meta_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Lumen's depth map from an LRI file via lri_process + GDepth XMP."
    )
    parser.add_argument("lri_file", help="Path to the .lri file")
    parser.add_argument(
        "--output-dir", "-o",
        default="depth",
        help="Output directory (default: ./depth)"
    )
    parser.add_argument(
        "--lris", default=None,
        help="Explicit path to .lris state file (auto-detected if not specified)"
    )
    args = parser.parse_args()

    if not Path(args.lri_file).exists():
        print(f"ERROR: LRI file not found: {args.lri_file}", file=sys.stderr)
        sys.exit(1)

    try:
        meta = extract_lumen_depth(
            lri_path=args.lri_file,
            output_dir=args.output_dir,
            lris_path=args.lris,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Done ===")
    stats = meta["depth_stats"]
    print(f"  Near / Far : {meta['near_m']:.3f} m / {meta['far_m']:.3f} m")
    print(f"  Shape      : {meta['image_height']} × {meta['image_width']}")
    print(f"  Depth range: {stats['min_m']:.3f} m – {stats['max_m']:.3f} m")
    print(f"  Median     : {stats['median_m']:.3f} m")
    print(f"  P5 / P95   : {stats['p5_m']:.3f} m / {stats['p95_m']:.3f} m")
    print(f"  Output dir : {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
