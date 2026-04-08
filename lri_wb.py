"""
lri_wb.py — Per-camera white balance and exposure normalization for the L16 fusion pipeline.

Applies simple per-channel scale factors to normalise each camera's image to a
reference camera.  No cross-channel mixing — just three independent multipliers.

Two adjustments are made in sequence:

  1. White Balance (WB)
     L16 calibration.json does not carry AsShotNeutral or any factory WB gains,
     so we fall back to a gray-world estimate computed per-camera from the source
     image.  The resulting gains are relative to the gray-world grey point; when
     applied to the src image the mean of each channel is equalized.  We then
     express the gains *relative to the reference camera* so that the reference
     is unchanged and all other cameras are shifted toward it.

  2. Exposure / Gain Normalization
     Each camera records analog_gain (ISP multiplier) and exposure_ns (integration
     time in nanoseconds) in calibration.json.  Their product is the effective
     photon count per DN, so the EV offset between cameras is:

         ev_offset = log2( (exp_ref * gain_ref) / (exp_src * gain_src) )

     A positive ev_offset means src is darker than ref; we multiply src by
     2**ev_offset to bring it to the same brightness.

     When analog_gain / exposure_ns are both absent (or identical across all
     cameras as in a locked-exposure shot), the EV scale factor is 1.0 — a no-op.

Public API
----------
gray_world_gains(img_uint16)
    -> (r_gain, g_gain, b_gain)  float64 per-channel gains for gray-world balance.

apply_wb_exposure(src_img, src_meta, ref_meta)
    -> np.ndarray  float32 (H, W, 3) with WB and exposure normalization applied.
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Gray-world white balance
# ---------------------------------------------------------------------------

def gray_world_gains(img: np.ndarray) -> tuple[float, float, float]:
    """
    Compute per-channel gains to make each channel's mean equal to the overall
    image mean (gray-world assumption).

    Parameters
    ----------
    img : np.ndarray
        uint16 or float32 (H, W, 3) linear image.  Channel order: R, G, B.

    Returns
    -------
    (r_gain, g_gain, b_gain) : float64
        Per-channel multiplicative gains.  Apply to the image as:
            corrected = img * [r_gain, g_gain, b_gain]
    """
    img_f = img.astype(np.float64)
    # Use pixels that are not clipped and not black to avoid skewing the mean
    luminance = img_f.mean(axis=2)
    valid = (luminance > 256) & (luminance < 60000)   # 16-bit image
    if valid.sum() < 1000:
        # Fallback: use all pixels
        valid = np.ones(luminance.shape, dtype=bool)

    means = np.array([
        img_f[:, :, 0][valid].mean(),
        img_f[:, :, 1][valid].mean(),
        img_f[:, :, 2][valid].mean(),
    ])
    # Avoid division by zero
    means = np.maximum(means, 1.0)
    overall = means.mean()
    gains = overall / means   # channel gain to bring each mean to `overall`
    return float(gains[0]), float(gains[1]), float(gains[2])


# ---------------------------------------------------------------------------
# EV normalization
# ---------------------------------------------------------------------------

def exposure_ev_scale(src_meta: dict, ref_meta: dict) -> float:
    """
    Compute the linear brightness scale factor to apply to src so it matches
    ref in terms of photon exposure.

    Uses analog_gain * exposure_ns as a proxy for effective exposure.

    Parameters
    ----------
    src_meta : dict
        Camera metadata dict for the source camera.  Expected keys:
        ``analog_gain`` (float), ``exposure_ns`` (int).
    ref_meta : dict
        Same for the reference camera.

    Returns
    -------
    float
        Linear scale factor to multiply into src.  Returns 1.0 when either
        camera is missing exposure data or the ratio is out of a sane range.
    """
    src_gain = src_meta.get('analog_gain')
    src_exp  = src_meta.get('exposure_ns')
    ref_gain = ref_meta.get('analog_gain')
    ref_exp  = ref_meta.get('exposure_ns')

    if None in (src_gain, src_exp, ref_gain, ref_exp):
        return 1.0
    if src_gain <= 0 or src_exp <= 0:
        return 1.0

    src_effective = float(src_gain) * float(src_exp)
    ref_effective = float(ref_gain) * float(ref_exp)

    if src_effective <= 0:
        return 1.0

    ratio = ref_effective / src_effective   # >1 means src is underexposed → brighten
    ev_offset = math.log2(ratio)

    # Sanity clamp: allow at most ±6 EV (64×) correction.
    # Larger differences are almost certainly metadata errors.
    ev_offset = max(-6.0, min(6.0, ev_offset))
    return float(2.0 ** ev_offset)


# ---------------------------------------------------------------------------
# Combined WB + exposure normalization
# ---------------------------------------------------------------------------

def apply_wb_exposure(
    src_img: np.ndarray,
    src_meta: dict,
    ref_meta: dict,
    ref_img: np.ndarray | None = None,
    use_gray_world: bool = True,
) -> np.ndarray:
    """
    Apply white balance and exposure normalization to ``src_img`` relative to
    the reference camera.

    Normalization is purely per-channel multiplicative — no cross-channel
    mixing.  The three scale factors applied to [R, G, B] are:

        scale_c = wb_relative_c * ev_scale

    where:
      * ``ev_scale`` corrects for the exposure / analog-gain difference between
        src and ref, derived from ``exposure_ns`` and ``analog_gain`` fields in
        the metadata dicts.
      * ``wb_relative_c`` aligns gray-world WB gains of src to those of ref,
        computed on *exposure-normalised* images so that the EV difference does
        not contaminate the colour comparison.  When ``ref_img`` is not
        provided, gray-world gains of the reference are assumed to be (1,1,1).
        Disabled (wb_rel=1.0 per channel) when ``use_gray_world=False``.

    Parameters
    ----------
    src_img : np.ndarray
        Source camera image, uint16 or float32 (H, W, 3), linear, R-G-B order.
    src_meta : dict
        Metadata for the source camera.  Required keys: ``analog_gain``,
        ``exposure_ns``.
    ref_meta : dict
        Metadata for the reference camera.  Required keys: ``analog_gain``,
        ``exposure_ns``.
    ref_img : np.ndarray or None
        Reference camera image (uint16 or float32, H, W, 3), used for
        gray-world relative WB.  When None the reference WB gains are assumed
        to be (1, 1, 1) — only the source's cast is removed.
    use_gray_world : bool
        When True (default) apply relative gray-world WB on top of EV scale.
        Set to False to apply only EV scale — useful for cameras with identical
        sensor type and overlapping FOV (e.g. A-cameras among themselves) where
        scene-content variation would otherwise corrupt the gray-world estimate.

    Returns
    -------
    np.ndarray
        float32 (H, W, 3) normalized image, clipped to [0, 65535].
    """
    src_f = src_img.astype(np.float32)

    # --- 1. EV scale (exposure + analog gain) ----------------------------
    ev = exposure_ev_scale(src_meta, ref_meta)

    if not use_gray_world:
        # EV-only mode: just scale brightness, preserve colour ratio.
        result = src_f * ev
        return np.clip(result, 0.0, 65535.0).astype(np.float32)

    # --- 2. Gray-world WB gains (computed on EV-normalised images) -------
    # Normalise both images by their EV first so that exposure differences
    # don't contaminate the colour comparison.  E.g. an A-camera with 2×
    # exposure would otherwise appear "greener" than the reference and receive
    # a spurious green-suppression gain.
    src_ev_normed = (src_f * ev).clip(0, 65535)
    src_rg, src_gg, src_bg = gray_world_gains(src_ev_normed)

    if ref_img is not None:
        ref_rg, ref_gg, ref_bg = gray_world_gains(ref_img.astype(np.float32))
    else:
        ref_rg, ref_gg, ref_bg = 1.0, 1.0, 1.0

    # Relative WB: what gains bring src's colour cast (post-EV) to match ref.
    # Sanity clamp: no more than 4× per channel.
    wb_r = float(np.clip(src_rg / max(ref_rg, 1e-6), 0.25, 4.0))
    wb_g = float(np.clip(src_gg / max(ref_gg, 1e-6), 0.25, 4.0))
    wb_b = float(np.clip(src_bg / max(ref_bg, 1e-6), 0.25, 4.0))

    # --- 3. Combine and apply -------------------------------------------
    # EV already applied above; wb brings colour balance in line with ref.
    scale = np.array([wb_r, wb_g, wb_b], dtype=np.float32)
    result = src_ev_normed * scale[None, None, :]
    return np.clip(result, 0.0, 65535.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def wb_exposure_summary(cameras_meta: dict, ref_name: str) -> None:
    """
    Print a table of per-camera WB and exposure scale factors relative to
    ``ref_name``.  Useful for verifying calibration.json exposure data.

    cameras_meta : dict
        Dict keyed by camera name; values have ``analog_gain`` and
        ``exposure_ns`` fields (as stored in the cameras dict after
        load_cameras() is called with include_exposure=True).
    ref_name : str
        Name of the reference camera (e.g. 'A1').
    """
    ref_meta = cameras_meta.get(ref_name, {})
    print(f"{'Camera':<8} {'analog_gain':>12} {'exposure_ns':>14} {'ev_offset':>10}")
    print("-" * 50)
    for name, meta in sorted(cameras_meta.items()):
        ag  = meta.get('analog_gain', float('nan'))
        exp = meta.get('exposure_ns', float('nan'))
        ev  = exposure_ev_scale(meta, ref_meta)
        ev_db = math.log2(ev) if ev > 0 else float('nan')
        print(f"{name:<8} {ag:>12.3f} {exp:>14} {ev_db:>+10.3f} EV  (scale={ev:.4f})")
