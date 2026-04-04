#!/usr/bin/env python3
"""
lri_lumen.py — Light L16 visual refocus + DNG export tool.

Mimics Light's Lumen: click the image to pick a focus point, adjust aperture
and focal-length sliders, apply computational bokeh, then export a 16-bit
linear DNG suitable for Lightroom / Capture One.

Requirements:
  pip3 install gradio tifffile opencv-python-headless numpy

Usage:
  python3 lri_lumen.py [frames_dir] [fused_dir] [--port PORT] [--share]

  frames_dir : dir containing A1.png (from lri_extract.py)
               default: /tmp/L16_00001_frames
  fused_dir  : dir containing fused_10cams_median_depth.png
               default: /tmp/L16_00001_fused
  --port     : local port (default 7860)
  --share    : create a public Gradio share link
"""

import os
import sys
import datetime
import argparse
import struct

import numpy as np
import cv2
import tifffile
import gradio as gr

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='L16 Lumen refocus tool')
    p.add_argument('frames_dir', nargs='?', default='/tmp/L16_00001_frames')
    p.add_argument('fused_dir',  nargs='?', default='/tmp/L16_00001_fused')
    p.add_argument('--port',  type=int,  default=7860)
    p.add_argument('--share', action='store_true')
    p.add_argument('--output-dir', default='/tmp/L16_lumen_output')
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(frames_dir: str, fused_dir: str):
    # Find best available reference frame.  A1 is ideal (wide, full-frame
    # coverage); fall back to B1, C1 for B/C-only files (2018-02 era).
    _CANDIDATES = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3']
    img_path = None
    for cand in _CANDIDATES:
        p = os.path.join(frames_dir, f'{cand}.png')
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(
            f'No reference frame found in {frames_dir}  '
            f'(tried {", ".join(_CANDIDATES)})')

    depth_path = os.path.join(fused_dir, 'fused_10cams_median_depth.png')

    # Load with IMREAD_UNCHANGED so 16-bit PNGs stay 16-bit.
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot load image: {img_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Normalise to float [0..1] linear ──────────────────────────────────────
    max_val = 65535.0 if img_rgb.dtype == np.uint16 else 255.0
    img_f = img_rgb.astype(np.float32) / max_val

    # ── Grey-world auto white-balance ─────────────────────────────────────────
    # L16 raw data has a strong green bias (G/R ≈ 1.6, G/B ≈ 1.3) inherited
    # from the sensor's Bayer CFA.  Without correction the initial display
    # looks greenish and WB sliders must be hand-tuned.  Applying grey-world
    # here gives a neutral starting point; the WB sliders allow fine-tuning.
    r_m, g_m, b_m = img_f[:, :, 0].mean(), img_f[:, :, 1].mean(), img_f[:, :, 2].mean()
    if r_m > 1e-5 and b_m > 1e-5 and g_m > 1e-5:
        img_f[:, :, 0] = np.clip(img_f[:, :, 0] * (g_m / r_m), 0, 1)
        img_f[:, :, 2] = np.clip(img_f[:, :, 2] * (g_m / b_m), 0, 1)

    # ── Gamma encode for display (linear → sRGB ≈ gamma 2.2) ─────────────────
    # The raw sensor data is linear-light.  Without gamma encoding the image
    # appears much darker than expected on a typical monitor.
    img_f = np.power(np.clip(img_f, 0, 1), 1.0 / 2.2)
    img_rgb = (img_f * 255.0).astype(np.uint8)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f'Cannot load depth: {depth_path}')
    depth = depth.astype(np.float32)

    return img_rgb, depth


# ── Depth map visualisation ───────────────────────────────────────────────────

def depth_colormap(depth: np.ndarray, alpha: float = 0.5, img_rgb: np.ndarray | None = None):
    """Return a jet-colourmap depth image, optionally blended with img_rgb."""
    valid = depth > 0
    dmin  = depth[valid].min() if valid.any() else 1.0
    dmax  = depth[valid].max() if valid.any() else 2.0

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - dmin) / (dmax - dmin + 1e-6)
    norm = (norm * 255).astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if img_rgb is not None:
        colored = (alpha * colored + (1 - alpha) * img_rgb).astype(np.uint8)
    return colored


# ── Circle of confusion ───────────────────────────────────────────────────────

def compute_coc(depth: np.ndarray,
                focus_mm: float,
                f_equiv_mm: float = 35.0,
                f_number: float = 2.0,
                img_width: int = 4160) -> np.ndarray:
    """
    Circle-of-confusion radius in pixels, using a 35mm-equivalent camera model.

    thin-lens far-field approx (valid when depth >> focal_length):
      coc_sensor_mm = f_equiv² / (N * focus_mm) * |focus_mm/depth − 1| * (focus_mm)
                    = f_equiv² / N * |1/focus_mm − 1/depth|
      coc_pixels    = coc_sensor_mm / pixel_size_mm
      pixel_size_mm = 36 mm / img_width   (36 mm = full-frame width)
    """
    sensor_width_mm = 36.0                       # full-frame equivalent
    pixel_size_mm   = sensor_width_mm / img_width

    med         = float(np.median(depth[depth > 0]))
    depth_safe  = np.where(depth > 0, depth, med).astype(np.float32)

    coc_sensor = (f_equiv_mm ** 2 / f_number) * np.abs(1.0 / focus_mm - 1.0 / depth_safe)
    coc_pixels = coc_sensor / pixel_size_mm

    return coc_pixels.astype(np.float32)


# ── Bokeh (focus stack blend) ─────────────────────────────────────────────────

def apply_bokeh(img_rgb: np.ndarray,
                depth: np.ndarray,
                focus_mm: float,
                f_number: float = 2.0,
                f_equiv_mm: float = 35.0,
                max_coc_px: float = 80.0,
                preview_scale: float = 1.0) -> np.ndarray:
    """
    Depth-of-field blur using focus-stack blending.

    Generates Gaussian-blurred layers at doubling sigmas, then per-pixel
    interpolates between adjacent layers based on each pixel's CoC radius.
    Fast: O(K × H × W) instead of naïve O(H × W × max_radius²).
    """
    if preview_scale < 1.0:
        H0, W0 = img_rgb.shape[:2]
        W_p = int(W0 * preview_scale)
        H_p = int(H0 * preview_scale)
        img_work  = cv2.resize(img_rgb, (W_p, H_p), interpolation=cv2.INTER_AREA)
        depth_work = cv2.resize(depth,  (W_p, H_p), interpolation=cv2.INTER_NEAREST)
    else:
        img_work   = img_rgb
        depth_work = depth

    H, W = img_work.shape[:2]
    coc = compute_coc(depth_work, focus_mm, f_equiv_mm, f_number, img_rgb.shape[1])
    coc = np.clip(coc, 0.0, max_coc_px)

    img_f = img_work.astype(np.float32)

    # Build sigma ladder: 0, 1, 2, 4, 8, 16, 32, 64 … up to needed max
    sigma_max = max(1.0, float(coc.max()) / 2.0)
    sigmas    = [0.0]
    s = 1.0
    while s <= sigma_max * 1.01:
        sigmas.append(s)
        s *= 2.0
    sigmas = np.array(sigmas, dtype=np.float32)

    # Pre-blur all layers
    layers = [img_f]
    for s in sigmas[1:]:
        k = int(2 * np.ceil(3 * s) + 1) | 1  # ensure odd kernel size
        layers.append(cv2.GaussianBlur(img_f, (k, k), float(s)))

    # CoC → effective sigma (CoC diameter ≈ 2 × sigma)
    sigma_map = (coc / 2.0).clip(0, sigmas[-1])

    result = np.zeros_like(img_f)
    for i in range(len(sigmas) - 1):
        s0, s1 = sigmas[i], sigmas[i + 1]
        mask = (sigma_map >= s0) & (sigma_map < s1)
        if not mask.any():
            continue
        t = ((sigma_map - s0) / (s1 - s0))[mask, np.newaxis]
        result[mask] = (1 - t) * layers[i][mask] + t * layers[i + 1][mask]

    beyond = sigma_map >= sigmas[-1]
    if beyond.any():
        result[beyond] = layers[-1][beyond]

    # Zero-depth pixels have no valid depth data — pass through original, never black
    no_depth = (depth_work == 0)
    if no_depth.any():
        result[no_depth] = img_f[no_depth]

    out = np.clip(result, 0, 255).astype(np.uint8)

    if preview_scale < 1.0:
        out = cv2.resize(out, (W0, H0), interpolation=cv2.INTER_LINEAR)

    return out


# ── DNG / TIFF export ─────────────────────────────────────────────────────────

# ── Post-processing (tone/colour adjustments) ─────────────────────────────────

def apply_adjustments(img_rgb: np.ndarray,
                      exposure: float = 0.0,
                      contrast: float = 1.0,
                      highlights: float = 0.0,
                      shadows: float = 0.0,
                      wb_r: float = 1.0,
                      wb_b: float = 1.0,
                      saturation: float = 1.0,
                      sharpness: float = 0.0) -> np.ndarray:
    """
    Apply Lumen-style tone and colour adjustments to an RGB uint8 image.

    exposure   : stops, e.g. +1 = 2× brighter, -1 = 0.5× (default 0)
    contrast   : linear multiplier around mid-grey (default 1.0 = no change)
    highlights : pull highlights down in [−1, 0] (default 0)
    shadows    : lift shadows in [0, 1] (default 0)
    wb_r/wb_b  : white-balance multipliers for R and B channels (default 1.0)
    saturation : 0 = greyscale, 1 = original, 2 = vivid (default 1.0)
    sharpness  : unsharp-mask amount 0–5 (default 0)
    """
    img = img_rgb.astype(np.float32) / 255.0

    # Exposure (in stops)
    if exposure != 0.0:
        img *= 2.0 ** exposure

    # White balance
    img[:, :, 0] *= wb_r
    img[:, :, 2] *= wb_b

    # Contrast (S-curve around 0.5)
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5

    # Highlights / shadows (gentle roll-off)
    if highlights != 0.0:
        # Pull bright areas: compress values above 0.5
        mask = img > 0.5
        img[mask] = img[mask] + highlights * img[mask] * (img[mask] - 0.5) * 2
    if shadows != 0.0:
        mask = img < 0.5
        img[mask] = img[mask] + shadows * (1 - img[mask]) * (0.5 - img[mask]) * 2

    img = np.clip(img, 0.0, 1.0)

    # Saturation (via HSV)
    if saturation != 1.0:
        img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Sharpness (unsharp mask)
    if sharpness > 0.0:
        img8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img8, (0, 0), 1.5)
        img = np.clip(img8.astype(np.float32) + sharpness * (img8.astype(np.float32) - blurred), 0, 255) / 255.0

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def export_dng(img_rgb: np.ndarray,
               output_path: str,
               focus_mm: float | None = None,
               f_equiv_mm: float | None = None,
               f_number: float | None = None) -> str:
    """
    Write a 16-bit linear DNG file (TIFF-based, Lightroom/Mylio/ACR compatible).

    Structure (mirrors camera-origin DNGs from Adobe DNG Converter):
      IFD0  — sRGB thumbnail (8-bit, NewSubfileType=1) + ALL DNG metadata tags
              SubIFDs tag → links to SubIFD below
      SubIFD— full-resolution 16-bit LinearRaw (NewSubfileType=0)
              DefaultCropOrigin/Size so readers know exact valid pixel area

    Putting metadata in IFD0 and full-res in SubIFD matches what Mylio,
    Lightroom, and ACR expect — they follow the camera DNG convention.

    img_rgb may be:
      - uint8  (H, W, 3) [0..255]  → scaled × 257 to fill 16-bit range
      - float32 (H, W, 3) [0..65535] → pre-gamma linear (from fused_image_16bit.png)
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if img_rgb.dtype == np.uint8:
        img16 = (img_rgb.astype(np.uint32) * 257).astype(np.uint16)
    else:
        img16 = np.clip(img_rgb, 0, 65535).astype(np.uint16)

    # Apply grey-world white balance to the export data.
    # The fused 16-bit PNG has native sensor colour bias (G/R ≈ 1.3); without
    # correction DNG apps display a strong green cast regardless of ColorMatrix.
    img_f = img16.astype(np.float32)
    r_mean = img_f[:, :, 0].mean()
    g_mean = img_f[:, :, 1].mean()
    b_mean = img_f[:, :, 2].mean()
    if r_mean > 0 and b_mean > 0 and g_mean > 0:
        img_f[:, :, 0] *= (g_mean / r_mean)
        img_f[:, :, 2] *= (g_mean / b_mean)
    img16 = np.clip(img_f, 0, 65535).astype(np.uint16)

    H, W = img16.shape[:2]
    now = datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S')
    desc_parts = ['Light L16 linear DNG - lri_lumen.py']
    if focus_mm   is not None: desc_parts.append(f'Focus {focus_mm:.0f} mm')
    if f_equiv_mm is not None: desc_parts.append(f'{f_equiv_mm:.0f} mm equiv')
    if f_number   is not None: desc_parts.append(f'f/{f_number:.1f}')
    description = ', '.join(desc_parts)

    # ── Thumbnail: 1/8 scale, gamma-corrected sRGB 8-bit ────────────────────
    # Gamma correction (linear → sRGB ≈ gamma 2.2) so thumbnail looks correct
    # in Mylio/Lightroom library view and as the DNG's embedded preview.
    thumb_lin = img16[::8, ::8].astype(np.float32) / 65535.0
    thumb_lin = np.clip(thumb_lin, 0.0, 1.0)
    thumb8 = np.power(thumb_lin, 1.0 / 2.2)
    thumb8 = np.clip(thumb8 * 255.0, 0, 255).astype(np.uint8)

    # ── IFD0 extra-tags (DNG metadata + thumbnail colour space) ─────────────
    # All DNG metadata MUST live in IFD0.  tifffile handles tags 254/282/283/296
    # via its own params (subfiletype, resolution, resolutionunit) — exclude here.
    # TIFF types: 1=BYTE 2=ASCII 3=SHORT 4=LONG 5=RATIONAL 7=UNDEF 10=SRATIONAL
    # RATIONAL/SRATIONAL value: flat int tuple (num, den, num, den, …).
    # Strictly ascending tag code order required.
    ifd0_tags = [
        (271,   2,   0,  'Light',                     True),  # Make
        (272,   2,   0,  'L16',                       True),  # Model
        (50706, 1,   4,  bytes([1, 4, 0, 0]),         True),  # DNGVersion = 1.4
        (50707, 1,   4,  bytes([1, 1, 0, 0]),         True),  # DNGBackwardVersion = 1.1
        (50708, 2,   0,  'Light L16',                 True),  # UniqueCameraModel
        (50714, 4,   1,  0,                           True),  # BlackLevel = 0
        (50717, 4,   1,  65535,                       True),  # WhiteLevel = 65535
        # ColorMatrix1: identity — data is already processed linear sRGB.
        # A real camera-to-XYZ matrix on pre-processed data would cause a
        # double-correction.  Identity passes colour through unchanged.
        (50721, 10,  9,  (1, 1,  0, 1,  0, 1,         # ColorMatrix1 = identity 3×3
                          0, 1,  1, 1,  0, 1,
                          0, 1,  0, 1,  1, 1),         True),
        (50728, 5,   3,  (1, 1,  1, 1,  1, 1),        True),  # AsShotNeutral = 1 × 3
        (50730, 10,  1,  (0, 1),                      True),  # BaselineExposure = 0
        (50731, 5,   1,  (1, 1),                      True),  # BaselineNoise = 1
        (50732, 5,   1,  (1, 1),                      True),  # BaselineSharpness = 1
        (50778, 3,   1,  21,                          True),  # CalibrationIlluminant1 = D65
        (50964, 10,  9,  (1, 1,  0, 1,  0, 1,         # ForwardMatrix1 = identity 3×3
                          0, 1,  1, 1,  0, 1,
                          0, 1,  0, 1,  1, 1),         True),
        (50970, 3,   1,  2,                           True),  # PreviewColorSpace = sRGB
    ]

    # ── SubIFD extra-tags (full-res image crop boundaries) ───────────────────
    subifd_tags = [
        (50719, 4,   2,  (0, 0),                      True),  # DefaultCropOrigin
        (50720, 4,   2,  (W, H),                      True),  # DefaultCropSize
    ]

    with tifffile.TiffWriter(output_path, bigtiff=False) as tw_obj:
        # IFD0 — thumbnail (8-bit sRGB) + all DNG metadata.
        # subifds=1 creates a SubIFDs tag in IFD0 pointing to the full-res SubIFD.
        # This matches the canonical camera DNG layout that Mylio/Lightroom expect.
        tw_obj.write(
            thumb8,
            photometric='rgb',
            compression=None,
            subfiletype=1,           # reduced-resolution preview
            subifds=1,               # next write() → SubIFD (full-res raw data)
            resolution=(72, 72, 'inch'),
            software='lri_lumen.py',
            datetime=now,
            description=description,
            extratags=ifd0_tags,
            metadata=None,
        )

        # SubIFD — full-resolution 16-bit LinearRaw.
        # Linked via SubIFDs tag in IFD0.  Readers that understand DNG will find
        # the full-res image here; legacy TIFF readers see the thumbnail in IFD0.
        tw_obj.write(
            img16,
            photometric=34892,       # LinearRaw (DNG processed RGB)
            compression=None,
            subfiletype=0,           # full-resolution primary image
            extratags=subifd_tags,
            metadata=None,
        )

    return output_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(img_rgb: np.ndarray,
             depth: np.ndarray,
             output_dir: str) -> gr.Blocks:

    os.makedirs(output_dir, exist_ok=True)

    # Precompute defaults
    valid_depths = depth[depth > 0]
    depth_median  = float(np.median(valid_depths))
    depth_min_mm  = float(valid_depths.min())
    depth_max_mm  = float(valid_depths.max())

    # Small preview for fast interaction (1/4 res)
    H0, W0 = img_rgb.shape[:2]
    PREV_W, PREV_H = W0 // 4, H0 // 4
    img_preview = cv2.resize(img_rgb, (PREV_W, PREV_H), interpolation=cv2.INTER_AREA)

    # ── Layout ────────────────────────────────────────────────────────────────
    css = '''
    .focus-crosshair { cursor: crosshair !important; }
    #focus-info { font-size: 1.1em; padding: 6px 0; }
    '''

    with gr.Blocks(title='L16 Lumen') as demo:
        gr.Markdown('# Light L16 Lumen - Computational Refocus')
        gr.Markdown(
            '**Click the image** to pick a focus point. '
            'Adjust aperture and focal length, then **Apply Bokeh**. '
            'Export a 16-bit DNG or JPEG when satisfied.'
        )

        with gr.Row():
            # ── Left: image viewer ─────────────────────────────────────────
            with gr.Column(scale=3):
                img_display = gr.Image(
                    value=img_preview,
                    label='Click to focus',
                    type='numpy',
                    elem_classes=['focus-crosshair'],
                    height=520,
                    interactive=True,
                )
                focus_info = gr.Markdown(
                    f'**Focus:** {depth_median:.0f} mm ({depth_median/1000:.2f} m)  '
                    f'— click image to change',
                    elem_id='focus-info',
                )

            # ── Right: controls ────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown('### Lens Settings')

                f_equiv_slider = gr.Slider(
                    minimum=24, maximum=200, value=35, step=1,
                    label='Focal length (35mm equiv)',
                    info='Longer = shallower depth of field',
                )
                f_number_slider = gr.Slider(
                    minimum=0.7, maximum=22.0, value=1.8, step=0.1,
                    label='f-number (aperture)',
                    info='Smaller = more background blur',
                )
                focus_slider = gr.Slider(
                    minimum=max(200, int(depth_min_mm * 0.8)),
                    maximum=int(depth_max_mm * 1.2),
                    value=int(depth_median),
                    step=10,
                    label='Focus distance (mm)',
                    info='Or click the image above',
                )

                gr.Markdown('### View')
                view_mode = gr.Radio(
                    choices=['Photo', 'Depth map', 'Depth overlay'],
                    value='Photo',
                    label='Display mode',
                )
                preview_quality = gr.Radio(
                    choices=['Preview (fast 1/4 res)', 'Full resolution'],
                    value='Preview (fast 1/4 res)',
                    label='Bokeh quality',
                )

                with gr.Accordion('Tone & Colour', open=False):
                    exposure_slider   = gr.Slider(-3.0, 3.0,   0.0, step=0.1, label='Exposure (stops)')
                    contrast_slider   = gr.Slider(0.5,  2.5,   1.0, step=0.05, label='Contrast')
                    highlights_slider = gr.Slider(-1.0, 0.0,   0.0, step=0.05, label='Highlights')
                    shadows_slider    = gr.Slider(0.0,  1.0,   0.0, step=0.05, label='Shadows')
                    with gr.Row():
                        wb_r_slider = gr.Slider(0.5, 2.0, 1.0, step=0.05, label='WB Red')
                        wb_b_slider = gr.Slider(0.5, 2.0, 1.0, step=0.05, label='WB Blue')
                    saturation_slider = gr.Slider(0.0, 3.0, 1.0, step=0.05, label='Saturation')
                    sharpness_slider  = gr.Slider(0.0, 5.0, 0.0, step=0.1, label='Sharpness')

                gr.Markdown('---')

                with gr.Row():
                    bokeh_btn = gr.Button('Apply Bokeh', variant='primary', scale=2)
                    reset_btn = gr.Button('Reset',       variant='secondary', scale=1)

                gr.Markdown('---')
                gr.Markdown('### Export')
                with gr.Row():
                    export_dng_btn  = gr.Button('Export DNG',  variant='secondary', scale=1)
                    export_jpg_btn  = gr.Button('Export JPEG', variant='secondary', scale=1)
                export_file = gr.File(label='Download', interactive=False)
                export_msg  = gr.Textbox(label='', interactive=False, max_lines=2)

        # ── State ──────────────────────────────────────────────────────────────
        state_focus_mm    = gr.State(value=depth_median)
        state_result_img  = gr.State(value=img_preview)
        state_bokeh_img   = gr.State(value=img_preview)   # bokeh-applied, pre-tone
        state_f_equiv     = gr.State(value=35.0)
        state_f_number    = gr.State(value=1.8)
        state_is_preview  = gr.State(value=True)

        # ── Helpers ───────────────────────────────────────────────────────────
        def _depth_at(xi: int, yi: int, scale: float = 0.25) -> float:
            """Return depth in mm at full-res pixel (xi, yi)."""
            # img_display shows preview at 1/4 res → coords already in display space
            xi_full = min(int(xi / scale), W0 - 1)
            yi_full = min(int(yi / scale), H0 - 1)
            d = depth[yi_full, xi_full]
            if d == 0:
                r = 30
                patch = depth[max(0, yi_full - r): yi_full + r,
                              max(0, xi_full - r): xi_full + r]
                valid = patch[patch > 0]
                d = float(np.median(valid)) if len(valid) else depth_median
            return float(d)

        def _render_view(mode: str, img_np: np.ndarray) -> np.ndarray:
            if mode == 'Depth map':
                return cv2.resize(
                    depth_colormap(depth),
                    (PREV_W, PREV_H), interpolation=cv2.INTER_AREA,
                )
            if mode == 'Depth overlay':
                return cv2.resize(
                    depth_colormap(depth, alpha=0.5, img_rgb=img_rgb),
                    (PREV_W, PREV_H), interpolation=cv2.INTER_AREA,
                )
            return img_np

        # ── Events ────────────────────────────────────────────────────────────

        def on_click(evt: gr.SelectData, view, result_img):
            """User clicked image -> set focus at that depth.
            evt.index is in img_preview pixel space (1/4 of full res).
            _depth_at default scale=0.25 converts to full-res coords.
            """
            x, y = int(evt.index[0]), int(evt.index[1])
            d = _depth_at(x, y)  # scale=0.25 default -> full-res lookup
            xi_full, yi_full = min(x * 4, W0 - 1), min(y * 4, H0 - 1)
            info = (
                f'**Focus:** {d:.0f} mm ({d/1000:.2f} m)  '
                f'(full-res pixel {xi_full}, {yi_full})'
            )
            return d, info, d

        img_display.select(
            on_click,
            inputs=[view_mode, state_result_img],
            outputs=[state_focus_mm, focus_info, focus_slider],
        )

        def on_focus_slider(v):
            return v, f'**Focus:** {v:.0f} mm ({v/1000:.2f} m)  — slider'

        focus_slider.change(
            on_focus_slider,
            inputs=[focus_slider],
            outputs=[state_focus_mm, focus_info],
        )

        def on_view_change(mode, result_img):
            return _render_view(mode, result_img)

        view_mode.change(
            on_view_change,
            inputs=[view_mode, state_result_img],
            outputs=[img_display],
        )

        _tone_inputs = [
            exposure_slider, contrast_slider, highlights_slider, shadows_slider,
            wb_r_slider, wb_b_slider, saturation_slider, sharpness_slider,
        ]

        def _render_bokeh(focus_mm, f_equiv, f_number, scale=1.0):
            """Apply bokeh only (no tone). Returns uint8 RGB."""
            return apply_bokeh(
                img_rgb, depth,
                focus_mm=float(focus_mm),
                f_number=float(f_number),
                f_equiv_mm=float(f_equiv),
                preview_scale=scale,
            )

        def _render_full(focus_mm, f_equiv, f_number,
                         exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp,
                         scale=1.0):
            """Render bokeh + tone at requested scale, returned as full-res uint8."""
            bokeh = _render_bokeh(focus_mm, f_equiv, f_number, scale=scale)
            return apply_adjustments(bokeh,
                exposure=float(exp), contrast=float(cont),
                highlights=float(hilite), shadows=float(shadow),
                wb_r=float(wb_r), wb_b=float(wb_b),
                saturation=float(sat), sharpness=float(sharp))

        def on_apply_bokeh(focus_mm, f_equiv, f_number, quality, view,
                           exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp):
            is_prev = 'fast' in quality
            scale = 0.25 if is_prev else 1.0
            gr.Info(f'Rendering (focus {float(focus_mm):.0f} mm, f/{float(f_number):.1f})...')
            bokeh = _render_bokeh(focus_mm, f_equiv, f_number, scale=scale)
            result = apply_adjustments(bokeh,
                exposure=float(exp), contrast=float(cont),
                highlights=float(hilite), shadows=float(shadow),
                wb_r=float(wb_r), wb_b=float(wb_b),
                saturation=float(sat), sharpness=float(sharp))
            result_prev = cv2.resize(result, (PREV_W, PREV_H), interpolation=cv2.INTER_AREA)
            bokeh_prev  = cv2.resize(bokeh,  (PREV_W, PREV_H), interpolation=cv2.INTER_AREA)
            display = _render_view(view, result_prev)
            return display, result_prev, bokeh_prev, float(f_equiv), float(f_number), is_prev

        bokeh_btn.click(
            on_apply_bokeh,
            inputs=[state_focus_mm, f_equiv_slider, f_number_slider,
                    preview_quality, view_mode] + _tone_inputs,
            outputs=[img_display, state_result_img, state_bokeh_img,
                     state_f_equiv, state_f_number, state_is_preview],
        )

        def on_tone_change(bokeh_img, view,
                           exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp):
            """Live tone slider update — re-applies tone to stored bokeh image."""
            result = apply_adjustments(bokeh_img,
                exposure=float(exp), contrast=float(cont),
                highlights=float(hilite), shadows=float(shadow),
                wb_r=float(wb_r), wb_b=float(wb_b),
                saturation=float(sat), sharpness=float(sharp))
            return _render_view(view, result), result

        for _s in _tone_inputs:
            _s.change(
                on_tone_change,
                inputs=[state_bokeh_img, view_mode] + _tone_inputs,
                outputs=[img_display, state_result_img],
            )

        def on_reset(view):
            info = (
                f'**Focus:** {depth_median:.0f} mm ({depth_median/1000:.2f} m)  - reset'
            )
            display = _render_view(view, img_preview)
            return display, img_preview, depth_median, info

        reset_btn.click(
            on_reset,
            inputs=[view_mode],
            outputs=[img_display, state_result_img, state_focus_mm, focus_info],
        )

        def _do_export_dng(focus_mm, f_equiv, f_number,
                           exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp):
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(output_dir, f'L16_lumen_{ts}.dng')
            gr.Info('Rendering full-resolution for DNG export...')
            full_res = _render_full(focus_mm, f_equiv, f_number,
                                    exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp,
                                    scale=1.0)
            export_dng(full_res, out_path,
                       focus_mm=float(focus_mm),
                       f_equiv_mm=float(f_equiv),
                       f_number=float(f_number))
            size_mb = os.path.getsize(out_path) / 1e6
            return out_path, f'DNG saved: {out_path}  ({size_mb:.1f} MB, 16-bit)'

        def _do_export_jpg(focus_mm, f_equiv, f_number,
                           exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp):
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(output_dir, f'L16_lumen_{ts}.jpg')
            gr.Info('Rendering full-resolution for JPEG export...')
            full_res = _render_full(focus_mm, f_equiv, f_number,
                                    exp, cont, hilite, shadow, wb_r, wb_b, sat, sharp,
                                    scale=1.0)
            cv2.imwrite(out_path,
                        cv2.cvtColor(full_res, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 97])
            size_mb = os.path.getsize(out_path) / 1e6
            return out_path, f'JPEG saved: {out_path}  ({size_mb:.1f} MB, quality 97)'

        _export_inputs = [state_focus_mm, state_f_equiv, state_f_number] + _tone_inputs

        export_dng_btn.click(
            _do_export_dng,
            inputs=_export_inputs,
            outputs=[export_file, export_msg],
        )
        export_jpg_btn.click(
            _do_export_jpg,
            inputs=_export_inputs,
            outputs=[export_file, export_msg],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f'Loading image: {args.frames_dir}/A1.png')
    print(f'Loading depth: {args.fused_dir}/fused_10cams_median_depth.png')

    img_rgb, depth = load_data(args.frames_dir, args.fused_dir)

    valid = depth[depth > 0]
    print(f'Image : {img_rgb.shape[1]}×{img_rgb.shape[0]} px')
    print(f'Depth : {depth.shape[1]}×{depth.shape[0]} px, '
          f'{valid.size:,} valid pixels, '
          f'range {valid.min():.0f}–{valid.max():.0f} mm, '
          f'median {np.median(valid):.0f} mm')

    demo = build_ui(img_rgb, depth, args.output_dir)

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )


if __name__ == '__main__':
    main()
