"""
lri_depth_pro.py — Metric depth from a single L16 frame using Apple Depth Pro.

Produces a per-pixel depth map in the virtual canvas coordinate system,
usable as the init_depth parameter in merge_cameras_with_flow() or a custom merge function.
"""
import os
import sys
import numpy as np
import cv2
import torch

# Depth Pro lives at ml-depth-pro/src
_DEPTH_PRO_SRC = os.path.join(os.path.dirname(__file__), 'ml-depth-pro', 'src')
if _DEPTH_PRO_SRC not in sys.path:
    sys.path.insert(0, _DEPTH_PRO_SRC)

from depth_pro import create_model_and_transforms, load_rgb
from depth_pro.depth_pro import DepthProConfig


_model = None
_transform = None
_device = None


def _load_model(device: str = 'mps'):
    """Load Depth Pro model (cached after first call)."""
    global _model, _transform, _device
    if _model is None:
        print('[depth_pro] Loading model...')
        try:
            # Try to use the requested device
            torch_device = torch.device(device)
        except Exception:
            print(f'[depth_pro] Device {device} not available, falling back to cpu')
            torch_device = torch.device('cpu')
        
        # Create config with absolute path to checkpoint
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            'ml-depth-pro', 'checkpoints', 'depth_pro.pt'
        )
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=checkpoint_path,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        
        _model, _transform = create_model_and_transforms(
            config=config,
            device=torch_device
        )
        _model.eval()
        _device = torch_device
        print(f'[depth_pro] Model loaded on {torch_device}')
    return _model, _transform, _device


def estimate_depth(
    frame_path: str,
    canvas_hw: tuple[int, int],
    focal_px: float | None = None,
    device: str = 'mps',
) -> np.ndarray:
    """
    Run Depth Pro on a single frame and return a depth map at canvas resolution.

    Parameters
    ----------
    frame_path : str
        Path to input frame (PNG or other OpenCV-readable format).
    canvas_hw : tuple[int, int]
        Target canvas shape (H, W) for depth map.
    focal_px : float, optional
        Focal length in pixels. If provided, used to rescale depth (advanced).
    device : str
        'mps' (default, Apple Silicon) or 'cpu'.

    Returns
    -------
    np.ndarray
        (H_canvas, W_canvas) float32 depth map in metres.
    """
    model, transform, actual_device = _load_model(device)
    
    print(f'[depth_pro] Loading frame: {frame_path}')
    
    # Try load_rgb from Depth Pro first (returns tuple: image, metadata, icc_profile)
    try:
        rgb_img, _, _ = load_rgb(frame_path)
    except Exception as e:
        print(f'[depth_pro] load_rgb failed ({e}), falling back to cv2.imread')
        bgr = cv2.imread(frame_path)
        if bgr is None:
            raise RuntimeError(f"Failed to load {frame_path}")
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] if needed
    if rgb_img.dtype == np.uint8:
        rgb_img = rgb_img.astype(np.float32) / 255.0
    elif rgb_img.dtype == np.uint16:
        rgb_img = rgb_img.astype(np.float32) / 65535.0
    
    print(f'[depth_pro] Input shape: {rgb_img.shape}')
    
    # Apply transform and run inference
    with torch.no_grad():
        img_tensor = transform(rgb_img)
        img_tensor = img_tensor.unsqueeze(0).to(actual_device)
        
        print(f'[depth_pro] Running inference...')
        # f_px can be a float, tensor, or None
        f_px_tensor = None
        if focal_px is not None:
            f_px_tensor = torch.tensor(focal_px, device=actual_device)
        prediction = model.infer(img_tensor, f_px=f_px_tensor)
        depth_raw = prediction["depth"].squeeze().cpu().numpy()
    
    print(f'[depth_pro] Raw depth shape: {depth_raw.shape}, range: [{depth_raw.min():.2f}, {depth_raw.max():.2f}]m')
    
    # Fill NaN/inf with median
    valid = np.isfinite(depth_raw)
    if not valid.all():
        median_depth = float(np.nanmedian(depth_raw[valid]))
        depth_raw = np.where(valid, depth_raw, median_depth)
        print(f'[depth_pro] Filled invalid pixels with median depth: {median_depth:.2f}m')
    
    # Resize to canvas resolution
    H_canvas, W_canvas = canvas_hw
    depth_canvas = cv2.resize(depth_raw, (W_canvas, H_canvas), interpolation=cv2.INTER_LINEAR)
    
    print(f'[depth_pro] Resized to canvas {W_canvas}×{H_canvas}')
    print(f'[depth_pro] Canvas depth range: [{depth_canvas.min():.2f}, {depth_canvas.max():.2f}]m')
    
    return depth_canvas.astype(np.float32)
