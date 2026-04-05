#!/usr/bin/env python3
"""
lri_virtual_camera.py — Synthetic output canvas for L16 multi-camera fusion.

Defines a VirtualCamera with two modes:

  tele (default):
    - FOV   = union of all wide cameras (mirror_type='NONE')
    - Pixel density = median telephoto focal length mapped to the wide FOV
    - Position (t) = centroid of wide camera translation vectors
    Yields a single rectilinear projection covering the full wide-angle view
    with telephoto-grade angular resolution (~75MP for standard L16 data).

  wide:
    - FOV   = union of all A-group cameras (28mm, fx≈3370)
    - Pixel density = median B-camera focal length (fx≈8276, telephoto resolution)
    - Position (t) = A1 translation vector (reference camera)
    Yields a 28mm-equivalent output at telephoto pixel density (~52-80MP).
    This is the primary mode described on Light.co's website: depth from the
    5×28mm cameras, then B cameras overlaid into the A reference frame.
"""

import numpy as np

# Camera name prefixes for the two groups
_A_CAMERAS = frozenset({'A1', 'A2', 'A3', 'A4', 'A5'})
_B_CAMERAS = frozenset({'B1', 'B2', 'B3', 'B4', 'B5'})

# Approximate focal length boundary between A (28mm) and B (70mm) cameras.
# A cameras: fx≈3370 px; B cameras: fx≈8276 px.
# Anything below this threshold is treated as a wide/A camera.
_WIDE_FX_THRESHOLD = 5000.0


class VirtualCamera:
    """
    Synthetic output canvas for the Light L16 fusion pipeline.

    Parameters
    ----------
    cameras : dict
        Camera dict produced by ``load_cameras()`` in lri_fuse_image.py.
        Each entry has keys: K, R, t, W, H, mirror_type.
    mode : str, optional
        Output mode.  One of:

        ``'tele'`` (default)
            fx≈8276 virtual camera covering the full wide FOV at telephoto
            pixel density.  Reference position = centroid of A cameras.

        ``'wide'``
            fx≈8276 virtual camera (B pixel density) over A-camera FOV (28mm).
            FOV = union of A-group cameras.  Reference position = A1
            translation vector (or centroid of A cameras if A1 is not
            available).  Yields 52-80MP output.

    Attributes
    ----------
    K : np.ndarray, shape (3, 3)
        Intrinsic matrix of the virtual output canvas.
    W, H : int
        Width and height of the output canvas in pixels.
    t : np.ndarray, shape (3,)
        Camera position in world coordinates.
        None if no translation vector is available.
    mode : str
        The mode this instance was built with (``'tele'`` or ``'wide'``).
    """

    def __init__(self, cameras: dict, mode: str = 'tele'):
        if mode not in ('tele', 'wide'):
            raise ValueError(f"mode must be 'tele' or 'wide', got {mode!r}")
        self.mode = mode

        if mode == 'tele':
            self._init_tele(cameras)
        else:
            self._init_wide(cameras)

    # ── Mode-specific initialisers ────────────────────────────────────────────

    def _init_tele(self, cameras: dict) -> None:
        """
        Tele mode (legacy default):
          FOV = union of A-group (wide) cameras.
          Pixel density = median B-group (telephoto) focal length.
          Position = centroid of A-camera translation vectors.
        """
        # ── Separate wide and telephoto cameras ──────────────────────────────
        wide = {n: c for n, c in cameras.items()
                if c.get('mirror_type', 'NONE') == 'NONE'}
        tele = {n: c for n, c in cameras.items()
                if c.get('mirror_type', 'NONE') != 'NONE'}

        if not wide:
            raise ValueError("No wide cameras (mirror_type='NONE') found.")
        if not tele:
            raise ValueError("No telephoto cameras found.")

        # ── Wide-camera FOV union ─────────────────────────────────────────────
        half_fov_h_list = []
        half_fov_v_list = []
        for cam in wide.values():
            K = cam['K']
            W = cam['W']
            H = cam['H']
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            half_fov_h_list.append(max(np.arctan(cx / fx),
                                       np.arctan((W - cx) / fx)))
            half_fov_v_list.append(max(np.arctan(cy / fy),
                                       np.arctan((H - cy) / fy)))

        half_fov_h = max(half_fov_h_list)
        half_fov_v = max(half_fov_v_list)

        # ── Telephoto pixel density ───────────────────────────────────────────
        tele_fx_values = [c['K'][0, 0] for c in tele.values()]
        tele_fy_values = [c['K'][1, 1] for c in tele.values()]
        median_tele_fx = float(np.median(tele_fx_values))
        median_tele_fy = float(np.median(tele_fy_values))

        output_fx = median_tele_fx
        output_fy = median_tele_fy

        W_out = int(round(2.0 * np.tan(half_fov_h) * output_fx))
        H_out = int(round(2.0 * np.tan(half_fov_v) * output_fy))
        cx_out = W_out / 2.0
        cy_out = H_out / 2.0

        self.K = np.array([[output_fx, 0.0,       cx_out],
                           [0.0,       output_fy, cy_out],
                           [0.0,       0.0,       1.0   ]], dtype=np.float64)
        self.W = W_out
        self.H = H_out

        # Position: centroid of wide-camera translations
        ts = [c['t'] for c in wide.values() if c.get('t') is not None]
        self.t = np.mean(np.stack(ts, axis=0), axis=0) if ts else None

    def _init_wide(self, cameras: dict) -> None:
        """
        Wide mode (28mm FOV, telephoto pixel density):
          FOV   = union of A-group cameras (28mm coverage).
          Pixel density = median B-group focal length (fx≈8276).
          Position = A1 translation vector, or centroid of A-group cameras.

        A cameras define the field-of-view; B cameras supply the pixel
        density that lifts the output to 52-80MP.  This matches Light's
        published architecture: depth from 5×28mm, then 70mm images overlaid.

        In wide mode ALL cameras (A and B) contribute to the output.
        B cameras with MOVABLE mirrors have stale R calibration — their
        images are still included but a warning is emitted by the fusion layer.
        B4 (GLUED mirror) has valid R calibration and is fully trusted.
        """
        # A-group cameras: named A1–A5, or fx < _WIDE_FX_THRESHOLD
        a_cams = {n: c for n, c in cameras.items()
                  if n in _A_CAMERAS or c['K'][0, 0] < _WIDE_FX_THRESHOLD}
        b_cams = {n: c for n, c in cameras.items()
                  if n in _B_CAMERAS or c['K'][0, 0] >= _WIDE_FX_THRESHOLD}
        if not a_cams:
            raise ValueError(
                "No A-group cameras found for wide mode "
                "(expected cameras named A1–A5 or fx < 5000)."
            )

        # ── A-camera FOV union ────────────────────────────────────────────────
        half_fov_h_list = []
        half_fov_v_list = []
        for cam in a_cams.values():
            K = cam['K']
            W = cam['W']
            H = cam['H']
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            half_fov_h_list.append(max(np.arctan(cx / fx),
                                       np.arctan((W - cx) / fx)))
            half_fov_v_list.append(max(np.arctan(cy / fy),
                                       np.arctan((H - cy) / fy)))

        half_fov_h = max(half_fov_h_list)
        half_fov_v = max(half_fov_v_list)

        # ── B-camera pixel density over A-camera FOV ─────────────────────────
        # B cameras (fx≈8276) define pixel density; A cameras define coverage.
        # Fallback to A-camera density if no B cameras present.
        if b_cams:
            b_fx_values = [c['K'][0, 0] for c in b_cams.values()]
            b_fy_values = [c['K'][1, 1] for c in b_cams.values()]
            output_fx = float(np.median(b_fx_values))
            output_fy = float(np.median(b_fy_values))
        else:
            a_fx_values = [c['K'][0, 0] for c in a_cams.values()]
            a_fy_values = [c['K'][1, 1] for c in a_cams.values()]
            output_fx = float(np.median(a_fx_values))
            output_fy = float(np.median(a_fy_values))

        W_out = int(round(2.0 * np.tan(half_fov_h) * output_fx))
        H_out = int(round(2.0 * np.tan(half_fov_v) * output_fy))
        cx_out = W_out / 2.0
        cy_out = H_out / 2.0

        self.K = np.array([[output_fx, 0.0,       cx_out],
                           [0.0,       output_fy, cy_out],
                           [0.0,       0.0,       1.0   ]], dtype=np.float64)
        self.W = W_out
        self.H = H_out

        # Position: prefer A1 as reference camera (matches Light.co architecture)
        if 'A1' in cameras and cameras['A1'].get('t') is not None:
            self.t = np.asarray(cameras['A1']['t'], dtype=np.float64).ravel()
        else:
            ts = [c['t'] for c in a_cams.values() if c.get('t') is not None]
            self.t = np.mean(np.stack(ts, axis=0), axis=0) if ts else None

    # ── Public interface ──────────────────────────────────────────────────────

    def output_shape(self) -> tuple[int, int]:
        """Return (H, W) of the output canvas."""
        return (self.H, self.W)

    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Convert output pixel coordinates to unit direction vectors.

        Parameters
        ----------
        u, v : np.ndarray, shape (N,)
            Pixel column and row coordinates on the output canvas.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Unit direction vectors in the virtual camera's coordinate frame.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        x = (u - cx) / fx
        y = (v - cy) / fy
        z = np.ones_like(x)

        rays = np.stack([x, y, z], axis=-1)          # (N, 3)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / norms

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3-D direction vectors onto the output canvas.

        Parameters
        ----------
        rays : np.ndarray, shape (N, 3)
            Direction vectors (need not be unit length).

        Returns
        -------
        u, v : np.ndarray, shape (N,)
            Pixel column and row coordinates on the output canvas.
        """
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Perspective projection (divide by z)
        z = rays[:, 2]
        u = fx * (rays[:, 0] / z) + cx
        v = fy * (rays[:, 1] / z) + cy
        return u, v


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from lri_fuse_image import load_cameras

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <calibration.json> [tele|wide]")
        sys.exit(1)

    cameras = load_cameras(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'tele'
    vc = VirtualCamera(cameras, mode=mode)
    print(f"Mode: {mode}")
    print(f"Output canvas: {vc.W}×{vc.H} = {vc.W * vc.H / 1e6:.1f}MP")
    print(f"K:\n{vc.K}")
    print(f"t: {vc.t}")
