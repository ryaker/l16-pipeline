#!/usr/bin/env python3
"""
lri_virtual_camera.py — Synthetic output canvas for L16 multi-camera fusion.

Defines a VirtualCamera whose:
  - FOV   = union of all wide cameras (mirror_type='NONE')
  - Pixel density = median telephoto focal length mapped to the wide FOV
  - Position (t) = centroid of wide camera translation vectors

This yields a single rectilinear projection that covers the full wide-angle
view with telephoto-grade angular resolution (~75MP for standard L16
calibration data).
"""

import numpy as np


class VirtualCamera:
    """
    Synthetic output canvas for the Light L16 fusion pipeline.

    Parameters
    ----------
    cameras : dict
        Camera dict produced by ``load_cameras()`` in lri_fuse_image.py.
        Each entry has keys: K, R, t, W, H, mirror_type.

    Attributes
    ----------
    K : np.ndarray, shape (3, 3)
        Intrinsic matrix of the virtual output canvas.
    W, H : int
        Width and height of the output canvas in pixels.
    t : np.ndarray, shape (3,)
        Camera position (centroid of wide-camera translation vectors).
        None if no wide camera has a translation vector.
    """

    def __init__(self, cameras: dict):
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
        # For each wide camera compute the half-angle FOV from the principal
        # point and sensor edges, then take the maximum across all cameras.
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
            # Angular half-extents to the far edge of the sensor
            half_fov_h_list.append(max(np.arctan(cx / fx),
                                       np.arctan((W - cx) / fx)))
            half_fov_v_list.append(max(np.arctan(cy / fy),
                                       np.arctan((H - cy) / fy)))

        half_fov_h = max(half_fov_h_list)   # radians
        half_fov_v = max(half_fov_v_list)   # radians

        # ── Telephoto pixel density ───────────────────────────────────────────
        # Use median telephoto fx and sensor half-width to derive the angular
        # resolution (pixels-per-radian) of the finest available camera.
        tele_fx_values = [c['K'][0, 0] for c in tele.values()]
        tele_fy_values = [c['K'][1, 1] for c in tele.values()]
        median_tele_fx = float(np.median(tele_fx_values))
        median_tele_fy = float(np.median(tele_fy_values))

        # ── Output canvas dimensions ──────────────────────────────────────────
        # Map the wide FOV at telephoto pixel density:
        #   output_W = 2 * tan(half_fov_h) * output_fx
        # where output_fx = median_tele_fx (pixels-per-unit-tan).
        output_fx = median_tele_fx
        output_fy = median_tele_fy

        W_out = int(round(2.0 * np.tan(half_fov_h) * output_fx))
        H_out = int(round(2.0 * np.tan(half_fov_v) * output_fy))

        # Principal point at centre of output canvas
        cx_out = W_out / 2.0
        cy_out = H_out / 2.0

        # ── Build output intrinsic matrix ─────────────────────────────────────
        self.K = np.array([[output_fx, 0.0,       cx_out],
                           [0.0,       output_fy, cy_out],
                           [0.0,       0.0,       1.0   ]], dtype=np.float64)
        self.W = W_out
        self.H = H_out

        # ── Camera position: centroid of wide-camera translations ─────────────
        ts = [c['t'] for c in wide.values() if c.get('t') is not None]
        if ts:
            self.t = np.mean(np.stack(ts, axis=0), axis=0)
        else:
            self.t = None

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
        print(f"Usage: {sys.argv[0]} <calibration.json>")
        sys.exit(1)

    cameras = load_cameras(sys.argv[1])
    vc = VirtualCamera(cameras)
    print(f"Output canvas: {vc.W}×{vc.H} = {vc.W * vc.H / 1e6:.1f}MP")
    print(f"K:\n{vc.K}")
