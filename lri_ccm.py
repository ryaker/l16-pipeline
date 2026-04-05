"""
lri_ccm.py — Color Correction Matrix (CCM) estimation and application for L16 mirror cameras.

Mirror cameras (GLUED, MOVABLE) have a spectral shift from the mirror coating.
This module estimates and applies 3x3 CCMs to correct mirror camera colors to
match the wide cameras (NONE mirror type), which have no mirror in the optical path.

All images are assumed to be linear uint16 (no gamma).
"""

import warnings

import cv2
import numpy as np


def identity_ccm() -> np.ndarray:
    """Return a 3x3 identity CCM (no correction).

    Used as a fallback when estimation fails or correction is disabled.

    Returns:
        (3, 3) float64 identity matrix.
    """
    return np.eye(3, dtype=np.float64)


def estimate_ccm(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    mask: np.ndarray = None,
    exposure_neutral: bool = False,
) -> np.ndarray:
    """Estimate a 3x3 CCM mapping src_img colors to ref_img colors.

    Uses least-squares: CCM @ src_pixels.T ≈ ref_pixels.T

    Args:
        src_img:          Mirror camera image, uint16 HxWx3, linear.
        ref_img:          Wide camera image warped into src_img's space, uint16 HxWx3, linear.
        mask:             Optional boolean mask (H, W) of valid overlap pixels.
        exposure_neutral: If True, equalize median luminance of src/ref before fitting so the
                          CCM corrects only spectral (chromaticity) differences and does not
                          absorb exposure differences between cameras.  Mirror cameras are
                          typically 1.5–2× brighter than wide cameras at the same scene
                          because the telephoto optics collect more light per pixel.  Without
                          this flag the CCM absorbs that scaling, reducing the mirror camera's
                          pixel values and destroying sharpness (variance ∝ intensity²).

    Returns:
        (3, 3) float64 CCM matrix. Falls back to identity on failure.
    """
    assert src_img.shape == ref_img.shape, "src_img and ref_img must have the same shape"
    assert src_img.ndim == 3 and src_img.shape[2] == 3, "Images must be HxWx3"

    src_f = src_img.astype(np.float64)
    ref_f = ref_img.astype(np.float64)

    # Thresholds in [0, 65535] space
    DARK_THRESH = 1000.0
    SAT_THRESH = 60000.0

    # Build valid-pixel mask: not too dark, not saturated in either image
    valid = (
        np.all(src_f > DARK_THRESH, axis=2) &
        np.all(src_f < SAT_THRESH, axis=2) &
        np.all(ref_f > DARK_THRESH, axis=2) &
        np.all(ref_f < SAT_THRESH, axis=2)
    )

    if mask is not None:
        valid = valid & mask.astype(bool)

    n_valid = int(np.sum(valid))
    MIN_PIXELS = 100

    if n_valid < MIN_PIXELS:
        warnings.warn(
            f"estimate_ccm: only {n_valid} valid pixels (need {MIN_PIXELS}); "
            "returning identity CCM.",
            RuntimeWarning,
        )
        return identity_ccm()

    # Flatten to (N, 3)
    src_pixels = src_f[valid]  # (N, 3)
    ref_pixels = ref_f[valid]  # (N, 3)

    if exposure_neutral:
        # Equalize median per-pixel luminance (simple channel mean) so the CCM
        # fits spectral differences only and does not absorb the exposure gap.
        lum_src = np.median(src_pixels.mean(axis=1))
        lum_ref = np.median(ref_pixels.mean(axis=1))
        if lum_ref > 1e-6:
            ref_pixels = ref_pixels * (lum_src / lum_ref)

    # Diagonal-only CCM: per-channel gain from median ratio, no cross-channel mixing.
    #
    # A full 3×3 CCM uses cross-channel mixing to explain both spectral differences
    # and residual exposure mismatches.  For telephoto mirror cameras the src/ref
    # brightness ratio can be 3-5×, causing the optimizer to use large off-diagonal
    # terms that effectively smooth the image (cross-channel averaging = low-pass
    # filter in disguise).  Laplacian variance scales as intensity², so even a 0.5×
    # luminance change from off-diagonals cuts sharpness by 4×.
    #
    # Diagonal CCM: independently estimate per-channel gain as the median ratio.
    # This corrects the mirror coating's per-channel spectral shift with zero spatial
    # frequency cost, then normalizes so the green channel (most stable) stays at 1.0.
    gains = np.zeros(3, dtype=np.float64)
    for c in range(3):
        ratios = ref_pixels[:, c] / np.maximum(src_pixels[:, c], 1.0)
        gains[c] = float(np.median(ratios))

    # Normalize to green channel so overall luminance is preserved
    if gains[1] > 1e-6:
        gains /= gains[1]

    ccm = np.diag(gains)

    # Sanity check: reject wildly non-physical matrices
    if not np.all(np.isfinite(ccm)):
        warnings.warn(
            "estimate_ccm: non-finite values in CCM; returning identity.",
            RuntimeWarning,
        )
        return identity_ccm()

    return ccm


def apply_ccm(img: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply a 3x3 CCM to a linear uint16 image.

    Args:
        img: (H, W, 3) uint16 linear image.
        ccm: (3, 3) float64 CCM matrix.

    Returns:
        (H, W, 3) uint16 image with CCM applied, clipped to [0, 65535].
    """
    assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3"
    assert ccm.shape == (3, 3), "ccm must be (3, 3)"

    H, W = img.shape[:2]
    img_f = img.astype(np.float32).reshape(-1, 3)  # (N, 3)

    # Apply CCM: out_col = CCM @ in_col  →  out.T = CCM @ in.T
    # Row-wise: out[i] = (CCM @ in[i].T).T = in[i] @ CCM.T
    ccm_f = ccm.astype(np.float32)
    out_f = img_f @ ccm_f.T  # (N, 3)

    out_f = np.clip(out_f, 0.0, 65535.0)
    return out_f.reshape(H, W, 3).astype(np.uint16)


def estimate_ccm_from_cameras(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_cam: dict,
    ref_cam: dict,
    exposure_neutral: bool = True,
) -> np.ndarray:
    """Higher-level helper: estimate CCM from two camera images using calibration.

    Warps ref_img into src_img's space using a homography derived from the
    camera calibration (K, R) under the far-field (infinite scene depth)
    approximation, then estimates CCM on the overlapping region.

    Args:
        src_img:          Mirror camera image, uint16 HxWx3.
        ref_img:          Wide camera image, uint16 HxWx3.
        src_cam:          Camera dict with keys 'K' (3x3), 'R' (3x3), 't' (3,), 'W', 'H'.
        ref_cam:          Camera dict with keys 'K' (3x3), 'R' (3x3), 't' (3,), 'W', 'H'.
        exposure_neutral: Passed to estimate_ccm; default True so callers get chromaticity-only
                          correction without absorbing the telephoto exposure advantage.

    Returns:
        (3, 3) float64 CCM matrix.
    """
    src_H, src_W = src_img.shape[:2]

    # Far-field homography: H = K_src * R_src * R_ref^{-1} * K_ref^{-1}
    H_rel = (
        src_cam['K']
        @ src_cam['R']
        @ np.linalg.inv(ref_cam['R'])
        @ np.linalg.inv(ref_cam['K'])
    )

    # Warp ref_img into src_img coordinate space
    ref_warped = cv2.warpPerspective(
        ref_img,
        H_rel,
        (src_W, src_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Valid overlap mask: warped ref pixels that are non-zero
    overlap_mask = np.any(ref_warped > 0, axis=2)

    return estimate_ccm(src_img, ref_warped, mask=overlap_mask, exposure_neutral=exposure_neutral)


if __name__ == '__main__':
    # Smoke test: identity CCM applied to an image should be unchanged
    img = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
    ccm = identity_ccm()
    out = apply_ccm(img, ccm)
    assert np.allclose(img.astype(np.float32), out.astype(np.float32), atol=1)
    print("CCM smoke test passed")
