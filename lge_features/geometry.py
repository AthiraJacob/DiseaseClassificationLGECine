"""Geometry helpers: angles, radial rays, and per-ray scar classification.

Coordinates are (x, y) in pixels with the image convention that y increases
downward. Angles are in degrees in [0, 360), measured with 0 degrees at the
3 o'clock position and increasing counter-clockwise as seen on screen.
"""

from typing import Tuple

import numpy as np

from .constants import (
    LAYER_SCAR_THRESHOLD,
    MIN_SCAR_EXTENT_THRESHOLD,
    N_RAY_SAMPLES,
    TRANSMURAL_PCT_HIGH,
    TRANSMURAL_PCT_LOW,
)


def angle_between(center: Tuple[float, float],
                  point: Tuple[float, float]) -> float:
    """Angle from `center` to `point`, in degrees in [0, 360)."""
    if center is None or point is None:
        return 0.0
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = np.degrees(np.arctan2(-dy, dx))   # -dy: screen y grows downward
    if angle < 0:
        angle += 360
    return angle


def ray_angles(start_angle: float, end_angle: float, n_rays: int) -> np.ndarray:
    """`n_rays` evenly spaced angles sweeping counter-clockwise."""
    if start_angle > end_angle:
        span = start_angle - end_angle
    else:
        span = start_angle + (360 - end_angle)
    offsets = np.linspace(0, span, n_rays, endpoint=False)
    return (start_angle - offsets) % 360


def classify_ray(myocardium: np.ndarray,
                 scar: np.ndarray,
                 lv_center: Tuple[float, float],
                 angle_deg: float) -> Tuple[str, float]:
    """Classify the scar seen along one radial ray through the myocardium.

    The ray is sampled outward from `lv_center`. The myocardium it crosses is
    divided into three equal layers -- endocardial, mid-myocardial and
    epicardial -- and the scar fraction of each is measured.

    Returns
    -------
    (classification, transmurality_pct)
        classification is one of 'none', 'endocardial', 'mid-myocardial',
        'epicardial', 'transmural'. transmurality_pct is the percentage of
        wall thickness occupied by scar along this ray.
    """
    angle_rad = np.radians(angle_deg)
    max_radius = max(myocardium.shape) // 2 + 10
    radii = np.linspace(0, max_radius, N_RAY_SAMPLES)

    ray_x = lv_center[0] + radii * np.cos(angle_rad)
    ray_y = lv_center[1] - radii * np.sin(angle_rad)

    rows, cols = myocardium.shape
    inside = (ray_x >= 0) & (ray_x < cols) & (ray_y >= 0) & (ray_y < rows)
    ray_x, ray_y = ray_x[inside], ray_y[inside]
    if len(ray_x) == 0:
        return ('none', 0.0)

    yi, xi = ray_y.astype(int), ray_x.astype(int)
    myo_samples = myocardium[yi, xi]
    scar_samples = scar[yi, xi]

    myo_idx = np.where(myo_samples > 0)[0]
    if len(myo_idx) < 3:
        return ('none', 0.0)

    endo_idx, epi_idx = myo_idx[0], myo_idx[-1]
    thickness = epi_idx - endo_idx + 1
    if thickness < 3:
        return ('none', 0.0)

    # Split the wall into three equal layers.
    layer = thickness / 3.0
    endo_end = endo_idx + round(layer)
    mid_end = endo_idx + round(2 * layer)

    endo_frac = _scar_fraction(scar_samples[endo_idx:endo_end])
    mid_frac = _scar_fraction(scar_samples[endo_end:mid_end])
    epi_frac = _scar_fraction(scar_samples[mid_end:epi_idx + 1])

    wall_scar = scar_samples[endo_idx:epi_idx + 1]
    transmurality_pct = (np.sum(wall_scar > 0) / thickness) * 100
    if transmurality_pct < MIN_SCAR_EXTENT_THRESHOLD:
        return ('none', 0.0)

    has_endo = endo_frac >= LAYER_SCAR_THRESHOLD
    has_mid = mid_frac >= LAYER_SCAR_THRESHOLD
    has_epi = epi_frac >= LAYER_SCAR_THRESHOLD

    if transmurality_pct >= TRANSMURAL_PCT_HIGH:
        return ('transmural', transmurality_pct)
    if transmurality_pct >= TRANSMURAL_PCT_LOW and has_endo and has_epi:
        return ('transmural', transmurality_pct)
    if has_endo and has_mid and has_epi:
        return ('transmural', transmurality_pct)

    if has_endo and not has_epi:
        return ('endocardial', transmurality_pct)
    if has_epi and not has_endo:
        return ('epicardial', transmurality_pct)
    if has_endo and has_epi:
        dominant = 'endocardial' if endo_frac >= epi_frac else 'epicardial'
        return (dominant, transmurality_pct)
    if has_mid:
        return ('mid-myocardial', transmurality_pct)

    fractions = {'endocardial': endo_frac,
                 'mid-myocardial': mid_frac,
                 'epicardial': epi_frac}
    return (max(fractions, key=fractions.get), transmurality_pct)


def _scar_fraction(samples: np.ndarray) -> float:
    """Fraction of a layer's samples that are scar."""
    return float(np.mean(samples > 0)) if len(samples) > 0 else 0.0
