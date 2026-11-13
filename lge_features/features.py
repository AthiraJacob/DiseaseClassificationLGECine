"""The 28 quantitative LGE features used in the paper.

Public entry point
------------------
    extract_lge_features(slices) -> dict[str, float]

`slices` is a list of `LGESlice`, normally three (basal, mid, apical). Every
feature is a ratio or a binary flag, so no pixel spacing is required and the
result is independent of image resolution.

The 28 features
---------------
    scar_burden                     scar pixels / myocardial pixels
    mean_transmurality, max_transmurality  over all scar-bearing rays
    extent_endocardial, extent_epicardial,
    extent_mid_myocardial, extent_transmural     ONE-HOT over the dominant
                                                 extent class, not independent
                                                 per-class indicators
    wall_anterior, wall_septal,
    wall_inferior, wall_lateral                  binary, scar in that wall
    rv_insertion_involved                        binary
    seg_1_burden ... seg_16_burden         per-AHA-segment scar fraction

Scar outside the myocardium is ignored everywhere: each slice's scar mask is
intersected with its myocardium mask on construction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage

from .aha import aha_segment_mask, segment_layout
from .constants import (
    EXTENT_PRIORITY,
    N_RAYS_PER_SEGMENT,
    RVIP_MAX_SNAP_PX,
    RVIP_SEARCH_RADIUS_PX,
    WALL_SEGMENTS,
)
from .geometry import angle_between, classify_ray, ray_angles


@dataclass
class LGESlice:
    """One short-axis LGE slice, already segmented.

    Attributes
    ----------
    myocardium : (H, W) array
        Binary myocardium mask. Non-zero is myocardium.
    scar : (H, W) array
        Binary scar mask, a subset of `myocardium`.
    level : {'basal', 'mid', 'apical'}
        Which third of the ventricle this slice represents.
    lv_center : (x, y)
        Centre of the LV cavity, in pixels.
    rv_insertion_anterior : (x, y)
        Anterior RV insertion point, in pixels. Anchors the AHA grid.
    rv_insertion_inferior : (x, y), optional
        Inferior RV insertion point. Only affects `rv_insertion_involved`.
    image : (H, W) array, optional
        The LGE image itself. Not used for feature extraction; carried so
        that a caller's segmentation model can be driven from the same
        object.
    """

    myocardium: np.ndarray
    scar: np.ndarray
    level: str
    lv_center: Tuple[float, float]
    rv_insertion_anterior: Tuple[float, float]
    rv_insertion_inferior: Optional[Tuple[float, float]] = None
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.myocardium = np.asarray(self.myocardium)
        self.scar = np.asarray(self.scar)

        for name, mask in (('myocardium', self.myocardium), ('scar', self.scar)):
            if mask.ndim != 2:
                raise ValueError(
                    f"{name} must be a 2-D (H, W) array, got {mask.ndim}-D "
                    f"with shape {mask.shape}")
            if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        if self.myocardium.shape != self.scar.shape:
            raise ValueError(
                f"myocardium {self.myocardium.shape} and scar {self.scar.shape} "
                "must have the same shape")

        segment_layout(self.level)   # raises if the level is not recognised

        # The anterior insertion point anchors the AHA grid, so both
        # landmarks must be present and inside the image.
        self.lv_center = _check_point(self.lv_center, 'lv_center',
                                      self.myocardium.shape)
        self.rv_insertion_anterior = _check_point(
            self.rv_insertion_anterior, 'rv_insertion_anterior',
            self.myocardium.shape)
        if self.rv_insertion_inferior is not None:
            self.rv_insertion_inferior = _check_point(
                self.rv_insertion_inferior, 'rv_insertion_inferior',
                self.myocardium.shape)

        # scar_burden is a fraction of myocardium, so scar is restricted
        # to the myocardium here.
        self.scar = (self.scar > 0) & (self.myocardium > 0)
        self.myocardium = self.myocardium > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_lge_features(slices: List[LGESlice]) -> Dict[str, float]:
    """Compute the 28 LGE features from a set of segmented slices."""
    if not slices:
        raise ValueError("at least one LGESlice is required")

    features: Dict[str, float] = {}
    features['scar_burden'] = scar_burden(slices)
    features.update(segmental_burdens(slices))
    features.update(transmurality_and_extent(slices))
    features['rv_insertion_involved'] = rv_insertion_involved(slices)
    return features


def scar_burden(slices: List[LGESlice]) -> float:
    """Total scar pixels divided by total myocardial pixels, over all slices.

    Scar has already been intersected with myocardium by `LGESlice`, so the
    result is always in [0, 1].
    """
    scar = sum(int(np.sum(s.scar > 0)) for s in slices)
    myo = sum(int(np.sum(s.myocardium > 0)) for s in slices)
    return scar / myo if myo > 0 else 0.0


def segmental_burdens(slices: List[LGESlice]) -> Dict[str, float]:
    """Scar fraction within each of the 16 AHA segments."""
    scar_px = {i: 0 for i in range(1, 17)}
    myo_px = {i: 0 for i in range(1, 17)}

    for s in slices:
        labels = aha_segment_mask(
            s.myocardium, s.lv_center, s.rv_insertion_anterior, s.level)
        for segment in range(1, 17):
            region = labels == segment
            if not region.any():
                continue
            myo_px[segment] += int(np.sum(region))
            scar_px[segment] += int(np.sum(s.scar[region] > 0))

    return {
        f'seg_{i}_burden': (scar_px[i] / myo_px[i]) if myo_px[i] > 0 else 0.0
        for i in range(1, 17)
    }


def _cast_rays(slices: List[LGESlice]):
    """Probe every AHA sector with radial rays and collect what they see.

    Returns (transmuralities, extent_counts, scarred_segments), where
    `transmuralities` holds one value per scar-bearing ray, `extent_counts`
    tallies the extent class of those rays, and `scarred_segments` is the set
    of AHA segments in which at least one ray found scar.
    """
    transmuralities: List[float] = []
    extent_counts = {'endocardial': 0, 'epicardial': 0,
                     'mid-myocardial': 0, 'transmural': 0}
    scarred_segments = set()

    for s in slices:
        reference = angle_between(s.lv_center, s.rv_insertion_anterior)
        n_sectors, sector_size, first_segment = segment_layout(s.level)

        for i in range(n_sectors):
            segment = first_segment + ((i + 1) % n_sectors)
            start = (reference + i * sector_size) % 360
            end = (reference + (i + 1) * sector_size) % 360

            has_scar = False
            for angle in ray_angles(end, start, N_RAYS_PER_SEGMENT):
                label, pct = classify_ray(s.myocardium, s.scar, s.lv_center, angle)
                if label != 'none':
                    transmuralities.append(pct / 100.0)
                    extent_counts[label] += 1
                    has_scar = True
            if has_scar:
                scarred_segments.add(segment)

    return transmuralities, extent_counts, scarred_segments


def _dominant(extent_counts: Dict[str, int]) -> str:
    """The most frequent extent class; ties broken by EXTENT_PRIORITY."""
    if sum(extent_counts.values()) == 0:
        return 'none'
    highest = max(extent_counts.values())
    for extent in EXTENT_PRIORITY:
        if extent_counts[extent] == highest:
            return extent
    return 'none'


def transmurality_and_extent(slices: List[LGESlice]) -> Dict[str, float]:
    """Transmurality, extent class flags, and wall involvement flags.

    Each AHA sector is probed with `N_RAYS_PER_SEGMENT` radial rays. Every ray
    that carries scar contributes its transmurality and its extent class.

    The four extent_* flags are a one-hot encoding of the DOMINANT extent
    class, that is, the class seen on the largest number of scar-bearing rays.
    Exactly one of the four is set whenever at least one ray finds scar, and
    none is set otherwise.

    Note that a scar too small or too thin to register on any ray leaves all
    four flags at zero while `scar_burden` is still positive; a ray
    must carry at least MIN_SCAR_EXTENT_THRESHOLD percent of wall thickness
    to count.
    """
    transmuralities, extent_counts, scarred_segments = _cast_rays(slices)
    dominant = _dominant(extent_counts)

    result: Dict[str, float] = {
        'mean_transmurality': float(np.mean(transmuralities)) if transmuralities else 0.0,
        'max_transmurality': float(np.max(transmuralities)) if transmuralities else 0.0,
        'extent_endocardial': int(dominant == 'endocardial'),
        'extent_epicardial': int(dominant == 'epicardial'),
        'extent_mid_myocardial': int(dominant == 'mid-myocardial'),
        'extent_transmural': int(dominant == 'transmural'),
    }
    for wall, segments in WALL_SEGMENTS.items():
        result[f'wall_{wall}'] = int(any(s in scarred_segments for s in segments))
    return result


def dominant_extent(slices: List[LGESlice]) -> str:
    """Most common extent class across all scar-bearing rays.

    Returns 'none', 'endocardial', 'mid-myocardial', 'epicardial' or
    'transmural'. This is the label the four extent_* features one-hot encode;
    it is offered separately because it is the readable form.
    """
    _, extent_counts, _ = _cast_rays(slices)
    return _dominant(extent_counts)


def rv_insertion_involved(slices: List[LGESlice]) -> int:
    """1 if scar touches either RV insertion point on any slice.

    The supplied point is snapped onto the nearest myocardial boundary pixel
    first, then scar is sought within `RVIP_SEARCH_RADIUS_PX` of it.
    """
    for s in slices:
        if np.sum(s.scar > 0) == 0 or np.sum(s.myocardium > 0) == 0:
            continue
        for point in (s.rv_insertion_anterior, s.rv_insertion_inferior):
            if point is None:
                continue
            snapped = _snap_to_boundary(point, s.myocardium)
            if snapped is not None and _scar_near(s.scar, s.myocardium, snapped):
                return 1
    return 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _check_point(point, name: str, shape) -> Tuple[float, float]:
    """Validate an (x, y) landmark and return it as a pair of floats."""
    if point is None:
        raise ValueError(
            f"{name} is required and must not be None; the AHA segment grid "
            "is anchored on it.")
    point = tuple(float(v) for v in point)
    if len(point) != 2:
        raise ValueError(f"{name} must be an (x, y) pair, got {point!r}")
    if not all(np.isfinite(v) for v in point):
        raise ValueError(f"{name} must be finite, got {point!r}")
    height, width = shape
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        raise ValueError(
            f"{name}={point} lies outside the image of shape {shape} "
            "(coordinates are (x, y) in pixels)")
    return point


def _boundary(myocardium: np.ndarray) -> np.ndarray:
    """Myocardial pixels that touch background on a 4-connected neighbour."""
    inside = myocardium > 0
    return inside & ndimage.binary_dilation(~inside, iterations=1)


def _snap_to_boundary(point, myocardium, max_distance=RVIP_MAX_SNAP_PX):
    """Nearest myocardial boundary pixel, or None if none is close enough."""
    boundary = _boundary(myocardium)
    if not boundary.any():
        return None
    rows, cols = np.where(boundary)
    distances = np.hypot(cols - point[0], rows - point[1])
    nearest = int(np.argmin(distances))
    if distances[nearest] > max_distance:
        return None
    return (float(cols[nearest]), float(rows[nearest]))


def _scar_near(scar, myocardium, point, radius=RVIP_SEARCH_RADIUS_PX) -> bool:
    """True if any myocardial scar pixel lies within `radius` of `point`."""
    rows, cols = scar.shape
    y, x = np.ogrid[:rows, :cols]
    within = (x - point[0]) ** 2 + (y - point[1]) ** 2 <= radius ** 2
    return bool(np.any((scar > 0) & within & (myocardium > 0)))
