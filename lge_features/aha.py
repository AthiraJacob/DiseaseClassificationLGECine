"""AHA 16-segment labelling of a short-axis myocardial mask.

The segment grid is anchored at the anterior RV insertion point and runs
counter-clockwise: basal and mid slices are divided into six 60-degree
sectors, apical slices into four 90-degree sectors.
"""

from typing import Tuple

import numpy as np

from .constants import (
    APICAL_SEGMENT_SIZE,
    APICAL_SEGMENTS,
    BASAL_SEGMENT_SIZE,
    BASAL_SEGMENTS,
    MID_SEGMENTS,
)
from .geometry import angle_between


def sector_mask(shape: Tuple[int, int],
                center: Tuple[float, float],
                start_angle: float,
                end_angle: float) -> np.ndarray:
    """Binary mask of the wedge running counter-clockwise start -> end."""
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    angles = np.degrees(np.arctan2(-(y - center[1]), x - center[0]))
    angles[angles < 0] += 360

    if start_angle > end_angle:
        wedge = (angles <= start_angle) & (angles > end_angle)
    else:
        wedge = (angles > end_angle) | (angles <= start_angle)
    return wedge.astype(np.uint8)


def segment_layout(level: str):
    """(number of sectors, sector size in degrees, first segment number)."""
    if level == 'apical':
        return len(APICAL_SEGMENTS), APICAL_SEGMENT_SIZE, APICAL_SEGMENTS[0]
    if level == 'mid':
        return len(MID_SEGMENTS), BASAL_SEGMENT_SIZE, MID_SEGMENTS[0]
    if level == 'basal':
        return len(BASAL_SEGMENTS), BASAL_SEGMENT_SIZE, BASAL_SEGMENTS[0]
    raise ValueError(f"level must be 'basal', 'mid' or 'apical', got {level!r}")


def aha_segment_mask(myocardium: np.ndarray,
                     lv_center: Tuple[float, float],
                     rv_insertion: Tuple[float, float],
                     level: str) -> np.ndarray:
    """Label every myocardial pixel with its AHA segment number.

    Parameters
    ----------
    myocardium : (H, W) array
        Binary myocardium mask.
    lv_center : (x, y)
        Centre of the left ventricular cavity, in pixels.
    rv_insertion : (x, y)
        Anterior RV insertion point, in pixels. Anchors the segment grid.
    level : {'basal', 'mid', 'apical'}

    Returns
    -------
    (H, W) uint8 array
        0 outside the myocardium, otherwise the AHA segment number
        (1-6 basal, 7-12 mid, 13-16 apical).
    """
    if np.sum(myocardium) == 0:
        return np.zeros_like(myocardium, dtype=np.uint8)

    reference = angle_between(lv_center, rv_insertion)
    n_sectors, sector_size, first_segment = segment_layout(level)

    labels = np.zeros(myocardium.shape, dtype=np.uint8)
    for i in range(n_sectors):
        segment = first_segment + ((i + 1) % n_sectors)
        start = (reference + i * sector_size) % 360
        end = (reference + (i + 1) * sector_size) % 360
        wedge = sector_mask(myocardium.shape, lv_center, end, start)
        labels[(myocardium > 0) & (wedge > 0)] = segment
    return labels
