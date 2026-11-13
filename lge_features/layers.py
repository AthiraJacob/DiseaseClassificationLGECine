"""Label the myocardial wall into endocardial, mid-myocardial and epicardial
layers.

The transmurality features classify each radial ray by splitting the
myocardium it crosses into three equal parts (see `geometry.classify_ray`).
This module labels every pixel by the same idea -- relative depth through the
wall, in equal thirds -- but measures that depth with a distance transform
rather than along a ray, which gives a smooth, gap-free mask independent of
any angular sampling.

The mask is provided for inspection. The features themselves are computed
from the rays.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage

ENDOCARDIAL, MID_MYOCARDIAL, EPICARDIAL = 1, 2, 3


def wall_layer_mask(myocardium: np.ndarray,
                    lv_center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Label each myocardial pixel by its depth through the wall.

    Depth runs from 0 at the endocardial (cavity) surface to 1 at the
    epicardial surface, and is split into equal thirds.

    Parameters
    ----------
    myocardium : (H, W) array
        Binary myocardium mask. The cavity is detected as the hole it
        encloses, so the mask must form a closed ring for the inner surface
        to be found.
    lv_center : (x, y), optional
        Only used as a fallback when the mask encloses no cavity, in which
        case depth is measured radially from this point.

    Returns
    -------
    (H, W) uint8 array
        0 outside the myocardium, 1 endocardial, 2 mid-myocardial,
        3 epicardial.
    """
    myo = np.asarray(myocardium) > 0
    labels = np.zeros(myo.shape, dtype=np.uint8)
    if not myo.any():
        return labels

    filled = ndimage.binary_fill_holes(myo)
    cavity = filled & ~myo
    background = ~filled

    if cavity.any():
        # Distance to the cavity and to the outside; their ratio is the
        # fractional depth through the wall.
        to_cavity = ndimage.distance_transform_edt(~cavity)
        to_outside = ndimage.distance_transform_edt(~background)
        total = to_cavity + to_outside
        depth = np.divide(to_cavity, total, out=np.zeros_like(total),
                          where=total > 0)
    elif lv_center is not None:
        # Open or solid mask: fall back to radial depth about the centre.
        depth = _radial_depth(myo, lv_center)
    else:
        raise ValueError(
            'myocardium encloses no cavity, so the endocardial surface '
            'cannot be found; pass lv_center to use radial depth instead')

    layer = np.where(depth < 1.0 / 3.0, ENDOCARDIAL,
                     np.where(depth < 2.0 / 3.0, MID_MYOCARDIAL, EPICARDIAL))
    labels[myo] = layer[myo].astype(np.uint8)
    return labels


def _radial_depth(myo: np.ndarray, lv_center: Tuple[float, float]) -> np.ndarray:
    """Fractional depth measured outward from a centre point."""
    ys, xs = np.nonzero(myo)
    radius = np.hypot(xs - lv_center[0], ys - lv_center[1])
    inner, outer = radius.min(), radius.max()
    span = outer - inner
    depth = np.zeros(myo.shape, dtype=float)
    depth[ys, xs] = (radius - inner) / span if span > 0 else 0.0
    return depth
