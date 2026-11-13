"""End-to-end pipeline: images -> segmentation -> landmarks -> features.

    run_pipeline(inputs, segmenter_for, landmarks_for, out_dir)

The segmentation and landmark models are supplied by the caller through the
interfaces in `interfaces.py`; none are included in this release.

Outputs written to `out_dir`
----------------------------
    features.json             the 28 features, named
    features.csv              the same, one header row and one data row
    aha_<level>_mask.png      AHA segment labels, 0 = background,
                              1-16 = segment
    layers_<level>_mask.png   wall layers, 0 = background, 1 = endocardial,
                              2 = mid-myocardial, 3 = epicardial
    overlay_<level>.png       myocardium, scar and segment boundaries (needs
                              matplotlib; skipped with a message if
                              unavailable)
"""

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aha import aha_segment_mask
from .features import LGESlice, dominant_extent, extract_lge_features
from .layers import wall_layer_mask

# Display palettes for the label masks. Index 0 is background; the remaining
# entries colour the label values. These affect display only -- the stored
# pixel values are the label numbers.
LAYER_PALETTE = [
    (0, 0, 0),          # 0 background
    (214, 69, 65),      # 1 endocardial
    (233, 196, 78),     # 2 mid-myocardial
    (62, 146, 204),     # 3 epicardial
]

AHA_PALETTE = [
    (0, 0, 0),          # 0 background
    (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
    (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
    (188, 128, 189), (204, 235, 197), (255, 237, 111), (166, 206, 227),
    (31, 120, 180), (178, 223, 138), (251, 154, 153), (227, 26, 28),
]

FEATURE_ORDER = (
    ['scar_burden', 'mean_transmurality', 'max_transmurality',
     'extent_endocardial', 'extent_epicardial', 'extent_mid_myocardial',
     'extent_transmural', 'wall_anterior', 'wall_septal', 'wall_inferior',
     'wall_lateral', 'rv_insertion_involved']
    + [f'seg_{i}_burden' for i in range(1, 17)]
)


def build_slices(inputs: Sequence[Tuple[str, np.ndarray]],
                 segmenter_for,
                 landmarks_for) -> List[LGESlice]:
    """Run segmentation and landmark detection to produce LGESlice objects.

    Parameters
    ----------
    inputs : sequence of (level, image)
        `level` is 'basal', 'mid' or 'apical'.
    segmenter_for : callable(level) -> ScarSegmenter
    landmarks_for : callable(level) -> LandmarkDetector
    """
    slices = []
    for level, image in inputs:
        myocardium, scar = segmenter_for(level).segment(image)
        marks = landmarks_for(level).detect(image, myocardium)
        slices.append(LGESlice(
            myocardium=myocardium,
            scar=scar,
            level=level,
            lv_center=marks['lv_center'],
            rv_insertion_anterior=marks['rv_insertion_anterior'],
            rv_insertion_inferior=marks.get('rv_insertion_inferior'),
            image=image,
        ))
    return slices


def run_pipeline(inputs: Sequence[Tuple[str, np.ndarray]],
                 segmenter_for,
                 landmarks_for,
                 out_dir: Optional[str] = None,
                 overlays: bool = True) -> Dict[str, float]:
    """Segment, locate landmarks, extract features, and write outputs."""
    slices = build_slices(inputs, segmenter_for, landmarks_for)
    features = extract_lge_features(slices)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        write_features(features, out_dir)
        write_aha_masks(slices, out_dir)
        write_layer_masks(slices, out_dir)
        if overlays:
            write_overlays(slices, out_dir)
    return features


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_features(features: Dict[str, float], out_dir: str) -> None:
    """Write features.json and features.csv."""
    ordered = {k: features[k] for k in FEATURE_ORDER}
    with open(os.path.join(out_dir, 'features.json'), 'w') as f:
        json.dump(ordered, f, indent=2)
        f.write('\n')
    with open(os.path.join(out_dir, 'features.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ordered.keys())
        writer.writerow([_round(v) for v in ordered.values()])


def write_aha_masks(slices: List[LGESlice], out_dir: str) -> None:
    """Write one PNG per slice holding the AHA segment labels (0-16)."""
    for s in slices:
        labels = aha_segment_mask(
            s.myocardium, s.lv_center, s.rv_insertion_anterior, s.level)
        _write_png(os.path.join(out_dir, f'aha_{s.level}_mask.png'),
                   labels, AHA_PALETTE)


def write_layer_masks(slices: List[LGESlice], out_dir: str) -> None:
    """Write one PNG per slice holding the wall layer labels (1-3)."""
    for s in slices:
        layers = wall_layer_mask(s.myocardium, s.lv_center)
        _write_png(os.path.join(out_dir, f'layers_{s.level}_mask.png'),
                   layers, LAYER_PALETTE)


def write_overlays(slices: List[LGESlice], out_dir: str) -> None:
    """Write a human-readable overlay per slice. Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed - skipping overlay PNGs)')
        return

    for s in slices:
        labels = aha_segment_mask(
            s.myocardium, s.lv_center, s.rv_insertion_anterior, s.level)
        fig, ax = plt.subplots(figsize=(4, 4))
        if s.image is not None:
            ax.imshow(s.image, cmap='gray')
        ax.imshow(np.ma.masked_where(labels == 0, labels),
                  cmap='tab20', alpha=0.45, vmin=1, vmax=16)
        ax.contour(s.scar > 0, levels=[0.5], colors='red', linewidths=1.2)
        ax.plot(*s.lv_center, 'w+', markersize=10)
        ax.plot(*s.rv_insertion_anterior, 'yo', markersize=5)
        ax.set_title(f'{s.level} - AHA segments, scar in red')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'overlay_{s.level}.png'), dpi=130)
        plt.close(fig)


def _write_png(path: str, array: np.ndarray, palette=None) -> None:
    """Save a label image losslessly as an indexed-colour PNG.

    The stored pixel values are the label numbers themselves, so reading the
    file back with `numpy.array(PIL.Image.open(path))` returns the labels
    unchanged. The palette only affects how a viewer displays them -- without
    it a mask holding values 0-3 or 0-16 looks almost uniformly black.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            'writing PNGs needs Pillow; install it, or call '
            'extract_lge_features directly and save the arrays yourself'
        ) from exc

    data = np.ascontiguousarray(array.astype(np.uint8))
    height, width = data.shape
    image = Image.frombytes('P', (width, height), data.tobytes())

    colours = list(palette or [])
    colours += [(0, 0, 0)] * (256 - len(colours))      # pad to a full palette
    image.putpalette([channel for colour in colours for channel in colour])
    image.save(path)


def _round(value, places: int = 6):
    return round(value, places) if isinstance(value, float) else value
