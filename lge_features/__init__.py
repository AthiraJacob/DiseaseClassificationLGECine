"""cmr-lge-features: quantitative LGE feature extraction from segmented
short-axis cardiac MR.

Computes the 28 scar features used in

    Jacob et al., "Automated Late Gadolinium Enhancement Quantification
    Improves Multi-Class Cardiac Disease Detection on CMR."

from binary myocardium and scar masks plus two landmark coordinates. No
segmentation model, no trained weights and no patient data are included; see
README.md.

Typical use
-----------
    from lge_features import LGESlice, extract_lge_features

    features = extract_lge_features([
        LGESlice(myocardium=myo_b, scar=scar_b, level='basal',
                 lv_center=(112, 110), rv_insertion_anterior=(140, 86)),
        ...
    ])
"""

from .aha import aha_segment_mask
from .constants import LEVELS, SEGMENT_NAMES
from .features import (
    LGESlice,
    dominant_extent,
    extract_lge_features,
    rv_insertion_involved,
    scar_burden,
    segmental_burdens,
    transmurality_and_extent,
)
from .interfaces import (
    LandmarkDetector,
    PrecomputedLandmarks,
    PrecomputedSegmenter,
    ScarSegmenter,
)
from .layers import wall_layer_mask
from .pipeline import FEATURE_ORDER, build_slices, run_pipeline

__version__ = '1.0.0'

__all__ = [
    'LGESlice',
    'extract_lge_features',
    'scar_burden',
    'segmental_burdens',
    'transmurality_and_extent',
    'rv_insertion_involved',
    'dominant_extent',
    'aha_segment_mask',
    'wall_layer_mask',
    'ScarSegmenter',
    'LandmarkDetector',
    'PrecomputedSegmenter',
    'PrecomputedLandmarks',
    'run_pipeline',
    'build_slices',
    'FEATURE_ORDER',
    'LEVELS',
    'SEGMENT_NAMES',
]
