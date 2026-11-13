"""Property tests for the LGE feature extractor.

These double as executable documentation: each test states, in code, what one
of the features is supposed to mean. They build their own small phantoms and
need no data.

    pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lge_features import (                                      # noqa: E402
    LGESlice,
    aha_segment_mask,
    extract_lge_features,
    wall_layer_mask,
)

SIZE = 224
CENTER = (112.0, 112.0)
R_IN, R_OUT = 30.0, 50.0
ANT_RVIP_ANGLE = 90.0


def polar(size=SIZE, center=CENTER):
    y, x = np.mgrid[:size, :size]
    dx, dy = x - center[0], y - center[1]
    return np.hypot(dx, dy), np.degrees(np.arctan2(-dy, dx)) % 360


def arc(angle, start, end):
    if start <= end:
        return (angle >= start) & (angle <= end)
    return (angle >= start) | (angle <= end)


def point_at(angle_deg, radius, center=CENTER):
    rad = np.radians(angle_deg)
    return (center[0] + radius * np.cos(rad), center[1] - radius * np.sin(rad))


def make_slice(scar_arc=None, depth=1.0, level='basal', scale=1.0,
               rvip_angle=ANT_RVIP_ANGLE):
    """One phantom slice; `scar_arc` is (start_deg, end_deg) or None."""
    size = int(SIZE * scale)
    center = (CENTER[0] * scale, CENTER[1] * scale)
    r_in, r_out = R_IN * scale, R_OUT * scale

    radius, angle = polar(size, center)
    myocardium = (radius >= r_in) & (radius <= r_out)

    if scar_arc is None:
        scar = np.zeros_like(myocardium)
    else:
        outer = r_in + depth * (r_out - r_in)
        scar = myocardium & arc(angle, *scar_arc) & (radius <= outer)

    return LGESlice(
        myocardium=myocardium,
        scar=scar,
        level=level,
        lv_center=center,
        rv_insertion_anterior=point_at(rvip_angle, r_out, center),
    )


# --------------------------------------------------------------------------

def test_no_scar_gives_all_zero_features():
    features = extract_lge_features([make_slice(scar_arc=None)])
    assert all(value == 0 for value in features.values())


def test_global_burden_is_scar_pixels_over_myocardial_pixels():
    s = make_slice(scar_arc=(90, 150))
    features = extract_lge_features([s])
    expected = s.scar.sum() / s.myocardium.sum()
    assert features['scar_burden'] == pytest.approx(expected)


def test_scar_filling_one_sector_isolates_that_segment():
    # Sector 0 runs counter-clockwise from the insertion point and, for a
    # basal slice, carries label 2.
    s = make_slice(scar_arc=(ANT_RVIP_ANGLE, ANT_RVIP_ANGLE + 60), depth=1.0)
    features = extract_lge_features([s])
    assert features['seg_2_burden'] > 0.95
    others = [features[f'seg_{i}_burden'] for i in range(1, 17) if i != 2]
    assert max(others) < 0.05


EXTENT_FLAGS = ['extent_endocardial', 'extent_epicardial',
                'extent_mid_myocardial', 'extent_transmural']


def test_extent_flags_are_one_hot_when_scar_is_present():
    # The four extent features encode a single dominant class, so exactly
    # one must be set whenever a ray finds scar.
    for arc_span, depth in [((90, 150), 1.0), ((90, 150), 0.3),
                            ((90, 270), 1.0), ((100, 130), 0.5)]:
        features = extract_lge_features(
            [make_slice(scar_arc=arc_span, depth=depth)])
        assert sum(features[f] for f in EXTENT_FLAGS) == 1


def test_extent_flags_are_all_zero_without_scar():
    features = extract_lge_features([make_slice(scar_arc=None)])
    assert sum(features[f] for f in EXTENT_FLAGS) == 0


def test_mixed_extent_reports_only_the_dominant_class():
    # A transmural basal slice and a thin subendocardial apical slice: the
    # transmural rays outnumber the subendocardial ones, so transmural wins
    # and the endocardial flag stays clear.
    slices = [make_slice(scar_arc=(90, 210), depth=1.0, level='basal'),
              make_slice(scar_arc=(95, 125), depth=0.3, level='apical')]
    features = extract_lge_features(slices)
    assert features['extent_transmural'] == 1
    assert features['extent_endocardial'] == 0
    assert sum(features[f] for f in EXTENT_FLAGS) == 1


def test_full_thickness_scar_is_transmural():
    features = extract_lge_features([make_slice(scar_arc=(90, 150), depth=1.0)])
    assert features['max_transmurality'] == pytest.approx(1.0, abs=0.05)
    assert features['extent_transmural'] == 1


def test_thin_subendocardial_rim_is_not_transmural():
    features = extract_lge_features([make_slice(scar_arc=(90, 150), depth=0.3)])
    assert features['extent_endocardial'] == 1
    assert features['extent_transmural'] == 0
    assert features['max_transmurality'] < 0.5


def test_septal_scar_sets_septal_wall_flag():
    # The 120-degree arc covers segments 2 and 3 (septal) and clips segment 1.
    # Wall flags have no burden threshold, so the clip raises wall_anterior too.
    s = make_slice(scar_arc=(ANT_RVIP_ANGLE, ANT_RVIP_ANGLE + 120), depth=1.0)
    features = extract_lge_features([s])
    assert features['wall_septal'] == 1
    assert features['wall_inferior'] == 0
    assert features['wall_lateral'] == 0
    assert features['wall_anterior'] == 1


def test_features_are_resolution_invariant():
    small = extract_lge_features([make_slice(scar_arc=(90, 150), scale=1.0)])
    large = extract_lge_features([make_slice(scar_arc=(90, 150), scale=2.0)])
    assert small['scar_burden'] == pytest.approx(
        large['scar_burden'], abs=0.01)
    for i in range(1, 17):
        assert small[f'seg_{i}_burden'] == pytest.approx(
            large[f'seg_{i}_burden'], abs=0.02)


def test_rotating_the_insertion_point_rotates_the_segment_labels():
    # Moving the insertion point one sector counter-clockwise should relabel
    # a fixed scar from segment 2 to segment 1 (the preceding sector).
    base = make_slice(scar_arc=(90, 150), rvip_angle=90)
    rotated = make_slice(scar_arc=(90, 150), rvip_angle=150)
    assert extract_lge_features([base])['seg_2_burden'] > 0.95
    assert extract_lge_features([rotated])['seg_1_burden'] > 0.95


def test_scar_at_the_insertion_point_is_detected():
    on_point = make_slice(scar_arc=(ANT_RVIP_ANGLE - 10, ANT_RVIP_ANGLE + 10))
    away = make_slice(scar_arc=(ANT_RVIP_ANGLE + 150, ANT_RVIP_ANGLE + 190))
    assert extract_lge_features([on_point])['rv_insertion_involved'] == 1
    assert extract_lge_features([away])['rv_insertion_involved'] == 0


def test_aha_mask_covers_the_myocardium_and_only_valid_labels():
    s = make_slice(scar_arc=None, level='basal')
    labels = aha_segment_mask(s.myocardium, s.lv_center,
                              s.rv_insertion_anterior, 'basal')
    assert set(np.unique(labels[s.myocardium])) <= set(range(1, 7))
    assert labels[~s.myocardium].max() == 0
    assert (labels[s.myocardium] > 0).all()


def test_layer_mask_covers_the_myocardium_with_three_labels():
    s = make_slice(scar_arc=None)
    layers = wall_layer_mask(s.myocardium, s.lv_center)
    assert set(np.unique(layers[s.myocardium])) == {1, 2, 3}
    assert layers[~s.myocardium].max() == 0
    assert (layers[s.myocardium] > 0).all()


def test_layer_mask_is_ordered_inner_to_outer():
    # Endocardial pixels must sit closer to the LV centre than epicardial
    # ones, with mid-myocardium between them.
    s = make_slice(scar_arc=None)
    layers = wall_layer_mask(s.myocardium, s.lv_center)
    ys, xs = np.nonzero(s.myocardium)
    radius = np.hypot(xs - s.lv_center[0], ys - s.lv_center[1])
    mean_radius = [radius[layers[ys, xs] == layer].mean() for layer in (1, 2, 3)]
    assert mean_radius[0] < mean_radius[1] < mean_radius[2]


def test_layer_mask_thirds_are_comparable_in_size():
    # Equal-depth thirds of an annulus: the outer ring has more pixels than
    # the inner one, but no layer should collapse or dominate.
    s = make_slice(scar_arc=None)
    layers = wall_layer_mask(s.myocardium, s.lv_center)
    counts = [int((layers == layer).sum()) for layer in (1, 2, 3)]
    assert min(counts) > 0.2 * sum(counts) / 3
    assert counts[0] < counts[2]          # outer third is geometrically larger


def test_layer_mask_is_empty_for_empty_myocardium():
    empty = np.zeros((32, 32), dtype=bool)
    assert wall_layer_mask(empty, (16.0, 16.0)).max() == 0


def test_apical_level_uses_four_segments():
    s = make_slice(scar_arc=None, level='apical')
    labels = aha_segment_mask(s.myocardium, s.lv_center,
                              s.rv_insertion_anterior, 'apical')
    assert set(np.unique(labels[s.myocardium])) <= {13, 14, 15, 16}


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        make_slice(scar_arc=None, level='middle')


def test_mismatched_mask_shapes_are_rejected():
    with pytest.raises(ValueError):
        LGESlice(myocardium=np.zeros((10, 10)), scar=np.zeros((12, 12)),
                 level='basal', lv_center=(5, 5), rv_insertion_anterior=(8, 5))


def test_scar_outside_the_myocardium_is_ignored():
    # A segmenter may emit scar beyond the myocardium. It must not leak into
    # scar_burden, which is a fraction OF the myocardium.
    s = make_slice(scar_arc=(90, 150), depth=1.0)
    everywhere = LGESlice(
        myocardium=s.myocardium,
        scar=np.ones_like(s.myocardium),      # scar over the whole frame
        level='basal',
        lv_center=s.lv_center,
        rv_insertion_anterior=s.rv_insertion_anterior,
    )
    burden = extract_lge_features([everywhere])['scar_burden']
    assert burden == pytest.approx(1.0)       # never above 1.0


def test_missing_landmarks_are_required():
    # The AHA grid is anchored on the anterior insertion point, so both
    # landmarks must be supplied.
    base = make_slice(scar_arc=(90, 150))
    for bad in ({'lv_center': None}, {'rv_insertion_anterior': None}):
        kwargs = dict(myocardium=base.myocardium, scar=base.scar,
                      level='basal', lv_center=base.lv_center,
                      rv_insertion_anterior=base.rv_insertion_anterior)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            LGESlice(**kwargs)


def test_landmarks_outside_the_image_are_rejected():
    base = make_slice(scar_arc=None)
    with pytest.raises(ValueError):
        LGESlice(myocardium=base.myocardium, scar=base.scar, level='basal',
                 lv_center=base.lv_center,
                 rv_insertion_anterior=(10_000.0, 10_000.0))


def test_non_2d_masks_are_rejected():
    with pytest.raises(ValueError):
        LGESlice(myocardium=np.zeros((4, 10, 10)), scar=np.zeros((4, 10, 10)),
                 level='basal', lv_center=(5, 5), rv_insertion_anterior=(8, 5))


@pytest.mark.parametrize('arc,depth', [
    (None, 1.0), ((90, 95), 0.1), ((90, 150), 0.3), ((90, 150), 1.0),
    ((0, 359), 1.0), ((200, 40), 0.6), ((45, 90), 0.05),
])
def test_extent_flags_are_never_more_than_one_hot(arc, depth):
    # The flags encode one dominant class, so their sum is 0 or 1 for any
    # input. A sum of 0 is legitimate: scar can be too thin to register on
    # any ray while still contributing to scar_burden.
    features = extract_lge_features([make_slice(scar_arc=arc, depth=depth)])
    assert sum(features[f] for f in EXTENT_FLAGS) in (0, 1)
