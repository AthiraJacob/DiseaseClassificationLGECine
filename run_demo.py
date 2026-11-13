"""Run the pipeline on the synthetic phantom and check the result.

    python run_demo.py            compute and check against examples/output/
    python run_demo.py --write    regenerate examples/output/

By default nothing is written: the features and label masks are computed in
memory and compared against the reference stored in `examples/output/`, and
the script prints PASS or FAIL. Pass `--write` to refresh that reference
after an intentional change.

The reference in `examples/output/` is produced by this same code, so PASS
means the output has not changed.

No patient data and no trained model are involved. The segmenter and landmark
detector used here are `PrecomputedSegmenter` and `PrecomputedLandmarks`,
which simply return the masks and points stored alongside the images.
Substitute your own implementations of `ScarSegmenter` and `LandmarkDetector`
to run the same pipeline on real data.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from lge_features import (
    FEATURE_ORDER,
    LEVELS,
    PrecomputedLandmarks,
    PrecomputedSegmenter,
    aha_segment_mask,
    build_slices,
    extract_lge_features,
    run_pipeline,
    wall_layer_mask,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PHANTOM = os.path.join(HERE, 'examples', 'phantom')
OUTPUT = os.path.join(HERE, 'examples', 'output')
TOLERANCE = 1e-6


def load_png(path):
    return np.array(Image.open(path))


def phantom_inputs():
    """Images, segmenters and landmark detectors for the three phantom slices."""
    with open(os.path.join(PHANTOM, 'landmarks.json')) as f:
        landmarks = json.load(f)

    inputs, segmenters, detectors = [], {}, {}
    for level in LEVELS:
        inputs.append((level, load_png(os.path.join(PHANTOM, f'{level}_image.png'))))
        segmenters[level] = PrecomputedSegmenter(
            myocardium=load_png(os.path.join(PHANTOM, f'{level}_myocardium_mask.png')),
            scar=load_png(os.path.join(PHANTOM, f'{level}_scar_mask.png')),
        )
        marks = landmarks[level]
        detectors[level] = PrecomputedLandmarks(
            lv_center=marks['lv_center'],
            rv_insertion_anterior=marks['rv_insertion_anterior'],
            rv_insertion_inferior=marks.get('rv_insertion_inferior'),
        )
    return inputs, segmenters, detectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='regenerate examples/output/ instead of checking it')
    args = parser.parse_args()

    inputs, segmenters, detectors = phantom_inputs()

    if args.write:
        features = run_pipeline(inputs, segmenters.get, detectors.get,
                                out_dir=OUTPUT)
        print_features(features)
        print(f'\nwrote {OUTPUT}')
        return 0

    slices = build_slices(inputs, segmenters.get, detectors.get)
    features = extract_lge_features(slices)
    print_features(features)
    return check(features, slices)


def print_features(features):
    print('28 LGE features from the synthetic phantom')
    print('-' * 46)
    for name in FEATURE_ORDER:
        value = features[name]
        print(f'  {name:<26} {value:.6f}' if isinstance(value, float)
              else f'  {name:<26} {value}')


def check(features, slices):
    """Compare computed features and masks against examples/output/."""
    reference = os.path.join(OUTPUT, 'features.json')
    if not os.path.exists(reference):
        print(f'\nFAIL - reference missing: {reference}')
        print('       run with --write to create it.')
        return 1

    with open(reference) as f:
        expected = json.load(f)

    problems = []
    for name in FEATURE_ORDER:
        got, want = features[name], expected.get(name)
        if want is None or abs(float(got) - float(want)) > TOLERANCE:
            problems.append(f'{name:<26} got {got}  expected {want}')

    # The label masks are first-class outputs, so check them too. They are
    # recomputed here rather than read back, so nothing is written.
    n_masks = 0
    for s in slices:
        for kind, computed in (
            ('aha', aha_segment_mask(s.myocardium, s.lv_center,
                                     s.rv_insertion_anterior, s.level)),
            ('layers', wall_layer_mask(s.myocardium, s.lv_center)),
        ):
            n_masks += 1
            path = os.path.join(OUTPUT, f'{kind}_{s.level}_mask.png')
            if not os.path.exists(path):
                problems.append(f'{kind}_{s.level}_mask.png missing')
            elif not np.array_equal(computed, load_png(path)):
                problems.append(f'{kind}_{s.level}_mask.png differs')

    print('-' * 46)
    if problems:
        print(f'FAIL - {len(problems)} difference(s) from examples/output/:')
        for problem in problems:
            print(f'  {problem}')
        return 1
    print(f'PASS - all {len(FEATURE_ORDER)} features and {n_masks} label '
          f'masks match examples/output/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
