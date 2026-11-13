# cmr-lge-features

Quantitative late gadolinium enhancement (LGE) feature extraction from
segmented short-axis cardiac MR, and code to reproduce the published results
of the accompanying manuscript (see [Citation](#citation)).

The package has two aims:

**Aim 1** — turn a myocardium mask, a scar mask and two landmark coordinates
into the 28 scar features used by the paper's classifiers.

**Aim 2** — regenerate the paper's result tables from the released test-set
predictions.

---

## Repository layout

```
cmr-lge-features/
│
├── lge_features/              ── Aim 1: the feature extractor
│   ├── constants.py             every threshold, in one place
│   ├── geometry.py              angles, radial rays, per-ray scar classification
│   ├── aha.py                   AHA 16-segment labelling
│   ├── layers.py                endocardial / mid / epicardial wall layers
│   ├── features.py              LGESlice and extract_lge_features  ← main entry point
│   ├── interfaces.py            ScarSegmenter / LandmarkDetector plug-in points
│   └── pipeline.py              end-to-end run and output writing
│
├── examples/                  ── Aim 1: worked example, synthetic data only
│   ├── phantom/                 input images, masks and landmarks
│   └── output/                  the result the phantom should produce
├── run_demo.py                  Aim 1: runs the phantom and self-checks
├── tests/                       Aim 1: property tests, one per feature definition
│
├── reproduce/                 ── Aim 2: the published results
│   ├── predictions/             de-identified held-out test-set predictions
│   ├── stats.py                 McNemar, DeLong, Fisher, bootstrap
│   └── reproduce_tables.py      regenerates Tables 3–4, S4, S5
│
├── README.md  LICENSE  requirements.txt
```

---

## What is and is not included

| Component | Included | Notes |
|---|---|---|
| LGE feature extraction (AHA mapping, transmurality, extent, wall location, RV insertion, segmental burden) | **Yes** | `lge_features/`, pure NumPy/SciPy |
| Wall layer labelling | **Yes** | `lge_features/layers.py` |
| Worked example on synthetic data | **Yes** | `examples/`, `run_demo.py` |
| Statistical tests (McNemar, DeLong, Fisher, bootstrap) | **Yes** | `reproduce/stats.py` |
| De-identified held-out test-set predictions | **Yes** | `reproduce/predictions/` |
| Script regenerating the paper's tables | **Yes** | `reproduce/reproduce_tables.py` |
| Scar segmentation model and weights | **No** | proprietary |
| Landmark detection model and weights | **No** | proprietary |
| Trained disease classifiers (XGBoost models) | **No** | proprietary |
| Classifier training code | **No** | proprietary |
| Patient images, masks or feature values | **No** | patient data; cannot be shared |

The segmentation and landmark models are not part of this release. The
pipeline exposes them as interfaces (`ScarSegmenter`, `LandmarkDetector`) so
that an independent implementation can be substituted and everything
downstream run unchanged.

Nothing in this package reads DICOM or assumes any particular image source: a
segmenter receives a 2-D array and returns two boolean masks.

---

## Install

```bash
pip install -r requirements.txt
```

NumPy and SciPy are the only requirements for feature extraction itself.

---

# Aim 1 — feature extraction

## Quick start

```bash
python run_demo.py        # runs the phantom and checks it against examples/output/
pytest tests/             # 31 property tests, no data needed
```

`run_demo.py` writes nothing by default: it computes the features and masks in
memory and compares them with the reference in `examples/output/`, printing
PASS or FAIL. Use `python run_demo.py --write` to refresh that reference after
an intentional change.

## Using it on your own data

```python
import numpy as np
from lge_features import LGESlice, extract_lge_features

slices = [
    LGESlice(myocardium=myo_basal,  scar=scar_basal,  level='basal',
             lv_center=(112.0, 110.0), rv_insertion_anterior=(140.0, 86.0)),
    LGESlice(myocardium=myo_mid,    scar=scar_mid,    level='mid',    ...),
    LGESlice(myocardium=myo_apical, scar=scar_apical, level='apical', ...),
]

features = extract_lge_features(slices)   # dict of 28 named values
```

Inputs per slice:

| Field | Type | Meaning |
|---|---|---|
| `myocardium` | `(H, W)` bool | myocardium mask |
| `scar` | `(H, W)` bool | scar mask, a subset of `myocardium` |
| `level` | str | `'basal'`, `'mid'` or `'apical'` |
| `lv_center` | `(x, y)` | centre of the LV cavity, pixels |
| `rv_insertion_anterior` | `(x, y)` | anterior RV insertion point; anchors the AHA grid |
| `rv_insertion_inferior` | `(x, y)` or `None` | optional; only affects `rv_insertion_involved` |

Every feature is a ratio or a binary flag, so no pixel spacing is required.
Burdens are area ratios and are insensitive to resolution. The ray-sampled
quantities -- transmurality and the extent class -- are measured on the
discrete pixel grid and shift slightly with it.

## Input conventions

The features depend on these as much as on the masks, so match them if you
want values comparable with the paper.

* **Three slices, one per level.** The paper uses one representative basal,
  mid and apical short-axis slice, not the full stack.
* **Assigning the level.** Order the candidate slices by the effective radius
  of the myocardium (`sqrt(area / pi)`): largest is basal, smallest apical.
  This package does not infer the level — it is a required input, because the
  choice depends on the whole stack.
* **Exclude LVOT slices.** Basal slices containing left ventricular outflow
  tract should be dropped before extraction, not passed in.
* **Clean the myocardium mask.** Spurious disconnected components should be
  removed; the paper keeps components whose centroid lies within 0.8–1.2x the
  effective radius of the largest component.
* **Scar must lie within the myocardium.** `LGESlice` intersects the two masks
  for you, so scar outside the myocardium is dropped rather than counted.
* **Landmarks are required.** `lv_center` and `rv_insertion_anterior` must
  be supplied and must fall inside the image; both are validated and an
  invalid value raises. The anterior insertion point anchors the AHA grid, so
  the segment numbering follows from it.

## The 28 features

| Feature | Range | Definition |
|---|---|---|
| `scar_burden` | 0–1 | scar pixels / myocardial pixels, summed over the analysed slices |
| `mean_transmurality` | 0–1 | mean, over scar-bearing rays, of the fraction of wall thickness occupied by scar |
| `max_transmurality` | 0–1 | maximum of the same |
| `extent_endocardial`<br>`extent_mid_myocardial`<br>`extent_epicardial`<br>`extent_transmural` | 0/1 | **one-hot** encoding of the *dominant* extent class — the class seen on the most scar-bearing rays. Exactly one is set when at least one ray finds scar; all four are zero otherwise, including for a scar too small to register on any ray. |
| `wall_anterior`, `wall_septal`,<br>`wall_inferior`, `wall_lateral` | 0/1 | scar present in any AHA segment of that wall |
| `rv_insertion_involved` | 0/1 | scar within 3 px of either RV insertion point, after snapping the point to the myocardial boundary |
| `seg_1_burden` … `seg_16_burden` | 0–1 | scar fraction within each AHA segment |

The AHA grid is anchored at the anterior RV insertion point and runs
counter-clockwise: six 60° sectors for basal (1–6) and mid (7–12) slices, four
90° sectors for apical slices (13–16).

The extent features are a one-hot encoding of a single dominant class.
`dominant_extent()` returns the same information as a string.

## Plugging in your own models

```python
from lge_features import ScarSegmenter, LandmarkDetector, run_pipeline

class MySegmenter(ScarSegmenter):
    def segment(self, image):
        return myocardium_mask, scar_mask        # (H, W) bool each

class MyLandmarks(LandmarkDetector):
    def detect(self, image, myocardium):
        return {'lv_center': (x, y), 'rv_insertion_anterior': (x, y)}

features = run_pipeline(
    [('basal', img_b), ('mid', img_m), ('apical', img_a)],
    segmenter_for=lambda level: MySegmenter(),
    landmarks_for=lambda level: MyLandmarks(),
    out_dir='my_output',
)
```

`run_pipeline` writes:

```
my_output/
├── features.json            the 28 features
├── features.csv             the same, one header row and one data row
├── aha_<level>_mask.png     AHA labels: 0 = background, 1–16 = segment
├── layers_<level>_mask.png  wall layers: 0 = background, 1 = endocardial,
│                            2 = mid-myocardial, 3 = epicardial
└── overlay_<level>.png      myocardium, scar and AHA segments
```

The two mask files are indexed-colour PNGs: the stored pixel values are the
label numbers themselves, so `numpy.array(PIL.Image.open(path))` returns the
labels unchanged, while a viewer shows each label in a distinct colour.

The layer mask is provided for inspection. It measures depth through the
wall with a distance transform, which gives a smooth labelling independent of
any angular sampling.

## Tests

```bash
pytest tests/        # 31 property tests
```

The property tests state each feature's definition in executable form: no scar
gives all zeros, a full-thickness scar is transmural, a thin subendocardial rim
is not, rotating the RV insertion point shifts the segment labels by one, and
rescaling the phantom leaves every feature unchanged.

---

# Aim 2 — reproducing the published results

`reproduce/predictions/model{1,2,3}.csv` hold the held-out test-set class
probabilities of the three models (n=263), with patient identifiers replaced
by a row index. No feature values and no patient attributes are included.

```bash
python reproduce/reproduce_tables.py
```

regenerates Tables 3–4 (performance), Table S4 (confusion matrices), Table S5
(per-class AUC with DeLong p-values) and the model comparisons. The 16
headline values — accuracy and macro/micro/weighted AUC for each model, plus
Model 2's four per-class AUCs — are printed next to the published value and
checked automatically. The confusion matrices, bootstrap intervals and
p-values are printed for inspection but not checked.

---

## Citation

The accompanying manuscript is currently under review. Please cite it as:

```bibtex
@unpublished{jacob_lge_cmr,
  title  = {Automated Late Gadolinium Enhancement Quantification Improves
            Multi-Class Cardiac Disease Detection on CMR},
  author = {Jacob and Chitiboi and Sharma and Emrich and Rueckert and
            Varga-Szemes},
  note   = {Manuscript under review},
  year   = {2026}
}
```

This entry will be replaced with the journal reference once the manuscript is
published.

## License

Research and non-commercial use only; see [LICENSE](LICENSE). Commercial use
requires a separate written agreement with the copyright holder.

**Not a medical device.** Not cleared, approved or certified for clinical use,
and not to be used for any diagnostic, prognostic or treatment decision.
