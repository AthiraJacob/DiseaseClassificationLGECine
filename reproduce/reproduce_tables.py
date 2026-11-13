"""Regenerate the paper's result tables from the released predictions.

    python reproduce/reproduce_tables.py

Reads reproduce/predictions/model{1,2,3}.csv and prints:

    Tables 3-4   held-out performance of the three models
    Table S4     confusion matrices
    Table S5     per-class AUC, Model 1 vs Model 2, with DeLong p-values
    Model comparisons  McNemar and Fisher-combined DeLong

The 16 headline values -- accuracy and macro/micro/weighted AUC for each of
the three models, plus Model 2's four per-class AUCs -- are printed next to
the published manuscript value and checked, so a mismatch is visible rather
than silent. The confusion matrices, bootstrap intervals and p-values are
printed for inspection but are not checked automatically.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats import (                                             # noqa: E402
    CLASSES,
    all_metrics,
    bootstrap_ci,
    confusion,
    delong_per_class,
    fisher_combine,
    mcnemar,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS = os.path.join(HERE, 'predictions')
TOLERANCE = 0.001

# Published values, for checking. Accuracy, macro, micro, weighted AUC.
PUBLISHED = {
    'model1': (0.760, 0.906, 0.929, 0.907),
    'model2': (0.817, 0.932, 0.948, 0.936),
    'model3': (0.829, 0.930, 0.946, 0.934),
}
PUBLISHED_MODEL2_PER_CLASS = {
    'normal': 0.961, 'dcm': 0.922, 'hcm': 0.938, 'ihd': 0.907,
}

LABELS = {
    'model1': 'Model 1  cine + clinical       (97 features)',
    'model2': 'Model 2  + summary LGE        (109 features)',
    'model3': 'Model 3  + segmental LGE      (125 features)',
}


def load(name):
    """Read one predictions CSV. Returns (case_ids, y, proba)."""
    path = os.path.join(PREDICTIONS, f'{name}.csv')
    case_ids, y, proba = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            case_ids.append(int(row['case_id']))
            y.append(CLASSES.index(row['true_label']))
            proba.append([float(row[f'p_{c}']) for c in CLASSES])
    return np.array(case_ids), np.array(y), np.array(proba)


def check(label, got, expected, failures):
    """Compare a value with the published one and record any mismatch."""
    ok = abs(got - expected) <= TOLERANCE
    if not ok:
        failures.append((label, got, expected))
    return 'ok' if ok else 'MISMATCH'


def main():
    loaded = {name: load(name) for name in ('model1', 'model2', 'model3')}

    # The three files describe the same patients in the same order; the paired
    # McNemar and DeLong tests below are only valid if that holds.
    reference_ids, reference_y, _ = loaded['model1']
    for name, (case_ids, y_other, _) in loaded.items():
        if not np.array_equal(case_ids, reference_ids):
            raise SystemExit(f'{name}.csv: case_id column is not aligned with '
                             'model1.csv; paired tests would be invalid')
        if not np.array_equal(y_other, reference_y):
            raise SystemExit(f'{name}.csv: true_label disagrees with model1.csv')

    models = {name: (y, proba) for name, (_, y, proba) in loaded.items()}
    y = reference_y
    failures = []

    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    print('=' * 72)
    print(f'  Held-out test set: n={len(y)}  {counts}')
    print('=' * 72)

    # ---- Tables 3-4 -------------------------------------------------------
    print('\nTABLES 3-4  Held-out performance')
    print('-' * 72)
    print('  "got" is recomputed here; "pub" is the published manuscript value.')
    print(f"\n{'':<34}{'accuracy':>18}{'macro AUC':>18}"
          f"{'micro AUC':>18}{'weighted AUC':>18}")
    print(f"{'':<34}" + ''.join(f"{'got':>9}{'pub':>9}" for _ in range(4)))
    for name, (yy, proba) in models.items():
        m = all_metrics(yy, proba)
        got = (m['accuracy'], m['macro_auc'], m['micro_auc'], m['weighted_auc'])
        cells = ''
        for value, expected in zip(got, PUBLISHED[name]):
            cells += f'{value:>9.4f}{expected:>9.3f}'
        print(f'  {LABELS[name]:<32}{cells}')
        for value, expected, metric in zip(got, PUBLISHED[name],
                                           ['accuracy', 'macro', 'micro', 'weighted']):
            check(f'{name} {metric}', value, expected, failures)

    # ---- 95% CIs ----------------------------------------------------------
    print('\n95% percentile bootstrap CIs (1,000 resamples)')
    print('-' * 72)
    for name, (yy, proba) in models.items():
        acc_lo, acc_hi = bootstrap_ci(yy, proba, 'accuracy')
        auc_lo, auc_hi = bootstrap_ci(yy, proba, 'macro_auc')
        m = all_metrics(yy, proba)
        print(f'  {name}  accuracy {m["accuracy"]:.3f} '
              f'({acc_lo:.3f}-{acc_hi:.3f})   '
              f'macro AUC {m["macro_auc"]:.3f} ({auc_lo:.3f}-{auc_hi:.3f})')

    # ---- Table S4 ---------------------------------------------------------
    print('\nTABLE S4  Confusion matrices (rows = true, columns = predicted)')
    print('-' * 72)
    for name, (yy, proba) in models.items():
        print(f'  {name}      ' + ''.join(f'{c:>9}' for c in CLASSES))
        for i, row in enumerate(confusion(yy, proba)):
            print(f'    {CLASSES[i]:<8}' + ''.join(f'{v:>9d}' for v in row))
        print()

    # ---- Table S5 ---------------------------------------------------------
    print('TABLE S5  Per-class AUC, Model 1 -> Model 2 (DeLong)')
    print('-' * 72)
    y1, p1 = models['model1']
    _, p2 = models['model2']
    rows = delong_per_class(y1, p1, p2)
    print(f"{'  class':<10}{'n':>5}{'Model 1':>10}{'Model 2':>10}"
          f"{'(pub)':>8}{'delta':>9}{'p':>10}")
    for r in rows:
        published = PUBLISHED_MODEL2_PER_CLASS[r['class']]
        print(f"  {r['class']:<8}{r['n']:>5}{r['auc_a']:>10.4f}"
              f"{r['auc_b']:>10.4f}{published:>8.3f}"
              f"{r['delta']:>+9.3f}{r['p']:>10.4f}")
        check(f"model2 auc_{r['class']}", r['auc_b'], published, failures)
    combined = fisher_combine([r['p'] for r in rows])
    print(f'\n  Fisher combined p = {combined:.3g}')

    # ---- model comparisons ------------------------------------------------
    print('\nMODEL COMPARISONS (paired, same patients)')
    print('-' * 72)
    for a, b in (('model1', 'model2'), ('model2', 'model3')):
        ya, pa = models[a]
        _, pb = models[b]
        mc = mcnemar(ya, pa.argmax(1), pb.argmax(1))
        per_class = delong_per_class(ya, pa, pb)
        combined = fisher_combine([r['p'] for r in per_class])
        acc_a = all_metrics(ya, pa)['accuracy']
        acc_b = all_metrics(ya, pb)['accuracy']
        print(f'  {a} -> {b}')
        print(f'    accuracy  {acc_a:.3f} -> {acc_b:.3f} ({acc_b - acc_a:+.3f})'
              f"   McNemar b={mc['b']} c={mc['c']} p={mc['p']:.4f}")
        print(f'    macro AUC {all_metrics(ya, pa)["macro_auc"]:.3f} -> '
              f'{all_metrics(ya, pb)["macro_auc"]:.3f}'
              f'   Fisher combined p={combined:.3g}')

    # ---- verdict ----------------------------------------------------------
    print('\n' + '=' * 72)
    if failures:
        print(f'  FAIL - {len(failures)} value(s) differ from the manuscript '
              f'by more than {TOLERANCE}:')
        for label, got, expected in failures:
            print(f'    {label:<30} got {got:.4f}  published {expected:.3f}')
        return 1
    print('  PASS - every checked value matches the published manuscript')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
