"""Statistical tests used in the paper.

McNemar's test with continuity correction for accuracy, DeLong's test for
correlated ROC AUCs, Fisher's method to combine the per-class DeLong p-values
into one omnibus statement about macro AUC, and percentile bootstrap
confidence intervals.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

CLASSES = ['normal', 'dcm', 'hcm', 'ihd']   # label order 0, 1, 2, 3


# --------------------------------------------------------------- metrics ---

def macro_auc(y: np.ndarray, proba: np.ndarray) -> float:
    """Unweighted mean of the four one-vs-rest AUCs."""
    aucs = [roc_auc_score((y == c).astype(int), proba[:, c])
            for c in range(proba.shape[1])
            if len(np.unique(y == c)) > 1]
    return float(np.mean(aucs)) if aucs else float('nan')


def all_metrics(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Accuracy plus macro, micro, weighted and per-class AUC."""
    onehot = np.eye(proba.shape[1], dtype=int)[y]
    out = {
        'accuracy': accuracy_score(y, proba.argmax(1)),
        'macro_auc': macro_auc(y, proba),
        'micro_auc': roc_auc_score(onehot, proba, average='micro'),
        'weighted_auc': roc_auc_score(onehot, proba, multi_class='ovr',
                                      average='weighted'),
    }
    for c, name in enumerate(CLASSES):
        out[f'auc_{name}'] = roc_auc_score((y == c).astype(int), proba[:, c])
    return out


def confusion(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, proba.argmax(1))


# ---------------------------------------------------------------- DeLong ---

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_binary: np.ndarray,
                proba_a: np.ndarray,
                proba_b: np.ndarray) -> Tuple[float, float, float, float]:
    """Two-sided DeLong test for two correlated AUCs on the same cases.

    Returns (auc_a, auc_b, z, p).
    """
    order = np.argsort(-y_binary, kind='mergesort')
    y_binary = y_binary[order]
    preds = np.vstack([proba_a[order], proba_b[order]])

    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return (float('nan'),) * 4

    positive, negative = preds[:, :n_pos], preds[:, n_pos:]
    tx = np.vstack([_midrank(positive[r]) for r in range(2)])
    ty = np.vstack([_midrank(negative[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])

    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if variance <= 0:
        return aucs[0], aucs[1], float('nan'), float('nan')

    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def delong_per_class(y: np.ndarray,
                     proba_a: np.ndarray,
                     proba_b: np.ndarray) -> List[Dict]:
    """DeLong test for each class in turn, one versus rest."""
    rows = []
    for c, name in enumerate(CLASSES):
        y_binary = (y == c).astype(int)
        auc_a, auc_b, z, p = delong_test(y_binary, proba_a[:, c], proba_b[:, c])
        rows.append({'class': name, 'n': int(y_binary.sum()),
                     'auc_a': auc_a, 'auc_b': auc_b,
                     'delta': auc_b - auc_a, 'z': z, 'p': p})
    return rows


# --------------------------------------------------------------- McNemar ---

def mcnemar(y: np.ndarray,
            pred_a: np.ndarray,
            pred_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test with continuity correction.

    b = model A right where B is wrong; c = B right where A is wrong.
    """
    correct_a, correct_b = (pred_a == y), (pred_b == y)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': float('nan'), 'p': 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {'b': b, 'c': c, 'chi2': chi2, 'p': float(stats.chi2.sf(chi2, 1))}


# ---------------------------------------------------------------- Fisher ---

def fisher_combine(pvalues: Sequence[float]) -> float:
    """Fisher's combined probability test over the per-class p-values."""
    usable = [p for p in pvalues if np.isfinite(p) and p > 0]
    if not usable:
        return float('nan')
    chi2 = -2 * np.sum(np.log(usable))
    return float(stats.chi2.sf(chi2, 2 * len(usable)))


# ------------------------------------------------------------- bootstrap ---

def bootstrap_ci(y: np.ndarray,
                 proba: np.ndarray,
                 metric: str = 'accuracy',
                 n_boot: int = 1000,
                 seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for one metric."""
    rng = np.random.RandomState(seed)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < proba.shape[1]:
            continue
        try:
            scores.append(all_metrics(y[idx], proba[idx])[metric])
        except ValueError:
            continue
    if not scores:
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return (float(lo), float(hi))
