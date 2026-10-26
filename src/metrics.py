"""Group-level evaluation metrics."""
import numpy as np
import pandas as pd


def _arrays(*args):
    return [np.asarray(a).ravel() for a in args]


def group_report(y_true, y_pred, groups):
    """Per-group metrics as a DataFrame, one row per group.

    Columns: n, base rate (share truly >50K), accuracy, selection rate (share
    predicted >50K), true positive rate and false positive rate.
    """
    y_true, y_pred, groups = _arrays(y_true, y_pred, groups)
    rows = {}
    for g in pd.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        rows[g] = {
            "n": int(m.sum()),
            "base rate": yt.mean(),
            "accuracy": (yt == yp).mean(),
            "selection rate": yp.mean(),
            "TPR": yp[yt == 1].mean() if (yt == 1).any() else np.nan,
            "FPR": yp[yt == 0].mean() if (yt == 0).any() else np.nan,
        }
    return pd.DataFrame(rows).T.sort_values("n", ascending=False)


def balanced_accuracy_over_groups(y_true, y_pred, groups):
    """Mean of the per-group accuracies (each group counts equally)."""
    return group_report(y_true, y_pred, groups)["accuracy"].mean()


def worst_group_accuracy(y_true, y_pred, groups):
    """Lowest per-group accuracy."""
    return group_report(y_true, y_pred, groups)["accuracy"].min()


def demographic_parity_difference(y_pred, groups):
    """Largest gap in selection rate between any two groups."""
    y_pred, groups = _arrays(y_pred, groups)
    rates = [y_pred[groups == g].mean() for g in pd.unique(groups)]
    return max(rates) - min(rates)


def equal_opportunity_difference(y_true, y_pred, groups):
    """Largest gap in true positive rate between any two groups."""
    tpr = group_report(y_true, y_pred, groups)["TPR"].dropna()
    return tpr.max() - tpr.min()


def bootstrap_ci(metric_fn, *arrays, n_boot=1000, alpha=0.05, random_state=0):
    """Percentile bootstrap confidence interval for metric_fn(*arrays)."""
    arrays = _arrays(*arrays)
    n = len(arrays[0])
    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(*[a[idx] for a in arrays]))
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])
