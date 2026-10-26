"""Group-level evaluation metrics.

TODO (you): implement these. Your DSM assignment 3 `evaluate` function already
computes per-group, balanced and worst-group accuracy.
"""
import numpy as np


def group_report(y_true, y_pred, groups):
    """Per-group metrics as a pandas DataFrame, one row per group.

    Suggested columns: n, accuracy, selection rate (share predicted >50K),
    true positive rate, false positive rate.
    """
    raise NotImplementedError


def balanced_accuracy_over_groups(y_true, y_pred, groups):
    """Mean of the per-group accuracies (each group counts equally)."""
    raise NotImplementedError


def worst_group_accuracy(y_true, y_pred, groups):
    """Lowest per-group accuracy."""
    raise NotImplementedError


def demographic_parity_difference(y_pred, groups):
    """Largest gap in selection rate between any two groups."""
    raise NotImplementedError


def equal_opportunity_difference(y_true, y_pred, groups):
    """Largest gap in true positive rate between any two groups."""
    raise NotImplementedError
