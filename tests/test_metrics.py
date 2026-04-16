"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from src.eval.metrics import compute_all


def test_compute_all_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    classes = ["a", "b", "c"]
    m = compute_all(y_true, y_pred, classes)

    assert "accuracy"    in m
    assert "f1_macro"    in m
    assert "f1_weighted" in m
    assert "per_class"   in m
    assert "confusion_matrix" in m


def test_perfect_accuracy():
    y = np.array([0, 1, 2])
    m = compute_all(y, y, ["a", "b", "c"])
    assert m["accuracy"]  == 1.0
    assert m["f1_macro"]  == 1.0


def test_roc_auc_with_scores():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    # Fake one-hot-ish softmax scores
    scores = np.eye(3)[y_true] * 0.8 + rng.dirichlet([1, 1, 1], size=6) * 0.2
    m = compute_all(y_true, y_pred, ["a", "b", "c"], y_score=scores)
    assert "roc_auc" in m
    assert m["roc_auc"] is not None
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_confusion_matrix_shape():
    n_classes = 4
    y = np.tile(np.arange(n_classes), 5)
    m = compute_all(y, y, [str(i) for i in range(n_classes)])
    assert m["confusion_matrix"].shape == (n_classes, n_classes)
