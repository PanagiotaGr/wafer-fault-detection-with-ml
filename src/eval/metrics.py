"""
Comprehensive evaluation metrics.
Replaces the accuracy-only reporting in the original pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray | None = None,
) -> dict:
    """
    Compute a full metrics report.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Human-readable class names in label-encoder order.
        y_score:     Softmax probabilities [N, C] — required for ROC-AUC.

    Returns:
        Dict with keys: accuracy, f1_macro, f1_weighted, roc_auc (optional),
        per_class (DataFrame), confusion_matrix (ndarray).
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report_dict).T.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )

    result = {
        "accuracy":    round(acc, 6),
        "f1_macro":    round(f1_macro, 6),
        "f1_weighted": round(f1_weighted, 6),
        "per_class":   per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_score is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            roc_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
            result["roc_auc"] = round(roc_auc, 6)
        except Exception:
            result["roc_auc"] = None

    return result


def print_summary(metrics: dict, title: str = "Evaluation") -> None:
    """Print a readable summary table."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)   : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"\nPer-class breakdown:")
    print(metrics["per_class"].to_string())
    print(f"{'─' * 50}\n")


def save(metrics: dict, output_dir: Path, prefix: str = "") -> None:
    """Save metrics to CSV and JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{prefix}_" if prefix else ""

    # Summary JSON
    summary = {k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}
    with open(output_dir / f"{tag}summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-class CSV
    metrics["per_class"].to_csv(output_dir / f"{tag}per_class.csv")

    # Confusion matrix CSV
    np.savetxt(
        output_dir / f"{tag}confusion_matrix.csv",
        metrics["confusion_matrix"],
        delimiter=",",
        fmt="%d",
    )

    print(f"[eval] Metrics saved to {output_dir} (prefix='{tag}')")
