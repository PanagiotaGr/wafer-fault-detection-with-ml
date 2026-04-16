"""
Visualization helpers.
All plot functions save to disk and optionally return the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ── Individual plot functions ─────────────────────────────────────────────────

def class_distribution(labels: np.ndarray, save_path: Path | None = None) -> plt.Figure:
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(unique, counts, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Class distribution")
    ax.set_xlabel("Defect class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    # Normalise by row (true labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(len(class_names) + 2, len(class_names) + 1))
    sns.heatmap(
        cm_norm,
        annot=cm,        # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def training_curves(history: dict, save_path: Path | None = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def fewshot_comparison(
    results: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    results: DataFrame with columns [k, variant, accuracy, f1_macro, f1_weighted]
    """
    variants = results["variant"].unique()
    k_values = sorted(results["k"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    titles  = ["Accuracy", "F1 (macro)", "F1 (weighted)"]

    for ax, metric, title in zip(axes, metrics, titles):
        for var in variants:
            sub = results[results["variant"] == var].sort_values("k")
            ax.plot(sub["k"], sub[metric], marker="o", label=var)
        ax.set_title(title)
        ax.set_xlabel("Samples per class (k)")
        ax.set_xticks(k_values)
        ax.legend(fontsize=8)

    fig.suptitle("Few-shot learning: performance vs. data scarcity", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def model_comparison_bar(
    results: dict[str, dict],
    metric: str = "accuracy",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing multiple models on a single metric.

    results: { model_name: {metric: value, ...} }
    """
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    colors = ["#4C72B0" if v < max(values) else "#2ecc71" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_xlim(0, min(1.05, max(values) * 1.15))
    ax.set_title(f"Model comparison — {metric}")
    ax.set_xlabel(metric.replace("_", " ").title())
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def wafer_sample_grid(
    df,
    samples_per_class: int = 2,
    save_path: Path | None = None,
) -> plt.Figure:
    classes = df["label"].unique()
    rows, cols = len(classes), samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))

    for r, cls in enumerate(classes):
        samples = df[df["label"] == cls].head(cols)
        for c, (_, row) in enumerate(samples.iterrows()):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.imshow(np.array(row["waferMap"]), cmap="gray")
            ax.set_title(cls if c == 0 else "", fontsize=8)
            ax.axis("off")

    fig.suptitle("Sample wafer maps per class", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
