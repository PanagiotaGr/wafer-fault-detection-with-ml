"""
Classical ML pipeline (Logistic Regression, SVM, Random Forest).

All heavy lifting lives in src/ — this file is just the orchestrator.
Run: python wafer_pipeline.py [--config config.yaml]
"""

import argparse
from pathlib import Path

import numpy as np

from src.data.loader import clean, extract_images, load_raw, split_data
from src.eval import metrics as eval_metrics
from src.eval.plots import (
    class_distribution,
    confusion_matrix_heatmap,
    model_comparison_bar,
    wafer_sample_grid,
)
from src.models.classical import ALL_MODELS
from src.utils import ensure_output_dirs, load_config, set_seed


def run(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])
    ensure_output_dirs(cfg)

    fig_dir = Path(cfg["outputs"]["figures_dir"])
    res_dir = Path(cfg["outputs"]["results_dir"])

    # ── Data ─────────────────────────────────────────────────────────────────
    df = load_raw(cfg["data"]["path"])
    df = clean(df, cfg["preprocessing"]["invalid_labels"])

    wafer_sample_grid(df, save_path=fig_dir / "sample_wafer_maps.png")
    class_distribution(df["label"].values, save_path=fig_dir / "class_distribution.png")

    X, y = extract_images(
        df,
        image_size=cfg["data"]["image_size"],
        interpolation=cfg["preprocessing"]["resize_interpolation"],
    )

    split = split_data(
        X.reshape(len(X), -1),   # flatten for sklearn
        y,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["data"]["seed"],
    )
    le = split["label_encoder"]
    X_train = np.vstack([split["X_train"], split["X_val"]])
    y_train = np.concatenate([split["y_train"], split["y_val"]])
    X_test, y_test = split["X_test"], split["y_test"]

    # ── Train & evaluate each model ────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for name, build_fn in ALL_MODELS.items():
        print(f"\n[pipeline] Training {name} …")
        model = build_fn(cfg["classical_ml"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = eval_metrics.compute_all(y_test, y_pred, le.classes_.tolist())
        eval_metrics.print_summary(m, title=name)
        eval_metrics.save(m, res_dir, prefix=name)

        confusion_matrix_heatmap(
            m["confusion_matrix"],
            le.classes_.tolist(),
            title=f"Confusion matrix — {name}",
            save_path=fig_dir / f"{name}_confusion_matrix.png",
        )
        all_results[name] = m

    # ── Summary bar chart ─────────────────────────────────────────────────
    for metric in ("accuracy", "f1_macro", "f1_weighted"):
        model_comparison_bar(
            all_results, metric=metric,
            save_path=fig_dir / f"classical_comparison_{metric}.png",
        )

    print("\n[pipeline] Classical ML pipeline complete.")
    print(f"  Results → {res_dir}")
    print(f"  Figures → {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
