"""
Few-shot learning experiment.
Tests all (loss, augmentation) variants at k=5, 10, 20 samples per class.

Run: python wafer_fewshot_focal_experiment.py [--config config.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import get_eval_transforms, get_train_transforms
from src.data.loader import clean, extract_images, load_raw, split_data
from src.eval import metrics as eval_metrics
from src.eval.plots import fewshot_comparison
from src.models.cnn import build_model
from src.training.losses import build_loss
from src.training.trainer import WaferDataset, evaluate, train
from src.utils import ensure_output_dirs, load_config, resolve_device, set_seed


def sample_few_shot(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample exactly k examples per class (stratified)."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
        idx.extend(chosen)
    idx = np.array(idx)
    return X[idx], y[idx]


def run(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])
    ensure_output_dirs(cfg)
    device = resolve_device(cfg["training"]["device"])

    fig_dir = Path(cfg["outputs"]["figures_dir"])
    res_dir = Path(cfg["outputs"]["results_dir"])
    model_dir = Path(cfg["outputs"]["models_dir"])

    # ── Data (full split — test set stays fixed) ───────────────────────────
    df = clean(load_raw(cfg["data"]["path"]), cfg["preprocessing"]["invalid_labels"])
    X, y = extract_images(df, image_size=cfg["data"]["image_size"],
                          interpolation=cfg["preprocessing"]["resize_interpolation"])
    split = split_data(X, y,
                       test_size=cfg["data"]["test_size"],
                       val_size=cfg["data"]["val_size"],
                       seed=cfg["data"]["seed"])
    le = split["label_encoder"]
    num_classes = len(le.classes_)
    image_size = cfg["data"]["image_size"]
    train_cfg = cfg["training"]

    test_ds = WaferDataset(
        split["X_test"], split["y_test"],
        transform=get_eval_transforms(image_size),
    )
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=0)

    records = []

    for k in cfg["few_shot"]["k_values"]:
        X_few, y_few = sample_few_shot(split["X_train"], split["y_train"],
                                       k=k, seed=cfg["training"]["seed"])
        # Use the few-shot set as both train and a tiny val (last 20%)
        n_val = max(1, int(len(X_few) * 0.2))
        X_tr, X_vl = X_few[:-n_val], X_few[-n_val:]
        y_tr, y_vl = y_few[:-n_val], y_few[-n_val:]

        for variant_cfg in cfg["few_shot"]["variants"]:
            name   = variant_cfg["name"]
            loss   = variant_cfg["loss"]
            augment = variant_cfg["augmentation"]

            print(f"\n[fewshot] k={k}  variant={name}  loss={loss}  aug={augment}")

            tr_ds = WaferDataset(X_tr, y_tr,
                                 transform=get_train_transforms(image_size, augment=augment))
            vl_ds = WaferDataset(X_vl, y_vl,
                                 transform=get_eval_transforms(image_size))

            tr_loader = DataLoader(tr_ds, batch_size=min(train_cfg["batch_size"], len(tr_ds)),
                                   shuffle=True, num_workers=0)
            vl_loader = DataLoader(vl_ds, batch_size=min(train_cfg["batch_size"], len(vl_ds)),
                                   shuffle=False, num_workers=0)

            model = build_model(num_classes,
                                {**cfg["cnn"], "image_size": image_size}).to(device)
            criterion = build_loss(loss, y_tr, device, cfg["losses"]["focal"]["gamma"])
            optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

            model_path = model_dir / f"fewshot_k{k}_{name}.pt"
            train(model, tr_loader, vl_loader, criterion, optimizer, device,
                  epochs=train_cfg["epochs"],
                  patience=train_cfg["early_stopping_patience"],
                  model_path=model_path)

            model.load_state_dict(torch.load(model_path, map_location=device))
            _, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)

            m = eval_metrics.compute_all(y_true, y_pred, le.classes_.tolist())
            eval_metrics.print_summary(m, title=f"k={k} | {name}")

            records.append({
                "k":           k,
                "variant":     name,
                "accuracy":    m["accuracy"],
                "f1_macro":    m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
            })

    # ── Save & plot results ────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(res_dir / "fewshot_summary.csv", index=False)
    print(f"\n[fewshot] Summary saved → {res_dir / 'fewshot_summary.csv'}")
    print(results_df.to_string(index=False))

    fewshot_comparison(results_df, save_path=fig_dir / "fewshot_comparison.png")
    print(f"[fewshot] Plot saved   → {fig_dir / 'fewshot_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
