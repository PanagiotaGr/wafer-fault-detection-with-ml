"""
CNN training pipeline (baseline, weighted loss, focal loss, focal+augmentation).

Run: python wafer_cnn_pipeline.py [--config config.yaml] [--variant baseline]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.augmentation import get_eval_transforms, get_train_transforms
from src.data.loader import clean, extract_images, load_raw, split_data
from src.eval import metrics as eval_metrics
from src.eval.plots import (
    class_distribution,
    confusion_matrix_heatmap,
    model_comparison_bar,
    training_curves,
)
from src.models.cnn import build_model
from src.training.losses import build_loss
from src.training.trainer import WaferDataset, evaluate, train
from src.utils import ensure_output_dirs, load_config, resolve_device, set_seed


VARIANT_LOSS = {
    "baseline": ("cross_entropy",  False),
    "weighted": ("weighted_cross_entropy", False),
    "focal":    ("focal",          False),
    "focal_aug":("focal",          True),
}


def run_variant(
    variant: str,
    split: dict,
    cfg: dict,
    device: torch.device,
    fig_dir: Path,
    res_dir: Path,
    model_dir: Path,
) -> dict:
    loss_name, augment = VARIANT_LOSS[variant]
    image_size = cfg["data"]["image_size"]
    train_cfg  = cfg["training"]
    le         = split["label_encoder"]
    num_classes = len(le.classes_)

    print(f"\n{'═' * 55}")
    print(f"  Variant: {variant}  |  loss: {loss_name}  |  aug: {augment}")
    print(f"{'═' * 55}")

    # Datasets
    train_ds = WaferDataset(
        split["X_train"], split["y_train"],
        transform=get_train_transforms(image_size, augment=augment),
    )
    val_ds = WaferDataset(
        split["X_val"], split["y_val"],
        transform=get_eval_transforms(image_size),
    )
    test_ds = WaferDataset(
        split["X_test"], split["y_test"],
        transform=get_eval_transforms(image_size),
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    model = build_model(num_classes, {**cfg["cnn"], "image_size": image_size}).to(device)
    criterion = build_loss(loss_name, split["y_train"], device, cfg["losses"]["focal"]["gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    model_path = model_dir / f"cnn_{variant}.pt"
    history = train(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs=train_cfg["epochs"],
        patience=train_cfg["early_stopping_patience"],
        model_path=model_path,
    )

    training_curves(history, save_path=fig_dir / f"cnn_{variant}_curves.png")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    _, _, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    m = eval_metrics.compute_all(y_true, y_pred, le.classes_.tolist())
    eval_metrics.print_summary(m, title=f"CNN {variant} — test set")
    eval_metrics.save(m, res_dir, prefix=f"cnn_{variant}")

    confusion_matrix_heatmap(
        m["confusion_matrix"],
        le.classes_.tolist(),
        title=f"Confusion matrix — CNN {variant}",
        save_path=fig_dir / f"cnn_{variant}_confusion_matrix.png",
    )
    return m


def run(cfg: dict, variants: list[str] | None = None) -> None:
    set_seed(cfg["training"]["seed"])
    ensure_output_dirs(cfg)
    device = resolve_device(cfg["training"]["device"])
    print(f"[pipeline] Device: {device}")

    fig_dir   = Path(cfg["outputs"]["figures_dir"])
    res_dir   = Path(cfg["outputs"]["results_dir"])
    model_dir = Path(cfg["outputs"]["models_dir"])

    # ── Data ──────────────────────────────────────────────────────────────
    df = clean(load_raw(cfg["data"]["path"]), cfg["preprocessing"]["invalid_labels"])
    class_distribution(df["label"].values, save_path=fig_dir / "class_distribution.png")

    X, y = extract_images(df, image_size=cfg["data"]["image_size"],
                          interpolation=cfg["preprocessing"]["resize_interpolation"])
    split = split_data(X, y,
                       test_size=cfg["data"]["test_size"],
                       val_size=cfg["data"]["val_size"],
                       seed=cfg["data"]["seed"])

    # ── Train each CNN variant ─────────────────────────────────────────────
    if variants is None:
        variants = list(VARIANT_LOSS.keys())

    all_results: dict[str, dict] = {}
    for variant in variants:
        all_results[variant] = run_variant(
            variant, split, cfg, device, fig_dir, res_dir, model_dir
        )

    # ── Summary ───────────────────────────────────────────────────────────
    for metric in ("accuracy", "f1_macro", "f1_weighted"):
        model_comparison_bar(
            all_results, metric=metric,
            save_path=fig_dir / f"cnn_comparison_{metric}.png",
        )

    print("\n[pipeline] CNN pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_LOSS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()
    variants = None if args.variant == "all" else [args.variant]
    run(load_config(args.config), variants=variants)
