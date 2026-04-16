"""
Generic training loop for PyTorch models.
Used by both the CNN pipeline and the few-shot experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ── Dataset ──────────────────────────────────────────────────────────────────

class WaferDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Callable | None = None,
    ) -> None:
        # images: [N, H, W] float32 → add channel dim for CNN
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]                  # [H, W] float32 in [0, 1]
        # Convert to uint8 for torchvision transforms, then back
        img_u8 = (img * 255).astype(np.uint8)

        if self.transform is not None:
            tensor = self.transform(img_u8)     # transform handles ToTensor()
        else:
            tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return tensor, self.labels[idx]


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Single pass. If optimizer is None, runs in eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_targets = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    n = len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return total_loss / n, accuracy_score(all_targets, all_preds), all_targets, all_preds


# ── Main trainer ──────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    patience: int = 5,
    model_path: Path | None = None,
) -> dict:
    """
    Full training loop with early stopping.
    Returns a history dict and saves the best checkpoint if model_path given.
    """
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = _run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val   loss={vl_loss:.4f} acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    history["best_val_acc"] = best_val_acc
    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run a single evaluation pass. Returns (loss, acc, y_true, y_pred)."""
    loss, acc, y_true, y_pred = _run_epoch(model, loader, criterion, None, device)
    return loss, acc, y_true, y_pred
