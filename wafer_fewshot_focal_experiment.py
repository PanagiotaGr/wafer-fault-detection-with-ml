import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"
FIG_DIR = "outputs/figures"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEWSHOT_VALUES = [5, 10, 20]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_pickle(path)
    print("[INFO] Loaded shape:", df.shape)
    print("[INFO] Columns:", df.columns.tolist())
    return df


def simplify_failure_type(x):
    if pd.isna(x):
        return "none"

    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return "none"
        return str(x[0]).strip().lower()

    s = str(x).strip().lower()

    # string τύπου "['edge-ring']"
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        s = s.strip("'").strip('"').strip()

    return s


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "waferMap" not in df.columns:
        raise ValueError("Column 'waferMap' not found.")
    if "failureType" not in df.columns:
        raise ValueError("Column 'failureType' not found.")

    df["failureType_clean"] = df["failureType"].apply(simplify_failure_type)

    print("\n[INFO] Raw class distribution:")
    print(df["failureType_clean"].value_counts(dropna=False))

    invalid_labels = {"none", "[]", "", "nan", "'none'", '"none"'}
    df = df[~df["failureType_clean"].isin(invalid_labels)].copy()

    df = df.reset_index(drop=True)

    print("\n[INFO] Clean class distribution:")
    print(df["failureType_clean"].value_counts())
    return df


def create_fewshot_dataset(df: pd.DataFrame, samples_per_class=10) -> pd.DataFrame:
    parts = []
    classes = sorted(df["failureType_clean"].unique())

    for cls in classes:
        cls_df = df[df["failureType_clean"] == cls]
        take_n = min(samples_per_class, len(cls_df))
        sampled = cls_df.sample(n=take_n, random_state=SEED)
        parts.append(sampled)

    few_df = pd.concat(parts).reset_index(drop=True)
    return few_df


def plot_class_distribution(df: pd.DataFrame, suffix="full"):
    counts = df["failureType_clean"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(f"Class Distribution ({suffix})")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"class_distribution_{suffix}.png"), dpi=200)
    plt.close()


def prepare_images_and_labels(df, image_size=64):
    images = []
    labels = []

    for _, row in df.iterrows():
        img = np.array(row["waferMap"], dtype=np.uint8)
        label = row["failureType_clean"]

        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)

        if img.max() > 0:
            img = img / img.max()

        images.append(img)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    return images, labels


def augment_wafer(img):
    out = img.copy()

    if random.random() < 0.5:
        out = np.fliplr(out)

    if random.random() < 0.5:
        out = np.flipud(out)

    k = random.choice([0, 1, 2, 3])
    out = np.rot90(out, k).copy()

    if random.random() < 0.3:
        noise = np.random.normal(0, 0.03, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    return out.astype(np.float32)


class WaferDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            img = augment_wafer(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


class SimpleWaferCNN(nn.Module):
    def __init__(self, num_classes, image_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        feat_size = image_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_size * feat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)


def save_confusion_matrix_plot(cm, classes, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def safe_split(images, labels_encoded):
    class_counts = np.bincount(labels_encoded)
    min_class_count = class_counts.min()

    if min_class_count < 6:
        # Για πολύ μικρό few-shot: train/test μόνο
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels_encoded,
            test_size=0.4,
            random_state=SEED,
            stratify=labels_encoded
        )
        X_val, y_val = X_test.copy(), y_test.copy()
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            images,
            labels_encoded,
            test_size=0.3,
            random_state=SEED,
            stratify=labels_encoded
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=SEED,
            stratify=y_temp
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_single_experiment(df_full: pd.DataFrame, k: int):
    print(f"\n{'='*60}")
    print(f"[INFO] FEW-SHOT EXPERIMENT: {k} samples/class")
    print(f"{'='*60}")

    df_k = create_fewshot_dataset(df_full, samples_per_class=k)
    plot_class_distribution(df_k, suffix=f"fewshot_{k}")

    images, labels = prepare_images_and_labels(df_k, image_size=IMAGE_SIZE)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    if len(np.unique(labels_encoded)) < 2:
        raise ValueError("Not enough classes after few-shot sampling.")

    X_train, X_val, X_test, y_train, y_val, y_test = safe_split(images, labels_encoded)

    train_ds = WaferDataset(X_train, y_train, augment=True)
    val_ds = WaferDataset(X_val, y_val, augment=False)
    test_ds = WaferDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimpleWaferCNN(num_classes=len(le.classes_), image_size=IMAGE_SIZE).to(DEVICE)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, f"best_fewshot_{k}.pt")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"[k={k}] Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[RESULT][k={k}] Test Accuracy: {test_acc:.4f}")
    print(report)

    with open(os.path.join(RESULTS_DIR, f"fewshot_{k}_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Few-shot setting: {k} samples/class\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    np.savetxt(
        os.path.join(RESULTS_DIR, f"fewshot_{k}_confusion_matrix.csv"),
        cm,
        delimiter=",",
        fmt="%d"
    )

    save_confusion_matrix_plot(
        cm,
        le.classes_,
        os.path.join(FIG_DIR, f"fewshot_{k}_confusion_matrix.png")
    )

    return {
        "samples_per_class": k,
        "num_total_samples": len(df_k),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "num_classes": len(le.classes_)
    }


def plot_summary(results_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["samples_per_class"], results_df["test_acc"], marker="o")
    plt.xlabel("Samples per class")
    plt.ylabel("Test Accuracy")
    plt.title("Few-Shot Performance vs Samples per Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fewshot_accuracy_comparison.png"), dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_dataset(df)

    results = []

    for k in FEWSHOT_VALUES:
        res = run_single_experiment(df, k)
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "fewshot_summary.csv"), index=False)

    print("\n[INFO] Few-shot summary:")
    print(results_df)

    plot_summary(results_df)

    print("\n[INFO] Few-shot experiment completed successfully.")
    print(f"[INFO] Summary saved to: {os.path.join(RESULTS_DIR, 'fewshot_summary.csv')}")


if __name__ == "__main__":
    main()
