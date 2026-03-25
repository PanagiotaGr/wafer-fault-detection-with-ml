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
from torch.utils.data import Dataset, DataLoader


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"
FIG_DIR = "outputs/figures"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return "none"
        return str(x[0]).strip().lower()
    if pd.isna(x):
        return "none"
    return str(x).strip().lower()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "waferMap" not in df.columns:
        raise ValueError("Column 'waferMap' not found.")
    if "failureType" not in df.columns:
        raise ValueError("Column 'failureType' not found.")

    df["failureType_clean"] = df["failureType"].apply(simplify_failure_type)

    print("\n[INFO] Raw class distribution:")
    print(df["failureType_clean"].value_counts(dropna=False))

    invalid_labels = {"none", "[]", "", "nan"}
    df = df[~df["failureType_clean"].isin(invalid_labels)].copy()

    print("\n[INFO] Clean class distribution:")
    print(df["failureType_clean"].value_counts())

    df = df.reset_index(drop=True)
    return df


def plot_class_distribution(df):
    counts = df["failureType_clean"].value_counts()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cnn_class_distribution.png"), dpi=200)
    plt.close()


def prepare_images_and_labels(df, image_size=64):
    images = []
    labels = []

    print("\n[INFO] Preparing images...")
    for _, row in df.iterrows():
        img = np.array(row["waferMap"], dtype=np.uint8)
        label = row["failureType_clean"]

        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)

        # normalize to [0, 1]
        if img.max() > 0:
            img = img / img.max()

        images.append(img)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    print("[INFO] Images shape:", images.shape)
    print("[INFO] Labels shape:", labels.shape)
    return images, labels


class WaferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # [N,1,H,W]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class SimpleWaferCNN(nn.Module):
    def __init__(self, num_classes):
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
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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


def plot_training_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cnn_loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cnn_accuracy_curve.png"), dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_dataset(df)
    plot_class_distribution(df)

    images, labels = prepare_images_and_labels(df, image_size=IMAGE_SIZE)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels_encoded,
        test_size=0.3,
        random_state=SEED,
        stratify=labels_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp
    )

    print("\n[INFO] Train:", X_train.shape, y_train.shape)
    print("[INFO] Val  :", X_val.shape, y_val.shape)
    print("[INFO] Test :", X_test.shape, y_test.shape)

    train_ds = WaferDataset(X_train, y_train)
    val_ds = WaferDataset(X_val, y_val)
    test_ds = WaferDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimpleWaferCNN(num_classes=len(le.classes_)).to(DEVICE)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_wafer_cnn.pt")

    print(f"\n[INFO] Training on device: {DEVICE}")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\n[INFO] Best validation accuracy: {best_val_acc:.4f}")
    plot_training_curves(history)

    # load best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n[RESULT] Test Loss: {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}")

    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n[RESULT] Classification Report:\n")
    print(report)

    print("\n[RESULT] Confusion Matrix:\n")
    print(cm)

    with open(os.path.join(RESULTS_DIR, "cnn_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    np.savetxt(
        os.path.join(RESULTS_DIR, "cnn_confusion_matrix.csv"),
        cm,
        delimiter=",",
        fmt="%d"
    )

    print("\n[INFO] CNN pipeline completed successfully.")
    print(f"[INFO] Model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
