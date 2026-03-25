import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "outputs/results"
FIG_DIR = "outputs/figures"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_pickle(path)
    print("[INFO] Dataset loaded successfully.")
    print("[INFO] Shape:", df.shape)
    print("[INFO] Columns:", df.columns.tolist())
    return df


def simplify_failure_type(x):
    """
    Το failureType στο WM-811K συχνά είναι list/array/string.
    Εδώ το κάνουμε καθαρό string label.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return "none"
        return str(x[0]).strip().lower()
    if pd.isna(x):
        return "none"
    return str(x).strip().lower()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "failureType" not in df.columns:
        raise ValueError("Δεν βρέθηκε η στήλη 'failureType' στο dataset.")

    if "waferMap" not in df.columns:
        raise ValueError("Δεν βρέθηκε η στήλη 'waferMap' στο dataset.")

    df["failureType_clean"] = df["failureType"].apply(simplify_failure_type)

    print("\n[INFO] Raw class distribution:")
    print(df["failureType_clean"].value_counts(dropna=False))

    invalid_labels = {"none", "[]", "", "nan"}
    df = df[~df["failureType_clean"].isin(invalid_labels)].copy()

    print("\n[INFO] Class distribution after removing unlabeled/invalid:")
    print(df["failureType_clean"].value_counts())

    df = df.reset_index(drop=True)
    return df


def save_class_distribution(df: pd.DataFrame):
    counts = df["failureType_clean"].value_counts()
    counts.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), header=["count"])

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Wafer Defect Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=200)
    plt.close()


def visualize_samples(df: pd.DataFrame, samples_per_class: int = 2):
    classes = df["failureType_clean"].unique().tolist()
    rows = len(classes)
    cols = samples_per_class

    plt.figure(figsize=(3 * cols, 2.5 * rows))

    plot_idx = 1
    for cls in classes:
        class_samples = df[df["failureType_clean"] == cls].head(samples_per_class)
        for _, row in class_samples.iterrows():
            img = np.array(row["waferMap"])
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(img, cmap="gray")
            plt.title(cls)
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sample_wafer_maps.png"), dpi=200)
    plt.close()


def preprocess_images(df: pd.DataFrame, image_size=(32, 32)):
    X = []
    y = []

    print("\n[INFO] Preprocessing images...")
    for _, row in df.iterrows():
        img = np.array(row["waferMap"], dtype=np.uint8)
        label = row["failureType_clean"]

        img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_NEAREST)
        X.append(img_resized.flatten())
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("[INFO] X shape:", X.shape)
    print("[INFO] y shape:", y.shape)
    return X, y


def train_baseline_random_forest(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\n[INFO] Training Random Forest baseline...")
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n[RESULT] Accuracy:", acc)
    print("\n[RESULT] Classification Report:\n")
    print(report)
    print("\n[RESULT] Confusion Matrix:\n")
    print(cm)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    np.savetxt(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
        cm,
        delimiter=",",
        fmt="%d"
    )

    return clf, le, acc


def main():
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_dataset(df)

    save_class_distribution(df)
    visualize_samples(df, samples_per_class=2)

    X, y = preprocess_images(df, image_size=(32, 32))
    _, _, acc = train_baseline_random_forest(X, y)

    print("\n[INFO] Pipeline completed successfully.")
    print(f"[INFO] Final baseline accuracy: {acc:.4f}")
    print(f"[INFO] Outputs saved in: {OUTPUT_DIR}")
    print(f"[INFO] Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
