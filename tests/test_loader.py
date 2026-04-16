"""
Unit tests for data loading and preprocessing.
These run without the actual dataset — they use synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import _parse_failure_type, clean, extract_images, split_data


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_df(n: int = 200, n_classes: int = 4) -> pd.DataFrame:
    """Create a minimal fake WM-811K DataFrame."""
    rng = np.random.default_rng(0)
    classes = [f"class_{i}" for i in range(n_classes)]
    records = []
    for i in range(n):
        records.append({
            "waferMap":    rng.integers(0, 3, size=(32, 32)).tolist(),
            "failureType": classes[i % n_classes],
        })
    return pd.DataFrame(records)


# ── _parse_failure_type ───────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    ("edge-ring",       "edge-ring"),
    (["center"],        "center"),
    (np.array(["loc"]), "loc"),
    ([],                "none"),
    (None,              "none"),
    (float("nan"),      "none"),
    ("  Scratch  ",     "scratch"),
])
def test_parse_failure_type(value, expected):
    assert _parse_failure_type(value) == expected


# ── clean ─────────────────────────────────────────────────────────────────────

def test_clean_removes_invalid_labels():
    df = make_dummy_df(100, 4)
    # Inject invalid rows using object-dtype column to allow mixed types
    df["failureType"] = df["failureType"].astype(object)
    df.at[0, "failureType"] = None
    df.at[1, "failureType"] = np.nan
    cleaned = clean(df)
    assert "none" not in cleaned["label"].values
    assert len(cleaned) <= len(df)


def test_clean_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="waferMap"):
        clean(df)


def test_clean_label_column_exists():
    df = make_dummy_df()
    cleaned = clean(df)
    assert "label" in cleaned.columns


# ── extract_images ────────────────────────────────────────────────────────────

def test_extract_images_shape():
    df = make_dummy_df(50)
    df = clean(df)
    X, y = extract_images(df, image_size=32)
    assert X.shape == (len(df), 32, 32)
    assert y.shape == (len(df),)


def test_extract_images_normalized():
    df = make_dummy_df(50)
    df = clean(df)
    X, _ = extract_images(df, image_size=16)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


# ── split_data ────────────────────────────────────────────────────────────────

def test_split_data_sizes():
    X = np.zeros((200, 32 * 32), dtype=np.float32)
    y = np.array(["a"] * 100 + ["b"] * 100)
    result = split_data(X, y, test_size=0.2, val_size=0.15, seed=42)

    total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
    assert total == 200


def test_split_data_label_encoder():
    X = np.zeros((100, 10), dtype=np.float32)
    y = np.array(["cat"] * 50 + ["dog"] * 50)
    result = split_data(X, y)
    le = result["label_encoder"]
    assert set(le.classes_) == {"cat", "dog"}
