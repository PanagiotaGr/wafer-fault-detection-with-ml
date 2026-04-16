"""
Classical ML classifiers (Random Forest, SVM, Logistic Regression).
Each returns a fitted sklearn pipeline so callers don't need to know
about the internals.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    rf_cfg = cfg.get("random_forest", {})
    return RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
        random_state=cfg.get("seed", 42),
        n_jobs=rf_cfg.get("n_jobs", -1),
    )


def build_svm(cfg: dict) -> Pipeline:
    """SVM with StandardScaler (required for RBF kernel)."""
    svm_cfg = cfg.get("svm", {})
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=svm_cfg.get("C", 1.0),
            kernel=svm_cfg.get("kernel", "rbf"),
            class_weight=svm_cfg.get("class_weight", "balanced"),
            random_state=cfg.get("seed", 42),
        )),
    ])


def build_logistic_regression(cfg: dict) -> Pipeline:
    """Logistic Regression with StandardScaler."""
    lr_cfg = cfg.get("logistic_regression", {})
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
        )),
    ])


ALL_MODELS = {
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
    "random_forest": build_random_forest,
}
