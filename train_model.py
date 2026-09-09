"""
Train the MindPulse multi-label emotion classifier and persist the
pipeline + MultiLabelBinarizer next to the Streamlit app.

Usage:
    python train_model.py

Outputs (created/overwritten):
    app/models/mental_health_model.pkl
    app/models/mlb.pkl
    models/metrics.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
TRAIN_JSON = ROOT / "DepressionEmo" / "Dataset" / "train.json"
TEST_JSON = ROOT / "DepressionEmo" / "Dataset" / "test.json"
VAL_JSON = ROOT / "DepressionEmo" / "Dataset" / "val.json"

APP_MODELS = ROOT / "app" / "models"
APP_MODELS.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ROOT / "models" / "metrics.json"
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def load_data() -> pd.DataFrame:
    frames = [df for df in (_load(p) for p in (TRAIN_JSON, VAL_JSON, TEST_JSON)) if not df.empty]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["emotions"] = df["emotions"].apply(lambda v: v if isinstance(v, list) else [])
    df = df[df["emotions"].map(len) > 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _build_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10_000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )


def _build_logreg_pipeline() -> Pipeline:
    """Companion interpretable model. Used only to surface the words that
    push each label up/down for a given input (coefficient × tfidf score)."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10_000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def _safe_predict_proba(pipe: Pipeline, x) -> np.ndarray:
    """OneVsRest(LogReg) collapses to 1-D if a label has only one class in
    the training fold. Force a 2-D array so downstream thresholding works."""
    proba = pipe.predict_proba(x)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    return proba


def train(df: pd.DataFrame, threshold: float = 0.4) -> dict:
    x = df["text"].astype(str).tolist()
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["emotions"].tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # ----- primary model: RandomForest (best F1) -----
    pipeline = _build_rf_pipeline()
    pipeline.fit(x_train, y_train)
    proba = _safe_predict_proba(pipeline, x_test)
    y_pred = (proba >= threshold).astype(int)

    # ----- companion model: LogisticRegression (interpretable) -----
    interpretable = _build_logreg_pipeline()
    interpretable.fit(x_train, y_train)
    interp_proba = _safe_predict_proba(interpretable, x_test)
    interp_pred = (interp_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "primary_model": "OneVsRest(RandomForest, n=200)",
        "interpretable_model": "OneVsRest(LogisticRegression, liblinear, C=1.0)",
        "primary": {
            "hamming_loss": float(hamming_loss(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_micro": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        },
        "interpretable": {
            "hamming_loss": float(hamming_loss(y_test, interp_pred)),
            "accuracy": float(accuracy_score(y_test, interp_pred)),
            "precision_micro": float(precision_score(y_test, interp_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_test, interp_pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, interp_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, interp_pred, average="macro", zero_division=0)),
        },
        "labels": list(mlb.classes_),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
        ),
    }

    joblib.dump(pipeline, APP_MODELS / "mental_health_model.pkl")
    joblib.dump(interpretable, APP_MODELS / "interpretable_model.pkl")
    joblib.dump(mlb, APP_MODELS / "mlb.pkl")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    print(f"[MindPulse] Loading data from {TRAIN_JSON.parent} ...")
    df = load_data()
    print(f"[MindPulse] {len(df):,} examples, {df['emotions'].explode().nunique()} unique emotions")

    print("[MindPulse] Training TF-IDF + OneVsRest(RandomForest) (primary) ...")
    print("[MindPulse] Training TF-IDF + OneVsRest(LogisticRegression) (interpretable companion) ...")
    metrics = train(df)

    print("\n=== Test metrics (primary RF) ===")
    for k, v in metrics["primary"].items():
        print(f"  {k:18s} {v:.4f}")
    print("\n=== Test metrics (interpretable LogReg) ===")
    for k, v in metrics["interpretable"].items():
        print(f"  {k:18s} {v:.4f}")

    print(f"\n[MindPulse] Saved artefacts to {APP_MODELS}")
    print(f"[MindPulse] Saved metrics    to {METRICS_PATH}")


if __name__ == "__main__":
    main()
