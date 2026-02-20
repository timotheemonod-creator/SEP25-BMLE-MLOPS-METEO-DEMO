from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_preparation import prepare_dataset
from src.retrain_dataset_builder import build_retrain_dataset


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
RETRAIN_RAW_PATH = Path(os.getenv("RETRAIN_RAW_PATH", "data/processed/weatherAUS_retrain.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline.joblib"))
OUT_PATH = Path(os.getenv("RETRAIN_QUALITY_PATH", "metrics/retrain_quality_eval.json"))
CV_SPLITS = int(os.getenv("RETRAIN_QUALITY_CV_SPLITS", "5"))
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/weatherAUS.csv"))
SCORED_PREDS_PATH = Path(os.getenv("SCORED_PREDS_PATH", "outputs/preds_api_scored.csv"))
RETRAIN_WATERMARK_PATH = Path(os.getenv("RETRAIN_WATERMARK_PATH", "metrics/retrain_watermark.json"))


def _read_retrain_watermark(watermark_path: Path) -> date | None:
    if not watermark_path.exists():
        return None
    try:
        payload = json.loads(watermark_path.read_text(encoding="utf-8"))
        value = payload.get("last_feature_date")
        if not value:
            return None
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return None


def _count_new_labeled_rows_for_retrain(raw_data_path: Path, scored_preds_path: Path, watermark_path: Path) -> tuple[int, date | None]:
    if not scored_preds_path.exists() or not raw_data_path.exists():
        return 0, None
    try:
        hist = pd.read_csv(raw_data_path, usecols=["Date"])
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.date
        max_hist_date = hist["Date"].max()
        if max_hist_date is None:
            return 0, None
        watermark_date = _read_retrain_watermark(watermark_path)
        cutoff_date = max_hist_date
        if watermark_date is not None and watermark_date > cutoff_date:
            cutoff_date = watermark_date

        scored = pd.read_csv(scored_preds_path)
        needed = {"truth_available", "feature_date", "y_true"}
        if not needed.issubset(scored.columns):
            return 0, cutoff_date

        scored = scored[scored["truth_available"] == True].copy()  # noqa: E712
        scored = scored.dropna(subset=["y_true"])
        feature_ts = pd.to_datetime(scored["feature_date"], errors="coerce", format="mixed")
        target_ts = pd.to_datetime(scored.get("target_date"), errors="coerce")
        inferred = target_ts - pd.Timedelta(days=1)
        feature_ts = feature_ts.fillna(inferred)
        scored["feature_date"] = feature_ts.dt.date
        scored = scored.dropna(subset=["feature_date"])
        scored = scored[scored["feature_date"] > cutoff_date]
        scored = scored.drop_duplicates(["location", "feature_date"], keep="last")
        return int(len(scored)), cutoff_date
    except Exception:
        return 0, None


def compute_retrain_quality() -> dict:
    retrain_meta = build_retrain_dataset()

    raw_path = PROJECT_ROOT / RETRAIN_RAW_PATH
    model_path = PROJECT_ROOT / MODEL_PATH
    out_path = PROJECT_ROOT / OUT_PATH
    raw_data_path = PROJECT_ROOT / RAW_DATA_PATH
    scored_preds_path = PROJECT_ROOT / SCORED_PREDS_PATH
    watermark_path = PROJECT_ROOT / RETRAIN_WATERMARK_PATH

    if not raw_path.exists():
        raise FileNotFoundError(f"Retrain dataset not found: {raw_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    new_rows_for_retrain, cutoff_date = _count_new_labeled_rows_for_retrain(
        raw_data_path=raw_data_path,
        scored_preds_path=scored_preds_path,
        watermark_path=watermark_path,
    )

    X, y, _, _ = prepare_dataset(str(raw_path))
    y = np.asarray(y).astype(int).ravel()

    if len(np.unique(y)) < 2:
        payload = {
            "status": "not_enough_classes",
            "rows_used": int(len(X)),
            "cv_splits": CV_SPLITS,
            "retrain_dataset_meta": retrain_meta,
            "new_rows_for_retrain": int(new_rows_for_retrain),
            "retrain_cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    pipeline = joblib.load(model_path)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "accuracy": "accuracy",
        },
        n_jobs=1,
    )

    payload = {
        "status": "ok",
        "rows_used": int(len(X)),
        "cv_splits": CV_SPLITS,
        "accuracy_combined_cv": float(np.mean(scores["test_accuracy"])),
        "precision_combined_cv": float(np.mean(scores["test_precision"])),
        "recall_combined_cv": float(np.mean(scores["test_recall"])),
        "f1_combined_cv": float(np.mean(scores["test_f1"])),
        "roc_auc_combined_cv": float(np.mean(scores["test_roc_auc"])),
        "retrain_dataset_meta": retrain_meta,
        "new_rows_for_retrain": int(new_rows_for_retrain),
        "retrain_cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(compute_retrain_quality(), indent=2))
