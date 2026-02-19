from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data_preparation import prepare_dataset_from_processed


PROCESSED_PATH = Path(os.getenv("PROCESSED_PATH", "data/processed/weatherAUS_processed.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline.joblib"))
LIVE_SCORED_PATH = Path(os.getenv("LIVE_SCORED_PATH", "outputs/preds_api_scored.csv"))
OUT_PATH = Path(os.getenv("COMBINED_METRICS_PATH", "metrics/combined_eval.json"))


def _safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if len(y_true.unique()) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_combined_metrics() -> dict:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    X_hist, y_hist, _, _ = prepare_dataset_from_processed(str(PROCESSED_PATH))
    model = joblib.load(MODEL_PATH)
    y_hist_pred = pd.Series(model.predict(X_hist)).astype(int)
    y_hist_proba = pd.Series(model.predict_proba(X_hist)[:, 1]).astype(float)
    y_hist_true = pd.Series(y_hist).astype(int)

    y_live_true = pd.Series(dtype="int64")
    y_live_pred = pd.Series(dtype="int64")
    y_live_proba = pd.Series(dtype="float64")

    if LIVE_SCORED_PATH.exists():
        live = pd.read_csv(LIVE_SCORED_PATH)
        if {"truth_available", "y_true", "predicted_rain"}.issubset(live.columns):
            live = live[live["truth_available"] == True].copy()  # noqa: E712
            live["y_true"] = pd.to_numeric(live["y_true"], errors="coerce")
            live["predicted_rain"] = pd.to_numeric(live["predicted_rain"], errors="coerce")
            live = live.dropna(subset=["y_true", "predicted_rain"])
            if not live.empty:
                y_live_true = live["y_true"].astype(int)
                y_live_pred = live["predicted_rain"].astype(int)
                if "rain_probability" in live.columns:
                    live["rain_probability"] = pd.to_numeric(live["rain_probability"], errors="coerce")
                    y_live_proba = live["rain_probability"].dropna().astype(float)

    y_true_all = pd.concat([y_hist_true, y_live_true], ignore_index=True)
    y_pred_all = pd.concat([y_hist_pred, y_live_pred], ignore_index=True)

    # For ROC AUC, keep rows that have a probability score.
    y_true_for_auc = y_hist_true.copy()
    y_score_for_auc = y_hist_proba.copy()
    if len(y_live_proba) > 0 and len(y_live_true) > 0:
        # Align by index after dropping NaN probabilities.
        live_auc = pd.DataFrame({"y_true": y_live_true, "proba": pd.to_numeric(y_live_proba, errors="coerce")}).dropna()
        if not live_auc.empty:
            y_true_for_auc = pd.concat([y_true_for_auc, live_auc["y_true"].astype(int)], ignore_index=True)
            y_score_for_auc = pd.concat([y_score_for_auc, live_auc["proba"].astype(float)], ignore_index=True)

    metrics = {
        "status": "ok",
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true_for_auc.astype(int), y_score_for_auc.astype(float)),
        "historical_rows": int(len(y_hist_true)),
        "live_labeled_rows": int(len(y_live_true)),
        "total_rows": int(len(y_true_all)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(compute_combined_metrics(), indent=2))
