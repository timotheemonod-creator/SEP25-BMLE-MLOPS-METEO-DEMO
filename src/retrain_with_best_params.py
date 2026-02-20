from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
RETRAIN_RAW_PATH = Path(os.getenv("RETRAIN_RAW_PATH", "data/processed/weatherAUS_retrain.csv"))
BEST_PARAMS_PATH = Path(os.getenv("BEST_PARAMS_PATH", "models/best_params.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline.joblib"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", "models/model_metadata.json"))


def _load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Best params file not found: {BEST_PARAMS_PATH}")
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("best_params", {})
    if not params:
        raise ValueError("best_params.json exists but contains no 'best_params'.")
    return payload


def train_with_best_params() -> dict:
    raw_path = PROJECT_ROOT / RETRAIN_RAW_PATH
    model_path = PROJECT_ROOT / MODEL_PATH
    model_meta_path = PROJECT_ROOT / MODEL_META_PATH
    best_payload = _load_best_params()
    best_params = dict(best_payload["best_params"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Retrain dataset not found: {raw_path}")

    X, y, cat_cols, num_cols = prepare_dataset(str(raw_path))
    y = np.asarray(y).astype(int).ravel()

    model_params = best_params.copy()
    model_params.update(
        {
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(cat_cols, num_cols)),
            ("scaling", StandardScaler(with_mean=False)),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(**model_params)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1", "accuracy": "accuracy"},
    )

    mlflow.set_experiment("weather-ml")
    with mlflow.start_run(run_name="retrain_from_optuna_best"):
        mlflow.log_params(model_params)
        mlflow.log_param("retrain_rows", int(len(X)))
        mlflow.log_param("retrain_raw_path", str(raw_path))
        mlflow.log_param("best_study_name", best_payload.get("study_name"))
        mlflow.log_metric("best_cv_value_from_optuna", float(best_payload.get("best_value", 0.0)))
        cv_roc_auc = float(np.mean(cv_scores["test_roc_auc"]))
        cv_precision = float(np.mean(cv_scores["test_precision"]))
        cv_recall = float(np.mean(cv_scores["test_recall"]))
        cv_f1 = float(np.mean(cv_scores["test_f1"]))
        cv_accuracy = float(np.mean(cv_scores["test_accuracy"]))
        # Metrics namespaced for detailed tracking
        mlflow.log_metric("retrain_cv_roc_auc", cv_roc_auc)
        mlflow.log_metric("retrain_cv_precision", cv_precision)
        mlflow.log_metric("retrain_cv_recall", cv_recall)
        mlflow.log_metric("retrain_cv_f1", cv_f1)
        mlflow.log_metric("retrain_cv_accuracy", cv_accuracy)
        # Canonical aliases so MLflow/DagsHub default metric columns are populated
        mlflow.log_metric("roc_auc", cv_roc_auc)
        mlflow.log_metric("precision", cv_precision)
        mlflow.log_metric("recall", cv_recall)
        mlflow.log_metric("f1", cv_f1)
        mlflow.log_metric("accuracy", cv_accuracy)

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1]
        mlflow.log_metric("retrain_fit_accuracy", float(accuracy_score(y, y_pred)))
        mlflow.log_metric("retrain_fit_precision", float(precision_score(y, y_pred, zero_division=0)))
        mlflow.log_metric("retrain_fit_recall", float(recall_score(y, y_pred, zero_division=0)))
        mlflow.log_metric("retrain_fit_f1", float(f1_score(y, y_pred, zero_division=0)))
        mlflow.log_metric("retrain_fit_roc_auc", float(roc_auc_score(y, y_proba)))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "retrain_raw_path": str(raw_path),
        "rows_used": int(len(X)),
        "best_study_name": best_payload.get("study_name"),
        "best_value": best_payload.get("best_value"),
        "best_params": best_payload.get("best_params"),
    }
    model_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_with_best_params(), indent=2))
