from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import optuna
import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
OPTUNA_DB = os.path.join(PROJECT_ROOT, "optimisations", "optuna_xgb_recall_small.db")
BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "best_params.json")
SCORED_PREDS_PATH = os.path.join(PROJECT_ROOT, "outputs", "preds_api_scored.csv")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "weatherAUS.csv")
RETRAIN_WATERMARK_PATH = os.path.join(PROJECT_ROOT, "metrics", "retrain_watermark.json")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


def export_best_params() -> None:
    storage = f"sqlite:///{OPTUNA_DB}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    valid = [s for s in summaries if s.best_trial is not None]
    if not valid:
        raise RuntimeError("No Optuna study with best_trial found in storage.")

    best_summary = max(valid, key=lambda s: s.best_trial.value)
    study = optuna.load_study(study_name=best_summary.study_name, storage=storage)

    payload = {
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
    }

    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def update_retrain_watermark(**context) -> None:
    if not os.path.exists(SCORED_PREDS_PATH) or not os.path.exists(RAW_DATA_PATH):
        return

    hist = pd.read_csv(RAW_DATA_PATH, usecols=["Date"])
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.date
    max_hist_date = hist["Date"].max()
    if max_hist_date is None:
        return

    scored = pd.read_csv(SCORED_PREDS_PATH)
    needed = {"truth_available", "feature_date", "y_true"}
    if not needed.issubset(scored.columns):
        return

    scored = scored[scored["truth_available"] == True].copy()  # noqa: E712
    scored = scored.dropna(subset=["y_true"])
    feature_ts = pd.to_datetime(scored["feature_date"], errors="coerce", format="mixed")
    target_ts = pd.to_datetime(scored.get("target_date"), errors="coerce")
    inferred = target_ts - pd.Timedelta(days=1)
    feature_ts = feature_ts.fillna(inferred)
    scored["feature_date"] = feature_ts.dt.date
    scored = scored.dropna(subset=["feature_date"])
    scored = scored[scored["feature_date"] > max_hist_date]
    scored = scored.drop_duplicates(["location", "feature_date"], keep="last")
    if scored.empty:
        return

    latest_consumed = scored["feature_date"].max()
    payload = {
        "last_feature_date": str(latest_consumed),
        "consumed_rows": int(len(scored)),
        "updated_at_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": context.get("run_id"),
    }
    os.makedirs(os.path.dirname(RETRAIN_WATERMARK_PATH), exist_ok=True)
    with open(RETRAIN_WATERMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


with DAG(
    dag_id="optuna_tuning_dag",
    description="Run Optuna tuning and export best params",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["weather", "optuna", "mlops"],
) as dag:

    run_optuna = BashOperator(
        task_id="run_optuna",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"PYTHONPATH={PROJECT_ROOT} python -m src.retrain_dataset_builder && "
            f"PYTHONPATH={PROJECT_ROOT} OPTUNA_RAW_PATH=data/processed/weatherAUS_retrain.csv "
            f"python optimisations/optuna_search_recall_small.py"
        ),
    )

    save_best_params = PythonOperator(
        task_id="save_best_params",
        python_callable=export_best_params,
    )

    mark_retrain_consumed = PythonOperator(
        task_id="mark_retrain_consumed",
        python_callable=update_retrain_watermark,
    )

    run_optuna >> save_best_params >> mark_retrain_consumed
