from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
import pandas as pd

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
RECALL_THRESHOLD = float(os.getenv("RECALL_THRESHOLD", "0.59"))
MIN_NEW_ROWS_FOR_RETRAIN = int(os.getenv("MIN_NEW_ROWS_FOR_RETRAIN", "60"))
LIVE_METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics", "live_eval.json")
COMBINED_METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics", "combined_eval.json")
RETRAIN_WATERMARK_PATH = os.path.join(PROJECT_ROOT, "metrics", "retrain_watermark.json")
SCORED_PREDS_PATH = os.path.join(PROJECT_ROOT, "outputs", "preds_api_scored.csv")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "weatherAUS.csv")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def read_quality_metrics() -> dict:
    with open(LIVE_METRICS_PATH, "r", encoding="utf-8") as f:
        live = json.load(f)
    combined = {}
    if os.path.exists(COMBINED_METRICS_PATH):
        with open(COMBINED_METRICS_PATH, "r", encoding="utf-8") as f:
            combined = json.load(f)
    new_rows_for_retrain, cutoff_date = count_new_labeled_rows_for_retrain()
    return {
        "live": live,
        "combined": combined,
        "new_rows_for_retrain": new_rows_for_retrain,
        "retrain_cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
    }


def _read_retrain_watermark() -> date | None:
    if not os.path.exists(RETRAIN_WATERMARK_PATH):
        return None
    try:
        with open(RETRAIN_WATERMARK_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        watermark = payload.get("last_feature_date")
        if not watermark:
            return None
        return pd.to_datetime(watermark, errors="coerce").date()
    except Exception:
        return None


def count_new_labeled_rows_for_retrain() -> tuple[int, date | None]:
    if not os.path.exists(SCORED_PREDS_PATH) or not os.path.exists(RAW_DATA_PATH):
        return 0, None
    try:
        hist = pd.read_csv(RAW_DATA_PATH, usecols=["Date"])
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce").dt.date
        max_hist_date = hist["Date"].max()
        if max_hist_date is None:
            return 0, None
        watermark_date = _read_retrain_watermark()
        cutoff_date = max_hist_date
        if watermark_date is not None and watermark_date > cutoff_date:
            cutoff_date = watermark_date

        scored = pd.read_csv(SCORED_PREDS_PATH)
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


def branch_on_recall(**context) -> str:
    payload = context["ti"].xcom_pull(task_ids="read_quality_metrics")
    live = payload.get("live", {})
    recall_live = float(live.get("recall_live", 1.0))
    new_rows_for_retrain = int(payload.get("new_rows_for_retrain", 0))
    if new_rows_for_retrain < MIN_NEW_ROWS_FOR_RETRAIN:
        return "skip_optuna"
    return "trigger_optuna" if recall_live < RECALL_THRESHOLD else "skip_optuna"


with DAG(
    dag_id="weather_main_dag",
    description="Main DAG: predict all stations, score against observed truth, trigger Optuna on low recall",
    start_date=datetime(2026, 1, 1),
    schedule="0 6,18 * * *",
    catchup=False,
    default_args=default_args,
    template_searchpath=[PROJECT_ROOT],
    tags=["weather", "main", "mlops"],
) as dag:

    predict_all_and_push = BashOperator(
        task_id="predict_all_and_push",
        bash_command=f"cd {PROJECT_ROOT} && bash ./scripts/predict_all_and_push.sh ",
    )

    compute_live_metrics = BashOperator(
        task_id="compute_live_metrics",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.live_monitoring",
    )

    compute_combined_metrics = BashOperator(
        task_id="compute_combined_metrics",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.combined_metrics",
    )

    read_live_recall = PythonOperator(
        task_id="read_quality_metrics",
        python_callable=read_quality_metrics,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_recall",
        python_callable=branch_on_recall,
    )

    trigger_optuna = TriggerDagRunOperator(
        task_id="trigger_optuna",
        trigger_dag_id="optuna_tuning_dag",
        wait_for_completion=False,
    )

    skip_optuna = EmptyOperator(task_id="skip_optuna")

    join_after_branch = EmptyOperator(
        task_id="join_after_branch",
        trigger_rule="none_failed_min_one_success",
    )

    predict_all_and_push >> compute_live_metrics >> compute_combined_metrics >> read_live_recall >> branch
    branch >> trigger_optuna >> join_after_branch
    branch >> skip_optuna >> join_after_branch
    join_after_branch
