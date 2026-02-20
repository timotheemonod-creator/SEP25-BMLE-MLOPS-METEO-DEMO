from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
RECALL_THRESHOLD = float(os.getenv("RECALL_THRESHOLD", "0.59"))
MIN_NEW_ROWS_FOR_RETRAIN = int(os.getenv("MIN_NEW_ROWS_FOR_RETRAIN", "60"))
RETRAIN_QUALITY_PATH = os.path.join(PROJECT_ROOT, "metrics", "retrain_quality_eval.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pipeline.joblib")
MODEL_META_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def branch_on_recall(**context) -> str:
    if not os.path.exists(RETRAIN_QUALITY_PATH):
        return "skip_optuna"
    try:
        with open(RETRAIN_QUALITY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return "skip_optuna"

    if payload.get("status") != "ok":
        return "skip_optuna"
    recall_combined_cv = float(payload.get("recall_combined_cv", 1.0))
    new_rows_for_retrain = int(payload.get("new_rows_for_retrain", 0))
    if new_rows_for_retrain < MIN_NEW_ROWS_FOR_RETRAIN:
        return "skip_optuna"
    return "trigger_optuna" if recall_combined_cv < RECALL_THRESHOLD else "skip_optuna"


def ensure_latest_model_ready() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    info = {
        "model_path": MODEL_PATH,
        "model_mtime_utc": datetime.utcfromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() + "Z",
    }
    if os.path.exists(MODEL_META_PATH):
        try:
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            info["model_metadata"] = meta
        except Exception:
            info["model_metadata"] = {"status": "unreadable"}
    else:
        info["model_metadata"] = {"status": "not_found"}
    return info


with DAG(
    dag_id="weather_main_dag",
    description="Main DAG: predict all stations, then run quality gate and trigger Optuna on low combined recall",
    start_date=datetime(2026, 1, 1),
    schedule="0 6,18 * * *",
    catchup=False,
    default_args=default_args,
    template_searchpath=[PROJECT_ROOT],
    tags=["weather", "main", "mlops"],
) as dag:

    ensure_latest_model = PythonOperator(
        task_id="ensure_latest_model",
        python_callable=ensure_latest_model_ready,
    )

    predict_all_and_push = BashOperator(
        task_id="predict_all_and_push",
        bash_command=f"cd {PROJECT_ROOT} && bash ./scripts/predict_all_and_push.sh ",
    )

    compute_live_metrics = BashOperator(
        task_id="compute_live_metrics",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.live_monitoring",
    )

    compute_retrain_quality = BashOperator(
        task_id="compute_retrain_quality",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.retrain_quality_monitor",
    )

    predictions_ready = EmptyOperator(task_id="predictions_ready")

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

    ensure_latest_model >> predict_all_and_push

    predict_all_and_push >> predictions_ready
    predict_all_and_push >> compute_live_metrics >> compute_retrain_quality >> branch

    branch >> trigger_optuna >> join_after_branch
    branch >> skip_optuna >> join_after_branch
    join_after_branch
