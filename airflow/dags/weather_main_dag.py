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
MIN_LIVE_LABELS = int(os.getenv("MIN_LIVE_LABELS", "72"))
LIVE_METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics", "live_eval.json")
COMBINED_METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics", "combined_eval.json")

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
    return {
        "live": live,
        "combined": combined,
    }


def branch_on_recall(**context) -> str:
    payload = context["ti"].xcom_pull(task_ids="read_quality_metrics")
    live = payload.get("live", {})
    recall_live = float(live.get("recall_live", 1.0))
    scored_rows = int(live.get("scored_rows", 0))
    if scored_rows < MIN_LIVE_LABELS:
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
