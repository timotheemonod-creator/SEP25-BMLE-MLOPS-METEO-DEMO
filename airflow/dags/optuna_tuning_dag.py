from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import optuna
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
OPTUNA_DB = os.path.join(PROJECT_ROOT, "optimisations", "optuna_xgb_recall_small.db")
BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "best_params.json")

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
        bash_command=f"cd {PROJECT_ROOT} && PYTHONPATH={PROJECT_ROOT} python optimisations/optuna_search_recall_small.py",
    )

    save_best_params = PythonOperator(
        task_id="save_best_params",
        python_callable=export_best_params,
    )

    run_optuna >> save_best_params
