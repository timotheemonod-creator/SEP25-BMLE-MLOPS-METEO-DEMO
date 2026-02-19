# SEP25-BMLE-MLOPS-METEO

End-to-end MLOps project for rainfall prediction (`RainTomorrow`) with:
- model training/evaluation,
- FastAPI inference service,
- MLflow experiment tracking,
- DVC + DagsHub data/artifact versioning,
- Airflow orchestration (prediction + monitoring + conditional Optuna retraining),
- Streamlit presentation/demo app.

This README is the operational reference for development, demo, and soutenance.

## 1) Project Goal

Business goal:
- Predict next-day rain per weather station to support operational planning.

Technical goal:
- Run a reproducible ML pipeline,
- Expose predictions through an API,
- Track model quality over time (offline + live),
- Trigger optimization automatically when live quality degrades.

Primary quality KPI:
- `recall` (priority), then `precision`, `f1`, `roc_auc`, `accuracy`.

## 2) Global Architecture

```text
                 +-----------------------+
                 |  Streamlit (demo UI) |
                 +-----------+-----------+
                             |
                             v
  +---------------------+  HTTP  +--------------------------+
  |  API clients/curl   +------->+ FastAPI (/predict,/health)|
  +---------------------+        +-------------+-------------+
                                              |
                                              v
                                      models/pipeline.joblib
                                              |
                                              v
                                    outputs/preds_api.csv
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
            src/live_monitoring.py                      DVC pointer (.dvc)
            src/combined_metrics.py                     + DagsHub storage
                         |
                         v
         metrics/live_eval.json + metrics/combined_eval.json
                         |
                         v
             Airflow weather_main_dag branch_on_recall
                         |
             +-----------+------------+
             |                        |
             v                        v
        skip_optuna            trigger optuna_tuning_dag
                                        |
                                        v
                         optimisations/optuna_search_recall_small.py
                                        |
                                        v
                               models/best_params.json
```

## 3) Tools and How They Complement Each Other

- `scikit-learn` / `imblearn` / `xgboost`: feature processing + classification pipeline.
- `FastAPI`: inference API and Swagger UI.
- `MLflow`: logs params, metrics, and model artifacts for each run.
- `DVC`: tracks data/artifact files with lightweight pointers in Git.
- `DagsHub`: remote storage for DVC and MLflow UI backend target.
- `Airflow`: schedules/automates prediction + monitoring + conditional tuning.
- `Docker` / `Docker Compose`: reproducible local microservices stack.
- `Streamlit`: soutenance app (slides + live demo tabs).
- `GitHub Actions`: scheduled/manual publication of `preds_api.csv` to DagsHub.

## 4) Repository Structure

```text
.
|- airflow/
|  |- dags/
|  |  |- weather_main_dag.py
|  |  `- optuna_tuning_dag.py
|  |- logs/
|  `- plugins/
|- api/
|  `- main.py
|- data/
|  |- raw/
|  |  |- weatherAUS.csv.dvc
|  |  `- weatherAUS.csv            (DVC-managed payload)
|  `- processed/
|- docker/
|  |- Dockerfile.api
|  |- Dockerfile.ml
|  |- Dockerfile.airflow
|  `- requirements.airflow.extra.txt
|- metrics/
|  |- eval.json
|  |- live_eval.json
|  `- combined_eval.json
|- models/
|  |- pipeline.joblib
|  `- best_params.json
|- optimisations/
|  |- optuna_search_recall_small.py
|  `- optuna_xgb_recall_small.db
|- outputs/
|  |- preds_api.csv
|  |- preds_api.csv.dvc
|  |- preds_api_scored.csv
|  `- predictions_daily.csv
|- scripts/
|  `- predict_all_and_push.sh
|- src/
|  |- data_preparation.py
|  |- training.py
|  |- evaluation.py
|  |- live_monitoring.py
|  `- combined_metrics.py
|- tests/
|  `- test_health.py
|- .github/workflows/
|  `- publish_preds.yml
|- docker-compose.yml
|- docker-compose.airflow.yml
|- dvc.yaml
|- Makefile
|- streamlit_app.py
`- README.md
```

## 5) What Is Stored Where

### In Git (source of truth for code/config)
- Python code, DAGs, Dockerfiles, Compose files,
- DVC pointers (`*.dvc`, `dvc.yaml`, `dvc.lock`),
- CI workflow files,
- Documentation.

### In DVC + DagsHub (data/artifact payloads)
- `data/raw/weatherAUS.csv` (via `data/raw/weatherAUS.csv.dvc`),
- `outputs/preds_api.csv` (via `outputs/preds_api.csv.dvc`),
- Other large tracked payloads depending on local `dvc add`.

### In MLflow
- Training/evaluation/optuna run metadata:
  - params,
  - metrics,
  - model artifacts.
- Backend can be local or DagsHub MLflow URI (via env).

### In Docker
- Runtime images and containers:
  - API image,
  - ML jobs image,
  - Airflow custom image (`meteo-airflow:2.10.5`).

### In Airflow
- DAG definitions: `airflow/dags/*.py`,
- Orchestration metadata DB: Postgres container volume `airflow_postgres_data`,
- Task logs: `airflow/logs/`.

### In API / Swagger
- Live docs and manual testing at `/docs`.

## 6) Main Pipelines

### 6.1 Training Pipeline
1. `src.data_preparation` prepares cleaned processed data.
2. `src.training` trains `models/pipeline.joblib`, logs metrics to MLflow.
3. `src.evaluation` computes offline metrics into `metrics/eval.json`.

### 6.2 Inference + Live Monitoring Pipeline
1. FastAPI `/predict` logs each prediction to `outputs/preds_api.csv`.
2. `src.live_monitoring` matches predictions against BOM observed rain and writes:
   - `outputs/preds_api_scored.csv`,
   - `metrics/live_eval.json`.
3. `src.combined_metrics` computes historical + live metrics:
   - `metrics/combined_eval.json`.

### 6.3 Airflow Decision Logic

`weather_main_dag` schedule: `0 6,18 * * *` (06:00 and 18:00 UTC)

Tasks:
1. `predict_all_and_push`
2. `compute_live_metrics`
3. `compute_combined_metrics`
4. `read_quality_metrics`
5. `branch_on_recall`
6. `trigger_optuna` OR `skip_optuna`

Decision rule:
- Trigger Optuna only if:
  - `recall_live < RECALL_THRESHOLD` (default `0.59`),
  - and `scored_rows >= MIN_LIVE_LABELS` (default `72`).

`optuna_tuning_dag`:
- Runs `optimisations/optuna_search_recall_small.py`,
- Exports best params to `models/best_params.json`.

## 7) Local Setup

## 7.1 Prerequisites
- Python 3.12 recommended,
- Docker Desktop + Compose v2,
- WSL integration enabled if running under WSL,
- Optional: DagsHub token for DVC/MLflow remote push.

## 7.2 Python environment

```bash
python -m venv .venv_meteo
source .venv_meteo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `streamlit` is not found:
```bash
python -m pip install streamlit
```

## 7.3 Environment files

- Copy and edit Docker env:
```bash
cp .env.docker.example .env.docker
```

- Airflow env already exists as `.env.airflow`:
  - set `MLFLOW_TRACKING_PASSWORD`,
  - set `_AIRFLOW_WWW_USER_USERNAME` / `_AIRFLOW_WWW_USER_PASSWORD`,
  - tune `RECALL_THRESHOLD` and `MIN_LIVE_LABELS` if needed.

## 8) Run Commands

## 8.1 Without Docker (quick dev loop)

```bash
make setup
make preprocess
make train
make evaluate
make api
```

## 8.2 Core Docker services (API + MLflow)

```bash
docker compose up -d --build api-inference mlflow-server
docker compose ps
```

One-shot ML jobs:
```bash
docker compose run --rm preprocess
docker compose run --rm train
docker compose run --rm evaluate
docker compose run --rm predict-batch
```

## 8.3 Airflow stack

Build Airflow image:
```bash
docker build -f docker/Dockerfile.airflow -t meteo-airflow:2.10.5 .
```

Start Airflow services:
```bash
set -a
source .env.airflow
set +a

docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d \
  airflow-postgres airflow-init airflow-webserver airflow-scheduler airflow-triggerer
```

Trigger main DAG manually:
```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags trigger weather_main_dag
```

Check run status:
```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags list-runs -d weather_main_dag
```

Check task states for a run:
```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow tasks states-for-dag-run weather_main_dag 'manual__YYYY-MM-DDTHH:MM:SS+00:00'
```

## 8.4 Streamlit soutenance app

```bash
streamlit run streamlit_app.py
```

## 9) URLs / UIs

- FastAPI Swagger: `http://127.0.0.1:8000/docs`
- FastAPI health: `http://127.0.0.1:8000/health`
- MLflow UI (local): `http://127.0.0.1:5000`
- Airflow UI: `http://127.0.0.1:8080`
- Streamlit UI: `http://127.0.0.1:8501`

## 10) API Quick Usage

Health:
```bash
curl -s http://127.0.0.1:8000/health
```

Prediction from latest BOM values:
```bash
curl -s -X POST \
  "http://127.0.0.1:8000/predict?use_latest=true&station_name=Sydney" \
  -H "Authorization: Bearer my_secret_api_key" \
  -H "accept: application/json"
```

## 11) Tests and Validation Checklist

Python tests:
```bash
pytest -q
```

DAG syntax checks:
```bash
python -m py_compile airflow/dags/weather_main_dag.py
python -m py_compile airflow/dags/optuna_tuning_dag.py
```

Compose validation:
```bash
docker compose -f docker-compose.yml config >/tmp/compose.check
docker compose -f docker-compose.yml -f docker-compose.airflow.yml config >/tmp/compose.airflow.check
```

Service status:
```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml ps
```

Airflow import errors:
```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags list-import-errors
```

## 12) CI: Publish Predictions to DagsHub

Workflow: `.github/workflows/publish_preds.yml`

Triggers:
- manual (`workflow_dispatch`),
- scheduled (`15 6,18 * * *`).

Required GitHub repository secrets:
- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`

What the workflow does:
1. checkout branch,
2. install DVC,
3. validate `outputs/preds_api.csv`,
4. `dvc add outputs/preds_api.csv`,
5. commit/push `.dvc` pointer,
6. `dvc push` to DagsHub remote.

## 13) Known Runtime Notes

- Airflow task commands should not call nested `docker compose` inside Airflow containers.
- Optuna runtime can be several minutes; for soutenance use a capped demo mode.
- If `states-for-dag-run` fails with `RUN_ID`, replace it with a real run id string.
- If predictions fail in Airflow, ensure API container `meteo-api` is running.

## 14) Additional Readmes

- `README_branche.md`: historical branch-focused notes (Phase 2 context).

---

If you need a minimal "demo-only" command set, use:
1. start API + Airflow + Streamlit,
2. trigger `weather_main_dag`,
3. show Airflow graph + Streamlit monitoring tabs + Swagger response.

