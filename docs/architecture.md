# Architecture technique (version active)

Ce document décrit l'architecture **en production projet** (et non l'historique).

## 1) Vue d'ensemble

- `FastAPI` expose `/health` et `/predict` et lit le modèle actif `models/pipeline.joblib`.
- `Airflow` orchestre:
  - `weather_main_dag` pour la prédiction + monitoring + décision,
  - `optuna_tuning_dag` pour tuning + retrain + watermark.
- `MLflow` trace les runs d'entraînement, d'évaluation, d'Optuna et de retrain.
- `DVC + DagsHub` versionnent les artefacts volumineux (dataset, logs de prédictions, base Optuna).
- `Streamlit` sert de support de soutenance (slides + démo live API).

## 2) Chaîne opérationnelle

1. `weather_main_dag` lance les prédictions sur 36 stations.
2. Les sorties sont enrichies avec les vérités observées (BOM) via `src.live_monitoring`.
3. Le DAG calcule les métriques live + combinées.
4. Branche de décision:
   - si `recall_live < seuil` **et** `new_rows_for_retrain >= MIN_NEW_ROWS_FOR_RETRAIN` -> trigger Optuna,
   - sinon skip.
5. `optuna_tuning_dag` exécute:
   - `src.retrain_dataset_builder` (historique + nouvelles lignes scorées),
   - `optimisations/prod/optuna_search_recall_small.py`,
   - export des meilleurs paramètres,
   - retrain final (`src.retrain_with_best_params`),
   - mise à jour du watermark (`metrics/retrain_watermark.json`).

## 3) Microservices Docker

- `api-inference` (`docker/Dockerfile.api`)
- `mlflow-server` (`docker/Dockerfile.ml`)
- Jobs ML (`preprocess`, `train`, `evaluate`, `predict-batch`) via `docker-compose.yml`
- Stack Airflow (`webserver`, `scheduler`, `triggerer`, `postgres`) via `docker-compose.airflow.yml`

## 4) Emplacements clés

- DAGs: `airflow/dags/`
- Code ML: `src/`
- API: `api/`
- Optimisation active: `optimisations/prod/`
- Recherche / benchmark: `optimisations/research/`
- Présentation: `streamlit_app.py`, `assets/slides/`
- Documentation: `README.md`, `docs/specifications.md`
