# SEP25-BMLE-MLOPS-METEO

Projet MLOps de prédiction de pluie (`RainTomorrow`) avec une chaîne complète:
- entraînement et évaluation de modèle,
- API FastAPI pour l'inférence,
- suivi MLflow,
- versioning DVC + DagsHub,
- orchestration Airflow (prédiction, monitoring, décision de relance Optuna),
- application Streamlit de soutenance (slides + démo live).

Ce README est la référence opérationnelle pour exécuter, comprendre et démontrer le projet.

## 1) Objectif du projet

Objectif métier:
- Prédire la pluie du lendemain pour chaque station météo afin d'aider à la planification opérationnelle.

Objectif technique:
- Avoir un pipeline reproductible et industrialisable,
- Servir les prédictions via API,
- Suivre la qualité du modèle dans le temps (offline + live),
- Déclencher automatiquement une optimisation quand la qualité live se dégrade.

Métriques suivies:
- `recall` (prioritaire),
- `precision`,
- `f1`,
- `roc_auc`,
- `accuracy`.

## 2) Architecture globale

```text
+----------------------+         HTTP         +--------------------------+
| UI Streamlit (demo)  +--------------------->+ FastAPI (/predict,/health)|
+----------+-----------+                      +-------------+------------+
           |                                                    |
           |                                                    v
           |                                            models/pipeline.joblib
           |                                                    |
           |                                                    v
           |                                          outputs/preds_api.csv
           |                                                    |
           |                            +-----------------------+----------------------+
           |                            |                                              |
           |                            v                                              v
           |                 src/live_monitoring.py                          DVC (.dvc)
           |                                                          + DagsHub
           |                            |
           |                            v
           |                     metrics/live_eval.json
           |                            |
           |                            v
           |                  Airflow: weather_main_dag
           |                            |
           |                    branch_on_recall
           |                            |
           |         +------------------+-------------------+
           |         |                                      |
           |         v                                      v
           |   skip_optuna                        trigger_optuna -> optuna_tuning_dag
           |                                                        |
           |                                                        v
           |                              optimisations/prod/optuna_search_recall_small.py
           |                                                        |
           |                         +------------------------------+-------------------+
           |                         |                                                  |
           |                         v                                                  v
           |                models/best_params.json                    metrics/retrain_watermark.json
```

## 3) Outils utilisés et complémentarité

- `scikit-learn`, `imbalanced-learn`, `xgboost`: préprocessing + modèle de classification.
- `FastAPI`: endpoints d'inférence et endpoint de santé.
- `MLflow`: tracking des runs, paramètres, métriques, artefacts.
- `DVC`: versioning des gros fichiers (dataset, logs de prédictions).
- `DagsHub`: stockage distant DVC + support tracking MLflow.
- `Airflow`: orchestration des exécutions planifiées et décisionnelles.
- `Docker` / `Docker Compose`: environnement exécutable de manière reproductible.
- `Streamlit`: interface de soutenance (pédagogie + démonstration live).
- `GitHub Actions`: automatisation de publication des prédictions versionnées.

## 4) Arborescence du dépôt

```text
.
|- airflow/
|  |- dags/
|  |  |- weather_main_dag.py
|  |  `- optuna_tuning_dag.py
|  `- logs/
|- api/
|  `- main.py
|- data/
|  |- raw/
|  |  |- weatherAUS.csv.dvc
|  |  `- weatherAUS.csv
|  `- processed/
|- docker/
|  |- Dockerfile.api
|  |- Dockerfile.ml
|  |- Dockerfile.airflow
|  `- requirements.airflow.extra.txt
|- metrics/
|  |- eval.json
|  |- live_eval.json
|  `- retrain_watermark.json
|- models/
|  |- pipeline.joblib
|  `- best_params.json
|- optimisations/
|  |- prod/
|  |  `- optuna_search_recall_small.py
|  |- research/
|  |  |- benchmark_models.py
|  |  |- lazyml_search.py
|  |  `- optuna_search.py
|  |- optuna_xgb_recall_small.db
|  `- README.md
|- outputs/
|  |- preds_api.csv
|  |- preds_api.csv.dvc
|  `- preds_api_scored.csv
|- scripts/
|  |- predict_all_and_push.sh
|  `- ensure_mlflow_env.sh
|- legacy/
|  `- scripts/
|     |- preprocess.sh
|     |- train.sh
|     |- evaluate.sh
|     |- predict.sh
|     `- cron.txt
|- src/
|  |- data_preparation.py
|  |- training.py
|  |- evaluation.py
|  |- live_monitoring.py
|  `- retrain_dataset_builder.py
|- tests/
|  `- test_health.py
|- .github/workflows/
|  `- publish_preds.yml
|- docker-compose.yml
|- docker-compose.airflow.yml
|- dvc.yaml
|- requirements.in
|- requirements-dev.in
|- streamlit_app.py
`- README.md
```

## 5) Où sont stockées les informations

Dans Git:
- code source,
- DAGs,
- Dockerfiles,
- workflows CI,
- pointeurs DVC (`*.dvc`).

Dans DVC + DagsHub:
- `data/raw/weatherAUS.csv`,
- `outputs/preds_api.csv`,
- autres artefacts volumineux suivis par DVC.

Dans MLflow:
- paramètres, métriques, artefacts des runs d'entraînement et d'optimisation.

Dans Airflow:
- métadonnées d'orchestration (Postgres Airflow),
- logs d'exécution des tasks.

Dans les fichiers de métriques:
- `metrics/live_eval.json`: métriques live sur prédictions scorées,
- `metrics/retrain_watermark.json`: point de consommation des nouvelles données après retrain/optuna.

## 6) Pipelines applicatifs

### 6.1 Pipeline entraînement (offline)

1. `src.data_preparation` prépare les données.
2. `src.training` entraîne le pipeline et sauvegarde `models/pipeline.joblib`.
3. `src.evaluation` calcule les métriques offline (`metrics/eval.json`).

### 6.2 Pipeline inférence et scoring live

1. L'API `/predict` écrit les requêtes/résultats dans `outputs/preds_api.csv`.
2. `src.live_monitoring` enrichit en vérité observée BOM et produit:
   - `outputs/preds_api_scored.csv`,
   - `metrics/live_eval.json`.

### 6.3 Pipeline dataset de retrain

`src.retrain_dataset_builder` construit:
- `data/processed/weatherAUS_retrain.csv` = historique + nouvelles lignes live labellisées exploitables,
- `metrics/retrain_dataset_meta.json` avec `historical_rows`, `added_rows`, `total_rows`.

Important:
- On n'ajoute pas les probabilités de prédiction comme feature de modèle.
- On ajoute des observations reconstruites avec les colonnes du schéma `weatherAUS`.

## 7) Logique détaillée des DAGs

## 7.1 `weather_main_dag`

Planification:
- `0 6,18 * * *` (06:00 et 18:00 UTC).

Enchaînement des tasks:
1. `predict_all_and_push`
2. `compute_live_metrics`
3. `read_quality_metrics`
4. `branch_on_recall`
5. `trigger_optuna` ou `skip_optuna`
6. `join_after_branch`

Règle de décision:
- Déclencher Optuna si et seulement si:
  - `recall_live < RECALL_THRESHOLD` (par défaut `0.59`),
  - et `new_rows_for_retrain >= MIN_NEW_ROWS_FOR_RETRAIN` (par défaut `60`).

Calcul `new_rows_for_retrain`:
- base = lignes live scorées (`truth_available=True`) dédupliquées par (`location`, `feature_date`),
- seules les dates strictement au-delà du `cutoff_date` sont comptées,
- `cutoff_date = max(max_hist_date, last_feature_date_watermark)`.

## 7.2 `optuna_tuning_dag`

Enchaînement des tasks:
1. `run_optuna`
   - reconstruit d'abord le dataset de retrain,
   - lance Optuna sur `data/processed/weatherAUS_retrain.csv`.
2. `save_best_params`
   - exporte les meilleurs hyperparamètres dans `models/best_params.json`.
3. `retrain_with_best`
   - réentraîne un modèle complet sur `weatherAUS_retrain.csv` avec les meilleurs paramètres,
   - remplace `models/pipeline.joblib` (modèle actif API).
4. `mark_retrain_consumed`
   - met à jour `metrics/retrain_watermark.json` avec la dernière `feature_date` consommée.

Effet du watermark:
- après un Optuna réussi, les lignes déjà utilisées ne sont plus recomptées comme "nouvelles".
- cela évite de relancer Optuna en boucle sur le même lot de données.

## 7.3 Schéma de décision (cas possibles)

```text
weather_main_dag
  -> predict_all_and_push
  -> compute_live_metrics
  -> read_quality_metrics
       lit recall_live + new_rows_for_retrain
       avec cutoff = max(max_hist_date, watermark)
  -> branch_on_recall
       si new_rows_for_retrain < MIN_NEW_ROWS_FOR_RETRAIN
            -> skip_optuna
       sinon si recall_live < RECALL_THRESHOLD
            -> trigger_optuna -> optuna_tuning_dag
       sinon
            -> skip_optuna
```

Table de comportements:
- Cas A: `new_rows` insuffisant, recall faible -> pas d'Optuna.
- Cas B: `new_rows` suffisant, recall bon -> pas d'Optuna.
- Cas C: `new_rows` suffisant, recall faible -> Optuna déclenché.
- Cas D: Optuna terminé avec succès -> watermark mis à jour, `new_rows` retombe.

## 7.4 Sorties des DAGs: fichiers, XCom, consommateurs

### `weather_main_dag`

| Task | Sortie | Type | Consommé par | But |
|---|---|---|---|---|
| `ensure_latest_model` | `return_value` (chemin modèle, mtime, metadata) | XCom | Diagnostic Airflow / debug | Vérifier qu'un modèle actif existe avant prédiction |
| `predict_all_and_push` | `outputs/preds_api.csv` | CSV | `src.live_monitoring`, CI (`publish_preds.yml`), Streamlit | Journal brut des prédictions API |
| `compute_live_metrics` | `outputs/preds_api_scored.csv` | CSV | `src.retrain_dataset_builder`, Streamlit | Prédictions enrichies avec vérité observée (`truth_available`, `y_true`, `feature_date`) |
| `compute_live_metrics` | `metrics/live_eval.json` | JSON | `read_quality_metrics`, Streamlit | Qualité live (accuracy/precision/recall/f1/roc_auc) |
| `read_quality_metrics` | `return_value` (`live`, `new_rows_for_retrain`, `retrain_cutoff_date`) | XCom | `branch_on_recall` | Centraliser les signaux de décision |
| `branch_on_recall` | branche cible (`trigger_optuna` ou `skip_optuna`) | XCom interne de branchement | Airflow (scheduler) | Appliquer la règle métier |

### `optuna_tuning_dag`

| Task | Sortie | Type | Consommé par | But |
|---|---|---|---|---|
| `run_optuna` | `data/processed/weatherAUS_retrain.csv` | CSV | `optimisations/prod/optuna_search_recall_small.py`, `retrain_with_best` | Dataset retrain = historique + nouvelles lignes labellisées |
| `run_optuna` | `metrics/retrain_dataset_meta.json` | JSON | Opérations / debug | Vérifier `historical_rows`, `added_rows`, `total_rows` |
| `run_optuna` | `optimisations/optuna_xgb_recall_small.db` | SQLite DB | `save_best_params` | Historique complet des trials Optuna |
| `save_best_params` | `models/best_params.json` | JSON | `retrain_with_best`, Streamlit (explication) | Publier les meilleurs hyperparamètres |
| `retrain_with_best` | `models/pipeline.joblib` | Binaire modèle | API `/predict`, `ensure_latest_model` | Activer le dernier modèle retrainé |
| `retrain_with_best` | `models/model_metadata.json` | JSON | `ensure_latest_model`, audit | Traçabilité du modèle entraîné (date, params, rows) |
| `mark_retrain_consumed` | `metrics/retrain_watermark.json` | JSON | `read_quality_metrics` (dans DAG principal) | Empêcher de recompter le même lot de lignes live |

### Exemple de payload XCom clé (`read_quality_metrics`)

```json
{
  "live": {
    "recall_live": 0.4193,
    "scored_rows": 138
  },
  "new_rows_for_retrain": 0,
  "retrain_cutoff_date": "2026-02-18"
}
```

Interprétation:
- `branch_on_recall` décide à partir de ce payload.
- Si `new_rows_for_retrain < MIN_NEW_ROWS_FOR_RETRAIN`, Optuna est skip même si le recall live est faible.

## 8) Installation et démarrage

## 8.1 Pré-requis

- Python 3.12,
- Docker + Docker Compose v2,
- accès Internet (BOM, DagsHub, etc.),
- token DagsHub si push DVC/MLflow distant.

## 8.2 Environnement Python local

```bash
python -m venv .venv_meteo
source .venv_meteo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Stratégie dépendances (recommandée):
- `requirements.in`: dépendances runtime minimales maintenues à la main.
- `requirements-dev.in`: dépendances de dev/test.
- `requirements.txt`: lock actuel (historique) encore supporté pour compatibilité.

Compilation lock (optionnelle, pip-tools):
```bash
python -m pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
```

Si Streamlit est manquant:
```bash
python -m pip install streamlit
```

## 8.3 Stack Docker API + MLflow

```bash
docker compose up -d --build api-inference mlflow-server
docker compose ps
```

## 8.4 Stack Airflow

```bash
docker build -f docker/Dockerfile.airflow -t meteo-airflow:2.10.5 .

docker compose --env-file .env.airflow -f docker-compose.yml -f docker-compose.airflow.yml up -d \
  airflow-postgres airflow-init airflow-webserver airflow-scheduler airflow-triggerer
```

Déclencher le DAG principal:
```bash
docker compose --env-file .env.airflow -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags trigger weather_main_dag
```

Lister les runs:
```bash
docker compose --env-file .env.airflow -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags list-runs -d weather_main_dag
```

Voir l'état des tasks d'un run:
```bash
docker compose --env-file .env.airflow -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow tasks states-for-dag-run weather_main_dag 'manual__YYYY-MM-DDTHH:MM:SS+00:00'
```

## 8.5 Streamlit (soutenance)

```bash
streamlit run streamlit_app.py
```

## 9) URLs utiles

- FastAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI health: `http://127.0.0.1:8000/health`
- Airflow: `http://127.0.0.1:8080`
- MLflow: `http://127.0.0.1:5000`
- Streamlit: `http://127.0.0.1:8501`

## 10) Commandes de vérification / tests

Tests Python:
```bash
pytest -q
```

Vérification syntaxe DAGs:
```bash
python -m py_compile airflow/dags/weather_main_dag.py
python -m py_compile airflow/dags/optuna_tuning_dag.py
```

Vérification import DAGs Airflow:
```bash
docker compose --env-file .env.airflow -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-webserver \
airflow dags list-import-errors
```

Recalcul métriques live:
```bash
python -m src.live_monitoring
cat metrics/live_eval.json
```

Recalcul des nouvelles lignes de retrain:
```bash
python -m src.retrain_dataset_builder
cat metrics/retrain_dataset_meta.json
```

## 11) CI GitHub Actions

Workflow: `.github/workflows/publish_preds.yml`

Déclenchement:
- manuel (`workflow_dispatch`),
- planifié (`15 6,18 * * *`).

Secrets requis:
- `DAGSHUB_USERNAME`,
- `DAGSHUB_TOKEN`.

Comportement:
1. checkout branche,
2. installation DVC,
3. validation de `outputs/preds_api.csv`,
4. `dvc add outputs/preds_api.csv`,
5. commit/push des pointeurs DVC,
6. `dvc push` vers DagsHub.

## 12) Notes d'exploitation

- `trigger_optuna=success` signifie que le run Optuna a été créé; vérifier ensuite le statut du `optuna_tuning_dag`.
- Les lignes live ajoutées manuellement doivent aller dans `outputs/preds_api.csv` (pas uniquement dans `preds_api_scored.csv`).
- Le watermark évite de retraiter en boucle les mêmes données.
- Pour les diagnostics XCom avancés, utiliser un script Python dans le conteneur Airflow (certaines commandes CLI XCom sont absentes selon version).

## 13) Résumé soutenance

Narratif recommandé:
1. besoin métier,
2. baseline et métriques,
3. API en production locale,
4. suivi live des performances,
5. logique de décision Airflow,
6. déclenchement conditionnel d'Optuna,
7. consommation des nouvelles données via watermark,
8. traçabilité Git + DVC + MLflow + Airflow.

