## README de branche — Phase 2 (Microservices, Suivi & Versioning)

### Objectif
Documenter tout le travail réalisé sur la branche `feature/mlflow-dvc-dagshub` pour la Phase 2 :
- Suivi d’expériences MLflow (params, métriques, artefacts)
- Versioning des données/modèles via DVC + Dagshub
- Pipeline DVC
- Benchmarks et optimisation hyperparamètres (Optuna)

### DVC + Dagshub
1. Initialisation DVC et configuration du remote Dagshub.
2. Tracking DVC des éléments suivants :
   - `data/` (données brutes)
   - `outputs/` (résultats/artefacts)
   - `models/pipeline.joblib` (binaire uniquement, le code reste en Git)
3. Pipeline DVC généré (`dvc.yaml`, `dvc.lock`) :
   - `train` → génère `models/pipeline.joblib`
   - `evaluate` → génère `metrics/eval.json`
   - `predict` → génère `outputs/predictions_daily.csv`

### MLflow (Dagshub)
- Tracking centralisé via Dagshub (UI Experiments).
- Logging :
  - paramètres (`mlflow.log_params`)
  - métriques (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`)
  - modèle en artefact

Tracking URI (Dagshub) :
    https://dagshub.com/timotheemonod-creator/SEP25-BMLE-MLOPS-METEO.mlflow

### Benchmarks & Optimisations
Tout est versionné dans `optimisations/` :
- `benchmark_models.py` + `benchmark_models.csv` (benchmark multi-modèles)
- `benchmark_rf_xgb.csv` (comparaison RF vs XGB)
- `lazyml_search.py` + `lazyml_models.csv` (tentative LazyML)
- `optuna_search.py` (Optuna ROC AUC)
- `optuna_search_recall_small.py` (Optuna Recall)
- `optuna_xgb_recall_small.db` (storage Optuna)

### Résultat du benchmark RF vs XGB (CV 5 folds)
- ROC AUC :
  - RF ≈ 0.8919
  - XGB ≈ 0.8905
- Recall/F1 :
  - XGB meilleur → choisi pour optimisation

### Optuna (ROC AUC)
Meilleur ROC AUC observé :
- Best CV ROC AUC ≈ 0.9051  
- Paramètres :
  - `n_estimators`: 598
  - `max_depth`: 8
  - `learning_rate`: 0.06023585382561457
  - `subsample`: 0.8218142946598608
  - `colsample_bytree`: 0.9462874097203866
  - `min_child_weight`: 2
  - `reg_alpha`: 0.00047133466463315196
  - `reg_lambda`: 0.0012785745299627295

### Optuna (Recall)
Meilleur Recall observé :
- Best CV Recall ≈ 0.59927  
- Paramètres :
  - `n_estimators`: 796
  - `max_depth`: 10
  - `learning_rate`: 0.11812558198339064
  - `subsample`: 0.7724069195699388
  - `colsample_bytree`: 0.7000620673916282
  - `min_child_weight`: 6
  - `reg_alpha`: 3.479275996079184e-05
  - `reg_lambda`: 1.3538647103817488e-06

### Paramètres finaux appliqués
Le modèle final dans `src/training.py` utilise les paramètres Optuna Recall (meilleure sensibilité).

### Liens utiles
Dagshub repo :
    https://dagshub.com/timotheemonod-creator/SEP25-BMLE-MLOPS-METEO

MLflow experiments :
    https://dagshub.com/timotheemonod-creator/SEP25-BMLE-MLOPS-METEO.mlflow
