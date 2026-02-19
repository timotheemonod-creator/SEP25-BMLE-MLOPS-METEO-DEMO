# Optimisations

## Dossier `prod/`
Contient les scripts utilisés par les DAGs Airflow en production.

- `optuna_search_recall_small.py`: recherche Optuna orientée recall utilisée par `optuna_tuning_dag`.

## Dossier `research/`
Contient les expérimentations et benchmarks historiques, non exécutés par les DAGs.

- `benchmark_models.py`, `benchmark_models.csv`, `benchmark_rf_xgb.csv`
- `lazyml_search.py`, `lazyml_models.csv`
- `optuna_search.py` (ancienne variante)

## Artefacts DVC

Le pointeur DVC `optimisations/optuna_xgb_recall_small.db.dvc` est conservé à la racine du dossier `optimisations` pour compatibilité avec l'existant.
