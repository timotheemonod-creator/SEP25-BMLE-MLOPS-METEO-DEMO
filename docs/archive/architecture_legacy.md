# Schéma complet de la solution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SOURCES DE DONNÉES                               │
│                                                                              │
│  Kaggle: weatherAUS.csv                                                       │
│  (10 ans d’observations quotidiennes en Australie)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
                        ┌──────────────────────────┐
                        │ data/raw/weatherAUS.csv  │
                        └─────────────┬────────────┘
                                      │
                                      v
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRÉPARATION DES DONNÉES                               │
│  src/data_preparation.py                                                      │
│  - drop NA cible (RainTomorrow)                                               │
│  - outliers -> NaN (WindSpeed9am, Evaporation, Rainfall)                      │
│  - features temps: Dayofyear, Month                                           │
│  - séparation X/y + détection colonnes cat/num                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ENTRAÎNEMENT (BASELINE)                               │
│  src/training.py                                                              │
│  - split train/test (80/20, stratifié)                                        │
│  - pipeline sklearn/imblearn:                                                 │
│      * imputation (fit sur train)                                             │
│      * one-hot encode (fit sur train)                                         │
│      * sin/cos (Dayofyear, Month)                                             │
│      * scaling (with_mean=False)                                              │
│      * SMOTE                                                                  │
│      * XGBoost                                                                │
│  - métriques: accuracy, precision, recall, f1, roc_auc                        │
│  - export modèle -> models/pipeline.joblib                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ÉVALUATION OFFLINE                                   │
│  src/evaluation.py                                                            │
│  - recharge pipeline.joblib                                                   │
│  - métriques sur test                                                         │
└──────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                             ORCHESTRATION LOCALE
═══════════════════════════════════════════════════════════════════════════════

Makefile (mon_projet_ml/Makefile)
- make setup      -> pip install -r requirements.txt
- make preprocess -> python -m src.data_preparation
- make train      -> python -m src.training
- make evaluate   -> python -m src.evaluation
- make predict    -> python -m src.predict_batch
- make api        -> uvicorn api.main:app

Scripts (mon_projet_ml/scripts/)
- preprocess.sh   -> python -m src.data_preparation
- train.sh        -> python -m src.training
- evaluate.sh     -> python -m src.evaluation
- predict.sh      -> python -m src.predict_batch

Cron (mon_projet_ml/scripts/cron.txt)
- 02:00 preprocess
- 02:15 train
- 02:30 evaluate
- 02:45 predict

═══════════════════════════════════════════════════════════════════════════════
                          PRÉDICTION BATCH QUOTIDIENNE
═══════════════════════════════════════════════════════════════════════════════

src/predict_batch.py
- lit data/raw/weatherAUS.csv
- sélectionne la dernière date disponible
- applique le pipeline entraîné
- écrit une ligne par exécution

Artefact :
- outputs/predictions_daily.csv
  (run_at, date, location, rain_probability, rain_tomorrow)

═══════════════════════════════════════════════════════════════════════════════
                               API D’INFÉRENCE
═══════════════════════════════════════════════════════════════════════════════

FastAPI (mon_projet_ml/api/main.py)
- GET  /health
- POST /predict
    Entrée : features météo + Date
    Traitement :
      * calcule Dayofyear + Month depuis Date
      * drop Date
      * predict_proba
    Sortie :
      * rain_tomorrow (0/1)
      * rain_probability (float)

Swagger : http://127.0.0.1:8000/docs

═══════════════════════════════════════════════════════════════════════════════
                               CONTENEURISATION
═══════════════════════════════════════════════════════════════════════════════

Dockerfile (mon_projet_ml/Dockerfile)
- base python:3.12-slim
- installe requirements.txt
- copie api/, src/, models/
- expose 8000
- lance uvicorn

Exécution :
- docker build -t meteo-api .
- docker run -p 8000:8000 -v $(pwd)/models:/app/models meteo-api

═══════════════════════════════════════════════════════════════════════════════
                               DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════════

docs/specifications.md
- cas d’usage
- objectifs
- métriques
- hypothèses
- sources et prétraitement
- API et tests manuels

README.md
- commandes
- schémas
- usage API
- Docker

═══════════════════════════════════════════════════════════════════════════════
                                 ARBORESCENCE
═══════════════════════════════════════════════════════════════════════════════

mon_projet_ml/
├── api/                    # FastAPI
├── data/
│   ├── raw/                # weatherAUS.csv
│   └── processed/
├── models/                 # pipeline.joblib
├── outputs/                # predictions_daily.csv
├── scripts/                # preprocess/train/evaluate/predict + cron
├── src/                    # data_preparation, training, evaluation, predict_batch
├── docs/                   # specifications.md
├── requirements.txt
├── Makefile
└── README.md
```
