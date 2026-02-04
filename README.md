## Projet Météo - Baseline MLOps

### Prérequis
- Python 3.10+ (WSL recommandé)
- Dataset Kaggle `weatherAUS.csv` placé ici :
  `data/raw/weatherAUS.csv`

### Installation
```bash
cd .
make setup
```

### Entraînement
```bash
make train
```
Le modèle est enregistré dans `models/pipeline.joblib`.

### Évaluation
```bash
make evaluate
```

### API (FastAPI)
```bash
make api
```
Puis ouvrir `http://127.0.0.1:8000/docs`.

Exemple de payload:
```json
{
  "Date": "2015-06-01",
  "Location": "Canberra",
  "MinTemp": 5.1,
  "MaxTemp": 14.2,
  "Rainfall": 0.0,
  "Evaporation": 2.3,
  "Sunshine": 7.5,
  "WindGustDir": "NW",
  "WindGustSpeed": 35,
  "WindDir9am": "N",
  "WindDir3pm": "NW",
  "WindSpeed9am": 9,
  "WindSpeed3pm": 15,
  "Humidity9am": 81,
  "Humidity3pm": 56,
  "Pressure9am": 1018.4,
  "Pressure3pm": 1015.2,
  "Cloud9am": 7,
  "Cloud3pm": 5,
  "Temp9am": 9.2,
  "Temp3pm": 13.7,
  "RainToday": "No"
}
```

### Conteneurisation (Phase 1)
```bash
docker build -t meteo-api .
docker run -p 8000:8000 meteo-api
```

### Scripts
```bash
./scripts/preprocess.sh
./scripts/train.sh
./scripts/evaluate.sh
./scripts/predict.sh
```

### Structure
```

├── data/raw/              # weatherAUS.csv
├── data/processed/
├── models/                # pipeline.joblib
├── outputs/               # predictions_daily.csv
├── src/                   # code Python
└── scripts/               # wrappers shell
```

### Schéma du pipeline (corrigé)
```
[weatherAUS.csv]
        |
        v
data/raw/weatherAUS.csv
        |
        v
src/data_preparation.py
  - drop NA sur cible
  - outliers -> NaN (WindSpeed9am, Evaporation, Rainfall)
  - features temps (Dayofyear, Month)
  - séparation X / y
  - liste colonnes cat/num
        |
        v
src/training.py
  - split train/test
  - build pipeline:
      * imputation (fit sur train)
      * one-hot (fit sur train)
      * features sin/cos (Dayofyear, Month)
      * scaling
      * SMOTE
      * XGBoost
  - entraînement
  - métriques
  - sauvegarde -> models/pipeline.joblib
        |
        v
src/evaluation.py
  - reload modèle
  - évaluation sur test (mêmes métriques)
```

### Schéma complet
Voir `docs/architecture.md`.
