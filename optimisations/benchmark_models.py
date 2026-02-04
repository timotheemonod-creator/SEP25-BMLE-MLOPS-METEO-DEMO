import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset


def main():
    raw_path = "data/raw/weatherAUS.csv"
    X, y, cat_cols, num_cols = prepare_dataset(raw_path)
    y = np.asarray(y).astype(int).ravel()

    models = {
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    mlflow.set_experiment("weather-ml")
    with mlflow.start_run(run_name="benchmark_rf_xgb"):
        results = []
        for name, model in models.items():
            pipe = Pipeline(
                steps=[
                    ("preprocessing", build_preprocessor(cat_cols, num_cols)),
                    ("scaling", StandardScaler(with_mean=False)),
                    ("model", model),
                ]
            )
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
            row = {
                "model": name,
                "roc_auc": float(np.mean(scores["test_roc_auc"])),
                "precision": float(np.mean(scores["test_precision"])),
                "recall": float(np.mean(scores["test_recall"])),
                "f1": float(np.mean(scores["test_f1"])),
            }
            for k, v in row.items():
                if k != "model":
                    mlflow.log_metric(f"{name}_cv_{k}", v)
            results.append(row)

        df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
        df.to_csv("metrics/benchmark_rf_xgb.csv", index=False)
        mlflow.log_artifact("metrics/benchmark_rf_xgb.csv")
        print(df)


if __name__ == "__main__":
    main()
