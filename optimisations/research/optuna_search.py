import mlflow
import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset


def objective(trial, X, y, cat_cols, num_cols):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaling", StandardScaler(with_mean=False)),
            ("model", model),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores))


def main():
    raw_path = "data/raw/weatherAUS.csv"
    X, y, cat_cols, num_cols = prepare_dataset(raw_path)
    y = np.asarray(y).astype(int).ravel()

    mlflow.set_experiment("weather-ml")
    with mlflow.start_run(run_name="optuna_xgb_cv"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X, y, cat_cols, num_cols), timeout=1800)

        # Log best ROC AUC and params
        mlflow.log_metric("optuna_best_cv_roc_auc", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(k, v)

        # Evaluate best params on extra metrics
        best_params = study.best_params.copy()
        best_params.update({
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        })

        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(cat_cols, num_cols)),
                ("scaling", StandardScaler(with_mean=False)),
                ("model", XGBClassifier(**best_params)),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        mlflow.log_metric("optuna_cv_roc_auc", float(np.mean(scores["test_roc_auc"])))
        mlflow.log_metric("optuna_cv_precision", float(np.mean(scores["test_precision"])))
        mlflow.log_metric("optuna_cv_recall", float(np.mean(scores["test_recall"])))
        mlflow.log_metric("optuna_cv_f1", float(np.mean(scores["test_f1"])))

        print("Best CV ROC AUC:", study.best_value)
        print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
