import mlflow
import optuna
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset


BEST_PARAMS = {
    "n_estimators": 598,
    "max_depth": 8,
    "learning_rate": 0.06023585382561457,
    "subsample": 0.8218142946598608,
    "colsample_bytree": 0.9462874097203866,
    "min_child_weight": 2,
    "reg_alpha": 0.00047133466463315196,
    "reg_lambda": 0.0012785745299627295,
}


class NoImprovementStopper:
    def __init__(self, min_delta=1e-4, patience=10):
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.count = 0

    def __call__(self, study, trial):
        if trial.value is None:
            return
        if self.best is None or (trial.value - self.best) > self.min_delta:
            self.best = trial.value
            self.count = 0
        else:
            self.count += 1
        if self.count >= self.patience:
            study.stop()


def objective(trial, X, y, cat_cols, num_cols):
    # narrow search around best params
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(cat_cols, num_cols)),
            ("scaling", StandardScaler(with_mean=False)),
            ("model", XGBClassifier(**params)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring={"recall": "recall"})
    return float(np.mean(scores["test_recall"]))


def main():
    raw_path = os.getenv("OPTUNA_RAW_PATH", "data/raw/weatherAUS.csv")
    X, y, cat_cols, num_cols = prepare_dataset(raw_path)
    y = np.asarray(y).astype(int).ravel()

    optuna_db = Path(__file__).resolve().parent / "optuna_xgb_recall_small.db"
    storage = f"sqlite:///{optuna_db}"
    study = optuna.create_study(direction="maximize", study_name="xgb_recall_small", storage=storage, load_if_exists=True)

    # seed with best-known params
    study.enqueue_trial(BEST_PARAMS)

    mlflow.set_experiment("weather-ml")
    with mlflow.start_run(run_name="optuna_xgb_recall_small"):
        stopper = NoImprovementStopper(min_delta=1e-4, patience=10)
        study.optimize(lambda t: objective(t, X, y, cat_cols, num_cols), n_trials=30, callbacks=[stopper])

        # log best recall
        mlflow.log_metric("optuna_best_cv_recall", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(k, v)

        # log full metrics for best params
        best_params = study.best_params.copy()
        best_params.update({"eval_metric": "logloss", "random_state": 42, "n_jobs": -1})

        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(cat_cols, num_cols)),
                ("scaling", StandardScaler(with_mean=False)),
                ("model", XGBClassifier(**best_params)),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1", "accuracy": "accuracy"}
        )

        cv_roc_auc = float(np.mean(scores["test_roc_auc"]))
        cv_precision = float(np.mean(scores["test_precision"]))
        cv_recall = float(np.mean(scores["test_recall"]))
        cv_f1 = float(np.mean(scores["test_f1"]))
        cv_accuracy = float(np.mean(scores["test_accuracy"]))

        # Metrics namespaced for detailed tracking
        mlflow.log_metric("optuna_cv_roc_auc", cv_roc_auc)
        mlflow.log_metric("optuna_cv_precision", cv_precision)
        mlflow.log_metric("optuna_cv_recall", cv_recall)
        mlflow.log_metric("optuna_cv_f1", cv_f1)
        mlflow.log_metric("optuna_cv_accuracy", cv_accuracy)
        # Canonical aliases so MLflow/DagsHub default metric columns are populated
        mlflow.log_metric("roc_auc", cv_roc_auc)
        mlflow.log_metric("precision", cv_precision)
        mlflow.log_metric("recall", cv_recall)
        mlflow.log_metric("f1", cv_f1)
        mlflow.log_metric("accuracy", cv_accuracy)

        print("Best CV Recall:", study.best_value)
        print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
