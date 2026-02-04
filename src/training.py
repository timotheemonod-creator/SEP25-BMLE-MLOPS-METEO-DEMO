import joblib
import mlflow
import mlflow.sklearn

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset


def train(raw_path, model_path):
    X, y, cat_cols, num_cols = prepare_dataset(raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaling", StandardScaler(with_mean=False)),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )

    mlflow.set_experiment("weather-ml")

    with mlflow.start_run(run_name="training"):
        mlflow.log_params(model.get_params())
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    joblib.dump(pipeline, model_path)
    return metrics


if __name__ == "__main__":
    raw_path = "data/raw/weatherAUS.csv"
    model_path = "models/pipeline.joblib"
    metrics = train(raw_path, model_path)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
