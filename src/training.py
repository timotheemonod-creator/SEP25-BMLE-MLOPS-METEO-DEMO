import joblib
import mlflow
import mlflow.sklearn

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_preparation import build_preprocessor, prepare_dataset_from_processed


def train(processed_path, model_path):
    X, y, cat_cols, num_cols = prepare_dataset_from_processed(processed_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBClassifier(
        n_estimators=796,
        max_depth=10,
        learning_rate=0.11812558198339064,
        subsample=0.7724069195699388,
        colsample_bytree=0.7000620673916282,
        min_child_weight=6,
        reg_alpha=3.479275996079184e-05,
        reg_lambda=1.3538647103817488e-06,
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
    processed_path = "data/processed/weatherAUS_processed.csv"
    model_path = "models/pipeline.joblib"
    metrics = train(processed_path, model_path)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
