import joblib

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_preparation import prepare_dataset


def evaluate(raw_path, model_path):
    X, y, _, _ = prepare_dataset(raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


if __name__ == "__main__":
    raw_path = "data/raw/weatherAUS.csv"
    model_path = "models/pipeline.joblib"
    metrics = evaluate(raw_path, model_path)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
