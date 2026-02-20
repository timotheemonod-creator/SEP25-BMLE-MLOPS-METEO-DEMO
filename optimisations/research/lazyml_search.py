import mlflow
import numpy as np
import pandas as pd
import importlib

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

from src.data_preparation import build_preprocessor, prepare_dataset


def to_df(X, prefix="f"):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=[f"{prefix}_{i}" for i in range(X.shape[1])])


def main():
    mlflow.sklearn.autolog(disable=True)
    try:
        importlib.import_module("mlflow.xgboost")
        mlflow.xgboost.autolog(disable=True)
    except Exception:
        pass
    try:
        importlib.import_module("mlflow.lightgbm")
        mlflow.lightgbm.autolog(disable=True)
    except Exception:
        pass

    mlflow.end_run()

    raw_path = "data/raw/weatherAUS.csv"
    X, y, cat_cols, num_cols = prepare_dataset(raw_path)
    y = np.asarray(y).astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_df = to_df(X_train_t, "x")
    X_test_df = to_df(X_test_t, "x")

    mlflow.set_experiment("weather-ml")
    with mlflow.start_run(run_name="lazyml"):
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)

        if models is not None and not models.empty:
            models.to_csv("metrics/lazyml_models.csv", index=True)
            mlflow.log_artifact("metrics/lazyml_models.csv")

            top = models.sort_values("ROC AUC", ascending=False).head(5)
            for idx, row in top.iterrows():
                mlflow.log_metric(f"lazyml_{idx}_roc_auc", float(row["ROC AUC"]))
            print(top)
        else:
            print("LazyML returned no models.")


if __name__ == "__main__":
    main()
