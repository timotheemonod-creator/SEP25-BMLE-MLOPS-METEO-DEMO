import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def sin_transform(x, period):
    return np.sin(x / period * 2 * np.pi)


def cos_transform(x, period):
    return np.cos(x / period * 2 * np.pi)


def load_raw(path):
    return pd.read_csv(path)


def basic_cleaning(df):
    df = df.copy()
    df = df.dropna(subset=["RainTomorrow"])
    df.loc[df["WindSpeed9am"] > 100, "WindSpeed9am"] = np.nan
    df.loc[df["Evaporation"] > 140, "Evaporation"] = np.nan
    df.loc[df["Rainfall"] > 350, "Rainfall"] = np.nan
    return df


def add_time_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dayofyear"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month
    df = df.drop(columns=["Date"])
    return df


def build_preprocessor(cat_cols, num_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, cat_cols),
            ("num", numeric_pipe, num_cols),
            ("dayofyear_sin", FunctionTransformer(sin_transform, kw_args={"period": 365}), ["Dayofyear"]),
            ("dayofyear_cos", FunctionTransformer(cos_transform, kw_args={"period": 365}), ["Dayofyear"]),
            ("month_sin", FunctionTransformer(sin_transform, kw_args={"period": 12}), ["Month"]),
            ("month_cos", FunctionTransformer(cos_transform, kw_args={"period": 12}), ["Month"]),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_dataset(raw_path):
    df = load_raw(raw_path)
    df = basic_cleaning(df)
    df = add_time_features(df)
    X = df.drop("RainTomorrow", axis=1)
    y = (df["RainTomorrow"] == "Yes").astype(int)
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(include=["int32", "int64", "float64"]).columns
    num_cols = [c for c in num_cols if c not in {"Dayofyear", "Month"}]
    return X, y, cat_cols, num_cols




def save_processed(raw_path, processed_path):
    df = load_raw(raw_path)
    df = basic_cleaning(df)
    df = add_time_features(df)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)


def prepare_dataset_from_processed(processed_path):
    df = pd.read_csv(processed_path)
    X = df.drop("RainTomorrow", axis=1)
    y = (df["RainTomorrow"] == "Yes").astype(int)
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(include=["int32", "int64", "float64"]).columns
    num_cols = [c for c in num_cols if c not in {"Dayofyear", "Month"}]
    return X, y, cat_cols, num_cols

if __name__ == "__main__":
    raw_path = "data/raw/weatherAUS.csv"
    processed_path = "data/processed/weatherAUS_processed.csv"
    save_processed(raw_path, processed_path)
    X, y, cat_cols, num_cols = prepare_dataset_from_processed(processed_path)
    print(f"Rows: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Categorical: {len(cat_cols)} | Numeric: {len(num_cols)}")
