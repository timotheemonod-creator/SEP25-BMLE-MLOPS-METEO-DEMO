import os
from datetime import datetime, timezone

import joblib
import pandas as pd


RAW_PATH = os.environ.get("RAW_PATH", "data/raw/weatherAUS.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/pipeline.joblib")
OUT_PATH = os.environ.get("PREDICTIONS_PATH", "outputs/predictions_daily.csv")


def build_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dayofyear"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month
    return df


def main():
    df = pd.read_csv(RAW_PATH)
    df = build_features(df)

    # take the latest available date as a daily batch example
    df = df.sort_values("Date")
    latest = df.tail(1).drop(columns=["RainTomorrow"])

    model = joblib.load(MODEL_PATH)
    proba = float(model.predict_proba(latest)[:, 1][0])
    pred = int(proba >= 0.5)

    out = pd.DataFrame(
        [
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "date": str(latest["Date"].iloc[0].date()),
                "location": latest["Location"].iloc[0],
                "rain_probability": proba,
                "rain_tomorrow": pred,
            }
        ]
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    header = not os.path.exists(OUT_PATH)
    out.to_csv(OUT_PATH, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
