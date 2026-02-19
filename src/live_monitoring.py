from __future__ import annotations

from datetime import date
from pathlib import Path
import io
import json
import math
import os
import csv
from typing import Dict, Tuple

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PRED_LOG_PATH = Path(os.getenv("PRED_LOG_PATH", "outputs/preds_api.csv"))
SCORED_LOG_PATH = Path(os.getenv("SCORED_LOG_PATH", "outputs/preds_api_scored.csv"))
LIVE_METRICS_PATH = Path(os.getenv("LIVE_METRICS_PATH", "metrics/live_eval.json"))
LIVE_WINDOW = int(os.getenv("LIVE_WINDOW", "200"))

STATIONS = {
    "Adelaide": "IDCJDW5081",
    "Albany": "IDCJDW6001",
    "Albury-Wodonga": "IDCJDW2002",
    "Alice Springs": "IDCJDW8002",
    "Ballarat": "IDCJDW3005",
    "Bendigo": "IDCJDW3008",
    "Brisbane": "IDCJDW4019",
    "Broome": "IDCJDW6015",
    "Cairns": "IDCJDW4024",
    "Canberra": "IDCJDW2801",
    "Casey": "IDCJDW9203",
    "Christmas Island": "IDCJDW6026",
    "Cocos Islands": "IDCJDW6027",
    "Darwin": "IDCJDW8014",
    "Davis": "IDCJDW9201",
    "Devonport": "IDCJDW7013",
    "Gold Coast": "IDCJDW4050",
    "Hobart": "IDCJDW7021",
    "Kalgoorlie-Boulder": "IDCJDW6061",
    "Launceston": "IDCJDW7025",
    "Lord Howe Island": "IDCJDW2077",
    "Macquarie Island": "IDCJDW9204",
    "Mawson": "IDCJDW9202",
    "Melbourne": "IDCJDW3033",
    "Mount Gambier": "IDCJDW5041",
    "Norfolk Island": "IDCJDW2100",
    "Penrith": "IDCJDW2111",
    "Perth": "IDCJDW6111",
    "Port Lincoln": "IDCJDW5055",
    "Renmark": "IDCJDW5059",
    "Sydney": "IDCJDW2124",
    "Tennant Creek": "IDCJDW8045",
    "Townsville": "IDCJDW4128",
    "Tuggeranong": "IDCJDW2802",
    "Wollongong": "IDCJDW2146",
    "Wynyard": "IDCJDW7057",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def _bom_url(station_name: str, year_month: str) -> str:
    station_code = STATIONS.get(station_name)
    if station_code is None:
        raise KeyError(f"Unknown station: {station_name}")
    return f"https://www.bom.gov.au/climate/dwo/{year_month}/text/{station_code}.{year_month}.csv"


def _safe_float(val):
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _fetch_bom_month(station_name: str, year_month: str) -> pd.DataFrame:
    response = requests.get(_bom_url(station_name, year_month), headers=HEADERS, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.split(",")[0].strip() == "")
    except StopIteration as exc:
        raise ValueError("Header line not found in BOM CSV") from exc
    return pd.read_csv(io.StringIO(response.text), skiprows=header_index, sep=",")


def _extract_rainfall_for_date(df: pd.DataFrame, target: date) -> float | None:
    date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    rain_col = next((c for c in df.columns if "Rainfall" in str(c) and "(mm)" in str(c)), None)
    if date_col is None or rain_col is None:
        return None

    dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
    row = df.loc[dates == target]
    if row.empty:
        return None
    return _safe_float(row.iloc[-1][rain_col])


def score_predictions() -> Dict[str, float | int | str]:
    if not PRED_LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {PRED_LOG_PATH}")

    preds = _load_prediction_log(PRED_LOG_PATH)
    required = {"logged_at_utc", "target_date", "location", "predicted_rain"}
    missing = required.difference(preds.columns)
    if missing:
        raise ValueError(f"Missing expected columns in prediction log: {sorted(missing)}")

    # Some historical logs mix naive timestamps and timezone-aware ones.
    # Parse robustly so we don't drop valid rows from monitoring/retrain logic.
    try:
        preds["logged_at_utc"] = pd.to_datetime(preds["logged_at_utc"], errors="coerce", utc=True, format="mixed")
    except TypeError:
        preds["logged_at_utc"] = pd.to_datetime(preds["logged_at_utc"], errors="coerce", utc=True)
    target_ts = pd.to_datetime(preds["target_date"], errors="coerce")
    preds["target_date"] = target_ts.dt.date
    # Backward compatibility: older logs may not have feature_date.
    # In this project target_date is feature_date + 1 day, so we can infer it.
    if "feature_date" not in preds.columns:
        preds["feature_date"] = (target_ts - pd.Timedelta(days=1)).dt.date
    else:
        feature_ts = pd.to_datetime(preds["feature_date"], errors="coerce")
        inferred = (target_ts - pd.Timedelta(days=1)).dt.date
        preds["feature_date"] = feature_ts.dt.date
        preds.loc[preds["feature_date"].isna(), "feature_date"] = inferred[preds["feature_date"].isna()]
    preds = preds.dropna(subset=["logged_at_utc", "target_date", "location"])
    preds = preds.sort_values("logged_at_utc").drop_duplicates(["location", "target_date"], keep="last")

    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    y_true = []
    y_pred = []
    observed_rain = []
    truth_available = []

    for _, row in preds.iterrows():
        station = str(row["location"])
        target_date = row["target_date"]
        ym = target_date.strftime("%Y%m")
        cache_key = (station, ym)

        df_month = cache.get(cache_key)
        if df_month is None:
            try:
                df_month = _fetch_bom_month(station, ym)
            except Exception:
                df_month = pd.DataFrame()
            cache[cache_key] = df_month

        rain_mm = _extract_rainfall_for_date(df_month, target_date) if not df_month.empty else None
        has_truth = rain_mm is not None
        truth_available.append(has_truth)
        observed_rain.append(rain_mm)

        if has_truth:
            y_true.append(1 if rain_mm > 0 else 0)
            y_pred.append(int(row["predicted_rain"]))

    preds["observed_rainfall_mm"] = pd.to_numeric(observed_rain, errors="coerce")
    preds["truth_available"] = truth_available
    preds["y_true"] = preds["observed_rainfall_mm"].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else None)
    )
    preds["predicted_rain"] = pd.to_numeric(preds["predicted_rain"], errors="coerce")

    SCORED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(SCORED_LOG_PATH, index=False)

    eval_df = preds[preds["truth_available"] == True].copy()  # noqa: E712
    eval_df = eval_df.dropna(subset=["y_true", "predicted_rain"])
    if LIVE_WINDOW > 0 and len(eval_df) > LIVE_WINDOW:
        eval_df = eval_df.tail(LIVE_WINDOW)

    if eval_df.empty:
        metrics = {
            "status": "no_ground_truth_available",
            "accuracy_live": None,
            "precision_live": None,
            "recall_live": 1.0,
            "f1_live": None,
            "roc_auc_live": None,
            "scored_rows": 0,
            "window": LIVE_WINDOW,
        }
    else:
        y_true = eval_df["y_true"].astype(int)
        y_pred = eval_df["predicted_rain"].astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        eval_df["rain_probability"] = pd.to_numeric(eval_df["rain_probability"], errors="coerce")
        auc_df = eval_df.dropna(subset=["rain_probability"])
        if len(auc_df) > 0 and len(auc_df["y_true"].astype(int).unique()) > 1:
            roc_auc_live = float(roc_auc_score(auc_df["y_true"].astype(int), auc_df["rain_probability"]))
        else:
            roc_auc_live = None
        metrics = {
            "status": "ok",
            "accuracy_live": float(accuracy_score(y_true, y_pred)),
            "precision_live": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_live": float(recall),
            "f1_live": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc_live": roc_auc_live,
            "scored_rows": int(len(eval_df)),
            "positives_true": int(y_true.sum()),
            "window": LIVE_WINDOW,
        }

    LIVE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _load_prediction_log(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for row in reader:
            if not row:
                continue
            if len(row) == 7:
                rows.append(
                    {
                        "logged_at_utc": row[0],
                        "target_date": row[1],
                        "location": row[2],
                        "use_latest": row[3],
                        "station_name": row[4],
                        "rain_probability": row[5],
                        "predicted_rain": row[6],
                    }
                )
            elif len(row) >= 8:
                rows.append(
                    {
                        "logged_at_utc": row[0],
                        "feature_date": row[1],
                        "target_date": row[2],
                        "location": row[3],
                        "use_latest": row[4],
                        "station_name": row[5],
                        "rain_probability": row[6],
                        "predicted_rain": row[7],
                    }
                )
    if not rows:
        raise ValueError(f"No usable prediction rows in {path}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = score_predictions()
    print(json.dumps(results, indent=2))
