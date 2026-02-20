from __future__ import annotations

import io
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


RAW_HIST_PATH = Path(os.getenv("RAW_HIST_PATH", "data/raw/weatherAUS.csv"))
SCORED_PREDS_PATH = Path(os.getenv("SCORED_PREDS_PATH", "outputs/preds_api_scored.csv"))
OUT_PATH = Path(os.getenv("RETRAIN_RAW_PATH", "data/processed/weatherAUS_retrain.csv"))
META_PATH = Path(os.getenv("RETRAIN_META_PATH", "metrics/retrain_dataset_meta.json"))

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

TARGET_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow",
]


def _normalize_col_name(col: str) -> str:
    return (
        str(col)
        .lower()
        .replace("â°", "c")
        .replace("°", "c")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _bom_url(station_name: str, year_month: str) -> str:
    station_code = STATIONS.get(station_name)
    if station_code is None:
        raise KeyError(f"Unknown station: {station_name}")
    return f"https://www.bom.gov.au/climate/dwo/{year_month}/text/{station_code}.{year_month}.csv"


def _fetch_bom_month(station_name: str, year_month: str) -> pd.DataFrame:
    r = requests.get(_bom_url(station_name, year_month), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    lines = r.text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.split(",")[0].strip() == "")
    return pd.read_csv(io.StringIO(r.text), skiprows=header_idx, sep=",")


def _extract_feature_row(df_month: pd.DataFrame, station_name: str, feature_date: date, y_true: int) -> dict[str, Any] | None:
    date_col = _pick_col(df_month, ["Date"])
    if date_col is None:
        return None

    dates = pd.to_datetime(df_month[date_col], errors="coerce").dt.date
    row = df_month.loc[dates == feature_date]
    if row.empty:
        return None
    row = row.iloc[-1]

    c_min = _pick_col(df_month, ["Minimum temperature (°C)", "Minimum temperature (Ã‚Â°C)"])
    c_max = _pick_col(df_month, ["Maximum temperature (°C)", "Maximum temperature (Ã‚Â°C)"])
    c_rain = _pick_col(df_month, ["Rainfall (mm)"])
    c_evap = _pick_col(df_month, ["Evaporation (mm)"])
    c_sun = _pick_col(df_month, ["Sunshine (hours)"])
    c_wgd = _pick_col(df_month, ["Direction of maximum wind gust"])
    c_wgs = _pick_col(df_month, ["Speed of maximum wind gust (km/h)"])
    c_w9d = _pick_col(df_month, ["9am wind direction"])
    c_w3d = _pick_col(df_month, ["3pm wind direction"])
    c_w9s = _pick_col(df_month, ["9am wind speed (km/h)"])
    c_w3s = _pick_col(df_month, ["3pm wind speed (km/h)"])
    c_h9 = _pick_col(df_month, ["9am relative humidity (%)"])
    c_h3 = _pick_col(df_month, ["3pm relative humidity (%)"])
    c_p9 = _pick_col(df_month, ["9am MSL pressure (hPa)"])
    c_p3 = _pick_col(df_month, ["3pm MSL pressure (hPa)"])
    c_c9 = _pick_col(df_month, ["9am cloud amount (oktas)"])
    c_c3 = _pick_col(df_month, ["3pm cloud amount (oktas)"])
    c_t9 = _pick_col(df_month, ["9am Temperature (°C)", "9am Temperature (Ã‚Â°C)"])
    c_t3 = _pick_col(df_month, ["3pm Temperature (°C)", "3pm Temperature (Ã‚Â°C)"])

    rain = pd.to_numeric(pd.Series([row[c_rain]]) if c_rain else pd.Series([None]), errors="coerce").iloc[0]
    rain_today = "Yes" if pd.notna(rain) and float(rain) > 0 else "No"

    return {
        "Date": feature_date.isoformat(),
        "Location": station_name,
        "MinTemp": row[c_min] if c_min else None,
        "MaxTemp": row[c_max] if c_max else None,
        "Rainfall": row[c_rain] if c_rain else None,
        "Evaporation": row[c_evap] if c_evap else None,
        "Sunshine": row[c_sun] if c_sun else None,
        "WindGustDir": row[c_wgd] if c_wgd else None,
        "WindGustSpeed": row[c_wgs] if c_wgs else None,
        "WindDir9am": row[c_w9d] if c_w9d else None,
        "WindDir3pm": row[c_w3d] if c_w3d else None,
        "WindSpeed9am": row[c_w9s] if c_w9s else None,
        "WindSpeed3pm": row[c_w3s] if c_w3s else None,
        "Humidity9am": row[c_h9] if c_h9 else None,
        "Humidity3pm": row[c_h3] if c_h3 else None,
        "Pressure9am": row[c_p9] if c_p9 else None,
        "Pressure3pm": row[c_p3] if c_p3 else None,
        "Cloud9am": row[c_c9] if c_c9 else None,
        "Cloud3pm": row[c_c3] if c_c3 else None,
        "Temp9am": row[c_t9] if c_t9 else None,
        "Temp3pm": row[c_t3] if c_t3 else None,
        "RainToday": rain_today,
        "RainTomorrow": "Yes" if int(y_true) == 1 else "No",
    }


def build_retrain_dataset() -> dict[str, int]:
    hist = pd.read_csv(RAW_HIST_PATH)
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    max_hist_date = hist["Date"].max().date()
    known_keys = {(d.date().isoformat(), loc) for d, loc in zip(hist["Date"], hist["Location"])}

    if not SCORED_PREDS_PATH.exists():
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(OUT_PATH, index=False)
        meta = {"historical_rows": int(len(hist)), "added_rows": 0, "total_rows": int(len(hist))}
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    scored = pd.read_csv(SCORED_PREDS_PATH)
    required = {"truth_available", "y_true", "feature_date", "location"}
    if not required.issubset(scored.columns):
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(OUT_PATH, index=False)
        meta = {"historical_rows": int(len(hist)), "added_rows": 0, "total_rows": int(len(hist))}
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    scored = scored[scored["truth_available"] == True].copy()  # noqa: E712
    scored = scored.dropna(subset=["y_true", "location"])
    feature_ts = pd.to_datetime(scored["feature_date"], errors="coerce")
    target_ts = pd.to_datetime(scored.get("target_date"), errors="coerce")
    inferred = target_ts - pd.Timedelta(days=1)
    feature_ts = feature_ts.fillna(inferred)
    scored["feature_date"] = feature_ts.dt.date
    scored = scored.dropna(subset=["feature_date"])
    scored = scored[scored["feature_date"] > max_hist_date]
    scored = scored.sort_values("feature_date").drop_duplicates(["location", "feature_date"], keep="last")

    if scored.empty:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(OUT_PATH, index=False)
        meta = {"historical_rows": int(len(hist)), "added_rows": 0, "total_rows": int(len(hist))}
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    added_rows: list[dict[str, Any]] = []

    for _, r in scored.iterrows():
        station = str(r["location"])
        fdate = r["feature_date"]
        key = (fdate.isoformat(), station)
        if key in known_keys:
            continue
        ym = fdate.strftime("%Y%m")
        cache_key = (station, ym)
        if cache_key not in cache:
            try:
                cache[cache_key] = _fetch_bom_month(station, ym)
            except Exception:
                cache[cache_key] = pd.DataFrame()
        if cache[cache_key].empty:
            continue
        row = _extract_feature_row(cache[cache_key], station, fdate, int(r["y_true"]))
        if row is None:
            continue
        added_rows.append(row)

    if added_rows:
        add_df = pd.DataFrame(added_rows)
        for col in TARGET_COLUMNS:
            if col not in add_df.columns:
                add_df[col] = None
        add_df = add_df[TARGET_COLUMNS]
        merged = pd.concat([hist[TARGET_COLUMNS], add_df], ignore_index=True)
    else:
        merged = hist[TARGET_COLUMNS].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    meta = {
        "historical_rows": int(len(hist)),
        "added_rows": int(len(added_rows)),
        "total_rows": int(len(merged)),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(build_retrain_dataset(), indent=2))
