from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
import joblib
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import requests
import math

app = FastAPI(title="Meteo RainTomorrow API", version="0.1.1")

MODEL_PATH = Path("models/pipeline.joblib")
PRED_LOG_PATH = Path("outputs/preds_api.csv")

# Charge d'abord .env puis le fichier "env" (présent dans ton repo)
load_dotenv()
if Path("env").exists():
    load_dotenv("env")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Missing API_KEY environment variable")

security = HTTPBearer()

class WeatherFeatures(BaseModel):
    Date: date
    Location: str
    MinTemp: Optional[float] = Field(default=None)
    MaxTemp: Optional[float] = Field(default=None)
    Rainfall: Optional[float] = Field(default=None)
    Evaporation: Optional[float] = Field(default=None)
    Sunshine: Optional[float] = Field(default=None)
    WindGustDir: Optional[str] = Field(default=None)
    WindGustSpeed: Optional[float] = Field(default=None)
    WindDir9am: Optional[str] = Field(default=None)
    WindDir3pm: Optional[str] = Field(default=None)
    WindSpeed9am: Optional[float] = Field(default=None)
    WindSpeed3pm: Optional[float] = Field(default=None)
    Humidity9am: Optional[float] = Field(default=None)
    Humidity3pm: Optional[float] = Field(default=None)
    Pressure9am: Optional[float] = Field(default=None)
    Pressure3pm: Optional[float] = Field(default=None)
    Cloud9am: Optional[float] = Field(default=None)
    Cloud3pm: Optional[float] = Field(default=None)
    Temp9am: Optional[float] = Field(default=None)
    Temp3pm: Optional[float] = Field(default=None)
    RainToday: Optional[str] = Field(default=None, description="Yes/No")

class PredictionResponse(BaseModel):
    rain_tomorrow: int
    rain_probability: float

station = {
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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_bom_url(station_name: str) -> str:
    today = datetime.today()
    year_month = today.strftime("%Y%m")
    station_code = station.get(station_name)
    if station_code is None:
        raise ValueError(f"Unknown station name: {station_name}")
    base_url = "https://www.bom.gov.au/climate/dwo/"
    subdir = "/text/"
    csv_url = f"{base_url}{year_month}{subdir}{station_code}.{year_month}.csv"
    return csv_url

def safe_float(val) -> Optional[float]:
    if val is None:
        return None

    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return None

    try:
        f = float(val)
        if math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None

def fetch_weather_data(station_name: str):
    r = requests.get(get_bom_url(station_name), headers=headers)
    r.raise_for_status()

    lines = r.text.splitlines()

    try:
        header_index = next(
            i for i, line in enumerate(lines) if line.split(",")[0].strip() == ""
        )
    except StopIteration:
        raise ValueError("Header line 'Date,' not found in BOM file")

    df = pd.read_csv(io.StringIO(r.text), skiprows=header_index, sep=",")
    return df

def latest_weather_features(station_name: str) -> WeatherFeatures:
    df = fetch_weather_data(station_name)
    latest = df.iloc[-1]

    rainfall = safe_float(latest.get("Rainfall (mm)"))

    weather_payload = WeatherFeatures(
        Date=pd.to_datetime(latest["Date"]).date(),
        Location=station_name,
        MinTemp=safe_float(latest.get("Minimum temperature (Â°C)")),
        MaxTemp=safe_float(latest.get("Maximum temperature (Â°C)")),
        Rainfall=rainfall,
        Evaporation=safe_float(latest.get("Evaporation (mm)")),
        Sunshine=safe_float(latest.get("Sunshine (hours)")),
        WindGustDir=str(latest.get("Direction of maximum wind gust") or ""),
        WindGustSpeed=safe_float(latest.get("Speed of maximum wind gust (km/h)")),
        WindDir9am=str(latest.get("9am wind direction") or ""),
        WindDir3pm=str(latest.get("3pm wind direction") or ""),
        WindSpeed9am=safe_float(latest.get("9am wind speed (km/h)")),
        WindSpeed3pm=safe_float(latest.get("3pm wind speed (km/h)")),
        Humidity9am=safe_float(latest.get("9am relative humidity (%)")),
        Humidity3pm=safe_float(latest.get("3pm relative humidity (%)")),
        Pressure9am=safe_float(latest.get("9am MSL pressure (hPa)")),
        Pressure3pm=safe_float(latest.get("3pm MSL pressure (hPa)")),
        Cloud9am=safe_float(latest.get("9am cloud amount (oktas)")),
        Cloud3pm=safe_float(latest.get("3pm cloud amount (oktas)")),
        Temp9am=safe_float(latest.get("9am Temperature (Â°C)")),
        Temp3pm=safe_float(latest.get("3pm Temperature (Â°C)")),
        RainToday="Yes" if (rainfall is not None and rainfall > 0) else "No",
    )
    return weather_payload

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    status_model = MODEL_PATH.exists()
    return {"status": "ok", "model_ready": status_model}

@app.get("/latest_weather", response_model=WeatherFeatures)
def get_latest_weather(station_name: str = Query(..., description="Nom de la station météo", enum=list(station.keys()))):
    return latest_weather_features(station_name)

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_token)])
def predict(
    payload: WeatherFeatures | None = Body(default=None),
    use_latest: bool = Query(False),
    station_name: Optional[str] = Query(
        default=None,
        description="Nom de la station météo",
        enum=list(station.keys())
    ),
):
    if use_latest:
        if not station_name:
            raise HTTPException(status_code=400, detail="station_name is required when use_latest is true")
        features = latest_weather_features(station_name)
    elif payload is not None:
        features = payload
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide payload or set use_latest=true"
        )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([features.model_dump()])
    date_series = pd.to_datetime(df["Date"])
    df["Dayofyear"] = date_series.dt.dayofyear
    df["Month"] = date_series.dt.month
    df = df.drop(columns=["Date"])

    try:
        y_proba = model.predict_proba(df)[:, 1][0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {exc}") from exc

    y_pred = int(y_proba >= 0.5)

    try:
        PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_date": str(features.Date),
            "location": features.Location,
            "use_latest": bool(use_latest),
            "station_name": station_name,
            "rain_probability": float(y_proba),
            "predicted_rain": y_pred,
        }
        df_log = pd.DataFrame([row])
        header = not PRED_LOG_PATH.exists()
        df_log.to_csv(PRED_LOG_PATH, mode="a", header=header, index=False)
    except Exception as exc:
        print(f"Warning: failed to log prediction: {exc}")

    return PredictionResponse(rain_tomorrow=y_pred, rain_probability=float(y_proba))
