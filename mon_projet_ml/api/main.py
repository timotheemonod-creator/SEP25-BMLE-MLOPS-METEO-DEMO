from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/pipeline.joblib")

app = FastAPI(title="Meteo RainTomorrow API", version="0.1.0")


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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: WeatherFeatures):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame([payload.model_dump()])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Dayofyear"] = df["Date"].dt.dayofyear
    df["Month"] = df["Date"].dt.month
    df = df.drop(columns=["Date"])
    y_proba = model.predict_proba(df)[:, 1][0]
    y_pred = int(y_proba >= 0.5)
    return PredictionResponse(rain_tomorrow=y_pred, rain_probability=float(y_proba))
