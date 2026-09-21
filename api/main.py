from pathlib import Path
import json
from fastapi import FastAPI
ROOT=Path(__file__).resolve().parents[1]
METRICS=json.loads((ROOT/"results/metrics.json").read_text())
app=FastAPI(title="European Power Forecasting Research API")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/metrics")
def metrics():
    return METRICS

@app.get("/signal")
def signal(predicted_price:float, day_ahead_price:float, threshold:float=0.5):
    spread=predicted_price-day_ahead_price
    position=1 if spread>threshold else (-1 if spread<-threshold else 0)
    return {"predicted_spread":spread,"position":position,"scope":"research-only"}
