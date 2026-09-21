import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="European Power Forecasting Research",layout="wide")
st.title("European Power Forecasting & Trading Research")
st.caption("Chronological forecasting benchmark + research-only spread-direction simulation. Not live trading.")

metrics=json.loads((ROOT/"results/metrics.json").read_text())
sample=pd.read_csv(ROOT/"results/test_predictions_sample.csv")

c1,c2,c3,c4=st.columns(4)
c1.metric("Test MAE",f'{metrics["test"]["mae"]:.2f} EUR/MWh')
c2.metric("Persistence MAE",f'{metrics["test"]["persistence_mae"]:.2f}')
c3.metric("Test R²",f'{metrics["test"]["r2"]:.3f}')
c4.metric("Held-out rows",f'{metrics["split"]["test_rows"]:,}')

st.subheader("Held-out prediction sample")
plot=sample[["time","price_actual","price_day_ahead","prediction"]].copy()
plot["time"]=pd.to_datetime(plot["time"])
st.line_chart(plot.set_index("time"))

st.subheader("Interactive spread-rule replay on bundled held-out sample")
threshold=st.slider("Absolute predicted spread threshold (EUR/MWh)",0.0,10.0,0.5,0.5)
cost=st.slider("Illustrative active-signal cost (EUR/MWh)",0.0,3.0,0.5,0.1)
pred_spread=sample["prediction"]-sample["price_day_ahead"]
realized=sample["price_actual"]-sample["price_day_ahead"]
pos=np.where(pred_spread>threshold,1,np.where(pred_spread<-threshold,-1,0))
active=pos!=0
net=pos*realized-cost*active
a,b,c=st.columns(3)
a.metric("Active rate",f"{active.mean():.1%}")
b.metric("Directional accuracy",f"{((np.sign(realized[active])==pos[active]).mean() if active.any() else 0):.1%}")
c.metric("Mean net unit-MWh capture",f"{net.mean():.2f} EUR/MWh")

with st.expander("Method and limitations"):
    st.write("""
    The offline benchmark uses Spanish hourly energy-market data (2015–2018), day-ahead market features,
    lagged actual prices/load/generation, and lagged observed weather. Model selection uses a chronological
    train/validation/test split. The trading panel is a research-only spread-direction simulation on a bundled
    held-out sample; it is not a live strategy, exchange contract, or realized portfolio P&L.
    """)
