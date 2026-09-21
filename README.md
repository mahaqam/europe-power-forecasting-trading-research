# European Power Forecasting & Trading Research

Portfolio research project for energy-market data science. It builds a chronological price-forecasting benchmark on Spanish hourly energy-market data and adds a deliberately limited, research-only spread-direction simulation.

## What was built

- Data validation and hourly alignment across the energy-market and five-city weather datasets.
- Calendar features plus lagged price, load, generation, and weather features.
- Chronological 70% / 15% / 15% train-validation-test split.
- Ridge and HistGradientBoosting regression compared with persistence and same-hour day-ahead price baselines.
- Validation-selected model refit on train + validation.
- Thresholded spread-direction simulation with an explicit 0.5 EUR/MWh active-signal cost assumption.
- Streamlit research dashboard and a small FastAPI metrics/signal API.

## Verified results

The uploaded energy file contained 35,064 hourly rows and the weather file 178,396 rows. After creating 168-hour lag features, 34,896 rows were usable for modeling.

| Metric | Verified result |
| --- | ---: |
| Held-out test rows | 5,235 |
| Test MAE | 1.506 EUR/MWh |
| Test RMSE | 2.018 EUR/MWh |
| Test R² | 0.935 |
| Persistence baseline MAE | 1.972 EUR/MWh |
| Same-hour day-ahead baseline MAE | 8.601 EUR/MWh |
| MAE improvement vs persistence | 23.7% |
| Selected spread threshold | 0.5 EUR/MWh |
| Directional accuracy on active research signals | 98.0% |
| Mean net unit-MWh spread capture | 8.01 EUR/MWh |
| Always-long baseline net mean | 7.42 EUR/MWh |

The spread simulation is **not** a live trading strategy or exchange-contract P&L. The dataset exhibits a strong positive realized spread in the held-out period, so the always-long comparison is shown explicitly rather than hiding that baseline.

## Leakage controls and methodology

Current-hour observed weather is not used as an input. Weather enters only through lagged observations. The model can use same-hour day-ahead price/load/wind/solar forecast fields plus historical actuals that would already be known. All model selection and threshold selection occur before the final held-out period.

## Reproduce

Dataset:

- Energy + weather: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

```bash
pip install -r requirements.txt
python src/train.py --energy energy_dataset.csv --weather weather_features.csv --output-dir results
```

ZIP files containing one CSV are also accepted.

## Demo

The Streamlit app uses a bundled sample from the held-out predictions so it can run without redistributing the full source dataset. It shows verified benchmark metrics and lets you replay the spread threshold/cost assumptions interactively.

```bash
streamlit run app/dashboard.py
```

FastAPI:

```bash
uvicorn api.main:app --reload
```

## Streamlit deployment

- Repository: `mahaqam/europe-power-forecasting-trading-research`
- Branch: `main`
- Main file: `app/dashboard.py`

No secrets are required for the deterministic public demo.

## Scope

This is a portfolio research prototype. It does not claim production deployment, live market connectivity, execution, portfolio optimization, or realized trading profit.
