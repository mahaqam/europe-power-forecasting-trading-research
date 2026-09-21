from __future__ import annotations
import argparse, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def read_csv_any(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(names)!=1:
                raise ValueError(f"Expected exactly one CSV in {path}, found {names}")
            with z.open(names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)

def build_dataset(energy_path: str, weather_path: str):
    energy=read_csv_any(energy_path)
    weather=read_csv_any(weather_path)
    energy["time"]=pd.to_datetime(energy["time"], utc=True)
    weather["dt_iso"]=pd.to_datetime(weather["dt_iso"], utc=True)
    weather["city_name"]=weather["city_name"].astype(str).str.strip()
    weather_num=["temp","pressure","humidity","wind_speed","rain_1h","clouds_all"]
    city_hour=weather.groupby(["city_name","dt_iso"],as_index=False)[weather_num].mean()
    weather_hour=(city_hour.groupby("dt_iso",as_index=False)[weather_num].mean()
                  .rename(columns={"dt_iso":"time"}))
    df=energy.merge(weather_hour,on="time",how="left").sort_values("time").reset_index(drop=True)

    df["hour"]=df["time"].dt.hour
    df["dow"]=df["time"].dt.dayofweek
    df["month"]=df["time"].dt.month
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]=np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]=np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"]=np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"]=np.cos(2*np.pi*(df["month"]-1)/12)

    lag_spec={
        "price actual":[1,24,168],
        "total load actual":[1,24,168],
        "generation wind onshore":[1,24],
        "generation solar":[1,24],
        "temp":[1,24],
        "humidity":[1,24],
        "wind_speed":[1,24],
        "rain_1h":[1,24],
        "clouds_all":[1,24],
    }
    for col,lags in lag_spec.items():
        for lag in lags:
            df[f"{col}_lag{lag}"]=df[col].shift(lag)

    features=[
        "price day ahead","total load forecast","forecast solar day ahead","forecast wind onshore day ahead",
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "price actual_lag1","price actual_lag24","price actual_lag168",
        "total load actual_lag1","total load actual_lag24","total load actual_lag168",
        "generation wind onshore_lag1","generation wind onshore_lag24",
        "generation solar_lag1","generation solar_lag24",
        "temp_lag1","temp_lag24","humidity_lag1","humidity_lag24",
        "wind_speed_lag1","wind_speed_lag24","rain_1h_lag1","rain_1h_lag24",
        "clouds_all_lag1","clouds_all_lag24",
    ]
    model_df=df[["time","price actual"]+features].iloc[168:].dropna(
        subset=["price actual","price day ahead"]).reset_index(drop=True)
    return energy, weather, model_df, features

def metrics(y,p):
    return {
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(mean_squared_error(y,p)**0.5),
        "r2":float(r2_score(y,p)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--energy", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--output-dir", default="results")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    energy,weather,df,features=build_dataset(args.energy,args.weather)
    n=len(df); i1=int(n*.70); i2=int(n*.85)
    train,val,test=df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]
    Xtr,ytr=train[features],train["price actual"]
    Xv,yv=val[features],val["price actual"]
    Xt,yt=test[features],test["price actual"]

    models={
        "ridge":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model",Ridge(alpha=10.0)),
        ]),
        "hist_gbr":Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("model",HistGradientBoostingRegressor(
                max_iter=300,learning_rate=.06,max_leaf_nodes=31,
                l2_regularization=1.0,random_state=42)),
        ]),
    }
    rows=[]; val_predictions={}
    for name,model in models.items():
        t=time.perf_counter(); model.fit(Xtr,ytr); fit=time.perf_counter()-t
        t=time.perf_counter(); pred=model.predict(Xv); infer=time.perf_counter()-t
        val_predictions[name]=pred
        r={"model":name,**{f"val_{k}":v for k,v in metrics(yv,pred).items()},
           "fit_seconds":fit,"val_infer_seconds":infer}
        rows.append(r)

    rows.append({"model":"day_ahead_baseline",
                 **{f"val_{k}":v for k,v in metrics(yv,val["price day ahead"]).items()}})
    rows.append({"model":"persistence_baseline",
                 **{f"val_{k}":v for k,v in metrics(yv,val["price actual_lag1"]).items()}})
    comp=pd.DataFrame(rows)
    best=comp[comp["model"].isin(models)].sort_values("val_mae").iloc[0]["model"]

    train_val=df.iloc[:i2]
    model=models[best]
    t=time.perf_counter(); model.fit(train_val[features],train_val["price actual"]); fit_final=time.perf_counter()-t
    t=time.perf_counter(); pred_test=model.predict(Xt); infer_final=time.perf_counter()-t
    test_m=metrics(yt,pred_test)

    # Research-only spread-direction simulation; not a live trading P&L.
    cost=0.5
    pval=val_predictions[best]
    real_val=yv.values-val["price day ahead"].values
    pred_spread_val=pval-val["price day ahead"].values
    threshold_rows=[]
    for thr in [0.0,0.5,1.0,2.0,3.0,5.0,7.5,10.0]:
        pos=np.where(pred_spread_val>thr,1,np.where(pred_spread_val<-thr,-1,0))
        active=pos!=0
        net=pos*real_val-cost*active
        threshold_rows.append({
            "threshold_eur_per_mwh":thr,
            "active_rate":float(active.mean()),
            "directional_accuracy_active":float((np.sign(real_val[active])==pos[active]).mean()) if active.any() else None,
            "mean_net_eur_per_mwh_all_hours":float(net.mean()),
            "mean_net_eur_per_mwh_active":float(net[active].mean()) if active.any() else None,
        })
    threshold_df=pd.DataFrame(threshold_rows)
    threshold=float(threshold_df.sort_values("mean_net_eur_per_mwh_all_hours",ascending=False).iloc[0]["threshold_eur_per_mwh"])

    day_test=test["price day ahead"].values
    real_test=yt.values-day_test
    pred_spread=pred_test-day_test
    pos=np.where(pred_spread>threshold,1,np.where(pred_spread<-threshold,-1,0))
    active=pos!=0
    net=pos*real_test-cost*active
    signal={
        "threshold_eur_per_mwh":threshold,
        "cost_assumption_eur_per_mwh_active":cost,
        "active_rate":float(active.mean()),
        "directional_accuracy_active":float((np.sign(real_test[active])==pos[active]).mean()),
        "mean_net_eur_per_mwh_all_hours":float(net.mean()),
        "mean_net_eur_per_mwh_active":float(net[active].mean()),
        "n_active_hours":int(active.sum()),
        "always_long_net_mean_eur_per_mwh":float((real_test-cost).mean()),
    }

    final={
        "dataset":{"energy_rows":int(len(energy)),"weather_rows":int(len(weather)),"usable_model_rows":int(len(df))},
        "split":{"train_rows":int(len(train)),"validation_rows":int(len(val)),"test_rows":int(len(test)),
                 "test_start":str(test["time"].min()),"test_end":str(test["time"].max())},
        "selected_model":best,
        "test":{**test_m,
                "persistence_mae":float(mean_absolute_error(yt,test["price actual_lag1"])),
                "day_ahead_mae":float(mean_absolute_error(yt,test["price day ahead"])),
                "final_fit_seconds":fit_final,"test_infer_seconds":infer_final},
        "signal":signal,
        "limitations":[
            "Chronological split; no shuffled validation.",
            "Current-hour observed weather is excluded; only lagged observed weather is used.",
            "Research-only spread-direction simulation, not live trading or a specific exchange contract P&L.",
            "Transaction/slippage assumption is illustrative at 0.5 EUR/MWh per active signal."
        ]
    }
    comp.to_csv(out/"model_comparison.csv",index=False)
    threshold_df.to_csv(out/"threshold_evaluation.csv",index=False)
    with open(out/"metrics.json","w") as f: json.dump(final,f,indent=2)

    pred_out=pd.DataFrame({
        "time":test["time"].astype(str),
        "price_actual":yt.values,
        "price_day_ahead":day_test,
        "prediction":pred_test,
        "predicted_spread":pred_spread,
        "realized_spread":real_test,
        "position":pos,
        "net_unit_mwh_eur":net,
    })
    pred_out.tail(300).to_csv(out/"test_predictions_sample.csv",index=False)
    print(json.dumps(final,indent=2))

if __name__=="__main__":
    main()
