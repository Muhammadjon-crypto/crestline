"""Train the production model (+3h, the honest headline horizon) and
save it with everything needed to reproduce a forecast at serve time.
"""
import json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "08076700"
UPSTREAM = {"08075900": "greens_us75", "08076000": "greens_hou",
            "08076500": "halls", "08076180": "garners"}
HORIZON_H = 3
TEST_FRACTION = 0.25

stage = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
stage.index = pd.to_datetime(stage.index, utc=True)
stage.columns = [str(c) for c in stage.columns]

rain = pd.read_csv("data/rain_stage4.csv", index_col=0, parse_dates=True)
rain.index = pd.to_datetime(rain.index, utc=True)
rain = rain.loc[:, ~rain.T.duplicated()].sort_index()

def rain_features(rain_df, target_index):
    f = pd.DataFrame(index=rain_df.index)
    for c in rain_df.columns:
        s = rain_df[c].fillna(0.0)
        f[f"{c}_1h"] = s
        for w in (3, 6, 12, 24, 72, 168):
            f[f"{c}_{w}h"] = s.rolling(w, min_periods=1).sum()
    return f.reindex(target_index, method="ffill")

def build_features(stage_df, rain_df):
    t = stage_df[TARGET]
    X = pd.DataFrame(index=stage_df.index)
    for sid, name in UPSTREAM.items():
        s = stage_df[sid]
        for lag_h in (0, 1, 2, 3, 4, 5, 6):
            X[f"{name}_lag{lag_h}h"] = s.shift(lag_h * 4)
        for w in (4, 12, 24):
            X[f"{name}_roc{w//4}h"] = s.diff(w)
    for lag_h in (0, 1, 2, 3, 6):
        X[f"target_lag{lag_h}h"] = t.shift(lag_h * 4)
    X["target_roc1h"] = t.diff(4)
    X["target_roc3h"] = t.diff(12)
    X["month"] = X.index.month
    X["hour"] = X.index.hour
    return X.join(rain_features(rain_df, stage_df.index))

if __name__ == "__main__":
    t = stage[TARGET]
    X = build_features(stage, rain)
    y = t.shift(-HORIZON_H * 4)
    ok = X.notna().all(axis=1) & y.notna() & t.notna()
    X, y, p = X[ok], y[ok], t[ok]

    sp = int(len(X) * (1 - TEST_FRACTION))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                          early_stopping=True,
                                          validation_fraction=0.15, random_state=0)
    model.fit(X.iloc[:sp], y.iloc[:sp])

    pred = pd.Series(model.predict(X.iloc[sp:]), index=X.index[sp:])
    yte, pte = y.iloc[sp:], p.iloc[sp:]
    hi = yte > t.quantile(0.95)
    r = lambda a, b, m=None: float(np.sqrt((((a-b) if m is None else (a[m]-b[m]))**2).mean()))

    metrics = {
        "horizon_hours": HORIZON_H,
        "trained_through": str(X.index[sp]),
        "test_start": str(X.index[sp]), "test_end": str(X.index[-1]),
        "n_test": int(len(yte)),
        "rmse_ft": r(yte, pred), "persistence_rmse_ft": r(yte, pte),
        "skill": 1 - r(yte, pred) / r(yte, pte),
        "rmse_high_ft": r(yte, pred, hi), "persistence_rmse_high_ft": r(yte, pte, hi),
        "skill_high": 1 - r(yte, pred, hi) / r(yte, pte, hi),
        "high_water_threshold_ft": float(t.quantile(0.95)),
        "feature_names": list(X.columns),
    }
    joblib.dump(model, "data/crestline_model_3h.joblib")
    json.dump(metrics, open("data/model_metrics.json", "w"), indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_names"}, indent=2))
    print(f"\n{len(X.columns)} features")
    print("Saved data/crestline_model_3h.joblib and data/model_metrics.json")
