"""Merge rainfall into the model and re-test the June 2026 event.

THE QUESTION: on 06-15/16, did the model see the surge coming, or was it
flat until the water was already up? Rainfall is the only input that can
know about a storm before the bayou does.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "08076700"
UPSTREAM = {"08075900": "greens_us75", "08076000": "greens_hou",
            "08076500": "halls", "08076180": "garners"}
HORIZONS = (3, 6, 12)
TEST_FRACTION = 0.25

stage = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
stage.index = pd.to_datetime(stage.index, utc=True)
stage.columns = [str(c) for c in stage.columns]

rain = pd.read_csv("data/rain_stage4.csv", index_col=0, parse_dates=True)
rain.index = pd.to_datetime(rain.index, utc=True)
rain = rain.loc[:, ~rain.T.duplicated()]          # drop duplicate grid cells
rain = rain.sort_index()
print(f"Distinct rainfall series: {list(rain.columns)}")

# rolling sums computed on the HOURLY grid, then upsampled (no double count)
feat_rain = pd.DataFrame(index=rain.index)
for c in rain.columns:
    s = rain[c].fillna(0.0)
    feat_rain[f"{c}_1h"] = s
    for w in (3, 6, 12, 24, 72, 168):
        feat_rain[f"{c}_{w}h"] = s.rolling(w, min_periods=1).sum()
feat_rain = feat_rain.reindex(stage.index, method="ffill")

t = stage[TARGET]

def build(with_rain):
    X = pd.DataFrame(index=stage.index)
    for sid, name in UPSTREAM.items():
        s = stage[sid]
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
    if with_rain:
        X = X.join(feat_rain)
    return X

results, keep6 = [], None
for H in HORIZONS:
    y = t.shift(-H * 4)
    for label, wr in [("gauges only", False), ("+ rainfall", True)]:
        X = build(wr)
        ok = X.notna().all(axis=1) & y.notna() & t.notna()
        Xf, yf, pf = X[ok], y[ok], t[ok]
        sp = int(len(Xf) * (1 - TEST_FRACTION))
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                          early_stopping=True,
                                          validation_fraction=0.15, random_state=0)
        m.fit(Xf.iloc[:sp], yf.iloc[:sp])
        pred = pd.Series(m.predict(Xf.iloc[sp:]), index=Xf.index[sp:])
        yte, pte = yf.iloc[sp:], pf.iloc[sp:]
        hi = yte > t.quantile(0.95)
        r = lambda a, b, msk=None: np.sqrt((((a-b) if msk is None else (a[msk]-b[msk]))**2).mean())
        results.append({"horizon": H, "model": label,
                        "RMSE": r(yte, pred), "skill": 1 - r(yte, pred)/r(yte, pte),
                        "RMSE_hi": r(yte, pred, hi),
                        "skill_hi": 1 - r(yte, pred, hi)/r(yte, pte, hi)})
        print(f"  +{H:2d}h {label:12s} skill {results[-1]['skill']:5.1%}  "
              f"high-water {results[-1]['skill_hi']:5.1%}")
        if H == 6 and wr:
            keep6 = pd.DataFrame({"truth": yte, "persistence": pte, "crestline": pred})

res = pd.DataFrame(results)
res.to_csv("data/rain_comparison.csv", index=False)
print("\n=== GAUGES ONLY vs + RAINFALL ===")
print(res.to_string(index=False))

# --- the event that mattered --------------------------------------------
k = keep6.copy()
k.index = k.index + pd.Timedelta(hours=6)          # valid time
ev = k.loc["2026-06-15":"2026-06-17"]
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(ev.index, ev["truth"], lw=2.6, color="#1f4e79", label="Observed stage")
ax.plot(ev.index, ev["persistence"], lw=1.4, ls="--", color="#999",
        label="Persistence (issued 6h earlier)")
ax.plot(ev.index, ev["crestline"], lw=2.2, color="#e07b39",
        label="CrestLine + rainfall (issued 6h earlier)")
ax.set_title("Greens Bayou at Ley Rd — June 2026 event, with rainfall features")
ax.set_ylabel("Stage (ft)"); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("data/event_with_rain.png", dpi=160)
print("\nSaved data/event_with_rain.png and data/rain_comparison.csv")
