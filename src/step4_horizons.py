"""Train at multiple horizons -> the lead-time table for the metrics page.
Also dumps the single biggest flood event in the test set as a plot:
truth vs persistence vs CrestLine. That plot is the video's money shot.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "08076700"
UPSTREAM = {"08075900": "greens_us75", "08076000": "greens_hou",
            "08076500": "halls", "08076180": "garners"}
HORIZONS = (1, 3, 6, 12, 18)
TEST_FRACTION = 0.25

df = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
df.columns = [str(c) for c in df.columns]
t = df[TARGET]

def build_X():
    X = pd.DataFrame(index=df.index)
    for sid, name in UPSTREAM.items():
        s = df[sid]
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
    return X

Xall = build_X()
hi_thresh = t.quantile(0.95)
rows, keep = [], {}

for H in HORIZONS:
    y = t.shift(-H * 4)
    ok = Xall.notna().all(axis=1) & y.notna() & t.notna()
    X, yy, pp = Xall[ok], y[ok], t[ok]
    split = int(len(X) * (1 - TEST_FRACTION))
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                      early_stopping=True, validation_fraction=0.15,
                                      random_state=0)
    m.fit(X.iloc[:split], yy.iloc[:split])
    pred = pd.Series(m.predict(X.iloc[split:]), index=X.index[split:])
    yte, pte = yy.iloc[split:], pp.iloc[split:]
    high = yte > hi_thresh

    def rmse(a, b, mask=None):
        e = (a - b) if mask is None else (a[mask] - b[mask])
        return np.sqrt((e ** 2).mean()) if len(e) else np.nan

    rp, rm = rmse(yte, pte), rmse(yte, pred)
    rph, rmh = rmse(yte, pte, high), rmse(yte, pred, high)
    rows.append({"horizon_h": H, "n_test": len(yte),
                 "persist_RMSE": rp, "crestline_RMSE": rm, "skill": 1 - rm / rp,
                 "persist_RMSE_hi": rph, "crestline_RMSE_hi": rmh,
                 "skill_hi": 1 - rmh / rph})
    if H == 6:
        keep = {"truth": yte, "persistence": pte, "crestline": pred}
    print(f"  +{H:2d}h  skill {1-rm/rp:5.1%}   high-water skill {1-rmh/rph:5.1%}")

res = pd.DataFrame(rows)
res.to_csv("data/lead_time_metrics.csv", index=False)
print("\n=== LEAD TIME TABLE ===")
print(res.to_string(index=False))

# --- hero plot: worst event in the test window --------------------------
k = pd.DataFrame(keep)
peak = k["truth"].idxmax()
win = k.loc[peak - pd.Timedelta("36h"): peak + pd.Timedelta("36h")]
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(win.index, win["truth"], lw=2.4, label="Actual water level")
ax.plot(win.index, win["persistence"], lw=1.4, ls="--", label="Persistence (no model)")
ax.plot(win.index, win["crestline"], lw=2.0, label="CrestLine forecast (+6h)")
ax.set_title(f"Greens Bayou at Ley Rd — {peak:%B %d, %Y} (crest {k['truth'].max():.1f} ft)")
ax.set_ylabel("Stage (ft)"); ax.legend(); ax.grid(alpha=.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("data/event_plot.png", dpi=160)
print(f"\nBiggest test-set event: {peak:%Y-%m-%d %H:%M}  crest {k['truth'].max():.2f} ft")
print("Saved data/lead_time_metrics.csv and data/event_plot.png")
