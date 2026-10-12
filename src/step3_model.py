"""First real model. Predict Greens Bayou @ Ley Rd H hours ahead from
upstream gauges only (no rainfall yet). Evaluated against persistence on
the same test set and the same slices.

LEAKAGE DISCIPLINE: features use only data at or before time t. The split
is chronological -- train on early years, test on the most recent. No
shuffling, ever, on a time series.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "08076700"
UPSTREAM = {"08075900": "greens_us75", "08076000": "greens_hou",
            "08076500": "halls", "08076180": "garners"}
HORIZON_H = 6
TEST_FRACTION = 0.25

df = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
df.columns = [str(c) for c in df.columns]

print("=== TRAVEL TIME ON CHANGES (not levels) ===")
tgt_d = df[TARGET].diff(4)
for sid, name in UPSTREAM.items():
    up_d = df[sid].diff(4)
    cors = {lag / 4: up_d.shift(lag).corr(tgt_d) for lag in range(0, 49, 2)}
    best = max(cors, key=lambda k: cors[k])
    print(f"  {name:12s} peak r={cors[best]:.3f} at {best:4.1f}h  (r={cors[0.0]:.3f} at 0h)")

X = pd.DataFrame(index=df.index)
for sid, name in UPSTREAM.items():
    s = df[sid]
    for lag_h in (0, 1, 2, 3, 4, 5, 6):
        X[f"{name}_lag{lag_h}h"] = s.shift(lag_h * 4)
    X[f"{name}_roc1h"] = s.diff(4)
    X[f"{name}_roc3h"] = s.diff(12)
    X[f"{name}_roc6h"] = s.diff(24)

t = df[TARGET]
for lag_h in (0, 1, 2, 3, 6):
    X[f"target_lag{lag_h}h"] = t.shift(lag_h * 4)
X["target_roc1h"] = t.diff(4)
X["target_roc3h"] = t.diff(12)
X["month"] = X.index.month
X["hour"] = X.index.hour

y = t.shift(-HORIZON_H * 4)
persist = t

ok = X.notna().all(axis=1) & y.notna() & persist.notna()
X, y, persist = X[ok], y[ok], persist[ok]
print(f"\nUsable rows: {len(X):,}  features: {X.shape[1]}")

split = int(len(X) * (1 - TEST_FRACTION))
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y.iloc[:split], y.iloc[split:]
pte = persist.iloc[split:]
print(f"Train: {Xtr.index.min():%Y-%m-%d} to {Xtr.index.max():%Y-%m-%d}  ({len(Xtr):,})")
print(f"Test : {Xte.index.min():%Y-%m-%d} to {Xte.index.max():%Y-%m-%d}  ({len(Xte):,})")

model = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_depth=None,
    early_stopping=True, validation_fraction=0.15, random_state=0)
model.fit(Xtr, ytr)
pred = pd.Series(model.predict(Xte), index=Xte.index)

roc = df[TARGET].diff(4).reindex(yte.index)
rising = roc > 0.05
high = yte > df[TARGET].quantile(0.95)

def rmse(a, b, m=None):
    e = (a - b) if m is None else (a[m] - b[m])
    return np.sqrt((e ** 2).mean()) if len(e) else np.nan

print(f"\n=== RESULTS @ +{HORIZON_H}h (test set only) ===")
print(f"{'slice':<14}{'n':>8}{'persistence':>14}{'CrestLine':>12}{'skill':>9}")
for label, m in [("all", None), ("rising", rising), ("top 5%", high)]:
    n = len(yte) if m is None else int(m.sum())
    rp, rm = rmse(yte, pte, m), rmse(yte, pred, m)
    skill = 1 - rm / rp if rp else np.nan
    print(f"{label:<14}{n:>8,}{rp:>14.3f}{rm:>12.3f}{skill:>8.1%}")

from sklearn.inspection import permutation_importance
sub = np.random.RandomState(0).choice(len(Xte), min(3000, len(Xte)), replace=False)
r = permutation_importance(model, Xte.iloc[sub], yte.iloc[sub],
                           n_repeats=3, random_state=0)
imp = pd.Series(r.importances_mean, index=X.columns)
print("\n=== TOP FEATURES ===")
print(imp.sort_values(ascending=False).head(12).to_string())

pd.DataFrame({"truth": yte, "persistence": pte, "crestline": pred}).to_csv(
    f"data/predictions_{HORIZON_H}h.csv")
print(f"\nSaved data/predictions_{HORIZON_H}h.csv")
