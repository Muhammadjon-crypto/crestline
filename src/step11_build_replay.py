"""Precompute the 15-16 June 2026 event replay.

This is a BACKTEST, not a mock: the frozen production model runs over the
real historical gauge and rainfall record, and every forecast in the output
is a genuine model prediction made from inputs available at that moment.
Nothing is canned.

Writes data/replay_june2026.json for /api/replay to serve.
"""
import sys, json
import numpy as np, pandas as pd, joblib

sys.path.insert(0, "src")
from step9_train_final import build_features, TARGET, HORIZON_H
from predict import condition, context_line, history, LEAD

START = "2026-06-15 00:00"
END   = "2026-06-17 12:00"
OUT   = "data/replay_june2026.json"

stage = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
stage.index = pd.to_datetime(stage.index, utc=True)
stage.columns = [str(c) for c in stage.columns]

rain = pd.read_csv("data/rain_stage4.csv", index_col=0, parse_dates=True)
rain.index = pd.to_datetime(rain.index, utc=True)
rain = rain.loc[:, ~rain.T.duplicated()].sort_index()

model = joblib.load("data/crestline_model_3h.joblib")
meta  = json.load(open("data/model_metrics.json"))
hist  = history()

print("Building features over the full record…")
X = build_features(stage, rain).reindex(columns=meta["feature_names"])

win   = slice(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"))
rows  = X.loc[win].dropna()
if rows.empty:
    sys.exit("No complete feature rows in that window — check the dates")

print(f"Running the frozen model on {len(rows):,} issue times…")
preds = model.predict(rows)

t      = stage[TARGET]
roc_1h = t.diff(4)
frames = []
for ts, pred in zip(rows.index, preds):
    now = float(t.loc[ts])
    r   = float(roc_1h.loc[ts]) if pd.notna(roc_1h.loc[ts]) else 0.0
    state, label = condition(now, float(pred), r, hist["high_water_ft"])
    frames.append({
        "t": ts.isoformat(),
        "now_ft": round(now, 2),
        "fcst_ft": round(float(pred), 2),
        "change_ft": round(float(pred) - now, 2),
        "roc_ft_per_h": round(r, 2),
        "state": state, "state_label": label,
        "context": context_line(state, float(pred), hist),
        "band_ft": round(meta["rmse_high_ft" if state in ("high", "elevated") else "rmse_ft"], 2),
        "band_basis": "high" if state in ("high", "elevated") else "normal",
        "upstream": [
            {"id": sid, "name": nm, "lead_hours": lh, "r": rr,
             "level_ft": round(float(stage[sid].loc[ts]), 2),
             "roc_ft_per_h": round(float(stage[sid].diff(4).loc[ts]), 2)}
            for sid, (nm, lh, rr) in LEAD.items()
            if sid in stage.columns and pd.notna(stage[sid].loc[ts])
        ],
    })

# full series for the chart, with a 24h lead-in before the first frame
lead  = rows.index[0] - pd.Timedelta(hours=24)
s_win = t.loc[lead:rows.index[-1] + pd.Timedelta(hours=HORIZON_H)].dropna()
r_win = rain.mean(axis=1).loc[lead:rows.index[-1]].dropna()

obs_peak = s_win.max()
out = {
    "event": {
        "name": "Greens Bayou flood — 15–16 June 2026",
        "short": "15 Jun 2026 event",
        "start": rows.index[0].isoformat(),
        "end":   rows.index[-1].isoformat(),
        "observed_crest_ft": round(float(obs_peak), 2),
        "observed_crest_at": s_win.idxmax().isoformat(),
        "rain_total_in": round(float(r_win.sum()), 2),
    },
    "horizon_hours": HORIZON_H,
    "high_water_ft": hist["high_water_ft"],
    "record_ft": hist["record_ft"],
    "stage": [{"t": i.isoformat(), "ft": round(float(v), 2)} for i, v in s_win.items()],
    "rain":  [{"t": i.isoformat(), "in": round(float(v), 3)} for i, v in r_win.items()],
    "frames": frames,
    "metrics": {
        "skill": round(meta["skill"], 3),
        "skill_high": round(meta["skill_high"], 3),
        "rmse_ft": round(meta["rmse_ft"], 2),
        "persistence_rmse_ft": round(meta["persistence_rmse_ft"], 2),
        "rmse_high_ft": round(meta["rmse_high_ft"], 2),
        "persistence_rmse_high_ft": round(meta["persistence_rmse_high_ft"], 2),
        "n_test": meta["n_test"],
        "test_start": meta["test_start"][:10],
        "test_end": meta["test_end"][:10],
    },
}
json.dump(out, open(OUT, "w"))

# how well did it actually do on this event?
f = pd.DataFrame(frames).set_index(pd.to_datetime([x["t"] for x in frames]))
truth = t.reindex(f.index + pd.Timedelta(hours=HORIZON_H)).values
err   = f["fcst_ft"].values - truth
ok    = ~np.isnan(err)
print(f"\nFrames        : {len(frames):,}  ({rows.index[0]:%d %b %H:%M} → {rows.index[-1]:%d %b %H:%M} UTC)")
print(f"Observed crest: {obs_peak:.2f} ft at {s_win.idxmax():%d %b %H:%M} UTC")
print(f"Forecast peak : {f['fcst_ft'].max():.2f} ft")
print(f"Rain total    : {r_win.sum():.2f} in")
print(f"RMSE on event : {np.sqrt((err[ok]**2).mean()):.2f} ft over {ok.sum()} forecasts")
print(f"\nSaved {OUT}")
