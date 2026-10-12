"""Redraw on a true clock axis + quantify crest bias.

Everything is plotted at VALID time (the time being forecast), which is
how an operational forecast is actually read.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 6
p = pd.read_csv(f"data/predictions_{H}h.csv", index_col=0, parse_dates=True)

# shift from issue time -> valid time
valid = p.index + pd.Timedelta(hours=H)
obs       = pd.Series(p["truth"].values,       index=valid, name="observed")
forecast  = pd.Series(p["crestline"].values,   index=valid, name="crestline")
persist   = pd.Series(p["persistence"].values, index=valid, name="persistence")

print("=== BIAS (positive = over-prediction) ===")
err = forecast - obs
hi = obs > obs.quantile(0.95)
print(f"  mean error, all      : {err.mean():+.3f} ft")
print(f"  mean error, high water: {err[hi].mean():+.3f} ft")
print(f"  90th pct over-predict : {err.quantile(0.90):+.3f} ft")
print(f"  10th pct under-predict: {err.quantile(0.10):+.3f} ft")

# --- crest error across the biggest independent events ------------------
print("\n=== CREST ERROR ON TOP EVENTS ===")
thresh = obs.quantile(0.995)
above = obs > thresh
grp = (above != above.shift()).cumsum()[above]
rows = []
for _, idx in obs[above].groupby(grp):
    w = slice(idx.index.min() - pd.Timedelta("12h"), idx.index.max() + pd.Timedelta("12h"))
    o, f = obs[w], forecast[w]
    if len(o) < 8:
        continue
    rows.append({"event": o.idxmax().strftime("%Y-%m-%d"),
                 "obs_crest": o.max(), "fcst_crest": f.max(),
                 "crest_err": f.max() - o.max(),
                 "timing_err_h": (f.idxmax() - o.idxmax()).total_seconds() / 3600})
ev = pd.DataFrame(rows).sort_values("obs_crest", ascending=False).head(10)
print(ev.to_string(index=False))
ev.to_csv("data/crest_errors.csv", index=False)

# --- the honest hero plot ------------------------------------------------
peak = obs.idxmax()
w = slice(peak - pd.Timedelta("30h"), peak + pd.Timedelta("30h"))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(obs[w].index, obs[w], lw=2.6, color="#1f4e79", label="Observed stage")
ax.plot(persist[w].index, persist[w], lw=1.5, ls="--", color="#999999",
        label=f"Persistence forecast (issued {H}h earlier)")
ax.plot(forecast[w].index, forecast[w], lw=2.2, color="#e07b39",
        label=f"CrestLine forecast (issued {H}h earlier)")
ax.set_title(f"Greens Bayou at Ley Rd — {peak:%B %d, %Y}\n"
             f"All lines at valid time; forecasts issued {H}h before", fontsize=12)
ax.set_ylabel("Stage (ft)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("data/event_plot_valid_time.png", dpi=160)
print("\nSaved data/event_plot_valid_time.png and data/crest_errors.csv")
