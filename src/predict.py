"""Live forecast for the CrestLine page.

Returns everything web/index.html needs in one payload: current stage, the
+3h forecast with its measured error band, 24h of observed stage and
rainfall, the four upstream inputs with their travel times, a condition
call, and the model's performance.

Imports build_features from step9 so serving and training cannot drift.
"""
import sys, json, datetime as dt
import numpy as np, pandas as pd, requests, joblib

sys.path.insert(0, "src")
from step9_train_final import build_features, TARGET, UPSTREAM, HORIZON_H

USGS = "https://waterservices.usgs.gov/nwis/iv/"
IEM  = "https://mesonet.agron.iastate.edu/iemre/hourly"
RAIN_POINTS = {"rain_upper": (29.96, -95.42), "rain_mid": (29.92, -95.31)}
LOOKBACK_DAYS = 10          # the 168h rainfall window needs a full week

# measured in step 2 — how far each gauge leads Ley Rd, and its correlation
LEAD = {
    "08075900": ("Greens Bayou @ US-75",    5.0, 0.79),
    "08076000": ("Greens Bayou nr Houston", 3.0, 0.87),
    "08076500": ("Halls Bayou",             2.5, 0.88),
    "08076180": ("Garners Bayou",           2.5, 0.89),
}

S = requests.Session()
S.headers.update({"User-Agent": "crestline-app"})


# ---------------------------------------------------------------- fetching
def fetch_stage_live(days=LOOKBACK_DAYS):
    sites = ",".join([TARGET] + list(UPSTREAM))
    r = S.get(USGS, params={"format": "json", "sites": sites,
                            "parameterCd": "00065", "period": f"P{days}D"},
              timeout=60)
    r.raise_for_status()
    out = {}
    for ts in r.json()["value"]["timeSeries"]:
        sid = ts["sourceInfo"]["siteCode"][0]["value"]
        vals = ts["values"][0]["value"]
        if not vals:
            continue
        d = pd.DataFrame(vals)
        d["dateTime"] = pd.to_datetime(d["dateTime"], utc=True, format="ISO8601")
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d.loc[d["value"] <= -999998, "value"] = np.nan
        out[sid] = d.set_index("dateTime")["value"].sort_index().resample("15min").mean()
    return pd.concat(out.values(), axis=1, keys=out.keys())


def fetch_rain_live(days=LOOKBACK_DAYS):
    today = dt.datetime.now(dt.UTC).date()
    series = {}
    for name, (lat, lon) in RAIN_POINTS.items():
        chunks = []
        for i in range(days + 1):
            day = today - dt.timedelta(days=days - i)
            try:
                rows = S.get(f"{IEM}/{day:%Y-%m-%d}/{lat}/{lon}/json",
                             timeout=45).json().get("data", [])
                if rows:
                    d = pd.DataFrame(rows)
                    d["valid_utc"] = pd.to_datetime(d["valid_utc"], utc=True)
                    chunks.append(d.set_index("valid_utc")["hourly_precip_in"])
            except Exception as e:
                print(f"  rain {name} {day}: {type(e).__name__}", file=sys.stderr)
        if chunks:
            s = pd.concat(chunks).sort_index()
            series[name] = s[~s.index.duplicated()].rename(name)
    return pd.concat(series.values(), axis=1)


# ------------------------------------------------------------- history ctx
_HIST = None

def history():
    """Percentiles and past crests from the 8-year record, computed once."""
    global _HIST
    if _HIST is not None:
        return _HIST
    df = pd.read_csv("data/greens_chain_15min.csv", index_col=0, parse_dates=True)
    df.columns = [str(c) for c in df.columns]
    s = df[TARGET].dropna()

    thresh = s.quantile(0.995)
    above = s > thresh
    grp = (above != above.shift()).cumsum()[above]
    crests = []
    for _, idx in s[above].groupby(grp):
        if len(idx) >= 4:
            crests.append({"when": idx.idxmax().strftime("%-d %b %Y"),
                           "ft": round(float(idx.max()), 2)})
    crests.sort(key=lambda c: -c["ft"])

    _HIST = {"high_water_ft": round(float(s.quantile(0.95)), 2),
             "record_ft": round(float(s.max()), 2),
             "median_ft": round(float(s.median()), 2),
             "crests": crests[:12]}
    return _HIST


def condition(now_ft, fcst_ft, roc_1h, hi):
    """Three states. The FORECAST decides, not the last hour of noise --
    a single noisy hour must never raise an alarm the model disagrees with."""
    peak = max(now_ft, fcst_ft)
    change = fcst_ft - now_ft
    if peak >= hi:
        return "high", ("Rising fast" if change > 0.5 else "High water")
    if peak >= 0.7 * hi or change > 0.75:
        return "elevated", "Elevated"
    return "normal", "Normal"


def context_line(state, fcst_ft, hist):
    """One plain sentence putting the number in context. No street claims."""
    hi = hist["high_water_ft"]
    if state == "normal":
        return (f"A normal day \u2014 the bayou sits below {hi} ft "
                f"about 95% of the time.")
    if fcst_ft < hi:
        return f"Still {round(hi - fcst_ft, 1)} ft below the {hi} ft high-water mark."
    below = [c for c in hist["crests"] if c["ft"] < fcst_ft]
    if below:
        c = below[0]
        return f"Higher than the {c['ft']} ft crest of {c['when']}."
    return (f"{round(fcst_ft - hi, 1)} ft above high water. "
            f"Record here is {hist['record_ft']} ft.")


# ------------------------------------------------------------------- main
def forecast():
    model = joblib.load("data/crestline_model_3h.joblib")
    meta  = json.load(open("data/model_metrics.json"))
    hist  = history()

    stage = fetch_stage_live()
    rain  = fetch_rain_live()

    X = build_features(stage, rain).reindex(columns=meta["feature_names"])
    valid = X.dropna()
    if valid.empty:
        raise RuntimeError("No complete feature row \u2014 a gauge may be offline")

    row    = valid.iloc[[-1]]
    issued = row.index[-1]
    pred   = float(model.predict(row)[0])

    t = stage[TARGET].dropna()
    now_ft = float(t.iloc[-1])
    roc_1h = float(t.iloc[-1] - t.iloc[-5]) if len(t) > 5 else 0.0
    state, label = condition(now_ft, pred, roc_1h, hist["high_water_ft"])

    # upstream panel
    upstream = []
    for sid, (name, lead_h, r) in LEAD.items():
        if sid not in stage.columns:
            continue
        s = stage[sid].dropna()
        if s.empty:
            continue
        upstream.append({
            "id": sid, "name": name,
            "level_ft": round(float(s.iloc[-1]), 2),
            "roc_ft_per_h": round(float(s.iloc[-1] - s.iloc[-5]), 2) if len(s) > 5 else 0.0,
            "lead_hours": lead_h, "r": r,
        })

    # 24h of observations
    recent = t.iloc[-96:]
    rain_24 = rain.mean(axis=1).iloc[-24:] if not rain.empty else pd.Series(dtype=float)

    return {
        "issued_utc": issued.isoformat(),
        "valid_utc": (issued + pd.Timedelta(hours=HORIZON_H)).isoformat(),
        "horizon_hours": HORIZON_H,

        "current_stage_ft": round(now_ft, 2),
        "forecast_stage_ft": round(pred, 2),
        "change_ft": round(pred - now_ft, 2),
        "roc_ft_per_h": round(roc_1h, 2),

        "state": state,
        "state_label": label,
        "context": context_line(state, pred, hist),

        "high_water_ft": hist["high_water_ft"],
        "record_ft": hist["record_ft"],

        "band_ft": round(meta["rmse_high_ft" if state == "high"
                              else "rmse_ft"], 2),

        "recent_stage": [{"t": ix.isoformat(), "ft": round(float(v), 2)}
                         for ix, v in recent.items() if pd.notna(v)],
        "recent_rain": [{"t": ix.isoformat(), "in": round(float(v), 3)}
                        for ix, v in rain_24.items() if pd.notna(v)],
        "rain_24h_total": round(float(rain_24.sum()), 2) if len(rain_24) else 0.0,

        "upstream": upstream,
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


if __name__ == "__main__":
    out = forecast()
    brief = {k: v for k, v in out.items()
             if k not in ("recent_stage", "recent_rain", "upstream", "metrics")}
    print(json.dumps(brief, indent=2))
    print(f"\nupstream      : {len(out['upstream'])} gauges")
    print(f"recent_stage  : {len(out['recent_stage'])} points")
    print(f"recent_rain   : {len(out['recent_rain'])} hours, "
          f"{out['rain_24h_total']} in total")
