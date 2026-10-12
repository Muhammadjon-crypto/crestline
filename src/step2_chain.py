import io, time, requests
import numpy as np, pandas as pd

BASE = "https://waterservices.usgs.gov/nwis"
S = requests.Session()
S.headers.update({"User-Agent": "tx29-flood-student-project"})

YEARS  = 8
TARGET = "08076700"   # Greens Bayou at Ley Rd  -- downstream, in TX-29
UPSTREAM = {
    "08075900": "Greens @ US-75",
    "08076000": "Greens nr Houston",
    "08076500": "Halls Bayou",
    "08076180": "Garners Bayou",
}


def fetch_iv(site, start, end, parameter_cd="00065"):
    frames = []
    edges = sorted(set(list(pd.date_range(start, end, freq="YS")) + [pd.Timestamp(start)]))
    for a in edges:
        b = min(a + pd.DateOffset(years=1), pd.Timestamp(end))
        if b <= a:
            continue
        try:
            r = S.get(f"{BASE}/iv/", params={
                "format": "json", "sites": site, "parameterCd": parameter_cd,
                "startDT": a.strftime("%Y-%m-%d"), "endDT": b.strftime("%Y-%m-%d"),
            }, timeout=180)
            if r.status_code != 200:
                continue
            ts = r.json()["value"]["timeSeries"]
            if not ts or not ts[0]["values"][0]["value"]:
                continue
            d = pd.DataFrame(ts[0]["values"][0]["value"])
            d["dateTime"] = pd.to_datetime(d["dateTime"], utc=True, format="ISO8601")
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d.loc[d["value"] <= -999998, "value"] = np.nan
            frames.append(d[["dateTime", "value"]])
        except Exception as e:
            print(f"    {site} {a:%Y}: {type(e).__name__}")
        time.sleep(0.4)
    if not frames:
        return pd.Series(dtype=float, name=site)
    s = pd.concat(frames).drop_duplicates("dateTime").set_index("dateTime")["value"]
    return s.sort_index().resample("15min").mean().rename(site)


def persistence_baseline(series, horizons_h=(1, 3, 6, 12)):
    rows, roc = [], series.diff(4)
    for h in horizons_h:
        err = (series.shift(-h * 4) - series).dropna()
        rising = roc.reindex(err.index) > 0.05
        high = series.reindex(err.index) > series.quantile(0.95)
        rows.append({
            "horizon_h": h,
            "RMSE_all": np.sqrt((err ** 2).mean()),
            "RMSE_rising": np.sqrt((err[rising] ** 2).mean()) if rising.sum() else np.nan,
            "n_rising": int(rising.sum()),
            "RMSE_top5pct": np.sqrt((err[high] ** 2).mean()) if high.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def travel_time(upstream, target, max_lag_h=12):
    """Correlate upstream level against target level at increasing lags.
    The lag with peak correlation is the physical travel time."""
    rows = []
    for lag_steps in range(0, max_lag_h * 4 + 1, 2):   # every 30 min
        c = upstream.shift(lag_steps).corr(target)
        rows.append({"lag_hours": lag_steps / 4, "corr": c})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=YEARS)

    print(f"Target: {TARGET} (Greens Bayou at Ley Rd)")
    print(f"Window: {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")

    series = {}
    print("Fetching target...")
    series[TARGET] = fetch_iv(TARGET, start, end)
    for sid, name in UPSTREAM.items():
        print(f"Fetching {name} ({sid})...")
        series[sid] = fetch_iv(sid, start, end)

    df = pd.concat(series.values(), axis=1)
    df.to_csv("data/greens_chain_15min.csv")

    print("\n=== COVERAGE ===")
    for col in df.columns:
        label = UPSTREAM.get(col, "TARGET Ley Rd")
        print(f"  {col} {label:22s} {100*df[col].notna().mean():5.1f}% "
              f"range {df[col].min():7.2f} to {df[col].max():7.2f} ft")

    tgt = df[TARGET]
    print("\n=== PERSISTENCE BASELINE AT LEY RD (beat this) ===")
    print(persistence_baseline(tgt).to_string(index=False))

    print("\n=== TRAVEL TIME: upstream -> Ley Rd ===")
    for sid, name in UPSTREAM.items():
        tt = travel_time(df[sid], tgt)
        best = tt.loc[tt["corr"].idxmax()]
        zero = tt.loc[0, "corr"]
        print(f"  {name:22s} peak r={best['corr']:.3f} at "
              f"{best['lag_hours']:4.1f}h   (r={zero:.3f} at 0h)")

    print("\nA peak at a lag > 0 means upstream water genuinely leads the target.")
    print("That lag is your forecast horizon, and persistence cannot see it.")
