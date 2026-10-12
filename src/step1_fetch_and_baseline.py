import io, time, requests
import numpy as np, pandas as pd

BBOX = (-95.50, 29.62, -95.02, 30.08)
TARGET = "08075900"
YEARS = 8
STAGE, RAIN = "00065", "00045"

BASE = "https://waterservices.usgs.gov/nwis"
S = requests.Session()
S.headers.update({"User-Agent": "tx29-flood-student-project"})


def discover(bbox, parameter_cd):
    r = S.get(f"{BASE}/site/", params={
        "format": "rdb",
        "bBox": ",".join(str(x) for x in bbox),
        "siteType": "ST", "hasDataTypeCd": "iv", "siteStatus": "all",
        "seriesCatalogOutput": "true", "parameterCd": parameter_cd,
    }, timeout=120)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:]
    df = df[(df["data_type_cd"] == "uv") & (df["parm_cd"] == parameter_cd)]
    df = df[["site_no", "station_nm", "dec_lat_va", "dec_long_va",
             "begin_date", "end_date", "count_nu"]].copy()
    for c in ["dec_lat_va", "dec_long_va"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("site_no").reset_index(drop=True)


def fetch_iv(site, parameter_cd, start, end):
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
                print(f"    {site} {a:%Y}: HTTP {r.status_code}")
                continue
            ts = r.json()["value"]["timeSeries"]
            if not ts or not ts[0]["values"][0]["value"]:
                continue
            d = pd.DataFrame(ts[0]["values"][0]["value"])
            d["dateTime"] = pd.to_datetime(d["dateTime"], utc=True, format="ISO8601")
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d.loc[d["value"] <= -999998, "value"] = np.nan
            frames.append(d[["dateTime", "value"]])
            print(f"    {site} {a:%Y}: {len(d):,} rows")
        except Exception as e:
            print(f"    {site} {a:%Y}: {type(e).__name__} {e}")
        time.sleep(0.4)
    if not frames:
        return pd.Series(dtype=float, name=site)
    out = pd.concat(frames).drop_duplicates("dateTime").set_index("dateTime")["value"]
    return out.sort_index().rename(site)


def to_15min(s):
    return s.resample("15min").mean()


def persistence_baseline(series, horizons_h=(1, 3, 6, 12)):
    rows, roc = [], series.diff(4)
    for h in horizons_h:
        err = (series.shift(-h * 4) - series).dropna()
        rising = roc.reindex(err.index) > 0.05
        high = series.reindex(err.index) > series.quantile(0.95)
        rows.append({
            "horizon_h": h, "n": len(err),
            "RMSE_all": np.sqrt((err ** 2).mean()),
            "MAE_all": err.abs().mean(),
            "RMSE_rising": np.sqrt((err[rising] ** 2).mean()) if rising.sum() else np.nan,
            "n_rising": int(rising.sum()),
            "RMSE_top5pct": np.sqrt((err[high] ** 2).mean()) if high.sum() else np.nan,
            "n_top5pct": int(high.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=YEARS)
    print(f"Window: {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")

    print("=== STAGE gauges in the TX-29 box ===")
    stage_sites = discover(BBOX, STAGE)
    print(stage_sites.to_string(), "\n")

    print("=== RAINFALL gauges in the same box ===")
    rain_sites = discover(BBOX, RAIN)
    print("NONE -- rainfall must come from HCFCD.\n" if rain_sites.empty
          else rain_sites.to_string() + "\n")

    print(f"=== Fetching target {TARGET} ({YEARS}y stage) ===")
    target = to_15min(fetch_iv(TARGET, STAGE, start, end))
    print(f"\n  rows on 15-min grid : {len(target):,}")
    print(f"  non-missing         : {target.notna().sum():,} ({100*target.notna().mean():.1f}%)")
    print(f"  range (ft)          : {target.min():.2f} to {target.max():.2f}")
    print(f"  span                : {target.index.min()} to {target.index.max()}\n")

    print("=== PERSISTENCE BASELINE (the number to beat) ===")
    print(persistence_baseline(target).to_string(index=False))

    target.to_csv("data/target_stage_15min.csv")
    stage_sites.to_csv("data/stage_sites.csv", index=False)
    print("\nSaved to data/")
