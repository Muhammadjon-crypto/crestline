"""Pull NCEP Stage IV hourly rainfall (radar + gauge-corrected, gridded)
over the Greens Bayou watershed from the IEM reanalysis API.

One HTTP call per day per point, so this is slow by design. It checkpoints
after every point -- safe to Ctrl+C and rerun; completed points are skipped.
"""
import os, time, json
import requests
import pandas as pd

YEARS = 8
OUT = "data/rain_stage4.csv"

# three points down the Greens Bayou watershed, upstream -> downstream
POINTS = {
    "rain_upper": (29.96, -95.42),   # near Greens @ US-75
    "rain_mid":   (29.92, -95.31),   # near Greens nr Houston
    "rain_lower": (29.84, -95.23),   # near Ley Rd (the target)
}

S = requests.Session()
S.headers.update({"User-Agent": "tx29-crestline-student-project"})
BASE = "https://mesonet.agron.iastate.edu/iemre/hourly"


def fetch_day(lat, lon, day):
    url = f"{BASE}/{day:%Y-%m-%d}/{lat}/{lon}/json"
    for attempt in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                js = r.json()
                rows = js.get("data", [])
                if not rows:
                    return None
                d = pd.DataFrame(rows)
                tcol = next((c for c in ("valid_utc", "valid_local", "valid")
                             if c in d.columns), None)
                pcol = next((c for c in ("hourly_precip_in", "precip_in", "p01i", "precip")
                             if c in d.columns), None)
                if tcol is None or pcol is None:
                    print(f"  unexpected columns: {list(d.columns)}")
                    return None
                d[tcol] = pd.to_datetime(d[tcol], utc=True, errors="coerce")
                d[pcol] = pd.to_numeric(d[pcol], errors="coerce")
                return d.set_index(tcol)[pcol].rename("precip_in")
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


if __name__ == "__main__":
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=YEARS)
    days = pd.date_range(start, end, freq="D")
    print(f"{len(days):,} days x {len(POINTS)} points = "
          f"{len(days)*len(POINTS):,} requests\n")

    existing = {}
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, index_col=0, parse_dates=True)
        existing = {c: prev[c] for c in prev.columns}
        print(f"Resuming -- already have: {list(existing)}\n")

    series = dict(existing)
    for name, (lat, lon) in POINTS.items():
        if name in series:
            continue
        print(f"=== {name} ({lat}, {lon}) ===")
        chunks, t0 = [], time.time()
        for i, day in enumerate(days):
            s = fetch_day(lat, lon, day)
            if s is not None:
                chunks.append(s)
            if i % 200 == 0 and i:
                el = time.time() - t0
                print(f"  {i:,}/{len(days):,}  {el/60:.1f} min elapsed, "
                      f"~{el/i*(len(days)-i)/60:.0f} min left")
            time.sleep(0.35)
        if chunks:
            series[name] = pd.concat(chunks).sort_index().rename(name)
            series[name] = series[name][~series[name].index.duplicated()]
            pd.concat(series.values(), axis=1).to_csv(OUT)
            print(f"  done: {len(series[name]):,} hours -> checkpointed\n")
        else:
            print("  NOTHING RETURNED\n")

    rain = pd.concat(series.values(), axis=1)
    rain.to_csv(OUT)
    print("=== SUMMARY ===")
    for c in rain.columns:
        s = rain[c]
        print(f"  {c:12s} {len(s):,} hrs | non-null {100*s.notna().mean():5.1f}% "
              f"| total {s.sum():8.1f} in | max hr {s.max():.2f} in")
    print(f"\nSaved {OUT}")
