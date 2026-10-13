"""CrestLine API. Refreshes the forecast on a timer, serves it instantly.

  GET /api/forecast  -> current forecast + recent observed curve
  GET /api/metrics   -> model performance for the metrics page
  GET /api/health    -> freshness check
"""
import sys, json, logging, threading, datetime as dt
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, "src")
from predict import forecast

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("crestline")

CACHE = {"data": None, "updated": None, "error": None}
LOCK = threading.Lock()
REFRESH_MIN = 15          # USGS publishes every 15 min; no point going faster

app = FastAPI(title="CrestLine API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def refresh():
    try:
        data = forecast()
        with LOCK:
            CACHE.update(data=data, updated=dt.datetime.now(dt.UTC), error=None)
        log.info("refreshed: %.2f ft -> %.2f ft",
                 data["current_stage_ft"], data["forecast_stage_ft"])
        Path("data").mkdir(exist_ok=True)
        json.dump(data, open("data/forecast.json", "w"), indent=2)
    except Exception as e:
        log.exception("refresh failed")
        with LOCK:
            CACHE["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def startup():
    # serve the last good forecast immediately, even before the first refresh
    p = Path("data/forecast.json")
    if p.exists():
        CACHE["data"] = json.load(open(p))
    threading.Thread(target=refresh, daemon=True).start()
    sch = BackgroundScheduler()
    sch.add_job(refresh, "interval", minutes=REFRESH_MIN)
    sch.start()


@app.get("/api/forecast")
def get_forecast():
    with LOCK:
        if CACHE["data"] is None:
            return {"status": "warming_up", "error": CACHE["error"]}
        now = dt.datetime.now(dt.UTC)
        data = CACHE["data"]

        # Two different clocks, two different failures.
        # fetch_age: minutes since the scheduler last succeeded. Catches a dead
        # refresh loop -- the server is up but no longer talking to USGS/IEM.
        fetch_age = ((now - CACHE["updated"]).total_seconds() / 60
                     if CACHE["updated"] else None)

        # age: minutes since the forecast was ISSUED. Stage IV rainfall publishes
        # hourly and lags, so the newest row with complete features sits behind
        # the wall clock -- a forecast can be an hour old the moment it is made.
        # This is the number that decides whether it is still worth reading.
        issued = data.get("issued_utc")
        valid = data.get("valid_utc")
        age = ((now - dt.datetime.fromisoformat(issued)).total_seconds() / 60
               if issued else None)
        expired = bool(valid and now > dt.datetime.fromisoformat(valid))

        return {"status": "ok",
                "stale": bool(expired or (fetch_age is not None and fetch_age > 45)),
                "expired": expired,
                "age_minutes": round(age, 1) if age is not None else None,
                "fetch_age_minutes": round(fetch_age, 1) if fetch_age is not None else None,
                "last_error": CACHE["error"], **data}


@app.get("/api/metrics")
def get_metrics():
    m = json.load(open("data/model_metrics.json"))
    m.pop("feature_names", None)
    return m


@app.get("/api/replay")
def get_replay():
    """A backtest of the frozen model over a real past flood. Precomputed by
    step11_build_replay.py -- every forecast here is a genuine prediction."""
    f = Path("data/replay_june2026.json")
    if not f.exists():
        return {"status": "unavailable"}
    return json.load(open(f))


@app.get("/api/health")
def health():
    with LOCK:
        return {"ok": CACHE["data"] is not None,
                "updated_utc": CACHE["updated"].isoformat() if CACHE["updated"] else None,
                "error": CACHE["error"]}


if Path("web").exists():
    app.mount("/", StaticFiles(directory="web", html=True), name="web")
