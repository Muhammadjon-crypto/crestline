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
        age = ((dt.datetime.now(dt.UTC) - CACHE["updated"]).total_seconds() / 60
               if CACHE["updated"] else None)
        return {"status": "ok", "stale": bool(age and age > 45),
                "age_minutes": round(age, 1) if age else None,
                "last_error": CACHE["error"], **CACHE["data"]}


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
