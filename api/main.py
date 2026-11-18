"""FastAPI backend (P4).

    EVAL_DIR=<folder with trip folders> uvicorn api.main:app --host 0.0.0.0 --port 8000

GET  /trips            trip ids found in EVAL_DIR (re-scanned on every call, so evaluator trips
                       dropped into the folder on the day show up immediately)
POST /predict          {"trip_id": "..."} -> GPS points, predicted route and accuracy metrics
GET  /health           liveness + which backend is active

Ground-truth arcs are never returned: only metric numbers computed on the server.

BACKEND
  "stub"      (11:30 checkpoint) returns the real GPS points of the trip but a placeholder route and
              null metrics, flagged with "stub": true. It exists so the UI and the two-laptop
              network path can be tested before the pipeline is wired in.
  "pipeline"  (wired in after the checkpoint) calls hmm.pipeline.run_trip().
Select with the PATHSNAP_BACKEND environment variable.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, load_trip  # noqa: E402

BACKEND = os.environ.get("PATHSNAP_BACKEND", "stub")
EVAL_DIR = Path(os.environ.get("EVAL_DIR", str(DATA_ROOT / "student_test")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.on_event("startup")
def _startup():
    # The trained model is loaded ONCE here when the real pipeline is enabled.
    log.info("backend=%s EVAL_DIR=%s", BACKEND, EVAL_DIR)
    if not EVAL_DIR.is_dir():
        log.warning("EVAL_DIR does not exist yet: %s", EVAL_DIR)


def _trip_ids() -> list[str]:
    if not EVAL_DIR.is_dir():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "eval_dir": str(EVAL_DIR)}


@app.get("/trips")
def trips():
    ids = _trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


def _predict_stub(trip_id: str) -> dict:
    trip = load_trip(EVAL_DIR / trip_id, validate=False)
    gps = trip["track"][["lat", "lon"]].to_numpy().tolist()
    return {
        "trip_id": trip_id,
        "stub": True,
        "gps_points": gps,
        "route_coords": gps,   # placeholder route: the raw GPS trace
        "metrics": {"top1": None, "route_continuity_pct": None, "inference_time_s": None},
    }


def _predict_pipeline(trip_id: str) -> dict:
    raise NotImplementedError("pipeline backend is wired in after the 11:30 checkpoint")


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in _trip_ids():
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")
    t0 = time.perf_counter()
    result = _predict_stub(req.trip_id) if BACKEND == "stub" else _predict_pipeline(req.trip_id)
    elapsed = time.perf_counter() - t0
    result["metrics"]["server_time_s"] = elapsed
    m = result["metrics"]
    log.info("POST /predict trip=%s backend=%s time=%.3fs top1=%s continuity=%s",
             req.trip_id, BACKEND, elapsed, m.get("top1"), m.get("route_continuity_pct"))
    return result
