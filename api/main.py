"""FastAPI backend (P4), runs on Laptop 1.

    EVAL_DIR=<folder with trip folders> uvicorn api.main:app --host 0.0.0.0 --port 8000

GET  /trips            trip ids found in EVAL_DIR (re-scanned on every call, so evaluator trips
                       dropped into the folder on the day show up immediately)
POST /predict          {"trip_id": "..."} -> GPS points, predicted route and accuracy metrics
GET  /health           liveness, backend and model info

The pipeline (hmm.pipeline.run_trip) runs on the server: candidates -> classifier -> Viterbi.
Accuracy is computed server-side against the trip's own .route file; ground-truth arcs are never
sent to the UI, only the metric numbers. The model is loaded ONCE at startup. Every request is
logged to the console with trip id, timing and accuracy.

PATHSNAP_BACKEND=stub switches to placeholder responses (used before the pipeline existed, kept
for network tests); the default is "pipeline".
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, load_trip  # noqa: E402

BACKEND = os.environ.get("PATHSNAP_BACKEND", "pipeline")
EVAL_DIR = Path(os.environ.get("EVAL_DIR", str(DATA_ROOT / "student_test")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting: backend=%s EVAL_DIR=%s", BACKEND, EVAL_DIR)
    if not EVAL_DIR.is_dir():
        log.warning("EVAL_DIR does not exist yet: %s", EVAL_DIR)
    if BACKEND == "pipeline":
        from hmm.pipeline import warm_up       # imported here so stub mode needs no model files
        t0 = time.perf_counter()
        STATE["hmm_params"] = warm_up()        # loads artifacts/model.pkl once
        from models.predict import load_artifacts
        cfg = load_artifacts()["config"]
        STATE["model_info"] = f"{cfg['model']} ({cfg['variant']}), {len(cfg['feature_cols'])} features"
        log.info("model loaded once in %.2fs: %s; HMM params: %s", time.perf_counter() - t0,
                 STATE["model_info"], STATE["hmm_params"].to_dict())
    yield


app = FastAPI(title="PathSnap map-matching API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids() -> list[str]:
    if not EVAL_DIR.is_dir():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "eval_dir": str(EVAL_DIR), "model": STATE.get("model_info")}


@app.get("/trips")
def trips():
    ids = _trip_ids()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


def _predict_stub(trip_id: str) -> dict:
    trip = load_trip(EVAL_DIR / trip_id, validate=False)
    gps = trip["track"][["lat", "lon"]].to_numpy().tolist()
    return {"trip_id": trip_id, "stub": True, "gps_points": gps, "route_segments": [gps],
            "classifier_route_segments": [], "metrics": {}}


def _predict_pipeline(trip_id: str) -> dict:
    from hmm.pipeline import run_trip
    r = run_trip(EVAL_DIR / trip_id, STATE["hmm_params"])
    m = r["metrics"]
    return {
        "trip_id": trip_id, "stub": False,
        "gps_points": r["gps"],
        "route_segments": r["route_segments"],                       # HMM route, connected polylines
        "classifier_route_segments": r["clf_route_segments"],        # classifier-only route
        "metrics": {                                                  # numbers only, never the reference arcs
            "top1": m.get("top1_hmm"), "top1_classifier_only": m.get("top1_classifier"),
            "route_continuity_pct": m.get("continuity_hmm_pct"),
            "route_continuity_classifier_pct": m.get("continuity_classifier_pct"),
            "arc_precision": m.get("arc_precision_hmm"), "arc_recall": m.get("arc_recall_hmm"),
            "arc_f1": m.get("arc_f1_hmm"),
            "n_points": r["n_points"], "n_scored_points": m.get("n_label_valid"),
            "inference_time_s": r["timing"]["inference_total_s"],
        },
        "timing": r["timing"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in _trip_ids():
        log.warning("POST /predict unknown trip_id=%r", req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")
    t0 = time.perf_counter()
    result = _predict_stub(req.trip_id) if BACKEND == "stub" else _predict_pipeline(req.trip_id)
    elapsed = time.perf_counter() - t0
    result["metrics"]["server_time_s"] = elapsed
    m = result["metrics"]
    if BACKEND == "stub":
        log.info("POST /predict trip=%s backend=stub time=%.3fs", req.trip_id, elapsed)
    elif m.get("top1") is None:
        log.info("POST /predict trip=%s points=%s inference=%.2fs total=%.2fs (no .route: accuracy not computed)",
                 req.trip_id, m["n_points"], m["inference_time_s"], elapsed)
    else:
        log.info("POST /predict trip=%s points=%s inference=%.2fs total=%.2fs top1=%.4f (classifier-only %.4f) "
                 "continuity=%.1f%% arcF1=%.3f", req.trip_id, m["n_points"], m["inference_time_s"], elapsed,
                 m["top1"], m["top1_classifier_only"], m["route_continuity_pct"], m["arc_f1"])
    return result
