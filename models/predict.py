"""Inference helper (P2 hand-off to P3 and P4).

    from models.predict import predict_scores
    scored = predict_scores(cand_df)      # cand_df + a `score` column in [0, 1]

The exported model (artifacts/model.pkl) and its feature list (artifacts/feature_config.json) are
loaded once and cached, so the API pays the loading cost at startup only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import REPO_ROOT  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
_STATE: dict = {}


def load_artifacts(artifacts_dir: Path | None = None) -> dict:
    """Load model, optional scaler and feature config once; later calls return the cache."""
    if _STATE and artifacts_dir is None:
        return _STATE
    d = Path(artifacts_dir) if artifacts_dir else ARTIFACTS
    cfg = json.loads((d / "feature_config.json").read_text())
    scaler_path = d / "scaler.pkl"
    _STATE.clear()
    _STATE.update({
        "model": joblib.load(d / "model.pkl"),
        "scaler": joblib.load(scaler_path) if scaler_path.exists() else None,
        "config": cfg,
    })
    return _STATE


def predict_scores(cand_df: pd.DataFrame) -> pd.DataFrame:
    """Return cand_df with a `score` column: P(candidate arc is the reference arc)."""
    st = load_artifacts()
    cols = st["config"]["feature_cols"]
    X = cand_df[cols]
    if st["scaler"] is not None:
        X = st["scaler"].transform(X)
    out = cand_df.copy()
    out["score"] = st["model"].predict_proba(X)[:, 1]
    return out
