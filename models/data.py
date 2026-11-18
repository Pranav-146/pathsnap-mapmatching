"""Loading helpers for the candidate tables (P2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.load import DATA_ROOT, REPO_ROOT, list_trips, load_trip  # noqa: E402
from features.build_features import FEATURE_COLS  # noqa: E402,F401

RESULTS = REPO_ROOT / "results"
ARTIFACTS = REPO_ROOT / "artifacts"


def load_split(name: str):
    """name: 'train' or 'studtest'. Returns (candidates, point_labels)."""
    cands = pd.read_parquet(RESULTS / f"candidates_{name}.parquet")
    points = pd.read_parquet(RESULTS / f"point_labels_{name}.parquet")
    return cands, points


def valid_only(cands: pd.DataFrame) -> pd.DataFrame:
    """Rows of points the .route covers (label_valid == 1): the only rows used for fitting and metrics."""
    return cands[cands["label_valid"] == 1].reset_index(drop=True)


def arcs_for_trips(partition_dir_name: str) -> dict:
    """trip_id -> arcs DataFrame, used for route continuity."""
    return {p.name: load_trip(p, validate=False)["arcs"] for p in list_trips(DATA_ROOT / partition_dir_name)}
