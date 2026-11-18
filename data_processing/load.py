"""Trip loader and projection helpers (P1).

All distance maths in this project is done in metres. GPS lon/lat is projected
to a per-trip local metric projection with pyproj (see LocalProjector); never
measure distances in raw degrees.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_data_root() -> Path:
    """Folder that holds `train/` and `student_test/`.

    Override with the PATHSNAP_DATA environment variable.
    """
    env = os.environ.get("PATHSNAP_DATA")
    if env:
        return Path(env)
    for cand in (
        REPO_ROOT / "data",
        REPO_ROOT.parent / "Student folder" / "Student folder",
        REPO_ROOT / "Student folder" / "Student folder",
    ):
        if (cand / "train").is_dir():
            return cand
    return REPO_ROOT / "data"


DATA_ROOT = _find_data_root()

class LocalProjector:
    """Per-trip local metric projection (azimuthal equidistant, metres).

    The plan suggests UTM 32N, but the dataset covers the whole world
    (Germany, Russia, USA, Brazil, Japan, ...), so one fixed UTM zone would
    distort distances badly for most trips. A projection centred on each trip
    keeps distance errors negligible for the <~1 degree extent of a trip.
    Everything downstream still measures distance in metres.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs"
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    @classmethod
    def from_track(cls, track: pd.DataFrame) -> "LocalProjector":
        return cls(track["lon"].mean(), track["lat"].mean())

    def project(self, lon, lat):
        """lon/lat (degrees) -> local x/y (metres)."""
        x, y = self._fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x), np.asarray(y)

    def unproject(self, x, y):
        """local x/y (metres) -> lon/lat (degrees)."""
        lon, lat = self._inv.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.asarray(lon), np.asarray(lat)


def list_trips(partition_dir) -> list[Path]:
    """Trip folders (each holds <id>.track/.nodes/.arcs/.route) in a partition folder."""
    partition_dir = Path(partition_dir)
    return sorted(p for p in partition_dir.iterdir() if p.is_dir())


def _read(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes, engine="c")


def load_trip(path, validate: bool = True) -> dict:
    """Load one trip folder.

    Returns a dict with:
      trip_id : str
      track   : DataFrame[lon, lat, t]    GPS points (~1 Hz)
      nodes   : DataFrame[lon, lat]       row index = node_id
      arcs    : DataFrame[from_node, to_node]  row index = arc_id
      route   : np.ndarray[int]           ground-truth arc ids in travel order
                (None-safe: empty array if the file has no rows)
    """
    path = Path(path)
    tid = path.name
    track = _read(path / f"{tid}.track", ["lon", "lat", "t"],
                  {"lon": "float64", "lat": "float64", "t": "float64"})
    nodes = _read(path / f"{tid}.nodes", ["lon", "lat"],
                  {"lon": "float64", "lat": "float64"})
    arcs = _read(path / f"{tid}.arcs", ["from_node", "to_node"],
                 {"from_node": "int64", "to_node": "int64"})
    route_path = path / f"{tid}.route"
    if route_path.exists() and route_path.stat().st_size > 0:
        route = pd.read_csv(route_path, header=None, dtype="int64").iloc[:, 0].to_numpy()
    else:
        route = np.zeros(0, dtype=np.int64)

    trip = {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
    if validate:
        validate_trip(trip)
    return trip


def validate_trip(trip: dict) -> None:
    """Check column counts, dtypes and timestamp order. Raises ValueError."""
    tid = trip["trip_id"]
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{tid}: unexpected column counts")
    if len(track) == 0:
        raise ValueError(f"{tid}: empty track")
    if not np.issubdtype(track["t"].dtype, np.floating):
        raise ValueError(f"{tid}: t must be numeric")
    if (np.diff(track["t"].to_numpy()) <= 0).any():
        raise ValueError(f"{tid}: timestamps not strictly increasing")
    n_nodes = len(nodes)
    if len(arcs) and (arcs.to_numpy().min() < 0 or arcs.to_numpy().max() >= n_nodes):
        raise ValueError(f"{tid}: arc endpoint outside node range")
    if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"{tid}: route arc id outside arc range")
