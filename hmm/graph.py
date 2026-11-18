"""Road graph and cached shortest paths for one trip (P3).

A networkx DiGraph per trip: nodes are node ids, every edge is an arc with its length in
metres (the arc id is stored on the edge). Shortest-path lookups between arc endpoints
are cached per source node, bounded by SP_CUTOFF metres.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.candidates import TripGeometry  # noqa: E402

SP_CUTOFF = 1500.0   # metres: network distances beyond this are treated as "not connected"
CACHE_LIMIT = 30000  # max cached source nodes before the cache is flushed


class RoadGraph:
    def __init__(self, geom: TripGeometry):
        self.geom = geom
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(geom.node_xy)))
        fn, tn, ln = geom.from_node, geom.to_node, geom.arc_len
        for a in np.argsort(-ln, kind="stable"):        # longest first, so the shortest duplicate wins
            self.G.add_edge(int(fn[a]), int(tn[a]), length=float(ln[a]), arc_id=int(a))
        self._cache: dict[int, dict[int, float]] = {}

    def lengths_from(self, src: int) -> dict[int, float]:
        """Network distance (m) from node `src` to every node within SP_CUTOFF (cached)."""
        d = self._cache.get(src)
        if d is None:
            if len(self._cache) > CACHE_LIMIT:
                self._cache.clear()
            d = nx.single_source_dijkstra_path_length(self.G, src, cutoff=SP_CUTOFF, weight="length")
            self._cache[src] = d
        return d

    def route_dist_matrix(self, arcs_a, frac_a, arcs_b, frac_b, disconnected: float = np.inf) -> np.ndarray:
        """Network distance from every (arc_a, frac_a) to every (arc_b, frac_b): shape (len a, len b).

        Same arc: |frac difference| * length (a small backward step is GPS noise, not a loop).
        Different arcs: rest of arc a + shortest path (end of a -> start of b) + start of arc b to the point.
        Unreachable within SP_CUTOFF -> `disconnected`.
        """
        g = self.geom
        out = np.full((len(arcs_a), len(arcs_b)), disconnected, dtype=float)
        for i, (a, fa) in enumerate(zip(arcs_a, frac_a)):
            d_src = self.lengths_from(int(g.to_node[a]))
            tail = (1.0 - fa) * g.arc_len[a]
            for j, (b, fb) in enumerate(zip(arcs_b, frac_b)):
                if a == b:
                    out[i, j] = abs(fb - fa) * g.arc_len[a]
                    continue
                d = d_src.get(int(g.from_node[b]))
                if d is not None:
                    out[i, j] = tail + d + fb * g.arc_len[b]
        return out

    def path_nodes(self, src: int, dst: int) -> list[int] | None:
        """Node sequence of the shortest road path src -> dst within SP_CUTOFF metres, or None."""
        if src == dst:
            return [src]
        try:
            _, path = nx.single_source_dijkstra(self.G, src, target=dst, cutoff=SP_CUTOFF, weight="length")
            return path
        except (nx.NetworkXNoPath, KeyError):
            return None

    def adjacency(self, arcs_a, arcs_b) -> np.ndarray:
        """1 where arc a flows directly into arc b (same arc counts), shape (len a, len b)."""
        g = self.geom
        a = np.asarray(arcs_a)[:, None]
        b = np.asarray(arcs_b)[None, :]
        return ((a == b) | (g.to_node[a] == g.from_node[b])).astype(float)
