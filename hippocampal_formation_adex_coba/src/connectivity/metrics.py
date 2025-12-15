
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any
import numpy as np

from .builder import PathwayEdges


def pathway_stats(edges: PathwayEdges, n_pre: int, n_post: int) -> Dict[str, Any]:
    if edges.pre_idx.size == 0:
        return {
            "n_edges": 0,
            "fanout_mean": 0.0,
            "fanin_mean": 0.0,
            "dist_mm_mean": None,
            "delay_ms_mean": None,
        }
    fanout = np.bincount(edges.pre_idx, minlength=n_pre)
    fanin = np.bincount(edges.post_idx, minlength=n_post)
    return {
        "n_edges": int(edges.pre_idx.size),
        "fanout_mean": float(fanout.mean()),
        "fanout_std": float(fanout.std()),
        "fanin_mean": float(fanin.mean()),
        "fanin_std": float(fanin.std()),
        "dist_mm_mean": float(edges.dist_mm.mean()),
        "dist_mm_std": float(edges.dist_mm.std()),
        "delay_ms_mean": float(edges.delay_ms.mean()),
        "delay_ms_std": float(edges.delay_ms.std()),
    }
