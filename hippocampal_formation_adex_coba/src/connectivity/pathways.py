
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass(frozen=True)
class SynapseSpec:
    # Conductance increments (siemens) per presyn spike
    w_ampa_nS: float = 0.0
    w_nmda_nS: float = 0.0
    w_gabaa_nS: float = 0.0
    w_gabab_nS: float = 0.0

    # Optional scaling for specific target layer (applied multiplicatively)
    layer_weight_scale: float = 1.0


@dataclass(frozen=True)
class PathwaySpec:
    name: str
    pre_pop: str
    post_pop: str
    kind: str  # "exc" or "inh"
    # Outdegree target per presyn neuron (mean/std)
    k_out_mean: float
    k_out_std: float

    # Distance kernel
    radius_mm: float
    sigma_mm: float

    # Delay: distance / velocity + base
    velocity_m_per_s: float
    base_delay_ms: float

    # Target layer distribution: dict layer->weight (not necessarily soma layer)
    target_layers: Dict[str, float]

    # Synaptic weights
    synapse: SynapseSpec

    exclude_self: bool = False


def load_pathways(path: str | Path) -> List[PathwaySpec]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "pathways" not in data:
        raise ValueError(f"Expected YAML with top-level key `pathways`: {p}")

    out: List[PathwaySpec] = []
    for item in data["pathways"]:
        syn = item.get("synapse", {}) or {}
        out.append(
            PathwaySpec(
                name=str(item["name"]),
                pre_pop=str(item["pre_pop"]),
                post_pop=str(item["post_pop"]),
                kind=str(item.get("kind", "exc")),
                k_out_mean=float(item.get("k_out_mean", 50)),
                k_out_std=float(item.get("k_out_std", 10)),
                radius_mm=float(item.get("radius_mm", 1.0)),
                sigma_mm=float(item.get("sigma_mm", 0.4)),
                velocity_m_per_s=float(item.get("velocity_m_per_s", 0.3)),
                base_delay_ms=float(item.get("base_delay_ms", 1.0)),
                target_layers=dict(item.get("target_layers", {})),
                synapse=SynapseSpec(
                    w_ampa_nS=float(syn.get("w_ampa_nS", 0.0)),
                    w_nmda_nS=float(syn.get("w_nmda_nS", 0.0)),
                    w_gabaa_nS=float(syn.get("w_gabaa_nS", 0.0)),
                    w_gabab_nS=float(syn.get("w_gabab_nS", 0.0)),
                    layer_weight_scale=float(syn.get("layer_weight_scale", 1.0)),
                ),
                exclude_self=bool(item.get("exclude_self", False)),
            )
        )
    return out
