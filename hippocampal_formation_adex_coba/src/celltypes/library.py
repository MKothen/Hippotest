
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..data.hippocampome import load_priors, HippocampomeCache
from .adex_params import PRESETS, AdExParams


@dataclass
class CellTypeSpec:
    name: str
    cls: str  # excitatory/inhibitory
    firing_phenotype: str  # RS/FS/LTS
    adex: AdExParams
    meta: Dict[str, Any]


def build_celltype_library(hippo_cfg: Dict[str, Any]) -> Dict[str, CellTypeSpec]:
    """
    Creates a library of cell types.

    Strategy:
    1) Load neuron-type priors (Hippocampome dump if reachable, else embedded priors).
    2) Map each type's firing phenotype to an AdEx preset.
    3) Allow overrides later via YAML configs (handled at simulation build time).
    """
    cache = HippocampomeCache(
        enabled=bool(hippo_cfg.get("enabled", True)),
        url=str(hippo_cfg.get("url", "https://hippocampome.org/")),
        json_endpoint=hippo_cfg.get("json_endpoint", None),
        cache_filename=str(hippo_cfg.get("cache_filename", "hippocampome_priors.json")),
    )
    priors = load_priors(cache)

    lib: Dict[str, CellTypeSpec] = {}
    for tname, meta in priors.items():
        phen = str(meta.get("firing_phenotype", "RS"))
        if phen not in PRESETS:
            phen = "RS"
        cls = str(meta.get("class", "excitatory"))
        lib[tname] = CellTypeSpec(
            name=tname,
            cls=cls,
            firing_phenotype=phen,
            adex=PRESETS[phen],
            meta=dict(meta),
        )
    return lib
