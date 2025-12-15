
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.paths import cache_root, ensure_dir


# -----------------------------
# Minimal priors (embedded)
# -----------------------------
# These are deliberately *not* claimed to be comprehensive Hippocampome exports.
# They serve as robust fallbacks when a machine-readable dump is not available.
DEFAULT_PRIORS: Dict[str, Dict[str, Any]] = {
    # Excitatory principal cells
    "DG_Granule": {
        "class": "excitatory",
        "region": "DG",
        "firing_phenotype": "RS",  # regular spiking
        "soma_layer": "GCL",
        "dendrite_layers": ["IML", "MML", "OML"],
        "axon_targets": ["CA3_LUC", "CA3_SR"],  # mossy fibers mainly stratum lucidum
    },
    "CA3_Pyramidal": {
        "class": "excitatory",
        "region": "CA3",
        "firing_phenotype": "RS",
        "soma_layer": "SP",
        "dendrite_layers": ["SO", "SR", "SLM"],
        "axon_targets": ["CA3_SO", "CA3_SR", "CA1_SR"],  # recurrent + Schaffer-like
    },
    "CA2_Pyramidal": {
        "class": "excitatory",
        "region": "CA2",
        "firing_phenotype": "RS",
        "soma_layer": "SP",
        "dendrite_layers": ["SO", "SR", "SLM"],
        "axon_targets": ["CA1_SR", "SUB"],
    },
    "CA1_Pyramidal": {
        "class": "excitatory",
        "region": "CA1",
        "firing_phenotype": "RS",
        "soma_layer": "SP",
        "dendrite_layers": ["SO", "SR", "SLM"],
        "axon_targets": ["SUB", "EC_deep"],
    },
    "SUB_Pyramidal": {
        "class": "excitatory",
        "region": "SUB",
        "firing_phenotype": "RS",
        "soma_layer": "SP",
        "dendrite_layers": ["MOLECULAR", "PYR", "POLY"],
        "axon_targets": ["EC_deep"],
    },
    "EC_L2_Stellate": {
        "class": "excitatory",
        "region": "EC",
        "firing_phenotype": "RS",
        "soma_layer": "L2",
        "dendrite_layers": ["L1", "L2", "L3"],
        "axon_targets": ["DG_ML", "CA3_SLM", "CA2_SLM"],
    },
    "EC_L3_Pyramidal": {
        "class": "excitatory",
        "region": "EC",
        "firing_phenotype": "RS",
        "soma_layer": "L3",
        "dendrite_layers": ["L1", "L2", "L3"],
        "axon_targets": ["CA1_SLM", "SUB_ML"],
    },
    # Inhibitory interneurons
    "PV_Basket": {
        "class": "inhibitory",
        "marker": "PV",
        "firing_phenotype": "FS",  # fast spiking
        "targeting": "perisomatic",
        "preferred_target_layers": ["SP", "GCL"],
    },
    "SOM_OLM": {
        "class": "inhibitory",
        "marker": "SOM",
        "firing_phenotype": "LTS",  # low-threshold / adapting
        "targeting": "distal_dendritic",
        "preferred_target_layers": ["SLM", "OML", "MML"],
    },
    "SOM_HIPP": {
        "class": "inhibitory",
        "marker": "SOM",
        "firing_phenotype": "LTS",
        "targeting": "distal_dendritic",
        "preferred_target_layers": ["OML", "MML"],
    },
}


@dataclass
class HippocampomeCache:
    enabled: bool = True
    url: str = "https://hippocampome.org/"  # base URL (used only if JSON endpoint is supplied)
    json_endpoint: Optional[str] = None     # optional full JSON URL to a machine-readable dump
    cache_filename: str = "hippocampome_priors.json"

    def cache_path(self) -> Path:
        return ensure_dir(cache_root()) / self.cache_filename


def _download_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "hippo3d/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def load_priors(cache: HippocampomeCache) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict of neuron-type priors. If `json_endpoint` is provided and reachable,
    it is cached locally and used. Otherwise, falls back to DEFAULT_PRIORS.

    This design is intentional: Hippocampome's web portal and data releases can change.
    We keep the build robust while still allowing opt-in machine-readable imports.
    """
    if not cache.enabled:
        return dict(DEFAULT_PRIORS)

    p = cache.cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    if cache.json_endpoint:
        try:
            priors = _download_json(cache.json_endpoint)
            if isinstance(priors, dict):
                p.write_text(json.dumps(priors, indent=2), encoding="utf-8")
                return priors  # type: ignore[return-value]
        except Exception:
            pass

    # fallback
    p.write_text(json.dumps(DEFAULT_PRIORS, indent=2), encoding="utf-8")
    return dict(DEFAULT_PRIORS)
