from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from brian2 import (
    Network, SpikeMonitor, StateMonitor, PoissonInput,
    ms, mV, nS, pA, Hz, second
)

from ..celltypes.library import CellTypeSpec
from ..connectivity.builder import PopulationGeometry, PathwayEdges
from .model import make_population, make_synapses, SynapseKinetics, DEFAULT_KINETICS
from .io import save_json, save_npz

# NEW: your single, information-dense activity plot
from .plots import save_activity_figure


@dataclass
class SimOutputs:
    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]]
    summary: Dict[str, Any]


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def run_simulation(
    config: Dict[str, Any],
    pop_specs: Dict[str, CellTypeSpec],
    pop_geom: Dict[str, PopulationGeometry],
    edges: Dict[str, PathwayEdges],
    out_dir: Path,
) -> SimOutputs:
    sim_cfg = config["simulation"]
    dt_ms = float(sim_cfg.get("dt_ms", 0.1))
    t_sim_s = float(sim_cfg.get("t_sim_s", 1.0))
    codegen = str(sim_cfg.get("codegen_target", "cython"))

    kinetics_cfg = sim_cfg.get("synapse_kinetics", {}) or {}
    kinetics = SynapseKinetics(**{k: float(v) for k, v in kinetics_cfg.items()}) if kinetics_cfg else DEFAULT_KINETICS

    # ---- resolve celltype keys robustly (your existing alias logic)
    def resolve_celltype_key(cell_type: str, pop_name: str) -> str:
        aliases = {
            "EC_L2_Stellate": "EC_L2",
            "EC_L2_Pyramidal": "EC_L2",
            "EC_L3_Pyramidal": "EC_L3",
            "EC_L3": "EC_L3",
            "EC_L2": "EC_L2",

            "DG_Granule": "DG_GC",
            "DG_GC": "DG_GC",

            "CA1_Pyramidal": "CA1_Pyr",
            "CA2_Pyramidal": "CA2_Pyr",
            "CA3_Pyramidal": "CA3_Pyr",
            "SUB_Pyramidal": "SUB_Pyr",

            "PV_Basket": "PV",
            "SOM_OLM": "SOM",
            "SOM_HIPP": "SOM",
        }

        if cell_type in pop_specs:
            return cell_type
        if cell_type in aliases and aliases[cell_type] in pop_specs:
            return aliases[cell_type]

        name = f"{pop_name} {cell_type}".upper()
        if "PV" in name and "PV" in pop_specs:
            return "PV"
        if ("SOM" in name or "OLM" in name or "HIPP" in name) and "SOM" in pop_specs:
            return "SOM"

        if "EC" in name and "EC_RS" in pop_specs:
            return "EC_RS"
        if "RS" in pop_specs:
            return "RS"
        return next(iter(pop_specs.keys()))

    # ---- build neuron groups
    groups: Dict[str, Any] = {}
    pop_sizes: Dict[str, int] = {}

    for pop_name, geom in pop_geom.items():
        ctype_key = resolve_celltype_key(geom.cell_type, pop_name)
        ctype = pop_specs[ctype_key]
        G = make_population(
            name=pop_name,
            n=int(geom.xyz_mm.shape[0]),
            adex=ctype.adex,
            kinetics=kinetics,
            dt_ms=dt_ms,
            codegen_target=codegen,
        )
        groups[pop_name] = G
        pop_sizes[pop_name] = int(geom.xyz_mm.shape[0])

    # ---- background excitation (PoissonInput goes to conductances directly)
    bg_cfg = sim_cfg.get("background", {}) or {}

    def iter_background_entries(cfg: Any) -> List[Dict[str, Any]]:
        if isinstance(cfg, list):
            return [entry for entry in cfg if isinstance(entry, dict)]

        if isinstance(cfg, dict):
            profiles = cfg.get("profiles")
            base = {k: v for k, v in cfg.items() if k != "profiles"}

            if profiles is None:
                return [base]

            if isinstance(profiles, dict):
                profile_entries = profiles.values()
            else:
                profile_entries = profiles

            entries: List[Dict[str, Any]] = []
            for entry in profile_entries:
                if isinstance(entry, dict):
                    merged = {**base, **entry}
                    entries.append(merged)
            return entries

        return []

    poisson_inputs = []
    for bg in iter_background_entries(bg_cfg):
        bg_rate_hz = float(bg.get("rate_hz", 0.0))
        bg_n = int(bg.get("n_sources", 0))
        bg_w_ampa_nS = float(bg.get("w_ampa_nS", 0.0))
        bg_w_nmda_nS = float(bg.get("w_nmda_nS", 0.0))
        bg_targets = list(bg.get("targets", list(groups.keys())))

        if bg_n <= 0 or bg_rate_hz <= 0:
            continue

        for pop_name in bg_targets:
            if pop_name not in groups:
                continue
            G = groups[pop_name]
            if bg_w_ampa_nS > 0:
                poisson_inputs.append(
                    PoissonInput(G, "g_ampa", N=bg_n, rate=bg_rate_hz * Hz, weight=bg_w_ampa_nS * nS)
                )
            if bg_w_nmda_nS > 0:
                poisson_inputs.append(
                    PoissonInput(G, "g_nmda", N=bg_n, rate=bg_rate_hz * Hz, weight=bg_w_nmda_nS * nS)
                )

    # ---- synapses for each pathway
    synapses = []
    for pname, e in edges.items():
        pre = groups[e.spec.pre_pop]
        post = groups[e.spec.post_pop]
        S = make_synapses(pre, post, name=f"syn_{pname}")
        if e.pre_idx.size > 0:
            S.connect(i=e.pre_idx, j=e.post_idx)
            S.w_ampa = e.w_ampa_nS * nS
            S.w_nmda = e.w_nmda_nS * nS
            S.w_gabaa = e.w_gabaa_nS * nS
            S.w_gabab = e.w_gabab_nS * nS
            S.delay = e.delay_ms * ms
        synapses.append(S)

    # ---- monitors
    spike_monitors: Dict[str, SpikeMonitor] = {}
    state_monitors: Dict[str, StateMonitor] = {}

    rec_cfg = sim_cfg.get("record", {}) or {}
    n_state = int(rec_cfg.get("n_state_neurons_per_pop", 3))

    for pop_name, G in groups.items():
        spike_monitors[pop_name] = SpikeMonitor(G, name=f"sp_{pop_name}")
        if n_state > 0:
            idx = np.arange(min(n_state, int(G.N)))
            state_monitors[pop_name] = StateMonitor(
                G,
                variables=["v", "g_ampa", "g_nmda", "g_gabaa", "g_gabab"],
                record=idx,
                name=f"st_{pop_name}",
            )

    # ---- run network
    net = Network()
    for obj in (
        list(groups.values())
        + synapses
        + list(spike_monitors.values())
        + list(state_monitors.values())
        + poisson_inputs
    ):
        net.add(obj)

    net.run(t_sim_s * second, report="text")

    # ---- collect spikes
    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    summary: Dict[str, Any] = {"t_sim_s": t_sim_s, "dt_ms": dt_ms, "populations": {}}

    for pop_name, mon in spike_monitors.items():
        t = _as_float_array(mon.t / second)          # seconds
        i = np.asarray(mon.i, dtype=int)
        spikes[pop_name] = (t, i)

        fr = float(i.size) / (t_sim_s * max(1, pop_sizes[pop_name]))
        summary["populations"][pop_name] = {
            "n": int(pop_sizes[pop_name]),
            "n_spikes": int(i.size),
            "mean_rate_hz": float(fr),
        }

    # ---- collect state traces for plotting
    state_traces: Dict[str, Dict[str, np.ndarray]] = {}
    for pop_name, mon in state_monitors.items():
        t_s = _as_float_array(mon.t / second)
        # mon.v etc are (n_rec, T) with Brian units
        state_traces[pop_name] = {
            "t_s": t_s,
            "v_mV": np.asarray(mon.v / mV, dtype=float),
            "g_ampa_nS": np.asarray(mon.g_ampa / nS, dtype=float),
            "g_nmda_nS": np.asarray(mon.g_nmda / nS, dtype=float),
            "g_gabaa_nS": np.asarray(mon.g_gabaa / nS, dtype=float),
            "g_gabab_nS": np.asarray(mon.g_gabab / nS, dtype=float),
        }

    # ---- save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    save_json(summary, out_dir / "summary.json")

    # Save a single rich activity overview figure
    save_activity_figure(
        spikes=spikes,
        pop_sizes=pop_sizes,
        out_path=out_dir / "plots" / "activity_overview.png",
        t_max_s=t_sim_s,
        max_raster_neurons_per_pop=int(rec_cfg.get("max_raster_neurons_per_pop", 250)),
        raster_subsample_seed=int(config.get("seed", 1)),
        rate_bin_ms=float(rec_cfg.get("rate_bin_ms", 5.0)),
        rate_smooth_ms=float(rec_cfg.get("rate_smooth_ms", 20.0)),
        state_traces=state_traces if len(state_traces) else None,
    )

    # Save raw spikes and state
    npz = {}
    for pop_name, (t, i) in spikes.items():
        npz[f"{pop_name}_spike_t_s"] = t
        npz[f"{pop_name}_spike_i"] = i

    for pop_name, tr in state_traces.items():
        npz[f"{pop_name}_state_t_s"] = tr["t_s"]
        npz[f"{pop_name}_v_mV"] = tr["v_mV"]
        npz[f"{pop_name}_g_ampa_nS"] = tr["g_ampa_nS"]
        npz[f"{pop_name}_g_nmda_nS"] = tr["g_nmda_nS"]
        npz[f"{pop_name}_g_gabaa_nS"] = tr["g_gabaa_nS"]
        npz[f"{pop_name}_g_gabab_nS"] = tr["g_gabab_nS"]

    save_npz(npz, out_dir / "data" / "sim_outputs.npz")

    return SimOutputs(spikes=spikes, summary=summary)
