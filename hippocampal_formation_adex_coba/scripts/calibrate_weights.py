from __future__ import annotations

"""
Automatic calibration of pathway synaptic weights via short Brian2 simulations.

Runs as:
  python scripts/calibrate_weights.py --config configs/small.yaml --trials 50 --downscale 0.30

Writes:
  runs/<run_name>_calib_<timestamp>/calibration/best_multipliers.yaml
  runs/<run_name>_calib_<timestamp>/calibration/trials.yaml
"""

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Make "import src...." work when running as a script on Windows
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

from brian2 import Network, SpikeMonitor, PoissonInput, ms, nS, Hz, prefs

from src.data.ccf_atlas import CCFAtlas
from src.celltypes.library import build_celltype_library, CellTypeSpec
from src.anatomy.regions import load_region_geometry, RegionGeometry
from src.anatomy.placement import layer_defs_from_config, LayerDef

# try to import your volumetric sampler; fallback to layer voxel sampling if absent
try:
    from src.anatomy.placement import sample_somata  # preferred (volumetric)
    _HAS_SAMPLE_SOMATA = True
except Exception:
    _HAS_SAMPLE_SOMATA = False
    from src.anatomy.placement import sample_somata_in_layer  # fallback

from src.connectivity.pathways import load_pathways
from src.connectivity.builder import PopulationGeometry, build_connectivity, PathwayEdges
from src.sim.model import make_population, make_synapses, SynapseKinetics, DEFAULT_KINETICS
from src.utils.seed import set_global_seeds


# -----------------------------
# Basic IO helpers
# -----------------------------

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_yaml(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _is_inhib_pop(pop_name: str) -> bool:
    u = pop_name.upper()
    return ("PV" in u) or ("SOM" in u) or ("INH" in u) or ("GABA" in u)


def _region_of(pop_name: str) -> str:
    for r in ("EC", "DG", "CA3", "CA2", "CA1", "SUB"):
        if pop_name.startswith(r + "_"):
            return r
    return "OTHER"


# -----------------------------
# Celltype resolution (match runner.py logic)
# -----------------------------

def resolve_celltype_key(cell_type: str, pop_name: str, pop_specs: Dict[str, CellTypeSpec]) -> str:
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

    # Excitatory defaults
    if "EC" in name and "EC_RS" in pop_specs:
        return "EC_RS"
    if "RS" in pop_specs:
        return "RS"

    return next(iter(pop_specs.keys()))


# -----------------------------
# Calibration structures
# -----------------------------

@dataclass
class Multipliers:
    exc_local: float
    exc_long: float
    inh_local: float


@dataclass
class Targets:
    min_hz: float = 0.2
    max_hz: float = 8.0
    min_spikes: int = 50
    window_s: float = 0.25
    max_latency_s: float = 0.40
    cascade: Tuple[str, ...] = ("EC", "DG", "CA3", "CA1", "SUB")


@dataclass
class TrialResult:
    score: float
    propagation: float
    rate_penalty: float
    multipliers: Multipliers
    details: Dict[str, Any]


# -----------------------------
# Build populations + connectivity ONCE
# -----------------------------

def build_static_model(
    cfg: Dict[str, Any],
    *,
    rng: np.random.Generator,
    downscale: float,
) -> Tuple[Dict[str, CellTypeSpec], Dict[str, PopulationGeometry], Dict[str, PathwayEdges], SynapseKinetics, Dict[str, Any]]:
    anat = cfg["anatomy"]
    atlas = CCFAtlas(atlas_name=str(anat.get("atlas_name", "allen_mouse_25um")))

    region_cfg: Dict[str, Any] = anat["regions"]
    region_geoms: Dict[str, RegionGeometry] = {}
    for region_key, rcfg in region_cfg.items():
        sname = str(rcfg["structure_name"])
        region_geoms[region_key] = load_region_geometry(atlas, sname)

    layer_defs: Dict[str, Dict[str, LayerDef]] = {}
    for region_key, rcfg in region_cfg.items():
        layer_defs[region_key] = layer_defs_from_config(rcfg["layers"])

    cell_lib = build_celltype_library(cfg.get("hippocampome", {}))

    pops_cfg: Dict[str, Any] = cfg["populations"]
    pop_geoms: Dict[str, PopulationGeometry] = {}

    for pop_name, pcfg in pops_cfg.items():
        region_key = str(pcfg["region"])
        layer_name = str(pcfg["soma_layer"])
        cell_type = str(pcfg["cell_type"])
        n0 = int(pcfg["n"])
        n = max(10, int(round(n0 * downscale)))

        geom = region_geoms[region_key]
        layer = layer_defs[region_key][layer_name]

        if _HAS_SAMPLE_SOMATA:
            xyz_mm = sample_somata(
                atlas=atlas,
                geom=geom,
                layer=layer,
                n=n,
                rng=rng,
                jitter_within_voxel=True,
            )
        else:
            # fallback (surface-like): sample voxels within laminar band, then convert to mm
            vox_zyx = sample_somata_in_layer(atlas, geom, layer, n=n, rng=rng)
            xyz_mm = atlas.vox_to_mm(vox_zyx)

        soma_layer = np.array([layer_name] * n, dtype=object)

        # NOTE: PopulationGeometry.cell_type stores the *cell_type string* from config
        # and will be resolved to a library key later (resolve_celltype_key).
        pop_geoms[pop_name] = PopulationGeometry(
            name=pop_name,
            xyz_mm=xyz_mm,
            soma_layer=soma_layer,
            region=region_key,
            cell_class=("inhibitory" if _is_inhib_pop(pop_name) else "excitatory"),
            cell_type=cell_type,
        )

    # Connectivity
    conn = cfg["connectivity"]
    pathways = load_pathways(conn["pathways_file"])
    scale = float(conn.get("scale", 1.0)) * downscale
    edges = build_connectivity(pathways, pop_geoms, rng=rng, scale=scale, show_progress=False)

    # Kinetics + sim cfg
    sim_cfg = cfg.get("simulation", {}) or {}
    kin_cfg = sim_cfg.get("synapse_kinetics", {}) or {}
    kinetics = SynapseKinetics(**{k: float(v) for k, v in kin_cfg.items()}) if kin_cfg else DEFAULT_KINETICS

    return cell_lib, pop_geoms, edges, kinetics, sim_cfg


# -----------------------------
# Short sim for a given multiplier set
# -----------------------------

def run_short_sim(
    *,
    pop_specs: Dict[str, CellTypeSpec],
    pop_geoms: Dict[str, PopulationGeometry],
    edges: Dict[str, PathwayEdges],
    kinetics: SynapseKinetics,
    sim_cfg: Dict[str, Any],
    multipliers: Multipliers,
    t_sim_s: float,
    dt_ms: float,
    codegen: str,
    seed: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    prefs.codegen.target = codegen

    # Build neuron groups
    groups: Dict[str, Any] = {}
    pop_sizes: Dict[str, int] = {}

    for pop_name, geom in pop_geoms.items():
        ckey = resolve_celltype_key(geom.cell_type, pop_name, pop_specs)
        ctype = pop_specs[ckey]

        G = make_population(
            name=pop_name,
            n=geom.xyz_mm.shape[0],
            adex=ctype.adex,
            kinetics=kinetics,
            dt_ms=dt_ms,
            codegen_target=codegen,
        )
        groups[pop_name] = G
        pop_sizes[pop_name] = int(G.N)

    # Background (exactly as runner)
    bg = (sim_cfg.get("background", {}) or {})
    bg_rate_hz = float(bg.get("rate_hz", 0.0))
    bg_n = int(bg.get("n_sources", 0))
    bg_w_ampa_nS = float(bg.get("w_ampa_nS", 0.0))
    bg_w_nmda_nS = float(bg.get("w_nmda_nS", 0.0))
    bg_targets = bg.get("targets", list(groups.keys()))

    poisson_inputs = []
    for pop_name in bg_targets:
        if pop_name not in groups:
            continue
        G = groups[pop_name]
        if bg_n > 0 and bg_rate_hz > 0 and bg_w_ampa_nS > 0:
            poisson_inputs.append(PoissonInput(G, "g_ampa", N=bg_n, rate=bg_rate_hz * Hz, weight=bg_w_ampa_nS * nS))
        if bg_n > 0 and bg_rate_hz > 0 and bg_w_nmda_nS > 0:
            poisson_inputs.append(PoissonInput(G, "g_nmda", N=bg_n, rate=bg_rate_hz * Hz, weight=bg_w_nmda_nS * nS))

    # Synapses (apply multipliers here; no need to rebuild edges)
    synapses = []
    for pname, e in edges.items():
        pre_name = e.spec.pre_pop
        post_name = e.spec.post_pop
        pre = groups[pre_name]
        post = groups[post_name]

        pre_reg = pop_geoms[pre_name].region
        post_reg = pop_geoms[post_name].region
        within = (pre_reg == post_reg)
        pre_inh = _is_inhib_pop(pre_name)
        pre_exc = not pre_inh

        if pre_exc and within:
            mult_e, mult_i = multipliers.exc_local, 1.0
        elif pre_exc and not within:
            mult_e, mult_i = multipliers.exc_long, 1.0
        else:
            mult_e, mult_i = 1.0, multipliers.inh_local

        S = make_synapses(pre, post, name=f"syn_{pname}")
        if e.pre_idx.size > 0:
            S.connect(i=e.pre_idx, j=e.post_idx)
            S.w_ampa = (e.w_ampa_nS * mult_e) * nS
            S.w_nmda = (e.w_nmda_nS * mult_e) * nS
            S.w_gabaa = (e.w_gabaa_nS * mult_i) * nS
            S.w_gabab = (e.w_gabab_nS * mult_i) * nS
            S.delay = e.delay_ms * ms
        synapses.append(S)

    # Spikes only
    spike_monitors: Dict[str, SpikeMonitor] = {p: SpikeMonitor(G, name=f"sp_{p}") for p, G in groups.items()}

    net = Network()
    for obj in list(groups.values()) + synapses + list(spike_monitors.values()) + poisson_inputs:
        net.add(obj)

    net.run(t_sim_s * 1000.0 * ms, report=None)

    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pop_name, mon in spike_monitors.items():
        t_s = np.asarray(mon.t / ms) / 1000.0
        i = np.asarray(mon.i, dtype=int)
        spikes[pop_name] = (t_s, i)

    return spikes, pop_sizes


# -----------------------------
# Scoring
# -----------------------------

def score_trial(
    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pop_sizes: Dict[str, int],
    t_sim_s: float,
    targets: Targets,
) -> Tuple[float, float, float, Dict[str, Any]]:
    # mean rate per pop
    rates: Dict[str, float] = {}
    for p, (t_s, i) in spikes.items():
        N = max(1, pop_sizes.get(p, 1))
        rates[p] = float(i.size) / (t_sim_s * N + 1e-12)

    # penalty for rate outside band
    pen = 0.0
    for p, r in rates.items():
        if r < targets.min_hz:
            pen += (targets.min_hz - r) / max(1e-9, targets.min_hz)
        elif r > targets.max_hz:
            pen += (r - targets.max_hz) / max(1e-9, targets.max_hz)

    # region grouping
    region_pops: Dict[str, List[str]] = {}
    for p in spikes.keys():
        region_pops.setdefault(_region_of(p), []).append(p)

    def region_first_spike(reg: str) -> Optional[float]:
        mins = []
        for p in region_pops.get(reg, []):
            t_s, _ = spikes[p]
            if t_s.size:
                mins.append(float(t_s.min()))
        return min(mins) if mins else None

    def region_spikes_in_window(reg: str, t0: float, t1: float) -> int:
        tot = 0
        for p in region_pops.get(reg, []):
            t_s, _ = spikes[p]
            if t_s.size:
                tot += int(np.sum((t_s >= t0) & (t_s < t1)))
        return tot

    cascade = list(targets.cascade)
    t_ec = region_first_spike("EC")

    if t_ec is None:
        propagation = 0.0
    else:
        ok = 0
        hops = 0
        t_up = t_ec
        for reg in cascade[1:]:
            hops += 1
            t_search_end = min(t_sim_s, t_up + targets.max_latency_s)
            t_win_end = min(t_sim_s, t_up + targets.window_s)
            count = region_spikes_in_window(reg, t_up, t_win_end)
            if count >= targets.min_spikes:
                ok += 1
                t_reg = region_first_spike(reg)
                t_up = t_reg if (t_reg is not None and t_reg < t_search_end) else t_search_end
            else:
                t_up = t_search_end
        propagation = ok / max(1, hops)

    score = 1.5 * propagation - 1.0 * pen
    details = {"rates_hz": rates, "propagation": propagation, "rate_penalty": pen}
    return float(score), float(propagation), float(pen), details


# -----------------------------
# Proposal distributions
# -----------------------------

def propose_random(rng: np.random.Generator) -> Multipliers:
    def logu(lo: float, hi: float) -> float:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    m = Multipliers(
        exc_local=logu(0.02, 2.0),
        exc_long=logu(0.05, 3.0),
        inh_local=logu(0.1, 8.0),
    )
    return Multipliers(
        exc_local=_clamp(m.exc_local, 0.01, 5.0),
        exc_long=_clamp(m.exc_long, 0.01, 8.0),
        inh_local=_clamp(m.inh_local, 0.01, 20.0),
    )


def propose_jitter(rng: np.random.Generator, best: Multipliers, sigma: float = 0.35) -> Multipliers:
    def j(x: float) -> float:
        return float(x * np.exp(rng.normal(0.0, sigma)))

    m = Multipliers(exc_local=j(best.exc_local), exc_long=j(best.exc_long), inh_local=j(best.inh_local))
    return Multipliers(
        exc_local=_clamp(m.exc_local, 0.01, 5.0),
        exc_long=_clamp(m.exc_long, 0.01, 8.0),
        inh_local=_clamp(m.inh_local, 0.01, 20.0),
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--downscale", type=float, default=0.30)
    ap.add_argument("--t_sim_s", type=float, default=1.2)
    ap.add_argument("--dt_ms", type=float, default=0.1)
    ap.add_argument("--codegen", type=str, default="numpy", choices=["numpy", "cython"])
    ap.add_argument("--random_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--min_hz", type=float, default=0.2)
    ap.add_argument("--max_hz", type=float, default=8.0)
    ap.add_argument("--min_spikes", type=int, default=50)
    ap.add_argument("--window_s", type=float, default=0.25)
    ap.add_argument("--max_latency_s", type=float, default=0.40)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_name = str(cfg.get("run_name", "hippo3d"))
    out_root = REPO_ROOT / "runs" / f"{run_name}_calib_{_now_tag()}"
    calib_dir = out_root / "calibration"
    _ensure_dir(calib_dir)

    rng = np.random.default_rng(int(args.seed))

    targets = Targets(
        min_hz=float(args.min_hz),
        max_hz=float(args.max_hz),
        min_spikes=int(args.min_spikes),
        window_s=float(args.window_s),
        max_latency_s=float(args.max_latency_s),
    )

    print(f"[calibrate] trials={args.trials} downscale={args.downscale} t_sim_s={args.t_sim_s} codegen={args.codegen}")

    # Build static model once
    pop_specs, pop_geoms, edges, kinetics, sim_cfg = build_static_model(
        cfg, rng=rng, downscale=float(args.downscale)
    )

    best: Optional[TrialResult] = None
    results: List[TrialResult] = []
    n_random = int(round(args.trials * float(args.random_frac)))

    for k in range(int(args.trials)):
        if best is None or k < n_random:
            m = propose_random(rng)
        else:
            m = propose_jitter(rng, best.multipliers, sigma=0.35)

        spikes, pop_sizes = run_short_sim(
            pop_specs=pop_specs,
            pop_geoms=pop_geoms,
            edges=edges,
            kinetics=kinetics,
            sim_cfg=sim_cfg,
            multipliers=m,
            t_sim_s=float(args.t_sim_s),
            dt_ms=float(args.dt_ms),
            codegen=str(args.codegen),
            seed=int(args.seed) + 1000 + k,
        )

        score, prop, pen, details = score_trial(
            spikes=spikes,
            pop_sizes=pop_sizes,
            t_sim_s=float(args.t_sim_s),
            targets=targets,
        )

        tr = TrialResult(score=score, propagation=prop, rate_penalty=pen, multipliers=m, details=details)
        results.append(tr)

        if best is None or tr.score > best.score:
            best = tr

        print(
            f"[trial {k:03d}] score={tr.score:+.3f}  prop={tr.propagation:.2f}  rate_pen={tr.rate_penalty:.2f}  "
            f"mE_loc={m.exc_local:.3g} mE_long={m.exc_long:.3g} mI={m.inh_local:.3g}"
        )

    assert best is not None

    _save_yaml(asdict(best.multipliers), calib_dir / "best_multipliers.yaml")
    _save_yaml(
        {
            "best": {
                "score": best.score,
                "propagation": best.propagation,
                "rate_penalty": best.rate_penalty,
                "multipliers": asdict(best.multipliers),
            },
            "targets": asdict(targets),
            "trials": [
                {
                    "score": r.score,
                    "propagation": r.propagation,
                    "rate_penalty": r.rate_penalty,
                    "multipliers": asdict(r.multipliers),
                }
                for r in results
            ],
        },
        calib_dir / "trials.yaml",
    )

    print("\n[calibrate] DONE")
    print(f"  best score: {best.score:+.3f}")
    print(f"  best multipliers: {asdict(best.multipliers)}")
    print(f"  wrote: {calib_dir/'best_multipliers.yaml'}")
    print(f"  wrote: {calib_dir/'trials.yaml'}")


if __name__ == "__main__":
    main()
