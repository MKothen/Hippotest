from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .plasticity import (
    BTSPLearningRule,
    STPMechanism,
    TemporalScalingConfig,
    bidirectional_plasticity,
    temporal_scaling_factor,
)


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _region_of(pop_name: str) -> str:
    for r in ("EC", "DG", "CA3", "CA2", "CA1", "SUB"):
        if pop_name.startswith(r + "_"):
            return r
    return "OTHER"


def _is_inhibitory(pop_name: str) -> bool:
    u = pop_name.upper()
    return ("PV" in u) or ("SOM" in u) or ("INH" in u) or ("GABA" in u)


def _sorted_pops(pop_names: List[str]) -> List[str]:
    # Region order first, then E before I, then name
    region_order = {"EC": 0, "DG": 1, "CA3": 2, "CA2": 3, "CA1": 4, "SUB": 5, "OTHER": 99}

    def key(n: str):
        return (region_order.get(_region_of(n), 99), 1 if _is_inhibitory(n) else 0, n)

    return sorted(pop_names, key=key)


def _bin_spikes_to_rate(
    t_s: np.ndarray,
    t_stop_s: float,
    n_neurons: int,
    bin_ms: float = 5.0,
    smooth_ms: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spike times to population rate (Hz) using binning + gaussian smoothing.
    """
    if t_stop_s <= 0:
        raise ValueError("t_stop_s must be > 0")

    dt = bin_ms / 1000.0
    edges = np.arange(0.0, t_stop_s + dt, dt)
    counts, _ = np.histogram(t_s, bins=edges)
    rate_hz = counts / (dt * max(1, n_neurons))

    # Gaussian smoothing in bins
    if smooth_ms > 0:
        sigma_bins = (smooth_ms / 1000.0) / dt
        # truncate at 4 sigma
        half = int(np.ceil(4 * sigma_bins))
        x = np.arange(-half, half + 1)
        k = np.exp(-0.5 * (x / sigma_bins) ** 2)
        k /= k.sum()
        rate_hz = np.convolve(rate_hz, k, mode="same")

    t_mid = 0.5 * (edges[:-1] + edges[1:])
    return t_mid, rate_hz


def save_activity_figure(
    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pop_sizes: Dict[str, int],
    out_path: Path,
    *,
    t_max_s: float,
    max_raster_neurons_per_pop: int = 250,
    raster_subsample_seed: int = 1,
    rate_bin_ms: float = 5.0,
    rate_smooth_ms: float = 20.0,
    state_traces: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    """
    Create a single, information-dense activity figure.

    spikes[pop] = (t_s, i)
      - t_s: spike times in seconds
      - i: neuron indices (int)
    pop_sizes[pop] = N
    state_traces (optional): per-pop arrays like {"t_s":..., "v_mV":..., "g_ampa_nS":..., "g_gabaa_nS":...}
    """
    _ensure_parent(out_path)

    pops = _sorted_pops(list(spikes.keys()))
    rng = np.random.default_rng(raster_subsample_seed)

    # Precompute per-pop binned rates
    rates_t: Dict[str, np.ndarray] = {}
    rates_hz: Dict[str, np.ndarray] = {}
    mean_rate: Dict[str, float] = {}

    for p in pops:
        t_s, i = spikes[p]
        N = int(pop_sizes.get(p, max(int(i.max()) + 1 if i.size else 1, 1)))
        tt, rr = _bin_spikes_to_rate(t_s, t_max_s, N, bin_ms=rate_bin_ms, smooth_ms=rate_smooth_ms)
        rates_t[p] = tt
        rates_hz[p] = rr
        mean_rate[p] = float(i.size) / (t_max_s * max(1, N))

    # Make a common time axis for heatmap (use the first)
    t_axis = rates_t[pops[0]]
    H = np.stack([np.interp(t_axis, rates_t[p], rates_hz[p]) for p in pops], axis=0)  # (P, T)

    # Region-summed traces
    regions = ["EC", "DG", "CA3", "CA2", "CA1", "SUB"]
    region_traces = {}
    for r in regions:
        idx = [k for k, p in enumerate(pops) if _region_of(p) == r]
        region_traces[r] = H[idx].sum(axis=0) if idx else np.zeros_like(t_axis)

    # --- Figure layout
    # 2 columns: left = raster, right = diagnostics panels
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.1, 1.0], width_ratios=[1.5, 1.0])

    ax_raster = fig.add_subplot(gs[:, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_region = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, 1])

    # --- Raster (stacked, subsampled per pop)
    y0 = 0
    yticks = []
    yticklabels = []
    separators = []

    for p in pops:
        t_s, i = spikes[p]
        N = int(pop_sizes[p])
        if N <= 0:
            continue

        # subsample neuron IDs for raster readability
        if N > max_raster_neurons_per_pop:
            keep = rng.choice(N, size=max_raster_neurons_per_pop, replace=False)
            keep = np.sort(keep)
            # map original i -> compact indices (only spikes from kept neurons)
            mask = np.isin(i, keep)
            t_plot = t_s[mask]
            i_plot = i[mask]
            # remap to 0..len(keep)-1
            remap = {int(k): j for j, k in enumerate(keep)}
            y = np.array([remap[int(ii)] for ii in i_plot], dtype=float) + y0
            height = max_raster_neurons_per_pop
        else:
            t_plot = t_s
            y = i.astype(float) + y0
            height = N

        ax_raster.plot(t_plot, y, linestyle="None", marker=".", markersize=1.0)

        mid = y0 + 0.5 * height
        yticks.append(mid)
        tag = "I" if _is_inhibitory(p) else "E"
        yticklabels.append(f"{p} [{tag}] (N={N})")

        y0 += height + 15  # gap between populations
        separators.append(y0 - 7.5)

    for s in separators[:-1]:
        ax_raster.axhline(s, linewidth=0.5)

    ax_raster.set_xlim(0, t_max_s)
    ax_raster.set_xlabel("time (s)")
    ax_raster.set_ylabel("stacked neuron index (subsampled per population)")
    ax_raster.set_yticks(yticks)
    ax_raster.set_yticklabels(yticklabels, fontsize=7)
    ax_raster.set_title("Stacked raster (populations grouped by region; subsampled neurons)")

    # --- Heatmap: pop x time
    im = ax_heat.imshow(
        H,
        aspect="auto",
        origin="lower",
        extent=[t_axis[0], t_axis[-1], 0, len(pops)],
    )
    ax_heat.set_title("Population rate heatmap (Hz per neuron)")
    ax_heat.set_xlabel("time (s)")
    ax_heat.set_ylabel("population index")
    ax_heat.set_yticks(np.arange(len(pops)) + 0.5)
    ax_heat.set_yticklabels(pops, fontsize=6)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # --- Region summed traces
    for r in regions:
        ax_region.plot(t_axis, region_traces[r], label=r)
    ax_region.set_xlim(0, t_max_s)
    ax_region.set_xlabel("time (s)")
    ax_region.set_ylabel("sum of pop rates (Hz/neuron, summed across pops)")
    ax_region.set_title("Region-summed activity (qualitative propagation / synchrony)")
    ax_region.legend(fontsize=8, ncol=3)

    # --- Mean rate bar plot
    xs = np.arange(len(pops))
    vals = np.array([mean_rate[p] for p in pops])
    ax_bar.bar(xs, vals)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(pops, rotation=90, fontsize=7)
    ax_bar.set_ylabel("mean rate (Hz)")
    ax_bar.set_title("Mean firing rate by population (sanity check)")

    # --- Optional: add a small inset with Vm & conductances if provided
    if state_traces:
        # pick one excitatory and one inhibitory population if available
        pick = None
        for p in pops:
            if not _is_inhibitory(p) and p in state_traces:
                pick = p
                break
        if pick is None:
            for p in pops:
                if p in state_traces:
                    pick = p
                    break

        if pick is not None:
            tr = state_traces[pick]
            # expected shapes: v_mV: (n_rec, T)
            t = tr.get("t_s", None)
            v = tr.get("v_mV", None)
            if t is not None and v is not None and v.ndim == 2 and v.shape[0] > 0:
                ax_in = ax_region.inset_axes([0.65, 0.55, 0.33, 0.4])
                ax_in.plot(t, v[0])
                ax_in.set_title(f"Example Vm: {pick}", fontsize=8)
                ax_in.set_xlabel("s", fontsize=7)
                ax_in.set_ylabel("mV", fontsize=7)
                ax_in.tick_params(labelsize=7)

    fig.suptitle("Hippocampal-formation network activity overview", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _simulate_stp(
    duration_s: float = 120.0,
    hfs_time_s: float = 10.0,
    stim_rate_hz_before: float = 0.5,
    stim_rate_hz_after: float = 5.0,
    rrp_scale: float = 2.5,
    pr_after: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an STP trajectory before and after a tetanus.

    The simulation uses 1-second steps for clarity: a brief HFS expands the
    readily releasable pool and increases release probability. Subsequent
    stimulation drives activity-dependent decay back toward baseline.
    """

    stp = STPMechanism()
    t = np.arange(0.0, duration_s + 1.0, 1.0)
    rrp: List[float] = []
    pr: List[float] = []

    for tt in t:
        if np.isclose(tt, hfs_time_s):
            stp.apply_hfs(rrp_scale=rrp_scale, pr_after=pr_after)

        stim_rate = stim_rate_hz_before if tt < hfs_time_s else stim_rate_hz_after
        stp.decay_with_activity(stim_rate)

        rrp.append(stp.rrp_current)
        pr.append(stp.pr)

    return t, np.asarray(rrp), np.asarray(pr)


def _btsp_probabilities(delta_t: np.ndarray, p_gate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Probability of LTP/LTD for binary BTSP weights across timing offsets."""

    rule = BTSPLearningRule(p_gate=p_gate, rng=np.random.default_rng(seed=1))
    p_ltp = np.zeros_like(delta_t)
    p_ltd = np.zeros_like(delta_t)

    for idx, dt in enumerate(delta_t):
        if -3.0 <= dt <= 2.0:
            # gate must pass and weight must be silent
            p_ltp[idx] = p_gate
        if (-6.0 <= dt < -3.0) or (2.0 < dt <= 4.0):
            p_ltd[idx] = p_gate

    return p_ltp, p_ltd


def _synaptic_tag_windows(config: TemporalScalingConfig) -> Dict[str, float]:
    p = config.plasticity_params
    return {
        "weak_tag": p.tag_lifetime_w_to_s_s / 60.0,
        "strong_tag": p.tag_lifetime_s_to_w_s / 60.0,
        "prp": p.prp_availability_s / 60.0,
    }


def _structural_trajectory(hours: int = 60, arc_onset_h: int = 12, active_every_h: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate survival/stabilization of a spine over several days."""

    from .plasticity import Spine  # local import to avoid cycles

    spine = Spine()
    status = []
    t = np.arange(0, hours + 1)
    pruned_at = None

    for h in t:
        if h >= arc_onset_h:
            spine.has_arc_expression = True
        outcome = spine.update(active_this_hour=(h % active_every_h == 0))
        if spine.is_stabilized:
            status.append(1)
        elif outcome == "PRUNE" and pruned_at is None:
            pruned_at = h
            status.append(-1)
        else:
            status.append(0)

    if pruned_at is not None:
        status = [s if idx <= pruned_at else -1 for idx, s in enumerate(status)]

    return t, np.asarray(status)


def save_plasticity_figure(
    out_path: Path,
    *,
    config: Optional[TemporalScalingConfig] = None,
    ca_range: Tuple[float, float] = (0.0, 2.0),
    p_gate: float = 0.5,
) -> None:
    """Create a multi-panel overview of plasticity primitives.

    Panels include:
    - STP timecourse (RRP and release probability before/after HFS)
    - Calcium-dependent bidirectional plasticity curve
    - BTSP timing-dependent probabilities for binary weights
    - Synaptic tagging/capture windows after temporal compression
    - Structural plasticity trajectory (Arc + activity dependent)
    - Temporal scaling comparison for core LTP stages
    """

    cfg = config or TemporalScalingConfig(target_sim_minutes=10.0, biological_max_hours=48.0)
    _ensure_parent(out_path)

    # --- Precompute traces
    t_stp, rrp, pr = _simulate_stp()
    ca = np.linspace(ca_range[0], ca_range[1], 400)
    plasticity_curve = np.array([bidirectional_plasticity(c) for c in ca])
    delta_t = np.linspace(-6.0, 4.0, 200)
    p_ltp, p_ltd = _btsp_probabilities(delta_t, p_gate)
    tag_windows = _synaptic_tag_windows(cfg)
    t_spine, spine_status = _structural_trajectory()

    p = cfg.plasticity_params
    durations = {
        "STP half-life": 35.0,  # minutes
        "LTP1": p.ltp1_duration_s / 60.0,
        "LTP2": p.ltp2_duration_s / 60.0,
        "LTP3 onset": p.ltp3_onset_s / 60.0,
    }
    original = {
        "STP half-life": 35.0,
        "LTP1": 120.0,
        "LTP2": 360.0,
        "LTP3 onset": 480.0,
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # STP
    ax = axes[0, 0]
    ax.plot(t_stp, rrp, label="RRP size")
    ax.plot(t_stp, pr, label="Release probability")
    ax.axvline(10.0, color="k", linestyle="--", alpha=0.6, label="HFS")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative value")
    ax.set_title("STP induction and use-dependent decay")
    ax.legend()

    # Calcium plasticity
    ax = axes[0, 1]
    ax.plot(ca, plasticity_curve, color="purple")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("[Ca] (µM)")
    ax.set_ylabel("Δ weight (arb)")
    ax.set_title("Bidirectional plasticity (sigmoid thresholds)")

    # BTSP
    ax = axes[1, 0]
    ax.plot(delta_t, p_ltp, label="P(LTP | weight=0)")
    ax.plot(delta_t, p_ltd, label="P(LTD | weight=1)")
    ax.axvspan(-6, -3, color="tab:orange", alpha=0.1)
    ax.axvspan(-3, 2, color="tab:green", alpha=0.1)
    ax.axvspan(2, 4, color="tab:orange", alpha=0.1)
    ax.set_xlabel("Δt = t_synapse - t_plateau (s)")
    ax.set_ylabel("probability")
    ax.set_title("BTSP stochasticity and temporal window")
    ax.legend()

    # Tagging windows
    ax = axes[1, 1]
    ax.barh(["Weak→Strong tag", "Strong→Weak PRP"], [tag_windows["weak_tag"], tag_windows["strong_tag"]], color=["tab:blue", "tab:red"])
    ax.barh(["PRP availability"], [tag_windows["prp"]], color="tab:gray")
    ax.set_xlabel("duration (minutes, compressed)")
    ax.set_title("Synaptic tagging & capture windows (scaled)")

    # Structural plasticity
    ax = axes[2, 0]
    ax.step(t_spine, spine_status, where="post")
    ax.set_xlabel("hours since spine formation")
    ax.set_ylabel("state (-1=pruned, 0=labile, 1=stabilized)")
    ax.set_title("Spine stabilization with Arc + activity")
    ax.set_ylim(-1.2, 1.2)

    # Temporal scaling comparison
    ax = axes[2, 1]
    y = np.arange(len(durations))
    width = 0.35
    ax.barh(y - width / 2, list(original.values()), height=width, label="biological (min)", color="tab:gray", alpha=0.5)
    ax.barh(y + width / 2, list(durations.values()), height=width, label="scaled (min)", color="tab:green")
    ax.set_yticks(y)
    ax.set_yticklabels(list(durations.keys()))
    ax.set_xlabel("duration (minutes)")
    ax.set_title(f"Temporal compression (CF={cfg.compression_factor:.0f}x)")
    ax.legend()

    fig.suptitle("Plasticity overview: synaptic, structural, and temporal scaling", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
