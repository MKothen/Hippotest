from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt


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
