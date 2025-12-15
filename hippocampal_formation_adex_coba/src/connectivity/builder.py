
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .pathways import PathwaySpec


@dataclass
class PopulationGeometry:
    name: str
    xyz_mm: np.ndarray  # (N,3)
    soma_layer: np.ndarray  # (N,) strings
    region: str
    cell_class: str  # excitatory/inhibitory
    cell_type: str   # library key


@dataclass
class PathwayEdges:
    spec: PathwaySpec
    pre_idx: np.ndarray  # (E,)
    post_idx: np.ndarray  # (E,)
    dist_mm: np.ndarray  # (E,)
    delay_ms: np.ndarray  # (E,)
    # weights in nS
    w_ampa_nS: np.ndarray
    w_nmda_nS: np.ndarray
    w_gabaa_nS: np.ndarray
    w_gabab_nS: np.ndarray
    target_layer: np.ndarray  # (E,) object dtype (strings)


def _sample_k(rng: np.random.Generator, mean: float, std: float, n: int) -> np.ndarray:
    k = rng.normal(loc=mean, scale=std, size=n)
    k = np.clip(np.round(k), 0, None).astype(int)
    return k


def build_edges_for_pathway(
    spec: PathwaySpec,
    pre: PopulationGeometry,
    post: PopulationGeometry,
    rng: np.random.Generator,
    scale: float = 1.0,
    show_progress: bool = True,
) -> PathwayEdges:
    """
    Builds synapses as a list of edges from `pre` to `post` using:
      - a radius cutoff
      - a Gaussian distance kernel exp(-d^2/(2 sigma^2))
      - per-pre-neuron target outdegree K ~ N(mean,std)
      - target-layer distribution (stored per edge)
      - distance-based conduction delays
    """
    if pre.xyz_mm.ndim != 2 or pre.xyz_mm.shape[1] != 3:
        raise ValueError("pre.xyz_mm must be (N,3)")
    if post.xyz_mm.ndim != 2 or post.xyz_mm.shape[1] != 3:
        raise ValueError("post.xyz_mm must be (M,3)")

    pre_xyz = pre.xyz_mm
    post_xyz = post.xyz_mm

    tree = cKDTree(post_xyz)

    # Sample outdegree per presyn neuron
    k_out = _sample_k(rng, spec.k_out_mean * scale, spec.k_out_std * scale, pre_xyz.shape[0])

    # Target layer distribution
    layers = list(spec.target_layers.keys()) if spec.target_layers else ["DEFAULT"]
    layer_w = np.array([spec.target_layers.get(l, 1.0) for l in layers], dtype=float)
    layer_w = layer_w / (layer_w.sum() + 1e-12)

    pre_idx_list: List[int] = []
    post_idx_list: List[int] = []
    dist_list: List[float] = []
    delay_list: List[float] = []
    tl_list: List[str] = []

    sigma = float(spec.sigma_mm)
    radius = float(spec.radius_mm)

    iterator = range(pre_xyz.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc=f"Pathway {spec.name}", leave=False)

    for i in iterator:
        ki = int(k_out[i])
        if ki <= 0:
            continue

        # Candidate posts within radius
        cand = tree.query_ball_point(pre_xyz[i], r=radius)
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)

        # Optionally exclude self (only meaningful if same population)
        if spec.exclude_self and pre is post:
            cand = cand[cand != i]
            if cand.size == 0:
                continue

        # Distances
        d = np.linalg.norm(post_xyz[cand] - pre_xyz[i], axis=1)
        # Distance kernel
        p = np.exp(-(d ** 2) / (2.0 * sigma ** 2 + 1e-12))
        p_sum = p.sum()
        if p_sum <= 0:
            continue
        p = p / p_sum

        # sample targets (with replacement if ki > cand)
        replace = ki > cand.size
        chosen = rng.choice(cand, size=ki, replace=replace, p=p)
        # compute distances for chosen
        dch = np.linalg.norm(post_xyz[chosen] - pre_xyz[i], axis=1)

        # delays
        # distance (mm) -> m: *1e-3; delay (s)=dist/v + base
        delay_s = (dch * 1e-3) / float(spec.velocity_m_per_s) + float(spec.base_delay_ms) * 1e-3
        delay_ms = delay_s * 1e3

        # target layers per edge
        tlay = rng.choice(layers, size=ki, replace=True, p=layer_w)

        pre_idx_list.extend([i] * ki)
        post_idx_list.extend(chosen.tolist())
        dist_list.extend(dch.tolist())
        delay_list.extend(delay_ms.tolist())
        tl_list.extend(tlay.tolist())

    pre_idx_arr = np.asarray(pre_idx_list, dtype=int)
    post_idx_arr = np.asarray(post_idx_list, dtype=int)
    dist_arr = np.asarray(dist_list, dtype=float)
    delay_arr = np.asarray(delay_list, dtype=float)
    tl_arr = np.asarray(tl_list, dtype=object)

    # weights (nS): base weights scaled by optional per-layer factor (if you want to model layer-dependent strength)
    # Here we keep a single scalar scale knob in config; per-edge scaling can be added later.
    w_ampa = np.full_like(dist_arr, float(spec.synapse.w_ampa_nS), dtype=float)
    w_nmda = np.full_like(dist_arr, float(spec.synapse.w_nmda_nS), dtype=float)
    w_gabaa = np.full_like(dist_arr, float(spec.synapse.w_gabaa_nS), dtype=float)
    w_gabab = np.full_like(dist_arr, float(spec.synapse.w_gabab_nS), dtype=float)

    return PathwayEdges(
        spec=spec,
        pre_idx=pre_idx_arr,
        post_idx=post_idx_arr,
        dist_mm=dist_arr,
        delay_ms=delay_arr,
        w_ampa_nS=w_ampa,
        w_nmda_nS=w_nmda,
        w_gabaa_nS=w_gabaa,
        w_gabab_nS=w_gabab,
        target_layer=tl_arr,
    )


def build_connectivity(
    pathways: List[PathwaySpec],
    pops: Dict[str, PopulationGeometry],
    rng: np.random.Generator,
    scale: float = 1.0,
    show_progress: bool = True,
) -> Dict[str, PathwayEdges]:
    edges: Dict[str, PathwayEdges] = {}
    for spec in pathways:
        if spec.pre_pop not in pops or spec.post_pop not in pops:
            raise KeyError(f"Unknown population in pathway {spec.name}: {spec.pre_pop}->{spec.post_pop}")
        pre = pops[spec.pre_pop]
        post = pops[spec.post_pop]
        edges[spec.name] = build_edges_for_pathway(spec, pre, post, rng, scale=scale, show_progress=show_progress)
    return edges
