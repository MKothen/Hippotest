from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..data.ccf_atlas import CCFAtlas
from ..celltypes.library import build_celltype_library, CellTypeSpec
from ..anatomy.regions import load_region_geometry, RegionGeometry
from ..anatomy.placement import layer_defs_from_config, LayerDef, sample_somata
from ..anatomy.visualize import (
    render_scene_from_geometries,
    render_activity_video_from_geometries,
    ActivityVideoSettings,
    PopulationViz,
)
from ..connectivity.pathways import load_pathways
from ..connectivity.builder import PopulationGeometry, build_connectivity
from ..connectivity.metrics import pathway_stats
from ..sim.runner import run_simulation
from ..sim.io import save_json, save_npz
from ..utils.seed import set_global_seeds


def build_and_run(config: Dict[str, Any], out_dir: Path) -> None:
    seed = int(config.get("seed", 1))
    set_global_seeds(seed)
    rng = np.random.default_rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Load atlas + regions (mask-derived geometry)
    # -----------------------
    anat = config["anatomy"]
    atlas = CCFAtlas(atlas_name=str(anat.get("atlas_name", "allen_mouse_25um")))
    hemisphere = str(anat.get("hemisphere", "both"))

    region_cfg: Dict[str, Any] = anat["regions"]
    region_geoms: Dict[str, RegionGeometry] = {}

    for region_key, rcfg in region_cfg.items():
        sname = str(rcfg["structure_name"])
        geom = load_region_geometry(atlas, sname, hemisphere=hemisphere)
        region_geoms[region_key] = geom

    # -----------------------
    # Celltype library
    # -----------------------
    cell_lib = build_celltype_library(config.get("hippocampome", {}))

    # -----------------------
    # Place neurons (sample directly in mm from region mask bands)
    # -----------------------
    pops_cfg: Dict[str, Any] = config["populations"]
    pop_geoms: Dict[str, PopulationGeometry] = {}
    pop_specs: Dict[str, CellTypeSpec] = {}

    # Precompute layer defs per region
    layer_defs: Dict[str, Dict[str, LayerDef]] = {}
    for region_key, rcfg in region_cfg.items():
        layer_defs[region_key] = layer_defs_from_config(rcfg["layers"])

    positions_npz: Dict[str, Any] = {}
    viz_pops: List[PopulationViz] = []

    for pop_name, pcfg in pops_cfg.items():
        region_key = str(pcfg["region"])
        layer_name = str(pcfg["soma_layer"])
        cell_type = str(pcfg["cell_type"])
        n = int(pcfg["n"])

        if region_key not in region_geoms:
            raise KeyError(f"Population {pop_name} references unknown region {region_key}")
        if cell_type not in cell_lib:
            raise KeyError(f"Unknown cell_type {cell_type} for population {pop_name}")

        geom = region_geoms[region_key]
        layer = layer_defs[region_key][layer_name]

        # IMPORTANT: this returns xyz_mm now (not vox)
        placement_cfg = pcfg.get("placement", {}) or {}
        xyz_mm = sample_somata(
            atlas,
            geom,
            layer,
            n=n,
            rng=rng,
            mode=str(placement_cfg.get("mode", "layer_weighted")),
            layer_sigma_u=placement_cfg.get("layer_sigma_u", None),
            restrict_to_layer_bounds=bool(placement_cfg.get("restrict_to_layer_bounds", False)),
            jitter_within_voxel=bool(placement_cfg.get("jitter_within_voxel", True)),
        )
        soma_layer = np.array([layer_name] * n, dtype=object)

        spec = cell_lib[cell_type]
        pop_specs[pop_name] = spec

        pop_geoms[pop_name] = PopulationGeometry(
            name=pop_name,
            xyz_mm=xyz_mm,
            soma_layer=soma_layer,
            region=region_key,
            cell_class=spec.cls,
            cell_type=cell_type,
        )

        positions_npz[f"{pop_name}_xyz_mm"] = xyz_mm
        positions_npz[f"{pop_name}_soma_layer"] = soma_layer

        # viz (color by class)
        color = (0, 120, 200) if spec.cls == "excitatory" else (200, 80, 80)
        viz_pops.append(
            PopulationViz(
                name=pop_name,
                xyz_mm=xyz_mm,
                color=color,
                point_size=float(pcfg.get("viz_point_size", 3.0)),
            )
        )

    save_npz(positions_npz, out_dir / "data" / "positions.npz")

    # -----------------------
    # Connectivity
    # -----------------------
    conn = config["connectivity"]
    pathways = load_pathways(conn["pathways_file"])
    scale = float(conn.get("scale", 1.0))

    edges = build_connectivity(pathways, pop_geoms, rng=rng, scale=scale, show_progress=True)

    # Sanity checks + save
    conn_stats: Dict[str, Any] = {}
    for pname, e in edges.items():
        st = pathway_stats(
            e,
            n_pre=pop_geoms[e.spec.pre_pop].xyz_mm.shape[0],
            n_post=pop_geoms[e.spec.post_pop].xyz_mm.shape[0],
        )
        conn_stats[pname] = st

    save_json(conn_stats, out_dir / "connectivity_summary.json")

    print("\nConnectivity summary (per pathway):")
    for pname, st in conn_stats.items():
        print(
            f"- {pname}: edges={st['n_edges']}, fanout_mean={st['fanout_mean']:.2f}, "
            f"fanin_mean={st['fanin_mean']:.2f}, dist_mean_mm={st['dist_mm_mean']}, "
            f"delay_mean_ms={st['delay_ms_mean']}"
        )

    # -----------------------
    # 3D Visualization (mask-derived meshes + mask-derived points; same frame)
    # -----------------------
    try:
        render_scene_from_geometries(
            region_geoms=region_geoms,
            populations=viz_pops,
            out_html=out_dir / "viz" / "scene.html",
            mesh_downsample=int(anat.get("viz_mesh_downsample", 2)),
            mesh_opacity=float(anat.get("viz_mesh_opacity", 0.12)),
            offscreen=True,
        )
        print(f"3D scene exported: {out_dir/'viz'/'scene.html'}")
    except Exception as e:
        print(f"[warn] 3D rendering skipped ({e})")

    # -----------------------
    # Simulation
    # -----------------------
    sim_out = run_simulation(config, pop_specs=cell_lib, pop_geom=pop_geoms, edges=edges, out_dir=out_dir)
    print("\nPhysiology summary (mean rates, Hz):")
    for pop_name, meta in sim_out.summary["populations"].items():
        print(
            f"  {pop_name:20s}  n={meta['n']:6d}  mean_rate={meta['mean_rate_hz']:.3f}  spikes={meta['n_spikes']}"
        )

    # -----------------------
    # Activity video
    # -----------------------
    sim_cfg = config.get("simulation", {}) or {}
    viz_cfg = sim_cfg.get("viz", {}) or {}
    activity_cfg = viz_cfg.get("activity_video", {}) or {}
    if bool(activity_cfg.get("enabled", False)):
        settings = ActivityVideoSettings(
            fps=int(activity_cfg.get("fps", 30)),
            window_ms=float(activity_cfg.get("window_ms", 20.0)),
            max_neurons_per_pop=int(activity_cfg.get("max_neurons_per_pop", 2000)),
            mesh_opacity=float(activity_cfg.get("mesh_opacity", 0.08)),
            mesh_downsample=int(activity_cfg.get("mesh_downsample", 2)),
            base_opacity=float(activity_cfg.get("base_opacity", 0.15)),
            active_point_size=float(activity_cfg.get("active_point_size", 7.0)),
            background_color=str(activity_cfg.get("background_color", "white")),
            show_meshes=bool(activity_cfg.get("show_meshes", True)),
        )
        try:
            render_activity_video_from_geometries(
                region_geoms=region_geoms,
                populations=viz_pops,
                spikes=sim_out.spikes,
                out_mp4=out_dir / "viz" / "activity_video.mp4",
                settings=settings,
                t_stop_s=float(sim_cfg.get("t_sim_s", 1.0)),
                offscreen=True,
            )
            print(f"3D activity video exported: {out_dir/'viz'/'activity_video.mp4'}")
        except Exception as e:
            print(f"[warn] 3D activity video skipped ({e})")
