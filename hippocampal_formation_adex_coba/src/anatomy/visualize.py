from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pyvista as pv

# Allow plotting empty meshes (required for PyVista >= 0.43)
try:
    pv.global_theme.allow_empty_mesh = True
except AttributeError:
    pass

from .regions import RegionGeometry


@dataclass
class PopulationViz:
    name: str
    xyz_mm: np.ndarray  # (N,3) in mm, atlas XYZ frame
    color: Tuple[int, int, int]
    point_size: float = 3.0


def _mask_to_mesh_mm(
    mask_zyx: np.ndarray,
    *,
    res_um_xyz: np.ndarray,
    downsample: int = 2,
    iso: float = 0.5,
) -> pv.PolyData:
    """
    Build an isosurface mesh from a boolean mask in (z,y,x), returned in mm coordinates.
    """
    if downsample < 1:
        downsample = 1

    # downsample in z,y,x
    m = mask_zyx[::downsample, ::downsample, ::downsample].astype(np.uint8)

    # convert to (x,y,z) volume for ImageData: (z,y,x) -> (x,y,z)
    vol_xyz = np.transpose(m, (2, 1, 0))

    # spacing in mm for (x,y,z)
    spacing_mm = (res_um_xyz / 1000.0) * float(downsample)

    grid = pv.ImageData(
        dimensions=vol_xyz.shape,
        spacing=(float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])),
        origin=(0.0, 0.0, 0.0),
    )

    # IMPORTANT: use Fortran order for pyvista/vtk point data layout
    grid.point_data["mask"] = vol_xyz.ravel(order="F")

    surf = grid.contour(isosurfaces=[iso], scalars="mask")
    return surf


def render_scene_from_geometries(
    region_geoms: Dict[str, RegionGeometry],
    populations: Iterable[PopulationViz],
    out_html: Path,
    *,
    mesh_opacity: float = 0.12,
    mesh_downsample: int = 2,
    offscreen: bool = True,
) -> None:
    """
    Render region meshes derived from **the same masks used for placement**, ensuring
    correct spatial organization in a single consistent frame.
    """
    pl = pv.Plotter(off_screen=offscreen)
    pl.background_color = "white"

    # draw meshes
    for region_name, geom in region_geoms.items():
        try:
            mesh = _mask_to_mesh_mm(
                geom.mask_zyx,
                res_um_xyz=geom.res_um_xyz,
                downsample=mesh_downsample,
            )
            pl.add_mesh(mesh, opacity=mesh_opacity, name=region_name)
        except Exception:
            continue

    # draw points
    for pop in populations:
        pts = pv.PolyData(np.asarray(pop.xyz_mm, dtype=float))
        pl.add_mesh(
            pts,
            render_points_as_spheres=True,
            point_size=float(pop.point_size),
            color=pop.color,
            name=pop.name,
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pl.export_html(str(out_html))
    pl.close()


@dataclass
class ActivityVideoSettings:
    fps: int = 30
    window_ms: float = 20.0
    max_neurons_per_pop: int = 2000
    mesh_opacity: float = 0.08
    mesh_downsample: int = 2
    base_opacity: float = 0.15
    active_point_size: float = 7.0
    background_color: str = "white"
    show_meshes: bool = True


def _brighten_color(color: Tuple[int, int, int], factor: float = 1.25) -> Tuple[int, int, int]:
    return tuple(int(min(255, max(0, c * factor))) for c in color)


def _pick_visual_subset(xyz_mm: np.ndarray, max_neurons: int, rng: np.random.Generator) -> np.ndarray:
    n = int(xyz_mm.shape[0])
    if max_neurons <= 0 or n <= max_neurons:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=max_neurons, replace=False))


def render_activity_video_from_geometries(
    region_geoms: Dict[str, RegionGeometry],
    populations: Iterable[PopulationViz],
    spikes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_mp4: Path,
    *,
    settings: Optional[ActivityVideoSettings] = None,
    t_stop_s: Optional[float] = None,
    offscreen: bool = True,
) -> None:
    settings = settings or ActivityVideoSettings()

    rng = np.random.default_rng(0)
    pop_entries = []

    for pop in populations:
        if pop.name not in spikes:
            continue
        t_s, i = spikes[pop.name]
        t_s = np.asarray(t_s, dtype=float)
        i = np.asarray(i, dtype=int)
        if t_s.size == 0:
            continue
        if t_s.size > 1 and np.any(np.diff(t_s) < 0):
            order = np.argsort(t_s)
            t_s = t_s[order]
            i = i[order]
        keep_indices = _pick_visual_subset(pop.xyz_mm, settings.max_neurons_per_pop, rng)
        index_map = np.full(pop.xyz_mm.shape[0], -1, dtype=int)
        index_map[keep_indices] = np.arange(keep_indices.size, dtype=int)
        keep_xyz = np.asarray(pop.xyz_mm[keep_indices], dtype=float)
        pop_entries.append(
            {
                "name": pop.name,
                "color": pop.color,
                "active_color": _brighten_color(pop.color, factor=1.35),
                "point_size": float(pop.point_size),
                "keep_xyz": keep_xyz,
                "index_map": index_map,
                "t_s": t_s,
                "i": i,
            }
        )

    if not pop_entries:
        return

    if t_stop_s is None:
        t_stop_s = max(float(entry["t_s"].max(initial=0.0)) for entry in pop_entries)

    dt_s = 1.0 / float(max(1, settings.fps))
    window_s = float(settings.window_ms) / 1000.0
    n_frames = max(1, int(np.ceil(t_stop_s / dt_s)))

    pl = pv.Plotter(off_screen=offscreen)
    pl.background_color = settings.background_color

    if settings.show_meshes:
        for region_name, geom in region_geoms.items():
            try:
                mesh = _mask_to_mesh_mm(
                    geom.mask_zyx,
                    res_um_xyz=geom.res_um_xyz,
                    downsample=settings.mesh_downsample,
                )
                pl.add_mesh(mesh, opacity=settings.mesh_opacity, name=f"mesh_{region_name}")
            except Exception:
                continue

    base_actors = {}
    active_polydata = {}
    for entry in pop_entries:
        base = pv.PolyData(entry["keep_xyz"])
        base_actor = pl.add_mesh(
            base,
            render_points_as_spheres=True,
            point_size=entry["point_size"],
            color=entry["color"],
            opacity=settings.base_opacity,
            name=f"base_{entry['name']}",
        )
        base_actors[entry["name"]] = base_actor

        active_poly = pv.PolyData(np.empty((0, 3), dtype=float))
        pl.add_mesh(
            active_poly,
            render_points_as_spheres=True,
            point_size=settings.active_point_size,
            color=entry["active_color"],
            opacity=1.0,
            name=f"active_{entry['name']}",
        )
        active_polydata[entry["name"]] = active_poly

    time_text = pl.add_text("t = 0.000 s", position="upper_left", font_size=12)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pl.open_movie(str(out_mp4), framerate=int(settings.fps))

    for frame_idx in range(n_frames):
        t0 = frame_idx * dt_s
        t1 = t0 + window_s
        for entry in pop_entries:
            t_s = entry["t_s"]
            i = entry["i"]
            idx0 = int(np.searchsorted(t_s, t0, side="left"))
            idx1 = int(np.searchsorted(t_s, t1, side="right"))
            if idx1 <= idx0:
                active_polydata[entry["name"]].points = np.empty((0, 3), dtype=float)
                continue

            spike_ids = i[idx0:idx1]
            mapped = entry["index_map"][spike_ids]
            mapped = mapped[mapped >= 0]
            if mapped.size == 0:
                active_polydata[entry["name"]].points = np.empty((0, 3), dtype=float)
                continue

            mapped = np.unique(mapped)
            active_polydata[entry["name"]].points = entry["keep_xyz"][mapped]

        time_text.SetText(0, f"t = {t0:0.3f} s")
        pl.render()
        pl.write_frame()

    pl.close()
