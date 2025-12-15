
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pyvista as pv

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
