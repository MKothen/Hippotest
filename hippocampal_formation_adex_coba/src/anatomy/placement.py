from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np

from ..data.ccf_atlas import CCFAtlas
from .regions import RegionGeometry


PlacementMode = Literal["volume", "layer_band", "layer_weighted"]


@dataclass(frozen=True)
class LayerDef:
    name: str
    u_min: float
    u_max: float

    @property
    def u_mid(self) -> float:
        return 0.5 * (self.u_min + self.u_max)

    @property
    def u_width(self) -> float:
        return max(1e-6, self.u_max - self.u_min)


def layer_defs_from_config(layer_map: Dict[str, List[float]]) -> Dict[str, LayerDef]:
    out: Dict[str, LayerDef] = {}
    for lname, bounds in layer_map.items():
        if len(bounds) != 2:
            raise ValueError(f"Layer bounds must be [u_min,u_max] for {lname}")
        out[lname] = LayerDef(name=lname, u_min=float(bounds[0]), u_max=float(bounds[1]))
    return out


# -------------------------
# Geometry compatibility
# -------------------------

def _get_region_vox_zyx_and_xyz_mm(
    atlas: CCFAtlas,
    geom: RegionGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      vox_zyx: (N,3) integer array (z,y,x)
      xyz_mm: (N,3) float array (x,y,z) in mm for each voxel center
      res_um_xyz: (3,) float array (x,y,z) µm/voxel
    Works across RegionGeometry schema variants.
    """
    # voxel points in (z,y,x)
    if hasattr(geom, "voxel_points_zyx"):
        vox_zyx = getattr(geom, "voxel_points_zyx")
    elif hasattr(geom, "voxel_points"):
        vox_zyx = getattr(geom, "voxel_points")
    else:
        raise AttributeError(
            "RegionGeometry has no voxel point field. Expected voxel_points_zyx or voxel_points."
        )

    # voxel resolution (x,y,z) in µm
    if hasattr(geom, "res_um_xyz"):
        res_um_xyz = np.asarray(getattr(geom, "res_um_xyz"), dtype=float).reshape(3)
    else:
        # fall back to atlas metadata
        res_um_xyz = np.asarray(getattr(atlas, "resolution_um", (25.0, 25.0, 25.0)), dtype=float).reshape(3)

    # voxel centers in mm for each voxel
    if hasattr(geom, "xyz_mm"):
        xyz_mm = np.asarray(getattr(geom, "xyz_mm"), dtype=float)
        if xyz_mm.shape[0] != vox_zyx.shape[0]:
            # compute from vox if inconsistent
            xyz_mm = atlas.vox_to_mm(vox_zyx)
    else:
        xyz_mm = atlas.vox_to_mm(vox_zyx)

    return np.asarray(vox_zyx), np.asarray(xyz_mm), res_um_xyz


def _jitter_mm(
    xyz_mm: np.ndarray,
    res_um_xyz: np.ndarray,
    rng: np.random.Generator,
    jitter_within_voxel: bool = True,
) -> np.ndarray:
    """
    Add continuous uniform jitter inside the voxel so somata aren't stuck at centers.
    """
    if not jitter_within_voxel:
        return xyz_mm
    res_mm = np.asarray(res_um_xyz, dtype=float) / 1000.0
    j = rng.uniform(-0.5, 0.5, size=xyz_mm.shape) * res_mm[None, :]
    return xyz_mm + j


# -------------------------
# Public sampler
# -------------------------

def sample_somata(
    atlas: CCFAtlas,
    geom: RegionGeometry,
    layer: LayerDef,
    n: int,
    rng: np.random.Generator,
    *,
    mode: PlacementMode = "layer_weighted",
    layer_sigma_u: Optional[float] = None,
    restrict_to_layer_bounds: bool = False,
    jitter_within_voxel: bool = True,
) -> np.ndarray:
    """
    Sample soma locations in mm.

    Modes:
      - "volume": uniform over the entire region volume.
      - "layer_band": uniform over voxels whose laminar_u lies in [u_min, u_max).
      - "layer_weighted": sample over region volume with probability biased toward layer.u_mid
                          (3D fill + laminar preference).

    Args:
      layer_sigma_u:
        For "layer_weighted": Gaussian sigma in u-units.
        Larger => more volumetric/uniform; smaller => more sheet-like.
        Default: 0.5*layer width.
      restrict_to_layer_bounds:
        For "layer_weighted": if True, only considers voxels within [u_min,u_max).
        If False (recommended), considers all voxels but weights them.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    vox_zyx, xyz_mm_all, res_um_xyz = _get_region_vox_zyx_and_xyz_mm(atlas, geom)
    u = np.asarray(geom.laminar_u, dtype=float)

    if u.shape[0] != vox_zyx.shape[0]:
        raise ValueError(
            f"geom.laminar_u has length {u.shape[0]} but voxel points have length {vox_zyx.shape[0]}"
        )

    Nvox = vox_zyx.shape[0]

    if mode == "volume":
        idx = np.arange(Nvox, dtype=int)
        chosen = rng.choice(idx, size=n, replace=(n > idx.size))
        xyz_mm = xyz_mm_all[chosen]

    elif mode == "layer_band":
        idx = np.where((u >= layer.u_min) & (u < layer.u_max))[0]
        if idx.size == 0:
            raise ValueError(
                f"No voxels in layer {layer.name} of {geom.name} for bounds [{layer.u_min},{layer.u_max})"
            )
        chosen = rng.choice(idx, size=n, replace=(n > idx.size))
        xyz_mm = xyz_mm_all[chosen]

    elif mode == "layer_weighted":
        if layer_sigma_u is None:
            layer_sigma_u = 0.5 * layer.u_width
        sigma = float(layer_sigma_u)
        if sigma <= 0:
            raise ValueError("layer_sigma_u must be > 0")

        if restrict_to_layer_bounds:
            idx = np.where((u >= layer.u_min) & (u < layer.u_max))[0]
        else:
            idx = np.arange(Nvox, dtype=int)

        if idx.size == 0:
            raise ValueError(f"No candidate voxels for placement in {geom.name} (mode={mode}).")

        uu = u[idx]
        mu = layer.u_mid

        w = np.exp(-0.5 * ((uu - mu) / sigma) ** 2) + 1e-12
        w = w / w.sum()

        replace = n > idx.size
        chosen_local = rng.choice(np.arange(idx.size, dtype=int), size=n, replace=replace, p=w)
        chosen = idx[chosen_local]
        xyz_mm = xyz_mm_all[chosen]

    else:
        raise ValueError(f"Unknown placement mode: {mode}")

    xyz_mm = _jitter_mm(xyz_mm, res_um_xyz, rng=rng, jitter_within_voxel=jitter_within_voxel)
    return xyz_mm
