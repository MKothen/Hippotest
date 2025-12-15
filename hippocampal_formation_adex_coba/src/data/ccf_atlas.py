
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from brainglobe_atlasapi import BrainGlobeAtlas
except Exception as e:  # pragma: no cover
    BrainGlobeAtlas = None  # type: ignore


@dataclass(frozen=True)
class StructureInfo:
    acronym: str
    name: str
    id: int
    rgb_triplet: Tuple[int, int, int]
    mesh_filename: Optional[Path]
    structure_id_path: List[int]


class CCFAtlas:
    """
    Thin wrapper around BrainGlobeAtlas for accessing Allen CCF-like atlas data.

    We rely on BrainGlobe to:
      - download + cache atlas volumes and meshes
      - provide a consistent `annotation` 3D volume (region ids) and mesh file paths

    Atlas space coordinates are in voxel indices; convert using `resolution_um`.
    """

    def __init__(self, atlas_name: str = "allen_mouse_25um"):
        if BrainGlobeAtlas is None:  # pragma: no cover
            raise ImportError(
                "brainglobe-atlasapi is required. Install with `pip install brainglobe-atlasapi`."
            )
        self.atlas = BrainGlobeAtlas(atlas_name)
        self.atlas_name = atlas_name

        # BrainGlobe uses (AP, DV, ML) convention for axis ordering in volumes (z,y,x-like).
        # We'll keep the raw order and consistently treat coordinates as (i,j,k) indices.
        self.annotation = np.asarray(self.atlas.annotation, dtype=np.int32)
        self.reference = np.asarray(self.atlas.reference)

        # Resolution in micrometers for each voxel axis
        self.resolution_um = tuple(float(x) for x in self.atlas.resolution)  # (um, um, um)

        # Structures: dict keyed by acronym
        self.structures: Dict[str, StructureInfo] = {}
        for acr, s in self.atlas.structures.items():
            self.structures[acr] = StructureInfo(
                acronym=s.get("acronym", acr),
                name=s.get("name", ""),
                id=int(s.get("id")),
                rgb_triplet=tuple(s.get("rgb_triplet", (200, 200, 200))),
                mesh_filename=s.get("mesh_filename", None),
                structure_id_path=list(s.get("structure_id_path", [])),
            )

        # For name lookup
        self._name_to_acronym = {info.name.lower(): acr for acr, info in self.structures.items()}

    def find_structure(self, *, name: Optional[str] = None, acronym: Optional[str] = None) -> StructureInfo:
        if acronym is not None:
            if acronym not in self.structures:
                raise KeyError(f"Unknown structure acronym: {acronym}")
            return self.structures[acronym]
        if name is not None:
            acr = self._name_to_acronym.get(name.lower())
            if acr is None:
                # fallback: substring match
                candidates = [info for info in self.structures.values() if name.lower() in info.name.lower()]
                if len(candidates) == 0:
                    raise KeyError(f"Could not find structure with name containing: {name}")
                if len(candidates) > 1:
                    # choose shortest name match
                    candidates.sort(key=lambda x: len(x.name))
                return candidates[0]
            return self.structures[acr]
        raise ValueError("Provide name or acronym.")

    def mask_for_id(self, structure_id: int) -> np.ndarray:
        return self.annotation == int(structure_id)

    def mask_for_structure(self, *, name: Optional[str] = None, acronym: Optional[str] = None) -> np.ndarray:
        info = self.find_structure(name=name, acronym=acronym)
        return self.mask_for_id(info.id)

    def mesh_path_for_structure(self, *, name: Optional[str] = None, acronym: Optional[str] = None) -> Optional[Path]:
        info = self.find_structure(name=name, acronym=acronym)
        return info.mesh_filename

    def vox_to_um(self, ijk: np.ndarray) -> np.ndarray:
        """Convert voxel indices (N,3) to micrometers in atlas space."""
        res = np.array(self.resolution_um, dtype=float)[None, :]
        return ijk.astype(float) * res

    def vox_to_mm(self, ijk: np.ndarray) -> np.ndarray:
        return self.vox_to_um(ijk) / 1000.0
