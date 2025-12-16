from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from ..data.ccf_atlas import CCFAtlas
from .laminar import LaminarAxes, pca_axes, laminar_coordinate, longitudinal_coordinate

# -----------------------------
# Canonical region name mapping
# -----------------------------
_CANONICAL: Dict[str, List[str]] = {
    "Dentate gyrus": ["DG", "Dentate gyrus"],
    "DG": ["DG", "Dentate gyrus"],

    "CA1": ["CA1"],
    "CA2": ["CA2"],
    "CA3": ["CA3"],

    "Subiculum": ["SUB", "Subiculum"],
    "SUB": ["SUB", "Subiculum"],

    "Entorhinal cortex": ["ENTl", "ENTm", "ENT", "Entorhinal"],
    "EC": ["ENTl", "ENTm", "ENT", "Entorhinal"],
}

ALLEN_STRUCTURE_GRAPH_URL = "https://api.brain-map.org/api/v2/structure_graph_download/1.json"
_ALLEN_GRAPH_CACHE: Optional[Dict[int, List[int]]] = None


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _get_attr(obj: Any, *names: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        return default
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _iter_structures(atlas: Any):
    s = getattr(atlas, "structures", None)
    if s is None:
        return
    if isinstance(s, dict):
        for k, v in s.items():
            yield k, v
    elif isinstance(s, (list, tuple)):
        for i, v in enumerate(s):
            yield i, v


def _structure_id_from_entry(key: Any, entry: Any) -> Optional[int]:
    if isinstance(key, (int, np.integer)):
        return int(key)
    sid = _get_attr(entry, "id", "structure_id", default=None)
    if sid is None:
        return None
    try:
        return int(sid)
    except Exception:
        return None


def _resolve_one_token_to_id(atlas: Any, token: str) -> Optional[int]:
    if str(token).isdigit():
        try:
            return int(token)
        except Exception:
            return None

    t = _normalize(token)

    # direct dict-key lookup (if keys are acronyms)
    s = getattr(atlas, "structures", None)
    if isinstance(s, dict) and token in s:
        sid = _structure_id_from_entry(token, s[token])
        if sid is not None:
            return sid

    exact_hits: List[int] = []
    contains_hits: List[int] = []

    for k, v in _iter_structures(atlas):
        sid = _structure_id_from_entry(k, v)
        if sid is None:
            continue
        acronym = _get_attr(v, "acronym", "abbr", default=None)
        name = _get_attr(v, "name", default=None)
        acr = _normalize(acronym) if isinstance(acronym, str) else None
        nm = _normalize(name) if isinstance(name, str) else None

        if acr == t or nm == t:
            exact_hits.append(sid)
        elif (acr and t in acr) or (nm and t in nm):
            contains_hits.append(sid)

    if exact_hits:
        return exact_hits[0]
    if contains_hits:
        return sorted(contains_hits)[0]
    return None


def resolve_structure_ids(atlas: Any, structure_name: str) -> List[int]:
    tokens = _CANONICAL.get(structure_name)
    if tokens is None:
        s = _normalize(structure_name)
        for k, v in _CANONICAL.items():
            if _normalize(k) == s:
                tokens = v
                break
    if tokens is None:
        tokens = [structure_name]

    ids: List[int] = []
    for tok in tokens:
        sid = _resolve_one_token_to_id(atlas, tok)
        if sid is not None and sid not in ids:
            ids.append(sid)

    if not ids:
        raise KeyError(
            f"Could not resolve structure '{structure_name}' to a numeric Allen structure id."
        )
    return ids


def _default_cache_dir(atlas: Any) -> Path:
    for attr in ("cache_dir", "cache_path", "cache_root"):
        if hasattr(atlas, attr):
            try:
                return Path(getattr(atlas, attr))
            except Exception:
                pass
    return Path(".cache") / "allen"


def _load_allen_structure_graph(cache_dir: Path) -> Dict[int, List[int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / "allen_structure_graph_1.json"

    if not fp.exists():
        r = requests.get(ALLEN_STRUCTURE_GRAPH_URL, timeout=60)
        r.raise_for_status()
        fp.write_text(r.text, encoding="utf-8")

    import json
    data = json.loads(fp.read_text(encoding="utf-8"))
    root = data.get("msg", data)

    id_to_children: Dict[int, List[int]] = {}

    def walk(node: Dict[str, Any]):
        nid = int(node["id"])
        kids = node.get("children", []) or []
        id_to_children[nid] = [int(ch["id"]) for ch in kids]
        for ch in kids:
            walk(ch)

    if isinstance(root, list):
        for n in root:
            walk(n)
    else:
        walk(root)

    return id_to_children


def _ensure_allen_graph(atlas: Any) -> Dict[int, List[int]]:
    global _ALLEN_GRAPH_CACHE
    if _ALLEN_GRAPH_CACHE is None:
        _ALLEN_GRAPH_CACHE = _load_allen_structure_graph(_default_cache_dir(atlas))
    return _ALLEN_GRAPH_CACHE


def descendant_ids(atlas: Any, root_id: int) -> List[int]:
    graph = _ensure_allen_graph(atlas)

    out: List[int] = []
    stack = list(graph.get(int(root_id), []))
    seen = set(stack)
    while stack:
        nid = stack.pop()
        out.append(nid)
        for ch in graph.get(nid, []):
            if ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return out


def mask_for_structure_ids(annotation: np.ndarray, ids: List[int]) -> np.ndarray:
    max_id = int(annotation.max())
    lut = np.zeros(max_id + 1, dtype=bool)
    for sid in ids:
        if 0 <= sid <= max_id:
            lut[sid] = True
    return lut[annotation]


def atlas_resolution_um(atlas: Any) -> np.ndarray:
    """
    Return (x,y,z) voxel size in µm. Uses magnitude heuristics for mm-vs-µm.
    """
    for attr in ("resolution_um", "voxel_size_um", "voxel_resolution_um", "resolution", "voxel_size"):
        if hasattr(atlas, attr):
            try:
                r = np.asarray(getattr(atlas, attr), dtype=float).reshape(-1)
                if r.size == 1:
                    r = np.repeat(r, 3)
                if r.size != 3:
                    continue
                if np.max(r) < 1.0:
                    return r * 1000.0  # mm -> µm
                return r
            except Exception:
                pass
    return np.array([25.0, 25.0, 25.0], dtype=float)


def zyx_to_xyz_vox(zyx: np.ndarray) -> np.ndarray:
    # np.argwhere gives (z,y,x); convert to (x,y,z)
    return zyx[:, [2, 1, 0]]


def xyz_vox_to_mm(xyz_vox: np.ndarray, res_um_xyz: np.ndarray) -> np.ndarray:
    return xyz_vox * (res_um_xyz[None, :] / 1000.0)


@dataclass
class RegionGeometry:
    name: str
    structure_id: int

    mask_zyx: np.ndarray            # (z,y,x) bool
    voxel_points_zyx: np.ndarray    # (N,3) (z,y,x)

    xyz_vox: np.ndarray             # (N,3) (x,y,z) voxel indices
    xyz_mm: np.ndarray              # (N,3) (x,y,z) mm

    axes: LaminarAxes
    laminar_u: np.ndarray           # (N,) in [0,1]
    longitudinal_u: np.ndarray      # (N,) in [0,1]

    res_um_xyz: np.ndarray          # (3,) µm/voxel


def _compute_laminar_per_hemi(
    xyz: np.ndarray,
    x_mid: float,
) -> Tuple[LaminarAxes, np.ndarray, np.ndarray]:
    """
    Compute laminar + longitudinal coordinates by PCA. If both hemispheres exist,
    compute PCA per hemisphere and stitch (prevents diagonal artifacts).
    """
    left = np.where(xyz[:, 0] < x_mid)[0]
    right = np.where(xyz[:, 0] >= x_mid)[0]

    lam = np.zeros(xyz.shape[0], dtype=float)
    lon = np.zeros(xyz.shape[0], dtype=float)

    # fallback axes: computed on all points
    axes_all = pca_axes(xyz)

    if left.size > 50:
        axL = pca_axes(xyz[left])
        lam[left] = laminar_coordinate(xyz[left], axL)
        lon[left] = longitudinal_coordinate(xyz[left], axL)
    if right.size > 50:
        axR = pca_axes(xyz[right])
        lam[right] = laminar_coordinate(xyz[right], axR)
        lon[right] = longitudinal_coordinate(xyz[right], axR)

    # if one side too small, use global
    if left.size <= 50 or right.size <= 50:
        lam = laminar_coordinate(xyz, axes_all)
        lon = longitudinal_coordinate(xyz, axes_all)

    return axes_all, lam, lon


def load_region_geometry(
    atlas: Any, structure_name: str, *, hemisphere: str = "both"
) -> RegionGeometry:
    root_ids = resolve_structure_ids(atlas, structure_name)

    all_ids: List[int] = []
    for rid in root_ids:
        all_ids.append(int(rid))
        all_ids.extend(descendant_ids(atlas, int(rid)))
    all_ids = sorted(set(all_ids))

    hemi = hemisphere.lower()
    if hemi not in {"both", "left", "right"}:
        raise ValueError("hemisphere must be one of: 'both', 'left', 'right'")

    mask_zyx = mask_for_structure_ids(atlas.annotation, all_ids)
    if not mask_zyx.any():
        raise ValueError(
            f"Empty/unresolved region mask for '{structure_name}' (root_ids={root_ids})."
        )

    # Restrict to a single hemisphere while keeping atlas-aligned dimensions.
    if hemi != "both":
        x_mid = 0.5 * float(mask_zyx.shape[2])
        x_axis = np.arange(mask_zyx.shape[2], dtype=float)
        if hemi == "left":
            hemi_mask = x_axis < x_mid
        else:  # "right"
            hemi_mask = x_axis >= x_mid
        mask_zyx = mask_zyx & hemi_mask[None, None, :]
        if not mask_zyx.any():
            raise ValueError(
                f"Hemisphere '{hemisphere}' empty for '{structure_name}' (root_ids={root_ids})."
            )

    vox_zyx = np.argwhere(mask_zyx)  # (z,y,x)
    res_um = atlas_resolution_um(atlas)  # (x,y,z) µm/voxel

    xyz_vox = zyx_to_xyz_vox(vox_zyx).astype(float)
    xyz_mm = xyz_vox_to_mm(xyz_vox, res_um)

    # mid-sagittal plane in voxel space
    x_mid = 0.5 * float(atlas.annotation.shape[2])

    axes, lam_u, lon_u = _compute_laminar_per_hemi(xyz_mm, x_mid=x_mid * (res_um[0] / 1000.0))

    return RegionGeometry(
        name=structure_name,
        structure_id=int(root_ids[0]),
        mask_zyx=mask_zyx,
        voxel_points_zyx=vox_zyx,
        xyz_vox=xyz_vox,
        xyz_mm=xyz_mm,
        axes=axes,
        laminar_u=lam_u,
        longitudinal_u=lon_u,
        res_um_xyz=res_um,
    )
