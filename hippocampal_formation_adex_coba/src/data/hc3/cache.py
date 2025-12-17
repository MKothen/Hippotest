from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import numpy as np

from .metadata import HC3Metadata, load_metadata
from .parse_res_clu import parse_res_clu


@dataclass
class HC3CacheData:
    cache_path: Path
    times_s: np.ndarray
    indices: np.ndarray
    unit_region: np.ndarray
    unit_celltype: np.ndarray
    topdir: str
    session: str
    sample_rate_hz: float

    @property
    def n_units(self) -> int:
        return int(self.unit_region.shape[0])

    @property
    def n_spikes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def spike_times_s(self) -> np.ndarray:
        """Alias for compatibility with replay inputs."""
        return self.times_s

    @property
    def spike_unit_idx(self) -> np.ndarray:
        """Alias for compatibility with replay inputs."""
        return self.indices

    @property
    def unit_metadata(self) -> list[dict[str, str]]:
        """List of metadata dicts for each unit."""
        return [
            {"region": str(region), "cell_type": str(celltype)}
            for region, celltype in zip(self.unit_region, self.unit_celltype)
        ]


_RES_RE = re.compile(r"\.res\.(\d+)$")
_CLU_RE = re.compile(r"\.clu\.(\d+)$")


def _infer_session_names(tar_path: Path) -> Tuple[str, str]:
    base = tar_path.name[:-7] if tar_path.name.endswith(".tar.gz") else tar_path.stem
    topdir = tar_path.parent.name
    session = base
    return topdir, session


def _read_sampling_rate_from_xml(bytes_obj: bytes) -> Optional[float]:
    try:
        root = ElementTree.fromstring(bytes_obj)
    except ElementTree.ParseError:
        return None

    for elem in root.iter():
        tag = elem.tag.lower()
        if "sampl" in tag and elem.text:
            try:
                val = float(elem.text)
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


def _iter_pairs(members: List[tarfile.TarInfo]) -> Dict[int, Tuple[tarfile.TarInfo, tarfile.TarInfo]]:
    res_members: Dict[int, tarfile.TarInfo] = {}
    clu_members: Dict[int, tarfile.TarInfo] = {}
    for m in members:
        res_match = _RES_RE.search(m.name)
        if res_match:
            res_members[int(res_match.group(1))] = m
        clu_match = _CLU_RE.search(m.name)
        if clu_match:
            clu_members[int(clu_match.group(1))] = m

    pairs: Dict[int, Tuple[tarfile.TarInfo, tarfile.TarInfo]] = {}
    for ele, res_m in res_members.items():
        if ele in clu_members:
            pairs[ele] = (res_m, clu_members[ele])
    return pairs


def _load_from_tar(
    tar_path: Path,
    *,
    metadata: HC3Metadata,
    regions: Optional[Sequence[str]] = None,
    cell_types: Optional[Sequence[str]] = None,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
    time_unit: str = "samples",
    sample_rate_hz: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], float]:
    regions_set = set(regions) if regions else None
    celltype_set = set(cell_types) if cell_types else None

    topdir, _ = _infer_session_names(tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        xml_members = [m for m in members if m.name.lower().endswith(".xml")]
        sr = sample_rate_hz
        if sr is None and xml_members:
            xml_bytes = tar.extractfile(xml_members[0]).read()
            sr = _read_sampling_rate_from_xml(xml_bytes)
        if sr is None and time_unit.lower().startswith("sample"):
            raise ValueError("Could not infer sampling rate from XML; please provide sample_rate_hz")

        pairs = _iter_pairs(members)

        all_times: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []
        unit_region: List[str] = []
        unit_celltype: List[str] = []
        for ele, (res_m, clu_m) in pairs.items():
            res_f = tar.extractfile(res_m)
            clu_f = tar.extractfile(clu_m)
            if res_f is None or clu_f is None:
                continue
            parsed = parse_res_clu(
                res_f,
                clu_f,
                time_unit=time_unit,
                sample_rate_hz=sr,
                t_start_s=t_start_s,
                t_stop_s=t_stop_s,
            )

            if parsed.cluster_ids.size == 0:
                continue

            local_map: Dict[int, int] = {}
            for local_idx, clu_id in enumerate(parsed.cluster_ids):
                meta = metadata.lookup(topdir, ele, int(clu_id))
                region = meta.region if meta else "unknown"
                ctype = meta.cell_type if meta else "unknown"
                if regions_set and region not in regions_set:
                    continue
                if celltype_set and ctype not in celltype_set:
                    continue
                global_idx = len(unit_region)
                local_map[local_idx] = global_idx
                unit_region.append(region)
                unit_celltype.append(ctype)

            if not local_map:
                continue

            mapping = np.full(parsed.cluster_ids.shape[0], -1, dtype=int)
            for local_idx, global_idx in local_map.items():
                mapping[local_idx] = global_idx

            mapped_indices = mapping[parsed.indices]
            keep = mapped_indices >= 0
            if keep.any():
                all_times.append(parsed.times_s[keep])
                all_indices.append(mapped_indices[keep])

        if not all_times:
            return np.array([]), np.array([], dtype=int), [], [], float(sr or 0)

        times_s = np.concatenate(all_times)
        indices = np.concatenate(all_indices)
        order = np.argsort(times_s, kind="stable")
        times_s = times_s[order]
        indices = indices[order]

        return times_s, indices.astype(np.int32), unit_region, unit_celltype, float(sr or 0.0)


def _default_cache_path(cache_dir: Path, topdir: str, session: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{topdir}__{session}.npz"
    return cache_dir / fname


def build_hc3_cache(
    session_tar_gz: str | Path,
    *,
    metadata_xlsx: str | Path,
    regions: Optional[Sequence[str]] = None,
    cell_types: Optional[Sequence[str]] = None,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
    time_unit: str = "samples",
    sample_rate_hz: float | None = None,
    cache_dir: str | Path = "runs/hc3_cache",
) -> HC3CacheData:
    tar_path = Path(session_tar_gz)
    meta = load_metadata(metadata_xlsx)
    topdir, session = _infer_session_names(tar_path)

    times_s, indices, unit_region, unit_celltype, sr = _load_from_tar(
        tar_path,
        metadata=meta,
        regions=regions,
        cell_types=cell_types,
        t_start_s=t_start_s,
        t_stop_s=t_stop_s,
        time_unit=time_unit,
        sample_rate_hz=sample_rate_hz,
    )

    cache_dir = Path(cache_dir)
    cache_path = _default_cache_path(cache_dir, topdir, session)

    np.savez(
        cache_path,
        times_s=times_s.astype(np.float32),
        indices=indices.astype(np.int32),
        unit_region=np.asarray(unit_region, dtype=object),
        unit_celltype=np.asarray(unit_celltype, dtype=object),
        topdir=topdir,
        session=session,
        sample_rate_hz=np.array([sr], dtype=float),
    )

    return HC3CacheData(
        cache_path=cache_path,
        times_s=times_s.astype(np.float32),
        indices=indices.astype(np.int32),
        unit_region=np.asarray(unit_region, dtype=object),
        unit_celltype=np.asarray(unit_celltype, dtype=object),
        topdir=topdir,
        session=session,
        sample_rate_hz=float(sr),
    )


def load_hc3_cache(cache_path: str | Path) -> HC3CacheData:
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=True)
    topdir = str(data.get("topdir", ""))
    session = str(data.get("session", ""))
    sr_arr = data.get("sample_rate_hz")
    sr = float(sr_arr[0]) if sr_arr is not None else 0.0
    return HC3CacheData(
        cache_path=p,
        times_s=np.asarray(data["times_s"], dtype=float),
        indices=np.asarray(data["indices"], dtype=int),
        unit_region=np.asarray(data["unit_region"], dtype=object),
        unit_celltype=np.asarray(data["unit_celltype"], dtype=object),
        topdir=topdir,
        session=session,
        sample_rate_hz=sr,
    )


def load_or_build_hc3_cache(
    session_tar_gz: str | Path,
    *,
    metadata_xlsx: str | Path,
    regions: Optional[Sequence[str]] = None,
    cell_types: Optional[Sequence[str]] = None,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
    time_unit: str = "samples",
    sample_rate_hz: float | None = None,
    cache_dir: str | Path = "runs/hc3_cache",
) -> HC3CacheData:
    tar_path = Path(session_tar_gz)
    cache_dir = Path(cache_dir)
    topdir, session = _infer_session_names(tar_path)
    cache_path = _default_cache_path(cache_dir, topdir, session)
    if cache_path.exists():
        return load_hc3_cache(cache_path)

    return build_hc3_cache(
        session_tar_gz=tar_path,
        metadata_xlsx=metadata_xlsx,
        regions=regions,
        cell_types=cell_types,
        t_start_s=t_start_s,
        t_stop_s=t_stop_s,
        time_unit=time_unit,
        sample_rate_hz=sample_rate_hz,
        cache_dir=cache_dir,
    )


# Backwards compatibility alias
def load_or_build_cache(*args, **kwargs):
    return load_or_build_hc3_cache(*args, **kwargs)
