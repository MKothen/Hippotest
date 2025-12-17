from __future__ import annotations

import io
from dataclasses import dataclass
from io import TextIOWrapper
from typing import BinaryIO, Iterable

import numpy as np


@dataclass
class ParsedResClu:
    times_s: np.ndarray
    indices: np.ndarray
    cluster_ids: np.ndarray


_DEF_TIME_UNIT = "samples"


def _read_int_lines(handle: Iterable[str]) -> np.ndarray:
    vals = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(int(line.split()[0]))
        except ValueError:
            continue
    return np.asarray(vals, dtype=np.int64)


def parse_res_clu(
    res_file: BinaryIO,
    clu_file: BinaryIO,
    *,
    time_unit: str = _DEF_TIME_UNIT,
    sample_rate_hz: float | None = None,
    t_start_s: float = 0.0,
    t_stop_s: float | None = None,
) -> ParsedResClu:
    """
    Parse hc-3 ``*.res.N`` and ``*.clu.N`` pairs.

    Parameters
    ----------
    res_file : file-like
        Spike times, one integer per line (samples or seconds).
    clu_file : file-like
        Cluster ids per spike; the first line is a header and is skipped.
    time_unit : str
        ``"samples"`` (default) or ``"seconds"``. When ``"samples"``,
        ``sample_rate_hz`` must be provided to convert to seconds.
    sample_rate_hz : float, optional
        Sampling rate used to convert sample indices to seconds.
    t_start_s, t_stop_s : float
        Optional window; spikes outside are discarded.
    """

    res_handle = res_file if isinstance(res_file, io.TextIOBase) else TextIOWrapper(res_file, encoding="utf-8")
    clu_handle = clu_file if isinstance(clu_file, io.TextIOBase) else TextIOWrapper(clu_file, encoding="utf-8")

    res_samples = _read_int_lines(res_handle)

    clu_lines = clu_handle.readlines()
    if not clu_lines:
        raise ValueError("Empty .clu file")
    clu_ids = np.asarray([int(x.split()[0]) for x in clu_lines[1:] if x.strip()], dtype=np.int64)

    if res_samples.shape[0] != clu_ids.shape[0]:
        raise ValueError(f".res and .clu lengths differ: {res_samples.shape[0]} vs {clu_ids.shape[0]}")

    keep_mask = clu_ids > 1
    res_samples = res_samples[keep_mask]
    clu_ids = clu_ids[keep_mask]

    time_unit_l = time_unit.lower()
    if time_unit_l in ("sample", "samples"):
        if sample_rate_hz is None or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be provided when time_unit='samples'")
        times_s = res_samples.astype(float) / float(sample_rate_hz)
    elif time_unit_l in ("s", "sec", "second", "seconds"):
        times_s = res_samples.astype(float)
    elif time_unit_l in ("ms", "millisecond", "milliseconds"):
        times_s = res_samples.astype(float) * 1e-3
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")

    if t_stop_s is not None:
        time_mask = (times_s >= t_start_s) & (times_s <= t_stop_s)
    else:
        time_mask = times_s >= t_start_s

    times_s = times_s[time_mask]
    clu_ids = clu_ids[time_mask]

    cluster_ids = np.unique(clu_ids)
    clu_to_idx = {c: i for i, c in enumerate(cluster_ids)}
    indices = np.asarray([clu_to_idx[c] for c in clu_ids], dtype=np.int32)

    order = np.argsort(times_s, kind="stable")
    times_s = times_s[order]
    indices = indices[order]

    return ParsedResClu(times_s=times_s, indices=indices, cluster_ids=cluster_ids)
