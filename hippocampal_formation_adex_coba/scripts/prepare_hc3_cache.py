from __future__ import annotations

import argparse
from pathlib import Path

from src.data.hc3.cache import build_hc3_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a cached npz of hc-3 EC spikes")
    p.add_argument("session_tar_gz", type=Path, help="Path to hc-3 session tar.gz")
    p.add_argument("metadata", type=Path, help="Path to hc3 metadata (xlsx or zip)")
    p.add_argument("--cache-dir", type=Path, default=Path("runs/hc3_cache"))
    p.add_argument("--regions", nargs="*", default=None, help="Filter units by region")
    p.add_argument("--cell-types", nargs="*", default=None, help="Filter units by cell type")
    p.add_argument("--t-start", type=float, default=0.0, help="Start time (s) for spikes")
    p.add_argument("--t-stop", type=float, default=None, help="Stop time (s) for spikes")
    p.add_argument("--time-unit", type=str, default="samples", help="Spike time unit in .res (samples or seconds)")
    p.add_argument("--sample-rate", type=float, default=None, help="Sampling rate (Hz) when time-unit is samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache = build_hc3_cache(
        session_tar_gz=args.session_tar_gz,
        metadata_xlsx=args.metadata,
        regions=args.regions,
        cell_types=args.cell_types,
        t_start_s=args.t_start,
        t_stop_s=args.t_stop,
        time_unit=args.time_unit,
        sample_rate_hz=args.sample_rate,
        cache_dir=args.cache_dir,
    )
    print(
        f"Cached {cache.n_spikes} spikes from {cache.n_units} units -> {cache.cache_path} (session={cache.session}, sr={cache.sample_rate_hz} Hz)"
    )


if __name__ == "__main__":
    main()
