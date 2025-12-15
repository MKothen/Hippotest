
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import time

import sys
# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml
from src.sim.pipeline import build_and_run


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and run 3D hippocampal-formation AdEx+COBA network")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "small.yaml"), help="Path to YAML config")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: runs/<run_name>_<timestamp>)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg.get("run_name", Path(args.config).stem)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (ROOT / "runs" / f"{run_name}_{ts}")
    build_and_run(cfg, out_dir)


if __name__ == "__main__":
    main()
