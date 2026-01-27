from pathlib import Path

from src.sim.plots import save_plasticity_figure
from src.sim.plasticity import TemporalScalingConfig


def test_save_plasticity_figure(tmp_path: Path):
    out = tmp_path / "plasticity_overview.png"
    cfg = TemporalScalingConfig(target_sim_minutes=10, biological_max_hours=48)
    save_plasticity_figure(out, config=cfg)
    assert out.exists()
    assert out.stat().st_size > 0
