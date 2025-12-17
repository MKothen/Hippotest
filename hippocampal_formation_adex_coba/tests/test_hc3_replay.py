from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.celltypes.adex_params import PRESETS
from src.celltypes.library import CellTypeSpec
from src.connectivity.builder import PopulationGeometry, PathwayEdges
from src.connectivity.pathways import PathwaySpec, SynapseSpec
from src.data.hc3.cache import build_hc3_cache
from src.sim.runner import run_simulation


def _make_tarball(
    tmpdir: Path, res: list[int], clu: list[int], sample_rate_hz: float, session_name: str
) -> Path:
    res_text = "\n".join(str(r) for r in res) + "\n"
    clu_text = "4\n" + "\n".join(str(c) for c in clu) + "\n"
    xml_text = f"<parameters><samplingRate>{sample_rate_hz}</samplingRate></parameters>"
    tar_path = tmpdir / f"{session_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, text in {
            "fake.res.1": res_text,
            "fake.clu.1": clu_text,
            "fake.xml": xml_text,
        }.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, fileobj=BytesIO(data))
    return tar_path


def test_hc3_replay_smoke(tmp_path: Path):
    # ----------------- build synthetic hc3 cache
    res = [100, 300, 700]
    clu = [2, 2, 3]
    sr_hz = 1000.0
    topdir_name = "ec013.45"
    session_name = "ec013.805"
    session_dir = tmp_path / topdir_name
    session_dir.mkdir()
    tar_path = _make_tarball(session_dir, res, clu, sr_hz, session_name)
    meta = pd.DataFrame(
        {
            "topdir": [topdir_name, topdir_name],
            "ele": [1, 1],
            "clu": [2, 3],
            "region": ["EC2", "EC2"],
            "cellType": ["p", "p"],
        }
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)

    cache = build_hc3_cache(
        session_tar_gz=tar_path,
        metadata_xlsx=meta_path,
        regions=["EC2"],
        cell_types=["p"],
        t_stop_s=0.5,
        time_unit="samples",
        sample_rate_hz=sr_hz,
        cache_dir=tmp_path / "cache",
    )
    assert cache.n_spikes == 2

    # ----------------- minimal network wiring
    pop_specs = {
        "EC_L2": CellTypeSpec("EC_L2", "excitatory", "RS", PRESETS["RS"], {}),
        "DG_GC": CellTypeSpec("DG_GC", "excitatory", "RS", PRESETS["RS"], {}),
    }

    pop_geom = {
        "EC_L2_Exc": PopulationGeometry(
            name="EC_L2_Exc",
            xyz_mm=np.zeros((2, 3)),
            soma_layer=np.array(["L2", "L2"], dtype=object),
            region="EC",
            cell_class="excitatory",
            cell_type="EC_L2",
        ),
        "DG_GC": PopulationGeometry(
            name="DG_GC",
            xyz_mm=np.zeros((2, 3)),
            soma_layer=np.array(["GCL", "GCL"], dtype=object),
            region="DG",
            cell_class="excitatory",
            cell_type="DG_GC",
        ),
    }

    path_spec = PathwaySpec(
        name="replay_to_gc",
        pre_pop="EC_L2_Exc",
        post_pop="DG_GC",
        kind="exc",
        k_out_mean=1,
        k_out_std=0,
        radius_mm=1.0,
        sigma_mm=1.0,
        velocity_m_per_s=0.5,
        base_delay_ms=1.0,
        target_layers={"GCL": 1.0},
        synapse=SynapseSpec(w_ampa_nS=0.5),
    )

    edges = {
        "replay_to_gc": PathwayEdges(
            spec=path_spec,
            pre_idx=np.array([0, 1], dtype=int),
            post_idx=np.array([0, 1], dtype=int),
            dist_mm=np.array([0.1, 0.1], dtype=float),
            delay_ms=np.array([1.0, 1.0], dtype=float),
            w_ampa_nS=np.array([0.5, 0.5], dtype=float),
            w_nmda_nS=np.zeros(2, dtype=float),
            w_gabaa_nS=np.zeros(2, dtype=float),
            w_gabab_nS=np.zeros(2, dtype=float),
            target_layer=np.array(["GCL", "GCL"], dtype=object),
        )
    }

    config = {
        "seed": 1,
        "simulation": {
            "dt_ms": 0.1,
            "t_sim_s": 1.0,
            "codegen_target": "numpy",
            "record": {"n_state_neurons_per_pop": 0, "max_raster_neurons_per_pop": 2},
            "background": {"rate_hz": 0.0, "n_sources": 0},
        },
        "ec_input": {
            "mode": "hc3_replay",
            "hc3": {
                "session_tar_gz": tar_path,
                "metadata_xlsx": meta_path,
                "regions": ["EC2"],
                "cell_types": ["p"],
                "cache_dir": cache.cache_path.parent,
                "t_stop_s": 0.5,
                "time_unit": "samples",
                "sample_rate_hz": sr_hz,
            },
        },
    }

    out_dir = tmp_path / "out"
    run_simulation(config, pop_specs=pop_specs, pop_geom=pop_geom, edges=edges, out_dir=out_dir)

    npz = np.load(out_dir / "data" / "sim_outputs.npz")
    assert npz["EC_L2_Exc_spike_t_s"].size == cache.n_spikes
    assert (out_dir / "plots" / "activity_overview.png").exists()
