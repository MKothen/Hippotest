
import numpy as np

from src.connectivity.pathways import PathwaySpec, SynapseSpec
from src.connectivity.builder import PopulationGeometry, build_edges_for_pathway


def test_build_edges_shapes():
    rng = np.random.default_rng(0)
    pre_xyz = rng.normal(size=(50, 3)) * 0.2  # mm
    post_xyz = rng.normal(size=(80, 3)) * 0.2

    pre = PopulationGeometry(
        name="pre", xyz_mm=pre_xyz,
        soma_layer=np.array(["SP"]*50, dtype=object),
        region="CA1", cell_class="excitatory", cell_type="CA1_Pyramidal"
    )
    post = PopulationGeometry(
        name="post", xyz_mm=post_xyz,
        soma_layer=np.array(["SP"]*80, dtype=object),
        region="CA1", cell_class="excitatory", cell_type="CA1_Pyramidal"
    )

    spec = PathwaySpec(
        name="test",
        pre_pop="pre", post_pop="post",
        kind="exc",
        k_out_mean=5, k_out_std=0,
        radius_mm=1.0, sigma_mm=0.3,
        velocity_m_per_s=0.3, base_delay_ms=1.0,
        target_layers={"SR": 1.0},
        synapse=SynapseSpec(w_ampa_nS=0.5, w_nmda_nS=0.2),
        exclude_self=False,
    )

    edges = build_edges_for_pathway(spec, pre, post, rng, scale=1.0, show_progress=False)
    assert edges.pre_idx.shape == edges.post_idx.shape == edges.dist_mm.shape == edges.delay_ms.shape
    assert edges.w_ampa_nS.shape == edges.pre_idx.shape
    assert np.all(edges.delay_ms >= 1.0)
