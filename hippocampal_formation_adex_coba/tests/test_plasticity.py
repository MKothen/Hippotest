import numpy as np

from src.sim.plasticity import (
    BTSPLearningRule,
    CA2Synapse,
    Spine,
    SynapticTag,
    TemporalScalingConfig,
    bidirectional_plasticity,
    check_observability,
    temporal_scaling_factor,
)


def test_bidirectional_plasticity_direction():
    ltp_value = bidirectional_plasticity(1.5)
    ltd_value = bidirectional_plasticity(0.1)
    assert ltp_value > 0
    assert ltd_value < 0


def test_btsp_learning_rule_weight_changes():
    rule = BTSPLearningRule(p_gate=1.0, rng=np.random.default_rng(seed=42))
    assert rule.update_weight(0, t_synapse=1.0, t_plateau=0.0) == 1
    assert rule.update_weight(1, t_synapse=-4.0, t_plateau=0.0) == 0


def test_synaptic_tagging_windows():
    tag = SynapticTag()
    tag.weak_stimulation()
    tag.strong_stimulation()
    assert tag.check_conversion_to_ltp3()
    tag.update(minutes_elapsed=200)
    assert not tag.tag_set
    assert not tag.prp_available


def test_spine_pruning_logic():
    spine = Spine()
    spine.has_arc_expression = True
    for _ in range(10):
        spine.update(active_this_hour=True)
    assert spine.is_stabilized
    spine = Spine()
    outcome = None
    for _ in range(50):
        outcome = spine.update(active_this_hour=False)
    assert outcome == "PRUNE"


def test_ca2_developmental_window():
    juvenile = CA2Synapse(age_days=9)
    adult = CA2Synapse(age_days=30)
    assert juvenile.can_induce_ltp()
    assert not adult.can_induce_ltp()


def test_temporal_scaling_and_observability():
    cf = temporal_scaling_factor(target_sim_time_minutes=10, biological_max_hours=48)
    assert cf == 288
    config = TemporalScalingConfig(target_sim_minutes=10, biological_max_hours=48)
    obs = check_observability(10, config)
    assert obs["STP"]
    assert obs["LTP1"]
