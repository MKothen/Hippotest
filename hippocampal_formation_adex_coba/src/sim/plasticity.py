"""Plasticity primitives and temporal scaling helpers.

This module collects small, testable building blocks for synaptic and
structural plasticity inspired by the neuroscience summary in the prompt.
They are self-contained utilities rather than Brian2/Nengo bindings so they
can be validated in isolation before integration into simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


class STPMechanism:
    """Short-term potentiation with activity-dependent decay.

    The model tracks the size of the readily releasable pool (RRP) and the
    release probability. A brief high-frequency stimulation expands the RRP
    and increases release probability. Decay only happens when the synapse is
    stimulated again, consistent with “use it or lose it” storage.
    """

    def __init__(self, rrp_baseline: float = 10.0, pr_baseline: float = 0.3) -> None:
        self.rrp_baseline = rrp_baseline
        self.rrp_current = rrp_baseline
        self.pr_baseline = pr_baseline
        self.pr = pr_baseline

    def apply_hfs(self, rrp_scale: float = 2.5, pr_after: float = 0.6) -> None:
        """Apply a brief high-frequency stimulation (HFS).

        Parameters
        ----------
        rrp_scale:
            Multiplicative expansion of the readily releasable pool.
        pr_after:
            Release probability immediately after HFS.
        """

        self.rrp_current = self.rrp_baseline * rrp_scale
        self.pr = pr_after

    def decay_with_activity(self, stim_rate_hz: float) -> None:
        """Let STP decay only when there is stimulation.

        Parameters
        ----------
        stim_rate_hz:
            Rate of synaptic activation. Higher rates accelerate decay back to
            baseline.
        """

        if stim_rate_hz <= 0:
            return

        decay_fraction = 0.01 * stim_rate_hz
        self.rrp_current -= (self.rrp_current - self.rrp_baseline) * decay_fraction
        self.pr -= (self.pr - self.pr_baseline) * decay_fraction


class BTSPLearningRule:
    """Binary-weight BTSP rule with a stochastic gating signal.

    The rule captures the probabilistic nature of behavioral time-scale
    plasticity: weight changes only occur if a gating signal passes and the
    synaptic input falls within the plasticity window around a plateau event.
    """

    def __init__(self, p_gate: float = 0.5, rng: np.random.Generator | None = None) -> None:
        self.p_gate = p_gate
        self.rng = rng or np.random.default_rng()

    def update_weight(self, weight: int, t_synapse: float, t_plateau: float) -> int:
        """Update a binary synaptic weight.

        Parameters
        ----------
        weight:
            Current weight (0 for silent, 1 for active).
        t_synapse:
            Timing of synaptic input (seconds).
        t_plateau:
            Timing of plateau potential (seconds).
        """

        delta_t = t_synapse - t_plateau
        if delta_t < -6.0 or delta_t > 4.0:
            return weight

        if self.rng.random() > self.p_gate:
            return weight

        if weight == 0 and -3.0 <= delta_t <= 2.0:
            return 1
        if weight == 1 and ((-6.0 <= delta_t < -3.0) or (2.0 < delta_t <= 4.0)):
            return 0

        return weight


def bidirectional_plasticity(ca_concentration: float, history_factor: float = 1.0) -> float:
    """Calcium-dependent bidirectional plasticity with sigmoid thresholds.

    Parameters
    ----------
    ca_concentration:
        Calcium concentration in micromolar.
    history_factor:
        Sliding threshold factor for BCM-like metaplasticity.
    """

    theta_m = 0.7 * history_factor
    ltp_mag = 2.0 / (1 + np.exp(-5 * (ca_concentration - theta_m)))
    ltd_mag = -1.0 / (1 + np.exp(5 * (ca_concentration - theta_m + 0.2)))
    return float(ltp_mag + ltd_mag)


class MossyFiberSynapse:
    """Presynaptic plasticity with optional preNMDAR modulation."""

    def __init__(self, pre_nmdar_active: bool = False, pr_baseline: float = 0.1) -> None:
        self.pre_nmdar_active = pre_nmdar_active
        self.pr_baseline = pr_baseline
        self.presynaptic_ca = 0.0

    def low_frequency_facilitation(self, stim_rate_hz: float = 1.0) -> None:
        if self.pre_nmdar_active and stim_rate_hz > 0:
            self.presynaptic_ca += 0.05
            self.pr_baseline += 0.02 * self.presynaptic_ca

    def apply_hfs_ltp(self) -> None:
        cAMP_level = 2.0
        if self.pre_nmdar_active:
            cAMP_level *= 1.3
        self.pr_baseline = 0.5 * cAMP_level / 2.0


class CA2Synapse:
    """Developmentally regulated CA2 plasticity."""

    def __init__(self, age_days: int) -> None:
        self.age_days = age_days
        self.pnn_maturity = self._calculate_pnn_maturity()

    def _calculate_pnn_maturity(self) -> float:
        if self.age_days < 8:
            return 0.0
        if self.age_days <= 14:
            return (self.age_days - 8) / 6.0
        return 1.0

    def can_induce_ltp(self) -> bool:
        return self.pnn_maturity < 0.5


class SynapticTag:
    """Asymmetric synaptic tagging and capture timers."""

    def __init__(self) -> None:
        self.tag_set = False
        self.tag_timer = 0.0
        self.prp_available = False
        self.prp_timer = 0.0

    def weak_stimulation(self) -> None:
        self.tag_set = True
        self.tag_timer = 60.0

    def strong_stimulation(self) -> None:
        self.prp_available = True
        self.prp_timer = 150.0

    def check_conversion_to_ltp3(self) -> bool:
        return self.tag_set and self.prp_available

    def update(self, minutes_elapsed: float) -> None:
        if self.tag_set:
            self.tag_timer -= minutes_elapsed
            if self.tag_timer <= 0:
                self.tag_set = False
        if self.prp_available:
            self.prp_timer -= minutes_elapsed
            if self.prp_timer <= 0:
                self.prp_available = False


class Spine:
    """Spine stabilization with activity and Arc expression requirements."""

    def __init__(self) -> None:
        self.age_hours = 0.0
        self.has_arc_expression = False
        self.recent_activity_count = 0
        self.is_stabilized = False

    def update(self, active_this_hour: bool) -> str | None:
        self.age_hours += 1
        if active_this_hour:
            self.recent_activity_count += 1

        if self.age_hours <= 48:
            if self.recent_activity_count >= 10 and self.has_arc_expression:
                self.is_stabilized = True

        if self.age_hours > 48 and not self.is_stabilized:
            return "PRUNE"
        return None


@dataclass
class ScaledPlasticityParameters:
    """Scaled time constants for synaptic/structural plasticity."""

    compression_factor: float

    stp_half_life_s: float = 35 * 60
    rrp_refill_time_s: float = 2.0
    ltp1_duration_s: float = 2 * 3600
    ltp2_duration_s: float = 6 * 3600
    ltp3_onset_s: float = 8 * 3600
    btsp_pre_window_s: float = -6.0
    btsp_post_window_s: float = 4.0
    plateau_duration_s: float = 0.1
    tag_lifetime_w_to_s_s: float = 60 * 60
    tag_lifetime_s_to_w_s: float = 150 * 60
    prp_availability_s: float = 180 * 60
    spine_rapid_actin_phase_s: float = 5 * 60
    spine_stabilization_phase_s: float = 60 * 60
    spine_consolidation_window_s: float = 36 * 3600

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "compression_factor":
                continue
            original_value = getattr(self, field_name)
            setattr(self, field_name, original_value / self.compression_factor)


@dataclass
class ScaledMolecularKinetics:
    """Scaled molecular and cellular kinetics."""

    compression_factor: float
    ca_rise_time_s: float = 0.010
    ca_decay_time_s: float = 0.100
    camkii_activation_time_s: float = 1.0
    camkii_persistent_time_s: float = 60 * 60
    local_translation_time_s: float = 10 * 60
    somatic_transcription_time_s: float = 30 * 60
    mrna_transport_time_s: float = 20 * 60
    ampar_insertion_time_s: float = 2 * 60
    ampar_removal_time_s: float = 10 * 60

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "compression_factor":
                continue
            original_value = getattr(self, field_name)
            setattr(self, field_name, original_value / self.compression_factor)


@dataclass
class ScaledHomeostasis:
    """Scaled homeostatic time constants."""

    compression_factor: float
    bcm_integration_window_s: float = 1 * 3600
    bcm_threshold_shift_time_s: float = 6 * 3600
    scaling_detection_window_s: float = 12 * 3600
    scaling_implementation_time_s: float = 24 * 3600
    spine_formation_rate_per_min: float = 1 / (24 * 60)
    spine_elimination_rate_per_min: float = 1 / (24 * 60)

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "compression_factor":
                continue
            original_value = getattr(self, field_name)
            setattr(self, field_name, original_value / self.compression_factor)


def temporal_scaling_factor(target_sim_time_minutes: float, biological_max_hours: float = 48.0) -> float:
    """Compute compression factor to fit biological time into the simulation."""

    biological_max_minutes = biological_max_hours * 60.0
    if target_sim_time_minutes <= 0:
        raise ValueError("target_sim_time_minutes must be positive")
    return biological_max_minutes / target_sim_time_minutes


class TemporalScalingConfig:
    """Master configuration that applies temporal compression across modules."""

    def __init__(
        self,
        target_sim_minutes: float = 10.0,
        biological_max_hours: float = 48.0,
    ) -> None:
        self.target_sim_minutes = target_sim_minutes
        self.biological_max_hours = biological_max_hours
        self.compression_factor = temporal_scaling_factor(target_sim_minutes, biological_max_hours)
        self.plasticity_params = ScaledPlasticityParameters(self.compression_factor)
        self.molecular_params = ScaledMolecularKinetics(self.compression_factor)
        self.homeostasis_params = ScaledHomeostasis(self.compression_factor)
        validate_hierarchy(self)

    def report(self) -> Dict[str, float]:
        return {
            "compression_factor": self.compression_factor,
            "ltp1_duration_s": self.plasticity_params.ltp1_duration_s,
            "ltp2_duration_s": self.plasticity_params.ltp2_duration_s,
            "tag_lifetime_w_to_s_s": self.plasticity_params.tag_lifetime_w_to_s_s,
            "spine_consolidation_window_s": self.plasticity_params.spine_consolidation_window_s,
        }


def validate_hierarchy(config: TemporalScalingConfig) -> None:
    """Ensure relative ordering of timescales survives compression."""

    p = config.plasticity_params
    m = config.molecular_params
    checks = [
        m.ca_rise_time_s < m.ca_decay_time_s,
        m.ca_decay_time_s < m.camkii_activation_time_s,
        p.stp_half_life_s < p.ltp1_duration_s,
        p.ltp1_duration_s < p.ltp2_duration_s,
        p.ltp2_duration_s < p.ltp3_onset_s,
        abs(p.btsp_pre_window_s) > abs(p.btsp_post_window_s),
        p.tag_lifetime_w_to_s_s < p.tag_lifetime_s_to_w_s,
        p.spine_rapid_actin_phase_s < p.spine_stabilization_phase_s,
        p.spine_stabilization_phase_s < p.spine_consolidation_window_s,
    ]
    if not all(checks):
        raise ValueError("Temporal hierarchy violated after scaling")


def check_observability(sim_time_minutes: float, config: TemporalScalingConfig) -> Dict[str, bool]:
    """Return which processes are observable within the simulation window."""

    sim_time_seconds = sim_time_minutes * 60.0
    p = config.plasticity_params
    observability = {
        "STP": (p.stp_half_life_s * np.log(2)) < sim_time_seconds,
        "LTP1": p.ltp1_duration_s < sim_time_seconds,
        "LTP2": p.ltp2_duration_s < sim_time_seconds * 0.6,
        "BTSP": p.btsp_post_window_s < sim_time_seconds * 0.1,
        "Tagging_W_to_S": p.tag_lifetime_w_to_s_s < sim_time_seconds * 0.4,
        "Tagging_S_to_W": p.tag_lifetime_s_to_w_s < sim_time_seconds * 0.7,
        "Spine_growth": p.spine_rapid_actin_phase_s < sim_time_seconds * 0.2,
    }
    return observability


__all__ = [
    "STPMechanism",
    "BTSPLearningRule",
    "bidirectional_plasticity",
    "MossyFiberSynapse",
    "CA2Synapse",
    "SynapticTag",
    "Spine",
    "ScaledPlasticityParameters",
    "ScaledMolecularKinetics",
    "ScaledHomeostasis",
    "TemporalScalingConfig",
    "temporal_scaling_factor",
    "validate_hierarchy",
    "check_observability",
]
