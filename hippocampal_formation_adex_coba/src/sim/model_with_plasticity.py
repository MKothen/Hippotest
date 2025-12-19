from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PopulationRateMonitor, StateMonitor,
    ms, mV, pF, nS, pA, Hz, siemens, volt, amp, second, prefs
)

from ..celltypes.adex_params import AdExParams


@dataclass
class SynapseKinetics:
    tau_ampa_ms: float = 5.0
    tau_nmda_ms: float = 100.0
    tau_gabaa_ms: float = 10.0
    tau_gabab_ms: float = 200.0
    E_ampa_mV: float = 0.0
    E_nmda_mV: float = 0.0
    E_gabaa_mV: float = -75.0
    E_gabab_mV: float = -90.0


@dataclass
class PlasticityConfig:
    """Configuration for plasticity mechanisms."""
    enable_stp: bool = True
    enable_stdp: bool = False
    enable_tagging: bool = False
    
    # STP parameters
    stp_tau_decay_ms: float = 1000.0  # decay time constant
    stp_u_baseline: float = 0.3       # baseline release probability
    stp_u_max: float = 0.9            # maximum release probability
    stp_tau_rec_ms: float = 50.0      # recovery time constant
    
    # STDP parameters
    stdp_tau_pre_ms: float = 20.0     # pre-synaptic trace time constant
    stdp_tau_post_ms: float = 20.0    # post-synaptic trace time constant
    stdp_A_plus: float = 0.005        # LTP amplitude
    stdp_A_minus: float = 0.00525     # LTD amplitude (slightly larger for depression bias)
    stdp_w_min: float = 0.0           # minimum weight
    stdp_w_max: float = 2.0           # maximum weight (relative to initial)
    
    # Synaptic tagging parameters
    tag_theta_tag: float = 0.5        # threshold for tag setting (normalized activity)
    tag_theta_prp: float = 0.8        # threshold for PRP synthesis
    tag_tau_tag_min: float = 60.0     # tag lifetime in minutes
    tag_tau_prp_min: float = 150.0    # PRP lifetime in minutes


DEFAULT_KINETICS = SynapseKinetics()
DEFAULT_PLASTICITY = PlasticityConfig()


ADEx_COBA_EQS = r"""
dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn + I_ext)/C : volt (unless refractory)
dw/dt = (a*(v - EL) - w)/tau_w : amp

dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_nmda/dt = -g_nmda/tau_nmda : siemens
dg_gabaa/dt = -g_gabaa/tau_gabaa : siemens
dg_gabab/dt = -g_gabab/tau_gabab : siemens

B = 1.0/(1.0 + exp(-0.062*(v/mV))/3.57) : 1
I_syn = g_ampa*(E_ampa - v) + g_nmda*B*(E_nmda - v) + g_gabaa*(E_gabaa - v) + g_gabab*(E_gabab - v) : amp

I_ext : amp

# STDP traces (for post-synaptic neurons)
dApre/dt = -Apre/tau_stdp_pre : 1 (event-driven)
dApost/dt = -Apost/tau_stdp_post : 1 (event-driven)

# parameters (constants per population)
C : farad (constant)
gL : siemens (constant)
EL : volt (constant)
VT : volt (constant)
DeltaT : volt (constant)
a : siemens (constant)
tau_w : second (constant)
b : amp (constant)
Vr : volt (constant)
Vcut : volt (constant)

tau_ampa : second (constant)
tau_nmda : second (constant)
tau_gabaa : second (constant)
tau_gabab : second (constant)

E_ampa : volt (constant)
E_nmda : volt (constant)
E_gabaa : volt (constant)
E_gabab : volt (constant)

tau_stdp_pre : second (constant)
tau_stdp_post : second (constant)
"""


def make_population(
    name: str,
    n: int,
    adex: AdExParams,
    kinetics: SynapseKinetics = DEFAULT_KINETICS,
    plasticity_cfg: PlasticityConfig = DEFAULT_PLASTICITY,
    dt_ms: float = 0.1,
    codegen_target: str = "cython",
    seed: int | None = None,
) -> NeuronGroup:
    prefs.codegen.target = codegen_target
    G = NeuronGroup(
        n,
        model=ADEx_COBA_EQS,
        threshold="v > Vcut",
        reset="""
            v = Vr
            w += b
            Apre += 1.0
            Apost += 1.0
        """,
        refractory=adex.t_ref_ms * ms,
        method="euler",
        name=name,
    )

    # Set parameters (unit conversions)
    G.C = adex.C_pF * pF
    G.gL = adex.gL_nS * nS
    G.EL = adex.EL_mV * mV
    G.VT = adex.VT_mV * mV
    G.DeltaT = adex.DeltaT_mV * mV
    G.a = adex.a_nS * nS
    G.tau_w = adex.tau_w_ms * ms
    G.b = adex.b_pA * pA
    G.Vr = adex.Vr_mV * mV
    G.Vcut = adex.Vcut_mV * mV

    G.tau_ampa = kinetics.tau_ampa_ms * ms
    G.tau_nmda = kinetics.tau_nmda_ms * ms
    G.tau_gabaa = kinetics.tau_gabaa_ms * ms
    G.tau_gabab = kinetics.tau_gabab_ms * ms

    G.E_ampa = kinetics.E_ampa_mV * mV
    G.E_nmda = kinetics.E_nmda_mV * mV
    G.E_gabaa = kinetics.E_gabaa_mV * mV
    G.E_gabab = kinetics.E_gabab_mV * mV

    G.tau_stdp_pre = plasticity_cfg.stdp_tau_pre_ms * ms
    G.tau_stdp_post = plasticity_cfg.stdp_tau_post_ms * ms

    # initial conditions
    G.v = adex.EL_mV * mV + (np.random.randn(n) * 2.0) * mV
    G.w = 0.0 * pA
    G.g_ampa = 0.0 * nS
    G.g_nmda = 0.0 * nS
    G.g_gabaa = 0.0 * nS
    G.g_gabab = 0.0 * nS
    G.I_ext = 0.0 * pA
    G.Apre = 0.0
    G.Apost = 0.0

    return G


def make_synapses_with_plasticity(
    pre: NeuronGroup, 
    post: NeuronGroup, 
    name: str,
    plasticity_cfg: PlasticityConfig = DEFAULT_PLASTICITY,
) -> Synapses:
    """
    Create synapses with optional plasticity mechanisms:
    - Short-term plasticity (STP): depression/facilitation via release probability
    - STDP: spike-timing dependent plasticity
    - Synaptic tagging: weak/strong stimulation tracking
    """
    
    # Build the model equations dynamically based on enabled plasticity
    model_eqs = r"""
        w_ampa_base : siemens
        w_nmda_base : siemens
        w_gabaa_base : siemens
        w_gabab_base : siemens
    """
    
    # STP variables
    if plasticity_cfg.enable_stp:
        model_eqs += r"""
        u : 1  # release probability (0-1)
        R : 1  # fraction of resources available (0-1)
        du/dt = (u_baseline - u) / tau_stp_decay : 1 (clock-driven)
        dR/dt = (1.0 - R) / tau_stp_rec : 1 (clock-driven)
        u_baseline : 1 (constant)
        u_max : 1 (constant)
        tau_stp_decay : second (constant)
        tau_stp_rec : second (constant)
        """
    
    # STDP variables
    if plasticity_cfg.enable_stdp:
        model_eqs += r"""
        w_plast : 1  # plastic weight multiplier (starts at 1.0)
        A_plus : 1 (constant)
        A_minus : 1 (constant)
        w_min : 1 (constant)
        w_max : 1 (constant)
        """
    
    # Synaptic tagging variables
    if plasticity_cfg.enable_tagging:
        model_eqs += r"""
        tag : 1  # tag state (0 or 1)
        prp : 1  # protein synthesis state (0 or 1)
        dtag/dt = -tag / tau_tag : 1 (clock-driven)
        dprp/dt = -prp / tau_prp : 1 (clock-driven)
        tau_tag : second (constant)
        tau_prp : second (constant)
        theta_tag : 1 (constant)
        theta_prp : 1 (constant)
        last_pre_spike : second
        """
    
    # Build on_pre equations
    on_pre_eqs = ""
    
    if plasticity_cfg.enable_stp:
        # STP: modulate release by u*R, then update u and R
        on_pre_eqs += r"""
        u_eff = clip(u * R, 0, 1)
        g_ampa_post += w_ampa_base * u_eff
        g_nmda_post += w_nmda_base * u_eff
        g_gabaa_post += w_gabaa_base * u_eff
        g_gabab_post += w_gabab_base * u_eff
        u = clip(u + u_baseline * (u_max - u), 0, u_max)
        R = clip(R - u_eff * R, 0, 1)
        """
    else:
        on_pre_eqs += r"""
        g_ampa_post += w_ampa_base
        g_nmda_post += w_nmda_base
        g_gabaa_post += w_gabaa_base
        g_gabab_post += w_gabab_base
        """
    
    if plasticity_cfg.enable_stdp:
        # STDP: LTD when pre-synaptic spike arrives
        on_pre_eqs += r"""
        w_plast = clip(w_plast - A_minus * Apost_post, w_min, w_max)
        """
    
    if plasticity_cfg.enable_tagging:
        on_pre_eqs += r"""
        last_pre_spike = t
        """
    
    # Build on_post equations
    on_post_eqs = ""
    
    if plasticity_cfg.enable_stdp:
        # STDP: LTP when post-synaptic spike arrives
        on_post_eqs += r"""
        w_plast = clip(w_plast + A_plus * Apre_pre, w_min, w_max)
        """
    
    if plasticity_cfg.enable_tagging:
        # Synaptic tagging: check activity levels
        on_post_eqs += r"""
        activity_level = Apre_pre
        tag = int(activity_level > theta_tag) * (1.0 - tag) + tag
        prp = int(activity_level > theta_prp) * (1.0 - prp) + prp
        """
    
    S = Synapses(
        pre,
        post,
        model=model_eqs,
        on_pre=on_pre_eqs if on_pre_eqs else None,
        on_post=on_post_eqs if on_post_eqs else None,
        method="euler",
        name=name,
    )
    
    return S


def initialize_synapses_with_plasticity(
    S: Synapses,
    plasticity_cfg: PlasticityConfig = DEFAULT_PLASTICITY,
) -> None:
    """Initialize plasticity-related synapse variables."""
    
    if plasticity_cfg.enable_stp:
        S.u = plasticity_cfg.stp_u_baseline
        S.R = 1.0
        S.u_baseline = plasticity_cfg.stp_u_baseline
        S.u_max = plasticity_cfg.stp_u_max
        S.tau_stp_decay = plasticity_cfg.stp_tau_decay_ms * ms
        S.tau_stp_rec = plasticity_cfg.stp_tau_rec_ms * ms
    
    if plasticity_cfg.enable_stdp:
        S.w_plast = 1.0
        S.A_plus = plasticity_cfg.stdp_A_plus
        S.A_minus = plasticity_cfg.stdp_A_minus
        S.w_min = plasticity_cfg.stdp_w_min
        S.w_max = plasticity_cfg.stdp_w_max
    
    if plasticity_cfg.enable_tagging:
        S.tag = 0.0
        S.prp = 0.0
        S.tau_tag = plasticity_cfg.tag_tau_tag_min * 60 * second
        S.tau_prp = plasticity_cfg.tag_tau_prp_min * 60 * second
        S.theta_tag = plasticity_cfg.tag_theta_tag
        S.theta_prp = plasticity_cfg.tag_theta_prp
        S.last_pre_spike = 0.0 * second


def make_synapses(pre: NeuronGroup, post: NeuronGroup, name: str) -> Synapses:
    """
    Backward compatible: create synapses without plasticity.
    """
    S = Synapses(
        pre,
        post,
        model=r"""
            w_ampa : siemens
            w_nmda : siemens
            w_gabaa : siemens
            w_gabab : siemens
        """,
        on_pre=r"""
            g_ampa_post += w_ampa
            g_nmda_post += w_nmda
            g_gabaa_post += w_gabaa
            g_gabab_post += w_gabab
        """,
        method="euler",
        name=name,
    )
    return S
