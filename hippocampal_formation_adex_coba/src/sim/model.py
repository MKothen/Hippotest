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


DEFAULT_KINETICS = SynapseKinetics()


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
"""


def make_population(
    name: str,
    n: int,
    adex: AdExParams,
    kinetics: SynapseKinetics = DEFAULT_KINETICS,
    dt_ms: float = 0.1,
    codegen_target: str = "cython",
    seed: int | None = None,
) -> NeuronGroup:
    prefs.codegen.target = codegen_target  # "numpy" or "cython" or "cpp_standalone"

    G = NeuronGroup(
        n,
        model=ADEx_COBA_EQS,
        threshold="v > Vcut",
        reset="v = Vr; w += b",
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

    # initial conditions
    G.v = adex.EL_mV * mV + (np.random.randn(n) * 2.0) * mV
    G.w = 0.0 * pA
    G.g_ampa = 0.0 * nS
    G.g_nmda = 0.0 * nS
    G.g_gabaa = 0.0 * nS
    G.g_gabab = 0.0 * nS
    G.I_ext = 0.0 * pA

    return G


def make_synapses(pre: NeuronGroup, post: NeuronGroup, name: str, enable_stp: bool = True) -> Synapses:
    """
    Create synapses with optional short-term plasticity (STP).
    
    Parameters
    ----------
    pre : NeuronGroup
        Presynaptic neuron group
    post : NeuronGroup
        Postsynaptic neuron group
    name : str
        Name for the synapse object
    enable_stp : bool
        Whether to enable short-term plasticity (Tsodyks-Markram model)
    
    Returns
    -------
    Synapses
        Brian2 Synapses object with STP if enabled
    """
    if enable_stp:
        # Tsodyks-Markram STP model
        # u: utilization of synaptic resources (release probability)
        # R: available resources (fraction)
        # Parameters can be set per-synapse after creation
        S = Synapses(
            pre,
            post,
            model=r"""
            w_ampa : siemens
            w_nmda : siemens
            w_gabaa : siemens
            w_gabab : siemens
            
            # STP variables (Tsodyks-Markram model)
                                    U : 1  # baseline release probability
            tau_rec : second  # recovery time constant
            tau_facil : second  # facilitation time constant
            
            du/dt = -u/tau_facil : 1 (clock-driven)
            dR/dt = (1 - R)/tau_rec : 1 (clock-driven)
            """,
            on_pre=r"""
            u += U * (1 - u)
            r_eff = u * R
            R -= r_eff
            
            g_ampa_post += w_ampa * r_eff
            g_nmda_post += w_nmda * r_eff
            g_gabaa_post += w_gabaa * r_eff
            g_gabab_post += w_gabab * r_eff
            """,
            method="euler",
            name=name,
        )
        # Default STP parameters (will be overridden per pathway)
        S.U = 0.5
        S.tau_rec = 100 * ms
        S.tau_facil = 50 * ms
        S.u = 0.0
        S.R = 1.0
    else:
        # Original static synapses
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
