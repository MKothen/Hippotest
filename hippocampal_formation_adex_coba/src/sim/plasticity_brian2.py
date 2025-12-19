"""Brian2-compatible plasticity mechanisms.

This module provides Brian2 equation strings and helper functions for
integrating synaptic plasticity into the hippocampal network simulation.
Supports:
- Short-term plasticity (STP) with facilitation and depression
- Calcium-based long-term plasticity with bidirectional rules
- BTSP-inspired timing-dependent plasticity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from brian2 import Synapses, NeuronGroup, ms, nS, second


@dataclass
class PlasticityConfig:
    """Configuration for plasticity mechanisms."""
    
    # STP parameters
    stp_enabled: bool = True
    U_baseline: float = 0.3  # baseline utilization (release probability)
    tau_rec_ms: float = 800.0  # recovery time constant (depression)
    tau_facil_ms: float = 50.0  # facilitation time constant
    
    # Calcium-based LTP/LTD parameters
    calcium_plasticity_enabled: bool = False
    tau_ca_ms: float = 50.0  # calcium decay time
    theta_ltp: float = 1.5  # LTP threshold (μM)
    theta_ltd: float = 0.8  # LTD threshold (μM)
    eta_ltp: float = 0.001  # LTP learning rate
    eta_ltd: float = 0.0005  # LTD learning rate
    w_min_nS: float = 0.0  # minimum weight
    w_max_nS: float = 5.0  # maximum weight
    
    # BTSP parameters
    btsp_enabled: bool = False
    tau_btsp_ms: float = 200.0  # BTSP eligibility trace decay
    btsp_window_pre_ms: float = -200.0  # pre-before-post window
    btsp_window_post_ms: float = 200.0  # post-before-pre window
    btsp_learning_rate: float = 0.0001
    
    # Homeostatic scaling
    homeostatic_scaling: bool = False
    target_rate_hz: float = 5.0
    tau_homeo_s: float = 3600.0  # 1 hour
    
    # Monitoring
    monitor_plasticity: bool = True
    n_synapses_to_monitor: int = 50


def make_plastic_synapses_stp(
    pre: NeuronGroup,
    post: NeuronGroup,
    config: PlasticityConfig,
    name: str
) -> Synapses:
    """
    Create synapses with short-term plasticity (Tsodyks-Markram model).
    
    The model includes:
    - u: running release probability (facilitation variable)
    - x: fraction of resources available (depression variable)
    
    Parameters
    ----------
    pre : NeuronGroup
        Presynaptic neuron group
    post : NeuronGroup
        Postsynaptic neuron group
    config : PlasticityConfig
        Plasticity configuration parameters
    name : str
        Name for the synapse group
        
    Returns
    -------
    Synapses
        Brian2 Synapses object with STP dynamics
    """
    
    model = f"""
    # Static weight parameters (can be scaled by x*u for STP)
    w_ampa_base : siemens
    w_nmda_base : siemens
    w_gabaa_base : siemens
    w_gabab_base : siemens
    
    # Short-term plasticity variables (Tsodyks-Markram)
    u : 1  # running release probability (facilitation)
    x : 1  # available resources (depression)
    
    # Effective weights modulated by STP
    w_ampa = w_ampa_base * x * u : siemens
    w_nmda = w_nmda_base * x * u : siemens
    w_gabaa = w_gabaa_base * x * u : siemens
    w_gabab = w_gabab_base * x * u : siemens
    
    # STP dynamics
    du/dt = -(u - {config.U_baseline}) / ({config.tau_facil_ms}*ms) : 1 (clock-driven)
    dx/dt = (1 - x) / ({config.tau_rec_ms}*ms) : 1 (clock-driven)
    """
    
    # On presynaptic spike: update u and x, then trigger conductance change
    on_pre = f"""
    u += {config.U_baseline} * (1 - u)  # facilitation increment
    x_release = x * u  # fraction released
    x -= x_release  # depletion
    
    # Update postsynaptic conductances with effective weight
    g_ampa_post += w_ampa_base * x_release * u
    g_nmda_post += w_nmda_base * x_release * u
    g_gabaa_post += w_gabaa_base * x_release * u
    g_gabab_post += w_gabab_base * x_release * u
    """
    
    S = Synapses(
        pre,
        post,
        model=model,
        on_pre=on_pre,
        method="euler",
        name=name,
    )
    
    return S


def make_plastic_synapses_calcium(
    pre: NeuronGroup,
    post: NeuronGroup,
    config: PlasticityConfig,
    name: str
) -> Synapses:
    """
    Create synapses with calcium-dependent long-term plasticity.
    
    Implements bidirectional plasticity based on postsynaptic calcium:
    - High Ca²⁺ (> theta_ltp) → LTP
    - Intermediate Ca²⁺ (< theta_ltd) → LTD
    - Between thresholds → no change
    
    Parameters
    ----------
    pre : NeuronGroup
        Presynaptic neuron group
    post : NeuronGroup
        Postsynaptic neuron group
    config : PlasticityConfig
        Plasticity configuration parameters
    name : str
        Name for the synapse group
        
    Returns
    -------
    Synapses
        Brian2 Synapses object with calcium-based plasticity
    """
    
    # Add calcium dynamics to postsynaptic group if not present
    post_has_calcium = hasattr(post, 'Ca')
    
    model = f"""
    # Plastic weights
    w_ampa : siemens
    w_nmda : siemens
    w_gabaa : siemens
    w_gabab : siemens
    
    # Last spike times for STDP-like timing
    t_last_pre : second
    t_last_post : second
    """
    
    # Calcium dynamics (postsynaptic)
    if not post_has_calcium:
        # Add calcium variable to model
        model += f"""
        Ca_post : 1 (summed)  # postsynaptic calcium concentration
        """
    
    on_pre = f"""
    # Update conductances
    g_ampa_post += w_ampa
    g_nmda_post += w_nmda
    g_gabaa_post += w_gabaa
    g_gabab_post += w_gabab
    
    # Calcium influx from spike
    Ca_post += 0.5
    
    # Store spike time
    t_last_pre = t
    """
    
    # Calcium-dependent weight update (evaluated after spikes)
    on_post = f"""
    Ca_post += 1.0  # larger calcium from postsynaptic spike
    t_last_post = t
    
    # Bidirectional plasticity based on calcium level
    w_ampa = clip(w_ampa + {config.eta_ltp}*nS * (Ca_post > {config.theta_ltp}) - 
                           {config.eta_ltd}*nS * (Ca_post < {config.theta_ltd} and Ca_post > 0.1),
                  {config.w_min_nS}*nS, {config.w_max_nS}*nS)
    """
    
    # Add calcium decay
    model += f"""
    dCa_post/dt = -Ca_post / ({config.tau_ca_ms}*ms) : 1 (clock-driven)
    """
    
    S = Synapses(
        pre,
        post,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="euler",
        name=name,
    )
    
    return S


def make_plastic_synapses_combined(
    pre: NeuronGroup,
    post: NeuronGroup,
    config: PlasticityConfig,
    name: str
) -> Synapses:
    """
    Create synapses with both STP and calcium-based LTP/LTD.
    
    Combines:
    - Short-term facilitation/depression
    - Long-term calcium-dependent weight changes
    
    Parameters
    ----------
    pre : NeuronGroup
        Presynaptic neuron group
    post : NeuronGroup
        Postsynaptic neuron group
    config : PlasticityConfig
        Plasticity configuration parameters
    name : str
        Name for the synapse group
        
    Returns
    -------
    Synapses
        Brian2 Synapses object with combined plasticity
    """
    
    model = f"""
    # Long-term plastic weights (slowly changing)
    w_ampa_lt : siemens
    w_nmda_lt : siemens
    w_gabaa_lt : siemens
    w_gabab_lt : siemens
    
    # STP variables
    u : 1  # facilitation variable
    x : 1  # depression variable
    
    # Effective weights (LT weight modulated by STP)
    w_ampa = w_ampa_lt * x * u : siemens
    w_nmda = w_nmda_lt * x * u : siemens
    w_gabaa = w_gabaa_lt * x * u : siemens
    w_gabab = w_gabab_lt * x * u : siemens
    
    # Calcium for LTP/LTD
    Ca_post : 1 (summed)
    
    # STP dynamics
    du/dt = -(u - {config.U_baseline}) / ({config.tau_facil_ms}*ms) : 1 (clock-driven)
    dx/dt = (1 - x) / ({config.tau_rec_ms}*ms) : 1 (clock-driven)
    
    # Calcium decay
    dCa_post/dt = -Ca_post / ({config.tau_ca_ms}*ms) : 1 (clock-driven)
    """
    
    on_pre = f"""
    # STP: facilitation and depression
    u += {config.U_baseline} * (1 - u)
    x_release = x * u
    x -= x_release
    
    # Deliver conductance change
    g_ampa_post += w_ampa_lt * x_release * u
    g_nmda_post += w_nmda_lt * x_release * u
    g_gabaa_post += w_gabaa_lt * x_release * u
    g_gabab_post += w_gabab_lt * x_release * u
    
    # Calcium influx
    Ca_post += 0.5
    """
    
    on_post = f"""
    # Postsynaptic spike causes larger calcium influx
    Ca_post += 1.0
    
    # Calcium-dependent plasticity (only for excitatory synapses)
    w_ampa_lt = clip(w_ampa_lt + {config.eta_ltp}*nS * (Ca_post > {config.theta_ltp}) - 
                                  {config.eta_ltd}*nS * (Ca_post < {config.theta_ltd} and Ca_post > 0.1),
                     {config.w_min_nS}*nS, {config.w_max_nS}*nS)
    """
    
    S = Synapses(
        pre,
        post,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="euler",
        name=name,
    )
    
    return S


def initialize_stp_state(synapses: Synapses, config: PlasticityConfig) -> None:
    """Initialize STP variables to baseline values."""
    if hasattr(synapses, 'u'):
        synapses.u = config.U_baseline
    if hasattr(synapses, 'x'):
        synapses.x = 1.0  # fully recovered initially


def configure_plasticity_from_dict(config_dict: Dict[str, Any]) -> PlasticityConfig:
    """Create PlasticityConfig from dictionary (e.g., from YAML)."""
    plasticity_params = config_dict.get('plasticity', {})
    return PlasticityConfig(**plasticity_params)


__all__ = [
    'PlasticityConfig',
    'make_plastic_synapses_stp',
    'make_plastic_synapses_calcium',
    'make_plastic_synapses_combined',
    'initialize_stp_state',
    'configure_plasticity_from_dict',
]
