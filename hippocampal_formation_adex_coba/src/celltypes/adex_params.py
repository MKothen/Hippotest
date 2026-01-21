
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class AdExParams:
    # Units are Brian2-friendly (SI base after conversion in sim builder)
    C_pF: float
    gL_nS: float
    EL_mV: float
    VT_mV: float
    DeltaT_mV: float
    a_nS: float
    tau_w_ms: float
    b_pA: float
    Vr_mV: float
    Vcut_mV: float
    t_ref_ms: float


# Canonical AdEx phenotypes (heuristic, literature-inspired; not a claim of a specific dataset fit).
# These are meant as *starting points* and are exposed to config overrides.
PRESETS: Dict[str, AdExParams] = {
    # Regular spiking / pyramidal-like
    "RS": AdExParams(
        C_pF=200.0, gL_nS=10.0, EL_mV=-70.0, VT_mV=-50.0, DeltaT_mV=2.0,
        a_nS=4.0, tau_w_ms=200.0, b_pA=40.0, Vr_mV=-58.0, Vcut_mV=-30.0, t_ref_ms=2.0
    ),
    # Fast spiking / PV-like
    "FS": AdExParams(
        C_pF=100.0, gL_nS=12.0, EL_mV=-65.0, VT_mV=-50.0, DeltaT_mV=0.5,
        a_nS=0.0, tau_w_ms=50.0, b_pA=0.0, Vr_mV=-55.0, Vcut_mV=-30.0, t_ref_ms=1.0
    ),
    # Low-threshold spiking / SOM-like (adapting)
    "LTS": AdExParams(
        C_pF=150.0, gL_nS=10.0, EL_mV=-68.0, VT_mV=-52.0, DeltaT_mV=1.5,
        a_nS=2.0, tau_w_ms=300.0, b_pA=60.0, Vr_mV=-58.0, Vcut_mV=-30.0, t_ref_ms=2.0
    ),
    "DG_GC": AdExParams(
        C_pF=100.0,       # Smaller capacitance (easier to charge)
        gL_nS=10.0,       
        EL_mV=-75.0, 
        VT_mV=-58.0,      # LOWER THRESHOLD (easier to fire)
        DeltaT_mV=2.0,
        a_nS=2.0,         # Less adaptation
        tau_w_ms=150.0, 
        b_pA=10.0,        # Less adaptation spike-triggered
        Vr_mV=-65.0, 
        Vcut_mV=-30.0, 
        t_ref_ms=2.0
    ),
}
