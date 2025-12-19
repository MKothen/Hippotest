# Plasticity Integration Guide

This document outlines the plasticty integration work completed for the Hippotest simulation.

## ✅ Completed Changes

### 1. Model.py - STP Integration (COMPLETED)

**File**: `hippocampal_formation_adex_coba/src/sim/model.py`

**Changes Made**:
- Added `enable_stp` parameter to `make_synapses()` function (default: True)
- Implemented Tsodyks-Markram STP model with:
  - `u`: utilization/release probability (dynamic)
  - `R`: available synaptic resources (0-1, dynamic)
  - `U`: baseline release probability (constant per synapse)
  - `tau_rec`: recovery time constant
  - `tau_facil`: facilitation time constant
- Modified `on_pre` to modulate transmission by `u * R`
- Set default STP parameters: U=0.5, tau_rec=100ms, tau_facil=50ms

**Commit**: "Integrate STP plasticity into model with Tsodyks-Markram dynamics"

### 2. Runner.py - Import Added (COMPLETED)

**File**: `hippocampal_formation_adex_coba/src/sim/runner.py`

**Changes Made**:
- Added import on line 31: `from .plasticity_plots import save_plasticity_overview`

**Commit**: "Add plasticity plotting import to runner"

## 🔄 Remaining Work

To fully integrate plasticity recording and plotting, you need to make the following changes to `runner.py`:

### Step 1: Add Plasticity Monitors

**Location**: Around line 297-312 (where spike_monitors and state_monitors are created)

**Add this code block after line 312**:

```python
# ---- plasticity monitors (for STP dynamics)
plasticity_monitors: Dict[str, StateMonitor] = {}
rec_plasticity = rec_cfg.get("record_plasticity", True)  # Enable by default

if rec_plasticity:
    for pname, S in zip(edges.keys(), synapses):
        # Only monitor if synapse has STP variables
        if hasattr(S, 'u') and hasattr(S, 'R'):
            # Record from a subset of synapses to save memory
            n_rec = min(50, len(S))  # Record up to 50 synapses per pathway
            if n_rec > 0:
                idx = np.linspace(0, len(S)-1, n_rec, dtype=int) if len(S) > 0 else []
                plasticity_monitors[pname] = StateMonitor(
                    S,
                    variables=['u', 'R'],
                    record=idx,
                    name=f"plast_{pname}"
                )
```

### Step 2: Add Monitors to Network

**Location**: Around line 315-325 (where network objects are added)

**Modify the net.add() section** to include plasticity_monitors:

```python
net = Network()
for obj in (
    list(groups.values())
    + replay_synapses
    + synapses
    + list(spike_monitors.values())
    + list(state_monitors.values())
    + list(plasticity_monitors.values())  # ADD THIS LINE
    + poisson_inputs
):
    net.add(obj)
```

### Step 3: Collect Plasticity Data

**Location**: Around line 365-390 (after spike and state trace collection)

**Add this code block**:

```python
# ---- collect plasticity traces
plasticity_data: Dict[str, Dict[str, np.ndarray]] = {}
for pname, mon in plasticity_monitors.items():
    if len(mon.t) > 0:
        t_s = _as_float_array(mon.t / second)
        plasticity_data[pname] = {
            't_s': t_s,
            'u': np.asarray(mon.u, dtype=float),  # shape: (n_synapses, n_timepoints)
            'R': np.asarray(mon.R, dtype=float),
        }
        # Also store initial values for comparison
        if pname in edges:
            pathway_idx = list(edges.keys()).index(pname)
            S = synapses[pathway_idx]
            if hasattr(S, 'U'):
                # Get baseline U values
                plasticity_data[pname]['U_baseline'] = np.asarray(S.U, dtype=float)
```

### Step 4: Save Plasticity Plots

**Location**: Around line 415-425 (after save_activity_figure is called)

**Add this code block**:

```python
# Save plasticity overview if data was recorded
if plasticity_data:
    save_plasticity_overview(
        plasticity_data=plasticity_data,
        out_path=out_dir / "plots" / "plasticity_overview.png",
        max_pathways=6
    )
```

### Step 5: Save Plasticity Data to NPZ

**Location**: Around line 430-445 (where npz dictionary is built)

**Add this code in the npz building section**:

```python
# Add plasticity data to npz
for pname, plast in plasticity_data.items():
    npz[f"{pname}_plast_t_s"] = plast['t_s']
    npz[f"{pname}_plast_u"] = plast['u']
    npz[f"{pname}_plast_R"] = plast['R']
    if 'U_baseline' in plast:
        npz[f"{pname}_plast_U_baseline"] = plast['U_baseline']
```

## 📊 Expected Output

After completing these changes, simulations will produce:

1. **activity_overview.png**: Existing spiking activity visualization
2. **plasticity_overview.png**: NEW - Shows STP dynamics for each pathway:
   - Release probability (u) over time
   - Available resources (R) over time
   - Effective release (u × R) over time

3. **sim_outputs.npz**: Will now include plasticity variables:
   - `<pathway>_plast_t_s`: Time points
   - `<pathway>_plast_u`: Release probability traces
   - `<pathway>_plast_R`: Resource availability traces
   - `<pathway>_plast_U_baseline`: Baseline U values

## 🔧 Testing

After making these changes:

1. Run a simulation:
   ```bash
   python -m hippocampal_formation_adex_coba.src.sim.runner
   ```

2. Check output directory for:
   - `plots/plasticity_overview.png`
   - Plasticity data in `data/sim_outputs.npz`

3. Verify STP is working by examining:
   - u values increase with spikes (facilitation)
   - R values decrease with spikes (depression)
   - Recovery during silent periods

## 📝 Notes

- STP is enabled by default (`enable_stp=True` in `make_synapses()`)
- Default parameters are generic; you may want to customize per pathway
- Memory usage scales with number of recorded synapses (currently max 50 per pathway)
- The plasticity plotting functions support multiple pathways and automatically handle different array shapes

## 🎯 Next Steps (Optional Enhancements)

1. **Pathway-specific STP parameters**: Modify synapses after creation to set biologically realistic U, tau_rec, tau_facil per pathway type
2. **STDP integration**: Add spike-timing dependent plasticity using similar monitoring approach
3. **Weight evolution plots**: Track actual weight changes if implementing long-term plasticity
4. **Calcium-dependent plasticity**: Integrate bidirectional_plasticity function from plasticity.py
