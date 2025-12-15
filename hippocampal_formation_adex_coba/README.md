
# hippo3d — Mouse hippocampal-formation 3D AdEx+COBA spiking network (laptop-feasible)

This project builds and simulates a **3D mouse hippocampal-formation** spiking network with:

- **Neuroanatomy from Allen CCF** (region masks + meshes in atlas space, cached locally)
- **Rule-based laminar organization** (explicit layer sub-volumes per region)
- **Adaptive Exponential I&F (AdEx)** neurons
- **Conductance-based synapses (COBA)**: AMPA, NMDA, GABA\_A, GABA\_B
- **Explicit inhibitory classes** (PV basket-like, SOM/O-LM-like) with **layer-specific targeting rules**
- Config-driven pathway definitions (YAML) implementing the canonical intrinsic pathway structure

> ⚠️ Scope note: this is a **mesoscale, rule-based** model intended to be *biologically plausible and scalable* on local machines. It does **not** claim cell-type–perfect or synapse-type–perfect precision where datasets do not provide direct numeric parameters.

---

## Quickstart

### Option A: conda (recommended)

```bash
conda create -n hippo3d python=3.11 -y
conda activate hippo3d
pip install -r requirements.txt
python scripts/build_and_run.py --config configs/small.yaml
```

### Option B: pip/venv

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_and_run.py --config configs/small.yaml
```

Outputs are written to `runs/<run_name>_<timestamp>/`:
- `viz/scene.html` (interactive 3D region meshes + soma points)
- `plots/raster.png`, `plots/rates.png`
- `connectivity_summary.json`
- `summary.json`
- `data/positions.npz`, `data/sim_outputs.npz`

---

## Configs

- `configs/small.yaml` (default): ~10–15k neurons, sparse pathways (`connectivity.scale=0.12`)
- `configs/medium.yaml`: ~3× neurons, denser (`scale=0.25`)
- `configs/large.yaml`: workstation-scale; uses `cpp_standalone` (Brian2) and much larger neuron counts

Connectivity is defined in `configs/pathways_default.yaml` and is intentionally **separated from code**.

---

## Neuroanatomy + layering

### Regions (minimum required set)

- Dentate gyrus (DG)
- CA3, CA2, CA1
- Subiculum (SUB)
- Entorhinal cortex (EC)

We obtain each parcel by name from the Allen CCF atlas and sample soma positions *inside the corresponding voxel mask*.

### Layers

For each region, we define a small set of key layers as **laminar coordinate bands** in `[0,1]` along a PCA-derived “radial” axis within that parcel.

- This is an **approximation**: it does not reconstruct curved laminar surfaces; it provides an explicit, configurable laminar stratification sufficient for layer-specific targeting rules.

You can edit layer boundaries per region in YAML.

---

## Cell types

### Included types (minimum set)

- Principal excitatory cells: DG granule; CA pyramidal; SUB pyramidal; EC LII/LIII excitatory
- Inhibitory interneurons:
  - **PV basket-like** (perisomatic targeting)
  - **SOM / O-LM-like** (distal dendritic targeting)

The code is designed to be extended (e.g., neurogliaform, axo-axonic, bistratified, IS interneurons).

### Hippocampome integration

`src/data/hippocampome.py` attempts to fetch a cached “priors” file. If a stable JSON endpoint is not accessible, it falls back to a small embedded prior table and **labels that as a fallback**. You can replace/extend priors by dropping a JSON file into the cache folder.

### AdEx parameter mapping

We map neuron firing phenotypes to AdEx presets (`RS`, `FS`, `LTS`) in `src/celltypes/adex_params.py`. These values are **documented starting points** and can be overridden (e.g., via config or code modifications). The mapping is explicitly described as heuristic unless backed by downloaded ephys descriptors.

---

## Connectivity

Connectivity is built by a **distance- and layer-aware** rule engine:

- Each pathway has:
  - outdegree distribution (mean/std)
  - radius + Gaussian distance kernel
  - conduction delay from distance / velocity + base delay
  - a target-layer distribution (explicit layer-specific targeting)
  - synaptic conductance increments (AMPA/NMDA or GABA\_A/GABA\_B)

Included canonical pathways:
- EC→DG (perforant-like)
- DG→CA3 (mossy fiber-like)
- CA3→CA1 (Schaffer collateral-like)
- CA1→SUB
- EC direct projections to CA3/CA2 and to CA1 (temporoammonic-like)

Included inhibition:
- PV→principal (perisomatic layers)
- SOM/O-LM→principal (distal dendritic layers, incl. SLM / DG molecular layers)

---

## Simulation

The default engine is **Brian2**.

- COBA currents are computed from evolving synaptic conductances.
- NMDA includes a simple voltage-dependent Mg block factor.
- Background activity is provided by `PoissonInput` onto AMPA (and optionally NMDA).

---

## Tests

```bash
pytest -q
```

Tests are intentionally small and do not require atlas downloads.

---

## Data sources + caching

- Allen CCF atlas data and meshes are obtained via **BrainGlobe AtlasAPI** and cached in `~/.brainglobe/`.
- Hippocampome integration is treated as “best effort” with caching in `~/.cache/hippo3d/`.

See `src/data/` for downloader/caching logic.

---

## License

Code is MIT-licensed (see `LICENSE`). Data sources retain their original licenses.
