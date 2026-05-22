# TechnoNet — Research Codebase

Overview
--------

TechnoNet is a project to evaluate the ability of ML models to detect Dyson Swarms from light curve data. There are two ML models: A Temporal Autoencoder and a Temporal Convolutional Network. To gather Dyson Swarm Light curve data, a novel simulation algorithm is implemented using the REBOUND N-body integrator developed by Hanno Rein and a Monte Carlo ray-tracing algorithm inspired by Bhowmick & Khaire, 2024.

Goals
-----
- Provide reproducible pipelines for generating synthetic light curves with injected signals.
- Run swarm/ring simulations and compute observational results.
- Train and evaluate classifiers and signal detectors (TCN and other models).
- Organize simulation parameters, results, and model checkpoints.

Repository layout
-----------------
Top-level files and directories (high-level summary):

- `misc.py` — miscellaneous helpers and small utilities referenced across scripts.
- `setup.sh` — environment / setup helper script (project-specific setup steps).
- `test.py` — quick tests / smoke checks used during development.
- `swarm_visualizer.py` — scripts to visualize swarm geometries and resulting signals.
- `swarm_tests.py` — integration or scenario tests for swarms.
- `checkpoints/` — pretrained model weights and training checkpoints (`*.pth` files).
- `config/Params.json` — default parameter file for simulations and experiments.
- `Data_Gen/` — data generation utilities and injection sets:
  - `light_curves.py` — functions used to build and manipulate synthetic light curves.
  - `postprocessing.py` — tools to process generated light curves and outputs.
  - `injection_sets/` — stored baseline files and injection metadata (npz, npy).
- `other/` — supporting experiments, one-off scripts, notebooks and experiment code.
- `results/` — main results output directory. Subfolders include `multi/`, `res/`, `thin/` and JSON parameter files describing each experiment run.
- `Simulation/` — the main simulator code (e.g., `sim.py`) and supporting utilities used to create swarm geometries and simulate their observational signatures.
- `swarms/` — specific swarm models and scenario definitions (e.g., `Diffuse.py`, `thin_ring.py`, `multi_ring.py`, `Resonant.py`).
- `TechnoNet/` — model code (training, preprocessing, helpers) used for classifier training, inference, and data caches.
- `temp_injection_lcs/` — temporary working area for injections and downloads (e.g., `mastDownload/`).

Key datasets and artifacts
-------------------------
- `Data_Gen/injection_sets/` contains baseline light curve datasets used as foundations for injecting synthetic signals.
- `TechnoNet/cached_lc*` and `TechnoNet/data/` contain prepared input datasets and caches used for model training and validation.
- `checkpoints/` stores PyTorch model weights (`*.pth`) from experiments (e.g., `best_TCN_Classifier_model.pth`).
- `results/` stores per-experiment `*.npz` result files with metrics, outputs, and `simulation_parameters.json` that describe experiment configuration.

Quickstart (environment & dependencies)
--------------------------------------
1. Python: use Python 3.8+ (3.10 or 3.11 recommended). Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (recommended). If the project does not include a `requirements.txt`, install the common scientific packages used by this repo:

```bash
pip install numpy scipy matplotlib pandas tqdm h5py jax astroquery lightkurve rebound scikit-learn
pip install torch torchvision  # if using GPU, install the appropriate CUDA build
```

3. Run `setup.sh` if present to perform any repo-specific setup steps (it may install or configure dependencies):

```bash
./setup.sh
```

Running simulations and generating data
--------------------------------------
- Run a full swarm simulation from `Simulation/sim.py`. Example usage (adapt arguments based on the script's CLI):

```bash
python Simulation/sim.py --config config/Params.json
```

- Generate or preprocess light curves using `Data_Gen/light_curves.py` functions. Example (import from Python):

```python
from Data_Gen.light_curves import generate_light_curve
lc = generate_light_curve(params)
```

- Use `swarms/` module classes to build swarm geometries. For example, import `thin_ring` or `Diffuse` to create specific scenarios.

Training and inference (models)
------------------------------
- The `TechnoNet/` folder contains model code and training utilities. Pretrained weights are in `checkpoints/`.
- Typical workflow:
  1. Prepare datasets in `TechnoNet/data/` or use cached datasets in `TechnoNet/cached_lc_train` and `TechnoNet/cached_lc_val`.
  2. Launch training scripts located under `TechnoNet/` (there may be a `setup.py` or dedicated train script).
  3. Save and load models using PyTorch `torch.save()` / `torch.load()`; checkpoints in `checkpoints/` follow this convention.

Running analysis and visualization
---------------------------------
- Quick evaluation: run `test.py` to perform smoke checks and simple evaluations.
- Visualize swarm geometry or light curves with `swarm_visualizer.py` (it contains plotting code using `matplotlib`).
- Run `swarm_tests.py` to execute test scenarios or batch experiments.

Results and interpreting files
------------------------------
- `results/*/*.npz` files contain arrays and numeric outputs produced by simulations and analysis. Use `numpy.load()` to inspect them:

```python
import numpy as np
d = np.load('results/multi/res_104.npz')
print(list(d.keys()))
```

- `simulation_parameters.json` files in each results folder record the configuration used for that experiment; keep them alongside results for reproducibility.

Best practices for reproducible experiments
-----------------------------------------
- Keep a copy of the `Params.json` used for each run and store it in the results folder.
- Record the git commit hash (e.g., `git rev-parse HEAD`) in your experiment metadata.
- Use the `checkpoints/` directory to version model weights and map them to experiment IDs in `results/`.

Common workflows and examples
----------------------------
1. Reproduce a published experiment:
   - Locate the corresponding `results/*/simulation_parameters.json` file.
   - Run `Simulation/sim.py` or the dedicated experiment script with those parameters.
   - Use the appropriate model checkpoint from `checkpoints/` for evaluation.

2. Train a new model:
   - Prepare training/validation datasets under `TechnoNet/final_training_data_TCN_train` and `TechnoNet/final_training_data_TCN_val`.
   - Run the training script in `TechnoNet/` (check for `train.py` or instructions in `TechnoNet/README` if present).
   - Save model weights to `checkpoints/` and track hyperparameters in a JSON file.

3. Quick visualization:
   - Use `swarm_visualizer.py` to plot geometry and light curves for a given simulation output file.

Testing and debugging
---------------------
- Use `test.py` and `swarm_tests.py` to run unit-like and scenario tests. Those scripts are intended as development tools; examine them to see expected inputs and outputs.
- If a script fails due to missing packages, install the relevant package (see Quickstart). If a file path is missing, confirm dataset locations under `TechnoNet/` and `Data_Gen/injection_sets/`.

Notes on file formats
---------------------
- `.npz` — compressed NumPy archive used for storing arrays and experiment outputs.
- `.npy` — single NumPy array files (often baselines or metadata arrays).
- `.pth` — PyTorch model checkpoint files.

Where to start (recommended)
----------------------------
1. Run `./setup.sh` to set up environment-specific dependencies.
2. Inspect `config/Params.json` to learn base parameters and defaults.
3. Run a small simulation: `python Simulation/sim.py --config config/Params.json`.
4. Visualize outputs with `python swarm_visualizer.py`.

License & contact
-----------------
This repository is licensed under the MIT License. See the `LICENSE` file for details. For questions about the code, contact the repository maintainer or the original research authors.

