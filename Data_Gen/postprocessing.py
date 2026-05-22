import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import json
from pathlib import Path

# ------------------- SETTINGS ------------------- #
VISUALIZING = True
INJECTING = False
RANDOM_SEED = 42

# ------------------- PATHS ------------------- #
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent

# Synthetic light curves directory (your generated swarms)
results_dir = base_dir / "results" / "res"  # Change to your swarm type directory

# Quiet baseline light curves from Sector 18
injection_dir = script_dir / "injection_sets"
quiet_baselines_path = injection_dir / "quiet_baselines_sector18.npz"

# Output directory for final injected light curves
output_dir = base_dir / "TechnoNet" / "dyson_positives_new"
output_dir.mkdir(parents=True, exist_ok=True)

# Simulation parameters JSON
# parameters_file_path = results_dir / "simulation_parameters.json"


# ------------------- INITIALIZATION ------------------- #
np.random.seed(RANDOM_SEED)

# Load simulation parameters
# with open(parameters_file_path, "r") as f:
#     params = json.load(f)

# List NPZ synthetic light curves
synth_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".npz")])
print(f"Found {len(synth_files)} synthetic light curves")

# Load quiet baseline light curves
if not quiet_baselines_path.exists():
    raise FileNotFoundError(
        f"Quiet baselines not found at {quiet_baselines_path}\n"
        f"Run your injection star script first to generate quiet_baselines_sector18.npz"
    )

quiet_data = np.load(quiet_baselines_path)
quiet_baselines = [quiet_data[key] for key in quiet_data.files]
print(f"Loaded {len(quiet_baselines)} quiet baseline light curves")

# Load metadata
metadata_path = injection_dir / "quiet_baselines_metadata.npy"
if metadata_path.exists():
    metadata = np.load(metadata_path, allow_pickle=True).item()
    print(f"Baseline variabilities: {[f'{v:.6f}' for v in metadata['variabilities']]}")


def match_dimensions_and_inject(synth_flux, baseline_flux, period_days=None):
    """
    Tiles synthetic signal to match TESS baseline length, preserving cadence.
    
    For Dyson swarms:
    - Synthetic should be ONE orbital period at 2-min cadence
    - This function tiles it to cover the full TESS observation
    - Preserves the transit signature timing
    
    Args:
        synth_flux: Synthetic signal for ONE period (normalized, mean~1)
        baseline_flux: TESS baseline light curve (normalized, mean~1)
        period_days: Orbital period in days (optional, for verification)
        
    Returns:
        injected_flux: Injected light curve at TESS cadence
    """
    if synth_flux.ndim == 2:
        # Average across Monte Carlo trials or particles
        synth_flux = synth_flux.mean(axis=1)  # Now shape (308,)
        print(f"  Averaged 2D flux to 1D, shape: {synth_flux.shape}")
    
    synth_flux = np.asarray(synth_flux).flatten()
    baseline_flux = np.asarray(baseline_flux).flatten()
    
    len_synth = len(synth_flux)
    len_baseline = len(baseline_flux)
    
    # If synthetic is already the right length, just inject
    if len_synth == len_baseline:
        injected = baseline_flux * synth_flux
        injected = injected / np.mean(injected)
        return injected
    
    # If synthetic is longer than baseline (shouldn't happen), interpolate down
    if len_synth > len_baseline:
        print(f"  Warning: Synthetic ({len_synth}) longer than baseline ({len_baseline}), interpolating down")
        t_synth = np.linspace(0, 1, len_synth)
        t_baseline = np.linspace(0, 1, len_baseline)
        interp_func = interp1d(t_synth, synth_flux, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        synth_resampled = interp_func(t_baseline)
        injected = baseline_flux * synth_resampled
        injected = injected / np.mean(injected)
        return injected
    
    # MAIN CASE: Tile synthetic to match baseline length
    # This preserves the cadence and repeats the orbital pattern
    
    # Calculate how many full periods fit
    n_full_periods = len_baseline // len_synth
    remainder = len_baseline % len_synth
    
    # Tile the synthetic signal
    synth_tiled = np.tile(synth_flux, n_full_periods)
    
    # Add partial period at the end if needed
    if remainder > 0:
        synth_tiled = np.concatenate([synth_tiled, synth_flux[:remainder]])
    
    # Verify length matches
    assert len(synth_tiled) == len_baseline, \
        f"Tiling failed: got {len(synth_tiled)}, expected {len_baseline}"
    
    # Multiplicative injection
    injected = baseline_flux * synth_tiled
    
    # Renormalize
    injected = injected / np.mean(injected)
    
    return injected

# ------------------- MAIN INJECTION LOOP ------------------- #
if INJECTING:
    print(f"\n{'='*60}")
    print(f"INJECTING SYNTHETIC SWARMS INTO QUIET BASELINES")
    print(f"{'='*60}")
    print(f"Synthetic swarms: {len(synth_files)}")
    print(f"Quiet baselines: {len(quiet_baselines)}")
    print(f"Total injected light curves: {len(synth_files) * len(quiet_baselines)}")
    print(f"{'='*60}\n")
    
    injection_count = 0
    
    for swarm_idx, synth_file in enumerate(synth_files):
        # Load synthetic swarm signal
        synth_data = np.load(results_dir / synth_file)
        synth_times = synth_data["times"]
        synth_flux = synth_data["flux"]

        synth_data = np.load(results_dir / synth_file)
        print(f"Keys in file: {synth_data.files}")
        print(f"synth_times shape: {synth_data['times'].shape}")
        print(f"synth_flux shape: {synth_data['flux'].shape}")
        
        if np.allclose(synth_flux, 1.0, atol=1e-5):
            print(f"  ⊘ Skipping {synth_file}: no observable transit (geometry not aligned)")
            continue

        swarm_base_name = Path(synth_file).stem
        
        # Inject into ALL quiet baselines
        for baseline_idx, baseline_flux in enumerate(quiet_baselines):
            
            # Match dimensions and inject
            injected_flux = match_dimensions_and_inject(synth_flux, baseline_flux)
            
            # Create output filename
            output_name = f"dyson_{swarm_base_name}_baseline{baseline_idx:02d}.npz"
            output_path = output_dir / output_name
            
            # Save injected light curve
            # Use baseline's time array (or create normalized one)
            time_array = np.arange(len(injected_flux))  # Simple index-based time
            
            np.savez_compressed(
                output_path,
                flux=injected_flux.astype(np.float32),
                time=time_array.astype(np.float32),
                metadata={
                    'swarm_file': synth_file,
                    'baseline_idx': baseline_idx,
                    'swarm_idx': swarm_idx,
                    'baseline_variability': metadata['variabilities'][baseline_idx] if metadata_path.exists() else None
                }
            )
            
            injection_count += 1
        
        # Progress update every 10 swarms
        if (swarm_idx + 1) % 10 == 0:
            print(f"  Processed {swarm_idx + 1}/{len(synth_files)} swarms "
                  f"({injection_count} total injections)")
    
    print(f"\n{'='*60}")
    print(f"INJECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total injected light curves: {injection_count}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")


# ------------------- VISUALIZATION ------------------- #
if VISUALIZING:
    print("\nVisualizing sample injections...")
    
    # Load a few examples
    injected_files = sorted(list(output_dir.glob("*.npz")))[:5]
    
    if len(injected_files) == 0:
        print("No injected files found to visualize")
    else:
        fig, axes = plt.subplots(len(injected_files), 1, 
                                figsize=(12, 3*len(injected_files)))
        
        if len(injected_files) == 1:
            axes = [axes]
        
        for i, injected_file in enumerate(injected_files):
            data = np.load(injected_file, allow_pickle=True)
            flux = data['flux']
            time = data['time']
            
            axes[i].plot(time, flux, linewidth=0.5, alpha=0.8)
            axes[i].set_ylabel("Normalized Flux")
            axes[i].set_title(f"{injected_file.name}")
            axes[i].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, linewidth=1)
            axes[i].grid(alpha=0.3)
        
        axes[-1].set_xlabel("Time (cadences)")
        plt.tight_layout()
        plt.savefig(output_dir / "sample_injections.png", dpi=150)
        print(f"✓ Saved visualization to {output_dir / 'sample_injections.png'}")
        plt.show()