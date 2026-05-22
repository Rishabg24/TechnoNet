import numpy as np
import json
from Simulation.utilities import mathematicals, coeffs
from Simulation.utilities.mathematicals import draw_chabrier_mass, mass_to_radius
import os
import sys # For error handling
from swarms import thin_ring, multi_ring, Diffuse, Resonant
from Simulation.utilities import Monte_carlo
import jax
import jax.numpy as jnp
from scipy import stats
from tqdm import tqdm

np.random.seed(42)

# --- Robust Path and Module Handling ---
try:
    # Assumes sim.py is in a 'Simulation' folder, and 'config' is in the parent dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '..', 'config', 'Params.json')
    with open(config_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file not found at {config_path}")
    print("Please ensure 'config/Params.json' exists in the project root directory.")
    sys.exit(1)
except ImportError:
    print("Error: Could not import swarm modules. Ensure 'swarms/thin_ring.py', etc. exist.")
    sys.exit(1)


def clip_to_bounds(val, bounds):
    """Clip a value to within given min/max bounds."""
    return np.clip(val, bounds["min"], bounds["max"])

def random_star_params(star_class):
    """Generate stellar parameters and limb-darkening coefficients."""
    conf = data["star type"][star_class]

    if star_class == "White_Dwarf":
        print("Warning: White Dwarf branch not fully implemented. It is incorrect as of now")
        mass, radius, T_eff = 0.6, 0.01, 10000
        u1, u2 = 0.1, 0.1
    else:
        m_bounds = conf["mass"]
        mass = draw_chabrier_mass(m_bounds["min"], m_bounds["max"])
        radius = mass_to_radius(mass)

        try:
            T_eff = mathematicals.compute_effective_temp(mass)
            T_eff = clip_to_bounds(T_eff, conf["Teff"])
        except ValueError:
            T_eff = np.mean([conf["Teff"]["min"], conf["Teff"]["max"]])

        try:
            u1, u2 = coeffs.sample_ld_main_coeffs(
                mass=mass,
                radius=radius,
                teff=T_eff,
                feh=conf["LimbDarkening"]["FeH"],
                vturb=conf["LimbDarkening"]["vturb_kms"],
            )
        except (AttributeError, ImportError, NameError):
            print("Warning: 'coeffs' module not found. Using dummy u1, u2 = (0.3, 0.1)")
            u1, u2 = 0.3, 0.1

    return mass, radius, u1, u2

def random_star_picker():
    star_classes = ["M_class", "K_class", "G_class"]
    weights = [0.76, 0.12, 0.08]
    return np.random.choice(star_classes, p=np.array(weights) / np.sum(weights))

def pick_swarm(type_sim: str, **kwargs):
    """
    Generic dispatcher for swarm generation.
    This will pass the correct 'inc_deg' or 'starting_inclination'
    based on the 'inclination' parameter from the main script.
    """
    try:
        match type_sim:
            case "thin":
                sim = thin_ring.make_thin_ring(
                    num_particles=kwargs["num_particles"],
                    a_AU=kwargs["AU"],
                    particle_radius=kwargs["particle_radius"],
                    ecc_max=kwargs.get("ecc_max", 0.0),
                    ecc_alpha=kwargs.get("ecc_alpha", 0.0),
                    inc_deg=kwargs.get("inc_deg", 89.0), # Use passed 'inclination'
                    star_mass=kwargs.get("star_mass", 1.0),
                    seed=kwargs.get("seed", None),
                )
            case "multi":
                sim = multi_ring.make_multi_ring(
                    num_particles_per_ring=kwargs["num_particles_per_ring"],
                    num_rings=kwargs["num_rings"],
                    ring_sep_AU=kwargs["ring_sep_AU"],
                    inc_dispersion_deg=kwargs.get("inc_dispersion_deg", 1.0),
                    particle_radius=kwargs["particle_radius"],
                    a_min_AU=kwargs["a_min_AU"],
                    starting_inclination=kwargs.get("starting_inclination", 89.0), # Use passed 'inclination'
                    ecc_alpha=kwargs.get("ecc_alpha", 0.0),
                    ecc_max=kwargs.get("ecc_max", 0.0),
                    star_mass=kwargs.get("star_mass", 1.0),
                    seed=kwargs.get("seed", None),
                )
            case "res":
                sim = Resonant.make_resonant_clumps(
                    num_particles_per_clump=kwargs["num_particles_per_clump"],
                    num_clumps=kwargs["num_clumps"],
                    a_AU=kwargs["a_AU"],
                    particle_radius=kwargs["particle_radius"],
                    inc_deg=kwargs.get("inc_deg", 89.0), # Use passed 'inclination'
                    clump_width=kwargs.get("clump_width", 0.05),
                    star_mass=kwargs.get("star_mass", 1.0),
                    ecc_alpha=kwargs.get("ecc_alpha", 0.0),
                    ecc_max=kwargs.get("ecc_max", 0.0),
                    seed=kwargs.get("seed", None),
                )
            case _:
                raise ValueError(f"Unknown swarm type: {type_sim}")
                
    except (NameError, AttributeError) as e:
        print(f"Error: Missing swarm module? (e.g., 'thin_ring', 'multi_ring', 'Resonant')")
        print(f"Details: {e}")
        raise
    return sim


def batched_flux(
    positions_all, radii_all, star_Radius, u1, u2, x_s_all, y_s_all, mu_all, observer
):
    """Wrapper for the JAX-based Monte Carlo flux calculation."""
    return Monte_carlo.compute_monte_carlo_flux(
        star_Radius,
        positions_all,
        radii_all,
        u1,
        u2,
        x_s_all,
        y_s_all,
        mu_all,
        observer,
    )

def warmup(positions_all):
    """Run a small dummy calculation to JIT-compile the JAX functions."""
    print("Warming up JAX kernel...")
    n_part_sample = min(128, positions_all.shape[1])
    dummy_pos = jnp.zeros((1, n_part_sample, 3), dtype=jnp.float32)
    dummy_r = jnp.zeros((n_part_sample,), dtype=jnp.float32)
    n_trial_sample, n_samp_sample = 2, 64
    dummy_x = jnp.zeros((n_trial_sample, n_samp_sample), dtype=jnp.float32)
    dummy_y = jnp.zeros((n_trial_sample, n_samp_sample), dtype=jnp.float32)
    dummy_mu = jnp.ones((n_trial_sample, n_samp_sample), dtype=jnp.float32)
    dummy_obs = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    
    _ = Monte_carlo.compute_monte_carlo_flux(
        1.0, dummy_pos, dummy_r, 0.1, 0.1, dummy_x, dummy_y, dummy_mu, dummy_obs
    ).block_until_ready()
    print("Warmup complete.")

# ===================
# Simulation Settings
# ===================
PARTICLE_COUNT_LOW = 10000
PARTICLE_COUNT_HIGH = 50000
NUM_LIGHT_CURVES = 1
TESS = True
VERBOSE = 1
NUM_MONTE_CARLO_TRIALS = 100       # Number of 'runs' for MC averaging
NUM_MONTE_CARLO_SAMPLES = int(5e4) # Points per star disk per run

# --- Select which swarm to run ---
# archetypes[0] = "thin"
# archetypes[1] = "multi"
# archetypes[2] = "res"
archetypes = ["thin", "multi", "res"]
SWARM_TO_RUN = archetypes[2] # <<< SET THIS TO 0, 1, or 2

master_key = jax.random.PRNGKey(42)

# =========================
# Beginning Data Collection
# =========================
print(f"Starting data generation for {NUM_LIGHT_CURVES} light curve(s)...")

for lc_idx in range(NUM_LIGHT_CURVES):
    print(f"\n--- Generating Light Curve {lc_idx + 1} / {NUM_LIGHT_CURVES} ---")

    PARTICLES = int(stats.loguniform.rvs(PARTICLE_COUNT_LOW, PARTICLE_COUNT_HIGH, size=()).item())
    star_type = random_star_picker()
    Mass, Radius_solar, u1, u2 = random_star_params(star_type)
    star_Radius_AU = mathematicals.convert_solar_radii_to_AU(Radius_solar)

    type_sim = SWARM_TO_RUN

    # =======================================================================
    # ===================== CORRECT TRANSIT GEOMETRY ========================
    # =======================================================================
    # 1. ORBITAL RADIUS: Set to be WIDE (a > R_star)
    #    e.g., 0.05 to 0.2 AU is ~10-40x a G-star's radius.
    ORBITAL_RADIUS_AU = np.random.uniform(0.05, 0.2) 

    # 2. INCLINATION: Set to be EDGE-ON (i ≈ 90 degrees)
    #    This ensures the projected path crosses the star.
    #    Impact parameter b = a * cos(i)
    #    We need b < R_star for a transit.
    #    e.g., a=0.1, R*=0.00465. We need cos(i) < 0.00465/0.1 = 0.0465
    #    This means i > arccos(0.0465) = 87.3 degrees.
    #    Sampling from [87.5, 89.9] guarantees a transit.
    min_inclination = np.degrees(np.arccos(star_Radius_AU / ORBITAL_RADIUS_AU))
    # Add a small buffer and ensure it's not > 90
    min_inclination = min(min_inclination + 0.1, 89.8) 
    inclination = np.random.uniform(min_inclination, 89.9) 

    # 3. OBSERVER: Fixed ("Philosophy A")
    observer = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    # =======================================================================

    # Calculate total particle count based on swarm type
    if type_sim == "res":
        num_clumps = np.random.randint(3, 10)
        PARTICLES_per_clump = int(PARTICLES / num_clumps)
        N_particles_total = PARTICLES_per_clump * num_clumps
    elif type_sim == "multi":
        num_rings = np.random.randint(2, 5)
        PARTICLES_per_ring = int(PARTICLES / num_rings)
        N_particles_total = PARTICLES_per_ring * num_rings
    else: # 'thin'
        num_clumps, num_rings = 0, 0 # For save file
        N_particles_total = PARTICLES

    # Re-sample particle radius based on *actual* N and desired depth
    # f_max=0.5 means the swarm *could* block up to 50% of the star
    PARTICLE_RADIUS_AU, covering_fraction = mathematicals.sample_particle_radius(
        star_Radius_AU, N_particles_total, f_min=0.2, f_max=0.5
    )

    ORBITAL_PERIOD = mathematicals.calculate_period(ORBITAL_RADIUS_AU, Mass)
    cadence = (2.0 / 1440.0) if TESS else (30.0 / 1440.0)
    tot_time = ORBITAL_PERIOD/4.0 # Simulate for 2 full orbits

    master_key, key_samples = jax.random.split(master_key)

    eccen = np.random.uniform(0.0, 0.05)
    ecc_alpha = 0.0 if eccen == 0.0 else -0.5
    
    # Swarm-specific structural params
    ring_sep_AU = np.random.uniform(0.001, 0.003) 
    within_ring_dispersion = np.random.uniform(0.5, 2.0) # Tighter dispersion
    inter_ring_tilts = (0.2, 1.5) # Tighter tilts
    clump_width_rad = np.random.uniform(0.01, 0.1) # Clump width in radians

    # --- Pick and build the swarm ---
    if type_sim == "res":
        sim = pick_swarm(
            type_sim=type_sim,
            num_particles_per_clump=PARTICLES_per_clump,
            num_clumps=num_clumps,
            a_AU=ORBITAL_RADIUS_AU, 
            particle_radius=PARTICLE_RADIUS_AU, 
            inc_deg=inclination,
            clump_width=clump_width_rad,
            star_mass=Mass, ecc_max=eccen, ecc_alpha=ecc_alpha, seed=None
        )
    elif type_sim == "multi":
         sim = pick_swarm(
            type_sim=type_sim,
            num_rings=num_rings,
            num_particles_per_ring=PARTICLES_per_ring,
            ring_sep_AU=ring_sep_AU,
            inc_dispersion_deg=within_ring_dispersion,  
            particle_radius=PARTICLE_RADIUS_AU,
            a_min_AU=ORBITAL_RADIUS_AU,
            starting_inclination=inclination,
            ecc_alpha=ecc_alpha, ecc_max=eccen, star_mass=Mass, seed=None,
        )
    else: # 'thin'
        sim = pick_swarm(
            type_sim=type_sim,
            num_particles=N_particles_total,
            AU=ORBITAL_RADIUS_AU,
            particle_radius=PARTICLE_RADIUS_AU,
            ecc_max=eccen, ecc_alpha=ecc_alpha, star_mass=Mass,
            inc_deg=inclination, seed=None
        )

    sim.integrator = "whfast"
    sim.dt = ORBITAL_PERIOD / 20.0
    times = np.arange(0, tot_time + cadence, cadence)

    x_s_all, y_s_all, mu_all, _ = Monte_carlo.make_all_disk_samples(
        key_samples, NUM_MONTE_CARLO_TRIALS, NUM_MONTE_CARLO_SAMPLES, star_Radius_AU
    )
   
    if VERBOSE == 1:
        print(f"Star: {star_type}, Mass: {Mass:.2f} M_sun, Radius: {Radius_solar:.3f} R_sun ({star_Radius_AU:.5f} AU)")
        print(f"Swarm: {type_sim}, N_particles: {N_particles_total}, p_radius: {PARTICLE_RADIUS_AU:.3e} AU")
        print(f"Orbit: a={ORBITAL_RADIUS_AU:.5f} AU, i={inclination:.2f} deg, P={ORBITAL_PERIOD:.4f} days")
        print(f"Observer: {observer}")
        print(f"Expected Max Covering Fraction (f): {covering_fraction:.3f}")
        print(f"Time: {len(times)} steps, dt={sim.dt:.4f} days, total={tot_time:.4f} days")
        print(f"Beginning integration for {type_sim} {lc_idx}...")

    # =================== 
    # integrate positions
    # ===================
    N_particles_sim = len(sim.particles) - 1
    if N_particles_sim != N_particles_total:
        print(f"Warning: N_particles in sim ({N_particles_sim}) != expected ({N_particles_total})")
        N_particles_total = N_particles_sim
        
    num_times = len(times)
    x_all = np.zeros((num_times, N_particles_total), dtype=np.float32)
    y_all = np.zeros((num_times, N_particles_total), dtype=np.float32)
    z_all = np.zeros((num_times, N_particles_total), dtype=np.float32)
    r_all = np.array([p.r for p in sim.particles[1:]], dtype=np.float32)

    for t_idx, t in enumerate(tqdm(times, desc="Integrating Orbits")):
        sim.integrate(t)
        x_all[t_idx, :] = [p.x for p in sim.particles[1:]]
        y_all[t_idx, :] = [p.y for p in sim.particles[1:]]
        z_all[t_idx, :] = [p.z for p in sim.particles[1:]]

    # ==============================================================
    # === Precompute static arrays for the full simulation =========
    # ==============================================================
    print("Preparing arrays for GPU...")
    positions_all = jnp.stack(
        [jnp.column_stack((x_all[t], y_all[t], z_all[t])) for t in range(num_times)],
        axis=0,  # (T, N, 3)
    )
    r_all_jax = jnp.array(r_all, dtype=jnp.float32)
    x_s_all_j = jnp.array(x_s_all, dtype=jnp.float32)
    y_s_all_j = jnp.array(y_s_all, dtype=jnp.float32)
    mu_all_j = jnp.array(mu_all, dtype=jnp.float32)
    observer_j = jnp.array(observer, dtype=jnp.float32)

    warmup(positions_all=positions_all)

    # ==============================================================
    # === GPU-accelerated batched flux computation =================
    # ==============================================================
    print("Computing fluxes on GPU...")
    fluxes_jax = batched_flux(
        positions_all,
        r_all_jax, # radii 
        star_Radius_AU,
        u1,
        u2,
        x_s_all_j,
        y_s_all_j,
        mu_all_j,
        observer_j,
    )
    fluxes = np.array(fluxes_jax) # Copy from GPU to CPU

    if VERBOSE == 1:
        print(f"Computed {len(fluxes)} flux values.")
        print(f"Flux stats: min={np.min(fluxes):.6f}, max={np.max(fluxes):.6f}, mean={np.mean(fluxes):.6f}")

    print(f"{type_sim}_{lc_idx} simulation finished.")

    # =========================
    # File Saving
    # ==========================
    
    # Create a simple plot to check results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(times, fluxes, "k.", markersize=1)
        plt.title(f"{type_sim} Light Curve (i={inclination:.2f} deg, a={ORBITAL_RADIUS_AU:.4f} AU, R*={star_Radius_AU:.4f} AU)")
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.ylim(max(0.0, np.min(fluxes) - 0.05), 1.05) # Auto-zoom y-axis
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save plot in the results directory
        results_dir = os.path.join(base_dir, "..", "results", type_sim)
        os.makedirs(results_dir, exist_ok=True)
        plot_filename = f"{type_sim}_{104+lc_idx}.png"
        full_plot_path = os.path.join(results_dir, plot_filename)
        
        plt.savefig(full_plot_path)
        print(f"Saved diagnostic plot to {full_plot_path}")
        plt.close()
    except ImportError:
        print("Matplotlib not found. Skipping diagnostic plot.")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # --- Save NPZ data ---
    file_name = f"{type_sim}_{104+lc_idx}.npz"
    full_file_path = os.path.join(results_dir, file_name)
    np.savez(full_file_path, times=times, flux=fluxes)
    print(f"Saved data to: {full_file_path}")

    # --- Save simulation parameters to one global JSON ---
    simulation_parameters = {
        "Stellar Mass": f"{Mass} Solar Masses",
        "Stellar Radius": f"{Radius_solar} Solar Radii",
        "Stellar Radius (AU)": f"{star_Radius_AU:.5e} AU",
        "Limb Darkening Coefficients": f"{u1}, {u2}",
        "Num_particles_total": f"{N_particles_total}",
        "Particle Radii (AU)": f"{PARTICLE_RADIUS_AU:.3e} AU",
        "Observer Vector": observer.tolist(),
        "Cadence": f"{cadence} days",
        "Total Integration Time": f"{tot_time} days",
        "Star Type": f"{star_type}",
        "Swarm Inclination": f"{inclination:.3f} deg",
        "Covering Fraction (f)": f"{covering_fraction:.4f}",
        "Orbital Radius (AU)": f"{ORBITAL_RADIUS_AU:.5e} AU",
        "Orbital Period": f"{ORBITAL_PERIOD:.4f} days",
        "Monte Carlo Trials": f"{NUM_MONTE_CARLO_TRIALS}",
        "Monte Carlo Samples": f"{NUM_MONTE_CARLO_SAMPLES}"
    }

    if type_sim == "multi":
        simulation_parameters["num rings"] = num_rings
        simulation_parameters["particles_per_ring"] = PARTICLES_per_ring
        simulation_parameters["ring_sep_AU"] = ring_sep_AU
        simulation_parameters["within_ring_dispersion"] = within_ring_dispersion
        simulation_parameters["inter_ring_tilts"] = inter_ring_tilts

    elif type_sim == "res":
        simulation_parameters["particles_per_clumps"] = PARTICLES_per_clump
        simulation_parameters["num clumps"] = num_clumps
        simulation_parameters["clump_width_rad"] = clump_width_rad

    global_json_path = os.path.join(results_dir, "simulation_parameters.json")

    try:
        if os.path.exists(global_json_path):
            with open(global_json_path, "r") as f:
                all_params = json.load(f)
        else:
            all_params = {}
    except json.JSONDecodeError:
        all_params = {} # Reset if file is corrupt

    sim_key = f"{type_sim}_{104+lc_idx}"
    all_params[sim_key] = simulation_parameters

    tmp_path = global_json_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(all_params, f, indent=4)
        os.replace(tmp_path, global_json_path)
    except Exception as e:
        print(f"Failed to save parameters to JSON: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Saved parameters for {sim_key} to {global_json_path}\n==================================")

print("Data Collection and Storage Complete. ")