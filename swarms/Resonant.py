import rebound as rb
import numpy as np
from Simulation.utilities.mathematicals import sample_ecc_powerlaw

def make_resonant_clumps(
    num_particles_per_clump,
    num_clumps,
    a_AU,
    particle_radius,
    inc_deg=90.0, # FIXED DEFAULT: Changed from 3.0 to 90.0 (edge-on)
    star_mass=1.0,
    ecc_alpha=-0.5,
    ecc_max=0.05,
    clump_width=0.001, # This is in radians (mean anomaly)
    seed=None,
):
    """
    Creates clumps of particles in a resonant-like structure.
    The inclination `inc_deg` is critical. 
    For a z-axis observer, inc_deg=90 is edge-on.
    """
    sim = rb.Simulation()
    sim.units = ("AU", "days", "Msun")
    sim.add(m=star_mass)
    sim.N_active = 1
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Global parameters - same for all clumps
    omega_global = rng.uniform(0, 2*np.pi)
    Omega_global = rng.uniform(0, 2*np.pi)
    
    # Use 'size' parameter for sample_ecc_powerlaw
    e_global = sample_ecc_powerlaw(alpha=ecc_alpha, emax=ecc_max, size=1)[0] 
    inc_global = np.radians(inc_deg)
    
    separation = (2*np.pi) / num_clumps

    for c in range(num_clumps):
        M_center = c * separation
        
        # ADD TINY VARIATIONS to each clump to prevent perfect alignment
        a_clump_base = a_AU + rng.normal(0, a_AU * 0.001)  # 0.1% a variation per clump
        inc_clump_base = inc_global + rng.normal(0, np.radians(0.05))  # 0.05° variation
        omega_clump_base = omega_global + rng.normal(0, np.radians(0.1))  # 0.1° variation
        Omega_clump_base = Omega_global + rng.normal(0, np.radians(0.1))  # 0.1° variation
        
        # Spread particles within the clump using normal distribution
        # clump_width is the std dev of Mean Anomaly
        M_0 = rng.normal(loc=M_center, scale=clump_width, size=num_particles_per_clump)
        
        # Add very small scatter within each clump
        a_particles = a_clump_base + rng.normal(0, 1e-7, size=num_particles_per_clump)
        inc_particles = inc_clump_base + rng.normal(0, np.radians(0.01), size=num_particles_per_clump)
        omega_particles = omega_clump_base + rng.normal(0, np.radians(0.02), size=num_particles_per_clump)
        Omega_particles = Omega_clump_base + rng.normal(0, np.radians(0.02), size=num_particles_per_clump)
        
        # Give each particle its own eccentricity from the distribution
        e_particles = sample_ecc_powerlaw(alpha=ecc_alpha, emax=ecc_max, size=num_particles_per_clump)

        for j in range(num_particles_per_clump):
            sim.add(
                a=a_particles[j],
                e=e_particles[j], 
                inc=inc_particles[j],
                M=M_0[j],
                omega=omega_particles[j],
                Omega=Omega_particles[j],
                r=particle_radius,
                primary=sim.particles[0],
            )

    return sim