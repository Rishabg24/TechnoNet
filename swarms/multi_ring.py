import rebound as rb
import numpy as np
from Simulation.utilities.mathematicals import sample_ecc_powerlaw


def make_multi_ring(
    num_particles_per_ring,    
    num_rings,
    ring_sep_AU,
    inc_dispersion_deg,
    particle_radius,
    starting_inclination=87.0,  # CRITICAL: Keep near 87° for edge-on
    a_min_AU=0.003,
    ecc_alpha=0.0,
    ecc_max=0.0,
    star_mass=1.0,
    seed=None,
    random_ring_orientations=False,
    inter_ring_tilt_deg=(0.5, 3.0)  # MUCH SMALLER
):
    
    sim = rb.Simulation()
    sim.units = ("AU", "days", "Msun")
    sim.add(m=star_mass)
    sim.N_active = 1

    if seed is not None:
        np.random.seed(seed)

    # Track inclination as we build 
    current_inc = np.radians(starting_inclination)
    current_Omega = np.random.uniform(0, 2 * np.pi)
    current_omega = np.random.uniform(0, 2 * np.pi)

    for i in range(num_rings):
        a_ring = a_min_AU + i * ring_sep_AU

        if random_ring_orientations:
            # CHAOTIC: Not recommended for transit modeling
            ring_mean_inc = np.random.uniform(0, np.pi)
            Omega_ring = np.random.uniform(0, 2 * np.pi)
            omega_ring = np.random.uniform(0, 2 * np.pi)
        else:
            # HIERARCHICAL: Small systematic tilts (RECOMMENDED)
            if i == 0:
                # First ring at base inclination
                ring_mean_inc = current_inc
                Omega_ring = current_Omega
                omega_ring = current_omega
            else:
                # Small random tilt from previous ring
                tilt_min, tilt_max = inter_ring_tilt_deg
                tilt_angle = np.random.uniform(tilt_min, tilt_max)
                tilt_rad = np.radians(tilt_angle)
                
                # Random tilt direction (up/down, left/right in sky plane)
                tilt_direction = np.random.uniform(0, 2 * np.pi)
                
                # SMALL perturbation to inclination (not additive!)
                # With small angle approx. 
                delta_inc = tilt_rad * np.sin(tilt_direction)
                ring_mean_inc = current_inc + delta_inc
                
                # Keep physically reasonable
                ring_mean_inc = np.clip(ring_mean_inc, 
                                       np.radians(80),   # Min: still quite edge-on
                                       np.radians(90))   # Max: exactly edge-on
                
                # Small rotation of ascending node
                delta_Omega = tilt_rad * np.cos(tilt_direction)
                Omega_ring = (current_Omega + delta_Omega) % (2 * np.pi)
                
                # Minor variation in arg of periapsis
                omega_ring = current_omega + np.random.uniform(-0.1, 0.1)
                omega_ring = omega_ring % (2 * np.pi)
                
                # Update for next ring
                current_inc = ring_mean_inc
                current_Omega = Omega_ring
                current_omega = omega_ring

        # Particle properties
        M_0 = np.random.uniform(0, 2 * np.pi, size=num_particles_per_ring)
        e = sample_ecc_powerlaw(alpha=ecc_alpha, emax=ecc_max, size=num_particles_per_ring)

        # Within-ring inclination scatter (keep small!)
        inc = np.random.normal(
            loc=ring_mean_inc,
            scale=np.radians(inc_dispersion_deg),
            size=num_particles_per_ring
        )
        inc = np.abs(inc)
        inc = np.clip(inc, np.radians(75), np.radians(90))  # Force edge-on range

        # Add particles
        for j in range(num_particles_per_ring):
            sim.add(
                m=0,
                a=a_ring,
                e=e[j],
                inc=inc[j],
                Omega=Omega_ring,
                omega=omega_ring,
                M=M_0[j],
                r=particle_radius,
                primary=sim.particles[0]
            )

    return sim


def make_multi_ring_enhanced(
    num_particles_per_ring,    
    num_rings,
    ring_radii_AU,
    ring_inclinations_deg,
    inc_dispersion_deg,
    particle_radius,
    ecc_alpha=0.0,
    ecc_max=0.0,
    star_mass=1.0,
    seed=None
):
    """
    Full manual control - specify exact inclination for each ring.
    
    USAGE FOR DEEP TRANSITS:
    -------------------------
    Keep all inclinations near edge-on (85-89°):
    
    sim = make_multi_ring_enhanced(
        ring_inclinations_deg=[87, 86.5, 88, 87.2, 86],  # All edge-on. 
        ...
    )
    """
    if len(ring_radii_AU) != num_rings:
        raise ValueError("ring_radii_AU must have length num_rings")
    if len(ring_inclinations_deg) != num_rings:
        raise ValueError("ring_inclinations_deg must have length num_rings")
    
    sim = rb.Simulation()
    sim.units = ("AU", "days", "Msun")
    sim.add(m=star_mass)
    sim.N_active = 1

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_rings):
        a_ring = ring_radii_AU[i]
        ring_mean_inc = np.radians(ring_inclinations_deg[i])
        
        Omega_ring = np.random.uniform(0, 2 * np.pi)
        omega_ring = np.random.uniform(0, 2 * np.pi)

        M_0 = np.random.uniform(0, 2 * np.pi, size=num_particles_per_ring)
        e = sample_ecc_powerlaw(alpha=ecc_alpha, emax=ecc_max, size=num_particles_per_ring)

        inc = np.random.normal(
            loc=ring_mean_inc,
            scale=np.radians(inc_dispersion_deg),
            size=num_particles_per_ring
        )
        inc = np.abs(inc)
        inc = np.clip(inc, np.radians(80), np.radians(90))  # Keep edge-on

        for j in range(num_particles_per_ring):
            sim.add(
                m=0,
                a=a_ring,
                e=e[j],
                inc=inc[j],
                Omega=Omega_ring,
                omega=omega_ring,
                M=M_0[j],
                r=particle_radius,
                primary=sim.particles[0]
            )

    return sim


# DIAGNOSTIC: Calculate expected transit depth
def estimate_transit_depth(
    num_particles,
    particle_radius_AU,
    stellar_radius_AU,
    ring_radius_AU,
    inclination_deg,
    filling_factor=0.3
):
    """
    Quick estimate of expected transit depth.
    """
    inc_rad = np.radians(inclination_deg)
    
    # Projected area of ring in observer plane
    ring_height = 2 * ring_radius_AU * np.sin(inc_rad)
    
    # Total particle cross-section
    total_particle_area = num_particles * np.pi * particle_radius_AU**2
    
    # Effective blocking area (accounting for filling factor)
    blocking_area = total_particle_area * filling_factor
    
    # Stellar disk area
    stellar_area = np.pi * stellar_radius_AU**2
    
    # Transit depth (cap at 100%)
    depth = min(blocking_area / stellar_area, 1.0)
    normalized_flux = 1.0 - depth
    
    print(f"Inclination: {inclination_deg}°")
    print(f"Ring projected height: {ring_height:.6f} AU")
    print(f"Total particle area: {total_particle_area:.6e} AU²")
    print(f"Stellar area: {stellar_area:.6e} AU²")
    print(f"Expected transit depth: {depth*100:.1f}%")
    print(f"Expected flux during transit: {normalized_flux:.3f}")
    
    return normalized_flux


if __name__ == "__main":
    print(estimate_transit_depth(
        num_particles=40000,  # 4 rings × 10000
        particle_radius_AU=6.68e-4,
        stellar_radius_AU=0.01,  # ~1 solar radius
        ring_radius_AU=0.008,
        inclination_deg=87,  # Edge-on
        filling_factor=0.3
    ))
