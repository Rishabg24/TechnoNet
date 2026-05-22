import rebound as rb
import numpy as np
from Simulation.utilities.mathematicals import sample_ecc_powerlaw


def make_thin_ring(
    num_particles,
    a_AU,
    particle_radius,
    ecc_max=0.0,
    ecc_alpha=0.0,
    inc_deg=0.0,
    star_mass=1.0,
    seed=None,
):
    sim = rb.Simulation()
    sim.units = ("AU", "days", "Msun")
    sim.add(m=star_mass)  # Central star
    sim.N_active = 1

    if seed is not None:
        np.random.seed(seed)

    M_0 = np.random.uniform(0, 2 * np.pi, size=num_particles)

    if ecc_max>0.0:
        Omega = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0, 2 * np.pi)
        if ecc_alpha == 0.0:
            raise ValueError("Need an alpha and max eccentricity value for eccentricity sampling")
        
        e_part = sample_ecc_powerlaw(alpha=ecc_alpha, emax=ecc_max, size = num_particles) # Accounting for higher probability of low eccentric orbits
    else:
        Omega, omega, e_part = 0, 0, 0

    inc = np.radians(inc_deg)

    for j in range(num_particles):
        if ecc_max>0.0:
            sim.add(
                m=0,
                a=a_AU,
                e=e_part[j],
                inc=inc,
                Omega=Omega,
                omega=omega,
                M=M_0[j],
                r = particle_radius,
                primary = sim.particles[0]
            )
        else:
            sim.add(
                m=0,
                a=a_AU,
                e=e_part,
                inc=inc,
                Omega=Omega,
                omega=omega,
                M=M_0[j],
                r = particle_radius,
                primary = sim.particles[0]
            )

    return sim
