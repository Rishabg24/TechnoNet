import rebound as rb
import numpy as np
from Simulation.utilities.mathematicals import sample_ecc_powerlaw
import ctypes

# # Load the compiled C shared library
# lib = ctypes.CDLL("/absolute/path/to/libdiffuse.so")

# # Declare argument types for safety
# lib.add_particles_fast.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_int,
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
#     np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
# ]
# lib.add_particles_fast.restype = None


# def make_diffuse_cloud(
#     star_mass, num_particles, a_min, a_max, particle_radius, seed=None
# ):
#     sim = rb.Simulation()
#     sim.units = ("AU", "days", "Msun")
#     sim.N_active = 1
#     sim.add(m=star_mass)

#     if seed is not None:
#         np.random.seed(seed)

#     # Vectorized particle generation
#     M_0 = np.ascontiguousarray(
#         np.random.uniform(0, 2 * np.pi, num_particles), dtype=np.float64
#     )
#     a = np.ascontiguousarray(
#         np.random.uniform(a_min, a_max, num_particles), dtype=np.float64
#     )
#     omega = np.ascontiguousarray(
#         np.random.uniform(0, 2 * np.pi, num_particles), dtype=np.float64
#     )
#     Omega = np.ascontiguousarray(
#         np.random.uniform(0, 2 * np.pi, num_particles), dtype=np.float64
#     )
#     e = np.ascontiguousarray(
#         sample_ecc_powerlaw(emax=0.5, size=num_particles), dtype=np.float64
#     )
#     inc = np.ascontiguousarray(
#         np.arccos(np.random.uniform(-1, 1, num_particles)), dtype=np.float64
#     )
#     r = np.ascontiguousarray(np.full(num_particles, particle_radius), dtype=np.float64)

#     # Call the C function
#     lib.add_particles_fast(
#         ctypes.c_void_p(sim._simulation_address),
#         num_particles,
#         a,
#         e,
#         inc,
#         Omega,
#         omega,
#         M_0,
#         r,
#     )

#     return sim


import rebound as rb
import numpy as np
from Simulation.utilities.mathematicals import sample_ecc_powerlaw


def make_diffuse_cloud(star_mass, num_particles, a_min, a_max, particle_radius):
    sim = rb.Simulation()
    sim.units = ("AU", "days", "Msun")
    sim.N_active = 1
    sim.add(m=star_mass)

    M_0 = np.random.uniform(0, 2 * np.pi,size=num_particles)

    a = np.random.uniform(a_min, a_max,size=num_particles)

    omega = np.random.uniform(0, 2 * np.pi,size=num_particles)
    Omega = np.random.uniform(0, 2 * np.pi,size=num_particles)

    e = sample_ecc_powerlaw(emax=0.5,size=num_particles)

    inc = np.arccos(np.random.uniform(-1, 1,size=num_particles))
    for j in range(num_particles):
        sim.add(a=a[j], e=e[j], inc=inc[j], omega=omega[j], Omega=Omega[j], M=M_0[j], r= particle_radius, primary = sim.particles[0])

    return sim
