import numpy as np
import rebound as rb
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from Simulation.utilities import Monte_carlo, mathematicals
from swarms import multi_ring, Diffuse, thin_ring, Resonant
# # thin_ring, Resonant, Diffuse

# # ========================
# # Load parameters
# # ========================
# # with open('config/Params.json') as file:
# #     data = json.load(file)

# # mass = np.random.normal(
# #     loc=data["star type"]['White_Dwarf']["mass"]["mean"],
# #     scale=data["star type"]['White_Dwarf']["mass"]["std"]
# # )

# mass = 1.0 # Solar Masses


# R_sun = 0.92 # Solar Radii
# star_radius = mathematicals.convert_solar_radii_to_AU(R_sun)
# closest_a = 0.01 # in AU


# # ========================
# # Build swarm simulation
# # ========================

# # sim = multi_ring.make_multi_ring(
# #     num_particles_per_ring=50000,
# #     num_rings=2,
# #     ring_sep_AU=0.0015,
# #     inc_dispersion_deg=45,
# #     particle_radius=6.68459e-6,
# #     a_min_AU=0.01,
# #     star_mass=0.9,
# # )

# sim = thin_ring.make_thin_ring(50000, closest_a, 7e-5,)
# # sim = Resonant.make_resonant_clumps(5000,2, closest_a, 6.68e-4, 0.0, mass,)
# # sim = Diffuse.make_diffuse_cloud(mass, 5000, closest_a, closest_a+0.02, particle_radius= 6.68e-4)

# sim.integrator = 'whfast'
# sim.dt = mathematicals.calculate_period(closest_a, mass)/20

# # ========================
# # Timing / cadence
# # ========================
# tot_time = mathematicals.calculate_period(closest_a,mass)
# cadence = 2.0 / 1440.0   # ~2 min
# times = np.arange(0, tot_time, cadence)

# print('Beginning integration...')

# # ========================
# # Pre-generate Monte Carlo disk samples
# # ========================
# n_trials = 200
# n_samples = int(1e4)

# master_key = jax.random.PRNGKey(1234)
# x_s_all, y_s_all, mu_all, _ = Monte_carlo.make_all_disk_samples(
#     master_key, n_trials, n_samples, star_radius
# )

# observer = Monte_carlo.sample_observer_edge_biased(key=master_key,max_cos=0.45)[0]
# # ========================
# # Loop over times
# # ========================

# fluxes = []

# print(mathematicals.calculate_period(closest_a, mass))

# for t in times:
#     sim.integrate(t)

#     # Convert REBOUND particle positions/radii to jnp
#     positions = jnp.array([[p.x, p.y, p.z] for p in sim.particles[1:]])
#     radii     = jnp.array([p.r for p in sim.particles[1:]])

#     # cpu prints (not jitted)
#     xs = np.array([p.x for p in sim.particles[1:]])
#     ys = np.array([p.y for p in sim.particles[1:]])
#     zs = np.array([p.z for p in sim.particles[1:]])
#     rs = np.array([p.r for p in sim.particles[1:]])
#     print(f"t={t:.3f} N={len(xs)}  front_count={(zs>0).sum()}") #  near_count={((xs**2+ys**2) <= (star_radius+rs)**2).sum()}
#     print("example z (first 5):", zs[:5])
#     print("example radii (first 5):", rs[:5])
#     print("example x (first 5):", xs[:5])
#     print("example y (first 5):", ys[:5])
    

#     flux = Monte_carlo.compute_monte_carlo_flux_jax_prepared(
#         star_radius,
#         positions,
#         radii,
#         u1=0.31,
#         u2=0.26,
#         x_s_all=x_s_all,
#         y_s_all=y_s_all,
#         mu_all=mu_all,
#         observer= observer
#     )
#     print(f"flux: {flux}\n====================")

#     fluxes.append(float(flux))

# # ========================
# # Plot + handle results
# # ========================
# plt.plot(times, fluxes, '-', color='orange', label='Flux')
# plt.xlabel("Time (days)")
# plt.ylabel("Normalized flux")
# plt.title("Time vs Normalized Flux")

# # Force y-axis to 0–1 (no autoscaling)
# plt.ylim(0.0, 1.0)
# plt.xlim(times[0], times[-1])  # optional: full time range
# plt.ticklabel_format(useOffset=False)  # disable '1e-7 + ...' notation

# plt.legend()
# plt.show()
# np.savetxt("output_flux.txt", np.column_stack((times, fluxes)), header="time flux raw data")

# # ========================
# # Subsample for visualization
# # ========================
# print("Preparing subset for visualization...")
# sub_sim = rb.Simulation()
# # Copy star
# p0 = sim.particles[0]
# sub_sim.add(m=p0.m, x=p0.x, y=p0.y, z=p0.z, vx=p0.vx, vy=p0.vy, vz=p0.vz, r=p0.r)

# # Copy random sample of swarm
# indices = np.random.choice(range(1, sim.N), size=min(10000, sim.N-1), replace=False)
# for i in indices:
#     p = sim.particles[int(i)]
#     sub_sim.add(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz, r=p.r)
# sub_sim.move_to_com()

# # ========================
# # Show 3D orbit plot
# # ========================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# main_star = sub_sim.particles[0]
# ax.scatter(main_star.x, main_star.y, main_star.z, color='yellow', s=100, marker='*')
# x = [p.x for p in sub_sim.particles[1:]]
# y = [p.y for p in sub_sim.particles[1:]]
# z = [p.z for p in sub_sim.particles[1:]]  
# ax.scatter(x, y, z, s=1)
# plt.show()

import os

results_dir = "results/multi"
for file_name in os.listdir(results_dir):
    if file_name.endswith('.npz'):
        file_path = os.path.join(results_dir, file_name)
        data = np.load(file_path)
        times = data["times"]
        flux = data["flux"]
        
        # Extract curve number from filename (assuming format like 'multi_104.npz')
        curve_num = file_name.split('.')[0].split('_')[-1]
        
        plt.figure(figsize=(8, 4))
        plt.plot(times, flux, label=f'Curve {curve_num}')
        plt.xlabel("Time [days]")
        plt.ylabel("Flux")
        plt.title(f"Light Curve #{curve_num}")
        plt.legend()
        plt.tight_layout()
        plt.show()
