import rebound as rb
import numpy as np
import json
from Simulation.utilities import mathematicals
from swarms import thin_ring, Resonant
import matplotlib.pyplot as plt


# ========================
# Load parameters
# ========================
with open("config/Params.json") as file:
    data = json.load(file)

conf = data["star type"]["M_class"]
m_bounds = conf["mass"]
mass = mass = mathematicals.draw_chabrier_mass(m_bounds["min"], m_bounds["max"])

# user-specified input (keep these in variables so we can reference them)
a_AU = 0.01  # semimajor axis used for the thin ring
R_sun_frac = 0.016  # fraction of solar radius (as in your original)

# convert star radius to AU
star_radius = mathematicals.convert_solar_radii_to_AU(R_sun_frac)

# Set escape radius sensibly relative to ring radius so particles at a_AU aren't flagged as escaped at t=0
escape_radius = max(6 * star_radius, 3 * a_AU)  # e.g. at least 3× a_AU (adjust as needed)
collision_radius = star_radius

# ========================
# Build swarm simulation
# ========================
# sim = thin_ring.make_thin_ring(
#     num_particles=7000,
#     a_AU=a_AU,
#     inc_deg=56,
#     particle_radius=6.68459e-3,
#     star_mass=mass,
#     ecc_alpha=-0.5,
#     ecc_max=0.3,
# )

closest_a = 0.01
sim = Resonant.make_resonant_clumps(num_particles_per_clump=5000, num_clumps=4, a_AU = closest_a, particle_radius=6.68e-4,)
sim.integrator = "whfast"

# use period at the ring a_AU for dt (was using 0.031 earlier — now we use the actual ring radius)
sim.dt = mathematicals.calculate_period(a_AU, mass) / 20.0

# ========================
# Subsample for visualization
# ========================
print("Preparing subset for visualization...")
sub_sim = rb.Simulation()
# Copy star
p0 = sim.particles[0]
sub_sim.add(m=p0.m, x=p0.x, y=p0.y, z=p0.z, vx=p0.vx, vy=p0.vy, vz=p0.vz, r=p0.r)

# Copy random sample of swarm
indices = np.random.choice(range(1, sim.N), size=min(10000, sim.N-1), replace=False)
for i in indices:
    p = sim.particles[int(i)]
    sub_sim.add(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz, r=p.r)
sub_sim.move_to_com()

# ========================
# Show 3D orbit plot
# ========================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# main_star = sub_sim.particles[0]
# ax.scatter(main_star.x, main_star.y, main_star.z, color='yellow', s=100, marker='*')
# x = [p.x for p in sub_sim.particles[1:]]
# y = [p.y for p in sub_sim.particles[1:]]
# z = [p.z for p in sub_sim.particles[1:]]
# ax.scatter(x, y, z, s=1)
# plt.show()

# ========================
# Subsample for diagnostics
# ========================
diag_N = min(2000, sim.N - 1)  # number of particles for diagnostics (skip the star at index 0)
diag_indices = np.random.choice(range(1, sim.N), size=diag_N, replace=False).astype(int)

# Extract star
star = sim.particles[0]

# ========================
# Helper functions
# ========================
def angular_momentum_drift(simulation, L0):
    """
    Return (relative_drift, L_mag_vector_norm)
    relative_drift = ||L - L0|| (absolute vector norm)
    """
    L = simulation.angular_momentum()  # returns 3-vector
    diff = L - L0
    rel_drift = np.linalg.norm(diff)
    L_mag = np.linalg.norm(L)
    return rel_drift, L_mag


def check_stability_vectorized(simulation, indices, escape_r, collision_r):
    """
    Vectorized distances for selected particle indices (skips star assumed at index 0)
    Returns number escaped, number collided (counts within the subset)
    """
    star = simulation.particles[0]
    # gather positions (ensure ints)
    positions = np.array(
        [[simulation.particles[int(i)].x, simulation.particles[int(i)].y, simulation.particles[int(i)].z] for i in indices]
    )
    star_pos = np.array([star.x, star.y, star.z])
    r = np.linalg.norm(positions - star_pos, axis=1)
    escaped = int(np.sum(r > escape_r))
    collided = int(np.sum(r < collision_r))
    return escaped, collided


# ========================
# Diagnostics storage
# ========================
times = []
L_rel_drifts = []  # store relative drift (normalized)
L_mags = []        # store absolute magnitude if wanted
escaped_counts = []
collided_counts = []

# initial angular momentum (vector)
L_0 = sim.angular_momentum()
L0_norm = np.linalg.norm(L_0)

# integration setup (total time in same time units that calculate_period returns — keep your tot_time)
tot_time = 365000
log_steps = np.unique(np.logspace(0, np.log10(tot_time), num=1000, dtype=int))

# initial diagnostic before integrating
initial_escaped, initial_collided = check_stability_vectorized(sim, diag_indices, escape_radius, collision_radius)
print(f"Initial escaped: {initial_escaped}/{len(diag_indices)}")
print(f"Initial collided: {initial_collided}/{len(diag_indices)}")

print("Beginning integration...")

for step in log_steps:
    # integrate to absolute time = step * dt (log-spaced)
    target_time = step * sim.dt
    sim.integrate(target_time)

    times.append(sim.t)

    rel_drift, L_mag = angular_momentum_drift(sim, L_0)

    # store normalized relative drift (avoid division by zero)
    if L0_norm > 0:
        L_rel_drifts.append(rel_drift / L0_norm)
    else:
        L_rel_drifts.append(rel_drift)

    L_mags.append(L_mag)

    escaped, collided = check_stability_vectorized(sim, diag_indices, escape_radius, collision_radius)
    escaped_counts.append(escaped)
    collided_counts.append(collided)

    if rel_drift / max(1e-30, L0_norm) > 1e-3:
        print(f"⚠️ Angular momentum relative drift too high at t={sim.t:.3e}")

print("Integration complete.")
print("N =", sim.N)

# ========================
# Plot Relative Angular Momentum Drift
plt.figure(figsize=(8,5))
plt.semilogy(times, np.maximum(L_rel_drifts, 1e-30), "-", label="Relative Angular Momentum Drift (||ΔL||/||L0||)")
plt.xlabel("Time")
plt.ylabel("Relative Angular Momentum Drift")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()

# # ========================
# # Plot Escaped/Collided Particles (subset)
# plt.figure(figsize=(8,5))
# plt.plot(times, escaped_counts, "-", label="Escaped Particles (subset)")
# plt.plot(times, collided_counts, "-", label="Collided Particles (subset)")
# plt.xlabel("Time")
# plt.ylabel("Number of Particles (subset)")
# plt.legend()
# plt.grid(True, ls="--", alpha=0.4)
# plt.show()

# # ========================
# # Visualization subset (3D scatter)
# sub_N = min(5000, sim.N - 1)
# sub_indices = np.random.choice(range(1, sim.N), size=sub_N, replace=False).astype(int)

# sub_sim = rb.Simulation()
# sub_sim.add(m=star.m, x=star.x, y=star.y, z=star.z, vx=star.vx, vy=star.vy, vz=star.vz, r=star.r)

# for i in sub_indices:
#     p = sim.particles[int(i)]
#     sub_sim.add(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz, r=p.r)

# sub_sim.move_to_com()

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(star.x, star.y, star.z, marker="*", s=100, label="Star")
# x = [p.x for p in sub_sim.particles[1:]]
# y = [p.y for p in sub_sim.particles[1:]]
# z = [p.z for p in sub_sim.particles[1:]]
# ax.scatter(x, y, z, s=1)
# ax.set_xlabel("x (AU?)")
# ax.set_ylabel("y (AU?)")
# ax.set_zlabel("z (AU?)")
# plt.legend()
# plt.show()


# import numpy as np, jax, jax.numpy as jnp
# # use your make_all_disk_samples function (same R_star units)
# Rstar = float(star_radius)   # from your script
# n_trials = 50
# n_samples = 2000
# mk = jax.random.PRNGKey(0)
# x_s_all, y_s_all, mu_all, _ = Monte_carlo.make_all_disk_samples(mk, n_trials, n_samples, Rstar)

# # single centered particle
# Rp = 0.1 * Rstar   # test radius = 10% of star
# positions = jnp.array([[0.0, 0.0, 1.0]])   # in front of star
# radii = jnp.array([Rp])
# flux = Monte_carlo.compute_monte_carlo_flux_jax_prepared(
#     Rstar, positions, radii, u1=0.0, u2=0.0,
#     x_s_all=x_s_all, y_s_all=y_s_all, mu_all=mu_all,
#     observer=(0.0,0.0,1.0)
# )
# flux = float(flux)
# expected = 1.0 - (Rp/Rstar)**2
# print("flux (MC)   :", flux)
# print("expected    :", float(expected))
# print("abs error   :", abs(flux - float(expected)))

# import rebound as rb
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
# from Simulation.utilities import mathematicals
# from swarms import multi_ring

# # ========================
# # Load parameters
# # ========================
# with open('config/Params.json') as file:
#     data = json.load(file)

# mass = np.random.normal(
#     loc=data["star type"]['White_Dwarf']["mass"]["mean"],
#     scale=data["star type"]['White_Dwarf']["mass"]["std"]
# )

# R_sun = 0.016
# star_radius = mathematicals.convert_solar_radii_to_AU(R_sun)

# # ========================
# # Build swarm simulation
# # ========================
# sim = multi_ring.make_multi_ring(
#     num_particles_per_ring=500000,
#     num_rings=4,
#     ring_sep_AU=0.0001,
#     inc_dispersion_deg=45,
#     a_min_AU=0.003,
#     particle_radius=6.68459e-5,   # in AU
#     starting_inclination=87,      # degrees
#     star_mass=mass,
# )

# sim.integrator = 'whfast'
# sim.dt = mathematicals.calculate_period(0.031, mass)/20

# # ========================
# # Downsample for visualization
# # ========================
# print("Preparing subset for animation...")
# sub_sim = rb.Simulation()
# sub_sim.add(sim.particles[0])  # central star

# indices = np.random.choice(range(1, sim.N), size=min(1000, sim.N-1), replace=False)
# for i in indices:
#     sub_sim.add(sim.particles[i])

# # ========================
# # Setup Matplotlib figure
# # ========================
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection="3d")
# ax.set_facecolor("black")

# # initial plot
# particles = sub_sim.particles
# x = [p.x for p in particles[1:]]
# y = [p.y for p in particles[1:]]
# z = [p.z for p in particles[1:]]
# scat = ax.scatter(x, y, z, s=2, color="deepskyblue")
# star = ax.scatter([0], [0], [0], s=100, color="gold", marker="*")

# ax.set_xlim(-0.01, 0.01)
# ax.set_ylim(-0.01, 0.01)
# ax.set_zlim(-0.01, 0.01)
# ax.set_xlabel("x [AU]")
# ax.set_ylabel("y [AU]")
# ax.set_zlabel("z [AU]")
# ax.set_title("3D Animated Dyson Swarm Visualization", color="white")
# ax.w_xaxis.line.set_color("white")
# ax.w_yaxis.line.set_color("white")
# ax.w_zaxis.line.set_color("white")
# ax.tick_params(colors="white")

# # ========================
# # Animation update function
# # ========================
# def update(frame):
#     sub_sim.integrate(sub_sim.t + sub_sim.dt)
#     x = [p.x for p in sub_sim.particles[1:]]
#     y = [p.y for p in sub_sim.particles[1:]]
#     z = [p.z for p in sub_sim.particles[1:]]
#     scat._offsets3d = (x, y, z)
#     ax.set_title(f"t = {sub_sim.t:.3e} yr", color="white")
#     return scat,

# # ========================
# # Create animation
# # ========================
# ani = FuncAnimation(fig, update, frames=500, interval=20, blit=False)

# plt.show()
