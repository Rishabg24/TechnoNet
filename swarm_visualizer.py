import rebound as rb
import numpy as np
import json
from Simulation.utilities import mathematicals
from swarms import multi_ring, Resonant, thin_ring
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================
# VISUALIZATION FLAGS
# ========================
VISUALIZE_MULTI_RING = True
VISUALIZE_RESONANT = True
VISUALIZE_THIN_RING = True

# ========================
# Load parameters
# ========================

mass = 1.0  # Solar masses 
R_sun = 1.0  # Solar radii
star_radius = mathematicals.convert_solar_radii_to_AU(R_sun)

print(f"Star radius: {star_radius:.6f} AU (~{star_radius/0.00465:.2f} solar radii)")

# ========================
# REALISTIC MULTI-RING SYSTEM
# ========================
if VISUALIZE_MULTI_RING:
    print("\n" + "="*60)
    print("REALISTIC MULTI-RING SYSTEM FOR TRANSITS")
    print("="*60)
    print("Configuration:")
    print("  - 5 rings with hierarchical tilts")
    print("  - 12000 particles per ring (60000 total)")
    print("  - Ring radii: 0.005 to 0.013 AU (well-separated)")
    print("  - Base inclination: 87° (edge-on)")
    print("  - Inter-ring tilts: 20-30° each")
    print("  - Inclination dispersion within ring: ±3°")
    print("  - Low eccentricity: ~0.01")
    print("="*60)

    NUM_PARTICLES_PER_RING = 12000
    NUM_RINGS = 4

    sim_multi = multi_ring.make_multi_ring(
        num_particles_per_ring=NUM_PARTICLES_PER_RING,
        num_rings=NUM_RINGS,
        ring_sep_AU=np.random.uniform(0.0015, 0.003),
        inc_dispersion_deg=3.0,
        particle_radius=6.68e-7,
        starting_inclination=0.0,
        a_min_AU=0.003,
        ecc_alpha=0.0,
        ecc_max=0.0,
        star_mass=mass,
        seed=None,
        random_ring_orientations=False,
        inter_ring_tilt_deg=(20.0, 30.0)
    )

    sim_multi.integrator = 'whfast'
    orbital_period_multi = mathematicals.calculate_period(0.005, mass)
    sim_multi.dt = orbital_period_multi / 100

    print(f"\nInnermost ring orbital period: {orbital_period_multi:.4f} days")
    print(f"Time step: {sim_multi.dt:.6f} days")
    print("Beginning integration for 1 full orbit of innermost ring...")

    for i in range(101):
        t = i * orbital_period_multi / 100
        sim_multi.integrate(t)
        if i % 20 == 0:
            p = sim_multi.particles[1]
            print(f"  t={sim_multi.t:.3f} days | First particle: ({p.x:.3e}, {p.y:.3e}, {p.z:.3e})")

    print(f"Integration complete. Total particles: {sim_multi.N}")

    # Print ring statistics
    print("\n" + "="*60)
    print("RING STATISTICS")
    print("="*60)
    for ring_idx in range(NUM_RINGS):
        start_idx = 1 + ring_idx * NUM_PARTICLES_PER_RING
        end_idx = start_idx + NUM_PARTICLES_PER_RING
        
        sample_size = min(100, NUM_PARTICLES_PER_RING)
        sample_indices = range(start_idx, min(start_idx + sample_size, sim_multi.N))
        
        a_vals = [sim_multi.particles[i].a for i in sample_indices]
        inc_vals = [np.degrees(sim_multi.particles[i].inc) for i in sample_indices]
        
        mean_a = np.mean(a_vals)
        mean_inc = np.mean(inc_vals)
        std_inc = np.std(inc_vals)
        
        print(f"Ring {ring_idx+1}:")
        print(f"  Semi-major axis: {mean_a:.6f} AU")
        print(f"  Inclination: {mean_inc:.2f}° ± {std_inc:.2f}°")
        print(f"  Radius in stellar radii: {mean_a/star_radius:.2f}")

    # Visualize Multi-Ring
    print("\nPreparing multi-ring visualization...")
    fig = plt.figure(figsize=(18, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    star = sim_multi.particles[0]
    ax1.scatter(star.x, star.y, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)

    colors = plt.cm.plasma(np.linspace(0, 1, NUM_RINGS))

    for ring_idx in range(NUM_RINGS):
        start_idx = 1 + ring_idx * NUM_PARTICLES_PER_RING
        end_idx = start_idx + NUM_PARTICLES_PER_RING
        
        n_sample = min(400, NUM_PARTICLES_PER_RING)
        all_indices = range(start_idx, min(end_idx, sim_multi.N))
        sampled_indices = np.random.choice(list(all_indices), size=n_sample, replace=False)
        
        x_ring = np.array([sim_multi.particles[int(i)].x for i in sampled_indices])
        y_ring = np.array([sim_multi.particles[int(i)].y for i in sampled_indices])
        z_ring = np.array([sim_multi.particles[int(i)].z for i in sampled_indices])
        
        ax1.scatter(x_ring, y_ring, z_ring, s=3, alpha=0.6, 
                   c=[colors[ring_idx]], label=f'Ring {ring_idx+1}')

    ax1.set_xlabel('x (AU)', fontsize=10)
    ax1.set_ylabel('y (AU)', fontsize=10)
    ax1.set_zlabel('z (AU)', fontsize=10)
    ax1.set_title('Multi-Ring System - 3D View', fontsize=11)
    legend1 = ax1.legend(markerscale=2, loc='upper left', fontsize=8)
    legend1.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax1.view_init(elev=25, azim=45)

    x_all = np.array([sim_multi.particles[i].x for i in range(1, min(3000, sim_multi.N))])
    y_all = np.array([sim_multi.particles[i].y for i in range(1, min(3000, sim_multi.N))])
    z_all = np.array([sim_multi.particles[i].z for i in range(1, min(3000, sim_multi.N))])

    max_range = max(x_all.max()-x_all.min(), y_all.max()-y_all.min(), 
                    z_all.max()-z_all.min()) * 0.55
    mid_x, mid_y, mid_z = 0, 0, 0

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # XY projection (top-down view - face-on)
    ax3 = fig.add_subplot(132)
    ax3.scatter(star.x, star.y, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)

    for ring_idx in range(NUM_RINGS):
        start_idx = 1 + ring_idx * NUM_PARTICLES_PER_RING
        end_idx = start_idx + NUM_PARTICLES_PER_RING
        
        n_sample = min(400, NUM_PARTICLES_PER_RING)
        all_indices = range(start_idx, min(end_idx, sim_multi.N))
        sampled_indices = np.random.choice(list(all_indices), size=n_sample, replace=False)
        
        x_ring = np.array([sim_multi.particles[int(i)].x for i in sampled_indices])
        y_ring = np.array([sim_multi.particles[int(i)].y for i in sampled_indices])
        
        ax3.scatter(x_ring, y_ring, s=3, alpha=0.6, 
                   c=[colors[ring_idx]], label=f'Ring {ring_idx+1}')

    ax3.set_xlabel('x (AU)', fontsize=10)
    ax3.set_ylabel('z (AU)', fontsize=10)
    ax3.set_title('Edge-On View (XZ Plane)', fontsize=11)
    ax3.set_aspect('equal')
    legend3 = ax3.legend(markerscale=2, loc='upper right', fontsize=8)
    legend3.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax3.grid(True, alpha=0.3)

    # XZ projection (edge-on view - transit geometry!)
    ax2 = fig.add_subplot(133)
    ax2.scatter(star.x, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)

    for ring_idx in range(NUM_RINGS):
        start_idx = 1 + ring_idx * NUM_PARTICLES_PER_RING
        end_idx = start_idx + NUM_PARTICLES_PER_RING
        
        n_sample = min(400, NUM_PARTICLES_PER_RING)
        all_indices = range(start_idx, min(end_idx, sim_multi.N))
        sampled_indices = np.random.choice(list(all_indices), size=n_sample, replace=False)
        
        x_ring = np.array([sim_multi.particles[int(i)].x for i in sampled_indices])
        z_ring = np.array([sim_multi.particles[int(i)].z for i in sampled_indices])
        
        ax2.scatter(x_ring, z_ring, s=3, alpha=0.6, 
                   c=[colors[ring_idx]], label=f'Ring {ring_idx+1}')

    ax2.set_xlabel('x (AU)', fontsize=10)
    ax2.set_ylabel('y (AU)', fontsize=10)
    ax2.set_title('Top View (XY Plane) - Transit Geometry', fontsize=11)
    ax2.set_aspect('equal')
    legend3 = ax2.legend(markerscale=2, loc='upper right', fontsize=8)
    legend3.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multi_ring_realistic.png', dpi=150)
    print("Saved visualization to 'multi_ring_realistic.png'")
    plt.show()

    print("\n" + "="*60)
    print("MULTI-RING SYSTEM SUMMARY")
    print("="*60)
    print(f"Total particles: {sim_multi.N - 1}")
    print(f"Number of rings: {NUM_RINGS}")
    print("\nExpected transit behavior:")
    print("  - Different rings present different projected areas")
    print("  - Some rings may be nearly edge-on (deep transits)")
    print("  - Some rings may be more face-on (shallow/grazing transits)")
    print("  - Multiple rings create complex, multi-component light curves")
    print("="*60)

# ========================
# RESONANT CLUMPS
# ========================
if VISUALIZE_RESONANT:
    print("\n" + "="*60)
    print("RESONANT CLUMPS SYSTEM")
    print("="*60)
    print("Configuration:")
    print("  - 8 clumps uniformly distributed around orbit")
    print("  - 10000 particles per clump (80000 total)")
    print("  - Orbital radius: 0.008 AU (~3.5 stellar radii)")
    print("  - Inclination: 85° (nearly edge-on)")
    print("  - Eccentricity: ~0.05 (slightly elliptical)")
    print("  - Clump width: 0.001 radians")
    print("="*60)

    sim_res = Resonant.make_resonant_clumps(
        num_particles_per_clump=10000,
        num_clumps=8,
        a_AU=0.008,
        particle_radius=6.68e-7,
        inc_deg=85.0,
        star_mass=mass,
        ecc_alpha=-0.5,
        ecc_max=0.05,
        clump_width=0.001,
        seed=42
    )

    sim_res.integrator = 'whfast'
    orbital_period_res = mathematicals.calculate_period(0.008, mass)
    sim_res.dt = orbital_period_res / 100

    print(f"\nOrbital period: {orbital_period_res:.4f} days")
    print(f"Time step: {sim_res.dt:.6f} days")
    print("Beginning integration for 1 full orbit...")

    for i in range(101):
        t = i * orbital_period_res / 100
        sim_res.integrate(t)
        if i % 20 == 0:
            p = sim_res.particles[1]
            print(f"  t={sim_res.t:.3f} days | First particle: ({p.x:.3e}, {p.y:.3e}, {p.z:.3e})")

    print(f"Integration complete. Total particles: {sim_res.N}")

    # Visualize Resonant Clumps
    print("\nPreparing resonant clump visualization...")
    fig = plt.figure(figsize=(18, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    star = sim_res.particles[0]
    ax1.scatter(star.x, star.y, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)

    n_viz = min(3200, sim_res.N - 1)
    indices = np.random.choice(range(1, sim_res.N), size=n_viz, replace=False)
    x = np.array([sim_res.particles[int(i)].x for i in indices])
    y = np.array([sim_res.particles[int(i)].y for i in indices])
    z = np.array([sim_res.particles[int(i)].z for i in indices])

    ax1.scatter(x, y, z, s=1, alpha=0.6, c='C0', label='Particles')
    ax1.set_xlabel('x (AU)', fontsize=10)
    ax1.set_ylabel('y (AU)', fontsize=10)
    ax1.set_zlabel('z (AU)', fontsize=10)
    ax1.set_title('Resonant Clumps - 3D View', fontsize=11)
    legend1 = ax1.legend(markerscale=2, loc='upper left', fontsize=8)
    legend1.legend_handles[0]._sizes = [30]  # Resize star in legend

    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) * 0.6
    mid_x, mid_y, mid_z = 0, 0, 0
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.view_init(elev=25, azim=45)

 
    # XZ projection (edge-on view)
    ax3 = fig.add_subplot(133)
    ax3.scatter(star.x, star.y, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)
    ax3.scatter(x, y, s=1, alpha=0.6, c='C0', label='Particles')
    ax3.set_xlabel('x (AU)', fontsize=10)
    ax3.set_ylabel('z (AU)', fontsize=10)
    ax3.set_title('Edge-On View (XZ Plane)', fontsize=11)
    ax3.set_aspect('equal')
    legend3 = ax3.legend(markerscale=2, loc='upper right', fontsize=8)
    legend3.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax3.grid(True, alpha=0.3)

    # XY projection (top-down view)
    ax2 = fig.add_subplot(132)
    ax2.scatter(star.x, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)
    ax2.scatter(x, z, s=1, alpha=0.6, c='C0', label='Particles')
    ax2.set_xlabel('x (AU)', fontsize=10)
    ax2.set_ylabel('y (AU)', fontsize=10)
    ax2.set_title('Top View (XY Plane) - Transit Geometry', fontsize=11)
    ax2.set_aspect('equal')
    legend2 = ax2.legend(markerscale=2, loc='upper right', fontsize=8)
    legend2.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resonant_clumps.png', dpi=150)
    print("Saved visualization to 'resonant_clumps.png'")
    plt.show()

    print("\n" + "="*60)
    print("RESONANT CLUMPS SUMMARY")
    print("="*60)
    print(f"Total particles: {sim_res.N - 1}")
    print(f"Number of clumps: 8")
    print("Expected transit behavior:")
    print("  - 8 distinct clumps in resonant configuration")
    print("  - Periodic, repeating transit pattern")
    print("  - Clumps at different orbital phases")
    print("="*60)

# ========================
# THIN RING
# ========================
if VISUALIZE_THIN_RING:
    print("\n" + "="*60)
    print("THIN RING SYSTEM")
    print("="*60)
    print("Configuration:")
    print("  - Single thin ring")
    print("  - 20000 particles")
    print("  - Orbital radius: 0.007 AU (~3.0 stellar radii)")
    print("  - Inclination: 85° (nearly edge-on)")
    print("  - Eccentricity: ~0.02")
    print("="*60)

    sim_thin = thin_ring.make_thin_ring(
        num_particles=20000,
        a_AU=0.007,
        particle_radius=6.68e-7,
        ecc_max=0.02,
        ecc_alpha=-0.5,
        inc_deg=85.0,
        star_mass=mass,
        seed=44
    )

    sim_thin.integrator = 'whfast'
    orbital_period_thin = mathematicals.calculate_period(0.007, mass)
    sim_thin.dt = orbital_period_thin / 100

    print(f"\nOrbital period: {orbital_period_thin:.4f} days")
    print(f"Time step: {sim_thin.dt:.6f} days")
    print("Beginning integration for 1 full orbit...")

    for i in range(101):
        t = i * orbital_period_thin / 100
        sim_thin.integrate(t)
        if i % 20 == 0:
            p = sim_thin.particles[1]
            print(f"  t={sim_thin.t:.3f} days | First particle: ({p.x:.3e}, {p.y:.3e}, {p.z:.3e})")

    print(f"Integration complete. Total particles: {sim_thin.N}")

    # Visualize Thin Ring
    print("\nPreparing thin ring visualization...")
    fig = plt.figure(figsize=(18, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    star = sim_thin.particles[0]
    ax1.scatter(star.x, star.y, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)

    n_viz = min(2000, sim_thin.N - 1)
    indices = np.random.choice(range(1, sim_thin.N), size=n_viz, replace=False)
    x = np.array([sim_thin.particles[int(i)].x for i in indices])
    y = np.array([sim_thin.particles[int(i)].y for i in indices])
    z = np.array([sim_thin.particles[int(i)].z for i in indices])

    ax1.scatter(x, y, z, s=1, alpha=0.6, c='C2', label='Particles')
    ax1.set_xlabel('x (AU)', fontsize=10)
    ax1.set_ylabel('y (AU)', fontsize=10)
    ax1.set_zlabel('z (AU)', fontsize=10)
    ax1.set_title('Thin Ring - 3D View', fontsize=11)
    legend1 = ax1.legend(markerscale=2, loc='upper left', fontsize=8)
    legend1.legend_handles[0]._sizes = [30]  # Resize star in legend

    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) * 0.6
    mid_x, mid_y, mid_z = 0, 0, 0
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.view_init(elev=25, azim=45)

       # XZ projection (edge-on view)
    ax3 = fig.add_subplot(133)
    ax3.scatter(star.x, star.y, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)
    ax3.scatter(x, y, s=1, alpha=0.6, c='C2', label='Particles')
    ax3.set_xlabel('x (AU)', fontsize=10)
    ax3.set_ylabel('x (AU)', fontsize=10)
    ax3.set_title('Edge-On View (XZ Plane)', fontsize=11)
    ax3.set_aspect('equal')
    legend3 = ax3.legend(markerscale=2, loc='upper right', fontsize=8)
    legend3.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax3.grid(True, alpha=0.3)


     # XY projection (top-down view)
    ax2 = fig.add_subplot(132)
    ax2.scatter(star.x, star.z, color='gold', s=400, marker='*', 
               label='Star', zorder=10, edgecolors='orange', linewidths=2)
    ax2.scatter(x, z, s=1, alpha=0.6, c='C2', label='Particles')
    ax2.set_xlabel('x (AU)', fontsize=10)
    ax2.set_ylabel('y (AU)', fontsize=10)
    ax2.set_title('Top View (XY Plane) - Transit Geometry', fontsize=11)
    ax2.set_aspect('equal')
    legend2 = ax2.legend(markerscale=2, loc='upper right', fontsize=8)
    legend2.legend_handles[0]._sizes = [30]  # Resize star in legend
    ax2.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig('thin_ring.png', dpi=150)
    print("Saved visualization to 'thin_ring.png'")
    plt.show()

    print("\n" + "="*60)
    print("THIN RING SUMMARY")
    print("="*60)
    print(f"Total particles: {sim_thin.N - 1}")
    print(f"Semi-major axis: 0.007 AU")
    print("Expected transit behavior:")
    print("  - Uniform ring structure")
    print("  - Smooth, periodic transits")
    print("  - Near edge-on geometry for deep transits")
    print("="*60)

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)