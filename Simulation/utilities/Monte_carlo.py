import jax
import jax.numpy as jnp

# Tunable static chunk size (Python int). Keep as a module-level constant so it's static
# in JITted code. Increase for GH200 (more memory) to reduce kernel launches.
_CHUNK_SIZE = 256


# --- Utilities / core math (unchanged semantics) ---
def return_sky_basis(observer, eps=1e-12):
    """
    Returns orthonormal basis (u, v, k) where:
    - k points from star toward observer
    - u, v span the plane perpendicular to k
    """
    k = jnp.asarray(observer, dtype=jnp.float32)
    k = k / (jnp.linalg.norm(k) + eps)

    ex = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    ey = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    ez = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    candidates = jnp.stack([ex, ey, ez], axis=0)  # (3,3)

    dots = jnp.abs(candidates @ k)  # (3,)
    idx = jnp.argmin(dots)  # index of least-aligned axis
    a = candidates[idx]

    u = jnp.cross(a, k)
    u = u / (jnp.linalg.norm(u) + eps)
    v = jnp.cross(k, u)
    v = v / (jnp.linalg.norm(v) + eps)

    return u, v, k

# Not Used. Depracated:
def sample_observer_edge_biased(key, n=1, max_cos=0.1):
    """
    Sample observer directions biased toward edge-on views.
    Returns (obs, new_key) where:
      obs: array of shape (n,3) with observer direction vectors
      new_key: the next PRNGKey to use
    """
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 2)
    key_u, key_phi = keys
    u = jax.random.uniform(key_u, (n,), minval=-max_cos, maxval=max_cos)
    phi = jax.random.uniform(key_phi, (n,), minval=0.0, maxval=2 * jnp.pi)
    sinth = jnp.sqrt(jnp.clip(1.0 - u * u, 0.0, 1.0))
    obs = jnp.stack([sinth * jnp.cos(phi), sinth * jnp.sin(phi), u], axis=-1)
    return obs.astype(jnp.float32), key


def compute_intensities(mu, u1, u2):
    return jnp.clip(1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2, 0.0, None)


def make_all_disk_samples(master_key, n_trials, n_samples, R_star):
    keys = jax.random.split(master_key, n_trials)

    def one_key_sample(key):
        key_u, key_theta = jax.random.split(key)
        u = jax.random.uniform(key_u, (n_samples,))
        r = R_star * jnp.sqrt(u)
        theta = 2 * jnp.pi * jax.random.uniform(key_theta, (n_samples,))
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        mu = jnp.sqrt(jnp.clip(1.0 - (r / R_star) ** 2, 0.0, 1.0))
        return jnp.stack([x, y, mu], axis=-1)

    samples = jax.vmap(one_key_sample)(keys)  # (n_trials, n_samples, 3)
    x_all = samples[..., 0].astype(jnp.float32)
    y_all = samples[..., 1].astype(jnp.float32)
    mu_all = samples[..., 2].astype(jnp.float32)
    return x_all, y_all, mu_all, keys


@jax.jit
def prune_to_transiting(star_radius, positions, radii, observer):
    """
    Keep only particles that are:
    1. In front of the star (pz > 0, i.e., on observer side)
    2. Within stellar disk when projected (px^2 + py^2 <= (R* + r)^2)
    """
    u_vec, v_vec, k_vec = return_sky_basis(observer)

    px = positions @ u_vec
    py = positions @ v_vec
    pz = positions @ k_vec

    maxdist2 = (star_radius + radii) ** 2
    # Keep particles in front (pz > 0) and within projected disk
    keep_mask = (pz > 0.0) & ((px * px + py * py) <= maxdist2)

    mask_f = keep_mask.astype(jnp.float32)
    mask_pos = mask_f[:, None]

    pos_masked = positions * mask_pos
    rad_masked = radii * mask_f

    return pos_masked, rad_masked, px, py, pz


# --- Core single-time kernel: robust chunk scanning using lax.scan ---
def _single_time_core(
    star_radius,
    positions_t,  # (M,3)
    radii,
    px,
    py,
    pz,  # (M,)
    u1,
    u2,
    x_s,
    y_s,
    mu,  # shapes: (n_trials, n_samples)
    observer,
):
    # --- Defensive normalization ---
    positions_t = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), positions_t)
    positions_t = jnp.asarray(positions_t, jnp.float32)

    if positions_t.ndim == 3:
        positions_t = positions_t.reshape(-1, 3)

    # enforce dtype
    positions_t = jnp.asarray(positions_t, jnp.float32)
    radii = jnp.asarray(radii, jnp.float32)
    x_s = jnp.asarray(x_s, jnp.float32)
    y_s = jnp.asarray(y_s, jnp.float32)
    mu = jnp.asarray(mu, jnp.float32)
    observer = jnp.asarray(observer, jnp.float32)
    u1 = jnp.asarray(u1, jnp.float32)
    u2 = jnp.asarray(u2, jnp.float32)
    star_radius = jnp.asarray(star_radius, jnp.float32)

    # --- Normalize shapes to (n_trials, n_samples) safely ---
    def _ensure_2d(arr):
        arr = jnp.asarray(arr)
        if arr.ndim == 1:
            return arr[None, :]
        elif arr.ndim >= 2:
            return arr.reshape((-1, arr.shape[-1]))

    x_s = _ensure_2d(x_s)
    y_s = _ensure_2d(y_s)
    mu = _ensure_2d(mu)

    # Check if any particles survived pruning
    any_keep = jnp.any(radii > 0.0)

    intens = compute_intensities(mu, u1, u2)  # (n_trials, n_samples)
    n_trials, n_samples = x_s.shape

    M_total = px.shape[0]
    chunk_size = int(_CHUNK_SIZE)
    num_chunks = (int(M_total) + chunk_size - 1) // chunk_size

    def no_block(_):
        return jnp.array(1.0, dtype=jnp.float32)

    def do_block(_):
        covered_init = jnp.zeros((n_trials, n_samples), dtype=bool)

        full_len = num_chunks * chunk_size
        pad_len = full_len - M_total

        px_pad = jnp.pad(px, (0, pad_len), constant_values=0.0)
        py_pad = jnp.pad(py, (0, pad_len), constant_values=0.0)
        pz_pad = jnp.pad(pz, (0, pad_len), constant_values=-1.0)  # behind for padding
        r_pad = jnp.pad(radii, (0, pad_len), constant_values=0.0)

        px_chunks = jnp.reshape(px_pad, (num_chunks, chunk_size))
        py_chunks = jnp.reshape(py_pad, (num_chunks, chunk_size))
        pz_chunks = jnp.reshape(pz_pad, (num_chunks, chunk_size))
        r_chunks = jnp.reshape(r_pad, (num_chunks, chunk_size))

        idxs = jnp.arange(num_chunks)

        def scan_body(covered, idx):
            px_c = jax.lax.dynamic_index_in_dim(px_chunks, idx, axis=0)
            py_c = jax.lax.dynamic_index_in_dim(py_chunks, idx, axis=0)
            pz_c = jax.lax.dynamic_index_in_dim(pz_chunks, idx, axis=0)
            r_c = jax.lax.dynamic_index_in_dim(r_chunks, idx, axis=0)

            px_c = jnp.reshape(px_c, (chunk_size,))
            py_c = jnp.reshape(py_c, (chunk_size,))
            pz_c = jnp.reshape(pz_c, (chunk_size,))
            r_c = jnp.reshape(r_c, (chunk_size,))

            base = idx * chunk_size
            local_idxs = base + jnp.arange(chunk_size)
            in_range = local_idxs < M_total

            # Particle exists and is in front
            keep_c = in_range & (r_c > 0.0) & (pz_c > 0.0)

            dx = x_s[:, None, :] - px_c[None, :, None]
            dy = y_s[:, None, :] - py_c[None, :, None]
            d2 = dx * dx + dy * dy

            r2 = (r_c**2)[None, :, None]
            keep_b = keep_c[None, :, None]

            mask = (d2 <= r2) & keep_b
            covered_chunk = jnp.any(mask, axis=1)

            covered = covered | covered_chunk
            return covered, None

        covered_final, _ = jax.lax.scan(scan_body, covered_init, idxs)
        visible_flux = jnp.sum(jnp.where(~covered_final, intens, 0.0), axis=1)
        total_flux = jnp.sum(intens, axis=1)
        frac = jnp.where(total_flux > 0.0, visible_flux / total_flux, 0.0)
        return jnp.mean(frac).astype(jnp.float32)

    flux = jax.lax.cond(any_keep, do_block, no_block, operand=None)
    return flux


@jax.jit
def flux_for_single_time_device(
    star_radius, positions_t, radii, u1, u2, x_s, y_s, mu, observer
):
    pos_pr, rad_pr, px_pr, py_pr, pz_pr = prune_to_transiting(
        star_radius, positions_t, radii, observer,
    )
    any_survivors = jnp.any(rad_pr > 0.0)
    return jax.lax.cond(
        any_survivors,
        lambda _: _single_time_core(
            star_radius,
            pos_pr,
            rad_pr,
            px_pr,
            py_pr,
            pz_pr,
            u1,
            u2,
            x_s,
            y_s,
            mu,
            observer
        ),
        lambda _: jnp.array(1.0, dtype=jnp.float32),
        operand=None,
    )


@jax.jit
def compute_fluxes_all_times_device(
    star_radius,
    positions_all,  # (T, N, 3)
    radii,  # (N,)
    u1,
    u2,
    x_s_all,
    y_s_all,
    mu_all,
    observer,
):
    per_time_fn = lambda pos_t: flux_for_single_time_device(
        star_radius, pos_t, radii, u1, u2, x_s_all, y_s_all, mu_all, observer
    )
    return jax.vmap(per_time_fn)(positions_all)


def compute_monte_carlo_flux(
    star_radius,
    positions_all,
    radii,
    u1,
    u2,
    x_s_all,
    y_s_all,
    mu_all,
    observer,
    max_times_per_chunk: int = 512,
):
    """
    Splits the T timesteps into device-friendly chunks; each chunk calls a single jitted kernel.
    Returns a jnp array of fluxes (length T).
    """
    radii_j = jnp.array(radii, dtype=jnp.float32)
    x_s_j = jnp.array(x_s_all, dtype=jnp.float32)
    y_s_j = jnp.array(y_s_all, dtype=jnp.float32)
    mu_j = jnp.array(mu_all, dtype=jnp.float32)
    observer_j = jnp.array(observer, dtype=jnp.float32)

    T = positions_all.shape[0]
    flux_list = []
    bstart = 0
    while bstart < T:
        bend = min(bstart + max_times_per_chunk, T)
        pos_chunk = jnp.array(
            positions_all[bstart:bend], dtype=jnp.float32
        )
        flux_chunk = compute_fluxes_all_times_device(
            star_radius,
            pos_chunk,
            radii_j,
            u1,
            u2,
            x_s_j,
            y_s_j,
            mu_j,
            observer_j,
        )
        flux_list.append(flux_chunk)
        bstart = bend

    return jnp.concatenate(flux_list, axis=0)


# =======================
# Testing the algorithm:
# =======================

# Incorrect need to fix to match sim.py 

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import random
    import sys

    jax.config.update("jax_enable_x64", False)

    def run_tests():
        """
        Comprehensive test suite for the flux computation script.
        """
        print("=== Starting Flux Computation Tests ===")
        print(f"JAX version: {jax.__version__}")
        print(f"Platform: {jax.devices()[0]}")
        print("======================================")

        master_key = random.key(42)
        key_samples, key_obs, key_multi = random.split(master_key, 3)

        R_star = 1.0
        n_trials, n_samples = 8, 4096
        u1, u2 = 0.3, 0.1
        r_p = 0.1 * R_star

        x_s_all, y_s_all, mu_all, _ = make_all_disk_samples(
            key_samples, n_trials, n_samples, R_star
        )

        obs, key_obs = sample_observer_edge_biased(key_obs, n=1, max_cos=0.1)
        observer_random = obs[0]
        incl_deg = jnp.degrees(jnp.arccos(observer_random[2]))

        print(f"Random observer vector: {observer_random}, inclination ≈ {incl_deg:.2f}°")
        print(f"Monte Carlo samples: {n_trials} trials, {n_samples} samples per trial")

        # ---------------------------------------------
        # Test 1: Single-particle sanity checks
        # ---------------------------------------------
        print("\n=== Test 1: Single-Particle Sanity Checks ===")

        # Deterministic observer along +Z (face-on, looking from +Z toward origin)
        observer = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

        # CORRECTED: Particle in front (positive Z, closer to observer)
        positions_front = jnp.array([[0.0, 0.0, 0.5]], dtype=jnp.float32)
        radii_single = jnp.array([r_p], dtype=jnp.float32)

        flux_front = flux_for_single_time_device(
            R_star,
            positions_front,
            radii_single,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )

        # CORRECTED: Particle behind star (negative Z, farther from observer)
        positions_behind = jnp.array([[0.0, 0.0, -0.5]], dtype=jnp.float32)
        flux_behind = flux_for_single_time_device(
            R_star,
            positions_behind,
            radii_single,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )

        # Particle off-edge (in front but outside stellar disk)
        positions_off = jnp.array([[2.0, 0.0, 0.5]], dtype=jnp.float32)
        flux_off = flux_for_single_time_device(
            R_star,
            positions_off,
            radii_single,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )

        print(f"Front particle flux: {float(flux_front):.6f} (expect <1)")
        print(f"Behind particle flux: {float(flux_behind):.6f} (expect 1.0)")
        print(f"Off-edge particle flux: {float(flux_off):.6f} (expect 1.0)")

        assert flux_front < 1.0, f"Front particle should cause occultation (flux < 1), got {flux_front}"
        assert jnp.isclose(
            flux_behind, 1.0, atol=1e-3
        ), f"Behind particle should have flux = 1, got {flux_behind}"
        assert jnp.isclose(
            flux_off, 1.0, atol=1e-3
        ), f"Off-edge particle should have flux = 1, got {flux_off}"

        # ---------------------------------------------
        # Test 2: Pruning function checks
        # ---------------------------------------------
        print("\n=== Test 2: Pruning Function Checks ===")

        observer_center = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        # CORRECTED: In front particle
        pos_center = jnp.array([[0.0, 0.0, 0.5]], dtype=jnp.float32)
        rad_center = jnp.array([0.001], dtype=jnp.float32)

        p_pruned_c, r_pruned_c, _, _, _ = prune_to_transiting(
            R_star, pos_center, rad_center, observer_center
        )
        kept_center = jnp.any(r_pruned_c > 0.0)
        print(f"Center particle kept: {kept_center} (expect True)")

        pos_off = jnp.array([[2.0, 0.0, 0.5]], dtype=jnp.float32)
        p_pruned_o, r_pruned_o, _, _, _ = prune_to_transiting(
            R_star, pos_off, rad_center, observer_center
        )
        kept_off = jnp.any(r_pruned_o > 0.0)
        print(f"Off-edge particle kept: {kept_off} (expect False)")

        assert kept_center, "Center particle should be kept"
        assert not kept_off, "Off-edge particle should be pruned"

        # ---------------------------------------------
        # Test 3: Multi-particle debug case
        # ---------------------------------------------
        print("\n=== Test 3: Multi-Particle Debug Case ===")

        # CORRECTED: Particles in front
        positions_multi = jnp.array(
            [
                [0.0, 0.0, 0.5],
                [0.05, 0.0, 0.5],
                [0.12, 0.0, 0.5],
            ],
            dtype=jnp.float32,
        )
        radii_multi = jnp.array([0.08, 0.08, 0.08], dtype=jnp.float32)

        flux_multi = flux_for_single_time_device(
            R_star,
            positions_multi,
            radii_multi,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )
        print(
            f"Multi-particle flux: {float(flux_multi):.6f} (expect <1 due to overlapping occultations)"
        )

        assert flux_multi < 1.0, "Multi-particle case should cause occultation"

        # ---------------------------------------------
        # Test 4: Time-series Monte Carlo flux
        # ---------------------------------------------
        print("\n=== Test 4: Time-Series Monte Carlo Flux ===")

        T = 100
        x_positions = jnp.linspace(-1.5 * R_star, 1.5 * R_star, T)
        # CORRECTED: z=0.5 (in front)
        positions_time = jnp.stack(
            [x_positions, jnp.zeros(T), 0.5 * jnp.ones(T)], axis=-1
        )[:, None, :].astype(jnp.float32)
        radii_time = jnp.array([r_p], dtype=jnp.float32)

        fluxes_time = compute_monte_carlo_flux(
            R_star,
            positions_time,
            radii_time,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
            max_times_per_chunk=50,
        )

        print(f"Time-series flux shape: {fluxes_time.shape} (expect ({T},))")
        print(
            f"Time-series flux range: min={float(jnp.min(fluxes_time)):.6f}, max={float(jnp.max(fluxes_time)):.6f}"
        )

        mid_transit_idx = T // 2
        assert fluxes_time[mid_transit_idx] < 1.0, "Flux should dip during transit"
        assert jnp.allclose(fluxes_time[0], 1.0, atol=1e-2) and jnp.allclose(
            fluxes_time[-1], 1.0, atol=1e-2
        ), "Flux should be 1.0 outside transit"

        # ---------------------------------------------
        # Test 5: Edge case - zero-radius particles
        # ---------------------------------------------
        print("\n=== Test 5: Zero-Radius Particles ===")

        positions_zero = jnp.array([[0.0, 0.0, 0.5]], dtype=jnp.float32)
        radii_zero = jnp.array([0.0], dtype=jnp.float32)
        flux_zero = flux_for_single_time_device(
            R_star,
            positions_zero,
            radii_zero,
            u1,
            u2,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )
        print(f"Zero-radius particle flux: {float(flux_zero):.6f} (expect 1.000000)")

        assert jnp.isclose(
            flux_zero, 1.0, atol=1e-3
        ), "Zero-radius particle should have flux = 1"

        # ---------------------------------------------
        # Test 6: Numerical stability with extreme limb darkening
        # ---------------------------------------------
        print("\n=== Test 6: Extreme Limb Darkening ===")

        u1_extreme, u2_extreme = 0.9, 0.9
        flux_extreme = flux_for_single_time_device(
            R_star,
            positions_front,
            radii_single,
            u1_extreme,
            u2_extreme,
            x_s_all,
            y_s_all,
            mu_all,
            observer,
        )
        print(
            f"Extreme limb darkening flux: {float(flux_extreme):.6f} (expect <1, stable computation)"
        )

        assert jnp.isfinite(
            flux_extreme
        ), "Flux should be finite with extreme limb darkening"
        assert 0.0 <= flux_extreme <= 1.0, "Flux should be in [0,1]"

        print("\n=== All Tests Passed Successfully! ===")

    try:
        run_tests()
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)