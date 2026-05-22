# mathematical function tools 
import numpy as np
import scipy
from scipy.special import erf, erfinv


def sample_ecc_powerlaw(alpha=-0.5, emax=0.3, size=None):
    if size is None:
        r = np.random.uniform()
        return (r * (emax ** (alpha + 1))) ** (1 / (alpha + 1))
    else:
        r = np.random.uniform(size=size)
        return (r * (emax ** (alpha + 1))) ** (1 / (alpha + 1))


# Add chabrier relation


def draw_chabrier_mass(min_mass, max_mass, rng=None):
    """
    Draw one stellar mass (M_sun) from a Chabrier-like IMF truncated to [min_mass, max_mass].
    - log-normal piece: 0 < m <= 1 M_sun (characteristic mass mc, width sigma_ln)
    - power-law piece: m > 1 M_sun with slope alpha (xi ~ m^(-alpha))
    Returns a float (single sample).
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (min_mass < max_mass):
        raise ValueError("min_mass must be < max_mass")
    if max_mass <= 0:
        raise ValueError("max_mass must be > 0")

    # Chabrier-ish parameters (adjust if you want exact paper values)
    mc = 0.2  # characteristic mass [M_sun]
    sigma_log10 = 0.55  # width in dex (log10)
    sigma_ln = sigma_log10 * np.log(10)  # convert to ln-space
    alpha = 2.35  # Salpeter-like slope for high mass tail

    sqrt2 = np.sqrt(2.0)

    # helpers for normal CDF and inverse via erf / erfinv (numpy provides erfinv)
    def normal_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt2))

    def lognormal_cdf_at(m):
        # CDF of lognormal with ln-mean = ln(mc) and sigma = sigma_ln
        # valid for m>0
        return normal_cdf((np.log(m) - np.log(mc)) / sigma_ln)

    # integrate log-normal piece on [a,b] (a>0)
    def lognormal_mass_between(a, b):
        if b <= a:
            return 0.0
        return lognormal_cdf_at(b) - lognormal_cdf_at(a)

    # integrate power-law m^-alpha on [a,b] (alpha != 1)
    def powerlaw_mass_between(a, b):
        if b <= a:
            return 0.0
        if np.isclose(alpha, 1.0):
            # special-case alpha==1: integral ln(b/a)
            return np.log(b / a)
        else:
            return (b ** (1.0 - alpha) - a ** (1.0 - alpha)) / (1.0 - alpha)

    # split interval at 1 M_sun
    a_low = max(min_mass, 1e-12)  # avoid zero
    b_low = min(max_mass, 1.0)
    a_high = max(min_mass, 1.0)
    b_high = max_mass

    # compute normalization masses (unnormalized; lognormal piece already uses proper PDF form)
    mass_low = lognormal_mass_between(a_low, b_low) if b_low > a_low else 0.0
    mass_high = powerlaw_mass_between(a_high, b_high) if b_high > a_high else 0.0

    total = mass_low + mass_high
    if total == 0.0:
        raise ValueError(
            "No probability mass in the requested interval (check min/max)."
        )

    # choose which piece to sample from
    pick = rng.random()
    if (mass_low > 0.0) and (pick < (mass_low / total)):
        # sample from truncated lognormal on [a_low, b_low]
        u0 = lognormal_cdf_at(a_low)
        u1 = lognormal_cdf_at(b_low)
        u = rng.random() * (u1 - u0) + u0  # uniform in the truncated CDF interval
        # invert: ln(m) = ln(mc) + sqrt(2)*sigma_ln * erfinv(2u-1)
        z = erfinv(2.0 * u - 1.0)
        ln_m = np.log(mc) + sqrt2 * sigma_ln * z
        return float(np.exp(ln_m))
    else:
        # sample from truncated power law on [a_high, b_high]
        a = a_high
        b = b_high
        if a <= 0:
            raise ValueError("Power-law segment requires a>0")
        if np.isclose(alpha, 1.0):
            # CDF ∝ ln(m); invert CDF: m = a * exp(u * ln(b/a))
            u = rng.random()
            return float(a * np.exp(u * np.log(b / a)))
        else:
            # CDF(m) = (m^{1-alpha} - a^{1-alpha}) / (b^{1-alpha} - a^{1-alpha})
            u = rng.random()
            pow_a = a ** (1.0 - alpha)
            pow_b = b ** (1.0 - alpha)
            val = u * (pow_b - pow_a) + pow_a
            m = val ** (1.0 / (1.0 - alpha))
            return float(m)


def mass_to_radius(mass_solar):
    """Convert mass (in solar masses) to radius (in solar radii) using a simple mass-radius relation.

    Parameters
    ----------
    mass_solar : float
        Mass in units of solar masses (M_sun).

    Returns
    -------
    float
        Radius in units of solar radii (R_sun).

    Notes
    -----
    This is a very approximate relation and may not be accurate for all types of stars.
    For main-sequence stars, a common approximation is R ~ M^0.8 for M < 1 M_sun and R ~ M^0.57 for M > 1 M_sun.
    """
    if mass_solar <= 0:
        raise ValueError("Mass must be positive.")

    if mass_solar < 1.0:
        radius_solar = mass_solar**0.8
    else:
        radius_solar = mass_solar**0.57

    return radius_solar


def compute_effective_temp(mass):
    """
    Only for main sequence Stars
    """
    # --- Below main-sequence range ---
    if mass < 0.38:
        raise ValueError(f"Mass of {mass} doesn't fit the bounds of the simulation")
    # --- Main-sequence intervals ---
    elif 0.38 <= mass <= 1.05:
        alpha = 4.841
        beta = 0.8 if mass < 1.0 else 0.57
    elif 1.05 < mass <= 2.4:
        alpha = 4.328
        beta = 0.57
    elif 2.4 < mass <= 7.0:
        alpha = 3.962
        beta = 0.57
    elif 7.0 < mass <= 32.0:
        alpha = 2.726
        beta = 0.57
    # --- Above main-sequence range ---
    else:
        raise ValueError(f"Mass of {mass} doesn't fit the bounds of the simulation")

    gamma = (alpha - 2 * beta) / 4
    return 5772 * mass**gamma


def generate_alternating_magnitude_random_numbers(exponents=(-5, -6, -7), signed=False):
    """
    Return a single scalar whose order of magnitude (base-10 exponent) is exactly
    one of the exponents provided. The mantissa is sampled uniformly in [1, 10),
    so the resulting value lies in [1e-exp, 10e-exp) and will never drop to a
    smaller order.

    Args:
        exponents (iterable of int): allowed negative exponents, e.g. (-4,-5,-6).
        signed (bool): if True, randomly flip sign with 50% probability.

    Returns:
        float: a single random number whose order-of-magnitude is one of exponents in units of AU.
    """
    rng = np.random.default_rng()
    weights = [0.45, 0.45, 0.10]
    exp = int(rng.choice(list(exponents), p=np.array(weights) / np.sum(weights)))
    mantissa = rng.uniform(1.0, 10.0)  # in [1,10)
    value = mantissa * (10.0**exp)
    if signed and rng.random() < 0.5:
        value = -value
    return value


def logg_from_mass_radius(M_sun_units, R_sun_units):
    G = 6.67430e-8  # cm^3 g^-1 s^-2
    M_sun = 1.98847e33  # g
    R_sun = 6.957e10

    M = M_sun_units * M_sun
    R = R_sun_units * R_sun
    g = G * M / (R * R)
    return np.log10(g)


def convert_AU_to_km(a_AU):
    AU_in_km = 1.495978707e8  # meters
    return a_AU * AU_in_km


def convert_km_to_AU(a_km):
    AU_in_km = 1.495978707e8  # meters
    return a_km / AU_in_km


def convert_solar_to_kg(m_solar):
    M_sun_kg = 1.98847e30  # kg
    return m_solar * M_sun_kg


def convert_kg_to_solar(m_kg):
    M_sun_kg = 1.98847e30  # kg
    return m_kg / M_sun_kg


def convert_solar_radii_to_AU(r_solar):
    """takes radius of star in solar radii and returns it in AU"""
    R_sun_in_AU = 0.00465  # AU
    return r_solar * R_sun_in_AU


def convert_AU_to_solar_radius(r_AU):
    """takes radius of star in AU and returns it in solar radii"""
    R_Sun_in_AU = 0.00465  # AU
    return r_AU / R_Sun_in_AU


def calculate_period(a_AU, mass_solar):
    """Keplerian orbital period in days for a given semi-major axis (AU) and mass (Msun)."""
    G_solar = 0.00029591220828559104  # AU^3 / (M_sun * day^2)
    return 2 * np.pi * np.sqrt(a_AU**3 / (G_solar * mass_solar))

def sample_particle_radius(
    R_star_AU,
    N, 
    f_min=0.15, 
    f_max=0.4, 
    rng=None, 
    log_uniform=True,
    min_particle_fraction=0.005
):
    rng = np.random.default_rng() if rng is None else rng

    if log_uniform:
        logf = rng.uniform(np.log10(f_min), np.log10(f_max))
        f = 10.0**logf
    else:
        f = rng.uniform(f_min, f_max)

    r_particle = R_star_AU * np.sqrt(f / N)

    # Enforce minimum particle radius
    r_particle = max(r_particle, min_particle_fraction * R_star_AU)
    
    return r_particle, f
