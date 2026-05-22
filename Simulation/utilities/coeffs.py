# Getting limb darkening coefficients 
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator, LinearNDInterpolator
import json

G = 6.67430e-8  # cgs

# --- Load white dwarf table ---
df_wd = pd.read_fwf(
    "Data_Gen/tableab.dat",
    comment="#",
    names=["Mod", "logg", "Teff", "Z", "a", "b", "sig", "I", "Be", "Filter"]
)

wd_tess = df_wd.copy()

# --- Load main-sequence table (ATLAS, table25.dat) ---
df_main_seq = pd.read_fwf(
    "Data_Gen/table25.dat",
    comment="#",
    header=None,
    names=["logg", "Teff", "FeH", "vturb", 
           "u1_LSM", "u2_LSM", "u1_PCM", "u2_PCM", "chi2", "Mod"]
)

main_tess = df_main_seq.copy()

# --- Build scattered interpolation grid for white dwarfs ---
wd_points = np.column_stack((wd_tess["Teff"], wd_tess["logg"]))
u1_wd_vals = wd_tess["a"].astype(float)
u2_wd_vals = wd_tess["b"].astype(float)

# --- Build main-sequence interpolator ---

# Sort for deterministic reshaping
main_tess = main_tess.sort_values(["logg", "Teff", "FeH", "vturb"]).reset_index(drop=True) # Table is sorted in order (logg, Teff, FeH, vturb) to match reshape axes

loggs = np.unique(main_tess["logg"])
teffs = np.unique(main_tess["Teff"])
fehs  = np.unique(main_tess["FeH"])
vturs = np.unique(main_tess["vturb"])

expected_len = len(loggs) * len(teffs) * len(fehs) * len(vturs)
if expected_len == len(main_tess):
    u1_arr = main_tess["u1_LSM"].values.reshape((len(loggs), len(teffs), len(fehs), len(vturs)))
    u2_arr = main_tess["u2_LSM"].values.reshape((len(loggs), len(teffs), len(fehs), len(vturs)))
    u1_main_interp = RegularGridInterpolator((loggs, teffs, fehs, vturs), u1_arr, bounds_error=False, fill_value=None)
    u2_main_interp = RegularGridInterpolator((loggs, teffs, fehs, vturs), u2_arr, bounds_error=False, fill_value=None)
else:
    # scattered fallback
    pts = np.column_stack((main_tess["logg"], main_tess["Teff"], main_tess["FeH"], main_tess["vturb"]))
    u1_main_interp = LinearNDInterpolator(pts, main_tess["u1_LSM"].values)
    u2_main_interp = LinearNDInterpolator(pts, main_tess["u2_LSM"].values)

# --- Utilities ---
def logg_from_mass_radius(M_sun, R_sun):
    M = M_sun * 1.989e33   # g
    R = R_sun * 6.957e10   # cm
    g = G * M / R**2
    return np.log10(g)

# --- Sampling functions ---
def sample_ld_wd_coeffs(mass, radius, teff, sigma_model=0.02):
    """Quadratic LD coefficients for white dwarfs (TESS)."""
    logg = logg_from_mass_radius(mass, radius)
    pt = np.array([[teff, logg]])
    u1 = float(griddata(wd_points, u1_wd_vals, pt, method="linear"))
    u2 = float(griddata(wd_points, u2_wd_vals, pt, method="linear"))
    if np.isnan(u1) or np.isnan(u2):
        u1 = float(griddata(wd_points, u1_wd_vals, pt, method="nearest"))
        u2 = float(griddata(wd_points, u2_wd_vals, pt, method="nearest"))
    u1 += np.random.normal(0, sigma_model)
    u2 += np.random.normal(0, sigma_model)
    return np.clip(u1, -0.5, 1.5), np.clip(u2, -0.5, 1.5)

def sample_ld_main_coeffs(mass = None, radius = None, logg = None, teff=None, feh=0.0, vturb=2.0):
    """Quadratic LD coefficients for main-sequence stars (TESS)."""
    if logg is None:
        assert mass is not None and radius is not None
        logg = logg_from_mass_radius(mass, radius)
    pt = np.array([logg, teff, feh, vturb])
    u1 = float(u1_main_interp(pt))
    u2 = float(u2_main_interp(pt))
    if np.isnan(u1) or np.isnan(u2):
        u1 = float(np.nan_to_num(u1, nan=0.5))
        u2 = float(np.nan_to_num(u2, nan=0.1))
    return u1, u2

