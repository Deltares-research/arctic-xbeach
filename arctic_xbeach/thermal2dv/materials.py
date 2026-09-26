"""Constituent properties and the enthalpy closure of the 2DV thermal model.

Every cell is a mixture of mineral solids (phi_s), total H2O (phi_h2o; liquid + ice,
counted as water-equivalent volume) and air (1 - phi_s - phi_h2o). The prognostic
variable is volumetric enthalpy H [J m-3] with the reference H = 0 for "fully frozen at
the freezing point" (CryoGrid-style, Westermann et al. 2023, GMD 16:2607):

    H = C_f (T - Tf)                  T < Tf        (frozen, sensible)
    H in [0, Lv]                      T = Tf        (latent plateau, sharp freezing)
    H = Lv + C_u (T - Tf)             T > Tf        (thawed, sensible)

with Lv = rho_w L_f phi_h2o. Tf depends on the (static) salinity of the cell. Ice
expansion is ignored (water-equivalent fractions), as in Arctic-XBeach v1.0.
Temperatures are in degrees Celsius throughout.
"""

import numpy as np

# === Constituent properties ===
RHO_W = 1000.0          # water density used for the latent heat [kg m-3]
L_F = 3.34e5            # latent heat of fusion [J kg-1]
LV_WATER = RHO_W * L_F  # volumetric latent heat of pure water [J m-3]

C_MINERAL = 2.0e6       # volumetric heat capacity [J m-3 K-1]
C_WATER = 4.2e6
C_ICE = 1.9e6
C_AIR = 1.2e3

K_MINERAL = 2.9         # thermal conductivity [W m-1 K-1]
K_WATER = 0.57
K_ICE = 2.2
K_AIR = 0.025

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W m-2 K-4]
T0K = 273.15


def freezing_point(S):
    """Freezing point of (pore) water [degC] for practical salinity S (UNESCO 1983, p = 0)."""
    S = np.asarray(S, dtype=float)
    return -0.0575 * S + 1.710523e-3 * S**1.5 - 2.154996e-4 * S**2


def heat_capacities(phi_s, phi_h2o):
    """Frozen and thawed volumetric heat capacity and latent heat content of each cell."""
    phi_a = np.clip(1.0 - phi_s - phi_h2o, 0.0, 1.0)
    c_frozen = phi_s * C_MINERAL + phi_h2o * C_ICE + phi_a * C_AIR
    c_thawed = phi_s * C_MINERAL + phi_h2o * C_WATER + phi_a * C_AIR
    lv = LV_WATER * phi_h2o
    return c_frozen, c_thawed, lv


def enthalpy_from_temperature(T, Tf, c_frozen, c_thawed, lv, frozen_at_tf=True):
    """Enthalpy for a given temperature. Cells exactly at Tf are taken as frozen (H = 0) unless
    frozen_at_tf is False (then fully thawed, H = Lv)."""
    H = np.where(T < Tf, c_frozen * (T - Tf), lv + c_thawed * (T - Tf))
    at_tf = T == Tf
    if np.any(at_tf):
        H = np.where(at_tf, 0.0 if frozen_at_tf else lv, H)
    return H


def temperature_from_enthalpy(H, Tf, c_frozen, c_thawed, lv):
    """Temperature and dT/dH (0 on the latent plateau) from enthalpy."""
    frozen = H < 0.0
    thawed = H > lv
    T = np.where(frozen, Tf + H / c_frozen, np.where(thawed, Tf + (H - lv) / c_thawed, Tf))
    dTdH = np.where(frozen, 1.0 / c_frozen, np.where(thawed, 1.0 / c_thawed, 0.0))
    return T, dTdH


def liquid_fraction(H, lv):
    """Fraction of the H2O that is liquid (1 for cells without H2O)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(lv > 0.0, np.clip(H / lv, 0.0, 1.0), 1.0)
    return f


def soil_conductivity(phi_s, phi_h2o, f_liq):
    """Geometric-mean (Farouki 1981) conductivity of a soil mixture. For saturated soil the
    Johansen (1975) and Cote & Konrad (2005) models reduce to this form."""
    theta_w = phi_h2o * f_liq
    theta_i = phi_h2o * (1.0 - f_liq)
    phi_a = np.clip(1.0 - phi_s - phi_h2o, 0.0, 1.0)
    return K_MINERAL**phi_s * K_WATER**theta_w * K_ICE**theta_i * K_AIR**phi_a
