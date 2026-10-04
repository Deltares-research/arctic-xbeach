"""Constituent properties and the enthalpy-salinity closure of the 2DV thermal model.

Every cell is a mixture of mineral solids (phi_s), total H2O (phi_h2o; liquid + ice, as
water-equivalent volume) and air (1 - phi_s - phi_h2o). It carries two conserved
quantities:

    H     volumetric enthalpy [J m-3], reference 0 for all-liquid at 0 degC (CryoGrid-style,
          Westermann et al. 2023, GMD 16:2607)
    salt  salt content = phi_h2o * S_bulk [psu x volume fraction], where S_bulk is the
          salinity of the H2O if it were all melted

Salt is excluded from ice, so it all sits in the liquid: S_liq = salt / theta_w. With a
linear liquidus Tf(S) = -m S (m = 0.054; Assur 1958, as in CICE / Bitz & Lipscomb 1999),
equilibrium below the bulk liquidus Tf_b = -m S_bulk requires T = -m S_liq, so

    theta_w(T) = m salt / (-T)                                  (salt-exclusion freezing curve)
    H(T) = (a + phi c_i) T - Lv phi - beta (c_w - c_i) - beta Lv_w / T,   beta = m salt

with a = phi_s c_s + phi_a c_a and Lv_w = rho_w L_f. For given (H, salt) this is a quadratic
in T, so T(H) is closed-form. Fresh H2O (salt = 0) keeps the sharp plateau at 0 degC. The
same curve gives mushy sea ice (brine pockets) and partly frozen saline permafrost.
Temperatures are in degrees Celsius; ice expansion is ignored, as in Arctic-XBeach v1.0.
"""

from types import SimpleNamespace

import numpy as np

# === Constituent properties ===
RHO_W = 1000.0          # water density used for the latent heat [kg m-3]
L_F = 3.34e5            # latent heat of fusion [J kg-1]
LV_WATER = RHO_W * L_F  # volumetric latent heat of pure water [J m-3]
M_LIQUIDUS = 0.054      # liquidus slope, Tf = -m S [K psu-1] (Assur 1958; CICE)

C_MINERAL = 2.0e6       # volumetric heat capacity [J m-3 K-1]
C_WATER = 4.2e6
C_ICE = 1.9e6
C_AIR = 1.2e3

K_MINERAL = 2.9         # thermal conductivity [W m-1 K-1]
K_WATER = 0.57
K_ICE = 2.2
K_AIR = 0.025

D_SALT = 1.0e-9         # molecular diffusivity of sea salt in water near 0 degC [m2 s-1]

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W m-2 K-4]
T0K = 273.15
_SALTY = 1e-12          # salt content below which a cell is treated as fresh (sharp plateau)


def freezing_point(S):
    """Linear liquidus [degC] for (brine) salinity S."""
    return -M_LIQUIDUS * np.asarray(S, dtype=float)


def unesco_freezing_point(S):
    """UNESCO (1983) seawater freezing point at p = 0, for reference (within 0.03 K of the
    linear liquidus for S = 0-40)."""
    S = np.asarray(S, dtype=float)
    return -0.0575 * S + 1.710523e-3 * S**1.5 - 2.154996e-4 * S**2


def heat_capacities(phi_s, phi_h2o):
    """Frozen and thawed volumetric heat capacity and latent heat content of a cell."""
    phi_a = np.clip(1.0 - phi_s - phi_h2o, 0.0, 1.0)
    c_frozen = phi_s * C_MINERAL + phi_h2o * C_ICE + phi_a * C_AIR
    c_thawed = phi_s * C_MINERAL + phi_h2o * C_WATER + phi_a * C_AIR
    return c_frozen, c_thawed, LV_WATER * phi_h2o


def closure(phi_s, phi_h2o, salt):
    """Per-cell coefficients of the H-T-salinity relation."""
    c_frozen, c_liquid, lv = heat_capacities(phi_s, phi_h2o)
    with np.errstate(divide="ignore", invalid="ignore"):
        S_bulk = np.where(phi_h2o > 0.0, salt / phi_h2o, 0.0)
    Tf = freezing_point(S_bulk)
    return SimpleNamespace(phi_h2o=phi_h2o, c_frozen=c_frozen, c_liquid=c_liquid, lv=lv,
                           beta=M_LIQUIDUS * salt, S_bulk=S_bulk, Tf=Tf, H_liq=c_liquid * Tf)


def temperature_from_enthalpy(H, c):
    """Temperature, dT/dH and liquid water content theta_w for enthalpy H and closure c."""
    liquid = H >= c.H_liq
    saline = c.beta > _SALTY
    A = c.c_frozen

    # all liquid
    T_liq = H / c.c_liquid
    # saline mushy: A T^2 + B T + C0 = 0, take the negative root in a cancellation-free form
    B = -c.lv - c.beta * (C_WATER - C_ICE) - H
    C0 = -c.beta * LV_WATER
    root = np.sqrt(np.maximum(B * B - 4.0 * A * C0, 0.0))
    q = -0.5 * (B + np.where(B >= 0.0, root, -root))
    with np.errstate(divide="ignore", invalid="ignore"):
        T_sal = np.where(B >= 0.0, q / A, C0 / q)
    # fresh: sharp plateau at 0 degC
    T_fresh = np.minimum(0.0, (H + c.lv) / A)

    T = np.where(liquid, T_liq, np.where(saline, T_sal, T_fresh))
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_sal = np.clip(c.beta / -T_sal, 0.0, c.phi_h2o)
        dTdH_sal = 1.0 / (A + c.beta * LV_WATER / T_sal**2)
        theta_fresh = np.where(c.lv > 0.0, np.clip(c.phi_h2o * (1.0 + H / c.lv), 0.0, c.phi_h2o), 0.0)
    theta_w = np.where(liquid, c.phi_h2o, np.where(saline, theta_sal, theta_fresh))
    dTdH = np.where(liquid, 1.0 / c.c_liquid,
                    np.where(saline, dTdH_sal, np.where(H < -c.lv, 1.0 / A, 0.0)))
    return T, dTdH, theta_w


def enthalpy_from_temperature(T, c, frozen_at_tf=True):
    """Enthalpy for temperature T. Fresh cells at exactly 0 degC are taken as frozen
    (H = -Lv) unless frozen_at_tf is False (then H = 0, all liquid)."""
    T = np.asarray(T, dtype=float)
    A = c.c_frozen
    with np.errstate(divide="ignore", invalid="ignore"):
        H_sal = A * T - c.lv - c.beta * (C_WATER - C_ICE) - c.beta * LV_WATER / T
    H_fresh = np.where(T < 0.0, A * T - c.lv, np.where(frozen_at_tf, -c.lv, 0.0))
    saline = c.beta > _SALTY
    return np.where(T >= c.Tf, np.where(saline | (T > 0.0), c.c_liquid * T, H_fresh),
                    np.where(saline, H_sal, H_fresh))


def liquid_fraction(theta_w, phi_h2o):
    """Fraction of the H2O that is liquid (1 for cells without H2O)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(phi_h2o > 0.0, np.clip(theta_w / phi_h2o, 0.0, 1.0), 1.0)


def soil_conductivity(phi_s, phi_h2o, f_liq):
    """Geometric-mean (Farouki 1981) conductivity of a soil mixture. For saturated soil the
    Johansen (1975) and Cote & Konrad (2005) models reduce to this form."""
    theta_w = phi_h2o * f_liq
    theta_i = phi_h2o * (1.0 - f_liq)
    phi_a = np.clip(1.0 - phi_s - phi_h2o, 0.0, 1.0)
    return K_MINERAL**phi_s * K_WATER**theta_w * K_ICE**theta_i * K_AIR**phi_a
