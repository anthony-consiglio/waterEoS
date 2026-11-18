"""Bridge29 equilibrium Gibbs equation and finite-barrier transport model.

The Gibbs equation is in ``equation.py``. Pure-state Gibbs derivatives use
molar units, K and MPa. The public results use the specific SI units of
watereos, with bulk moduli in MPa.
"""
import json
from importlib.resources import files

import numpy as np
from scipy.special import expit

from .equation import G_SCALE, M, P_SCALE, R, Equation  # noqa: F401

with (files("watereos") / "data" / "bridge29.json").open() as stream:
    _data = json.load(stream)
EQUATION = Equation(_data["thermodynamics"])
Q, RM = EQUATION.Q, EQUATION.RM
PS, VS, PHI, CUBIC = EQUATION.PS, EQUATION.VS, EQUATION.PHI, EQUATION.CUBIC
_high, _low = EQUATION.high, EQUATION.low


def _transport(T, P):
    shape, dp = _data["shape"], P-0.1
    midpoint = (shape["T50_K"] + shape["slope_K_per_MPa"]*dp
                + shape["curvature_K_per_100_MPa_squared"]*(dp/100.)**2)
    f = expit((midpoint-T)*220/(T*shape["width_K"]))
    result = {"f": f}
    for key, c in _data["transport"].items():
        barrier = (1000*(c["high_E_1000K"] + c["increment_E_1000K"]*f)
                   + dp*c["pressure_E_K_per_MPa"])
        sign = -1 if key == "D" else 1
        prefactor = T/273.15 if key == "eta" else 1
        result[key] = prefactor*np.exp(c["intercept"] + sign*barrier/T)
    return result


def compute_batch(T, P, branch="stable", *, x=None):
    return EQUATION.compute_batch(T, P, branch, x=x, extra=_transport)
