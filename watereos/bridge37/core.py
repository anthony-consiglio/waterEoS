"""Bridge37: Bridge29's Gibbs equation with quartic spinodal functions,
refitted to liquid water at 100-400 K and -140 to 400 MPa together with
Bridge29's amorphous-water and ice constraints.

The equation is in ``watereos/bridge29/equation.py`` and the coefficients
in ``watereos/data/bridge37.json``: the high-density state's spinodal
pressure, volume, curvature and cubic coefficient are Bernstein quartics in
T_eff on 100-420 K instead of Chebyshev quadratics. Bridge37 has no
transport model.
"""
import json
from importlib.resources import files

from watereos.bridge29.equation import BERNSTEIN_RANGE_K, Equation

with (files("watereos") / "data" / "bridge37.json").open() as stream:
    _data = json.load(stream)
_basis = _data["spinodal_basis"]
if _basis["kind"] != "bernstein" or tuple(_basis["T_eff_K"]) != BERNSTEIN_RANGE_K:
    raise ValueError(f"bridge37.json: unsupported spinodal basis {_basis}")
EQUATION = Equation(_data["thermodynamics"])
Q, RM = EQUATION.Q, EQUATION.RM
#: The fit's scope: temperature [K] and pressure [MPa].
T_RANGE = tuple(_data["scope"]["T_K"])
P_RANGE = tuple(_data["scope"]["p_MPa"])


def compute_batch(T, P, branch="stable", *, x=None):
    return EQUATION.compute_batch(T, P, branch, x=x)
