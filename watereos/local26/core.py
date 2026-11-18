"""Local26: Bridge29's Gibbs equation refitted to liquid water at 175–300 K
and −50 to 250 MPa, with 26 coefficients.

The equation is in ``watereos/bridge29/equation.py`` and the coefficients
in ``watereos/data/local26.json``. Local26 has no transport model.
"""
import json
from importlib.resources import files

from watereos.bridge29.equation import Equation

with (files("watereos") / "data" / "local26.json").open() as stream:
    _data = json.load(stream)
EQUATION = Equation(_data["thermodynamics"])
Q, RM = EQUATION.Q, EQUATION.RM
#: Calibration window: temperature [K] and pressure [MPa].
T_RANGE = tuple(_data["calibration_window"]["T_K"])
P_RANGE = tuple(_data["calibration_window"]["p_MPa"])


def compute_batch(T, P, branch="stable", *, x=None):
    return EQUATION.compute_batch(T, P, branch, x=x)
