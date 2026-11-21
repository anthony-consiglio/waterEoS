"""Liquid–liquid critical point, binodal and spinodals of Bridge37."""
from watereos.bridge29.phase import PhaseDiagram

from .core import EQUATION

_BRIDGE37 = PhaseDiagram(EQUATION)
vapor_spinodal = _BRIDGE37.vapor_spinodal
find_LLCP = _BRIDGE37.find_LLCP
compute_spinodal_curve = _BRIDGE37.compute_spinodal_curve
compute_binodal_curve = _BRIDGE37.compute_binodal_curve
liquid_lines = _BRIDGE37.liquid_lines
