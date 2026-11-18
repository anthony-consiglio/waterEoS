"""Liquid–liquid critical point, binodal and spinodals of Local26."""
from watereos.bridge29.phase import PhaseDiagram

from .core import EQUATION

_LOCAL26 = PhaseDiagram(EQUATION)
vapor_spinodal = _LOCAL26.vapor_spinodal
find_LLCP = _LOCAL26.find_LLCP
compute_spinodal_curve = _LOCAL26.compute_spinodal_curve
compute_binodal_curve = _LOCAL26.compute_binodal_curve
liquid_lines = _LOCAL26.liquid_lines
