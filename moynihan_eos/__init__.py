"""
Moynihan (1996) two-species / nonideal-solution model for water.

Two interconvertible species, LDA-like and HDA-like, in a regular solution
with four constants fitted to Mishima's LDA <-> HDA transformation pressures
at 135 K and to the heat capacity, expansivity and compressibility of
supercooled water: dV0 = -4.5 cm3/mol, dH0 = 1940 J/mol,
dS0 = 0.98 R = 8.1 J/(mol K), W = 3610 J/mol.  Liquid-liquid critical point
at 217 K and 38 MPa.

The pure-species backgrounds are not modeled, so this is not a full
equation of state. It returns the low-density fraction ``x``, excess
responses ``Cp_ex``, ``V_alpha_ex`` and ``V_kappa_ex``, and approximate
``V``, ``rho``, ``alpha`` and ``Kt`` based on the paper's constant
background. See ``moynihan_eos.core`` for the phase-line functions and units.

Reference: C. T. Moynihan, MRS Proceedings 455, 411 (1996),
doi:10.1557/PROC-455-411.
"""

from .moynihan_eos import getProp
from .core import (
    compute_batch, compute_properties, critical_point, coexistence_pressure,
    coexistence_volumes, spinodal_pressures,
)

__all__ = ['getProp', 'compute_batch', 'compute_properties',
           'critical_point', 'coexistence_pressure', 'coexistence_volumes',
           'spinodal_pressures']
