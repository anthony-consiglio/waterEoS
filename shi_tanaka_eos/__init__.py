"""
Shi & Tanaka (2020) hierarchical two-state EoS for liquid water.

A two-state model that determines the LFTS (low-density tetrahedral) fraction
analytically (negligible cooperativity, J ≈ 0) and uses a constrained
13-parameter polynomial background for the DNLS (disordered normal-liquid)
state. Fitted to scattering, density, compressibility, expansion, and heat
capacity data over ≈ 240-320 K at ambient pressure with validation to negative
pressures down to -110 MPa.

Reference: R. Shi and H. Tanaka, "The anomalies and criticality of liquid
water", PNAS 117, 26591 (2020). All parameters from Table S2 of the SI.
"""

from .shi_tanaka_eos import getProp
from .core import compute_batch, compute_properties

__all__ = ['getProp', 'compute_batch', 'compute_properties']
