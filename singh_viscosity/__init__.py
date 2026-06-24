"""
Singh, Issenmann & Caupin (2017) two-state transport properties model.

Predicts viscosity (eta), self-diffusion coefficient (D), and rotational
correlation time (tau_r) using the LDS fraction f(T,P) from a two-state
thermodynamic model (Holten et al. 2014 by default).

Reference: L. P. Singh, B. Issenmann, F. Caupin, PNAS 114, 4312 (2017).
"""

from .singh_viscosity import getProp
from .core import compute_properties, compute_batch

__all__ = ['getProp', 'compute_properties', 'compute_batch']
