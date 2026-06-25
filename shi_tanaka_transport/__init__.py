"""
Shi & Tanaka (2020) hierarchical transport properties for liquid water.

Provides viscosity (η), self-diffusion coefficient (D), and rotational
relaxation time (τ_R) from a generalized Arrhenius law coupled to the
*dynamic* order parameter s^D of the hierarchical two-state framework.
The thermodynamic backbone is the same Shi-Tanaka EoS (shi_tanaka_eos),
so transport and thermodynamics are internally consistent within one
model — unlike the Singh (2017) module which couples its Arrhenius law
to a Holten (2014) thermo backbone.

Reference: R. Shi and H. Tanaka, PNAS 117, 26591 (2020). Parameters from
Table S3 of the supporting information.
"""

from .shi_tanaka_transport import getProp
from .core import compute_batch, compute_properties

__all__ = ['getProp', 'compute_batch', 'compute_properties']
