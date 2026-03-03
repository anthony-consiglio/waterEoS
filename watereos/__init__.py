"""
waterEoS — thermodynamic properties of supercooled water.

Provides a unified interface to five equation-of-state models for
liquid water, including three two-state models that predict a
liquid--liquid critical point (LLCP) in the deeply supercooled regime.

Quick start
-----------
Use ``compute()`` for the simplest interface::

    from watereos import compute
    out = compute(T_K=250, P_MPa=100, model='duska2020')
    print(out.rho, out.Cp, out.G)

Or ``getProp()`` for SeaFreeze-compatible grid/scatter input::

    import numpy as np
    from watereos import getProp
    PT = np.array([np.array([0.1, 100, 200]),   # pressures (MPa)
                   np.array([250, 260, 270])],   # temperatures (K)
                  dtype=object)
    out = getProp(PT, 'holten2014')   # returns grid of shape (3, 3)

Available models: ``'holten2014'``, ``'caupin2019'``, ``'duska2020'``,
``'grenke2025'``, ``'singh2017'``, ``'water1'``, ``'IAPWS95'``.

See ``watereos.model_registry.MODEL_REGISTRY`` for metadata on each model,
or call ``list_models()`` for a quick summary.
"""

from .watereos import getProp, list_models, compute
from .tv_phase_diagram import compute_tv_phase_diagram, compute_isochore
from .model_registry import MODEL_REGISTRY, MODEL_ORDER, ModelInfo

__all__ = ['getProp', 'compute', 'list_models', 'compute_tv_phase_diagram',
           'compute_isochore', 'MODEL_REGISTRY', 'MODEL_ORDER', 'ModelInfo']
