"""
Equations of state fitted to computer simulations of model waters.

THESE DESCRIBE MODEL WATERS, NOT REAL WATER.  Each equation was fitted by its
authors to molecular-simulation results for one water potential (TIP4P/2005,
ST2, mW, WAIL, q-TIP4P/F, ML-BOP, a SCAN-based neural network, ...).  They
are included for comparison and visualization only.  Their critical points,
densities and energies are those of the model, and their Gibbs energies keep
each paper's own zero, so energies cannot be compared with real water.

Every model key names the source and the water model, for example
``'gartner2020_scan'``; the display names in
``watereos.model_registry.MODEL_REGISTRY`` say "model water" and the
``water_model`` field of each entry names the potential.

Usage::

    import numpy as np
    from model_water_eos import getProp, MODELS
    PT = np.array([np.array([0.1, 100.0]), np.array([250.0, 300.0])],
                  dtype=object)
    out = getProp(PT, 'gartner2020_scan')

Each submodule exposes ``MODELS`` (model key -> ``compute_batch``),
``BRANCH_MODELS`` (keys that accept ``branch='hdl'`` / ``'ldl'``) and the
printed parameters with their page references.
"""

import importlib

from watereos.batch_getprop import getProp as _getProp
from watereos.two_state_branch import normalize_branch

_SUBMODULES = (
    'singh2016',
    'biddle2017',
    'gartner2020',
    'holten2013',
    'holten2014',
    'weis2022',
    'eltareb2022',
    'zorzi2024',
    'yu2023',
    'handle2018',
    'eltareb2024',
)

MODELS = {}
BRANCH_MODELS = set()
_NAN_KEYS = {}

for _name in _SUBMODULES:
    _mod = importlib.import_module(f'{__name__}.{_name}')
    MODELS.update(_mod.MODELS)
    BRANCH_MODELS.update(getattr(_mod, 'BRANCH_MODELS', ()))
    _NAN_KEYS.update(getattr(_mod, 'NAN_KEYS', {}))

# Two-state models, like the other two-state models of the package, do not
# compute dKt/dP in batch mode; it is returned as NaN.
_TWO_STATE_NAN = ('Kp', 'Kp_A', 'Kp_B')


def getProp(PT, model, branch='stable'):
    """Properties of a model water on SeaFreeze-style (P, T) input.

    Parameters
    ----------
    PT : numpy array
        Grid ``np.array([P_MPa, T_K], dtype=object)`` or scatter input.
    model : str
        A key of ``MODELS``.
    branch : {'stable', 'hdl', 'ldl'}
        Liquid branch, for the keys in ``BRANCH_MODELS``.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model water '{model}'. "
                         f"Choose from: {', '.join(sorted(MODELS))}")
    branch = normalize_branch(branch)
    nan_keys = _NAN_KEYS.get(model, _TWO_STATE_NAN)
    if model in BRANCH_MODELS:
        return _getProp(PT, MODELS[model], nan_keys=nan_keys, branch=branch)
    if branch != 'stable':
        raise ValueError(f"branch='{branch}' is not available for '{model}'")
    return _getProp(PT, MODELS[model], nan_keys=nan_keys)


__all__ = ['getProp', 'MODELS', 'BRANCH_MODELS']
