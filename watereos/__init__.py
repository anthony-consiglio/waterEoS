"""
waterEoS — thermodynamic properties of supercooled water.

Provides a unified interface to ten equation-of-state and transport
models for liquid water, including four primary two-state thermodynamic
EoS authors (Holten 2014, Caupin 2019 with optional Kim variant,
Duska 2020, Shi & Tanaka 2020) that predict a liquid--liquid critical
point (LLCP) in the deeply supercooled regime.

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

Available models: ``'holten2014'``, ``'caupin2019'``, ``'caupin2019_kim'``,
``'duska2020'``, ``'shi_tanaka2020'``, ``'shi_tanaka2020_transport'``,
``'grenke2025'``, ``'singh2017'``, ``'singh2017_caupin2019'``,
``'singh2017_duska2020'``, ``'water1'``, ``'IAPWS95'``.

See ``watereos.model_registry.MODEL_REGISTRY`` for metadata on each model,
or call ``list_models()`` for a quick summary. Call ``backend()`` to see
which backend (Rust/JAX/numpy) is dispatching each two-state model.
"""

from .watereos import getProp, list_models, compute
from .tv_phase_diagram import compute_tv_phase_diagram, compute_isochore
from .model_registry import MODEL_REGISTRY, MODEL_ORDER, ModelInfo

# One-time warning if the Rust backend failed to import. Without Rust the
# two-state models silently fall back to JAX (if installed) or pure-Python
# numpy, both of which can be 2-5x slower than the published wheel's default.
try:
    import watereos_rs as _watereos_rs  # noqa: F401
except ImportError:
    import warnings as _warnings
    _warnings.warn(
        "watereos_rs (Rust backend) is not available; two-state EoS models "
        "will use the slower JAX or numpy fallback. Reinstall the wheel from "
        "PyPI or build the Rust extension to restore the fast path. "
        "Call watereos.backend() to confirm which backend each model is using.",
        RuntimeWarning,
        stacklevel=2,
    )


def backend(model=None):
    """Return which backend (rust/jax/numpy) is being used per two-state model.

    Parameters
    ----------
    model : str or None
        If None (default), return a dict mapping every two-state model key
        to its active backend string. Otherwise return the backend string
        for the named model (e.g. ``'caupin2019'``, ``'holten2014'``,
        ``'duska2020'``, ``'caupin2019_kim'``).

    Returns
    -------
    dict or str
        Backend label(s): ``'rust'``, ``'jax'``, ``'numpy'``, ``'unknown'``,
        or ``'error: <msg>'`` if the module failed to import.

    Notes
    -----
    Useful for verifying that the fast Rust path is actually engaged. If
    ``watereos_rs`` failed to import you may be silently paying a 2-5x
    slowdown despite the package being installed.
    """
    import importlib
    result = {}
    for mkey, modname in [('caupin2019', 'caupin_eos.caupin_eos'),
                          ('caupin2019_kim', 'caupin_eos.caupin_kim_eos'),
                          ('holten2014', 'holten_eos.holten_eos'),
                          ('duska2020', 'duska_eos.duska_eos'),
                          ('shi_tanaka2020', 'shi_tanaka_eos.shi_tanaka_eos')]:
        try:
            mod = importlib.import_module(modname)
            cb = getattr(mod, 'compute_batch', None)
            if cb is None:
                result[mkey] = 'unknown'
                continue
            origin = getattr(cb, '__module__', '') or ''
            if 'watereos_rs' in origin:
                result[mkey] = 'rust'
            elif 'core_ad' in origin:
                result[mkey] = 'jax'
            elif 'core' in origin:
                result[mkey] = 'numpy'
            elif modname == 'caupin_eos.caupin_kim_eos' and origin == modname:
                # caupin_kim's fallback wrapper just rebinds params on top of
                # the numpy core, so the module name attribution is itself.
                result[mkey] = 'numpy'
            else:
                result[mkey] = 'unknown'
        except Exception as exc:
            result[mkey] = f'error: {exc}'
    if model is None:
        return result
    return result.get(model, 'unknown')


__all__ = ['getProp', 'compute', 'list_models', 'compute_tv_phase_diagram',
           'compute_isochore', 'MODEL_REGISTRY', 'MODEL_ORDER', 'ModelInfo',
           'backend']
