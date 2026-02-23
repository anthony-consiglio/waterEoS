"""
Unified water equation-of-state dispatcher.

Provides a single ``getProp(PT, model)`` interface that dispatches to:

  - ``'water1'``      -- SeaFreeze water1
  - ``'IAPWS95'``     -- SeaFreeze IAPWS-95
  - ``'holten2014'``  -- Holten, Sengers & Anisimov (2014)
  - ``'caupin2019'``  -- Caupin & Anisimov (2019)
  - ``'duska2020'``   -- Duska (2020) EOS-VaT
"""

_MODELS = ['water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020']

# Case-insensitive lookup  ->  canonical name
_CANONICAL = {name.lower(): name for name in _MODELS}


def list_models():
    """Return list of available model name strings."""
    return list(_MODELS)


def getProp(PT, model):
    """Compute thermodynamic properties for water.

    Parameters
    ----------
    PT : array-like
        Pressure (MPa) and temperature (K) input, in the format used by
        SeaFreeze (``np.array([P_array, T_array], dtype=object)``).
    model : str
        One of: 'water1', 'IAPWS95', 'holten2014', 'caupin2019', 'duska2020'.
        Matching is case-insensitive.

    Returns
    -------
    out : object
        Result object whose attributes (``rho``, ``Cp``, ``vel``, etc.) depend
        on the chosen backend.
    """
    canonical = _CANONICAL.get(model.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {', '.join(_MODELS)}"
        )

    if canonical == 'water1':
        from seafreeze.seafreeze import getProp as _sf
        return _sf(PT, 'water1')

    if canonical == 'IAPWS95':
        from seafreeze.seafreeze import getProp as _sf
        return _sf(PT, 'water_IAPWS95')

    if canonical == 'holten2014':
        from holten_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'caupin2019':
        from caupin_eos import getProp as _gp
        return _gp(PT)

    if canonical == 'duska2020':
        from duska_eos import getProp as _gp
        return _gp(PT)
