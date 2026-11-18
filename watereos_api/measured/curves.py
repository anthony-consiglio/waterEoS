"""Curves drawn over the measured data.

* Model isobars, isotherms and surfaces from the waterEoS registry.
* Reference lines that do not depend on a model: the IAPWS R14-08 melting
  curves of the stable ices, the homogeneous nucleation line of Holten,
  Sengers & Anisimov (2014, Appendix A), and the Murphy & Koop (2005) vapor
  pressures of supercooled water and ice Ih.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np

from watereos_api.measured import schema as S


def _finite_list(a):
    return [float(v) if math.isfinite(v) else None for v in np.asarray(a, float).ravel().tolist()]


# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------
def p_melt(T, ice):
    """Melting pressure (MPa) of ice ``ice`` at T (K), IAPWS R14-08(2011)."""
    if ice == 'Ih':
        th = T / 273.16
        a = (0.119539337e7, 0.808183159e5, 0.333826860e4)
        b = (0.300000e1, 0.257500e2, 0.103750e3)
        return 611.657e-6 * (1 + sum(ai * (1 - th ** bi) for ai, bi in zip(a, b)))
    if ice == 'III':
        return 208.566 * (1 - 0.299948 * (1 - (T / 251.165) ** 60))
    if ice == 'V':
        return 350.100 * (1 - 1.18721 * (1 - (T / 256.164) ** 8))
    if ice == 'VI':
        return 632.400 * (1 - 1.07476 * (1 - (T / 273.31) ** 4.6))
    if ice == 'VII':
        th = T / 355.0
        return 2216.0 * np.exp(1.73683 * (1 - 1 / th) - 0.0544606 * (1 - th ** 5) + 0.806106e-7 * (1 - th ** 22))
    raise ValueError(ice)


# Stable branches between the triple points: (ice, T_lo, T_hi) in K.
_MELT_BRANCHES = [('Ih', 251.165, 273.16), ('III', 251.165, 256.164), ('V', 256.164, 273.31),
                  ('VI', 273.31, 355.0), ('VII', 355.0, 715.0)]


def holten_th_line():
    """T_H(P) of Holten et al. 2014, Appendix A: Eq. (A1) up to 200 MPa (and into
    negative pressure down to -60 MPa), Eq. (A2) from 200 to 1000 MPa."""
    T1 = np.linspace(181.0, 240.0, 600)
    u = T1 / 235.15
    P1 = 0.1 * (1 + 2282.7 * (1 - u ** 6.243) + 157.24 * (1 - u ** 79.81))
    keep = (P1 <= 200.0) & (P1 >= -60.0)
    T1, P1 = T1[keep], P1[keep]
    P2 = np.linspace(200.0, 1000.0, 160)
    T2 = 172.82 + 0.03718 * P2 + 3.403e-5 * P2 ** 2 - 1.573e-8 * P2 ** 3
    order = np.argsort(P1)
    return np.concatenate([T1[order], T2]), np.concatenate([P1[order], P2])


def murphy_koop_liquid(T):
    T = np.asarray(T, float)
    lnp = (54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
           + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
    return np.exp(lnp) * 1e-6


def murphy_koop_ice(T):
    T = np.asarray(T, float)
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T) * 1e-6


@lru_cache(maxsize=1)
def reference_lines():
    melting = []
    for ice, lo, hi in _MELT_BRANCHES:
        T = np.linspace(lo, hi, 120)
        melting.append({'label': f'ice {ice}', 'T_K': _finite_list(T), 'P_MPa': _finite_list(p_melt(T, ice))})
    Tth, Pth = holten_th_line()
    Tl = np.linspace(123.0, 332.0, 300)
    Ti = np.linspace(110.0, 273.16, 300)
    return {
        'melting': {'source': 'IAPWS R14-08(2011)', 'branches': melting},
        'th': {'source': 'Holten, Sengers & Anisimov (2014), Appendix A', 'T_K': _finite_list(Tth),
               'P_MPa': _finite_list(Pth)},
        'vapor': {'source': 'Murphy & Koop (2005)', 'branches': [
            {'label': 'liquid', 'T_K': _finite_list(Tl), 'P_MPa': _finite_list(murphy_koop_liquid(Tl))},
            {'label': 'ice Ih', 'T_K': _finite_list(Ti), 'P_MPa': _finite_list(murphy_koop_ice(Ti))},
        ]},
    }


# ---------------------------------------------------------------------------
# Model curves
# ---------------------------------------------------------------------------
def overlay_models(prop):
    """[(key, display_name)] of registry models that can be drawn over ``prop``.

    Only equations for real water: the registry's fits to simulated model
    waters (``is_model_water``) and the vapor-phase model are left out, since
    the data are measurements on real H2O.
    """
    from watereos.model_registry import MODEL_ORDER, MODEL_REGISTRY
    spec = S.OVERLAY_PROPERTIES.get(prop)
    if spec is None:
        return []
    attr = spec[0]
    out = []
    for k in MODEL_ORDER:
        info = MODEL_REGISTRY[k]
        if k in S.OVERLAY_EXCLUDED_MODELS or getattr(info, 'is_model_water', False):
            continue
        if attr in info.properties:
            out.append((k, info.display_name))
    return out


def _model_info(model):
    from watereos.model_registry import MODEL_REGISTRY
    return MODEL_REGISTRY[model]


def _evaluate(model, attr, invert, T, P):
    """Property on the T x P grid, shape (len(P), len(T)); NaN where the model fails."""
    import watereos
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            out = watereos.compute(T_K=list(map(float, T)), P_MPa=list(map(float, P)), model=model)
            Z = np.asarray(getattr(out, attr), float).reshape(len(P), len(T))
        except Exception:  # noqa: BLE001 - a model that fails draws nothing
            Z = np.full((len(P), len(T)), np.nan)
    if invert:
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = 1.0 / Z
    Z[~np.isfinite(Z)] = np.nan
    return Z


@lru_cache(maxsize=512)
def model_curve(model, prop, along, fixed, lo, hi, n):
    """Along an isobar (along='T', fixed = P in MPa) or an isotherm (along='P', fixed = T in K)."""
    attr, invert = S.OVERLAY_PROPERTIES[prop]
    x = np.linspace(lo, hi, n)
    info = _model_info(model)
    if along == 'T':
        y = _evaluate(model, attr, invert, x, [fixed])[0]
        inr = (x >= info.T_min) & (x <= info.T_max) & (info.P_min <= fixed <= info.P_max)
    else:
        y = _evaluate(model, attr, invert, [fixed], x)[:, 0]
        inr = (x >= info.P_min) & (x <= info.P_max) & (info.T_min <= fixed <= info.T_max)
    return {'model': model, 'fixed': fixed, 'x': _finite_list(x), 'y': _finite_list(y),
            'in_range': [bool(b) for b in inr]}


@lru_cache(maxsize=64)
def model_surface(model, prop, T_lo, T_hi, P_lo, P_hi, n):
    attr, invert = S.OVERLAY_PROPERTIES[prop]
    T = np.linspace(T_lo, T_hi, n)
    P = np.linspace(P_lo, P_hi, n)
    Z = _evaluate(model, attr, invert, T, P)
    info = _model_info(model)
    inr = ((T[None, :] >= info.T_min) & (T[None, :] <= info.T_max)
           & (P[:, None] >= info.P_min) & (P[:, None] <= info.P_max))
    return {'model': model, 'T_K': _finite_list(T), 'P_MPa': _finite_list(P),
            'z': [_finite_list(row) for row in Z], 'in_range': [[bool(b) for b in row] for row in inr]}
