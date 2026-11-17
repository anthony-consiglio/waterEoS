"""
Equations fitted to simulations of model waters (package ``model_water_eos``).

Checks that apply to every one of them:

- the registry and the package list the same keys, and each registry entry
  is labeled as a model water (``water_model`` set, "model water" in the
  display name) and comes after the real-water models in ``MODEL_ORDER``;
- the dispatcher evaluates each key, and branch selection is accepted only
  for the keys that support it;
- the returned properties are thermodynamically consistent: for keys that
  return a Gibbs energy, S = -(dG/dT)_P, V = (dG/dP)_T, Cp = T (dS/dT)_P;
  for every key, Kt and alpha are the derivatives of the returned V.

Paper-specific check values live in ``tests/test_model_water_<paper>.py``.
"""

import numpy as np
import pytest

import watereos
from model_water_eos import BRANCH_MODELS, MODELS
from watereos.model_registry import MODEL_ORDER, MODEL_REGISTRY, model_water_keys

KEYS = sorted(MODELS)


def test_registry_and_package_agree():
    assert set(model_water_keys()) == set(MODELS)


@pytest.mark.parametrize('key', KEYS)
def test_labeled_as_model_water(key):
    info = MODEL_REGISTRY[key]
    assert info.is_model_water
    assert 'model water' in info.display_name
    assert info.water_model and info.water_model in info.display_name


def test_model_waters_listed_after_real_water():
    flags = [MODEL_REGISTRY[k].is_model_water for k in MODEL_ORDER]
    first = flags.index(True)
    assert all(flags[first:])
    assert not any(MODEL_REGISTRY[k].is_model_water
                   for k in MODEL_REGISTRY if k not in MODEL_ORDER)


@pytest.mark.parametrize('key', KEYS)
def test_dispatch_and_branch_support(key):
    PT = np.array([np.array([50.0, 150.0]), np.array([260.0, 280.0])],
                  dtype=object)
    out = watereos.getProp(PT, key)
    assert out.rho.shape == (2, 2)
    if key in BRANCH_MODELS:
        hdl = watereos.getProp(PT, key, branch='hdl')
        assert hdl.metastable.shape == (2, 2)
    else:
        with pytest.raises(ValueError):
            watereos.getProp(PT, key, branch='hdl')


def _points(key):
    """A spread of (T, P) points inside the registry range of a model."""
    info = MODEL_REGISTRY[key]
    T = np.linspace(info.T_min, info.T_max, 7)[1:-1]
    P = np.linspace(info.P_min, info.P_max, 7)[1:-1]
    TT, PP = np.meshgrid(T, P)
    return TT.ravel(), PP.ravel()


def _branches(key):
    return ['stable', 'hdl', 'ldl'] if key in BRANCH_MODELS else ['stable']


def _usable(*arrays):
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok


@pytest.mark.parametrize('key', KEYS)
def test_volume_derivatives(key):
    fn = MODELS[key]
    T, P = _points(key)
    hT, hP = 1e-3, 1e-3
    for branch in _branches(key):
        kw = {} if branch == 'stable' else {'branch': branch}
        o = fn(T, P, **kw)
        Vtp = fn(T + hT, P, **kw)['V']
        Vtm = fn(T - hT, P, **kw)['V']
        Vpp = fn(T, P + hP, **kw)['V']
        Vpm = fn(T, P - hP, **kw)['V']
        ok = _usable(o['V'], Vtp, Vtm, Vpp, Vpm)
        # stay off the liquid-liquid transition on the default branch
        if branch == 'stable' and 'x' in o:
            xs = [fn(T + d, P)['x'] for d in (-hT, hT)] + \
                 [fn(T, P + d)['x'] for d in (-hP, hP)]
            ok &= np.all([np.abs(x - o['x']) < 0.05 for x in xs], axis=0)
        if not ok.any():
            continue
        alpha = (Vtp - Vtm) / (2 * hT) / o['V']
        Kt = -o['V'] / ((Vpp - Vpm) / (2 * hP))
        np.testing.assert_allclose(alpha[ok], o['alpha'][ok], rtol=2e-5,
                                   atol=1e-9, err_msg=f'{key} {branch}')
        np.testing.assert_allclose(Kt[ok], o['Kt'][ok], rtol=2e-5,
                                   err_msg=f'{key} {branch}')


@pytest.mark.parametrize('key', KEYS)
def test_gibbs_derivatives(key):
    fn = MODELS[key]
    T, P = _points(key)
    probe = fn(T[:1], P[:1])
    if 'G' not in probe:
        pytest.skip(f'{key} has no caloric part')
    hT, hP = 1e-2, 1e-2
    for branch in _branches(key):
        kw = {} if branch == 'stable' else {'branch': branch}
        o = fn(T, P, **kw)
        tp, tm = fn(T + hT, P, **kw), fn(T - hT, P, **kw)
        pp, pm = fn(T, P + hP, **kw), fn(T, P - hP, **kw)
        ok = _usable(o['G'], tp['G'], tm['G'], pp['G'], pm['G'])
        if branch == 'stable' and 'x' in o:
            for nb in (tp, tm, pp, pm):
                ok &= np.abs(nb['x'] - o['x']) < 0.05
        if not ok.any():
            continue
        S = -(tp['G'] - tm['G']) / (2 * hT)
        V = (pp['G'] - pm['G']) / (2 * hP * 1e6)
        Cp = T * (tp['S'] - tm['S']) / (2 * hT)
        scale = np.maximum(np.abs(o['S']), 1.0)
        np.testing.assert_allclose(S[ok] / scale[ok], o['S'][ok] / scale[ok],
                                   atol=1e-5, err_msg=f'{key} {branch} S')
        np.testing.assert_allclose(V[ok], o['V'][ok], rtol=1e-6,
                                   err_msg=f'{key} {branch} V')
        np.testing.assert_allclose(Cp[ok], o['Cp'][ok], rtol=1e-4,
                                   err_msg=f'{key} {branch} Cp')
