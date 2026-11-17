import numpy as np
import pytest

from model_water_eos import holten2013 as mw


@pytest.mark.parametrize('key, expected', [(mw.KEY, 0.28), (mw.KEY_CLUSTERED, 0.18)])
def test_low_density_fraction_fig2(key, expected):
    result = mw.MODELS[key]([220.0], [0.1])
    assert result['x'][0] == pytest.approx(expected, abs=0.01)


def test_athermal_density_fig1():
    result = mw.MODELS[mw.KEY]([240.0, 240.0], [0.1, 1013.0])
    np.testing.assert_allclose(result['rho'], [1003.0, 1120.0], rtol=0, atol=2.0)


def test_clustered_heat_capacity_fig5():
    result = mw.MODELS[mw.KEY_CLUSTERED]([220.0], [0.1])
    assert result['Cp'][0] / 1000.0 == pytest.approx(3.2, abs=0.1)


def test_weak_crystallization_fig7():
    result = mw.MODELS[mw.KEY_WEAK_CRYSTALLIZATION]([220.0], [0.1])
    assert result['Cp'][0] / 1000.0 == pytest.approx(3.2, abs=0.1)
    assert result['rho'][0] == pytest.approx(998.0, abs=2.0)


def test_weak_crystallization_domain():
    run = mw.MODELS[mw.KEY_WEAK_CRYSTALLIZATION]
    # Eq. (26) gives a rounded stability limit of 175 - 0.093 P/MPa.
    expected_limit = 175.0 - 0.093 * 228.0
    actual_limit = mw.TS0 - mw.A_PRIME * 228e6 * mw.V0 / mw.R
    assert actual_limit == pytest.approx(expected_limit, abs=0.5)
    result = run([actual_limit - 0.01, actual_limit + 0.01], [228.0, 228.0])
    assert np.isnan(result['G'][0])
    assert np.isfinite(result['G'][1])
    with pytest.raises(ValueError, match='single liquid branch'):
        run([220.0], [0.1], branch='ldl')


@pytest.mark.parametrize('key', mw.MODELS)
def test_registry_probe(key):
    result = mw.MODELS[key]([273.15], [0.1])
    for prop in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp'):
        assert result[prop].shape == (1,)
        assert np.isfinite(result[prop][0])


@pytest.mark.parametrize(('key', 'branch'), [
    (key, branch)
    for key in mw.MODELS
    for branch in (('stable', 'hdl', 'ldl') if key in mw.BRANCH_MODELS else ('stable',))
])
def test_gibbs_derivatives(key, branch):
    run = mw.MODELS[key]
    if branch == 'ldl':
        T, P = np.array([185.0, 192.0, 195.0]), np.array([57.0, 0.1, 0.1])
    else:
        T, P = np.array([215.0, 240.0, 265.0]), np.array([57.0, 228.0, 810.0])
    dt, dp = 0.002, 0.002
    base = run(T, P, branch)
    hot, cold = run(T + dt, P, branch), run(T - dt, P, branch)
    high, low = run(T, P + dp, branch), run(T, P - dp, branch)
    for prop, derivative in {
        'S': -(hot['G'] - cold['G']) / (2.0 * dt),
        'V': (high['G'] - low['G']) / (2.0 * dp * 1e6),
        'Cp': T * (hot['S'] - cold['S']) / (2.0 * dt),
        'Kt': -2.0 * dp * base['V'] / (high['V'] - low['V']),
        'alpha': (hot['V'] - cold['V']) / (2.0 * dt * base['V']),
    }.items():
        assert np.isfinite(base[prop]).all(), prop
        np.testing.assert_allclose(base[prop], derivative, rtol=2e-6, atol=0, err_msg=prop)
    if branch != 'stable':
        assert not base['metastable'].any()
