import numpy as np
import pytest

from model_water_eos import holten2014 as st2


# Tables I and III (mean field) and Tables II and IV (crossover).
CRITICAL_POINTS = {
    st2.KEY_ST2_I: (253.5, 160.0, 52.478),
    st2.KEY_ST2_II: (249.0, 146.0, 53.261),
    st2.KEY_ST2_I_CROSSOVER: (252.0, 168.0, 52.739),
    st2.KEY_ST2_II_CROSSOVER: (246.75, 155.0, 53.645),
}


@pytest.mark.parametrize('key', CRITICAL_POINTS)
def test_critical_point(key):
    T, P, molar_density = CRITICAL_POINTS[key]
    run = st2.MODELS[key]
    for branch in ('stable', 'hdl', 'ldl'):
        result = run([T], [P], branch)
        assert result['rho'][0] / (st2.M * 1000.0) == pytest.approx(
            molar_density, abs=0.0005
        )
        assert result['x'][0] == 0.5
        assert result['Kt'][0] == 0.0
        assert np.isposinf(result['Cp'][0])


@pytest.mark.parametrize('key, T, expected_p, expected_rho', [
    (st2.KEY_ST2_I, 240.0, 228.64, (1083.0, 916.0)),
    (st2.KEY_ST2_I_CROSSOVER, 240.0, 229.55, (1086.0, 918.0)),
])
def test_coexistence_fig5(key, T, expected_p, expected_rho):
    P = _coexistence_pressure(key, np.array([T]))
    assert P[0] == pytest.approx(expected_p, abs=0.01)
    run = st2.MODELS[key]
    hdl = run([T], P, 'hdl')
    ldl = run([T], P, 'ldl')
    np.testing.assert_allclose(hdl['G'], ldl['G'], rtol=0, atol=1e-7)
    assert not hdl['metastable'][0]
    assert not ldl['metastable'][0]
    assert hdl['rho'][0] == pytest.approx(expected_rho[0], abs=4.0)
    assert ldl['rho'][0] == pytest.approx(expected_rho[1], abs=4.0)


@pytest.mark.parametrize('key, expected', [
    (st2.KEY_ST2_II, [7.05, 5.90]),
    (st2.KEY_ST2_II_CROSSOVER, [7.44, 6.00]),
])
def test_critical_isobar_fig10(key, expected):
    Tc, Pc, _ = CRITICAL_POINTS[key]
    result = st2.MODELS[key]([Tc - 0.1, Tc + 0.1], [Pc, Pc])
    np.testing.assert_allclose(result['Cv'] / 1000.0, expected, rtol=0, atol=0.1)


@pytest.mark.parametrize('key', (st2.KEY_ST2_I_CROSSOVER, st2.KEY_ST2_II_CROSSOVER))
def test_crossover_branch_limits(key):
    run = st2.MODELS[key]
    cold = np.array([241.0])
    coexistence = _coexistence_pressure(key, cold)
    for branch, direction in [('hdl', -1.0), ('ldl', 1.0)]:
        local_minimum = run(cold, coexistence + direction * 0.01, branch)
        assert local_minimum['metastable'][0]
        past_spinodal = run(cold, coexistence + direction * 20.0, branch)
        assert np.isnan(past_spinodal['rho'][0])
        assert not past_spinodal['metastable'][0]
    warm = np.array([260.0])
    symmetry_pressure = _coexistence_pressure(key, warm)
    assert run(warm, symmetry_pressure)['x'][0] == 0.5
    assert run(warm, symmetry_pressure - 0.01)['x'][0] > 0.5
    assert run(warm, symmetry_pressure + 0.01)['x'][0] < 0.5


def _coexistence_pressure(key, T):
    Tc, Pc, molar_density = CRITICAL_POINTS[key]
    # Tables I-IV give the slopes through their different a values.
    a = {
        st2.KEY_ST2_I: 0.085811, st2.KEY_ST2_II: 0.1199,
        st2.KEY_ST2_I_CROSSOVER: 0.085496, st2.KEY_ST2_II_CROSSOVER: 0.12053,
    }[key]
    return Pc - (T - Tc) * molar_density * 1000.0 * st2.R / (a * 1e6)


@pytest.mark.parametrize('key', st2.MODELS)
def test_registry_probe(key):
    result = st2.MODELS[key]([273.15], [0.1])
    for prop in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp'):
        assert result[prop].shape == (1,)
        assert np.isfinite(result[prop][0])


@pytest.mark.parametrize('key', st2.MODELS)
@pytest.mark.parametrize('branch', ('stable', 'hdl', 'ldl'))
def test_gibbs_derivatives(key, branch):
    run = st2.MODELS[key]
    if branch == 'stable':
        T, P = np.array([242.0, 268.0, 298.0]), np.array([110.0, 170.0, 210.0])
    else:
        T = np.array([240.5, 241.5, 242.5])
        P = _coexistence_pressure(key, T) + (-0.03 if branch == 'hdl' else 0.03)
    dt, dp = 0.0005, 0.0005
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
        np.testing.assert_allclose(base[prop], derivative, rtol=4e-6, atol=0, err_msg=prop)
    if branch != 'stable':
        assert base['metastable'].all()
        assert (base['G'] > run(T, P)['G']).all()
