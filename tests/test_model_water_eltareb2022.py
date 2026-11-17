"""Published q-TIP4P/F checks and consistency for all three fits."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from model_water_eos import eltareb2022 as eltareb
from watereos.two_state_general import M_D2O, M_H2O


@pytest.mark.parametrize('key, Tc, Pc, rho_c', [
    ('eltareb2022_qtip4pf_h2o_classical', 175.0, 203.0, 1030.0),
    ('eltareb2022_qtip4pf_h2o_quantum', 159.0, 167.0, 1020.0),
    ('eltareb2022_qtip4pf_d2o_quantum', 177.0, 176.0, 1130.0),
])
def test_critical_densities_tables_s1_to_s3(key, Tc, Pc, rho_c):
    out = eltareb.MODELS[key]([Tc], [Pc])
    assert out['x'][0] == pytest.approx(0.5, abs=5e-6)
    assert out['rho'][0] == pytest.approx(rho_c, abs=10.0)


@pytest.mark.parametrize('key, rho_fig', [
    (eltareb.H2O_CLASSICAL, 1002.0),  # Fig. 1C
    (eltareb.H2O_QUANTUM, 995.0),  # Fig. 1D
    (eltareb.D2O_QUANTUM, 1110.0),  # Fig. S4
])
def test_ambient_pressure_isobars(key, rho_fig):
    out = eltareb.MODELS[key]([300.0], [0.1])
    assert out['rho'][0] == pytest.approx(rho_fig, abs=3.0)


def _ambient_cp_peak(key):
    compute = eltareb.MODELS[key]
    peak = minimize_scalar(lambda T: -compute([T], [0.1])['Cp'][0],
                           method='bounded', bounds=(190.0, 245.0))
    assert peak.success
    return peak.x, -peak.fun


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'The Table S2 325 K fit yields 211.87 K; p. 8 reports 216 K'))
def test_quantum_h2o_cp_peak_temperature():
    T, _ = _ambient_cp_peak(eltareb.H2O_QUANTUM)
    assert T == pytest.approx(216.0, abs=1.0)


def test_quantum_d2o_cp_peak_on_fit_basis_fig_s7b():
    T, Cp = _ambient_cp_peak(eltareb.D2O_QUANTUM)
    assert T == pytest.approx(224.0, abs=2.0)
    assert Cp * M_H2O == pytest.approx(96.0, abs=2.0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'A physical D2O mole gives 106.9 J/(mol K); Fig. S7b gives about 96'))
def test_quantum_d2o_cp_peak_per_physical_mole_fig_s7b():
    _, Cp = _ambient_cp_peak(eltareb.D2O_QUANTUM)
    assert Cp * M_D2O == pytest.approx(96.0, abs=2.0)


def _check_derivatives(compute, T, P, branch):
    T, P = np.asarray(T), np.asarray(P)
    dt, dp = 0.001, 0.001  # K, MPa
    out = compute(T, P, branch=branch)
    hot = compute(T + dt, P, branch=branch)
    cold = compute(T - dt, P, branch=branch)
    high = compute(T, P + dp, branch=branch)
    low = compute(T, P - dp, branch=branch)
    expected = {
        'S': -(hot['G'] - cold['G']) / (2 * dt),
        'V': (high['G'] - low['G']) / (2 * dp * 1e6),
        'Cp': T * (hot['S'] - cold['S']) / (2 * dt),
        'Kt': -out['V'] * (2 * dp) / (high['V'] - low['V']),
        'alpha': (hot['V'] - cold['V']) / (2 * dt * out['V']),
    }
    for name, value in expected.items():
        assert np.isfinite(out[name]).all()
        np.testing.assert_allclose(out[name], value, rtol=2e-6, atol=1e-10,
                                   err_msg=name)


@pytest.mark.parametrize('key', eltareb.MODELS)
@pytest.mark.parametrize('branch, T, P', [
    ('stable', [200.0, 240.0, 295.0], [-30.0, 50.0, 250.0]),
    ('hdl', [230.0, 265.0, 300.0], [100.0, 160.0, 300.0]),
    ('ldl', [191.0, 200.0, 215.0], [-35.0, -20.0, -25.0]),
])
def test_thermodynamic_consistency(key, branch, T, P):
    _check_derivatives(eltareb.MODELS[key], T, P, branch)


@pytest.mark.parametrize('key, T, P', [
    (eltareb.H2O_CLASSICAL, 167.0, 224.55),
    (eltareb.H2O_QUANTUM, 151.0, 182.13),
    (eltareb.D2O_QUANTUM, 169.0, 193.98),
])
@pytest.mark.parametrize('branch, metastable', [
    ('hdl', [True, False]),
    ('ldl', [False, True]),
])
def test_metastable_branches(key, T, P, branch, metastable):
    compute = eltareb.MODELS[key]
    P = P + np.array([-0.5, 0.5])
    np.testing.assert_array_equal(compute(T, P, branch=branch)['metastable'],
                                  metastable)
    _check_derivatives(compute, T, P, branch)


@pytest.mark.parametrize('key', eltareb.MODELS)
def test_registry_probe(key):
    out = eltareb.MODELS[key]([273.15], [0.1])
    for name in ('rho', 'V', 'alpha', 'S', 'G', 'H', 'Cp'):
        assert np.isfinite(out[name][0])
