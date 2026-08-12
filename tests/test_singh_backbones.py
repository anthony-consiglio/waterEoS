"""
Tests for the Singh (2017) transport model with alternative EoS backbones.

The published parameters (Holten 2014 backbone) must be untouched by the
backbone generalization; the refitted caupin2019/duska2020 parameter sets
must reproduce the experimental viscosity to a similar accuracy.
"""

import numpy as np
import pytest

BACKBONES = ['holten2014', 'caupin2019', 'duska2020']
MODEL_NAMES = {
    'holten2014': 'singh2017',
    'caupin2019': 'singh2017_caupin2019',
    'duska2020': 'singh2017_duska2020',
}


# ---------------------------------------------------------------------------
# Backward compatibility: holten2014 path identical to published model
# ---------------------------------------------------------------------------

def test_published_holten_params_unchanged():
    """Module-level constants must equal Table 1 of the PNAS paper."""
    from singh_viscosity import params as P
    assert P.T_0 == 147.75
    assert P.A0_eta == 38.75e-6
    assert P.E_LDS_k_eta == 2262.0
    assert P.E_HDS_k_eta == 421.9
    assert P.dv_HDS_eta == 2.44e-30
    assert P.A0_D == 40330e-12
    assert P.A0_tau == 86.2e-15


def test_default_backbone_is_holten():
    from singh_viscosity.core import compute_batch
    T = np.array([253.15, 273.15])
    p = np.array([0.1, 100.0])
    default = compute_batch(T, p)
    explicit = compute_batch(T, p, backbone='holten2014')
    for k in ('eta', 'D', 'tau_r'):
        np.testing.assert_array_equal(default[k], explicit[k])


# ---------------------------------------------------------------------------
# New backbones: registry + dispatch
# ---------------------------------------------------------------------------

def test_new_models_registered():
    from watereos import list_models
    models = list_models()
    assert 'singh2017_caupin2019' in models
    assert 'singh2017_duska2020' in models


def test_registry_metadata():
    from watereos.model_registry import MODEL_REGISTRY, MODEL_ORDER
    for key in ('singh2017_caupin2019', 'singh2017_duska2020'):
        info = MODEL_REGISTRY[key]
        assert info.has_transport
        assert not info.has_phase_diagram
        assert key in MODEL_ORDER


def test_unknown_backbone_raises():
    from singh_viscosity.core import compute_batch
    with pytest.raises(ValueError, match='unknown Singh-2017 backbone'):
        compute_batch(np.array([260.0]), np.array([0.1]), backbone='nope')


# ---------------------------------------------------------------------------
# Physical sanity of every backbone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backbone', BACKBONES)
def test_ambient_viscosity(backbone):
    """eta at 298.15 K, 0.1 MPa must be ~0.89 mPa*s for every backbone."""
    from singh_viscosity import getProp
    PT = np.array([[0.1], [298.15]], dtype=object)
    out = getProp(PT, backbone=backbone)
    eta_mPas = out.eta.flat[0] * 1e3
    assert abs(eta_mPas - 0.890) < 0.03, f"{backbone}: eta={eta_mPas:.4f}"


@pytest.mark.parametrize('backbone', BACKBONES)
def test_viscosity_at_0C(backbone):
    """eta at 273.15 K, 0.1 MPa must be ~1.79 mPa*s for every backbone."""
    from singh_viscosity import getProp
    PT = np.array([[0.1], [273.15]], dtype=object)
    out = getProp(PT, backbone=backbone)
    eta_mPas = out.eta.flat[0] * 1e3
    assert abs(eta_mPas - 1.793) < 0.06, f"{backbone}: eta={eta_mPas:.4f}"


@pytest.mark.parametrize('backbone', BACKBONES)
def test_supercooled_viscosity_vs_experiment(backbone):
    """Reproduce two Singh 2017 Table S1 measurements within ~3 sigma."""
    from singh_viscosity import getProp
    # (T_K, P_MPa, eta_mPas, sigma_mPas) from Table S1
    checks = [(252.8, 20.5, 3.854, 0.053),
              (244.3, 200.0, 4.697, 0.106)]
    for T, p, eta_exp, sig in checks:
        PT = np.array([[p], [T]], dtype=object)
        eta_mod = getProp(PT, backbone=backbone).eta.flat[0] * 1e3
        assert abs(eta_mod - eta_exp) < 4 * sig + 0.10, (
            f"{backbone}: eta({T} K, {p} MPa) = {eta_mod:.3f}, "
            f"exp {eta_exp:.3f} +/- {sig:.3f}")


@pytest.mark.parametrize('backbone', BACKBONES)
def test_pressure_anomaly(backbone):
    """At 250 K viscosity must decrease from 0.1 to 100 MPa (anomaly)."""
    from singh_viscosity import getProp
    lo = np.array([[0.1], [250.0]], dtype=object)
    hi = np.array([[100.0], [250.0]], dtype=object)
    eta_lo = getProp(lo, backbone=backbone).eta.flat[0]
    eta_hi = getProp(hi, backbone=backbone).eta.flat[0]
    assert eta_hi < eta_lo, f"{backbone}: no pressure anomaly at 250 K"


@pytest.mark.parametrize('backbone', BACKBONES)
def test_transport_finite_and_positive(backbone):
    from singh_viscosity.core import compute_batch
    T = np.linspace(240.0, 300.0, 13)
    p = np.full_like(T, 50.0)
    out = compute_batch(T, p, backbone=backbone)
    for k in ('eta', 'D', 'tau_r'):
        assert np.all(np.isfinite(out[k])), f"{backbone}: {k} not finite"
        assert np.all(out[k] > 0), f"{backbone}: {k} not positive"
    # eta decreases with T along an isobar; D increases
    assert np.all(np.diff(out['eta']) < 0)
    assert np.all(np.diff(out['D']) > 0)


@pytest.mark.parametrize('backbone', BACKBONES)
def test_dispatcher_variants(backbone):
    """watereos.getProp model names dispatch to the right backbone."""
    from watereos import getProp
    from singh_viscosity import getProp as sgp
    PT = np.array([[50.0], [260.0]], dtype=object)
    via_registry = getProp(PT, MODEL_NAMES[backbone])
    direct = sgp(PT, backbone=backbone)
    assert via_registry.eta.flat[0] == direct.eta.flat[0]
    assert via_registry.f.flat[0] == direct.f.flat[0]


def test_backbone_fraction_passthrough():
    """f must equal the backbone's own x, not Holten's, for each variant."""
    from singh_viscosity import getProp
    from caupin_eos.core import compute_batch as caupin_batch
    PT = np.array([[0.1], [273.15]], dtype=object)
    out = getProp(PT, backbone='caupin2019')
    x_direct = caupin_batch(np.array([273.15]), np.array([0.1]))['x'][0]
    assert abs(out.f.flat[0] - x_direct) < 1e-12
    # And it must differ substantially from the Holten fraction (~0.097)
    assert out.f.flat[0] > 0.2
