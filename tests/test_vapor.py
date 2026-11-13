"""
Water vapor: formulation, thermodynamic consistency, backends, reference state.

The vapor is the IAPWS-95 ideal gas with the second and third virial terms
of the IAPWS-95 residual part, written as a Gibbs function of (T, p). The
tests check

  * the coefficients against published IAPWS-95 check values,
  * that every property is a consistent derivative of one Gibbs function,
  * that the Rust and NumPy backends agree,
  * that the vapor sits on the reference state of ice Ih and of the liquid
    models: equating Gibbs energies reproduces the IAPWS sublimation and
    saturation pressures,
  * the public entry points (``getProp``, ``compute``, the SeaFreeze-style
    calls with phase ``'vapor'``, the model registry),
  * agreement with the full IAPWS-95 formulation (``iapws`` package, if
    installed).
"""

import warnings

import numpy as np
import pytest

from vapor_eos import (
    compute_batch,
    compute_grid,
    getProp as vapor_getProp,
    saturation_pressure,
    sublimation_pressure,
    vapor_pressure,
)
from vapor_eos import core, params

R = params.R
KEYS = core.KEYS

# Saturation pressure of IAPWS-95 (Maxwell construction) in Pa, computed
# with the iapws package; the 300 K entry is the 3.53681e-3 MPa of the
# IAPWS-95 release.
IAPWS95_PSAT = {
    273.16: 611.655, 280.0: 991.820, 290.0: 1919.999, 300.0: 3536.807,
    310.0: 6231.118, 320.0: 10545.976, 330.0: 17213.152, 340.0: 27187.814,
    350.0: 41681.730, 360.0: 62193.565, 373.15: 101417.997,
}


def _cloud(n=4000, seed=0):
    """Scattered (T, p) from deep vacuum to somewhat beyond saturation."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(190.0, 400.0, n)
    p_coex = np.where(T < 273.16, sublimation_pressure(np.minimum(T, 273.16)),
                      saturation_pressure(np.maximum(T, 273.16)))
    p = p_coex * 10.0 ** rng.uniform(-4.0, 0.1, n)
    return T, p


# ───────────────────────────────────────────────────────────────────────
# 1. Coefficients against IAPWS check values
# ───────────────────────────────────────────────────────────────────────

class TestFormulation:
    def test_ideal_gas_part_matches_iapws95_table(self):
        """IAPWS-95 release, Table 6: phi0 and its tau derivatives at
        T = 500 K, rho = 838.025 kg/m^3."""
        T, rho = 500.0, 838.025
        tau, delta = params.T_C / T, rho / params.RHO_C
        _, F, tF, ttF, *_ = core._tau_terms(np.array([T]))
        assert np.log(delta) + F[0] == pytest.approx(0.204797733e1, rel=1e-8)
        assert tF[0] / tau == pytest.approx(0.904611106e1, rel=1e-8)
        assert ttF[0] / tau**2 == pytest.approx(-0.193249185e1, rel=1e-8)

    @pytest.mark.parametrize("T, B", [
        (300.0, -0.066682), (450.0, -0.013028), (500.0, -0.0094137),
    ])
    def test_second_virial_coefficient(self, T, B):
        """Second virial coefficient of IAPWS-95 in m^3/kg (five-digit
        values as quoted in the iapws package documentation)."""
        b1 = core._tau_terms(np.array([T]))[4][0]
        assert b1 / params.RHO_C == pytest.approx(B, rel=2e-5)

    def test_ideal_gas_limit(self):
        T = np.array([200.0, 273.16, 373.15])
        p = np.full(3, 1e-12)
        o = compute_batch(T, p)
        np.testing.assert_allclose(o['V'] * p * 1e6 / (R * T), 1.0, rtol=1e-9)
        np.testing.assert_allclose(o['Cp'] - o['Cv'], R, rtol=1e-9)
        np.testing.assert_allclose(o['Kt'], p, rtol=1e-9)
        np.testing.assert_allclose(o['alpha'] * T, 1.0, rtol=1e-9)
        np.testing.assert_allclose(o['Kp'], 1.0, rtol=1e-9)
        # cp0 / R = 1 + n3 + vibrational terms: just above 4 at these T
        assert np.all((o['Cp'] / R > 4.0) & (o['Cp'] / R < 4.2))

    def test_correlation_check_values(self):
        # IAPWS R14-08 computer-program check value
        assert sublimation_pressure(230.0) == pytest.approx(8.94735e-6, rel=1e-6)
        # IAPWS SR1-86 / Wagner & Pruss (1993): triple, normal boiling
        # and critical points
        assert saturation_pressure(273.16) == pytest.approx(611.657e-6, rel=1e-6)
        assert saturation_pressure(373.1243) == pytest.approx(0.101325, rel=1e-6)
        assert saturation_pressure(647.096) == pytest.approx(22.064, rel=1e-12)
        # the two curves meet at the triple point
        assert sublimation_pressure(273.16) == pytest.approx(611.657e-6, rel=1e-12)


# ───────────────────────────────────────────────────────────────────────
# 2. One Gibbs function: every property is a consistent derivative
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def state():
    T, p = _cloud()
    return T, p, core.compute_batch(T, p)


class TestConsistency:
    def test_all_finite_in_range(self, state):
        _, _, o = state
        for k in KEYS:
            assert np.all(np.isfinite(o[k])), k

    def test_first_derivatives(self, state):
        T, p, o = state
        dT, h = 1e-3, 1e-4
        g = lambda T_, p_: core.compute_batch(T_, p_)['G']
        S_fd = -(g(T + dT, p) - g(T - dT, p)) / (2 * dT)
        V_fd = (g(T, p * (1 + h)) - g(T, p * (1 - h))) / (2 * h * p * 1e6)
        np.testing.assert_allclose(S_fd, o['S'], rtol=1e-8)
        np.testing.assert_allclose(V_fd, o['V'], rtol=1e-7)

    def test_second_derivatives(self, state):
        T, p, o = state
        dT, h = 1e-3, 1e-4
        f = lambda T_, p_: core.compute_batch(T_, p_)
        Cp_fd = T * (f(T + dT, p)['S'] - f(T - dT, p)['S']) / (2 * dT)
        al_fd = (f(T + dT, p)['V'] - f(T - dT, p)['V']) / (2 * dT) / o['V']
        dVdp = (f(T, p * (1 + h))['V'] - f(T, p * (1 - h))['V']) / (2 * h * p)
        Kp_fd = (f(T, p * (1 + h))['Kt'] - f(T, p * (1 - h))['Kt']) / (2 * h * p)
        np.testing.assert_allclose(Cp_fd, o['Cp'], rtol=1e-7)
        np.testing.assert_allclose(al_fd, o['alpha'], rtol=1e-7)
        np.testing.assert_allclose(-o['V'] / dVdp, o['Kt'], rtol=1e-6)
        np.testing.assert_allclose(Kp_fd, o['Kp'], rtol=1e-6)

    def test_identities(self, state):
        T, p, o = state
        pV = p * 1e6 * o['V']
        np.testing.assert_allclose(o['G'] + T * o['S'], o['H'], rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(o['H'] - pV, o['U'], rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(o['G'] - pV, o['A'], rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(o['rho'] * o['V'], 1.0, rtol=1e-14)
        Cv = o['Cp'] - T * o['V'] * o['alpha']**2 * o['Kt'] * 1e6
        np.testing.assert_allclose(Cv, o['Cv'], rtol=1e-11)
        np.testing.assert_allclose(o['Kt'] * o['Cp'] / o['Cv'], o['Ks'], rtol=1e-13)
        np.testing.assert_allclose(np.sqrt(o['Ks'] * 1e6 * o['V']), o['vel'], rtol=1e-13)

    def test_invalid_states_are_nan(self):
        T = np.array([300.0, 300.0, -5.0, 0.0, np.nan, 300.0, 200.0])
        p = np.array([0.0, -1e-3, 1e-3, 1e-3, 1e-3, np.nan, 0.1])
        for o in (core.compute_batch(T, p), compute_batch(T, p)):
            for k in KEYS:
                assert np.all(np.isnan(o[k])), k
        # 200 K at 1 bar is 6e5 times the sublimation pressure: the series
        # has no meaning there. Moderately supersaturated vapor does.
        ok = compute_batch(np.array([273.15]), np.array([0.1]))
        assert all(np.isfinite(ok[k][0]) for k in KEYS)

    def test_grid_matches_batch(self):
        T = np.array([200.0, 250.0, 273.16, 300.0, 373.15])
        p = np.array([1e-7, 1e-5, 1e-4, 6e-4, 1e-2, 0.1])
        g = core.compute_grid(T, p)
        b = core.compute_batch(np.tile(T, p.size), np.repeat(p, T.size))
        for k in KEYS:
            assert g[k].shape == (p.size, T.size)
            np.testing.assert_array_equal(g[k].ravel(), b[k])


# ───────────────────────────────────────────────────────────────────────
# 3. Rust backend against the NumPy reference
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rs():
    rs = pytest.importorskip("watereos_rs", reason="Rust backend not built")
    if not hasattr(rs, 'compute_batch_vapor'):
        pytest.skip("installed watereos_rs predates the vapor")
    return rs


class TestBackends:
    @pytest.mark.parametrize("n", [1, 7, 500, 20000])
    def test_scatter(self, rs, n):
        # 20000 points exercise the multi-threaded path
        T, p = _cloud(n, seed=n)
        # include points outside the meaning of the series
        p[::5] *= 1e3
        a = rs.compute_batch_vapor(T, p)
        b = core.compute_batch(T, p)
        assert set(a) == set(KEYS)
        for k in KEYS:
            assert a[k].shape == (n,)
            np.testing.assert_array_equal(np.isnan(a[k]), np.isnan(b[k]), err_msg=k)
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, err_msg=k)

    @pytest.mark.parametrize("n_p, n_t", [(1, 1), (3, 5), (120, 110)])
    def test_grid(self, rs, n_p, n_t):
        T = np.linspace(190.0, 400.0, n_t)
        p = np.geomspace(1e-8, 0.2, n_p)
        a = rs.compute_grid_vapor(T, p)
        b = core.compute_grid(T, p)
        for k in KEYS:
            assert a[k].shape == (n_p, n_t)
            np.testing.assert_array_equal(np.isnan(a[k]), np.isnan(b[k]), err_msg=k)
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, err_msg=k)

    def test_non_contiguous_and_mismatched_input(self, rs):
        T = np.linspace(250.0, 300.0, 20)
        p = np.geomspace(1e-6, 1e-4, 20)
        a = rs.compute_batch_vapor(T[::2], p[::2])
        b = core.compute_batch(T[::2].copy(), p[::2].copy())
        np.testing.assert_allclose(a['G'], b['G'], rtol=1e-12)
        with pytest.raises(ValueError):
            rs.compute_batch_vapor(T, p[:5])
        with pytest.raises(ValueError):
            core.compute_batch(T, p[:5])

    def test_dispatch_reports_backend(self, rs):
        import watereos
        assert watereos.backend('vapor') == 'rust'


# ───────────────────────────────────────────────────────────────────────
# 4. Reference state: coexistence with ice Ih and with the liquids
# ───────────────────────────────────────────────────────────────────────

def _condensed_G(phase, T, p):
    from watereos._seafreeze_native import getProp_scatter
    from watereos import getProp
    if phase == 'Ih':
        return np.asarray(getProp_scatter('Ih', p, T).G)
    PT = np.empty(T.size, dtype=object)
    PT[:] = list(zip(p.tolist(), T.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(getProp(PT, phase).G)


class TestReferenceState:
    def test_gibbs_continuity_with_ice_Ih_on_sublimation_curve(self):
        """On the IAPWS R14-08 sublimation curve the vapor and SeaFreeze
        ice Ih have the same Gibbs energy to 2e-4 R T (about 25 J/kg)."""
        T = np.linspace(190.0, 273.16, 84)
        p = sublimation_pressure(T)
        dG = compute_batch(T, p)['G'] - _condensed_G('Ih', T, p)
        assert np.max(np.abs(dG) / (R * T)) < 2e-4

    def test_sublimation_pressure_from_gibbs_equality(self):
        T = np.linspace(190.0, 273.16, 84)
        rel = vapor_pressure(T, 'Ih') / sublimation_pressure(T) - 1.0
        assert np.max(np.abs(rel)) < 2e-4
        # down to the limit of the IAPWS-95 ideal-gas part
        T = np.linspace(130.0, 190.0, 13)
        rel = vapor_pressure(T, 'Ih') / sublimation_pressure(T) - 1.0
        assert np.max(np.abs(rel)) < 3e-4

    # (model, highest T tested [K], tolerance on the saturation pressure)
    LIQUIDS = [
        ('IAPWS95', 373.15, 5e-5),
        ('water1', 373.15, 6e-5),
        ('holten2014', 300.0, 5e-5),
        ('caupin2019', 300.0, 5e-5),
        ('caupin2019_kim', 300.0, 5e-5),
        ('grenke2025', 300.0, 1e-4),
        ('duska2020', 373.15, 5e-5),
    ]

    @pytest.mark.parametrize("model, T_hi, tol", LIQUIDS)
    def test_gibbs_continuity_with_liquid_on_saturation_curve(self, model, T_hi, tol):
        """On the IAPWS-95 saturation curve the vapor and the liquid have
        the same Gibbs energy, within the liquid model's own accuracy."""
        T = np.array([t for t in IAPWS95_PSAT if t <= T_hi])
        p = np.array([IAPWS95_PSAT[t] for t in T]) * 1e-6
        dG = compute_batch(T, p)['G'] - _condensed_G(model, T, p)
        assert np.max(np.abs(dG) / (R * T)) < tol

    @pytest.mark.parametrize("model, T_hi, tol", LIQUIDS)
    def test_saturation_pressure_from_gibbs_equality(self, model, T_hi, tol):
        T = np.array([t for t in IAPWS95_PSAT if t <= T_hi])
        p_ref = np.array([IAPWS95_PSAT[t] for t in T]) * 1e-6
        rel = vapor_pressure(T, model) / p_ref - 1.0
        assert np.max(np.abs(rel)) < tol

    def test_triple_point(self):
        p_ice = float(vapor_pressure(273.16, 'Ih'))
        p_liq = float(vapor_pressure(273.16, 'water1'))
        assert p_ice == pytest.approx(611.657e-6, rel=5e-5)
        assert p_liq == pytest.approx(611.657e-6, rel=5e-5)
        assert p_ice == pytest.approx(p_liq, rel=1e-5)

    def test_supercooled_water_vapor_pressure(self):
        """Murphy & Koop (2005) values for supercooled water, in Pa."""
        T = np.array([263.15, 253.15, 243.15, 233.15])
        ref = np.array([286.6, 125.6, 51.0, 18.9])
        p = vapor_pressure(T, 'holten2014') * 1e6
        np.testing.assert_allclose(p, ref, rtol=3e-3)
        # supercooled water has the higher vapor pressure
        assert np.all(p > sublimation_pressure(T) * 1e6)

    def test_vapor_pressure_shapes_and_failures(self):
        assert np.ndim(vapor_pressure(250.0, 'Ih')) == 0
        T = np.array([[230.0, 240.0], [250.0, 260.0]])
        assert vapor_pressure(T, 'Ih').shape == (2, 2)
        with pytest.raises(ValueError):
            vapor_pressure(250.0, 'no_such_phase')


# ───────────────────────────────────────────────────────────────────────
# 5. Public entry points
# ───────────────────────────────────────────────────────────────────────

class TestPublicAPI:
    P = np.array([1e-6, 1e-4, 6e-4])
    T = np.array([230.0, 250.0, 273.16, 300.0])

    def test_registered(self):
        from watereos import list_models, MODEL_REGISTRY, MODEL_ORDER
        from watereos.model_registry import EMPIRICAL_PROPS
        assert 'vapor' in list_models()
        assert 'vapor' in MODEL_ORDER
        info = MODEL_REGISTRY['vapor']
        assert info.model_key == 'vapor'
        assert not info.is_two_state and not info.has_transport
        assert set(info.properties) == set(EMPIRICAL_PROPS) == set(KEYS)
        assert (info.T_min, info.T_max) == (params.T_MIN, params.T_MAX)
        assert (info.P_min, info.P_max) == (params.P_MIN, params.P_MAX)

    def test_getProp_grid_and_compute(self):
        from watereos import getProp, compute
        PT = np.array([self.P, self.T], dtype=object)
        out = getProp(PT, 'vapor')
        ref = core.compute_grid(self.T, self.P)
        via_compute = compute(self.T, self.P, 'Vapor')
        for k in KEYS:
            assert getattr(out, k).shape == (3, 4)
            np.testing.assert_allclose(getattr(out, k), ref[k], rtol=1e-12)
            np.testing.assert_allclose(getattr(via_compute, k), ref[k], rtol=1e-12)

    def test_getProp_scatter(self):
        from watereos import getProp
        PT = np.empty(3, dtype=object)
        PT[0], PT[1], PT[2] = (1e-6, 230.0), (1e-4, 250.0), (6e-4, 273.16)
        out = getProp(PT, 'vapor')
        ref = core.compute_batch(self.T[:3], self.P)
        for k in KEYS:
            assert getattr(out, k).shape == (3,)
            np.testing.assert_allclose(getattr(out, k), ref[k], rtol=1e-12)
        np.testing.assert_allclose(vapor_getProp(PT).G, ref['G'], rtol=1e-12)

    def test_seafreeze_style_calls(self):
        """'vapor' is a phase name next to 'Ih', 'II', ... in the
        SeaFreeze-style calls, with the same argument order."""
        from watereos._seafreeze_native import getProp, getProp_grid, getProp_scatter
        ref_g = core.compute_grid(self.T, self.P)
        ref_s = core.compute_batch(self.T[:3], self.P)
        PT_grid = np.array([self.P, self.T], dtype=object)
        PT_scatter = np.empty(3, dtype=object)
        PT_scatter[:] = list(zip(self.P.tolist(), self.T[:3].tolist()))
        for out, ref in ((getProp(PT_grid, 'vapor'), ref_g),
                         (getProp_grid('vapor', self.P, self.T), ref_g),
                         (getProp(PT_scatter, 'vapor'), ref_s),
                         (getProp_scatter('vapor', self.P, self.T[:3]), ref_s)):
            for k in KEYS:
                np.testing.assert_allclose(getattr(out, k), ref[k], rtol=1e-12)
        # same call for an ice: Gibbs energies are directly comparable
        ice = getProp_scatter('Ih', self.P, self.T[:3])
        assert ice.G.shape == ref_s['G'].shape
        with pytest.raises(ValueError):
            getProp_scatter('vapor', self.P, self.T)

    def test_batch_accepts_lists_and_integers(self):
        a = compute_batch([250, 300], [1e-4, 1e-3])
        b = core.compute_batch(np.array([250.0, 300.0]), np.array([1e-4, 1e-3]))
        np.testing.assert_allclose(a['G'], b['G'], rtol=1e-12)
        g = compute_grid([250, 300], [1e-4])
        assert g['G'].shape == (1, 2)

    def test_no_liquid_branch(self):
        """branch= selects a liquid of a two-state model; the vapor has none."""
        from watereos import getProp, compute
        PT = np.array([self.P, self.T], dtype=object)
        np.testing.assert_array_equal(getProp(PT, 'vapor', branch='stable').G,
                                      getProp(PT, 'vapor').G)
        with pytest.raises(ValueError, match="not available for model 'vapor'"):
            getProp(PT, 'vapor', branch='hdl')
        with pytest.raises(ValueError, match="not available for model 'vapor'"):
            compute(self.T, self.P, 'vapor', branch='ldl')

    def test_bounds_warning(self):
        from watereos import getProp
        with pytest.warns(UserWarning, match="outside suggested range for vapor"):
            getProp(np.array([[1.0], [300.0]], dtype=object), 'vapor')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            getProp(np.array([[1e-3], [300.0]], dtype=object), 'vapor')


# ───────────────────────────────────────────────────────────────────────
# 6. Against the full IAPWS-95 formulation and IAPWS correlations (iapws)
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iapws():
    return pytest.importorskip("iapws", reason="iapws not installed")


class TestAgainstIapws:
    def test_sublimation_pressure(self, iapws):
        from iapws._iapws import _Sublimation_Pressure
        T = np.linspace(130.0, 273.16, 60)
        ref = np.array([_Sublimation_Pressure(float(t)) for t in T])
        np.testing.assert_allclose(sublimation_pressure(T), ref, rtol=1e-12)

    def test_saturation_pressure(self, iapws):
        """The auxiliary equation against the Maxwell construction of
        IAPWS-95, and the hard-coded table used above."""
        T = np.array(sorted(IAPWS95_PSAT))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = np.array([iapws.IAPWS95(T=float(t), x=0).P for t in T])
        np.testing.assert_allclose(saturation_pressure(T), ref, rtol=1e-4)
        table = np.array([IAPWS95_PSAT[t] for t in T]) * 1e-6
        np.testing.assert_allclose(table, ref, rtol=2e-6)

    @pytest.mark.parametrize("fraction, tol", [
        # (fraction of the saturation pressure, tolerances)
        (1.0, dict(G=3e-5, V=1e-4, S=1.2e-3, H=1.2e-3, Cp=1.5e-2, Cv=1.7e-2, vel=1.5e-3)),
        (0.5, dict(G=4e-6, V=1e-5, S=2e-4, H=2e-4, Cp=2.5e-3, Cv=3e-3, vel=3e-4)),
    ])
    def test_properties_up_to_saturation(self, iapws, fraction, tol):
        """Triple point to 100 C: the deviation from the full formulation
        is its terms of order rho^3 and higher, which the vapor drops."""
        for T in (273.16, 285.0, 300.0, 323.15, 348.15, 373.15):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sat = iapws.IAPWS95(T=T, x=1)
                st = sat if fraction == 1.0 else iapws.IAPWS95(T=T, P=sat.P * fraction)
            o = {k: v[0] for k, v in compute_batch([T], [st.P]).items()}
            RT = R * T
            assert abs(o['G'] - st.g * 1e3) / RT < tol['G'], T
            assert abs(o['V'] / st.v - 1) < tol['V'], T
            assert abs(o['S'] - st.s * 1e3) / R < tol['S'], T
            assert abs(o['H'] - st.h * 1e3) / RT < tol['H'], T
            assert abs(o['Cp'] / (st.cp * 1e3) - 1) < tol['Cp'], T
            assert abs(o['Cv'] / (st.cv * 1e3) - 1) < tol['Cv'], T
            assert abs(o['vel'] / st.w - 1) < tol['vel'], T

    def test_better_than_ideal_gas_at_100C(self, iapws):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            st = iapws.IAPWS95(T=373.15, x=1)
        V = compute_batch([373.15], [st.P])['V'][0]
        V_ideal = R * 373.15 / (st.P * 1e6)
        assert abs(V_ideal / st.v - 1) > 1.5e-2      # ideal gas: 1.6 % off
        assert abs(V / st.v - 1) < 1e-4


# ───────────────────────────────────────────────────────────────────────
# 7. The vapor branch of the T-V phase diagram
# ───────────────────────────────────────────────────────────────────────

class TestPhaseDiagramBranch:
    def test_branch_starts_on_the_common_tangent(self):
        from watereos.tv_phase_diagram import _evaluate_vapor
        T = np.array([200.0, 250.0, 273.0, 280.0, 350.0])
        d = _evaluate_vapor(T, n_V=12, V_max_factor=100.0)
        assert d['T_mask'].all()
        assert d['V'].shape == d['A'].shape == d['P'].shape == (12, 5)
        # first sample: coexistence with ice Ih (below the triple point)
        # or with the liquid; G = A + P V equals the condensed phase's G
        p0 = d['P'][0]
        np.testing.assert_allclose(p0[:3], vapor_pressure(T[:3], 'Ih'), rtol=1e-12)
        np.testing.assert_allclose(p0[3:], vapor_pressure(T[3:], 'water1'), rtol=1e-12)
        G0 = d['A'][0] + p0 * 1e6 * d['V'][0]
        G_cond = np.concatenate([_condensed_G('Ih', T[:3], p0[:3]),
                                 _condensed_G('water1', T[3:], p0[3:])])
        np.testing.assert_allclose(G0, G_cond, rtol=0, atol=1e-6)
        # the branch runs to 100 times lower pressure, volume increasing
        np.testing.assert_allclose(d['P'][0] / d['P'][-1], 100.0, rtol=1e-12)
        assert np.all(np.diff(d['V'], axis=0) > 0)

    def test_diagram_coexistence_pressures(self):
        pytest.importorskip("iapws", reason="the diagram's overlay curves need iapws")
        from watereos.tv_phase_diagram import compute_tv_phase_diagram
        d = compute_tv_phase_diagram(T_min=250.0, T_max=300.0, dT=5.0, verbose=False)
        bounds = {tuple(str(x) for x in k): v for k, v in d['two_phase_bounds'].items()}
        ice, liq = bounds[('Ih', 'vapor')], bounds[('water1', 'vapor')]
        on_grid = np.isclose(ice['T'] % 5.0, 0.0)
        np.testing.assert_allclose(ice['P_coex'][on_grid],
                                   sublimation_pressure(ice['T'][on_grid]), rtol=2e-4)
        on_grid = np.isclose(liq['T'] % 5.0, 0.0)
        np.testing.assert_allclose(liq['P_coex'][on_grid],
                                   saturation_pressure(liq['T'][on_grid]), rtol=2e-4)
