"""Liquid-liquid lines of watereos.computation.compute_phase_diagram_data.

These lines feed the phase overlays of the Property Explorer and surface
figures. They are traced as contours (watereos.eos_phase_diagram), so they
must not carry the spurious low-temperature roots of the per-pressure
solvers (caupin2019's HDL limit used to jump from 190.8 K to 16.8 K at
123.8 MPa, duska2020's from 131.5 K to 19.4 K, holten2014's fell to the
solver's 5 K floor).
"""
import numpy as np
import pytest

from watereos import eos_phase_diagram as epd
from watereos.computation import (
    _PHASE_DIAGRAM_DEFAULTS,
    _load_precomputed,
    compute_phase_diagram_data,
    compute_property_at_forced_x,
)

MODELS = sorted(epd.MODELS)
#: line -> its fraction keys
LINES = {'binodal': ('x_hdl', 'x_ldl'), 'hdl_spinodal': ('x',), 'ldl_spinodal': ('x',)}
#: models with a precomputed JSON in watereos/data
PRECOMPUTED = ('caupin2019', 'duska2020', 'holten2014')
P_MAX = _PHASE_DIAGRAM_DEFAULTS['P_max']


def pieces(curve, *keys):
    """Pieces of a line as (n, 2 + len(keys)) arrays, split at its NaN rows."""
    table = np.column_stack([np.asarray(curve[k], float) for k in ('T_K', 'p_MPa', *keys)])
    gap = ~np.isfinite(table[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, (~gap).astype(int), 0]))
    return [table[a:b] for a, b in zip(edges[::2], edges[1::2])]


def from_critical_point(curve, llcp, *keys):
    """The piece that starts at the critical point, up to where its pressure starts to fall."""
    (piece,) = [p for p in pieces(curve, *keys)
                if np.allclose(p[0, :2], (llcp['T_K'], llcp['p_MPa']), atol=1e-9)]
    rising = np.r_[True, np.diff(piece[:, 1]) >= 0]
    return piece[:np.argmin(rising) if not rising.all() else len(piece)]


@pytest.fixture(scope='module', params=MODELS)
def model_data(request):
    return request.param, compute_phase_diagram_data(request.param)


def test_lines_are_continuous_and_stay_in_range(model_data):
    model, d = model_data
    llcp = d['LLCP']
    for key, fractions in LINES.items():
        parts = pieces(d[key], *fractions)
        assert parts, f'{model} {key} is empty'
        assert any(np.allclose(p[0, :2], (llcp['T_K'], llcp['p_MPa']), atol=1e-9) for p in parts)
        for part in parts:
            T, p = part[:, 0], part[:, 1]
            assert T.min() >= 100.0 - 1e-6, f'{model} {key} goes below 100 K'
            assert p.max() <= P_MAX + 1e-9
            # No jump between neighboring points (the traced grid is 0.5 K by 1 MPa).
            assert np.abs(np.diff(T)).max() < 20.0, f'{model} {key} jumps in temperature'
            assert np.abs(np.diff(p)).max() < 20.0, f'{model} {key} jumps in pressure'
            assert np.isfinite(part[:, 2:]).all(), f'{model} {key} has missing fractions'


def test_old_output_keys_are_gone(model_data):
    _, d = model_data
    assert 'spinodal' not in d
    assert set(d['binodal']) == {'T_K', 'p_MPa', 'x_hdl', 'x_ldl'}
    assert set(d['hdl_spinodal']) == set(d['ldl_spinodal']) == {'T_K', 'p_MPa', 'x'}


def test_stability_limits_are_named_for_the_liquid_that_ceases(model_data):
    """Below hdl_spinodal the high-density liquid no longer exists; above ldl_spinodal the low-density one."""
    model, d = model_data
    llcp = d['LLCP']
    p = llcp['p_MPa'] + 30.0
    T = {}
    for key in LINES:
        piece = from_critical_point(d[key], llcp)
        T[key] = float(np.interp(p, piece[:, 1], piece[:, 0]))
    assert T['hdl_spinodal'] < T['binodal'] < T['ldl_spinodal']

    m = epd._model(model)
    branch = lambda t, name: float(epd._evaluate(m['batch'], [t], [p], name)['x'][0])  # noqa: E731
    assert np.isnan(branch(T['hdl_spinodal'] - 0.05, 'hdl'))
    assert np.isfinite(branch(T['hdl_spinodal'] + 0.05, 'hdl'))
    assert np.isnan(branch(T['ldl_spinodal'] + 0.05, 'ldl'))
    assert np.isfinite(branch(T['ldl_spinodal'] - 0.05, 'ldl'))

    # The per-pressure solver calls these T_lower and T_upper.
    spinodal = m['spinodal'](np.array([p]))
    assert T['hdl_spinodal'] == pytest.approx(float(spinodal['T_lower'][0]), abs=0.01)
    assert T['ldl_spinodal'] == pytest.approx(float(spinodal['T_upper'][0]), abs=0.01)


def test_fractions_belong_to_the_right_liquid(model_data):
    model, d = model_data
    llcp = d['LLCP']
    b = from_critical_point(d['binodal'], llcp, 'x_hdl', 'x_ldl')
    assert b[0, 2] == b[0, 3] == pytest.approx(llcp['x'])
    assert (b[1:, 2] < b[1:, 3]).all()
    # The coexisting liquids' fractions are those of the model's 'hdl' and 'ldl' branches.
    m = epd._model(model)
    mid = b[len(b) // 2]
    assert epd._evaluate(m['batch'], [mid[0]], [mid[1]], 'hdl')['x'][0] == pytest.approx(mid[2], abs=1e-9)
    assert epd._evaluate(m['batch'], [mid[0]], [mid[1]], 'ldl')['x'][0] == pytest.approx(mid[3], abs=1e-9)


def test_spinodal_fraction_is_taken_inside_the_limit(model_data):
    """A liquid's heat capacity diverges on its stability limit; x is taken 1e-3 inside it, so it stays finite."""
    model, d = model_data
    llcp = d['LLCP']
    for key in ('hdl_spinodal', 'ldl_spinodal'):
        part = from_critical_point(d[key], llcp, 'x')
        part = part[part[:, 1] > llcp['p_MPa'] + 5.0]
        cp = compute_property_at_forced_x(model, 'Cp', part[:, 0], part[:, 1], part[:, 2])
        assert np.isfinite(cp).all()
        # A few times 1e5 J/(kg K) 1e-3 K inside the limit; at the limit itself it exceeds 1e8.
        assert np.median(np.abs(cp)) < 3e6, f'{model} {key}: median Cp {np.median(np.abs(cp)):.3g}'


@pytest.mark.parametrize('model', PRECOMPUTED)
def test_precomputed_json_matches_the_live_computation(model):
    """watereos/data/<model>_phase_diagram.json is current (regenerate with scripts/precompute_phase_diagrams.py)."""
    stored = _load_precomputed(model)
    assert stored is not None
    live = compute_phase_diagram_data(model, use_precomputed=False)
    assert stored['LLCP']['T_K'] == pytest.approx(live['LLCP']['T_K'], abs=1e-9)
    assert stored['LLCP']['p_MPa'] == pytest.approx(live['LLCP']['p_MPa'], abs=1e-9)
    for key, fractions in LINES.items():
        for name in ('T_K', 'p_MPa', *fractions):
            a, b = np.asarray(stored[key][name], float), np.asarray(live[key][name], float)
            assert a.shape == b.shape, f'{model} {key}.{name}'
            assert np.allclose(a, b, atol=1e-3, equal_nan=True), f'{model} {key}.{name}'


def test_liquid_fraction_inset():
    """liquid_fraction steps off a point just outside the limit, and with an inset moves inside it."""
    lines = epd.liquid_liquid_lines('caupin2019')
    hdl = max(lines['hdl_limit'], key=len)
    T, p = hdl[50:60, 0], hdl[50:60, 1]
    at_limit = epd.liquid_fraction('caupin2019', T, p, 'hdl')
    inside = epd.liquid_fraction('caupin2019', T, p, 'hdl', inset=1e-3)
    assert np.isfinite(at_limit).all() and np.isfinite(inside).all()
    # The HDL fraction falls on the way into the region where the HDL exists.
    assert (inside < at_limit).all() and (at_limit - inside).max() < 0.02
    with pytest.raises(ValueError):
        epd.liquid_fraction('caupin2019', T, p, 'vapor')
