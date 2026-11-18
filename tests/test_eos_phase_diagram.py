"""EoS phase diagram of the two-state models (watereos.eos_phase_diagram)."""
import json

import numpy as np
import pytest

from watereos import eos_phase_diagram as epd

ATM = 0.101325


def temperatures_at(paths, pressure):
    """Temperatures where the paths cross an isobar (linear interpolation)."""
    out = []
    for path in paths:
        t, p = path.T
        side = p >= pressure
        for i in np.flatnonzero(side[:-1] != side[1:]):
            out.append(t[i] + (pressure - p[i]) * (t[i + 1] - t[i]) / (p[i + 1] - p[i]))
    return out


@pytest.fixture(scope='module')
def caupin():
    return epd.compute_eos_phase_diagram('caupin2019')


@pytest.mark.parametrize('model', sorted(epd.MODELS))
def test_every_model_has_all_curves(model):
    d = epd.compute_eos_phase_diagram(model)
    tc, pc = d['llcp']['T_K'], d['llcp']['p_MPa']
    assert np.allclose(d['binodal'][0], (tc, pc))
    assert len(d['binodal']) > 100 and d['binodal'][:, 0].min() >= epd.T_RANGE[0] - 1e-9
    for key in ('hdl_limit', 'ldl_limit'):
        assert d[key] and any(np.allclose(path[0], (tc, pc)) for path in d[key])
    assert d['widom']
    for ice in epd.ICES:
        assert len(d['melting'][ice]) == 1
    for key, _, _ in epd.EXTREMA:
        assert d['extrema'][key]['max'] or d['extrema'][key]['min']
    assert (d['lv_spinodal'] is None) == (model == 'holten2014')
    # Everything stays inside the computed window.
    paths = [d['binodal'], *d['hdl_limit'], *d['ldl_limit'], *d['widom'],
             *(p for v in d['melting'].values() for p in v),
             *(p for x in d['extrema'].values() for v in x.values() for p in v)]
    allpoints = np.vstack(paths)
    assert allpoints[:, 0].min() >= epd.T_RANGE[0] - 1e-9 and allpoints[:, 0].max() <= epd.T_RANGE[1] + 1e-9
    assert allpoints[:, 1].max() <= epd.P_RANGE[1] + 1e-9


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        epd.compute_eos_phase_diagram('water1')


@pytest.mark.parametrize('model', sorted(epd.MODELS))
def test_liquid_liquid_lines_match_the_per_pressure_solution(model):
    """Where the model's per-pressure solver has a single valid root, the traced lines agree with it."""
    d = epd.compute_eos_phase_diagram(model)
    m = epd._model(model)
    tc, pc = d['llcp']['T_K'], d['llcp']['p_MPa']
    p = np.linspace(pc + 3.0, 400.0, 300)
    spinodal = m['spinodal'](p)
    binodal = m['binodal'](p, spinodal=spinodal)
    P, upper = np.asarray(spinodal['p_array']), np.asarray(spinodal['T_upper'])
    for path in d['ldl_limit']:
        sel = (path[:, 1] > pc + 3.0) & (path[:, 1] < 399.0)
        assert np.abs(path[sel, 0] - np.interp(path[sel, 1], P, upper)).max() < 0.01
    bt, bp = np.asarray(binodal['T_K']), np.asarray(binodal['p_MPa'])
    ok = np.isfinite(bt) & (bp > pc + 3.0) & (bt > epd.T_RANGE[0] + 0.5)   # inside the plotted window
    traced = d['binodal'][d['binodal'][:, 0] > bt[ok].min() - 1.0]   # the branch the solver follows
    order = np.argsort(traced[:, 0])
    assert np.abs(np.interp(bt[ok], traced[order, 0], traced[order, 1]) - bp[ok]).max() < 0.05


def test_caupin2019_reentrant_liquid_liquid_lines(caupin):
    """Below about 150 K the extrapolated model's coexistence line turns back to lower pressure."""
    tc, pc = caupin['llcp']['T_K'], caupin['llcp']['p_MPa']
    assert tc == pytest.approx(218.13, abs=0.05) and pc == pytest.approx(71.95, abs=0.05)
    b = caupin['binodal']
    top = b[np.argmax(b[:, 1])]
    assert top[1] == pytest.approx(330.4, abs=0.5) and top[0] == pytest.approx(122.0, abs=2.0)
    assert b[-1] == pytest.approx([100.0, 315.6], abs=0.5)
    # The high-density liquid's stability limit has a pressure maximum and reaches 100 K.
    hdl = max(caupin['hdl_limit'], key=len)
    assert hdl[:, 1].max() == pytest.approx(125.3, abs=1.0)
    assert hdl[:, 0].min() == pytest.approx(100.0, abs=1e-6)


def test_caupin2019_melting_lines(caupin):
    (ih,) = caupin['melting']['Ih']
    (t_atm,) = temperatures_at([ih], ATM)
    assert t_atm == pytest.approx(273.152, abs=0.05)
    # The stretched ice Ih line ends on the liquid-vapor stability limit.
    low = ih[np.argmin(ih[:, 1])]
    assert low == pytest.approx([278.07, -270.9], abs=0.5)
    assert ih[:, 1].max() == pytest.approx(400.0, abs=1.0)
    # Ices III and V against the IAPWS R14-08 melting curves (254.96 K at 300 MPa, 259.82 K at 400 MPa).
    assert np.allclose(temperatures_at(caupin['melting']['III'], 300.0), 254.96, atol=0.5)
    assert np.allclose(temperatures_at(caupin['melting']['V'], 399.5), 259.8, atol=0.5)


def test_caupin2019_response_extrema_at_one_atmosphere(caupin):
    (tmd,) = temperatures_at(caupin['extrema']['density']['max'], ATM)
    assert tmd == pytest.approx(277.13, abs=0.5)   # measured 277.13 K
    warm = max(temperatures_at(caupin['extrema']['compressibility']['min'], ATM))
    assert warm == pytest.approx(319.67, abs=2.0)  # Kell (1975): 319.67 K


def test_vapor_liquid_and_nucleation(caupin):
    t, p = caupin['vapor_liquid'].T
    assert t[0] == epd.T_RANGE[0] and np.all(p >= 0) and np.all(np.diff(p) > 0)
    assert np.interp(273.16, t, p) == pytest.approx(611.657e-6, rel=1e-3)
    assert epd.nucleation_temperature([0.1])[0] == pytest.approx(235.15, abs=1e-6)
    assert epd.nucleation_temperature([200.0, 300.0]) == pytest.approx([180.70, 186.61], abs=0.01)


def test_ice_ih_gibbs_matches_iapws():
    _Ice = pytest.importorskip('iapws._iapws')._Ice
    for T, p in ((250.0, 0.1), (260.0, 50.0), (270.0, 100.0)):
        assert epd.ice_ih_gibbs(T, p) / 1e3 == pytest.approx(_Ice(T, p)['g'], abs=1e-9)


def test_reference_points_have_sources():
    ref = epd.reference_points()
    assert len(ref['melting']['Ih']['T_K']) > 200
    assert len(ref['melting']['III']['T_K']) > 30 and len(ref['melting']['V']['T_K']) > 30
    assert len(ref['density_maxima']['T_K']) > 40
    used = set()
    for entry in (*ref['melting'].values(), ref['density_maxima']):
        assert len(entry['T_K']) == len(entry['p_MPa']) == len(entry['source_key'])
        used.update(entry['source_key'])
    for entry in (*ref['warm_extrema'].values(), *ref['ambient_extrema'].values(), ref['heat_capacity_minima']):
        used.update(entry['sources'])
    assert used <= set(ref['sources'])
    assert all(ref['sources'][key]['citation'] for key in used)


def test_json_copy_is_serializable(caupin):
    data = epd.eos_phase_diagram_to_json(caupin)
    text = json.dumps(data)
    assert data['binodal']['T_K'][0] == pytest.approx(caupin['llcp']['T_K'], abs=1e-3)
    assert len(text) < 400_000


def test_plotly_figure_layers_and_themes():
    pytest.importorskip('plotly')
    from watereos.plot_theme import THEMES
    fig = epd.plot_eos_phase_diagram('caupin2019')
    legend = [t.name for t in fig.data if t.showlegend is not False]
    for key in epd.DEFAULT_LAYERS:
        assert epd.LAYERS[key] in legend
    assert epd.LAYERS['kauzmann'] not in legend
    assert list(fig.layout.xaxis.range) == list(epd.T_RANGE)
    assert list(fig.layout.yaxis.range) == list(epd.PLOT_P_RANGE)
    assert fig.layout.paper_bgcolor == THEMES['light']['bg_elev']
    assert any('critical point' in a.text for a in fig.layout.annotations)

    dark = epd.plot_eos_phase_diagram('caupin2019', show=['binodal', 'tmd', 'nuc_ih', 'kauzmann'], theme='dark')
    names = {t.name for t in dark.data}
    assert names >= {epd.LAYERS['binodal'], epd.LAYERS['density'], epd.LAYERS['nucleation'],
                     epd.LAYERS['kauzmann']}
    assert dark.layout.paper_bgcolor == THEMES['dark']['bg_elev']
    with pytest.raises(ValueError):
        epd.plot_eos_phase_diagram('caupin2019', theme='neon')
