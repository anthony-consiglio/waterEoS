"""Phase overlays of the Property Explorer and surface figures (watereos_api.figures).

The liquid-liquid lines are traced as contours and can be reentrant, so the
overlays split them where they leave the plotted window instead of drawing a
segment across the gap. Their names are those of the EoS Phase Diagram tab.
"""
import numpy as np
import pytest

from watereos.eos_phase_diagram import LAYERS, MODELS
from watereos_api import figures

#: A window reaching below 20 K. The old per-pressure lines jumped there:
#: caupin2019's HDL stability limit from 190.8 K to 16.8 K, duska2020's from
#: 131.5 K to 19.4 K. The traced lines end at 100 K.
WINDOW = dict(T_range=[10.0, 300.0], P_range=[0.0, 300.0])
T_FLOOR = 100.0
LINE_KEYS = ('binodal', 'hdl_spinodal', 'ldl_spinodal')
MAX_STEP = 20.0   # K along T, MPa along p


def line_traces(fig):
    return [t for t in fig.data if t.legendgroup in LINE_KEYS]


def max_step(values):
    values = np.asarray(values, dtype=float)
    return float(np.abs(np.diff(values)).max()) if values.size > 1 else 0.0


@pytest.mark.parametrize('model', sorted(MODELS))
def test_surface2d_overlay(model):
    fig = figures.build_surface2d_figure(
        model=model, prop='rho', n_points=20, colormap='rdbu', show_phase=True,
        theme='light', units=None, **WINDOW)
    lines = line_traces(fig)
    # One legend entry per line, named as on the EoS Phase Diagram tab.
    assert sorted(t.name for t in lines if t.showlegend) == sorted(LAYERS[k] for k in LINE_KEYS)
    for t in lines:
        assert t.name == LAYERS[t.legendgroup]
        assert max_step(t.x) < MAX_STEP and max_step(t.y) < MAX_STEP
        assert min(t.x) >= WINDOW['T_range'][0] and max(t.y) <= WINDOW['P_range'][1]
    # Every model's HDL stability limit is traced down to 100 K, and nothing
    # lower. Local26's ends at 133.0 K, where it meets the fold of the
    # high-density state's expansion and that state stops existing.
    hdl = [t for t in lines if t.legendgroup == 'hdl_spinodal']
    end = 133.00626 if model == 'local26' else T_FLOOR
    assert min(min(t.x) for t in hdl) == pytest.approx(end, abs=1e-5)
    assert min(min(t.x) for t in lines) >= T_FLOOR - 1e-6
    assert any((t.name or '').startswith('LLCP') for t in fig.data)


@pytest.mark.parametrize('model', sorted(MODELS))
@pytest.mark.parametrize('isobar_mode', [True, False])
def test_curves_overlay(model, isobar_mode):
    fig = figures.build_curves_figure(
        model=model, prop='rho', n_curves=3, n_points=40, isobar_mode=isobar_mode,
        show_phase=True, theme='dark', units=None, **WINDOW)
    lines = line_traces(fig)
    assert {t.legendgroup for t in lines} == set(LINE_KEYS)
    for t in lines:
        assert max_step(t.x) < MAX_STEP   # temperature on isobars, pressure on isotherms
        assert np.isfinite(np.asarray(t.y, dtype=float)).all()


@pytest.mark.parametrize('model', sorted(MODELS))
def test_surface3d_segments(model):
    segments = figures._phase_segments_3d(model, 'rho', WINDOW['T_range'], WINDOW['P_range'])
    lines = [s for s in segments if s['type'] in ('spinodal', 'binodal')]
    assert {s['group'] for s in lines} == set(LINE_KEYS)
    for s in lines:
        assert max_step(s['T']) < MAX_STEP and max_step(s['P']) < MAX_STEP
        assert min(s['T']) >= T_FLOOR - 1e-6
    assert any(s['type'] == 'llcp' for s in segments)


def test_line_leaving_the_window_is_split():
    """caupin2019's HDL stability limit rises to 125 MPa and turns back: a 100 MPa ceiling cuts it in two."""
    fig = figures.build_surface2d_figure(
        model='caupin2019', prop='rho', n_points=20, colormap='rdbu', show_phase=True,
        theme='light', units=None, T_range=[100.0, 300.0], P_range=[0.0, 100.0])
    hdl = [t for t in fig.data if t.legendgroup == 'hdl_spinodal']
    assert len(hdl) == 2
    assert sum(t.showlegend for t in hdl) == 1
    for t in hdl:
        assert max(t.y) <= 100.0 and max_step(t.x) < MAX_STEP
