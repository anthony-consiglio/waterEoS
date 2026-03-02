"""Tab: H2O Phase Diagram — callbacks.

Pre-computes the multi-phase T-V diagram at high resolution on import
and caches all three Plotly figures (T-V, T-P, 3D P-T-V) for instant display.
"""

import copy
import plotly.graph_objects as go
from dash import Input, Output, State

from watereos.tv_phase_diagram import (
    compute_tv_phase_diagram,
    plot_tv_phase_diagram_plotly,
    plot_tp_phase_diagram_plotly,
    plot_ptv_phase_diagram_plotly,
)
from watereos_dash.style import make_layout, make_layout_3d, DEFAULTS, COLORS

# ---------------------------------------------------------------------------
# Pre-compute at import time (runs once on app startup)
# ---------------------------------------------------------------------------
print("H2O Phase Diagram: computing high-resolution diagram (dT=0.25)...")
_diagram = compute_tv_phase_diagram(
    T_min=190.0, T_max=300.0, dT=0.25, verbose=False,
)

print("H2O Phase Diagram: generating T-V figure...")
_FIG_TV = plot_tv_phase_diagram_plotly(
    _diagram, V_min=7e-4, V_max=1.1e-3, T_min=190, T_max=300,
)

print("H2O Phase Diagram: generating T-P figure...")
_FIG_TP = plot_tp_phase_diagram_plotly(
    _diagram, T_min=190, T_max=300, P_min=1e-4, P_max=1000,
)

print("H2O Phase Diagram: generating 3D P-T-V figure...")
_FIG_PTV = plot_ptv_phase_diagram_plotly(
    _diagram, T_stride=4, n_pts_per_phase=80,
    V_min=7e-4, V_max=1.1e-3, P_max=1000,
)

_FIGURES = {'tv': _FIG_TV, 'tp': _FIG_TP, 'ptv': _FIG_PTV}
print("H2O Phase Diagram: all figures cached and ready.")


def _apply_style_2d(fig, settings):
    """Apply dark-theme styling from app settings to a 2D figure."""
    s = settings or DEFAULTS
    bg = s.get('bg_color', DEFAULTS['bg_color'])
    fs = s.get('font_size', DEFAULTS['font_size'])
    grid_on = s.get('grid_enabled', DEFAULTS['grid_enabled'])

    dark = bg and _is_dark(bg)
    text_color = COLORS['text'] if dark else '#1e293b'
    grid_color = COLORS['grid'] if dark else '#e2e4e8'

    fig.update_layout(
        paper_bgcolor=COLORS['page_bg'] if dark else '#ffffff',
        plot_bgcolor=bg,
        font=dict(family='Helvetica, Arial, sans-serif', size=fs,
                  color=text_color),
        xaxis=dict(gridcolor=grid_color, showgrid=grid_on,
                   linecolor=COLORS['border'] if dark else '#cbd5e1'),
        yaxis=dict(gridcolor=grid_color, showgrid=grid_on,
                   linecolor=COLORS['border'] if dark else '#cbd5e1'),
        legend=dict(
            bgcolor='rgba(17,24,39,0.8)' if dark else 'rgba(255,255,255,0.9)',
            font=dict(size=fs - 1, color=text_color),
            bordercolor=COLORS['border'] if dark else '#cbd5e1',
            borderwidth=1,
        ),
    )


def _apply_style_3d(fig, settings):
    """Apply dark-theme styling from app settings to a 3D figure."""
    s = settings or DEFAULTS
    bg = s.get('bg_color', DEFAULTS['bg_color'])
    fs = s.get('font_size', DEFAULTS['font_size'])
    grid_on = s.get('grid_enabled', DEFAULTS['grid_enabled'])

    dark = bg and _is_dark(bg)
    text_color = COLORS['text'] if dark else '#1e293b'
    grid_color = COLORS['grid'] if dark else '#e2e4e8'

    axis_update = dict(
        gridcolor=grid_color,
        showgrid=grid_on,
        backgroundcolor=bg,
        color=text_color,
        zerolinecolor=grid_color,
    )
    fig.update_layout(
        paper_bgcolor=COLORS['page_bg'] if dark else '#ffffff',
        font=dict(family='Helvetica, Arial, sans-serif', size=fs,
                  color=text_color),
        scene=dict(
            xaxis=axis_update,
            yaxis=axis_update,
            zaxis=axis_update,
            bgcolor=bg,
        ),
    )


def _is_dark(bg_hex):
    bg = bg_hex.lstrip('#')
    if len(bg) == 3:
        bg = ''.join(c * 2 for c in bg)
    r, g, b = int(bg[:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.3


def register(app):
    @app.callback(
        Output('h2o-pd-graph', 'figure'),
        [Input('h2o-pd-projection', 'value'),
         Input('h2o-pd-vmin', 'value'),
         Input('h2o-pd-vmax', 'value'),
         Input('h2o-pd-tmin', 'value'),
         Input('h2o-pd-tmax', 'value'),
         Input('h2o-pd-pmin', 'value'),
         Input('h2o-pd-pmax', 'value')],
        State('settings-store', 'data'),
    )
    def display(projection, vmin, vmax, tmin, tmax, pmin, pmax, settings):
        cached = _FIGURES.get(projection)
        if cached is None:
            return go.Figure()

        fig = copy.deepcopy(cached)

        try:
            vmin = float(vmin) if vmin is not None else 7e-4
            vmax = float(vmax) if vmax is not None else 1.1e-3
            tmin = float(tmin) if tmin is not None else 190
            tmax = float(tmax) if tmax is not None else 300
            pmin = float(pmin) if pmin is not None else 0.001
            pmax = float(pmax) if pmax is not None else 1000
        except (TypeError, ValueError):
            return fig

        if projection == 'tv':
            _apply_style_2d(fig, settings)
            fig.update_layout(
                xaxis=dict(range=[vmin, vmax]),
                yaxis=dict(range=[tmin, tmax]),
            )
        elif projection == 'tp':
            _apply_style_2d(fig, settings)
            fig.update_layout(
                xaxis=dict(range=[tmin, tmax]),
                yaxis=dict(range=[pmin, pmax]),
            )
        elif projection == 'ptv':
            _apply_style_3d(fig, settings)
            # 3D axes: x=V, y=T, z=P
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[vmin, vmax]),
                    yaxis=dict(range=[tmin, tmax]),
                    zaxis=dict(range=[0, pmax]),
                ),
            )

        return fig
