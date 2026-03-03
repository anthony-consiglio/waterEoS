"""
waterEoS-Visualizer — Plotly Dash web application.

Run:  python watereos_visualizer/app.py
Open:  http://127.0.0.1:8050
"""

import sys
from pathlib import Path

# Ensure the parent directory (waterEoS repo root) is on sys.path so that
# watereos_visualizer and watereos are importable.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import dash
from dash import dcc, html
import dash_mantine_components as dmc

from watereos_visualizer.style import DEFAULTS
from watereos_visualizer.layouts import (
    info_layout,
    property_explorer_layout,
    phase_diagram_layout,
    model_comparison_layout,
    point_calculator_layout,
    settings_layout,
    h2o_phase_diagram_layout,
)
from watereos_visualizer.callbacks import register_all

# --- App init ---
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='waterEoS-Visualizer',
    update_title='waterEoS-Visualizer — computing...',
)
server = app.server  # for deployment (gunicorn / Render / etc.)

# --- Global stores ---
stores = html.Div([
    dcc.Store(id='settings-store', storage_type='local', data=dict(DEFAULTS)),
    dcc.Store(id='pd-store', storage_type='memory'),
    dcc.Store(id='clicked-point-store', storage_type='memory'),
    dcc.Store(id='pe-data-store', storage_type='memory'),
])

# --- Tab definitions ---
_TAB_DEFS = [
    ('tab-info', 'Info'),
    ('tab-pe', 'Property Explorer'),
    ('tab-h2o', 'H2O Phase Diagram'),
    ('tab-pd', 'EoS Phase Diagram'),
    ('tab-mc', 'Model Comparison'),
    ('tab-pc', 'Point Calculator'),
    ('tab-st', 'Settings'),
]

tabs = dmc.Tabs(
    id='main-tabs',
    value='tab-info',
    children=[
        dmc.TabsList([
            dmc.TabsTab(label, value=tid)
            for tid, label in _TAB_DEFS
        ]),
    ],
)

# --- Layout (all tabs rendered, show/hide for persistence) ---
_hidden = {'display': 'none'}

app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    children=[
        stores,
        # Header
        dmc.Group(
            gap="lg",
            style={
                'borderBottom': '1px solid var(--mantine-color-dark-4)',
                'padding': '0 16px',
                'backgroundColor': 'var(--mantine-color-dark-7)',
            },
            children=[
                dmc.Text(
                    'waterEoS-Visualizer',
                    fw=700,
                    size="lg",
                    c="blue",
                    style={'padding': '10px 0', 'letterSpacing': '0.5px'},
                ),
                html.Div(tabs, style={'flex': 1, 'minWidth': 0, 'overflow': 'visible'}),
            ],
        ),
        # Tab content — all rendered, toggled via CSS display
        html.Div(id='tab-info-content', children=info_layout()),
        html.Div(id='tab-pe-content', children=property_explorer_layout(), style=_hidden),
        html.Div(id='tab-pd-content', children=phase_diagram_layout(), style=_hidden),
        html.Div(id='tab-mc-content', children=model_comparison_layout(), style=_hidden),
        html.Div(id='tab-pc-content', children=point_calculator_layout(), style=_hidden),
        html.Div(id='tab-st-content', children=settings_layout(), style=_hidden),
        html.Div(id='tab-h2o-content', children=h2o_phase_diagram_layout(), style=_hidden),
    ],
)


# --- Tab visibility switching ---
@app.callback(
    [dash.Output('tab-info-content', 'style'),
     dash.Output('tab-pe-content', 'style'),
     dash.Output('tab-pd-content', 'style'),
     dash.Output('tab-mc-content', 'style'),
     dash.Output('tab-pc-content', 'style'),
     dash.Output('tab-st-content', 'style'),
     dash.Output('tab-h2o-content', 'style')],
    dash.Input('main-tabs', 'value'),
)
def switch_tab(tab):
    shown = {}
    hidden = {'display': 'none'}
    idx = {
        'tab-info': 0, 'tab-pe': 1, 'tab-pd': 2,
        'tab-mc': 3, 'tab-pc': 4, 'tab-st': 5,
        'tab-h2o': 6,
    }
    styles = [hidden] * 7
    styles[idx.get(tab, 0)] = shown
    return styles


# --- Register all callbacks ---
register_all(app)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050)
