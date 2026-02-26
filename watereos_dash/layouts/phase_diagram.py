"""Tab 2: Phase Diagram — layout definition."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from watereos_gui.utils.model_registry import MODEL_REGISTRY, models_with_phase_diagram

_pd_models = models_with_phase_diagram()
_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in _pd_models
]
_default = _pd_models[0] if _pd_models else 'duska2020'
_info = MODEL_REGISTRY[_default]


def layout():
    sidebar = html.Div(className='sidebar', children=[
        html.Div(className='control-label', children='Model'),
        dcc.Dropdown(
            id='pd-model',
            options=_model_options,
            value=_default,
            clearable=False,
            className='dash-dropdown',
        ),

        # Display toggles
        html.Div(className='control-label', children='Show'),
        dcc.Checklist(
            id='pd-show',
            options=[
                {'label': ' Binodal', 'value': 'binodal'},
                {'label': ' Spinodal', 'value': 'spinodal'},
                {'label': ' LLCP', 'value': 'LLCP'},
            ],
            value=['binodal', 'spinodal', 'LLCP'],
            style={'marginBottom': '12px'},
        ),

        # Auto limits
        dcc.Checklist(
            id='pd-auto-limits',
            options=[{'label': ' Auto limits', 'value': 'auto'}],
            value=['auto'],
            style={'marginBottom': '8px'},
        ),

        # Manual T/P range (shown when auto unchecked)
        html.Div(id='pd-manual-limits', children=[
            html.Div(className='control-label', children='Temperature Range (K)'),
            dbc.Row([
                dbc.Col(dcc.Input(id='pd-tmin', type='number', value=_info.T_min,
                                  debounce=True, style={'width': '100%'}), width=6),
                dbc.Col(dcc.Input(id='pd-tmax', type='number', value=_info.T_max,
                                  debounce=True, style={'width': '100%'}), width=6),
            ], className='g-1'),
            html.Div(className='control-label', children='Pressure Range (MPa)'),
            dbc.Row([
                dbc.Col(dcc.Input(id='pd-pmin', type='number', value=_info.P_min,
                                  debounce=True, style={'width': '100%'}), width=6),
                dbc.Col(dcc.Input(id='pd-pmax', type='number', value=_info.P_max,
                                  debounce=True, style={'width': '100%'}), width=6),
            ], className='g-1'),
        ], style={'display': 'none'}),

        html.Button('Compute', id='pd-compute', className='btn-primary',
                     n_clicks=0, style={'marginTop': '12px'}),

        html.Div(id='pd-status', style={'marginTop': '8px', 'color': '#94a3b8',
                                         'fontSize': '12px'}),
    ], style={'width': '280px', 'flexShrink': 0})

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#3b82f6',
            children=dcc.Graph(
                id='pd-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
