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
                {'label': ' LL Binodal', 'value': 'binodal'},
                {'label': ' HDL Spinodal', 'value': 'hdl_spinodal'},
                {'label': ' LDL Spinodal', 'value': 'ldl_spinodal'},
                {'label': ' LLCP', 'value': 'LLCP'},
                {'label': ' TMD', 'value': 'tmd'},
                {'label': ' Widom line', 'value': 'widom'},
                {'label': ' Ice Ih liquidus', 'value': 'ice_ih'},
                {'label': ' Ice III liquidus', 'value': 'ice_iii'},
                {'label': ' Ih nucleation', 'value': 'nuc_ih'},
                {'label': ' III nucleation', 'value': 'nuc_iii'},
                {'label': ' HDL Kauzmann', 'value': 'kauzmann_hdl'},
                {'label': ' LDL Kauzmann', 'value': 'kauzmann_ldl'},
            ],
            value=['binodal', 'hdl_spinodal', 'ldl_spinodal', 'LLCP',
                   'tmd', 'widom', 'ice_ih', 'ice_iii', 'nuc_ih',
                   'nuc_iii', 'kauzmann_hdl', 'kauzmann_ldl'],
            style={'marginBottom': '12px'},
        ),

        # Auto limits
        dcc.Checklist(
            id='pd-auto-limits',
            options=[{'label': ' Auto limits', 'value': 'auto'}],
            value=[],
            style={'marginBottom': '8px'},
        ),

        # Manual T/P range (shown when auto unchecked)
        html.Div(id='pd-manual-limits', children=[
            html.Div(className='control-label', children='Temperature Range (K)'),
            dbc.Row([
                dbc.Col(dcc.Input(id='pd-tmin', type='number', value=150,
                                  debounce=True, style={'width': '100%'}), width=6),
                dbc.Col(dcc.Input(id='pd-tmax', type='number', value=300,
                                  debounce=True, style={'width': '100%'}), width=6),
            ], className='g-1'),
            html.Div(className='control-label', children='Pressure Range (MPa)'),
            dbc.Row([
                dbc.Col(dcc.Input(id='pd-pmin', type='number', value=0,
                                  debounce=True, style={'width': '100%'}), width=6),
                dbc.Col(dcc.Input(id='pd-pmax', type='number', value=300,
                                  debounce=True, style={'width': '100%'}), width=6),
            ], className='g-1'),
        ], style={'display': 'block'}),

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
