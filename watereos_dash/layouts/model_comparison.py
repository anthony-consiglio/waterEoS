"""Tab 3: Model Comparison — layout definition."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from watereos_gui.utils.model_registry import (
    MODEL_REGISTRY, MODEL_ORDER, get_display_label, get_common_properties,
)

_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in MODEL_ORDER
]

_default_models = ['duska2020', 'holten2014']
_common = get_common_properties(_default_models)
_prop_options = [{'label': get_display_label(p), 'value': p} for p in _common]


def layout():
    sidebar = html.Div(className='sidebar', children=[
        html.Div(className='control-label', children='Models (select 2+)'),
        dcc.Dropdown(
            id='mc-models',
            options=_model_options,
            value=_default_models,
            multi=True,
            clearable=False,
            className='dash-dropdown',
        ),

        html.Div(className='control-label', children='Property'),
        dcc.Dropdown(
            id='mc-property',
            options=_prop_options,
            value='rho' if 'rho' in _common else (_common[0] if _common else None),
            clearable=False,
            className='dash-dropdown',
        ),

        html.Div(className='control-label', children='Temperature Range (K)'),
        dbc.Row([
            dbc.Col(dcc.Input(id='mc-tmin', type='number', value=200,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='mc-tmax', type='number', value=300,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        html.Div(className='control-label', children='Pressure Range (MPa)'),
        dbc.Row([
            dbc.Col(dcc.Input(id='mc-pmin', type='number', value=0.1,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='mc-pmax', type='number', value=200,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        html.Div(className='control-label', children='Number of Curves'),
        dcc.Input(id='mc-ncurves', type='number', value=5, min=1, max=50,
                  debounce=True, style={'width': '100%'}),

        html.Div(className='control-label', children='Points per Curve'),
        dcc.Input(id='mc-npoints', type='number', value=200, min=10, max=1000,
                  debounce=True, style={'width': '100%'}),

        html.Div(className='control-label', children='Curve Type'),
        dcc.RadioItems(
            id='mc-curve-type',
            options=[
                {'label': ' Isobars', 'value': 'isobar'},
                {'label': ' Isotherms', 'value': 'isotherm'},
            ],
            value='isobar',
            inline=True,
            style={'marginBottom': '8px'},
        ),

        html.Div(className='control-label', children='Layout'),
        dcc.RadioItems(
            id='mc-layout',
            options=[
                {'label': ' Overlay', 'value': 'overlay'},
                {'label': ' Side by Side', 'value': 'sidebyside'},
            ],
            value='overlay',
            style={'marginBottom': '12px'},
        ),

        html.Button('Update Plot', id='mc-update', className='btn-primary',
                     n_clicks=0),
    ], style={'width': '280px', 'flexShrink': 0})

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#3b82f6',
            children=dcc.Graph(
                id='mc-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
