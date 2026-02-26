"""Tab 1: Property Explorer — layout definition."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from watereos_gui.utils.model_registry import (
    MODEL_REGISTRY, MODEL_ORDER, PROPERTY_LABELS, get_display_label,
)

# Build model options
_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in MODEL_ORDER
]

# Default model
_default_model = 'duska2020'
_default_info = MODEL_REGISTRY[_default_model]
_default_props = [
    {'label': get_display_label(p), 'value': p}
    for p in _default_info.properties
]


def layout():
    sidebar = html.Div(className='sidebar', children=[
        # Update button at top
        html.Button('Update Plot', id='pe-update', className='btn-primary',
                     n_clicks=0, style={'marginBottom': '8px'}),

        # Model
        html.Div(className='control-label', children='Model'),
        dcc.Dropdown(
            id='pe-model',
            options=_model_options,
            value=_default_model,
            clearable=False,
            className='dash-dropdown',
        ),

        # Property
        html.Div(className='control-label', children='Property'),
        dcc.Dropdown(
            id='pe-property',
            options=_default_props,
            value='rho',
            clearable=False,
            className='dash-dropdown',
        ),

        # T range
        html.Div(className='control-label', children='T Range (K)'),
        dbc.Row([
            dbc.Col(dcc.Input(id='pe-tmin', type='number', value=_default_info.T_min,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='pe-tmax', type='number', value=_default_info.T_max,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        # P range
        html.Div(className='control-label', children='P Range (MPa)'),
        dbc.Row([
            dbc.Col(dcc.Input(id='pe-pmin', type='number', value=_default_info.P_min,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='pe-pmax', type='number', value=_default_info.P_max,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        # Curves / points — side by side
        dbc.Row([
            dbc.Col([
                html.Div(className='control-label', children='Curves'),
                dcc.Input(id='pe-ncurves', type='number', value=5, min=1, max=50,
                          debounce=True, style={'width': '100%'}),
            ], width=6),
            dbc.Col([
                html.Div(className='control-label', children='Points'),
                dcc.Input(id='pe-npoints', type='number', value=200, min=10, max=1000,
                          debounce=True, style={'width': '100%'}),
            ], width=6),
        ], className='g-1'),

        # Isobar / Isotherm
        html.Div(className='control-label', children='Curve Type'),
        dcc.RadioItems(
            id='pe-curve-type',
            options=[
                {'label': ' Isobars', 'value': 'isobar'},
                {'label': ' Isotherms', 'value': 'isotherm'},
            ],
            value='isobar',
            inline=True,
            style={'marginBottom': '4px'},
        ),

        # Display mode
        html.Div(className='control-label', children='Display Mode'),
        dcc.RadioItems(
            id='pe-display-mode',
            options=[
                {'label': ' Curves', 'value': 'curves'},
                {'label': ' 2D Surface', 'value': 'surface2d'},
                {'label': ' 3D Surface', 'value': 'surface3d'},
            ],
            value='curves',
            style={'marginBottom': '4px'},
        ),

        # Z-axis choice (hidden in curves mode)
        html.Div(id='pe-zaxis-container', children=[
            html.Div(className='control-label', children='Z / Color Axis'),
            dcc.Dropdown(
                id='pe-zaxis',
                options=[
                    {'label': 'Property', 'value': 'property'},
                    {'label': 'Temperature', 'value': 'T'},
                    {'label': 'Pressure', 'value': 'P'},
                ],
                value='property',
                clearable=False,
                className='dash-dropdown',
            ),
        ], style={'display': 'none'}),

        # Phase boundaries
        dcc.Checklist(
            id='pe-phase-boundaries',
            options=[{'label': ' Show phase boundaries', 'value': 'show'}],
            value=[],
            style={'marginTop': '6px', 'marginBottom': '6px'},
        ),

        # CSV download
        html.Button('Download CSV', id='pe-download-btn', className='btn-secondary',
                     n_clicks=0, style={'marginTop': '4px', 'width': '100%'}),
        dcc.Download(id='pe-download'),
    ], style={'width': '260px', 'flexShrink': 0})

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#3b82f6',
            children=dcc.Graph(
                id='pe-graph',
                config={'displayModeBar': True, 'toImageButtonOptions': {
                    'format': 'png', 'width': 1200, 'height': 800, 'scale': 2,
                }},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
