"""Tab: H2O Phase Diagram — layout definition."""

from dash import dcc, html
import dash_bootstrap_components as dbc


def layout():
    sidebar = html.Div(className='sidebar', children=[
        html.Div(className='control-label', children='Projection'),
        dcc.RadioItems(
            id='h2o-pd-projection',
            options=[
                {'label': ' T–V', 'value': 'tv'},
                {'label': ' T–P', 'value': 'tp'},
                {'label': ' 3D P–T–V', 'value': 'ptv'},
            ],
            value='tv',
            style={'marginBottom': '16px'},
            labelStyle={'display': 'block', 'marginBottom': '4px'},
        ),

        html.Hr(style={'borderColor': '#1e3a5f', 'margin': '12px 0'}),

        html.Div(className='control-label', children='Display Limits'),

        html.Div(className='control-label',
                 children='Specific Volume (m³/kg)',
                 style={'fontSize': '11px', 'marginTop': '8px'}),
        dbc.Row([
            dbc.Col(dcc.Input(id='h2o-pd-vmin', type='number', value=7e-4,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='h2o-pd-vmax', type='number', value=1.1e-3,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        html.Div(className='control-label',
                 children='Temperature (K)',
                 style={'fontSize': '11px', 'marginTop': '8px'}),
        dbc.Row([
            dbc.Col(dcc.Input(id='h2o-pd-tmin', type='number', value=190,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='h2o-pd-tmax', type='number', value=300,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),

        html.Div(className='control-label',
                 children='Pressure (MPa)',
                 style={'fontSize': '11px', 'marginTop': '8px'}),
        dbc.Row([
            dbc.Col(dcc.Input(id='h2o-pd-pmin', type='number', value=0.001,
                              debounce=True, style={'width': '100%'}), width=6),
            dbc.Col(dcc.Input(id='h2o-pd-pmax', type='number', value=1000,
                              debounce=True, style={'width': '100%'}), width=6),
        ], className='g-1'),
    ], style={'width': '280px', 'flexShrink': 0})

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#3b82f6',
            children=dcc.Graph(
                id='h2o-pd-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
