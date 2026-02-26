"""Tab 4: Point Calculator — layout definition."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from watereos_gui.utils.model_registry import MODEL_REGISTRY, MODEL_ORDER

_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in MODEL_ORDER
]


def layout():
    sidebar = html.Div(className='sidebar', children=[
        html.Div(className='control-label', children='Temperature (K)'),
        dcc.Input(id='pc-temperature', type='number', value=273.15,
                  debounce=True, style={'width': '100%'}),

        html.Div(className='control-label', children='Pressure (MPa)'),
        dcc.Input(id='pc-pressure', type='number', value=0.1,
                  debounce=True, style={'width': '100%'}),

        html.Div(className='control-label', children='Models'),
        dcc.Dropdown(
            id='pc-models',
            options=_model_options,
            value=['duska2020', 'holten2014'],
            multi=True,
            clearable=False,
            className='dash-dropdown',
        ),

        html.Button('Calculate', id='pc-calculate', className='btn-primary',
                     n_clicks=0, style={'marginTop': '16px'}),

        html.Div(id='pc-status', style={'marginTop': '8px', 'color': '#94a3b8',
                                         'fontSize': '12px'}),
    ], style={'width': '280px', 'flexShrink': 0})

    main = html.Div(style={'flex': 1, 'padding': '16px', 'minWidth': 0,
                            'overflowY': 'auto'}, children=[
        dcc.Loading(
            type='circle',
            color='#3b82f6',
            children=html.Div(id='pc-table-container', children=[
                dash_table.DataTable(
                    id='pc-table',
                    columns=[
                        {'name': 'Property', 'id': 'property'},
                        {'name': 'Unit', 'id': 'unit'},
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#111827',
                        'color': '#e2e8f0',
                        'fontWeight': 'bold',
                        'borderColor': '#1e3a5f',
                    },
                    style_cell={
                        'backgroundColor': '#0f1629',
                        'color': '#e2e8f0',
                        'borderColor': '#1e3a5f',
                        'textAlign': 'right',
                        'padding': '8px 12px',
                        'fontFamily': 'monospace',
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'property'}, 'textAlign': 'left',
                         'fontFamily': 'Helvetica, Arial, sans-serif',
                         'fontWeight': '500'},
                        {'if': {'column_id': 'unit'}, 'textAlign': 'left',
                         'fontFamily': 'Helvetica, Arial, sans-serif',
                         'color': '#94a3b8'},
                    ],
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'},
                         'backgroundColor': '#111827'},
                    ],
                ),
            ]),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
