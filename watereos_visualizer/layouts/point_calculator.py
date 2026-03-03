"""Tab 4: Point Calculator — layout definition."""

from dash import dcc, html, dash_table
import dash_mantine_components as dmc

from watereos.model_registry import MODEL_REGISTRY, MODEL_ORDER

_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in MODEL_ORDER
]


def layout():
    sidebar = dmc.Paper(
        shadow="xs",
        p="sm",
        style={
            'width': '280px',
            'flexShrink': 0,
            'borderRight': '1px solid var(--mantine-color-dark-4)',
            'minHeight': 'calc(100vh - 60px)',
            'overflowY': 'auto',
        },
        children=dmc.Stack(gap="xs", children=[
            dmc.Text('Temperature (K)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.NumberInput(id='pc-temperature', value=273.15, decimalScale=2),

            dmc.Text('Pressure (MPa)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.NumberInput(id='pc-pressure', value=0.1, decimalScale=2),

            dmc.Text('Models', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.MultiSelect(
                id='pc-models',
                data=_model_options,
                value=['duska2020', 'holten2014'],
            ),

            dmc.Button('Calculate', id='pc-calculate', fullWidth=True, n_clicks=0),

            dmc.Text(id='pc-status', size="xs", c="dimmed"),
        ]),
    )

    main = html.Div(style={'flex': 1, 'padding': '16px', 'minWidth': 0,
                            'overflowY': 'auto'}, children=[
        dcc.Loading(
            type='circle',
            color='#339af0',
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
                        'backgroundColor': '#25262b',
                        'color': '#c1c2c5',
                        'fontWeight': 'bold',
                        'borderColor': '#373a40',
                    },
                    style_cell={
                        'backgroundColor': '#1a1b1e',
                        'color': '#c1c2c5',
                        'borderColor': '#373a40',
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
                         'color': '#909296'},
                    ],
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'},
                         'backgroundColor': '#25262b'},
                    ],
                ),
            ]),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
