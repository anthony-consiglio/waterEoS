"""Tab 1: Property Explorer — layout definition."""

from dash import dcc, html
import dash_mantine_components as dmc

from watereos.model_registry import (
    MODEL_REGISTRY, MODEL_ORDER, get_display_label,
)

_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in MODEL_ORDER
]

_default_model = 'duska2020'
_default_info = MODEL_REGISTRY[_default_model]
_default_props = [
    {'label': get_display_label(p), 'value': p}
    for p in _default_info.properties
]


def layout():
    sidebar = dmc.Paper(
        shadow="xs",
        p="sm",
        style={
            'width': '260px',
            'flexShrink': 0,
            'borderRight': '1px solid var(--mantine-color-dark-4)',
            'minHeight': 'calc(100vh - 60px)',
            'overflowY': 'auto',
        },
        children=dmc.Stack(gap="xs", children=[
            dmc.Button('Update Plot', id='pe-update', fullWidth=True, n_clicks=0),

            dmc.Text('Model', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='pe-model',
                data=_model_options,
                value=_default_model,
                allowDeselect=False,
            ),

            dmc.Text('Property', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='pe-property',
                data=_default_props,
                value='rho',
                allowDeselect=False,
            ),

            dmc.Text('T Range (K)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='pe-tmin', value=_default_info.T_min, decimalScale=2),
                dmc.NumberInput(id='pe-tmax', value=_default_info.T_max, decimalScale=2),
            ]),

            dmc.Text('P Range (MPa)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='pe-pmin', value=_default_info.P_min, decimalScale=2),
                dmc.NumberInput(id='pe-pmax', value=_default_info.P_max, decimalScale=2),
            ]),

            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.Stack(gap=2, children=[
                    dmc.Text('Curves', size="xs", fw=600, c="dimmed", tt="uppercase"),
                    dmc.NumberInput(id='pe-ncurves', value=5, min=1, max=50),
                ]),
                dmc.Stack(gap=2, children=[
                    dmc.Text('Points', size="xs", fw=600, c="dimmed", tt="uppercase"),
                    dmc.NumberInput(id='pe-npoints', value=200, min=10, max=1000),
                ]),
            ]),

            dmc.Text('Curve Type', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SegmentedControl(
                id='pe-curve-type',
                data=[
                    {'label': 'Isobars', 'value': 'isobar'},
                    {'label': 'Isotherms', 'value': 'isotherm'},
                ],
                value='isobar',
                fullWidth=True,
            ),

            dmc.Text('Display Mode', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SegmentedControl(
                id='pe-display-mode',
                data=[
                    {'label': 'Curves', 'value': 'curves'},
                    {'label': '2D', 'value': 'surface2d'},
                    {'label': '3D', 'value': 'surface3d'},
                ],
                value='curves',
                fullWidth=True,
            ),

            html.Div(id='pe-zaxis-container', children=[
                dmc.Text('Z / Color Axis', size="xs", fw=600, c="dimmed", tt="uppercase"),
                dmc.Select(
                    id='pe-zaxis',
                    data=[
                        {'label': 'Property', 'value': 'property'},
                        {'label': 'Temperature', 'value': 'T'},
                        {'label': 'Pressure', 'value': 'P'},
                    ],
                    value='property',
                    allowDeselect=False,
                ),
            ], style={'display': 'none'}),

            dmc.Checkbox(
                id='pe-phase-boundaries',
                label='Show phase boundaries',
                checked=False,
            ),

            dmc.Button('Download CSV', id='pe-download-btn', variant="outline",
                       fullWidth=True, n_clicks=0),
            dcc.Download(id='pe-download'),
        ]),
    )

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#339af0',
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
