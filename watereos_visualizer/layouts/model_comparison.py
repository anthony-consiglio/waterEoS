"""Tab 3: Model Comparison — layout definition."""

from dash import dcc, html
import dash_mantine_components as dmc

from watereos.model_registry import (
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
            dmc.Text('Models (select 2+)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.MultiSelect(
                id='mc-models',
                data=_model_options,
                value=_default_models,
            ),

            dmc.Text('Property', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='mc-property',
                data=_prop_options,
                value='rho' if 'rho' in _common else (_common[0] if _common else None),
                allowDeselect=False,
            ),

            dmc.Text('Temperature Range (K)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='mc-tmin', value=200, decimalScale=2),
                dmc.NumberInput(id='mc-tmax', value=300, decimalScale=2),
            ]),

            dmc.Text('Pressure Range (MPa)', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='mc-pmin', value=0.1, decimalScale=2),
                dmc.NumberInput(id='mc-pmax', value=200, decimalScale=2),
            ]),

            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.Stack(gap=2, children=[
                    dmc.Text('Curves', size="xs", fw=600, c="dimmed", tt="uppercase"),
                    dmc.NumberInput(id='mc-ncurves', value=5, min=1, max=50),
                ]),
                dmc.Stack(gap=2, children=[
                    dmc.Text('Points/Curve', size="xs", fw=600, c="dimmed", tt="uppercase"),
                    dmc.NumberInput(id='mc-npoints', value=200, min=10, max=1000),
                ]),
            ]),

            dmc.Text('Curve Type', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SegmentedControl(
                id='mc-curve-type',
                data=[
                    {'label': 'Isobars', 'value': 'isobar'},
                    {'label': 'Isotherms', 'value': 'isotherm'},
                ],
                value='isobar',
                fullWidth=True,
            ),

            dmc.Text('Layout', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SegmentedControl(
                id='mc-layout',
                data=[
                    {'label': 'Overlay', 'value': 'overlay'},
                    {'label': 'Side by Side', 'value': 'sidebyside'},
                ],
                value='overlay',
                fullWidth=True,
            ),

            dmc.Button('Update Plot', id='mc-update', fullWidth=True, n_clicks=0),
        ]),
    )

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#339af0',
            children=dcc.Graph(
                id='mc-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
