"""Tab 2: Phase Diagram — layout definition."""

from dash import dcc, html
import dash_mantine_components as dmc

from watereos.model_registry import MODEL_REGISTRY, models_with_phase_diagram

_pd_models = models_with_phase_diagram()
_model_options = [
    {'label': MODEL_REGISTRY[k].display_name, 'value': k}
    for k in _pd_models
]
_default = _pd_models[0] if _pd_models else 'duska2020'


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
            dmc.Text('Model', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='pd-model',
                data=_model_options,
                value=_default,
                allowDeselect=False,
            ),

            dmc.Text('Show', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.CheckboxGroup(
                id='pd-show',
                children=dmc.Stack(gap=4, children=[
                    dmc.Checkbox(label='LL Binodal', value='binodal'),
                    dmc.Checkbox(label='HDL Spinodal', value='hdl_spinodal'),
                    dmc.Checkbox(label='LDL Spinodal', value='ldl_spinodal'),
                    dmc.Checkbox(label='LLCP', value='LLCP'),
                    dmc.Checkbox(label='TMD', value='tmd'),
                    dmc.Checkbox(label='Widom line', value='widom'),
                    dmc.Checkbox(label='Ice Ih liquidus', value='ice_ih'),
                    dmc.Checkbox(label='Ice III liquidus', value='ice_iii'),
                    dmc.Checkbox(label='Ih nucleation', value='nuc_ih'),
                    dmc.Checkbox(label='III nucleation', value='nuc_iii'),
                    dmc.Checkbox(label='Kauzmann', value='kauzmann'),
                ]),
                value=['binodal', 'hdl_spinodal', 'ldl_spinodal', 'LLCP',
                       'tmd', 'widom', 'ice_ih', 'ice_iii', 'nuc_ih',
                       'nuc_iii', 'kauzmann'],
            ),

            dmc.Checkbox(id='pd-auto-limits', label='Auto limits', checked=False),

            html.Div(id='pd-manual-limits', children=[
                dmc.Text('Temperature Range (K)', size="xs", fw=600, c="dimmed", tt="uppercase"),
                dmc.SimpleGrid(cols=2, spacing="xs", children=[
                    dmc.NumberInput(id='pd-tmin', value=150, decimalScale=1),
                    dmc.NumberInput(id='pd-tmax', value=300, decimalScale=1),
                ]),
                dmc.Text('Pressure Range (MPa)', size="xs", fw=600, c="dimmed", tt="uppercase"),
                dmc.SimpleGrid(cols=2, spacing="xs", children=[
                    dmc.NumberInput(id='pd-pmin', value=0, decimalScale=1),
                    dmc.NumberInput(id='pd-pmax', value=300, decimalScale=1),
                ]),
            ], style={'display': 'block'}),

            dmc.Button('Compute', id='pd-compute', fullWidth=True, n_clicks=0),

            dmc.Text(id='pd-status', size="xs", c="dimmed"),
        ]),
    )

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#339af0',
            children=dcc.Graph(
                id='pd-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
