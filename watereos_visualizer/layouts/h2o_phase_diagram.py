"""Tab: H2O Phase Diagram — layout definition."""

from dash import dcc, html
import dash_mantine_components as dmc


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
            dmc.Text('Projection', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.RadioGroup(
                id='h2o-pd-projection',
                children=dmc.Stack(gap=4, children=[
                    dmc.Radio(label='T-V', value='tv'),
                    dmc.Radio(label='T-P', value='tp'),
                    dmc.Radio(label='3D P-T-V', value='ptv'),
                ]),
                value='tv',
            ),

            dmc.Divider(my="xs"),

            dmc.Text('Display Limits', size="xs", fw=600, c="dimmed", tt="uppercase"),

            dmc.Text('Specific Volume (m\u00b3/kg)', size="xs", c="dimmed"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='h2o-pd-vmin', value=7e-4, decimalScale=6, step=1e-4),
                dmc.NumberInput(id='h2o-pd-vmax', value=1.1e-3, decimalScale=6, step=1e-4),
            ]),

            dmc.Text('Temperature (K)', size="xs", c="dimmed"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='h2o-pd-tmin', value=190, decimalScale=1),
                dmc.NumberInput(id='h2o-pd-tmax', value=300, decimalScale=1),
            ]),

            dmc.Text('Pressure (MPa)', size="xs", c="dimmed"),
            dmc.SimpleGrid(cols=2, spacing="xs", children=[
                dmc.NumberInput(id='h2o-pd-pmin', value=0.001, decimalScale=4),
                dmc.NumberInput(id='h2o-pd-pmax', value=1000, decimalScale=1),
            ]),
        ]),
    )

    main = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dcc.Loading(
            type='circle',
            color='#339af0',
            children=dcc.Graph(
                id='h2o-pd-graph',
                config={'displayModeBar': True},
                style={'height': 'calc(100vh - 80px)'},
            ),
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[sidebar, main])
