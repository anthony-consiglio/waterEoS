"""Tab 5: Settings — layout definition."""

from dash import dcc, html
import dash_mantine_components as dmc

from watereos_visualizer.style import PALETTE_OPTIONS, CMAP_OPTIONS, DEFAULTS, BG_OPTIONS
from watereos_visualizer.units import UNIT_OPTIONS, UNIT_DEFAULTS, CATEGORY_LABELS

_palette_options = [{'label': k, 'value': k} for k in PALETTE_OPTIONS]
_cmap_options = [{'label': c, 'value': c} for c in CMAP_OPTIONS]

# Ordered list of unit setting keys for consistent UI
_UNIT_KEYS = [
    'unit_density', 'unit_volume', 'unit_energy',
    'unit_entropy', 'unit_bulk_modulus', 'unit_viscosity',
]


def _unit_controls():
    """Build the list of unit Select dropdowns."""
    controls = []
    for key in _UNIT_KEYS:
        controls.append(
            dmc.Text(CATEGORY_LABELS[key], size="xs", fw=600, c="dimmed",
                     tt="uppercase"),
        )
        controls.append(
            dmc.Select(
                id=f'st-{key.replace("_", "-")}',
                data=UNIT_OPTIONS[key],
                value=DEFAULTS.get(key, UNIT_DEFAULTS[key]),
                allowDeselect=False,
            ),
        )
    return controls


def layout():
    controls = dmc.Paper(
        shadow="xs",
        p="sm",
        style={
            'width': '320px',
            'flexShrink': 0,
            'borderRight': '1px solid var(--mantine-color-dark-4)',
            'minHeight': 'calc(100vh - 60px)',
            'overflowY': 'auto',
        },
        children=dmc.Stack(gap="sm", children=[
            dmc.Text('Appearance', size="lg", fw=600),

            dmc.Text('Curve Palette', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='st-palette',
                data=_palette_options,
                value=DEFAULTS['curve_palette'],
                allowDeselect=False,
            ),

            dmc.Text('Surface Colormap', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Select(
                id='st-cmap',
                data=_cmap_options,
                value=DEFAULTS['surface_cmap'],
                allowDeselect=False,
            ),

            dmc.Text('Phase Colors', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.SimpleGrid(cols=3, spacing="xs", children=[
                dmc.Stack(gap=2, children=[
                    dmc.Text('Binodal', size="xs", c="dimmed"),
                    dmc.ColorInput(id='st-binodal-color',
                                   value=DEFAULTS['binodal_color'],
                                   format="hex"),
                ]),
                dmc.Stack(gap=2, children=[
                    dmc.Text('Spinodal', size="xs", c="dimmed"),
                    dmc.ColorInput(id='st-spinodal-color',
                                   value=DEFAULTS['spinodal_color'],
                                   format="hex"),
                ]),
                dmc.Stack(gap=2, children=[
                    dmc.Text('LLCP', size="xs", c="dimmed"),
                    dmc.ColorInput(id='st-llcp-color',
                                   value=DEFAULTS['llcp_color'],
                                   format="hex"),
                ]),
            ]),

            dmc.Text('Line Width', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Slider(
                id='st-line-width',
                min=0.5, max=5.0, step=0.5, value=DEFAULTS['line_width'],
                marks=[{'value': v, 'label': str(v)} for v in [0.5, 2, 3.5, 5]],
            ),

            dmc.Text('Phase Line Width', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Slider(
                id='st-phase-line-width',
                min=0.5, max=4.0, step=0.5, value=DEFAULTS['phase_line_width'],
                marks=[{'value': v, 'label': str(v)} for v in [0.5, 2, 4]],
            ),

            dmc.Text('Font Size', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.Slider(
                id='st-font-size',
                min=8, max=20, step=1, value=DEFAULTS['font_size'],
                marks=[{'value': v, 'label': str(v)} for v in [8, 12, 16, 20]],
            ),

            dmc.Checkbox(
                id='st-grid',
                label='Show grid',
                checked=DEFAULTS['grid_enabled'],
            ),

            dmc.Text('Background', size="xs", fw=600, c="dimmed", tt="uppercase"),
            dmc.RadioGroup(
                id='st-background',
                children=dmc.Stack(gap=4, children=[
                    dmc.Radio(label=k, value=v)
                    for k, v in BG_OPTIONS.items()
                ]),
                value=DEFAULTS['bg_color'],
            ),

            dmc.Divider(my="sm"),
            dmc.Text('Units', size="lg", fw=600),
            *_unit_controls(),

            dmc.Divider(my="sm"),
            dmc.Button('Reset to Defaults', id='st-reset', variant="outline",
                       fullWidth=True, n_clicks=0),
        ]),
    )

    preview = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        dmc.Text('Live Preview', size="lg", fw=600, mb="xs"),
        dcc.Graph(
            id='st-preview',
            config={'displayModeBar': False},
            style={'height': 'calc(100vh - 120px)'},
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[controls, preview])
