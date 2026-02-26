"""Tab 5: Settings — layout definition."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from watereos_dash.style import PALETTE_OPTIONS, CMAP_OPTIONS, DEFAULTS, BG_OPTIONS

_palette_options = [{'label': k, 'value': k} for k in PALETTE_OPTIONS]
_cmap_options = [{'label': c, 'value': c} for c in CMAP_OPTIONS]
_bg_options = [{'label': k, 'value': v} for k, v in BG_OPTIONS.items()]


def layout():
    controls = html.Div(className='sidebar', style={'width': '320px', 'flexShrink': 0},
                         children=[
        html.H5('Appearance', style={'color': '#e2e8f0', 'marginBottom': '16px'}),

        # Curve palette
        html.Div(className='control-label', children='Curve Palette'),
        dcc.Dropdown(
            id='st-palette',
            options=_palette_options,
            value=DEFAULTS['curve_palette'],
            clearable=False,
            className='dash-dropdown',
        ),

        # Surface colormap
        html.Div(className='control-label', children='Surface Colormap'),
        dcc.Dropdown(
            id='st-cmap',
            options=_cmap_options,
            value=DEFAULTS['surface_cmap'],
            clearable=False,
            className='dash-dropdown',
        ),

        # Phase colors (dbc.Input supports type='color')
        html.Div(className='control-label', children='Phase Colors'),
        dbc.Row([
            dbc.Col([
                html.Div('Binodal', style={'color': '#94a3b8', 'fontSize': '11px'}),
                dbc.Input(id='st-binodal-color', type='color',
                          value=DEFAULTS['binodal_color']),
            ], width=4),
            dbc.Col([
                html.Div('Spinodal', style={'color': '#94a3b8', 'fontSize': '11px'}),
                dbc.Input(id='st-spinodal-color', type='color',
                          value=DEFAULTS['spinodal_color']),
            ], width=4),
            dbc.Col([
                html.Div('LLCP', style={'color': '#94a3b8', 'fontSize': '11px'}),
                dbc.Input(id='st-llcp-color', type='color',
                          value=DEFAULTS['llcp_color']),
            ], width=4),
        ], className='g-1', style={'marginBottom': '8px'}),

        # Line width
        html.Div(className='control-label', children='Line Width'),
        dcc.Slider(
            id='st-line-width',
            min=0.5, max=5.0, step=0.5, value=DEFAULTS['line_width'],
            marks={0.5: '0.5', 2: '2', 3.5: '3.5', 5: '5'},
        ),

        # Phase line width
        html.Div(className='control-label', children='Phase Line Width'),
        dcc.Slider(
            id='st-phase-line-width',
            min=0.5, max=4.0, step=0.5, value=DEFAULTS['phase_line_width'],
            marks={0.5: '0.5', 2: '2', 4: '4'},
        ),

        # Font size
        html.Div(className='control-label', children='Font Size'),
        dcc.Slider(
            id='st-font-size',
            min=8, max=20, step=1, value=DEFAULTS['font_size'],
            marks={8: '8', 12: '12', 16: '16', 20: '20'},
        ),

        # Grid
        dcc.Checklist(
            id='st-grid',
            options=[{'label': ' Show grid', 'value': 'grid'}],
            value=['grid'] if DEFAULTS['grid_enabled'] else [],
            style={'marginTop': '12px'},
        ),

        # Background
        html.Div(className='control-label', children='Background'),
        dcc.RadioItems(
            id='st-background',
            options=[{'label': f' {k}', 'value': v} for k, v in BG_OPTIONS.items()],
            value=DEFAULTS['bg_color'],
            style={'marginBottom': '12px'},
        ),

        # Reset
        html.Button('Reset to Defaults', id='st-reset', className='btn-secondary',
                     n_clicks=0, style={'width': '100%', 'marginTop': '12px'}),
    ])

    preview = html.Div(style={'flex': 1, 'padding': '8px', 'minWidth': 0}, children=[
        html.H5('Live Preview', style={'color': '#e2e8f0', 'marginBottom': '8px'}),
        dcc.Graph(
            id='st-preview',
            config={'displayModeBar': False},
            style={'height': 'calc(100vh - 120px)'},
        ),
    ])

    return html.Div(style={'display': 'flex', 'height': 'calc(100vh - 60px)'},
                     children=[controls, preview])
