"""
Plotly figure theming for the waterEoS-Visualizer app.

Provides palette/colormap options, default settings, layout builders,
and phase-diagram trace generators — all Plotly-native (no matplotlib).
"""

from watereos.model_registry import MODEL_ORDER

PALETTE_OPTIONS = {
    'Seaborn Deep': [
        '#4c72b0', '#dd8452', '#55a868', '#c44e52',
        '#8172b3', '#937860', '#da8bc3',
    ],
    'Seaborn Muted': [
        '#4878d0', '#ee854a', '#6acc64', '#d65f5f',
        '#956cb4', '#8c613c', '#dc7ec0',
    ],
    'Seaborn Bright': [
        '#023eff', '#ff7c00', '#1ac938', '#e8000b',
        '#8b2be2', '#9f4800', '#f14cc1',
    ],
    'Seaborn Colorblind': [
        '#0173b2', '#de8f05', '#029e73', '#d55e00',
        '#cc78bc', '#ca9161', '#fbafe4',
    ],
    'Tab10': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2',
    ],
    'Nature': [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
        '#F39B7F', '#8491B4', '#91D1C2',
    ],
    'Science': [
        '#0C5DA5', '#00B945', '#FF9500', '#FF2C00',
        '#845B97', '#474747', '#9E9E9E',
    ],
    'Biostasis': [
        '#F53A33', '#4DBEEE', '#7ED957', '#DAD4CE',
        '#C77DFF', '#FFB347', '#FF6B9D',
    ],
}

# Plotly colorscale names (lowercase)
CMAP_OPTIONS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'bluered', 'rdbu', 'spectral', 'turbo',
]

OCEAN_PALETTE = [
    '#3b82f6', '#06b6d4', '#22d3ee', '#a5b4fc',
    '#818cf8', '#38bdf8', '#67e8f9',
]
PALETTE_OPTIONS['Ocean'] = OCEAN_PALETTE

# --- Mantine-aligned dark theme colors (for Plotly figures) ---
COLORS = {
    'page_bg': '#1a1b1e',
    'sidebar_bg': '#25262b',
    'plot_bg': '#1a1b1e',
    'primary': '#339af0',
    'secondary': '#74c0fc',
    'text': '#c1c2c5',
    'muted': '#909296',
    'border': '#373a40',
    'grid': '#2c2e33',
    'card_bg': '#25262b',
}

BG_OPTIONS = {
    'Dark': '#1a1b1e',
    'Light gray': '#e5e7eb',
    'White': '#ffffff',
}

# --- Default settings (stored in dcc.Store) ---
DEFAULTS = {
    'curve_palette': 'Biostasis',
    'surface_cmap': 'rdbu',
    'binodal_color': '#339af0',
    'spinodal_color': '#ff922b',
    'llcp_color': '#51cf66',
    'line_width': 2.0,
    'phase_line_width': 1.5,
    'font_size': 12,
    'grid_enabled': True,
    'bg_color': '#1a1b1e',
    # Unit preferences (native SI defaults)
    'unit_density': 'kg/m\u00b3',
    'unit_volume': 'm\u00b3/kg',
    'unit_energy': 'J/kg',
    'unit_entropy': 'J/(kg\u00b7K)',
    'unit_bulk_modulus': 'MPa',
    'unit_viscosity': 'Pa\u00b7s',
}


def get_palette(settings=None):
    """Return the active curve color list."""
    s = settings or DEFAULTS
    name = s.get('curve_palette', DEFAULTS['curve_palette'])
    return PALETTE_OPTIONS.get(name, PALETTE_OPTIONS['Biostasis'])


def get_model_colors(settings=None):
    """Return model_key -> color mapping from active palette."""
    pal = get_palette(settings)
    return {k: pal[i % len(pal)] for i, k in enumerate(MODEL_ORDER)}


def make_layout(settings=None, title=None, xaxis_title=None, yaxis_title=None):
    """Build a go.Layout-compatible dict from settings."""
    s = settings or DEFAULTS
    bg = s.get('bg_color', DEFAULTS['bg_color'])
    fs = s.get('font_size', DEFAULTS['font_size'])
    grid_on = s.get('grid_enabled', DEFAULTS['grid_enabled'])

    dark = _is_dark(bg)
    text_color = COLORS['text'] if dark else '#1e293b'
    grid_color = COLORS['grid'] if dark else '#e2e4e8'

    layout = dict(
        paper_bgcolor=COLORS['page_bg'] if dark else '#ffffff',
        plot_bgcolor=bg,
        font=dict(family='Helvetica, Arial, sans-serif', size=fs, color=text_color),
        title=dict(text=title, font=dict(size=fs + 2)) if title else None,
        xaxis=dict(
            title=xaxis_title,
            gridcolor=grid_color,
            showgrid=grid_on,
            zeroline=False,
            linecolor=COLORS['border'] if dark else '#cbd5e1',
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=grid_color,
            showgrid=grid_on,
            zeroline=False,
            linecolor=COLORS['border'] if dark else '#cbd5e1',
        ),
        legend=dict(
            bgcolor='rgba(37,38,43,0.8)' if dark else 'rgba(255,255,255,0.9)',
            font=dict(size=fs - 1, color=text_color),
            bordercolor=COLORS['border'] if dark else '#cbd5e1',
            borderwidth=1,
            x=1.02, y=1, xanchor='left', yanchor='top',
        ),
        margin=dict(l=60, r=180, t=50, b=50),
        hovermode='closest',
    )
    return layout


def make_layout_3d(settings=None, title=None,
                   xaxis_title=None, yaxis_title=None, zaxis_title=None):
    """Build layout dict for 3D scenes."""
    s = settings or DEFAULTS
    bg = s.get('bg_color', DEFAULTS['bg_color'])
    fs = s.get('font_size', DEFAULTS['font_size'])
    grid_on = s.get('grid_enabled', DEFAULTS['grid_enabled'])
    dark = _is_dark(bg)
    text_color = COLORS['text'] if dark else '#1e293b'
    grid_color = COLORS['grid'] if dark else '#e2e4e8'

    axis_common = dict(
        gridcolor=grid_color,
        showgrid=grid_on,
        backgroundcolor=bg,
        color=text_color,
        zerolinecolor=grid_color,
    )

    layout = dict(
        paper_bgcolor=COLORS['page_bg'] if dark else '#ffffff',
        font=dict(family='Helvetica, Arial, sans-serif', size=fs, color=text_color),
        title=dict(text=title, font=dict(size=fs + 2)) if title else None,
        scene=dict(
            xaxis=dict(title=xaxis_title, **axis_common),
            yaxis=dict(title=yaxis_title, **axis_common),
            zaxis=dict(title=zaxis_title, **axis_common),
            bgcolor=bg,
            aspectmode='cube',
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return layout


def get_phase_traces(pd_data, settings=None):
    """Build a list of plotly go.Scatter traces for phase boundaries.

    Expects pd_data with keys: 'hdl_spinodal', 'ldl_spinodal', 'binodal', 'LLCP'.
    Each should have 'T_K' and 'p_MPa' arrays.
    """
    import plotly.graph_objects as go

    s = settings or DEFAULTS
    traces = []

    binodal_color = s.get('binodal_color', DEFAULTS['binodal_color'])
    spinodal_color = s.get('spinodal_color', DEFAULTS['spinodal_color'])
    llcp_color = s.get('llcp_color', DEFAULTS['llcp_color'])
    plw = s.get('phase_line_width', DEFAULTS['phase_line_width'])

    first_spinodal = True
    for sp_key in ('hdl_spinodal', 'ldl_spinodal'):
        sp = pd_data.get(sp_key)
        if sp and 'T_K' in sp and 'p_MPa' in sp:
            traces.append(go.Scatter(
                x=sp['T_K'], y=sp['p_MPa'],
                mode='lines',
                name='Spinodal' if first_spinodal else None,
                showlegend=first_spinodal,
                line=dict(color=spinodal_color, width=plw, dash='dash'),
                hovertemplate='T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra>Spinodal</extra>',
            ))
            first_spinodal = False

    if pd_data.get('binodal') and 'T_K' in pd_data['binodal']:
        bn = pd_data['binodal']
        traces.append(go.Scatter(
            x=bn['T_K'], y=bn['p_MPa'],
            mode='lines', name='Binodal',
            line=dict(color=binodal_color, width=plw),
            hovertemplate='T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra>Binodal</extra>',
        ))

    if pd_data.get('LLCP') and 'T_K' in pd_data['LLCP']:
        llcp = pd_data['LLCP']
        T_c = float(llcp['T_K'])
        P_c = float(llcp['p_MPa'])
        traces.append(go.Scatter(
            x=[T_c], y=[P_c],
            mode='markers', name=f'LLCP ({T_c:.1f} K, {P_c:.1f} MPa)',
            marker=dict(color=llcp_color, size=10, symbol='circle',
                        line=dict(width=1, color='white')),
            hovertemplate=f'LLCP<br>T={T_c:.2f} K<br>P={P_c:.2f} MPa<extra></extra>',
        ))

    return traces


def _is_dark(bg_hex):
    """Return True if bg is dark."""
    bg = bg_hex.lstrip('#')
    if len(bg) == 3:
        bg = ''.join(c * 2 for c in bg)
    r, g, b = int(bg[:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.3
