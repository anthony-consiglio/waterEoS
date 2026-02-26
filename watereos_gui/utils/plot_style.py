"""
Plot style — seaborn-inspired rcParams for the waterEoS GUI.

Mutable settings are stored in ``_current`` and exposed via getter/setter
functions.  Other modules read settings through the getters so that changes
made in the Settings tab take effect on the next "Update Plot".
"""

import copy

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------

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
    'Seaborn Pastel': [
        '#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b',
        '#d0bbff', '#debb9b', '#fab0e4',
    ],
    'Seaborn Dark': [
        '#001c7f', '#b1400d', '#12711c', '#8c0800',
        '#591e71', '#592f0d', '#a23582',
    ],
    'Tab10': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2',
    ],
    'Set1': [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf',
    ],
    'Set2': [
        '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
        '#a6d854', '#ffd92f', '#e5c494',
    ],
    'Dark2': [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
        '#66a61e', '#e6ab02', '#a6761d',
    ],
    'Paired': [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
        '#fb9a99', '#e31a1c', '#fdbf6f',
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

CMAP_OPTIONS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'coolwarm', 'RdBu_r', 'Spectral', 'turbo',
]

# Convenience aliases (kept for backward-compat imports)
SEABORN_DEEP = PALETTE_OPTIONS['Seaborn Deep']

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    'app_theme': 'dark',
    'curve_palette': 'Biostasis',
    'surface_cmap': 'coolwarm',
    'binodal_color': '#4DBEEE',
    'spinodal_color': '#F53A33',
    'llcp_color': '#7ED957',
    'line_width': 2.0,
    'phase_line_width': 1.5,
    'axes_linewidth': 0.8,
    'font_size': 12,
    'export_dpi': 200,
    'grid_enabled': True,
    'box_enabled': False,
    'bg_color': '#1a1a1a',
}

# Mutable runtime state
_current = copy.deepcopy(_DEFAULTS)

# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

def get_curve_palette():
    """Return the active curve color list."""
    return list(PALETTE_OPTIONS[_current['curve_palette']])

def get_curve_palette_name():
    return _current['curve_palette']

def get_surface_cmap():
    return _current['surface_cmap']

def get_phase_styles():
    """Return current PHASE_DIAGRAM_STYLES dict (rebuilt from _current)."""
    return {
        'binodal':  dict(color=_current['binodal_color'], linestyle='-',
                         linewidth=_current['phase_line_width'], label='Binodal'),
        'spinodal': dict(color=_current['spinodal_color'], linestyle='--',
                         linewidth=_current['phase_line_width'], label='Spinodal'),
        'LLCP':     dict(color=_current['llcp_color'], marker='o',
                         markersize=8, linestyle='none', label='LLCP'),
    }

def get_export_dpi():
    return _current['export_dpi']

def get_line_width():
    return _current['line_width']

def get_phase_line_width():
    return _current['phase_line_width']

def get_axes_linewidth():
    return _current['axes_linewidth']

def get_font_size():
    return _current['font_size']

def get_grid_enabled():
    return _current['grid_enabled']

def get_box_enabled():
    return _current['box_enabled']

def get_bg_color():
    return _current['bg_color']

def get_app_theme():
    return _current['app_theme']

# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

def set_curve_palette(name):
    if name in PALETTE_OPTIONS:
        _current['curve_palette'] = name

def set_surface_cmap(name):
    _current['surface_cmap'] = name

def set_phase_color(key, color):
    """key is one of 'binodal', 'spinodal', 'llcp'."""
    ckey = f'{key}_color'
    if ckey in _current:
        _current[ckey] = color

def set_export_dpi(val):
    _current['export_dpi'] = int(val)

def set_line_width(val):
    _current['line_width'] = float(val)

def set_phase_line_width(val):
    _current['phase_line_width'] = float(val)

def set_axes_linewidth(val):
    _current['axes_linewidth'] = float(val)

def set_font_size(val):
    _current['font_size'] = int(val)

def set_grid_enabled(enabled):
    _current['grid_enabled'] = bool(enabled)

def set_box_enabled(enabled):
    _current['box_enabled'] = bool(enabled)

def set_bg_color(color):
    _current['bg_color'] = color

def set_app_theme(theme):
    if theme in ('dark', 'light'):
        _current['app_theme'] = theme

def reset_defaults():
    _current.update(copy.deepcopy(_DEFAULTS))

# ---------------------------------------------------------------------------
# Map model keys to consistent colors (derived from active palette)
# ---------------------------------------------------------------------------

_MODEL_KEYS = [
    'duska2020', 'holten2014', 'caupin2019', 'grenke2025',
    'singh2017', 'water1', 'IAPWS95',
]

def get_model_colors():
    """Return MODEL_COLORS dict rebuilt from the active palette."""
    pal = get_curve_palette()
    return {k: pal[i % len(pal)] for i, k in enumerate(_MODEL_KEYS)}

# Backward-compat: static dict (used only at import time; prefer get_model_colors())
MODEL_COLORS = {k: SEABORN_DEEP[i % len(SEABORN_DEEP)] for i, k in enumerate(_MODEL_KEYS)}

# Legacy alias — modules that import PHASE_DIAGRAM_STYLES directly will get
# the *default* values; prefer get_phase_styles() for dynamic reads.
PHASE_DIAGRAM_STYLES = {
    'binodal':  dict(color=SEABORN_DEEP[0], linestyle='-',  linewidth=1.5, label='Binodal'),
    'spinodal': dict(color=SEABORN_DEEP[3], linestyle='--', linewidth=1.5, label='Spinodal'),
    'LLCP':     dict(color=SEABORN_DEEP[2], marker='o', markersize=8, linestyle='none', label='LLCP'),
}

# ---------------------------------------------------------------------------
# rcParams application
# ---------------------------------------------------------------------------

def _is_dark_bg(bg_hex):
    """Return True if the background is dark (luminance < 0.3)."""
    bg = bg_hex.lstrip('#')
    if len(bg) == 3:
        bg = ''.join(c * 2 for c in bg)
    r, g, b = int(bg[:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    # Relative luminance (sRGB approximation)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.3


def _build_rcparams():
    """Build rcParams dict from _current settings."""
    fs = _current['font_size']
    pal = get_curve_palette()
    bg = _current['bg_color']
    grid_on = _current['grid_enabled']
    dark = _is_dark_bg(bg)

    if dark:
        text_color = '#DAD4CE'
        edge_color = '#444444'
        grid_color = '#333333'
        fig_face = '#141419'
        legend_edge = '#444444'
    elif bg == '#ffffff':
        text_color = '#333333'
        edge_color = '#cccccc'
        grid_color = '#e0e0e0'
        fig_face = 'white'
        legend_edge = '#cccccc'
    else:
        text_color = '#333333'
        edge_color = '#cccccc'
        grid_color = 'white'
        fig_face = 'white'
        legend_edge = '#cccccc'

    return {
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': fs,
        'axes.labelsize': fs + 1,
        'axes.titlesize': fs + 2,
        'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1,
        'legend.fontsize': fs - 2,

        # Text / tick colors
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,

        # Axes
        'axes.facecolor': bg,
        'axes.edgecolor': edge_color,
        'axes.linewidth': _current['axes_linewidth'],
        'axes.grid': grid_on,
        'axes.axisbelow': True,
        'axes.prop_cycle': mpl.cycler(color=pal),

        # Grid
        'grid.color': grid_color,
        'grid.linewidth': 0.6 if dark else 1.2,
        'grid.alpha': 0.7 if dark else 1.0,

        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,

        # Lines
        'lines.linewidth': _current['line_width'],
        'lines.markersize': 6,

        # Figure
        'figure.facecolor': fig_face,
        'figure.dpi': 100,
        'savefig.dpi': _current['export_dpi'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': fig_face,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.85 if dark else 0.9,
        'legend.edgecolor': legend_edge,
        'legend.facecolor': bg,
        'legend.labelcolor': text_color,
    }


def apply_watereos_style():
    """Apply the waterEoS GUI style globally via rcParams."""
    mpl.rcParams.update(_build_rcparams())


def style_axes(ax):
    """Apply additional per-axes styling that rcParams can't handle."""
    # Update the figure facecolor (figures created before rcParams changed
    # keep their old facecolor unless explicitly updated)
    fig_face = mpl.rcParams.get('figure.facecolor', 'white')
    ax.figure.set_facecolor(fig_face)

    box = _current['box_enabled']
    dark = _is_dark_bg(_current['bg_color'])
    ax.spines['top'].set_visible(box)
    ax.spines['right'].set_visible(box)
    lw = _current['axes_linewidth']
    if box:
        color = '#DAD4CE' if dark else '#000000'
    else:
        color = '#444444' if dark else '#cccccc'
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Qt application stylesheets
# ---------------------------------------------------------------------------

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #141419;
    color: #DAD4CE;
}
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    background: #141419;
}
QTabBar::tab {
    background: #1a1a1a;
    color: #8C857F;
    border: 1px solid #2a2a2a;
    padding: 8px 18px;
    margin-right: 2px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #222228;
    color: #F3F3F5;
    border-bottom: 2px solid #F53A33;
}
QTabBar::tab:hover:!selected {
    background: #1e1e24;
    color: #DAD4CE;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #F3F3F5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #F3F3F5;
}
QLabel {
    color: #DAD4CE;
}
QPushButton {
    background-color: #222228;
    color: #DAD4CE;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #2a2a30;
    border-color: #F53A33;
    color: #F3F3F5;
}
QPushButton:pressed {
    background-color: #F53A33;
    color: #F3F3F5;
}
QComboBox {
    background-color: #1a1a1a;
    color: #DAD4CE;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 4px 8px;
}
QComboBox:hover {
    border-color: #F53A33;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    color: #DAD4CE;
    selection-background-color: #F53A33;
    selection-color: #F3F3F5;
    border: 1px solid #3a3a3a;
}
QSpinBox, QDoubleSpinBox {
    background-color: #1a1a1a;
    color: #DAD4CE;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 3px 6px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #F53A33;
}
QCheckBox, QRadioButton {
    color: #DAD4CE;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #8C857F;
    background: #1a1a1a;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #F53A33;
    border-color: #F53A33;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    text-align: center;
    color: #DAD4CE;
}
QProgressBar::chunk {
    background-color: #F53A33;
}
QSplitter::handle {
    background-color: #2a2a2a;
}
QScrollBar:vertical {
    background: #141419;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #3a3a3a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #F53A33;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QToolTip {
    background-color: #222228;
    color: #DAD4CE;
    border: 1px solid #3a3a3a;
    padding: 4px;
}
NavigationToolbar2QT {
    background-color: #141419;
    border: none;
}
NavigationToolbar2QT QToolButton {
    background-color: #222228;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 3px;
    color: #DAD4CE;
}
NavigationToolbar2QT QToolButton:hover {
    background-color: #2a2a30;
    border-color: #F53A33;
}
"""

LIGHT_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f0f0f0;
    color: #333333;
}
QTabWidget::pane {
    border: 1px solid #c0c0c0;
    background: #f0f0f0;
}
QTabBar::tab {
    background: #e0e0e0;
    color: #666666;
    border: 1px solid #c0c0c0;
    padding: 8px 18px;
    margin-right: 2px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #222222;
    border-bottom: 2px solid #F53A33;
}
QTabBar::tab:hover:!selected {
    background: #e8e8e8;
    color: #333333;
}
QGroupBox {
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #222222;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #222222;
}
QLabel {
    color: #333333;
}
QPushButton {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #b0b0b0;
    border-radius: 3px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #f5f5f5;
    border-color: #F53A33;
    color: #222222;
}
QPushButton:pressed {
    background-color: #F53A33;
    color: #ffffff;
}
QComboBox {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #b0b0b0;
    border-radius: 3px;
    padding: 4px 8px;
}
QComboBox:hover {
    border-color: #F53A33;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #333333;
    selection-background-color: #F53A33;
    selection-color: #ffffff;
    border: 1px solid #b0b0b0;
}
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #b0b0b0;
    border-radius: 3px;
    padding: 3px 6px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #F53A33;
}
QCheckBox, QRadioButton {
    color: #333333;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #999999;
    background: #ffffff;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #F53A33;
    border-color: #F53A33;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QProgressBar {
    background-color: #e0e0e0;
    border: 1px solid #c0c0c0;
    border-radius: 3px;
    text-align: center;
    color: #333333;
}
QProgressBar::chunk {
    background-color: #F53A33;
}
QSplitter::handle {
    background-color: #c0c0c0;
}
QScrollBar:vertical {
    background: #f0f0f0;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #b0b0b0;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #F53A33;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QToolTip {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #b0b0b0;
    padding: 4px;
}
NavigationToolbar2QT {
    background-color: #f0f0f0;
    border: none;
}
NavigationToolbar2QT QToolButton {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 3px;
    padding: 3px;
    color: #333333;
}
NavigationToolbar2QT QToolButton:hover {
    background-color: #f5f5f5;
    border-color: #F53A33;
}
"""


def get_app_stylesheet():
    """Return the Qt stylesheet for the current app theme."""
    if _current['app_theme'] == 'light':
        return LIGHT_STYLESHEET
    return DARK_STYLESHEET
