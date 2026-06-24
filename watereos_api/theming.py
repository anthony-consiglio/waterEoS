"""Plotly theming derived from the React prototype's design tokens."""

import plotly.graph_objects as go

FONT_SANS = '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif'
FONT_MONO = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace'

CURVE_PALETTE = [
    "#ef4444", "#38bdf8", "#84cc16", "#e2e8f0",
    "#a78bfa", "#f59e0b", "#ec4899",
]

# Phase-diagram semantic colors (verbatim from plan spec §5b / prototype tokens)
PHASE_SPINODAL_COLOR = "#ef4444"
PHASE_BINODAL_COLOR = "#5b8def"
PHASE_LLCP_COLOR = "#a78bfa"

THEMES = {
    "dark": {
        "bg": "#0A0A0B", "bg_elev": "#111114",
        "border": "#1E1E22", "border_strong": "#2A2A30",
        "text": "#ECECF0", "text_muted": "#9398A2", "text_faint": "#5E626A",
        "grid": "#1A1A1E", "grid_strong": "#25252B", "accent": "#5b8def",
    },
    "light": {
        "bg": "#FAFAFA", "bg_elev": "#FFFFFF",
        "border": "#ECECEE", "border_strong": "#DEDEE1",
        "text": "#0A0A0A", "text_muted": "#6B7280", "text_faint": "#9CA3AF",
        "grid": "#EEEEF0", "grid_strong": "#DCDCDF", "accent": "#5b8def",
    },
}

_3D_TRACE_TYPES = (
    go.Surface, go.Scatter3d, go.Mesh3d, go.Cone,
    go.Streamtube, go.Isosurface, go.Volume,
)


def _axis(c):
    return dict(
        gridcolor=c["grid"], zerolinecolor=c["grid_strong"],
        linecolor=c["grid_strong"],
        tickfont=dict(family=FONT_MONO, color=c["text_faint"], size=11),
        title=dict(font=dict(color=c["text_muted"], size=12)),
    )


def apply_theme(fig, theme: str):
    """Mutate fig.layout in place with the named theme. Raises ValueError if unknown.
    Returns the figure to allow chaining."""
    if theme not in THEMES:
        raise ValueError(f"unknown theme: {theme!r}")
    c = THEMES[theme]
    fig.update_layout(
        paper_bgcolor=c["bg_elev"], plot_bgcolor=c["bg_elev"],
        font=dict(family=FONT_SANS, color=c["text"], size=12),
        colorway=CURVE_PALETTE,
        # asymmetric: extra right margin for legend/colorbar
        margin=dict(l=64, r=110, t=24, b=44),
        legend=dict(font=dict(family=FONT_MONO, size=11, color=c["text_muted"]),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=c["bg_elev"], bordercolor=c["border_strong"],
                        font=dict(family=FONT_MONO, size=12, color=c["text"])),
    )
    fig.update_xaxes(**_axis(c))
    fig.update_yaxes(**_axis(c))
    if any(isinstance(t, _3D_TRACE_TYPES) for t in fig.data):
        sc = dict(xaxis=_axis(c), yaxis=_axis(c), zaxis=_axis(c),
                  bgcolor=c["bg_elev"])
        fig.update_layout(scene=sc)
    return fig
