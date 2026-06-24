import plotly.graph_objects as go
from watereos_api.theming import apply_theme, CURVE_PALETTE, THEMES


def test_themes_defined():
    assert set(THEMES) == {"dark", "light"}
    assert THEMES["dark"]["bg_elev"] == "#111114"
    assert THEMES["light"]["bg_elev"] == "#FFFFFF"
    assert CURVE_PALETTE[0] == "#ef4444" and len(CURVE_PALETTE) == 7


def test_apply_theme_sets_layout_colors():
    fig = go.Figure(go.Scatter(x=[1], y=[1]))
    apply_theme(fig, "dark")
    lay = fig.layout
    assert lay.paper_bgcolor == "#111114"
    assert lay.plot_bgcolor == "#111114"
    assert lay.colorway == tuple(CURVE_PALETTE)
    apply_theme(fig, "light")
    assert fig.layout.paper_bgcolor == "#FFFFFF"


def test_apply_theme_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        apply_theme(go.Figure(), "neon")


def test_apply_theme_3d_scene_only_for_3d_figures():
    surf = go.Figure(go.Surface(z=[[1, 2], [3, 4]]))
    apply_theme(surf, "dark")
    assert surf.layout.scene.xaxis.gridcolor == "#1A1A1E"
    assert surf.layout.scene.zaxis.gridcolor == "#1A1A1E"
    assert surf.layout.scene.bgcolor == "#111114"
    # 2-D figure must NOT get a populated scene
    flat = go.Figure(go.Scatter(x=[1], y=[1]))
    apply_theme(flat, "light")
    assert flat.layout.scene.bgcolor is None
