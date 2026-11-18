"""Dash-independent Plotly figure builders ported from watereos_visualizer."""
import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from watereos import getProp
from watereos.computation import (
    compute_multi_model_curves,
    compute_phase_diagram_data,
    compute_property_at_forced_x,
    compute_property_curves,
    compute_property_surface,
)
from watereos.model_registry import MODEL_REGISTRY
from watereos.eos_phase_diagram import LAYERS as EOS_PHASE_LAYERS, plot_eos_phase_diagram
from watereos.tv_phase_diagram import (
    compute_tv_phase_diagram,
    plot_tv_phase_diagram_plotly,
    plot_tp_phase_diagram_plotly,
    plot_ptv_phase_diagram_plotly,
)
from watereos.units import convert_array, display_label, get_factor
from watereos_api.theming import (
    apply_theme, CURVE_PALETTE,
    PHASE_SPINODAL_COLOR, PHASE_BINODAL_COLOR, PHASE_LLCP_COLOR,
)

logger = logging.getLogger(__name__)


def build_curves_figure(*, model, prop, T_range, P_range, n_curves,
                        n_points, isobar_mode, show_phase, theme, units):
    data = compute_property_curves(model, prop, tuple(T_range),
                                   tuple(P_range), n_curves, n_points,
                                   isobar_mode)
    fig = go.Figure()
    for i, (xs, ys, lbl) in enumerate(zip(
            data["x_values"], data["y_values"], data["curve_labels"])):
        y = convert_array(prop, list(ys), units)
        fig.add_trace(go.Scatter(
            x=list(xs), y=y, mode="lines", name=lbl,
            line=dict(color=CURVE_PALETTE[i % len(CURVE_PALETTE)], width=2),
        ))
    if show_phase and MODEL_REGISTRY[model].has_phase_diagram:
        pd = compute_phase_diagram_data(model)
        _add_phase_traces(fig, pd, model, prop, isobar_mode, T_range, P_range, units)
    fig.update_layout(
        title=dict(text=data.get("title")),
        xaxis_title=data.get("x_label"),
        yaxis_title=display_label(prop, units),
    )
    apply_theme(fig, theme)
    return fig


def _compute_along_curve(model_key, prop_key, T_arr, P_arr):
    """Evaluate a property along a T(P) curve using scatter mode."""
    PT = np.array([P_arr, T_arr], dtype=object)
    result = getProp(PT, model_key)
    Z = np.asarray(getattr(result, prop_key), dtype=float)
    return np.diag(Z)


# ---------------------------------------------------------------------------
# Liquid-liquid phase overlays
# ---------------------------------------------------------------------------

#: Liquid-liquid lines of compute_phase_diagram_data drawn by the overlays:
#: (data key, overlay kind, fraction keys of the liquids whose property is
#: drawn in property space). Trace names are those of the EoS Phase Diagram
#: tab (EOS_PHASE_LAYERS).
_PHASE_LINES = (
    ("ldl_spinodal", "spinodal", ("x",)),
    ("hdl_spinodal", "spinodal", ("x",)),
    ("binodal", "binodal", ("x_hdl", "x_ldl")),
)


def _phase_line_style(kind, width=1.5):
    if kind == "spinodal":
        return dict(color=PHASE_SPINODAL_COLOR, width=width, dash="dash")
    return dict(color=PHASE_BINODAL_COLOR, width=width)


def _runs(mask, min_len=2):
    """Slices of the runs of at least min_len consecutive True values of a boolean array."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def _phase_line_pieces(pd, key, fraction_keys):
    """Pieces of one liquid-liquid line, split at its NaN rows: [(T, P, [x, ...]), ...]."""
    curve = pd.get(key)
    if not curve:
        return []
    T = np.asarray(curve.get("T_K", []), dtype=float)
    P = np.asarray(curve.get("p_MPa", []), dtype=float)
    if T.size == 0 or T.shape != P.shape:
        return []
    fractions = []
    for fraction_key in fraction_keys:
        x = np.asarray(curve.get(fraction_key, []), dtype=float)
        fractions.append(x if x.shape == T.shape else None)
    return [(T[s], P[s], [None if x is None else x[s] for x in fractions])
            for s in _runs(np.isfinite(T) & np.isfinite(P), min_len=1)]


def _phase_line_property(model_key, prop_key, T, P, x, llcp):
    """Property along part of a liquid-liquid line.

    Evaluated for the liquid whose fraction is x where the model supports it
    (compute_property_at_forced_x), otherwise for the stable liquid. Points at
    the critical point llcp = (T_c, P_c) take the stable liquid's value there.
    Returns an array, NaN where the evaluation failed, or None.
    """
    vals = None
    if x is not None and not prop_key.endswith(("_A", "_B")):
        vals = compute_property_at_forced_x(model_key, prop_key, T, P, x)
    if vals is None or np.all(np.isnan(vals)):
        try:
            vals = _compute_along_curve(model_key, prop_key, T, P)
        except Exception:
            logger.debug("phase line property evaluation failed", exc_info=True)
            return None
    vals = np.array(vals, dtype=float)
    if llcp is not None:
        at_llcp = (np.abs(T - llcp[0]) < 1e-9) & (np.abs(P - llcp[1]) < 1e-9)
        if at_llcp.any():
            try:
                vals[at_llcp] = _compute_along_curve(
                    model_key, prop_key, np.array([llcp[0]]), np.array([llcp[1]]))[0]
            except Exception:
                logger.debug("LLCP property eval failed", exc_info=True)
    return vals


def _phase_line_runs(pd, model_key, T_range, P_range, prop_key=None):
    """Runs of the liquid-liquid lines inside the T-P window, one per contiguous stretch.

    The lines of compute_phase_diagram_data are traced as contours and can be
    reentrant: a line may leave the window and come back. Each stretch inside
    the window (and, with prop_key, with a finite property value) is its own
    run, so no segment is drawn across a gap.

    Returns a list of dicts with keys key (data key), kind ('spinodal' or
    'binodal'), T, P and prop (None without prop_key). In property space the
    coexistence line gives one run per coexisting liquid.
    """
    llcp_dict = pd.get("LLCP")
    llcp = None if llcp_dict is None else (float(llcp_dict["T_K"]), float(llcp_dict["p_MPa"]))
    runs = []
    for key, kind, fraction_keys in _PHASE_LINES:
        if prop_key is not None and kind == "binodal" and prop_key.endswith(("_A", "_B")):
            continue
        for T, P, fractions in _phase_line_pieces(pd, key, fraction_keys):
            inside = _clip_phase_tp(T, P, T_range, P_range)
            if not inside.any():
                continue
            if prop_key is None:
                runs += [dict(key=key, kind=kind, T=T[s], P=P[s], prop=None)
                         for s in _runs(inside)]
                continue
            for x in fractions:
                vals = _phase_line_property(
                    model_key, prop_key, T[inside], P[inside],
                    None if x is None else x[inside], llcp)
                if vals is None:
                    continue
                prop = np.full(T.shape, np.nan)
                prop[inside] = vals
                runs += [dict(key=key, kind=kind, T=T[s], P=P[s], prop=prop[s])
                         for s in _runs(inside & np.isfinite(prop))]
    return runs


def _add_phase_traces(fig, pd, model_key, prop_key, isobar_mode,
                      T_range, P_range, units):
    """Add the liquid-liquid lines and the LLCP, evaluated in property space.

    Each stability limit is drawn with the property of the liquid that ceases
    to exist there, and the coexistence line with that of each coexisting
    liquid, on the same axes as the isobar/isotherm property curves.

    Stability limits: PHASE_SPINODAL_COLOR dashed
    Coexistence:      PHASE_BINODAL_COLOR solid
    LLCP:             PHASE_LLCP_COLOR marker
    """
    info = MODEL_REGISTRY[model_key]
    T_lo = T_range[0] if T_range else info.T_min - 10
    T_hi = T_range[1] if T_range else info.T_max + 10
    P_lo = P_range[0] if P_range else -200.0
    P_hi = P_range[1] if P_range else 400.0

    shown = set()
    for run in _phase_line_runs(pd, model_key, (T_lo, T_hi), (P_lo, P_hi), prop_key):
        key = run["key"]
        x_vals = run["T"] if isobar_mode else run["P"]
        fig.add_trace(go.Scatter(
            x=x_vals.tolist(),
            y=convert_array(prop_key, run["prop"].tolist(), units),
            mode="lines",
            name=EOS_PHASE_LAYERS[key], legendgroup=key,
            showlegend=key not in shown,
            line=_phase_line_style(run["kind"]),
        ))
        shown.add(key)

    llcp = pd.get("LLCP")
    if llcp is not None:
        T_c = float(llcp["T_K"])
        P_c = float(llcp["p_MPa"])
        if T_lo <= T_c <= T_hi and P_lo <= P_c <= P_hi:
            try:
                prop_c = _compute_along_curve(
                    model_key, prop_key, np.array([T_c]), np.array([P_c]))
                if np.isfinite(prop_c[0]):
                    x_c = T_c if isobar_mode else P_c
                    y_c = convert_array(prop_key, [float(prop_c[0])], units)
                    fig.add_trace(go.Scatter(
                        x=[x_c], y=y_c,
                        mode="markers",
                        name=f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)",
                        marker=dict(color=PHASE_LLCP_COLOR, size=10,
                                    line=dict(width=1, color="white")),
                    ))
            except Exception:
                logger.debug("LLCP property eval failed", exc_info=True)


# ---------------------------------------------------------------------------
# Surface figure builders (2D heatmap + contour; 3D surface)
# ---------------------------------------------------------------------------

def _convert_z(prop_key, z, units):
    """Apply unit conversion factor to a 2D list of Z values."""
    if not units:
        return z
    f = get_factor(prop_key, units)
    if f == 1.0:
        return z
    return [[(v * f) for v in row] for row in z]


def _clip_phase_tp(T_arr, P_arr, T_range, P_range):
    """Boolean mask: which (T, P) points sit inside the user's bounding box."""
    T = np.asarray(T_arr)
    P = np.asarray(P_arr)
    return ((T >= T_range[0]) & (T <= T_range[1])
            & (P >= P_range[0]) & (P <= P_range[1]))


def _phase_segments_3d(model_key, prop_key, T_range, P_range):
    """Compute spinodal/binodal/LLCP curves in (T, P, property) space.

    Returns a list of dicts with keys {T, P, prop, type, group, name,
    show_legend}. Mirrors
    watereos_visualizer/callbacks/property_explorer.py::_compute_phase_3d.
    Each segment is one contiguous stretch of a line inside the user's
    bounding box (see _phase_line_runs); stretches whose property
    evaluation fails entirely are dropped.
    """
    try:
        pd = compute_phase_diagram_data(model_key)
    except Exception:
        logger.warning("phase 3D computation failed for %s", model_key, exc_info=True)
        return []

    T_lo, T_hi = float(T_range[0]), float(T_range[1])
    P_lo, P_hi = float(P_range[0]), float(P_range[1])
    segments = []

    shown = set()
    for run in _phase_line_runs(pd, model_key, (T_lo, T_hi), (P_lo, P_hi), prop_key):
        key = run["key"]
        segments.append({
            "T": run["T"].tolist(),
            "P": run["P"].tolist(),
            "prop": run["prop"].tolist(),
            "type": run["kind"],
            "group": key,
            "name": EOS_PHASE_LAYERS[key],
            "show_legend": key not in shown,
        })
        shown.add(key)

    llcp = pd.get("LLCP")
    if llcp is not None:
        T_c = float(llcp["T_K"])
        P_c = float(llcp["p_MPa"])
        if T_lo <= T_c <= T_hi and P_lo <= P_c <= P_hi:
            try:
                prop_c = _compute_along_curve(
                    model_key, prop_key, np.array([T_c]), np.array([P_c]))
                if np.isfinite(prop_c[0]):
                    segments.append({
                        "T": [T_c],
                        "P": [P_c],
                        "prop": [float(prop_c[0])],
                        "type": "llcp",
                        "name": f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)",
                        "show_legend": True,
                    })
            except Exception:
                logger.debug("LLCP 3D eval failed", exc_info=True)

    return segments


def _add_phase_overlay_surface2d(fig, model_key, T_range, P_range):
    """Add spinodal/binodal/LLCP traces in T-P space on top of a 2D heatmap.

    Properties are not evaluated for the 2D overlay — the boundaries are
    drawn purely on the (T, P) plane the heatmap occupies.
    """
    if not MODEL_REGISTRY[model_key].has_phase_diagram:
        return
    try:
        pd = compute_phase_diagram_data(model_key)
    except Exception:
        logger.warning("phase 2D computation failed for %s", model_key, exc_info=True)
        return

    # Liquid-liquid lines in T-P coordinates, one trace per stretch inside
    # the window; one legend entry per line.
    shown = set()
    for run in _phase_line_runs(pd, model_key, T_range, P_range):
        key = run["key"]
        label = EOS_PHASE_LAYERS[key]
        fig.add_trace(go.Scatter(
            x=run["T"].tolist(), y=run["P"].tolist(),
            mode="lines",
            name=label, legendgroup=key, showlegend=key not in shown,
            line=_phase_line_style(run["kind"]),
            hovertemplate=(
                f"{label}<br>T=%{{x:.2f}} K<br>P=%{{y:.2f}} MPa<extra></extra>"
            ),
        ))
        shown.add(key)

    # LLCP: single marker.
    llcp = pd.get("LLCP")
    if llcp is not None:
        T_c, P_c = float(llcp["T_K"]), float(llcp["p_MPa"])
        if (T_range[0] <= T_c <= T_range[1]
                and P_range[0] <= P_c <= P_range[1]):
            fig.add_trace(go.Scatter(
                x=[T_c], y=[P_c],
                mode="markers",
                name=f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)",
                marker=dict(color=PHASE_LLCP_COLOR, size=10,
                            line=dict(width=1, color="white")),
            ))


def _add_phase_overlay_surface3d(fig, model_key, prop_key, T_range, P_range, units):
    """Add spinodal/binodal/LLCP Scatter3d traces evaluated in property space.

    Each phase boundary point is lifted to (T, P, property) so the overlay
    sits on the rendered 3D surface rather than at the base plane.
    """
    if not MODEL_REGISTRY[model_key].has_phase_diagram:
        return
    segments = _phase_segments_3d(model_key, prop_key, T_range, P_range)
    if not segments:
        return

    for seg in segments:
        prop_vals = convert_array(prop_key, list(seg["prop"]), units)
        common = dict(
            x=list(seg["T"]), y=list(seg["P"]), z=list(prop_vals),
            name=seg.get("name"), showlegend=bool(seg.get("show_legend")),
            legendgroup=seg.get("group"),
        )
        if seg["type"] == "llcp":
            fig.add_trace(go.Scatter3d(
                **common, mode="markers",
                marker=dict(color=PHASE_LLCP_COLOR, size=5, symbol="circle",
                            line=dict(width=1, color="white")),
            ))
        elif seg["type"] == "spinodal":
            fig.add_trace(go.Scatter3d(
                **common, mode="lines",
                line=dict(color=PHASE_SPINODAL_COLOR, width=4, dash="dash"),
            ))
        else:  # binodal
            fig.add_trace(go.Scatter3d(
                **common, mode="lines",
                line=dict(color=PHASE_BINODAL_COLOR, width=4),
            ))


def build_surface2d_figure(*, model, prop, T_range, P_range, n_points,
                           colormap, show_phase, theme, units):
    """Build a 2D heatmap + contour figure of a property over T-P space.

    Mirrors watereos_visualizer/callbacks/property_explorer.py::_render_surface_2d.
    T_grid has shape (n_P, n_T) with default meshgrid indexing, so:
      T_1d = T_grid[0]        (T values, constant along rows)
      P_1d = P_grid[:, 0]     (P values, constant along columns)

    When show_phase is True the figure picks up spinodal/binodal/LLCP
    traces drawn in T-P coordinates on top of the heatmap (the legacy
    Dash app behaviour).
    """
    d = compute_property_surface(model, prop, tuple(T_range),
                                 tuple(P_range), n_points)
    z = _convert_z(prop, [list(row) for row in d["Z"]], units)
    # T varies along columns (axis=1), P varies along rows (axis=0)
    T_1d = d["T_grid"][0].tolist()        # shape (n_T,)
    P_1d = d["P_grid"][:, 0].tolist()     # shape (n_P,)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=T_1d, y=P_1d, z=z,
        colorscale=colormap,
        colorbar=dict(title=display_label(prop, units)),
        hovertemplate="T=%{x:.2f} K<br>P=%{y:.2f} MPa<br>%{z:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        x=T_1d, y=P_1d, z=z,
        colorscale=colormap, showscale=False, ncontours=12,
        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        line=dict(width=0.5, color="rgba(255,255,255,0.5)"),
        hoverinfo="skip",
    ))
    if show_phase:
        _add_phase_overlay_surface2d(fig, model, T_range, P_range)
    fig.update_layout(
        xaxis_title="Temperature [K]",
        yaxis_title="Pressure [MPa]",
    )
    apply_theme(fig, theme)
    if show_phase:
        # Pin the legend to the upper-left of the plot so it doesn't collide
        # with the colorbar on the right edge. Merges with the font/bgcolor
        # set by apply_theme.
        fig.update_layout(legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ))
    return fig


def build_surface3d_figure(*, model, prop, T_range, P_range, n_points,
                           colormap, show_phase, theme, units):
    """Build a rotatable 3D surface figure of a property over T-P space.

    Mirrors watereos_visualizer/callbacks/property_explorer.py::_render_surface_3d.
    x=T_grid, y=P_grid, z=Z puts T on x-axis and P on y-axis (property on z).

    When show_phase is True the figure additionally carries Scatter3d
    traces for the spinodal/binodal/LLCP, each evaluated in property
    space so the overlay sits on the surface.
    """
    d = compute_property_surface(model, prop, tuple(T_range),
                                 tuple(P_range), n_points)
    z = _convert_z(prop, [list(row) for row in d["Z"]], units)
    prop_label = display_label(prop, units)
    fig = go.Figure(go.Surface(
        x=d["T_grid"].tolist(), y=d["P_grid"].tolist(), z=z,
        colorscale=colormap, opacity=0.9,
        colorbar=dict(title=prop_label),
    ))
    if show_phase:
        _add_phase_overlay_surface3d(fig, model, prop, T_range, P_range, units)
    fig.update_layout(scene=dict(
        xaxis_title="Temperature [K]",
        yaxis_title="Pressure [MPa]",
        zaxis_title=prop_label,
        aspectmode="cube",
    ))
    apply_theme(fig, theme)
    if show_phase:
        # Pin the legend to the upper-left of the plot so it doesn't collide
        # with the colorbar on the right edge. Merges with the font/bgcolor
        # set by apply_theme.
        fig.update_layout(legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ))
    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison figure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EoS phase diagram figure builder
# ---------------------------------------------------------------------------

def build_eos_phase_figure(*, model, show, theme,
                           T_range=None, P_range=None, auto_limits=True):
    """EoS phase diagram of a two-state model (watereos.eos_phase_diagram).

    With auto_limits the figure keeps the diagram's standard window,
    100-375 K and -400 to 420 MPa; otherwise T_range and P_range apply.
    """
    return plot_eos_phase_diagram(
        model, show=show, theme=theme,
        T_range=None if auto_limits else T_range,
        P_range=None if auto_limits else P_range)


# ---------------------------------------------------------------------------
# H2O phase diagram figure builders (T-V, T-P, P-T-V projections)
# ---------------------------------------------------------------------------

_DIAGRAM_CACHE = {}


def _get_diagram():
    if "d" not in _DIAGRAM_CACHE:
        _DIAGRAM_CACHE["d"] = compute_tv_phase_diagram(verbose=False)
    return _DIAGRAM_CACHE["d"]


def build_h2o_figure(*, projection, V_range, T_range, P_range, theme):
    diagram = _get_diagram()
    if projection == "tv":
        v0, v1 = (V_range or [7e-4, 1.1e-3])
        t0, t1 = (T_range or [190, 300])
        fig = plot_tv_phase_diagram_plotly(diagram, V_min=v0, V_max=v1,
                                           T_min=t0, T_max=t1)
    elif projection == "tp":
        t0, t1 = (T_range or [190, 300])
        p0, p1 = (P_range or [1e-4, 1000])
        fig = plot_tp_phase_diagram_plotly(diagram, T_min=t0, T_max=t1,
                                           P_min=p0, P_max=p1)
    else:  # ptv
        v0, v1 = (V_range or [7e-4, 1.1e-3])
        p1 = (P_range or [0, 1000])[1]
        fig = plot_ptv_phase_diagram_plotly(diagram, T_stride=4,
                                            n_pts_per_phase=80,
                                            V_min=v0, V_max=v1, P_max=p1)
    # Self-title so the React shell can render the card bare.
    _proj_label = {
        "tv": "T–V projection",
        "tp": "T–P projection",
        "ptv": "3D P–T–V",
    }.get(projection, projection)
    fig.update_layout(title=dict(text=f"H₂O Phase Diagram — {_proj_label}"))
    apply_theme(fig, theme)
    return fig


def build_compare_figure(*, model_keys, prop, T_range, P_range, n_curves,
                         n_points, isobar_mode, layout, theme, units):
    multi = compute_multi_model_curves(model_keys, prop, tuple(T_range),
                                       tuple(P_range), n_curves, n_points,
                                       isobar_mode)
    if layout == "sidebyside":
        titles = [MODEL_REGISTRY[m].display_name for m in model_keys]
        fig = make_subplots(rows=1, cols=len(model_keys),
                            subplot_titles=titles, shared_yaxes=True)
        for col, mk in enumerate(model_keys, start=1):
            d = multi[mk]
            for i, (xs, ys, lbl) in enumerate(zip(
                    d["x_values"], d["y_values"], d["curve_labels"])):
                fig.add_trace(go.Scatter(
                    x=list(xs), y=convert_array(prop, list(ys), units),
                    mode="lines", name=f"{mk}:{lbl}", showlegend=(col == 1),
                    line=dict(color=CURVE_PALETTE[i % len(CURVE_PALETTE)])),
                    row=1, col=col)
        # Apply the shared y-axis title; set x-axis title on each subplot.
        x_label = multi[model_keys[0]].get("x_label")
        fig.update_layout(yaxis_title=display_label(prop, units))
        for col in range(1, len(model_keys) + 1):
            fig.update_xaxes(title_text=x_label, row=1, col=col)
    else:
        fig = go.Figure()
        for j, mk in enumerate(model_keys):
            d = multi[mk]
            for i, (xs, ys, lbl) in enumerate(zip(
                    d["x_values"], d["y_values"], d["curve_labels"])):
                fig.add_trace(go.Scatter(
                    x=list(xs), y=convert_array(prop, list(ys), units),
                    mode="lines",
                    name=f"{MODEL_REGISTRY[mk].display_name} · {lbl}",
                    legendgroup=mk,
                    showlegend=(i == 0),
                    line=dict(color=CURVE_PALETTE[
                        (j * n_curves + i) % len(CURVE_PALETTE)])))
        fig.update_layout(
            xaxis_title=multi[model_keys[0]].get("x_label"),
            yaxis_title=display_label(prop, units),
        )
    # Self-title so the React shell can render the card bare. Title appears
    # above both layouts (subplot-suptitle for sidebyside, normal title for
    # overlay).
    _prop_label = display_label(prop, units)
    _n = len(model_keys)
    _plural = "s" if _n != 1 else ""
    fig.update_layout(title=dict(
        text=f"{_prop_label} — {_n} model{_plural} compared"))
    apply_theme(fig, theme)
    return fig
