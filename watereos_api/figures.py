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


def _extend_to_llcp(T_arr, P_arr, prop_arr, pd_data, model_key, prop_key):
    """Prepend the LLCP point so phase boundary curves converge at the critical point.

    Ported verbatim from watereos_visualizer/callbacks/property_explorer.py.
    """
    if 'LLCP' not in pd_data:
        return T_arr, P_arr, prop_arr
    llcp = pd_data['LLCP']
    T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])
    prop_c = _compute_along_curve(model_key, prop_key, np.array([T_c]), np.array([P_c]))
    if not np.isfinite(prop_c[0]):
        return T_arr, P_arr, prop_arr
    return (np.concatenate([[T_c], T_arr]),
            np.concatenate([[P_c], P_arr]),
            np.concatenate([prop_c, prop_arr]))


def _add_phase_traces(fig, pd, model_key, prop_key, isobar_mode,
                      T_range, P_range, units):
    """Add spinodal, binodal, and LLCP traces evaluated in property space.

    Mirrors watereos_visualizer/callbacks/property_explorer.py::_compute_phase_curves
    semantics: evaluates the property along each phase boundary curve, extends each
    branch to the LLCP via _extend_to_llcp, and plots on the same axes as the
    isobar/isotherm property curves.

    Spinodal: PHASE_SPINODAL_COLOR dashed
    Binodal:  PHASE_BINODAL_COLOR solid
    LLCP:     PHASE_LLCP_COLOR marker
    """
    info = MODEL_REGISTRY[model_key]
    T_lo = T_range[0] if T_range else info.T_min - 10
    T_hi = T_range[1] if T_range else info.T_max + 10
    P_lo = P_range[0] if P_range else -200.0
    P_hi = P_range[1] if P_range else 400.0

    spinodal = pd.get("spinodal")
    if spinodal is not None:
        p_arr = np.asarray(spinodal["p_array"])
        first_spinodal = True
        for branch_key, x_key in [
            ("T_upper", "x_hi_upper"),
            ("T_lower", "x_lo_lower"),
        ]:
            T_branch = np.asarray(spinodal.get(branch_key, []))
            if T_branch.size == 0:
                continue
            valid = (
                np.isfinite(T_branch)
                & (T_branch >= T_lo) & (T_branch <= T_hi)
                & (p_arr >= P_lo) & (p_arr <= P_hi)
            )
            if not np.any(valid):
                continue
            T_b, P_b = T_branch[valid], p_arr[valid]

            x_sp = np.asarray(spinodal.get(x_key, []))
            prop_vals = None
            if x_sp.size == p_arr.size and not prop_key.endswith(("_A", "_B")):
                prop_vals = compute_property_at_forced_x(
                    model_key, prop_key, T_b, P_b, x_sp[valid])
            if prop_vals is None or np.all(np.isnan(prop_vals)):
                try:
                    prop_vals = _compute_along_curve(model_key, prop_key, T_b, P_b)
                except Exception:
                    logger.debug("spinodal property eval failed", exc_info=True)
                    continue

            Te, Pe, prop_ext = _extend_to_llcp(T_b, P_b, prop_vals, pd, model_key, prop_key)

            x_vals = Te if isobar_mode else Pe
            y_vals = convert_array(prop_key, list(prop_ext), units)
            mask = np.isfinite(prop_ext)
            x_plot = x_vals[mask].tolist()
            y_plot = [y_vals[i] for i, m in enumerate(mask) if m]

            if not x_plot:
                continue

            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot,
                mode="lines",
                name="Spinodal" if first_spinodal else None,
                showlegend=first_spinodal,
                line=dict(color=PHASE_SPINODAL_COLOR, width=1.5, dash="dash"),
            ))
            first_spinodal = False

    binodal = pd.get("binodal")
    if binodal is not None:
        p_arr = np.asarray(binodal["p_array"])
        T_bn = np.asarray(binodal.get("T_binodal", []))
        if T_bn.size > 0 and not prop_key.endswith(("_A", "_B")):
            valid = (
                np.isfinite(T_bn)
                & (T_bn >= T_lo) & (T_bn <= T_hi)
                & (p_arr >= P_lo) & (p_arr <= P_hi)
            )
            if np.any(valid):
                T_b, P_b = T_bn[valid], p_arr[valid]
                x_lo_arr = np.asarray(binodal.get("x_lo", []))
                x_hi_arr = np.asarray(binodal.get("x_hi", []))
                for x_arr, lbl in [(x_lo_arr, "Binodal"), (x_hi_arr, None)]:
                    prop_vals = None
                    if x_arr.size == p_arr.size:
                        prop_vals = compute_property_at_forced_x(
                            model_key, prop_key, T_b, P_b, x_arr[valid])
                    if prop_vals is None or np.all(np.isnan(prop_vals)):
                        try:
                            prop_vals = _compute_along_curve(
                                model_key, prop_key, T_b, P_b)
                        except Exception:
                            logger.debug("binodal property eval failed", exc_info=True)
                            continue

                    Te, Pe, prop_ext = _extend_to_llcp(
                        T_b, P_b, prop_vals, pd, model_key, prop_key)

                    x_vals = Te if isobar_mode else Pe
                    y_vals = convert_array(prop_key, list(prop_ext), units)
                    mask = np.isfinite(prop_ext)
                    x_plot = x_vals[mask].tolist()
                    y_plot = [y_vals[i] for i, m in enumerate(mask) if m]

                    if not x_plot:
                        continue

                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot,
                        mode="lines",
                        name=lbl,
                        showlegend=lbl is not None,
                        line=dict(color=PHASE_BINODAL_COLOR, width=1.5),
                    ))

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

    Returns a list of dicts with keys {T, P, prop, type, name, show_legend}.
    Mirrors watereos_visualizer/callbacks/property_explorer.py::_compute_phase_3d.
    Each segment is clipped to the user's bounding box; segments whose
    property evaluation fails entirely are dropped.
    """
    try:
        pd = compute_phase_diagram_data(model_key)
    except Exception:
        logger.warning("phase 3D computation failed for %s", model_key, exc_info=True)
        return []

    T_lo, T_hi = float(T_range[0]), float(T_range[1])
    P_lo, P_hi = float(P_range[0]), float(P_range[1])
    segments = []

    spinodal = pd.get("spinodal")
    if spinodal is not None:
        p_arr = np.asarray(spinodal["p_array"])
        first = True
        for branch_key, x_key in [("T_upper", "x_hi_upper"),
                                  ("T_lower", "x_lo_lower")]:
            T_branch = np.asarray(spinodal.get(branch_key, []))
            if T_branch.size == 0:
                continue
            valid = (np.isfinite(T_branch)
                     & (T_branch >= T_lo) & (T_branch <= T_hi)
                     & (p_arr >= P_lo) & (p_arr <= P_hi))
            if not np.any(valid):
                continue
            T_b, P_b = T_branch[valid], p_arr[valid]
            x_sp = np.asarray(spinodal.get(x_key, []))
            prop_vals = None
            if x_sp.size == p_arr.size and not prop_key.endswith(("_A", "_B")):
                prop_vals = compute_property_at_forced_x(
                    model_key, prop_key, T_b, P_b, x_sp[valid])
            if prop_vals is None or np.all(np.isnan(prop_vals)):
                try:
                    prop_vals = _compute_along_curve(model_key, prop_key, T_b, P_b)
                except Exception:
                    logger.debug("spinodal 3D eval failed", exc_info=True)
                    continue
            Te, Pe, prop_ext = _extend_to_llcp(
                T_b, P_b, prop_vals, pd, model_key, prop_key)
            mask = np.isfinite(prop_ext)
            if not mask.any():
                continue
            segments.append({
                "T": Te[mask].tolist(),
                "P": Pe[mask].tolist(),
                "prop": prop_ext[mask].tolist(),
                "type": "spinodal",
                "name": "Spinodal" if first else None,
                "show_legend": first,
            })
            first = False

    binodal = pd.get("binodal")
    if binodal is not None:
        p_arr = np.asarray(binodal["p_array"])
        T_bn = np.asarray(binodal.get("T_binodal", []))
        if T_bn.size > 0 and not prop_key.endswith(("_A", "_B")):
            valid = (np.isfinite(T_bn)
                     & (T_bn >= T_lo) & (T_bn <= T_hi)
                     & (p_arr >= P_lo) & (p_arr <= P_hi))
            if np.any(valid):
                T_b, P_b = T_bn[valid], p_arr[valid]
                x_lo_arr = np.asarray(binodal.get("x_lo", []))
                x_hi_arr = np.asarray(binodal.get("x_hi", []))
                for x_arr, lbl in [(x_lo_arr, "Binodal"), (x_hi_arr, None)]:
                    prop_vals = None
                    if x_arr.size == p_arr.size:
                        prop_vals = compute_property_at_forced_x(
                            model_key, prop_key, T_b, P_b, x_arr[valid])
                    if prop_vals is None or np.all(np.isnan(prop_vals)):
                        try:
                            prop_vals = _compute_along_curve(
                                model_key, prop_key, T_b, P_b)
                        except Exception:
                            logger.debug("binodal 3D eval failed", exc_info=True)
                            continue
                    Te, Pe, prop_ext = _extend_to_llcp(
                        T_b, P_b, prop_vals, pd, model_key, prop_key)
                    mask = np.isfinite(prop_ext)
                    if not mask.any():
                        continue
                    segments.append({
                        "T": Te[mask].tolist(),
                        "P": Pe[mask].tolist(),
                        "prop": prop_ext[mask].tolist(),
                        "type": "binodal",
                        "name": lbl,
                        "show_legend": lbl is not None,
                    })

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

    # Spinodal: two branches, single legend entry.
    spinodal = pd.get("spinodal")
    if spinodal is not None:
        p_arr = np.asarray(spinodal["p_array"])
        first = True
        for branch_key in ("T_upper", "T_lower"):
            T_branch = np.asarray(spinodal.get(branch_key, []))
            if T_branch.size == 0:
                continue
            mask = (np.isfinite(T_branch)
                    & _clip_phase_tp(T_branch, p_arr, T_range, P_range))
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=T_branch[mask].tolist(), y=p_arr[mask].tolist(),
                mode="lines",
                name="Spinodal" if first else None,
                showlegend=first,
                line=dict(color=PHASE_SPINODAL_COLOR, width=1.5, dash="dash"),
                hovertemplate=(
                    "Spinodal<br>T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra></extra>"
                ),
            ))
            first = False

    # Binodal: single curve in (T_binodal, p_array) coords.
    binodal = pd.get("binodal")
    if binodal is not None:
        T_bn = np.asarray(binodal.get("T_binodal", []))
        p_arr = np.asarray(binodal.get("p_array", []))
        if T_bn.size > 0 and p_arr.size == T_bn.size:
            mask = (np.isfinite(T_bn)
                    & _clip_phase_tp(T_bn, p_arr, T_range, P_range))
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=T_bn[mask].tolist(), y=p_arr[mask].tolist(),
                    mode="lines", name="Binodal",
                    line=dict(color=PHASE_BINODAL_COLOR, width=1.5),
                    hovertemplate=(
                        "Binodal<br>T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra></extra>"
                    ),
                ))

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

_PD_STYLES = {
    "binodal":      dict(color="#9333ea", dash="solid", width=2),
    "hdl_spinodal": dict(color="#ec4899", dash="dash", width=2),
    "ldl_spinodal": dict(color="#be185d", dash="dot", width=2),
    "tmd":          dict(color="#ffffff", dash="dash", width=2),
    "widom":        dict(color="#f97316", dash="dashdot", width=2),
    "ice_ih":       dict(color="#3b82f6", dash="solid", width=2),
    "ice_iii":      dict(color="#ef4444", dash="solid", width=2),
    # nuc_ih is the homogeneous Ih nucleation locus (Thom / T_H line). It
    # used to share gray with nuc_iii, but with the Kauzmann line no longer
    # in the default Info-hero overlay set, the bright green it used to
    # occupy is now repurposed for the Thom curve so it stands out as the
    # most physically relevant freezing limit.
    "nuc_ih":       dict(color="#22c55e", dash="solid", width=2),
    "nuc_iii":      dict(color="#9ca3af", dash="dash", width=1.5),
    "kauzmann":     dict(color="#22c55e", dash="solid", width=2),
}

_PD_LABELS = {
    'binodal':      'Binodal',
    'hdl_spinodal': 'HDL Spinodal',
    'ldl_spinodal': 'LDL Spinodal',
    'tmd':          'TMD (max density)',
    'widom':        'Widom line',
    'ice_ih':       'Ice Ih liquidus',
    'ice_iii':      'Ice III liquidus',
    'nuc_ih':       'Homogeneous nucleation (Ih)',
    'nuc_iii':      'Homogeneous nucleation (III)',
    'kauzmann':     'Kauzmann temperature',
    'LLCP':         'LLCP',
}

# Maps show-key to actual dict key in compute_phase_diagram_data return value
_PD_KEYMAP = {
    "ice_ih": "ice_ih_liquidus",
    "ice_iii": "ice_iii_liquidus",
    "nuc_ih": "nucleation_ih",
    "nuc_iii": "nucleation_iii",
}


def build_eos_phase_figure(*, model, show, theme,
                           T_range=None, P_range=None, auto_limits=True):
    pd = compute_phase_diagram_data(model)
    fig = go.Figure()

    for key in show:
        if key == "LLCP":
            llcp = pd.get("LLCP") or {}
            if llcp.get("T_K") is not None:
                fig.add_trace(go.Scatter(
                    x=[float(llcp["T_K"])], y=[float(llcp["p_MPa"])],
                    mode="markers", name="LLCP",
                    marker=dict(color="#9333ea", size=12,
                                line=dict(width=1, color="white"))))
            continue

        data_key = _PD_KEYMAP.get(key, key)
        d = pd.get(data_key) or {}
        st = _PD_STYLES.get(key, dict(color="#888", dash="solid", width=2))

        # binodal uses {T_binodal, p_array} from compute_phase_diagram_data;
        # the Dash callback silently skipped this curve because it checked
        # for T_K/p_MPa which are absent here. The API renders it correctly.
        if key == "binodal":
            T_vals = d.get("T_binodal")
            P_vals = d.get("p_array")
        else:
            T_vals = d.get("T_K")
            P_vals = d.get("p_MPa")

        if T_vals is None or P_vals is None:
            continue

        fig.add_trace(go.Scatter(
            x=list(T_vals), y=list(P_vals), mode="lines", name=_PD_LABELS.get(key, key),
            line=dict(color=st["color"], dash=st["dash"], width=st["width"])))

    # Triple point marker (shown alongside ice_iii)
    tp = pd.get("triple_point") or {}
    if tp.get("T_K") is not None and ("ice_ih" in show or "ice_iii" in show):
        fig.add_trace(go.Scatter(
            x=[float(tp["T_K"])], y=[float(tp["p_MPa"])], mode="markers",
            name="Triple point",
            marker=dict(color="#166534", size=12, symbol="square",
                        line=dict(width=1, color="white"))))

    # Self-titled so the React UI can render the chart card "bare" (no
    # outer card title) without losing context. Falls back to the bare key
    # if a registry entry is missing (e.g. an unfamiliar model).
    try:
        _display = MODEL_REGISTRY[model].display_name
    except KeyError:
        _display = model
    fig.update_layout(
        title=dict(text=f"{_display} — EoS Phase Diagram"),
        xaxis_title="Temperature [K]",
        yaxis_title="Pressure [MPa]",
    )
    apply_theme(fig, theme)
    if not auto_limits:
        if T_range is not None:
            fig.update_xaxes(range=list(T_range))
        if P_range is not None:
            fig.update_yaxes(range=list(P_range))
    return fig


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
