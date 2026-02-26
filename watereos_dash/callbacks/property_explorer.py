"""Tab 1: Property Explorer — callbacks."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, no_update, dcc

from watereos_gui.utils.model_registry import (
    MODEL_REGISTRY, get_display_label, PROPERTY_LABELS, PROPERTY_UNITS,
)
from watereos_gui.backend.computation import (
    compute_property_curves, compute_property_surface,
    compute_phase_diagram_data, compute_property_at_forced_x,
)
from watereos_dash.style import (
    get_palette, get_phase_traces, make_layout, make_layout_3d, DEFAULTS,
)


def register(app):
    # --- Update property dropdown when model changes ---
    @app.callback(
        [Output('pe-property', 'options'),
         Output('pe-property', 'value'),
         Output('pe-tmin', 'value'),
         Output('pe-tmax', 'value'),
         Output('pe-pmin', 'value'),
         Output('pe-pmax', 'value'),
         Output('pe-phase-boundaries', 'options')],
        Input('pe-model', 'value'),
        prevent_initial_call=True,
    )
    def update_model(model_key):
        if not model_key or model_key not in MODEL_REGISTRY:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        info = MODEL_REGISTRY[model_key]
        prop_opts = [{'label': get_display_label(p), 'value': p} for p in info.properties]
        default_prop = 'rho' if 'rho' in info.properties else info.properties[0]

        phase_opts = [{'label': ' Show phase boundaries', 'value': 'show'}]
        if not info.has_phase_diagram:
            phase_opts = [{'label': ' Show phase boundaries (N/A)', 'value': 'show',
                           'disabled': True}]

        return (prop_opts, default_prop,
                info.T_min, info.T_max, info.P_min, info.P_max,
                phase_opts)

    # --- Show/hide Z-axis dropdown ---
    @app.callback(
        Output('pe-zaxis-container', 'style'),
        Input('pe-display-mode', 'value'),
        prevent_initial_call=True,
    )
    def toggle_zaxis(mode):
        if mode == 'curves':
            return {'display': 'none'}
        return {'display': 'block'}

    # --- Compute callback: button click → store data ---
    @app.callback(
        Output('pe-data-store', 'data'),
        Input('pe-update', 'n_clicks'),
        [State('pe-model', 'value'),
         State('pe-property', 'value'),
         State('pe-tmin', 'value'),
         State('pe-tmax', 'value'),
         State('pe-pmin', 'value'),
         State('pe-pmax', 'value'),
         State('pe-ncurves', 'value'),
         State('pe-npoints', 'value'),
         State('pe-curve-type', 'value'),
         State('pe-display-mode', 'value'),
         State('pe-zaxis', 'value'),
         State('pe-phase-boundaries', 'value')],
        prevent_initial_call=True,
    )
    def compute_and_store(n_clicks, model_key, prop_key, tmin, tmax, pmin, pmax,
                          n_curves, n_points, curve_type, display_mode, z_choice,
                          phase_bnd):
        if not model_key or not prop_key:
            return no_update

        T_range = (float(tmin), float(tmax))
        P_range = (float(pmin), float(pmax))
        n_curves = int(n_curves or 5)
        n_points = int(n_points or 200)
        isobar_mode = curve_type == 'isobar'
        show_phase = 'show' in (phase_bnd or [])
        z_choice = z_choice or 'property'

        if display_mode == 'curves':
            return _store_curves(model_key, prop_key, T_range, P_range,
                                 n_curves, n_points, isobar_mode, show_phase)
        elif display_mode == 'surface2d':
            return _store_surface_2d(model_key, prop_key, T_range, P_range,
                                     n_points, z_choice, show_phase)
        else:
            return _store_surface_3d(model_key, prop_key, T_range, P_range,
                                     n_points, z_choice, show_phase)

    # --- Render callback: stored data + settings + z-axis → figure ---
    @app.callback(
        Output('pe-graph', 'figure'),
        [Input('pe-data-store', 'data'),
         Input('settings-store', 'data'),
         Input('pe-zaxis', 'value')],
        prevent_initial_call=True,
    )
    def render_plot(pe_data, settings, z_choice):
        if not pe_data:
            return no_update

        settings = settings or DEFAULTS
        mode = pe_data['mode']

        # Use live z_choice for surfaces (overrides stored value)
        if z_choice and mode in ('surface2d', 'surface3d'):
            pe_data = dict(pe_data, z_choice=z_choice)

        if mode == 'curves':
            return _render_curves(pe_data, settings)
        elif mode == 'surface2d':
            return _render_surface_2d(pe_data, settings)
        else:
            return _render_surface_3d(pe_data, settings)

    # --- CSV download ---
    @app.callback(
        Output('pe-download', 'data'),
        Input('pe-download-btn', 'n_clicks'),
        [State('pe-model', 'value'),
         State('pe-property', 'value'),
         State('pe-tmin', 'value'),
         State('pe-tmax', 'value'),
         State('pe-pmin', 'value'),
         State('pe-pmax', 'value'),
         State('pe-ncurves', 'value'),
         State('pe-npoints', 'value'),
         State('pe-curve-type', 'value')],
        prevent_initial_call=True,
    )
    def download_csv(n_clicks, model_key, prop_key, tmin, tmax, pmin, pmax,
                     n_curves, n_points, curve_type):
        if not model_key or not prop_key:
            return no_update

        T_range = (float(tmin), float(tmax))
        P_range = (float(pmin), float(pmax))
        isobar_mode = curve_type == 'isobar'
        data = compute_property_curves(
            model_key, prop_key, T_range, P_range,
            int(n_curves or 5), int(n_points or 200), isobar_mode)

        lines = [f"# {data['title']}"]
        for i, (x, y, label) in enumerate(
                zip(data['x_values'], data['y_values'], data['curve_labels'])):
            lines.append(f"\n# Curve {i+1}: {label}")
            lines.append(f"{data['x_label']},{data['y_label']}")
            for xv, yv in zip(x, y):
                lines.append(f"{xv},{yv}")

        return dcc.send_string('\n'.join(lines), filename=f'{model_key}_{prop_key}.csv')


# =====================================================================
# Data storage helpers (compute → JSON-serializable dict)
# =====================================================================

def _store_curves(model_key, prop_key, T_range, P_range,
                  n_curves, n_points, isobar_mode, show_phase):
    data = compute_property_curves(
        model_key, prop_key, T_range, P_range, n_curves, n_points, isobar_mode)

    curves = []
    for x, y, label in zip(data['x_values'], data['y_values'], data['curve_labels']):
        mask = np.isfinite(y)
        curves.append({
            'x': x[mask].tolist(),
            'y': y[mask].tolist(),
            'label': label,
        })

    stored = {
        'mode': 'curves',
        'title': data['title'],
        'x_label': data['x_label'],
        'y_label': data['y_label'],
        'curves': curves,
        'phase': None,
    }

    if show_phase and MODEL_REGISTRY[model_key].has_phase_diagram:
        stored['phase'] = _compute_phase_curves(model_key, prop_key, isobar_mode)

    return stored


def _store_surface_2d(model_key, prop_key, T_range, P_range,
                      n_points, z_choice, show_phase):
    data = compute_property_surface(model_key, prop_key, T_range, P_range, n_points)
    info = MODEL_REGISTRY[model_key]

    stored = {
        'mode': 'surface2d',
        'model_display_name': info.display_name,
        'prop_label': get_display_label(prop_key),
        'z_choice': z_choice,
        'T_1d': data['T_grid'][0].tolist(),
        'P_1d': data['P_grid'][:, 0].tolist(),
        'Z': data['Z'].tolist(),
        'phase_pd': None,
    }

    if show_phase and info.has_phase_diagram:
        stored['phase_pd'] = _serialize_phase_diagram(model_key)

    return stored


def _store_surface_3d(model_key, prop_key, T_range, P_range,
                      n_points, z_choice, show_phase):
    data = compute_property_surface(model_key, prop_key, T_range, P_range, n_points)
    info = MODEL_REGISTRY[model_key]

    stored = {
        'mode': 'surface3d',
        'model_display_name': info.display_name,
        'prop_label': get_display_label(prop_key),
        'z_choice': z_choice,
        'T_grid': data['T_grid'].tolist(),
        'P_grid': data['P_grid'].tolist(),
        'Z': data['Z'].tolist(),
        'phase': None,
    }

    if show_phase and info.has_phase_diagram:
        stored['phase'] = _compute_phase_3d(model_key, prop_key)

    return stored


# =====================================================================
# Render helpers (stored data + settings → Plotly figure)
# =====================================================================

def _render_curves(pe_data, settings):
    palette = get_palette(settings)
    lw = settings.get('line_width', DEFAULTS['line_width'])

    fig = go.Figure()
    for i, curve in enumerate(pe_data['curves']):
        fig.add_trace(go.Scatter(
            x=curve['x'], y=curve['y'],
            mode='lines', name=curve['label'],
            line=dict(color=palette[i % len(palette)], width=lw),
            hovertemplate=f"{curve['label']}<br>%{{x:.2f}}<br>%{{y:.6g}}<extra></extra>",
        ))

    _add_phase_traces_2d(fig, pe_data.get('phase'), settings)

    layout = make_layout(settings, title=pe_data['title'],
                         xaxis_title=pe_data['x_label'],
                         yaxis_title=pe_data['y_label'])
    fig.update_layout(**layout)
    return fig


def _render_surface_2d(pe_data, settings):
    cmap = settings.get('surface_cmap', DEFAULTS['surface_cmap'])
    z_choice = pe_data['z_choice']
    prop_label = pe_data['prop_label']
    T_1d = pe_data['T_1d']
    P_1d = pe_data['P_1d']
    Z = pe_data['Z']

    fig = go.Figure()

    if z_choice == 'property':
        fig.add_trace(go.Heatmap(
            x=T_1d, y=P_1d, z=Z,
            colorscale=cmap, colorbar=dict(title=prop_label),
            hovertemplate='T=%{x:.2f} K<br>P=%{y:.2f} MPa<br>%{z:.6g}<extra></extra>',
        ))
        fig.add_trace(go.Contour(
            x=T_1d, y=P_1d, z=Z,
            colorscale=cmap, showscale=False, ncontours=12,
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            line=dict(width=0.5, color='rgba(255,255,255,0.5)'),
            hoverinfo='skip',
        ))
        xlabel, ylabel = 'Temperature [K]', 'Pressure [MPa]'
    elif z_choice == 'T':
        Z_arr = np.array(Z)
        fig.add_trace(go.Heatmap(
            x=P_1d, y=T_1d, z=Z_arr.T.tolist(),
            colorscale=cmap, colorbar=dict(title=prop_label),
        ))
        xlabel, ylabel = 'Pressure [MPa]', 'Temperature [K]'
    else:  # P
        fig.add_trace(go.Heatmap(
            x=T_1d, y=P_1d, z=Z,
            colorscale=cmap, colorbar=dict(title=prop_label),
        ))
        xlabel, ylabel = 'Temperature [K]', 'Pressure [MPa]'

    # Phase boundaries from stored phase diagram data
    if pe_data.get('phase_pd'):
        for trace in get_phase_traces(pe_data['phase_pd'], settings):
            fig.add_trace(trace)

    title = f"{pe_data['model_display_name']} — {prop_label}"
    layout = make_layout(settings, title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.update_layout(**layout)
    return fig


def _render_surface_3d(pe_data, settings):
    cmap = settings.get('surface_cmap', DEFAULTS['surface_cmap'])
    z_choice = pe_data['z_choice']
    prop_label = pe_data['prop_label']
    T_grid = pe_data['T_grid']
    P_grid = pe_data['P_grid']
    Z = pe_data['Z']

    if z_choice == 'property':
        X, Y, ZZ = T_grid, P_grid, Z
        xlabel, ylabel, zlabel = 'Temperature [K]', 'Pressure [MPa]', prop_label
    elif z_choice == 'T':
        X, Y, ZZ = P_grid, Z, T_grid
        xlabel, ylabel, zlabel = 'Pressure [MPa]', prop_label, 'Temperature [K]'
    else:
        X, Y, ZZ = T_grid, Z, P_grid
        xlabel, ylabel, zlabel = 'Temperature [K]', prop_label, 'Pressure [MPa]'

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=ZZ,
        colorscale=cmap, opacity=0.9,
        colorbar=dict(title=zlabel),
        hovertemplate=(
            f'{xlabel}=%{{x:.2f}}<br>'
            f'{ylabel}=%{{y:.4g}}<br>'
            f'{zlabel}=%{{z:.4g}}<extra></extra>'
        ),
    ))

    _add_phase_traces_3d(fig, pe_data.get('phase'), z_choice, settings)

    title = f"{pe_data['model_display_name']} — {prop_label}"
    layout = make_layout_3d(settings, title=title,
                            xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel)
    fig.update_layout(**layout)
    return fig


# =====================================================================
# Phase boundary helpers — shared by compute and render
# =====================================================================

def _add_phase_traces_2d(fig, phase_traces, settings):
    """Add pre-computed 2D phase traces to a figure using current settings."""
    if not phase_traces:
        return
    plw = settings.get('phase_line_width', DEFAULTS['phase_line_width'])
    spinodal_color = settings.get('spinodal_color', DEFAULTS['spinodal_color'])
    binodal_color = settings.get('binodal_color', DEFAULTS['binodal_color'])
    llcp_color = settings.get('llcp_color', DEFAULTS['llcp_color'])

    for pt in phase_traces:
        if pt['type'] == 'llcp':
            fig.add_trace(go.Scatter(
                x=pt['x'], y=pt['y'],
                mode='markers', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                marker=dict(color=llcp_color, size=10, symbol='circle',
                            line=dict(width=1, color='white')),
            ))
        elif pt['type'] == 'spinodal':
            fig.add_trace(go.Scatter(
                x=pt['x'], y=pt['y'],
                mode='lines', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                line=dict(color=spinodal_color, width=plw, dash='dash'),
                hovertemplate='Spinodal<br>%{x:.2f}<br>%{y:.6g}<extra></extra>',
            ))
        else:  # binodal
            fig.add_trace(go.Scatter(
                x=pt['x'], y=pt['y'],
                mode='lines', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                line=dict(color=binodal_color, width=plw),
                hovertemplate='Binodal<br>%{x:.2f}<br>%{y:.6g}<extra></extra>',
            ))


def _add_phase_traces_3d(fig, phase_traces, z_choice, settings):
    """Add pre-computed 3D phase traces to a figure using current settings.

    Phase traces store raw T/P/prop arrays; z_choice determines axis mapping.
    """
    if not phase_traces:
        return
    plw = settings.get('phase_line_width', DEFAULTS['phase_line_width'])
    spinodal_color = settings.get('spinodal_color', DEFAULTS['spinodal_color'])
    binodal_color = settings.get('binodal_color', DEFAULTS['binodal_color'])
    llcp_color = settings.get('llcp_color', DEFAULTS['llcp_color'])

    def _map3d(T, P, prop):
        if z_choice == 'property':
            return T, P, prop
        elif z_choice == 'T':
            return P, prop, T
        else:
            return T, prop, P

    for pt in phase_traces:
        x, y, z = _map3d(pt['T'], pt['P'], pt['prop'])
        if pt['type'] == 'llcp':
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                marker=dict(color=llcp_color, size=6, symbol='circle',
                            line=dict(width=1, color='white')),
            ))
        elif pt['type'] == 'spinodal':
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                line=dict(color=spinodal_color, width=plw + 2, dash='dash'),
            ))
        else:  # binodal
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines', name=pt.get('name'),
                showlegend=pt.get('show_legend', True),
                line=dict(color=binodal_color, width=plw + 2),
            ))


# =====================================================================
# Phase data computation (model-dependent, settings-independent)
# =====================================================================

def _compute_along_curve(model_key, prop_key, T_arr, P_arr):
    """Compute property values along a (T, P) curve."""
    from watereos import getProp
    PT = np.array([P_arr, T_arr], dtype=object)
    result = getProp(PT, model_key)
    Z = np.asarray(getattr(result, prop_key), dtype=float)
    return np.diag(Z)


def _extend_to_llcp(T_arr, P_arr, prop_arr, pd_data, model_key, prop_key):
    """Prepend LLCP point so curves converge at the critical point."""
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


def _compute_phase_curves(model_key, prop_key, isobar_mode):
    """Compute phase traces for curves mode. Returns list of serializable dicts."""
    try:
        pd_data = compute_phase_diagram_data(model_key)
    except Exception:
        return None

    info = MODEL_REGISTRY[model_key]
    traces = []

    # Spinodal branches
    if 'spinodal' in pd_data:
        sp = pd_data['spinodal']
        p_arr = np.asarray(sp['p_array'])
        for branch_key, x_key, show_label in [
            ('T_upper', 'x_hi_upper', True),
            ('T_lower', 'x_lo_lower', False),
        ]:
            T_branch = np.asarray(sp[branch_key])
            valid = (np.isfinite(T_branch)
                     & (T_branch >= info.T_min - 10)
                     & (T_branch <= info.T_max + 10))
            if not np.any(valid):
                continue
            T_b, P_b = T_branch[valid], p_arr[valid]

            x_sp = np.asarray(sp.get(x_key, []))
            prop_vals = None
            if x_sp.size == p_arr.size and not prop_key.endswith(('_A', '_B')):
                prop_vals = compute_property_at_forced_x(
                    model_key, prop_key, T_b, P_b, x_sp[valid])
            if prop_vals is None or np.all(np.isnan(prop_vals)):
                prop_vals = _compute_along_curve(model_key, prop_key, T_b, P_b)

            T_b, P_b, prop_vals = _extend_to_llcp(
                T_b, P_b, prop_vals, pd_data, model_key, prop_key)

            x_vals = T_b if isobar_mode else P_b
            mask = np.isfinite(prop_vals)
            traces.append({
                'x': x_vals[mask].tolist(),
                'y': prop_vals[mask].tolist(),
                'type': 'spinodal',
                'name': 'Spinodal' if show_label else None,
                'show_legend': show_label,
            })

    # Binodal dome
    if 'binodal' in pd_data:
        bn = pd_data['binodal']
        p_arr = np.asarray(bn['p_array'])
        T_bn = np.asarray(bn['T_binodal'])
        valid = (np.isfinite(T_bn)
                 & (T_bn >= info.T_min - 10)
                 & (T_bn <= info.T_max + 10))
        if np.any(valid) and not prop_key.endswith(('_A', '_B')):
            T_b, P_b = T_bn[valid], p_arr[valid]
            eps = 0.1
            prop_hi = _compute_along_curve(model_key, prop_key, T_b + eps, P_b)
            prop_lo = _compute_along_curve(model_key, prop_key, T_b - eps, P_b)
            for branch, lbl in [(prop_lo, 'Binodal'), (prop_hi, None)]:
                Te, Pe, br = _extend_to_llcp(
                    T_b, P_b, branch, pd_data, model_key, prop_key)
                x_vals = Te if isobar_mode else Pe
                mask = np.isfinite(br)
                traces.append({
                    'x': x_vals[mask].tolist(),
                    'y': br[mask].tolist(),
                    'type': 'binodal',
                    'name': lbl,
                    'show_legend': lbl is not None,
                })

    # LLCP marker
    if 'LLCP' in pd_data:
        llcp = pd_data['LLCP']
        T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])
        prop_c = _compute_along_curve(model_key, prop_key,
                                       np.array([T_c]), np.array([P_c]))
        x_c = T_c if isobar_mode else P_c
        if np.isfinite(prop_c[0]):
            traces.append({
                'x': [x_c],
                'y': [float(prop_c[0])],
                'type': 'llcp',
                'name': f'LLCP ({T_c:.1f} K, {P_c:.1f} MPa)',
                'show_legend': True,
            })

    return traces


def _serialize_phase_diagram(model_key):
    """Serialize phase diagram data (T-P) for 2D surface overlay."""
    try:
        pd_data = compute_phase_diagram_data(model_key)
    except Exception:
        return None

    result = {}
    for key in ('LLCP', 'spinodal', 'binodal'):
        if key not in pd_data or pd_data[key] is None:
            result[key] = None
            continue
        d = pd_data[key]
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            else:
                out[k] = v
        result[key] = out
    return result


def _compute_phase_3d(model_key, prop_key):
    """Compute phase traces for 3D surface mode.

    Stores raw T/P/prop arrays (axis mapping deferred to render time).
    """
    try:
        pd_data = compute_phase_diagram_data(model_key)
    except Exception:
        return None

    info = MODEL_REGISTRY[model_key]
    traces = []

    # Spinodal
    if 'spinodal' in pd_data:
        sp = pd_data['spinodal']
        p_arr = np.asarray(sp['p_array'])
        for branch_key, x_key, show_label in [
            ('T_upper', 'x_hi_upper', True), ('T_lower', 'x_lo_lower', False)
        ]:
            T_branch = np.asarray(sp[branch_key])
            valid = (np.isfinite(T_branch)
                     & (T_branch >= info.T_min - 10)
                     & (T_branch <= info.T_max + 10))
            if not np.any(valid):
                continue
            T_b, P_b = T_branch[valid], p_arr[valid]
            x_sp = np.asarray(sp.get(x_key, []))
            prop_vals = None
            if x_sp.size == p_arr.size and not prop_key.endswith(('_A', '_B')):
                prop_vals = compute_property_at_forced_x(
                    model_key, prop_key, T_b, P_b, x_sp[valid])
            if prop_vals is None or np.all(np.isnan(prop_vals)):
                prop_vals = _compute_along_curve(model_key, prop_key, T_b, P_b)
            T_b, P_b, prop_vals = _extend_to_llcp(
                T_b, P_b, prop_vals, pd_data, model_key, prop_key)
            mask = np.isfinite(prop_vals)
            traces.append({
                'T': T_b[mask].tolist(),
                'P': P_b[mask].tolist(),
                'prop': prop_vals[mask].tolist(),
                'type': 'spinodal',
                'name': 'Spinodal' if show_label else None,
                'show_legend': show_label,
            })

    # Binodal dome
    if 'binodal' in pd_data:
        bn = pd_data['binodal']
        p_arr = np.asarray(bn['p_array'])
        T_bn = np.asarray(bn['T_binodal'])
        valid = (np.isfinite(T_bn)
                 & (T_bn >= info.T_min - 10)
                 & (T_bn <= info.T_max + 10))
        if np.any(valid) and not prop_key.endswith(('_A', '_B')):
            T_b, P_b = T_bn[valid], p_arr[valid]
            eps = 0.1
            prop_hi = _compute_along_curve(model_key, prop_key, T_b + eps, P_b)
            prop_lo = _compute_along_curve(model_key, prop_key, T_b - eps, P_b)
            for branch, lbl in [(prop_lo, 'Binodal'), (prop_hi, None)]:
                Te, Pe, br = _extend_to_llcp(
                    T_b, P_b, branch, pd_data, model_key, prop_key)
                mask = np.isfinite(br)
                traces.append({
                    'T': Te[mask].tolist(),
                    'P': Pe[mask].tolist(),
                    'prop': br[mask].tolist(),
                    'type': 'binodal',
                    'name': lbl,
                    'show_legend': lbl is not None,
                })

    # LLCP
    if 'LLCP' in pd_data:
        llcp = pd_data['LLCP']
        T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])
        prop_c = _compute_along_curve(model_key, prop_key,
                                       np.array([T_c]), np.array([P_c]))
        if np.isfinite(prop_c[0]):
            traces.append({
                'T': [T_c],
                'P': [P_c],
                'prop': [float(prop_c[0])],
                'type': 'llcp',
                'name': f'LLCP ({T_c:.1f} K, {P_c:.1f} MPa)',
                'show_legend': True,
            })

    return traces
