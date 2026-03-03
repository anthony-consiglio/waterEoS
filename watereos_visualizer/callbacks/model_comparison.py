"""Tab 3: Model Comparison — callbacks."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, no_update

from watereos.model_registry import (
    MODEL_REGISTRY, get_display_label, get_common_properties,
)
from watereos.computation import compute_multi_model_curves
from watereos_visualizer.style import (
    get_palette, get_model_colors, make_layout, DEFAULTS,
)


def register(app):
    @app.callback(
        [Output('mc-property', 'data'),
         Output('mc-property', 'value')],
        Input('mc-models', 'value'),
        State('mc-property', 'value'),
        prevent_initial_call=True,
    )
    def update_props(model_keys, current_prop):
        if not model_keys:
            return [], None
        common = get_common_properties(model_keys)
        opts = [{'label': get_display_label(p), 'value': p} for p in common]
        val = current_prop if current_prop in common else (common[0] if common else None)
        return opts, val

    @app.callback(
        Output('mc-graph', 'figure'),
        Input('mc-update', 'n_clicks'),
        [State('mc-models', 'value'),
         State('mc-property', 'value'),
         State('mc-tmin', 'value'),
         State('mc-tmax', 'value'),
         State('mc-pmin', 'value'),
         State('mc-pmax', 'value'),
         State('mc-ncurves', 'value'),
         State('mc-npoints', 'value'),
         State('mc-curve-type', 'value'),
         State('mc-layout', 'value'),
         Input('settings-store', 'data')],
        prevent_initial_call=True,
    )
    def update_plot(n_clicks, model_keys, prop_key, tmin, tmax, pmin, pmax,
                    n_curves, n_points, curve_type, layout_mode, settings):
        if not model_keys or len(model_keys) < 2 or not prop_key:
            return no_update

        settings = settings or DEFAULTS
        T_range = (float(tmin), float(tmax))
        P_range = (float(pmin), float(pmax))
        n_curves = int(n_curves or 5)
        n_points = int(n_points or 200)
        isobar_mode = curve_type == 'isobar'

        results = compute_multi_model_curves(
            model_keys, prop_key, T_range, P_range,
            n_curves, n_points, isobar_mode)

        if layout_mode == 'overlay':
            return _build_overlay(results, model_keys, prop_key, settings)
        else:
            return _build_sidebyside(results, model_keys, prop_key, settings)


def _build_overlay(results, model_keys, prop_key, settings):
    model_colors = get_model_colors(settings)
    lw = settings.get('line_width', DEFAULTS['line_width'])

    fig = go.Figure()
    x_label = y_label = ''

    for mk in model_keys:
        data = results.get(mk)
        if not data:
            continue
        x_label = data['x_label']
        y_label = data['y_label']
        color = model_colors.get(mk, '#888888')
        name = MODEL_REGISTRY[mk].display_name
        n = len(data['x_values'])

        for i, (x, y, label) in enumerate(
                zip(data['x_values'], data['y_values'], data['curve_labels'])):
            mask = np.isfinite(y)
            alpha = 0.4 + 0.6 * (i / max(n - 1, 1))
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                mode='lines',
                name=f'{name} — {label}',
                legendgroup=mk,
                showlegend=(i == 0),
                line=dict(color=color, width=lw),
                opacity=alpha,
                hovertemplate=f'{name} — {label}<br>%{{x:.2f}}<br>%{{y:.6g}}<extra></extra>',
            ))

    prop_label = get_display_label(prop_key)
    layout = make_layout(settings, title=f'Model Comparison — {prop_label}',
                         xaxis_title=x_label, yaxis_title=y_label)
    fig.update_layout(**layout)
    return fig


def _build_sidebyside(results, model_keys, prop_key, settings):
    n_models = len(model_keys)
    palette = get_palette(settings)
    lw = settings.get('line_width', DEFAULTS['line_width'])

    titles = [MODEL_REGISTRY[mk].display_name for mk in model_keys]
    fig = make_subplots(rows=1, cols=n_models, subplot_titles=titles,
                        shared_yaxes=True)

    x_label = y_label = ''
    for col, mk in enumerate(model_keys, 1):
        data = results.get(mk)
        if not data:
            continue
        x_label = data['x_label']
        y_label = data['y_label']

        for i, (x, y, label) in enumerate(
                zip(data['x_values'], data['y_values'], data['curve_labels'])):
            mask = np.isfinite(y)
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                mode='lines', name=label,
                legendgroup=f'{mk}_{label}',
                showlegend=(col == 1),
                line=dict(color=palette[i % len(palette)], width=lw),
                hovertemplate=f'{label}<br>%{{x:.2f}}<br>%{{y:.6g}}<extra></extra>',
            ), row=1, col=col)

    prop_label = get_display_label(prop_key)
    base = make_layout(settings, title=f'Model Comparison — {prop_label}')
    fig.update_layout(**base)

    for col in range(1, n_models + 1):
        xref = f'xaxis{col}' if col > 1 else 'xaxis'
        fig.update_layout(**{xref: dict(title=x_label)})
    fig.update_layout(yaxis=dict(title=y_label))

    return fig
