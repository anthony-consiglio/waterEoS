"""Tab: EoS Phase Diagram — callbacks.

Computes and renders the liquid--liquid phase diagram in the T--P plane
for two-state models.  The computation result is serialized to
``pd-store`` so that toggling individual curves or changing style
settings triggers only a re-render, not a re-computation.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from watereos.model_registry import MODEL_REGISTRY
from watereos.computation import compute_phase_diagram_data
from watereos_visualizer.style import make_layout, DEFAULTS


_CURVE_STYLES = {
    'binodal':       dict(color='#9333ea', dash='solid', width=2),
    'hdl_spinodal':  dict(color='#ec4899', dash='dash', width=2),
    'ldl_spinodal':  dict(color='#ec4899', dash='dash', width=2),
    'tmd':           dict(color='#ffffff', dash='dash', width=2),
    'widom':         dict(color='#f97316', dash='dashdot', width=2),
    'ice_ih':        dict(color='#3b82f6', dash='solid', width=2),
    'ice_iii':       dict(color='#ef4444', dash='solid', width=2),
    'nuc_ih':        dict(color='#9ca3af', dash='solid', width=1.5),
    'nuc_iii':       dict(color='#9ca3af', dash='dash', width=1.5),
    'kauzmann':      dict(color='#22c55e', dash='solid', width=2),
}

_CURVE_LABELS = {
    'binodal':       'LL binodal (LLTL)',
    'hdl_spinodal':  'HDL spinodal',
    'ldl_spinodal':  'LDL spinodal',
    'tmd':           'Density maximum (TMD)',
    'widom':         'Widom line (Cp max)',
    'ice_ih':        'Ice Ih liquidus',
    'ice_iii':       'Ice III liquidus',
    'nuc_ih':        'Ih nucleation (Holten)',
    'nuc_iii':       'III nucleation (Holten)',
    'kauzmann':      'Kauzmann temperature',
}

# Draw order: background curves first, then phase boundaries on top
_SHOW_TO_DATA = {
    'ice_ih':        'ice_ih_liquidus',
    'ice_iii':       'ice_iii_liquidus',
    'nuc_ih':        'nucleation_ih',
    'nuc_iii':       'nucleation_iii',
    'kauzmann':      'kauzmann',
    'tmd':           'tmd',
    'widom':         'widom',
    'hdl_spinodal':  'hdl_spinodal',
    'ldl_spinodal':  'ldl_spinodal',
    'binodal':       'binodal',
}


def register(app):
    @app.callback(
        Output('pd-manual-limits', 'style'),
        Input('pd-auto-limits', 'checked'),
        prevent_initial_call=True,
    )
    def toggle_manual(auto):
        """Show or hide the manual T/P range inputs based on the auto-limits checkbox."""
        if auto:
            return {'display': 'none'}
        return {'display': 'block'}

    @app.callback(
        [Output('pd-store', 'data'),
         Output('pd-status', 'children')],
        Input('pd-compute', 'n_clicks'),
        State('pd-model', 'value'),
        prevent_initial_call=True,
    )
    def compute(n_clicks, model_key):
        """Run the phase diagram computation and store the serialized result."""
        if not model_key:
            return no_update, ''
        try:
            pd_data = compute_phase_diagram_data(model_key)
            serializable = _serialize_pd(pd_data)
            serializable['_model_key'] = model_key
            return serializable, f'Computed for {MODEL_REGISTRY[model_key].display_name}'
        except Exception as e:
            return no_update, f'Error: {e}'

    @app.callback(
        Output('pd-graph', 'figure'),
        [Input('pd-store', 'data'),
         Input('pd-show', 'value'),
         Input('pd-auto-limits', 'checked'),
         Input('pd-tmin', 'value'),
         Input('pd-tmax', 'value'),
         Input('pd-pmin', 'value'),
         Input('pd-pmax', 'value'),
         Input('settings-store', 'data')],
        prevent_initial_call=True,
    )
    def replot(pd_data, show, auto_limits, tmin, tmax, pmin, pmax, settings):
        """Rebuild the figure from stored data, showing only the selected curves."""
        if not pd_data:
            return _empty_figure(settings)

        settings = settings or DEFAULTS
        show = show or []
        pd_native = _deserialize_pd(pd_data)

        model_key = pd_data.get('_model_key', '')
        model_name = MODEL_REGISTRY.get(model_key, None)
        title = (f'Water Phase Diagram ({model_name.display_name})'
                 if model_name else 'Water Phase Diagram')

        fig = go.Figure()

        for show_key, data_key in _SHOW_TO_DATA.items():
            if show_key not in show:
                continue
            d = pd_native.get(data_key)
            if d is None or 'T_K' not in d or 'p_MPa' not in d:
                continue
            T = np.asarray(d['T_K'])
            P = np.asarray(d['p_MPa'])
            if T.size == 0:
                continue

            style = _CURVE_STYLES[show_key]
            label = _CURVE_LABELS[show_key]

            fig.add_trace(go.Scatter(
                x=T, y=P,
                mode='lines',
                name=label,
                line=dict(color=style['color'], width=style['width'],
                          dash=style['dash']),
                hovertemplate=(f'{label}<br>T=%{{x:.2f}} K<br>'
                               f'P=%{{y:.2f}} MPa<extra></extra>'),
            ))

        if 'LLCP' in show:
            llcp = pd_native.get('LLCP')
            if llcp and 'T_K' in llcp and 'p_MPa' in llcp:
                T_c = float(llcp['T_K'])
                P_c = float(llcp['p_MPa'])
                fig.add_trace(go.Scatter(
                    x=[T_c], y=[P_c],
                    mode='markers',
                    name=f'LLCP ({T_c:.1f} K, {P_c:.1f} MPa)',
                    marker=dict(color='#9333ea', size=12, symbol='circle',
                                line=dict(width=1, color='white')),
                    hovertemplate=(f'LLCP<br>T={T_c:.2f} K<br>'
                                   f'P={P_c:.2f} MPa<extra></extra>'),
                ))

        tp = pd_native.get('triple_point')
        if tp and ('ice_ih' in show or 'ice_iii' in show):
            T_tp = float(tp['T_K'])
            P_tp = float(tp['p_MPa'])
            fig.add_trace(go.Scatter(
                x=[T_tp], y=[P_tp],
                mode='markers',
                name=f'Ih/III/liq triple pt ({T_tp:.1f} K, {P_tp:.0f} MPa)',
                marker=dict(color='#166534', size=12, symbol='square',
                            line=dict(width=1, color='white')),
                hovertemplate=(f'Ih/III/liq triple pt<br>T={T_tp:.2f} K<br>'
                               f'P={P_tp:.2f} MPa<extra></extra>'),
            ))

        layout = make_layout(settings, title=title,
                             xaxis_title='Temperature [K]',
                             yaxis_title='Pressure [MPa]')

        if not auto_limits:
            try:
                layout['xaxis']['range'] = [float(tmin), float(tmax)]
                layout['yaxis']['range'] = [float(pmin), float(pmax)]
            except (TypeError, ValueError):
                pass

        fig.update_layout(**layout)
        return fig

    @app.callback(
        Output('clicked-point-store', 'data'),
        Input('pd-graph', 'clickData'),
        prevent_initial_call=True,
    )
    def capture_click(click_data):
        """Forward a graph click's (T, P) to the Point Calculator via the shared store."""
        if not click_data or 'points' not in click_data:
            return no_update
        pt = click_data['points'][0]
        return {'T': pt.get('x'), 'P': pt.get('y')}


def _empty_figure(settings=None):
    """Return a placeholder figure with a 'Click Compute' annotation."""
    settings = settings or DEFAULTS
    fig = go.Figure()
    layout = make_layout(settings, title='EoS Phase Diagram',
                         xaxis_title='Temperature [K]',
                         yaxis_title='Pressure [MPa]')
    fig.update_layout(**layout)
    fig.add_annotation(
        text='Click "Compute" to generate phase diagram',
        xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color='#909296'),
    )
    return fig


_ALL_KEYS = [
    'LLCP', 'spinodal', 'hdl_spinodal', 'ldl_spinodal', 'binodal',
    'tmd', 'widom', 'ice_ih_liquidus', 'ice_iii_liquidus',
    'nucleation_ih', 'nucleation_iii', 'kauzmann',
    'triple_point',
]


def _serialize_pd(pd_data):
    """Convert phase diagram arrays to JSON-serializable lists for ``dcc.Store``."""
    result = {}
    for key in _ALL_KEYS:
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


def _deserialize_pd(pd_data):
    """Restore numpy arrays from the JSON-serialized phase diagram dict."""
    result = {}
    for key in _ALL_KEYS:
        if key not in pd_data or pd_data[key] is None:
            result[key] = None
            continue
        d = pd_data[key]
        out = {}
        for k, v in d.items():
            if isinstance(v, list):
                out[k] = np.array(v)
            else:
                out[k] = v
        result[key] = out
    return result
