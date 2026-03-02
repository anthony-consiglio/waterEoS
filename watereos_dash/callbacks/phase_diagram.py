"""Tab: EoS Phase Diagram — callbacks."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from watereos_gui.utils.model_registry import MODEL_REGISTRY
from watereos_gui.backend.computation import compute_phase_diagram_data
from watereos_dash.style import make_layout, DEFAULTS


# Curve styling matching Caupin & Anisimov 2019 reference figure
_CURVE_STYLES = {
    'binodal':       dict(color='#9333ea', dash='solid', width=2),    # purple
    'hdl_spinodal':  dict(color='#ec4899', dash='dash', width=2),     # magenta
    'ldl_spinodal':  dict(color='#ec4899', dash='dash', width=2),     # magenta
    'tmd':           dict(color='#ffffff', dash='dash', width=2),     # white
    'widom':         dict(color='#f97316', dash='dashdot', width=2),  # orange
    'ice_ih':        dict(color='#3b82f6', dash='solid', width=2),    # blue
    'ice_iii':       dict(color='#ef4444', dash='solid', width=2),    # red
    'nuc_ih':        dict(color='#9ca3af', dash='solid', width=1.5),  # gray
    'nuc_iii':       dict(color='#9ca3af', dash='dash', width=1.5),   # gray dashed
    'kauzmann_hdl':  dict(color='#22c55e', dash='solid', width=2),    # green
    'kauzmann_ldl':  dict(color='#16a34a', dash='dash', width=2),     # dark green dashed
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
    'kauzmann_hdl':  'HDL Kauzmann temperature',
    'kauzmann_ldl':  'LDL Kauzmann temperature',
}

# Map show-checklist value -> data key in pd_data
_SHOW_TO_DATA = {
    'binodal':       'binodal',
    'hdl_spinodal':  'hdl_spinodal',
    'ldl_spinodal':  'ldl_spinodal',
    'tmd':           'tmd',
    'widom':         'widom',
    'ice_ih':        'ice_ih_liquidus',
    'ice_iii':       'ice_iii_liquidus',
    'nuc_ih':        'nucleation_ih',
    'nuc_iii':       'nucleation_iii',
    'kauzmann_hdl':  'kauzmann_hdl',
    'kauzmann_ldl':  'kauzmann_ldl',
}


def register(app):
    @app.callback(
        Output('pd-manual-limits', 'style'),
        Input('pd-auto-limits', 'value'),
        prevent_initial_call=True,
    )
    def toggle_manual(auto):
        if 'auto' in (auto or []):
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
         Input('pd-auto-limits', 'value'),
         Input('pd-tmin', 'value'),
         Input('pd-tmax', 'value'),
         Input('pd-pmin', 'value'),
         Input('pd-pmax', 'value')],
        State('settings-store', 'data'),
        prevent_initial_call=True,
    )
    def replot(pd_data, show, auto_limits, tmin, tmax, pmin, pmax, settings):
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

        # --- Line curves ---
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

        # --- LLCP marker ---
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

        # --- Triple point marker ---
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

        if 'auto' not in (auto_limits or []):
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
        if not click_data or 'points' not in click_data:
            return no_update
        pt = click_data['points'][0]
        return {'T': pt.get('x'), 'P': pt.get('y')}


def _empty_figure(settings=None):
    settings = settings or DEFAULTS
    fig = go.Figure()
    layout = make_layout(settings, title='EoS Phase Diagram',
                         xaxis_title='Temperature [K]',
                         yaxis_title='Pressure [MPa]')
    fig.update_layout(**layout)
    fig.add_annotation(
        text='Click "Compute" to generate phase diagram',
        xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color='#94a3b8'),
    )
    return fig


_ALL_KEYS = [
    'LLCP', 'spinodal', 'hdl_spinodal', 'ldl_spinodal', 'binodal',
    'tmd', 'widom', 'ice_ih_liquidus', 'ice_iii_liquidus',
    'nucleation_ih', 'nucleation_iii', 'kauzmann_hdl', 'kauzmann_ldl',
    'triple_point',
]


def _serialize_pd(pd_data):
    """Convert phase diagram data (with numpy arrays) to JSON-safe dicts."""
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
    """Convert stored JSON lists back to numpy arrays where needed."""
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
