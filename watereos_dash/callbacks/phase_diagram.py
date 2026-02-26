"""Tab 2: Phase Diagram — callbacks."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from watereos_gui.utils.model_registry import MODEL_REGISTRY
from watereos_gui.backend.computation import compute_phase_diagram_data
from watereos_dash.style import make_layout, get_phase_traces, DEFAULTS


def register(app):
    # Toggle manual limits visibility
    @app.callback(
        Output('pd-manual-limits', 'style'),
        Input('pd-auto-limits', 'value'),
        prevent_initial_call=True,
    )
    def toggle_manual(auto):
        if 'auto' in (auto or []):
            return {'display': 'none'}
        return {'display': 'block'}

    # Compute phase diagram and store data
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
            # Convert numpy arrays to lists for JSON serialization
            serializable = _serialize_pd(pd_data)
            return serializable, f'Computed for {MODEL_REGISTRY[model_key].display_name}'
        except Exception as e:
            return no_update, f'Error: {e}'

    # Replot from stored data (fast — no recompute)
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

        # Deserialize arrays
        pd_native = _deserialize_pd(pd_data)

        fig = go.Figure()

        # Filter traces based on show checklist
        plw = settings.get('phase_line_width', DEFAULTS['phase_line_width'])
        spinodal_color = settings.get('spinodal_color', DEFAULTS['spinodal_color'])
        binodal_color = settings.get('binodal_color', DEFAULTS['binodal_color'])
        llcp_color = settings.get('llcp_color', DEFAULTS['llcp_color'])

        if 'spinodal' in show and 'spinodal' in pd_native and pd_native['spinodal']:
            sp = pd_native['spinodal']
            fig.add_trace(go.Scatter(
                x=sp['T_K'], y=sp['p_MPa'],
                mode='lines', name='Spinodal',
                line=dict(color=spinodal_color, width=plw, dash='dash'),
                hovertemplate='T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra>Spinodal</extra>',
            ))

        if 'binodal' in show and 'binodal' in pd_native and pd_native['binodal']:
            bn = pd_native['binodal']
            fig.add_trace(go.Scatter(
                x=bn['T_K'], y=bn['p_MPa'],
                mode='lines', name='Binodal',
                line=dict(color=binodal_color, width=plw),
                hovertemplate='T=%{x:.2f} K<br>P=%{y:.2f} MPa<extra>Binodal</extra>',
            ))

        if 'LLCP' in show and 'LLCP' in pd_native and pd_native['LLCP']:
            llcp = pd_native['LLCP']
            T_c = float(llcp['T_K'])
            P_c = float(llcp['p_MPa'])
            fig.add_trace(go.Scatter(
                x=[T_c], y=[P_c],
                mode='markers',
                name=f'LLCP ({T_c:.1f} K, {P_c:.1f} MPa)',
                marker=dict(color=llcp_color, size=10, symbol='circle',
                            line=dict(width=1, color='white')),
                hovertemplate=f'LLCP<br>T={T_c:.2f} K<br>P={P_c:.2f} MPa<extra></extra>',
            ))

        layout = make_layout(settings, title='Phase Diagram (T–P)',
                             xaxis_title='Temperature [K]',
                             yaxis_title='Pressure [MPa]')

        # Apply manual limits if auto is unchecked
        if 'auto' not in (auto_limits or []):
            try:
                layout['xaxis']['range'] = [float(tmin), float(tmax)]
                layout['yaxis']['range'] = [float(pmin), float(pmax)]
            except (TypeError, ValueError):
                pass

        fig.update_layout(**layout)
        return fig

    # Click on plot → send (T, P) to shared store for Point Calculator
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
    layout = make_layout(settings, title='Phase Diagram (T–P)',
                         xaxis_title='Temperature [K]',
                         yaxis_title='Pressure [MPa]')
    fig.update_layout(**layout)
    fig.add_annotation(
        text='Click "Compute" to generate phase diagram',
        xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color='#94a3b8'),
    )
    return fig


def _serialize_pd(pd_data):
    """Convert phase diagram data (with numpy arrays) to JSON-safe dicts."""
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


def _deserialize_pd(pd_data):
    """Convert stored JSON lists back to numpy arrays where needed."""
    result = {}
    for key in ('LLCP', 'spinodal', 'binodal'):
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
