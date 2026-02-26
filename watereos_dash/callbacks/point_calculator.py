"""Tab 4: Point Calculator — callbacks."""

from dash import Input, Output, State, no_update

from watereos_gui.utils.model_registry import (
    MODEL_REGISTRY, PROPERTY_LABELS, PROPERTY_UNITS, get_display_label,
)
from watereos_gui.backend.computation import compute_point_properties


def register(app):
    # Pre-populate T, P from phase diagram click
    @app.callback(
        [Output('pc-temperature', 'value'),
         Output('pc-pressure', 'value')],
        Input('clicked-point-store', 'data'),
        prevent_initial_call=True,
    )
    def from_phase_click(click_data):
        if not click_data:
            return no_update, no_update
        T = click_data.get('T')
        P = click_data.get('P')
        if T is None or P is None:
            return no_update, no_update
        return round(float(T), 2), round(float(P), 2)

    # Calculate
    @app.callback(
        [Output('pc-table', 'columns'),
         Output('pc-table', 'data'),
         Output('pc-status', 'children')],
        Input('pc-calculate', 'n_clicks'),
        [State('pc-temperature', 'value'),
         State('pc-pressure', 'value'),
         State('pc-models', 'value')],
        prevent_initial_call=True,
    )
    def calculate(n_clicks, T, P, model_keys):
        if not model_keys or T is None or P is None:
            return no_update, no_update, 'Provide T, P, and at least one model.'

        T_K = float(T)
        P_MPa = float(P)

        try:
            results = compute_point_properties(model_keys, T_K, P_MPa)
        except Exception as e:
            return no_update, no_update, f'Error: {e}'

        # Build columns
        columns = [
            {'name': 'Property', 'id': 'property'},
            {'name': 'Unit', 'id': 'unit'},
        ]
        for mk in model_keys:
            columns.append({
                'name': MODEL_REGISTRY[mk].display_name,
                'id': mk,
            })

        # Collect all properties across selected models
        all_props = []
        seen = set()
        for mk in model_keys:
            for p in MODEL_REGISTRY[mk].properties:
                if p not in seen:
                    all_props.append(p)
                    seen.add(p)

        # Build rows
        rows = []
        for prop in all_props:
            row = {
                'property': PROPERTY_LABELS.get(prop, prop),
                'unit': PROPERTY_UNITS.get(prop, ''),
            }
            for mk in model_keys:
                val = results.get(mk, {}).get(prop)
                if val is not None:
                    row[mk] = f'{val:.6g}'
                else:
                    row[mk] = '\u2014'
            rows.append(row)

        status = f'Computed at T = {T_K:.2f} K, P = {P_MPa:.2f} MPa'
        return columns, rows, status
