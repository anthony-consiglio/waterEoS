"""Tab 5: Settings — callbacks."""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update

from watereos_dash.style import (
    PALETTE_OPTIONS, CMAP_OPTIONS, DEFAULTS, BG_OPTIONS, make_layout,
)


def register(app):
    # Build settings dict from controls and store it
    @app.callback(
        Output('settings-store', 'data'),
        [Input('st-palette', 'value'),
         Input('st-cmap', 'value'),
         Input('st-binodal-color', 'value'),
         Input('st-spinodal-color', 'value'),
         Input('st-llcp-color', 'value'),
         Input('st-line-width', 'value'),
         Input('st-phase-line-width', 'value'),
         Input('st-font-size', 'value'),
         Input('st-grid', 'value'),
         Input('st-background', 'value'),
         Input('st-reset', 'n_clicks')],
        prevent_initial_call=True,
    )
    def update_settings(palette, cmap, bin_color, spin_color, llcp_color,
                        line_w, phase_lw, font_sz, grid, bg, reset_clicks):
        from dash import callback_context
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'st-reset.n_clicks':
            return dict(DEFAULTS)

        return {
            'curve_palette': palette or DEFAULTS['curve_palette'],
            'surface_cmap': cmap or DEFAULTS['surface_cmap'],
            'binodal_color': bin_color or DEFAULTS['binodal_color'],
            'spinodal_color': spin_color or DEFAULTS['spinodal_color'],
            'llcp_color': llcp_color or DEFAULTS['llcp_color'],
            'line_width': float(line_w) if line_w else DEFAULTS['line_width'],
            'phase_line_width': float(phase_lw) if phase_lw else DEFAULTS['phase_line_width'],
            'font_size': int(font_sz) if font_sz else DEFAULTS['font_size'],
            'grid_enabled': 'grid' in (grid or []),
            'bg_color': bg or DEFAULTS['bg_color'],
        }

    # Reset controls to defaults
    @app.callback(
        [Output('st-palette', 'value'),
         Output('st-cmap', 'value'),
         Output('st-binodal-color', 'value'),
         Output('st-spinodal-color', 'value'),
         Output('st-llcp-color', 'value'),
         Output('st-line-width', 'value'),
         Output('st-phase-line-width', 'value'),
         Output('st-font-size', 'value'),
         Output('st-grid', 'value'),
         Output('st-background', 'value')],
        Input('st-reset', 'n_clicks'),
        prevent_initial_call=True,
    )
    def reset_controls(n):
        return (
            DEFAULTS['curve_palette'],
            DEFAULTS['surface_cmap'],
            DEFAULTS['binodal_color'],
            DEFAULTS['spinodal_color'],
            DEFAULTS['llcp_color'],
            DEFAULTS['line_width'],
            DEFAULTS['phase_line_width'],
            DEFAULTS['font_size'],
            ['grid'] if DEFAULTS['grid_enabled'] else [],
            DEFAULTS['bg_color'],
        )

    # Live preview
    @app.callback(
        Output('st-preview', 'figure'),
        Input('settings-store', 'data'),
        prevent_initial_call=False,
    )
    def update_preview(settings):
        settings = settings or DEFAULTS
        palette_name = settings.get('curve_palette', DEFAULTS['curve_palette'])
        palette = PALETTE_OPTIONS.get(palette_name, PALETTE_OPTIONS['Biostasis'])
        lw = settings.get('line_width', DEFAULTS['line_width'])
        plw = settings.get('phase_line_width', DEFAULTS['phase_line_width'])
        binodal_color = settings.get('binodal_color', DEFAULTS['binodal_color'])
        spinodal_color = settings.get('spinodal_color', DEFAULTS['spinodal_color'])
        llcp_color = settings.get('llcp_color', DEFAULTS['llcp_color'])

        fig = go.Figure()

        # Sample sine curves
        x = np.linspace(0, 4 * np.pi, 200)
        for i in range(3):
            fig.add_trace(go.Scatter(
                x=x, y=np.sin(x + i * np.pi / 3) * (1 - 0.15 * i),
                mode='lines', name=f'Curve {i+1}',
                line=dict(color=palette[i % len(palette)], width=lw),
            ))

        # Phase boundary lines
        fig.add_trace(go.Scatter(
            x=[2, 8, 11], y=[0.6, -0.4, 0.3],
            mode='lines', name='Binodal',
            line=dict(color=binodal_color, width=plw),
        ))
        fig.add_trace(go.Scatter(
            x=[3, 7, 10], y=[0.4, -0.6, 0.1],
            mode='lines', name='Spinodal',
            line=dict(color=spinodal_color, width=plw, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=[6], y=[0.0],
            mode='markers', name='LLCP',
            marker=dict(color=llcp_color, size=10, symbol='circle',
                        line=dict(width=1, color='white')),
        ))

        layout = make_layout(settings, title='Preview',
                             xaxis_title='X', yaxis_title='Y')
        fig.update_layout(**layout)
        return fig
