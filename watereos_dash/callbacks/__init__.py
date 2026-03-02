from . import property_explorer, phase_diagram, model_comparison, point_calculator, settings, h2o_phase_diagram


def register_all(app):
    """Register all tab callbacks with the Dash app."""
    property_explorer.register(app)
    phase_diagram.register(app)
    model_comparison.register(app)
    point_calculator.register(app)
    settings.register(app)
    h2o_phase_diagram.register(app)
