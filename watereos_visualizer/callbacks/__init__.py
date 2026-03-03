"""Callback registration for all web app tabs.

Each tab module exposes a ``register(app)`` function that attaches its
Dash callbacks to the application instance.  ``register_all()`` calls
them all in the correct order.
"""

from . import property_explorer, phase_diagram, model_comparison, point_calculator, settings, h2o_phase_diagram


def register_all(app):
    """Register every tab's callbacks with the Dash *app*."""
    property_explorer.register(app)
    phase_diagram.register(app)
    model_comparison.register(app)
    point_calculator.register(app)
    settings.register(app)
    h2o_phase_diagram.register(app)
