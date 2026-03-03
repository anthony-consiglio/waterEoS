"""Layout definitions for each web app tab.

Every module exposes a ``layout()`` function that returns the Dash
component tree for that tab.  Layouts are imported here under
``<name>_layout`` aliases for use in ``app.py``.
"""

from .info import layout as info_layout
from .property_explorer import layout as property_explorer_layout
from .phase_diagram import layout as phase_diagram_layout
from .model_comparison import layout as model_comparison_layout
from .point_calculator import layout as point_calculator_layout
from .settings import layout as settings_layout
from .h2o_phase_diagram import layout as h2o_phase_diagram_layout
