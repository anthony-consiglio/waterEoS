"""Shared utilities for waterEoS model interfaces."""

import numpy as np


class ThermodynamicStates:
    """Container for thermodynamic output, accessed via attributes."""
    pass


def _is_grid_input(PT):
    """Detect whether PT is grid format or scatter format."""
    if PT.dtype == object and len(PT) == 2:
        try:
            if hasattr(PT[0], '__len__') and not isinstance(PT[0], tuple):
                return True
        except (TypeError, IndexError):
            pass
    return False
