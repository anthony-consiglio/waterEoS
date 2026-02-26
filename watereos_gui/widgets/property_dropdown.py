"""
Property dropdown — QComboBox that adapts to selected models.
"""

from PyQt6.QtWidgets import QComboBox

from watereos_gui.utils.model_registry import (
    get_common_properties,
    get_display_label,
)


class PropertyDropdown(QComboBox):
    """Dropdown listing properties common to the currently-selected models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_keys = []

    def update_for_models(self, model_keys):
        """Rebuild the dropdown for the intersection of properties."""
        prev = self.current_property()
        self.blockSignals(True)
        self.clear()
        self._current_keys = get_common_properties(model_keys)
        for key in self._current_keys:
            self.addItem(get_display_label(key), userData=key)
        # Try to restore previous selection
        if prev and prev in self._current_keys:
            self.setCurrentIndex(self._current_keys.index(prev))
        self.blockSignals(False)

    def current_property(self):
        idx = self.currentIndex()
        if idx < 0:
            return None
        return self._current_keys[idx]
