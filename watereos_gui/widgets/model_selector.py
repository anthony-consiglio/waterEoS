"""
Model selector widget — checkbox or radio-button group.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox, QRadioButton

from watereos_gui.utils.model_registry import MODEL_REGISTRY, MODEL_ORDER


class ModelSelector(QGroupBox):
    """Checkbox (multi) or radio (single) group for selecting models."""

    selection_changed = pyqtSignal(list)   # emits list of selected model keys

    def __init__(self, title='Models', multi=True, filter_fn=None, parent=None):
        super().__init__(title, parent)
        self._multi = multi
        self._buttons = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        for key in MODEL_ORDER:
            info = MODEL_REGISTRY[key]
            if filter_fn and not filter_fn(info):
                continue
            if multi:
                btn = QCheckBox(info.display_name)
            else:
                btn = QRadioButton(info.display_name)
            btn.setProperty('model_key', key)
            btn.toggled.connect(self._on_toggled)
            layout.addWidget(btn)
            self._buttons[key] = btn

    def _on_toggled(self, _checked):
        self.selection_changed.emit(self.selected_models())

    def selected_models(self):
        return [k for k, btn in self._buttons.items() if btn.isChecked()]

    def set_selected(self, keys):
        for k, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(k in keys)
            btn.blockSignals(False)
        self.selection_changed.emit(self.selected_models())
