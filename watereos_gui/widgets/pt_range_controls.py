"""
Pressure-Temperature range controls — spinboxes for T/P min/max, curve count,
and isobar/isotherm mode toggle.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QRadioButton, QHBoxLayout, QWidget,
)


class PTRangeControls(QGroupBox):
    """Controls for temperature range, pressure range, and curve mode."""

    range_changed = pyqtSignal()

    def __init__(self, title='T / P Range', parent=None):
        super().__init__(title, parent)
        layout = QFormLayout(self)

        # Mode: isobars vs isotherms
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.rb_isobars = QRadioButton('Isobars')
        self.rb_isotherms = QRadioButton('Isotherms')
        self.rb_isobars.setChecked(True)
        mode_layout.addWidget(self.rb_isobars)
        mode_layout.addWidget(self.rb_isotherms)
        layout.addRow('Mode:', mode_widget)

        # Temperature range
        self.T_min = QDoubleSpinBox()
        self.T_min.setRange(100, 700)
        self.T_min.setDecimals(1)
        self.T_min.setValue(220)
        self.T_min.setSuffix(' K')
        layout.addRow('T min:', self.T_min)

        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(100, 700)
        self.T_max.setDecimals(1)
        self.T_max.setValue(320)
        self.T_max.setSuffix(' K')
        layout.addRow('T max:', self.T_max)

        # Pressure range
        self.P_min = QDoubleSpinBox()
        self.P_min.setRange(-200, 3000)
        self.P_min.setDecimals(1)
        self.P_min.setValue(0.1)
        self.P_min.setSuffix(' MPa')
        layout.addRow('P min:', self.P_min)

        self.P_max = QDoubleSpinBox()
        self.P_max.setRange(-200, 3000)
        self.P_max.setDecimals(1)
        self.P_max.setValue(200)
        self.P_max.setSuffix(' MPa')
        layout.addRow('P max:', self.P_max)

        # Number of curves
        self.n_curves = QSpinBox()
        self.n_curves.setRange(1, 50)
        self.n_curves.setValue(5)
        layout.addRow('Curves:', self.n_curves)

        # Number of points per curve
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 1000)
        self.n_points.setValue(200)
        layout.addRow('Points:', self.n_points)

    def is_isobar_mode(self):
        return self.rb_isobars.isChecked()

    def get_T_range(self):
        return self.T_min.value(), self.T_max.value()

    def get_P_range(self):
        return self.P_min.value(), self.P_max.value()

    def get_n_curves(self):
        return self.n_curves.value()

    def get_n_points(self):
        return self.n_points.value()

    def set_ranges_from_model(self, model_info):
        """Populate spinbox defaults from a ModelInfo."""
        self.T_min.setValue(model_info.T_min)
        self.T_max.setValue(model_info.T_max)
        self.P_min.setValue(model_info.P_min)
        self.P_max.setValue(model_info.P_max)
