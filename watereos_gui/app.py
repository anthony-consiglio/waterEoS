"""
Main application window — 5-tab interface for waterEoS GUI.
"""

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import QSize

from watereos_gui.tabs.property_explorer import PropertyExplorerTab
from watereos_gui.tabs.phase_diagram_viewer import PhaseDiagramViewerTab
from watereos_gui.tabs.model_comparison import ModelComparisonTab
from watereos_gui.tabs.point_calculator import PointCalculatorTab
from watereos_gui.tabs.settings import SettingsTab
from watereos_gui.utils import plot_style


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('waterEoS — Water Equation of State Explorer')
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1280, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.property_explorer = PropertyExplorerTab()
        self.phase_diagram = PhaseDiagramViewerTab()
        self.model_comparison = ModelComparisonTab()
        self.point_calculator = PointCalculatorTab()
        self.settings = SettingsTab()

        self.tabs.addTab(self.property_explorer, 'Property Explorer')
        self.tabs.addTab(self.phase_diagram, 'Liquid-Liquid Phase Diagram')
        self.tabs.addTab(self.model_comparison, 'Model Comparison')
        self.tabs.addTab(self.point_calculator, 'Point Calculator')
        self.tabs.addTab(self.settings, 'Settings')

        # Cross-tab: right-click on phase diagram canvas → send (T, P) to Point Calculator
        self.phase_diagram.canvas_widget.canvas.mpl_connect(
            'button_press_event', self._on_phase_diagram_click,
        )

    def _on_phase_diagram_click(self, event):
        """Right-click on phase diagram sends point to Point Calculator tab."""
        if event.button == 3 and event.inaxes:  # right-click
            T_K = event.xdata
            P_MPa = event.ydata
            self.point_calculator.set_point(T_K, P_MPa)
            self.tabs.setCurrentWidget(self.point_calculator)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(plot_style.get_app_stylesheet())
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
