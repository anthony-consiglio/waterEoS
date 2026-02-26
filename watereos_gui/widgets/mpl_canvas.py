"""
Matplotlib canvas widget for embedding in PyQt6.
"""

from PyQt6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from watereos_gui.utils.plot_style import apply_watereos_style, style_axes


class MplCanvas(FigureCanvasQTAgg):
    """A single matplotlib Figure + Axes embedded in Qt."""

    def __init__(self, parent=None, width=6, height=4.5):
        apply_watereos_style()
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        style_axes(self.ax)
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)

    def clear(self):
        self.ax.clear()
        style_axes(self.ax)
        self.draw()


class MplCanvasWidget(QWidget):
    """Canvas + NavigationToolbar bundled in a QWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

    @property
    def fig(self):
        return self.canvas.fig

    @property
    def ax(self):
        return self.canvas.ax

    def clear(self):
        self.canvas.clear()

    def redraw(self):
        self.canvas.draw()
