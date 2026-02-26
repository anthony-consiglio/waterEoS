"""
Export toolbar — buttons for saving figures and data.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton


class ExportToolbar(QWidget):
    """Horizontal bar with 'Export Figure' and 'Export Data' buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        self.btn_export_fig = QPushButton('Export Figure')
        self.btn_export_data = QPushButton('Export Data')

        layout.addStretch()
        layout.addWidget(self.btn_export_fig)
        layout.addWidget(self.btn_export_data)
