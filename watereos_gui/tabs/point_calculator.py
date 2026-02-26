"""
Tab 4: Point Calculator — compute all properties at a single (T, P) for selected models.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QSplitter,
)

from watereos_gui.widgets.model_selector import ModelSelector
from watereos_gui.utils.model_registry import (
    MODEL_REGISTRY, PROPERTY_LABELS, PROPERTY_UNITS, get_common_properties,
)


class PointCalculatorTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._last_results = {}
        self._init_ui()

    def _init_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # --- Left sidebar ---------------------------------------------------
        sidebar = QWidget()
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 4, 4, 4)

        # T,P input
        tp_group = QGroupBox('Conditions')
        tp_form = QFormLayout(tp_group)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(100, 700)
        self.spin_T.setDecimals(2)
        self.spin_T.setValue(273.15)
        self.spin_T.setSuffix(' K')
        tp_form.addRow('Temperature:', self.spin_T)

        self.spin_P = QDoubleSpinBox()
        self.spin_P.setRange(-200, 3000)
        self.spin_P.setDecimals(2)
        self.spin_P.setValue(0.1)
        self.spin_P.setSuffix(' MPa')
        tp_form.addRow('Pressure:', self.spin_P)

        sb_layout.addWidget(tp_group)

        # Model selector (multi-check)
        self.model_selector = ModelSelector(title='Models', multi=True)
        self.model_selector.set_selected(['duska2020'])
        sb_layout.addWidget(self.model_selector)

        # Calculate button
        self.btn_calc = QPushButton('Calculate')
        self.btn_calc.setStyleSheet('QPushButton { padding: 8px; font-weight: bold; }')
        sb_layout.addWidget(self.btn_calc)

        sb_layout.addStretch()
        sidebar.setMaximumWidth(280)

        # --- Right panel: results table --------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_header = QLabel('Select models and click Calculate.')
        self.lbl_header.setStyleSheet('font-size: 13px; color: #555;')
        right_layout.addWidget(self.lbl_header)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.table)

        splitter.addWidget(sidebar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- Connections (backend wired in Phase 3) --------------------------
        self.btn_calc.clicked.connect(self._on_calculate)

    # -- public API -----------------------------------------------------------

    def set_point(self, T_K, P_MPa):
        """Called externally (e.g. from phase diagram right-click)."""
        self.spin_T.setValue(T_K)
        self.spin_P.setValue(P_MPa)

    # -- slots (computation wired in Phase 3) ---------------------------------

    def _on_calculate(self):
        models = self.model_selector.selected_models()
        if not models:
            self.lbl_header.setText('Please select at least one model.')
            return
        T = self.spin_T.value()
        P = self.spin_P.value()
        self.btn_calc.setEnabled(False)
        self.lbl_header.setText('Computing...')

        from watereos_gui.backend.workers import PointCalcWorker
        self._worker = PointCalcWorker(models, T, P)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, results_dict):
        self.btn_calc.setEnabled(True)
        self._display_results(results_dict)

    def _on_error(self, msg):
        self.btn_calc.setEnabled(True)
        self.lbl_header.setText(f'Error: {msg[:200]}')

    def _display_results(self, results_dict):
        """Populate the table from {model_key: {prop: value}}."""
        if not results_dict:
            return
        self._last_results = results_dict
        models = list(results_dict.keys())
        props = get_common_properties(models)
        if not props:
            return

        self.table.setRowCount(len(props))
        self.table.setColumnCount(len(models) + 2)

        headers = ['Property', 'Unit'] + [MODEL_REGISTRY[m].display_name for m in models]
        self.table.setHorizontalHeaderLabels(headers)

        for row, prop_key in enumerate(props):
            self.table.setItem(row, 0, QTableWidgetItem(PROPERTY_LABELS.get(prop_key, prop_key)))
            self.table.setItem(row, 1, QTableWidgetItem(PROPERTY_UNITS.get(prop_key, '')))
            for col, model_key in enumerate(models):
                val = results_dict[model_key].get(prop_key)
                text = f'{val:.6g}' if val is not None else 'N/A'
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col + 2, item)

        T = self.spin_T.value()
        P = self.spin_P.value()
        self.lbl_header.setText(f'Properties at T = {T:.2f} K, P = {P:.2f} MPa')
