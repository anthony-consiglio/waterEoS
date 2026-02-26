"""
Tab 2: Liquid-Liquid Phase Diagram Viewer — T-P diagram with toggleable overlays.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QCheckBox, QGroupBox, QProgressBar, QSplitter,
    QDoubleSpinBox, QFormLayout,
)

from watereos_gui.widgets.mpl_canvas import MplCanvasWidget
from watereos_gui.widgets.export_toolbar import ExportToolbar
from watereos_gui.utils.model_registry import MODEL_REGISTRY, models_with_phase_diagram


class PhaseDiagramViewerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._cached_data = {}     # model_key -> phase diagram dict
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

        # Model dropdown (two-state only)
        lbl = QLabel('Model:')
        lbl.setStyleSheet('font-weight: bold;')
        sb_layout.addWidget(lbl)
        self.model_combo = QComboBox()
        for key in models_with_phase_diagram():
            self.model_combo.addItem(MODEL_REGISTRY[key].display_name, userData=key)
        sb_layout.addWidget(self.model_combo)

        # Overlay checkboxes
        overlay_group = QGroupBox('Overlays')
        overlay_layout = QVBoxLayout(overlay_group)

        self.chk_binodal = QCheckBox('Binodal')
        self.chk_binodal.setChecked(True)
        self.chk_spinodal = QCheckBox('Spinodal')
        self.chk_spinodal.setChecked(True)
        self.chk_llcp = QCheckBox('LLCP')
        self.chk_llcp.setChecked(True)

        overlay_layout.addWidget(self.chk_binodal)
        overlay_layout.addWidget(self.chk_spinodal)
        overlay_layout.addWidget(self.chk_llcp)

        sb_layout.addWidget(overlay_group)

        # Axis limits
        axes_group = QGroupBox('Axis Limits')
        axes_form = QFormLayout(axes_group)

        self.T_min = QDoubleSpinBox()
        self.T_min.setRange(100, 700)
        self.T_min.setDecimals(1)
        self.T_min.setSuffix(' K')
        self.T_min.setValue(200)
        axes_form.addRow('T min:', self.T_min)

        self.T_max = QDoubleSpinBox()
        self.T_max.setRange(100, 700)
        self.T_max.setDecimals(1)
        self.T_max.setSuffix(' K')
        self.T_max.setValue(240)
        axes_form.addRow('T max:', self.T_max)

        self.P_min = QDoubleSpinBox()
        self.P_min.setRange(-200, 3000)
        self.P_min.setDecimals(1)
        self.P_min.setSuffix(' MPa')
        self.P_min.setValue(-50)
        axes_form.addRow('P min:', self.P_min)

        self.P_max = QDoubleSpinBox()
        self.P_max.setRange(-200, 3000)
        self.P_max.setDecimals(1)
        self.P_max.setSuffix(' MPa')
        self.P_max.setValue(250)
        axes_form.addRow('P max:', self.P_max)

        self.chk_auto_limits = QCheckBox('Auto limits')
        self.chk_auto_limits.setChecked(True)
        self.chk_auto_limits.toggled.connect(self._on_auto_limits_toggled)
        axes_form.addRow(self.chk_auto_limits)

        sb_layout.addWidget(axes_group)

        # Disable manual spinboxes initially (auto is on)
        self._set_limit_spinboxes_enabled(False)

        # Update button
        self.btn_update = QPushButton('Update Diagram')
        self.btn_update.setStyleSheet('QPushButton { padding: 8px; font-weight: bold; }')
        sb_layout.addWidget(self.btn_update)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        sb_layout.addWidget(self.progress)

        # Status
        self.lbl_status = QLabel('')
        self.lbl_status.setStyleSheet('color: #666; font-size: 11px;')
        sb_layout.addWidget(self.lbl_status)

        sb_layout.addStretch()
        sidebar.setMaximumWidth(280)

        # --- Right panel: canvas + export ------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_widget = MplCanvasWidget()
        right_layout.addWidget(self.canvas_widget, stretch=1)

        self.export_bar = ExportToolbar()
        right_layout.addWidget(self.export_bar)

        splitter.addWidget(sidebar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- Connections -----------------------------------------------------
        self.btn_update.clicked.connect(self._on_update)
        # Checkbox toggles re-plot from cache (no recompute)
        self.chk_binodal.toggled.connect(self._replot_from_cache)
        self.chk_spinodal.toggled.connect(self._replot_from_cache)
        self.chk_llcp.toggled.connect(self._replot_from_cache)
        self.export_bar.btn_export_fig.clicked.connect(self._export_figure)

    def _set_limit_spinboxes_enabled(self, enabled):
        for sb in (self.T_min, self.T_max, self.P_min, self.P_max):
            sb.setEnabled(enabled)

    def _on_auto_limits_toggled(self, checked):
        self._set_limit_spinboxes_enabled(not checked)
        self._replot_from_cache()

    def _export_figure(self):
        from PyQt6.QtWidgets import QFileDialog
        from watereos_gui.utils.plot_style import get_export_dpi
        path, _ = QFileDialog.getSaveFileName(self, 'Save Figure', '', 'PNG (*.png)')
        if path:
            self.canvas_widget.fig.savefig(path, dpi=get_export_dpi(), bbox_inches='tight')

    def _current_model_key(self):
        return self.model_combo.currentData()

    def _on_update(self):
        key = self._current_model_key()
        if not key:
            return
        self.btn_update.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate
        self.lbl_status.setText('Computing phase diagram...')

        from watereos_gui.backend.workers import PhaseDiagramWorker
        self._worker = PhaseDiagramWorker(key)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, data):
        self.btn_update.setEnabled(True)
        self.progress.setVisible(False)
        key = self._current_model_key()
        self._cached_data[key] = data
        self.lbl_status.setText('Done.')
        self._plot_phase_diagram(data)

    def _on_error(self, msg):
        self.btn_update.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(f'Error: {msg[:200]}')

    def _replot_from_cache(self):
        key = self._current_model_key()
        if key in self._cached_data:
            self._plot_phase_diagram(self._cached_data[key])

    def _plot_phase_diagram(self, data):
        from watereos_gui.utils.plot_style import get_phase_styles, style_axes

        ax = self.canvas_widget.ax
        ax.clear()
        style_axes(ax)

        model_key = self._current_model_key()
        model_name = MODEL_REGISTRY[model_key].display_name

        phase_styles = get_phase_styles()

        if self.chk_binodal.isChecked() and 'binodal' in data:
            bd = data['binodal']
            s = phase_styles['binodal']
            ax.plot(bd['T_K'], bd['p_MPa'],
                    color=s['color'], linestyle=s['linestyle'],
                    linewidth=s['linewidth'], label=s['label'])

        if self.chk_spinodal.isChecked() and 'spinodal' in data:
            sp = data['spinodal']
            s = phase_styles['spinodal']
            ax.plot(sp['T_K'], sp['p_MPa'],
                    color=s['color'], linestyle=s['linestyle'],
                    linewidth=s['linewidth'], label=s['label'])

        if self.chk_llcp.isChecked() and 'LLCP' in data:
            llcp = data['LLCP']
            s = phase_styles['LLCP']
            ax.plot(llcp['T_K'], llcp['p_MPa'],
                    color=s['color'], marker=s['marker'],
                    markersize=s['markersize'], linestyle=s['linestyle'],
                    label=f"LLCP ({llcp['T_K']:.1f} K, {llcp['p_MPa']:.1f} MPa)")

        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [MPa]')
        ax.set_title(f'{model_name} — Liquid-Liquid Phase Diagram')
        ax.legend()

        # Apply axis limits
        if not self.chk_auto_limits.isChecked():
            ax.set_xlim(self.T_min.value(), self.T_max.value())
            ax.set_ylim(self.P_min.value(), self.P_max.value())

        self.canvas_widget.redraw()
