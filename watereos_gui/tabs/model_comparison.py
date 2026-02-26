"""
Tab 3: Model Comparison — overlay or side-by-side comparison of models.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QRadioButton, QSplitter,
)

from watereos_gui.widgets.model_selector import ModelSelector
from watereos_gui.widgets.property_dropdown import PropertyDropdown
from watereos_gui.widgets.pt_range_controls import PTRangeControls
from watereos_gui.widgets.mpl_canvas import MplCanvasWidget
from watereos_gui.widgets.export_toolbar import ExportToolbar


class ModelComparisonTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._plot_data = None
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

        # Model selector (multi-check)
        self.model_selector = ModelSelector(title='Models', multi=True)
        self.model_selector.set_selected(['duska2020', 'holten2014'])
        sb_layout.addWidget(self.model_selector)

        # Property dropdown
        lbl = QLabel('Property:')
        lbl.setStyleSheet('font-weight: bold;')
        sb_layout.addWidget(lbl)
        self.prop_dropdown = PropertyDropdown()
        self.prop_dropdown.update_for_models(['duska2020', 'holten2014'])
        sb_layout.addWidget(self.prop_dropdown)

        # T/P range
        self.pt_controls = PTRangeControls()
        sb_layout.addWidget(self.pt_controls)

        # Layout mode
        layout_lbl = QLabel('Layout:')
        layout_lbl.setStyleSheet('font-weight: bold;')
        sb_layout.addWidget(layout_lbl)
        mode_row = QHBoxLayout()
        self.rb_overlay = QRadioButton('Overlay')
        self.rb_sidebyside = QRadioButton('Side by Side')
        self.rb_overlay.setChecked(True)
        mode_row.addWidget(self.rb_overlay)
        mode_row.addWidget(self.rb_sidebyside)
        sb_layout.addLayout(mode_row)

        # Update button
        self.btn_update = QPushButton('Update Plot')
        self.btn_update.setStyleSheet('QPushButton { padding: 8px; font-weight: bold; }')
        sb_layout.addWidget(self.btn_update)

        sb_layout.addStretch()
        sidebar.setMaximumWidth(280)

        # --- Right panel -----------------------------------------------------
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
        self.model_selector.selection_changed.connect(self._on_models_changed)
        self.btn_update.clicked.connect(self._on_update)
        self.export_bar.btn_export_fig.clicked.connect(self._export_figure)
        self.export_bar.btn_export_data.clicked.connect(self._export_data)

    def _on_models_changed(self, models):
        self.prop_dropdown.update_for_models(models)

    def _on_update(self):
        models = self.model_selector.selected_models()
        if not models:
            return
        prop = self.prop_dropdown.current_property()
        if not prop:
            return

        self.btn_update.setEnabled(False)

        from watereos_gui.backend.workers import MultiModelPropertyWorker
        self._worker = MultiModelPropertyWorker(
            model_keys=models,
            prop_key=prop,
            T_range=self.pt_controls.get_T_range(),
            P_range=self.pt_controls.get_P_range(),
            n_curves=self.pt_controls.get_n_curves(),
            n_points=self.pt_controls.get_n_points(),
            isobar_mode=self.pt_controls.is_isobar_mode(),
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, all_data):
        self.btn_update.setEnabled(True)
        self._plot_data = all_data
        self._plot_results(all_data)

    def _on_error(self, msg):
        self.btn_update.setEnabled(True)
        ax = self.canvas_widget.ax
        ax.clear()
        ax.text(0.5, 0.5, f'Error:\n{msg[:300]}', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='red')
        self.canvas_widget.redraw()

    def _plot_results(self, all_data):
        from watereos_gui.utils.plot_style import get_model_colors, style_axes
        from watereos_gui.utils.model_registry import MODEL_REGISTRY, get_display_label
        import numpy as np

        side_by_side = self.rb_sidebyside.isChecked()
        fig = self.canvas_widget.fig
        fig.clear()

        models = list(all_data.keys())
        if not models:
            return

        model_colors = get_model_colors()

        if side_by_side and len(models) > 1:
            axes = fig.subplots(1, len(models), sharey=True)
            if len(models) == 1:
                axes = [axes]
            for idx, mk in enumerate(models):
                ax = axes[idx]
                style_axes(ax)
                data = all_data[mk]
                color = model_colors.get(mk, '#333333')
                for x, y, lbl in zip(data['x_values'], data['y_values'], data['curve_labels']):
                    mask = np.isfinite(y)
                    ax.plot(x[mask], y[mask], color=color, label=lbl, alpha=0.8)
                ax.set_xlabel(data['x_label'])
                if idx == 0:
                    ax.set_ylabel(data['y_label'])
                ax.set_title(MODEL_REGISTRY[mk].display_name)
                ax.legend(fontsize=8)
        else:
            ax = fig.add_subplot(111)
            style_axes(ax)
            first = all_data[models[0]]
            for mk in models:
                data = all_data[mk]
                color = model_colors.get(mk, '#333333')
                name = MODEL_REGISTRY[mk].display_name
                for i, (x, y, lbl) in enumerate(zip(
                        data['x_values'], data['y_values'], data['curve_labels'])):
                    mask = np.isfinite(y)
                    label = f'{name} {lbl}' if i == 0 else f'  {lbl}'
                    ax.plot(x[mask], y[mask], color=color, label=label, alpha=0.8)
            ax.set_xlabel(first['x_label'])
            ax.set_ylabel(first['y_label'])
            prop = self.prop_dropdown.current_property()
            ax.set_title(f'Model Comparison — {get_display_label(prop)}')
            ax.legend(fontsize=8)

        fig.set_tight_layout(True)
        self.canvas_widget.redraw()

    def _export_figure(self):
        from PyQt6.QtWidgets import QFileDialog
        from watereos_gui.utils.plot_style import get_export_dpi
        path, _ = QFileDialog.getSaveFileName(self, 'Save Figure', '', 'PNG (*.png)')
        if path:
            self.canvas_widget.fig.savefig(path, dpi=get_export_dpi(), bbox_inches='tight')

    def _export_data(self):
        if not self._plot_data:
            return
        from PyQt6.QtWidgets import QFileDialog
        from watereos_gui.utils.data_export import export_property_curves_csv
        path, _ = QFileDialog.getSaveFileName(self, 'Save Data', '', 'CSV (*.csv)')
        if path:
            # Multi-model: write each model's data
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                for mk, data in self._plot_data.items():
                    writer.writerow([f'Model: {mk}'])
                    for x, y, lbl in zip(data['x_values'], data['y_values'], data['curve_labels']):
                        writer.writerow([data['x_label'], data['y_label'], lbl])
                        for xi, yi in zip(x, y):
                            writer.writerow([xi, yi])
                        writer.writerow([])
