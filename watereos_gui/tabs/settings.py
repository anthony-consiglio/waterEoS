"""
Tab 5: Settings — customize visual styles for all plots.

Changes are written to ``plot_style._current`` and take effect on the next
"Update Plot" click in any tab.  The right-hand preview redraws immediately.
"""

import numpy as np
import matplotlib.cm as _cm

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QRadioButton, QGroupBox, QDoubleSpinBox,
    QSpinBox, QSplitter, QColorDialog, QButtonGroup,
)

from watereos_gui.widgets.mpl_canvas import MplCanvasWidget
from watereos_gui.utils import plot_style


def _cmap_icon(cmap_name, width=120, height=16):
    """Render a colormap as a QIcon gradient swatch."""
    cmap = _cm.get_cmap(cmap_name)
    arr = np.linspace(0, 1, width)
    rgba = (cmap(arr)[:, :3] * 255).astype(np.uint8)  # (width, 3)
    # Tile vertically
    row = rgba.tobytes()
    img = QImage(rgba.data, width, 1, width * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(img).scaled(
        QSize(width, height), Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QIcon(pix)


class _ColorButton(QPushButton):
    """Small push-button that shows a color swatch and opens a QColorDialog."""

    def __init__(self, initial_color, parent=None):
        super().__init__(parent)
        self._color = initial_color
        self._apply_style()
        self.setFixedSize(50, 24)
        self.clicked.connect(self._pick)

    def _apply_style(self):
        self.setStyleSheet(
            f'QPushButton {{ background-color: {self._color}; border: 1px solid #999; }}'
        )

    def color(self):
        return self._color

    def set_color(self, c):
        self._color = c
        self._apply_style()

    def _pick(self):
        from PyQt6.QtGui import QColor
        c = QColorDialog.getColor(QColor(self._color), self, 'Choose Color')
        if c.isValid():
            self._color = c.name()
            self._apply_style()
            # Trigger the parent tab's _on_any_change via objectName
            p = self.parent()
            while p and not isinstance(p, SettingsTab):
                p = p.parent()
            if p:
                p._on_any_change()


class SettingsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._sync_from_style()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------

    def _init_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # --- Left sidebar: controls ----------------------------------------
        sidebar = QWidget()
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(4, 4, 4, 4)

        # -- App Theme group ------------------------------------------------
        theme_group = QGroupBox('App Theme')
        thg = QHBoxLayout(theme_group)
        self.rb_theme_dark = QRadioButton('Dark')
        self.rb_theme_light = QRadioButton('Light')
        self._theme_group = QButtonGroup(self)
        self._theme_group.addButton(self.rb_theme_dark)
        self._theme_group.addButton(self.rb_theme_light)
        thg.addWidget(self.rb_theme_dark)
        thg.addWidget(self.rb_theme_light)
        sb.addWidget(theme_group)

        # -- Colors group ---------------------------------------------------
        color_group = QGroupBox('Colors')
        cg = QVBoxLayout(color_group)

        # Curve palette
        row = QHBoxLayout()
        row.addWidget(QLabel('Curve palette:'))
        self.combo_palette = QComboBox()
        for name in plot_style.PALETTE_OPTIONS:
            self.combo_palette.addItem(name)
        row.addWidget(self.combo_palette)
        cg.addLayout(row)

        # Surface colormap (with gradient preview icons)
        row = QHBoxLayout()
        row.addWidget(QLabel('Surface cmap:'))
        self.combo_cmap = QComboBox()
        self.combo_cmap.setIconSize(QSize(120, 16))
        for name in plot_style.CMAP_OPTIONS:
            self.combo_cmap.addItem(_cmap_icon(name), name)
        row.addWidget(self.combo_cmap)
        cg.addLayout(row)

        # Phase boundary colors
        row = QHBoxLayout()
        row.addWidget(QLabel('Binodal:'))
        self.btn_binodal = _ColorButton(plot_style._DEFAULTS['binodal_color'])
        row.addWidget(self.btn_binodal)
        row.addWidget(QLabel('Spinodal:'))
        self.btn_spinodal = _ColorButton(plot_style._DEFAULTS['spinodal_color'])
        row.addWidget(self.btn_spinodal)
        row.addWidget(QLabel('LLCP:'))
        self.btn_llcp = _ColorButton(plot_style._DEFAULTS['llcp_color'])
        row.addWidget(self.btn_llcp)
        cg.addLayout(row)

        sb.addWidget(color_group)

        # -- Lines group ----------------------------------------------------
        line_group = QGroupBox('Lines')
        lg = QVBoxLayout(line_group)

        row = QHBoxLayout()
        row.addWidget(QLabel('Line width:'))
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.5, 5.0)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.setDecimals(1)
        row.addWidget(self.spin_lw)
        lg.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Phase boundary width:'))
        self.spin_phase_lw = QDoubleSpinBox()
        self.spin_phase_lw.setRange(0.5, 4.0)
        self.spin_phase_lw.setSingleStep(0.5)
        self.spin_phase_lw.setDecimals(1)
        row.addWidget(self.spin_phase_lw)
        lg.addLayout(row)

        sb.addWidget(line_group)

        # -- Text group -----------------------------------------------------
        text_group = QGroupBox('Text')
        tg = QVBoxLayout(text_group)

        row = QHBoxLayout()
        row.addWidget(QLabel('Font size:'))
        self.spin_font = QSpinBox()
        self.spin_font.setRange(8, 20)
        row.addWidget(self.spin_font)
        tg.addLayout(row)

        sb.addWidget(text_group)

        # -- Export group ---------------------------------------------------
        export_group = QGroupBox('Export')
        eg = QVBoxLayout(export_group)

        row = QHBoxLayout()
        row.addWidget(QLabel('DPI:'))
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setSingleStep(50)
        row.addWidget(self.spin_dpi)
        eg.addLayout(row)

        sb.addWidget(export_group)

        # -- Axes group -----------------------------------------------------
        axes_group = QGroupBox('Axes')
        ag = QVBoxLayout(axes_group)

        self.chk_grid = QCheckBox('Grid')
        ag.addWidget(self.chk_grid)

        self.chk_box = QCheckBox('Box (show all spines)')
        ag.addWidget(self.chk_box)

        row = QHBoxLayout()
        row.addWidget(QLabel('Axes line width:'))
        self.spin_axes_lw = QDoubleSpinBox()
        self.spin_axes_lw.setRange(0.2, 3.0)
        self.spin_axes_lw.setSingleStep(0.2)
        self.spin_axes_lw.setDecimals(1)
        row.addWidget(self.spin_axes_lw)
        ag.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Background:'))
        self.rb_dark = QRadioButton('Dark')
        self.rb_light_gray = QRadioButton('Light gray')
        self.rb_white = QRadioButton('White')
        self._bg_group = QButtonGroup(self)
        self._bg_group.addButton(self.rb_dark)
        self._bg_group.addButton(self.rb_light_gray)
        self._bg_group.addButton(self.rb_white)
        row.addWidget(self.rb_dark)
        row.addWidget(self.rb_light_gray)
        row.addWidget(self.rb_white)
        ag.addLayout(row)

        sb.addWidget(axes_group)

        # -- Reset button ---------------------------------------------------
        self.btn_reset = QPushButton('Reset to Defaults')
        self.btn_reset.setStyleSheet('QPushButton { padding: 6px; }')
        sb.addWidget(self.btn_reset)

        sb.addStretch()
        sidebar.setMaximumWidth(320)

        # --- Right panel: preview canvas -----------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel('Preview')
        lbl.setStyleSheet('font-weight: bold; font-size: 13px; padding: 4px;')
        right_layout.addWidget(lbl)

        self.canvas_widget = MplCanvasWidget()
        right_layout.addWidget(self.canvas_widget, stretch=1)

        splitter.addWidget(sidebar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- Connections ---------------------------------------------------
        self.rb_theme_dark.toggled.connect(self._on_theme_change)
        self.rb_theme_light.toggled.connect(self._on_theme_change)
        self.combo_palette.currentIndexChanged.connect(self._on_any_change)
        self.combo_cmap.currentIndexChanged.connect(self._on_any_change)
        self.spin_lw.valueChanged.connect(self._on_any_change)
        self.spin_phase_lw.valueChanged.connect(self._on_any_change)
        self.spin_font.valueChanged.connect(self._on_any_change)
        self.spin_dpi.valueChanged.connect(self._on_any_change)
        self.chk_grid.toggled.connect(self._on_any_change)
        self.chk_box.toggled.connect(self._on_any_change)
        self.spin_axes_lw.valueChanged.connect(self._on_any_change)
        self.rb_dark.toggled.connect(self._on_any_change)
        self.rb_light_gray.toggled.connect(self._on_any_change)
        self.rb_white.toggled.connect(self._on_any_change)
        self.btn_reset.clicked.connect(self._on_reset)

    # -----------------------------------------------------------------
    # Sync widgets ↔ plot_style
    # -----------------------------------------------------------------

    def _sync_from_style(self):
        """Load current plot_style values into widgets (no signals)."""
        self.rb_theme_dark.blockSignals(True)
        self.rb_theme_light.blockSignals(True)
        if plot_style.get_app_theme() == 'light':
            self.rb_theme_light.setChecked(True)
        else:
            self.rb_theme_dark.setChecked(True)
        self.rb_theme_dark.blockSignals(False)
        self.rb_theme_light.blockSignals(False)

        self.combo_palette.blockSignals(True)
        idx = self.combo_palette.findText(plot_style.get_curve_palette_name())
        if idx >= 0:
            self.combo_palette.setCurrentIndex(idx)
        self.combo_palette.blockSignals(False)

        self.combo_cmap.blockSignals(True)
        idx = self.combo_cmap.findText(plot_style.get_surface_cmap())
        if idx >= 0:
            self.combo_cmap.setCurrentIndex(idx)
        self.combo_cmap.blockSignals(False)

        self.btn_binodal.set_color(plot_style._current['binodal_color'])
        self.btn_spinodal.set_color(plot_style._current['spinodal_color'])
        self.btn_llcp.set_color(plot_style._current['llcp_color'])

        self.spin_lw.blockSignals(True)
        self.spin_lw.setValue(plot_style.get_line_width())
        self.spin_lw.blockSignals(False)

        self.spin_phase_lw.blockSignals(True)
        self.spin_phase_lw.setValue(plot_style.get_phase_line_width())
        self.spin_phase_lw.blockSignals(False)

        self.spin_font.blockSignals(True)
        self.spin_font.setValue(plot_style.get_font_size())
        self.spin_font.blockSignals(False)

        self.spin_dpi.blockSignals(True)
        self.spin_dpi.setValue(plot_style.get_export_dpi())
        self.spin_dpi.blockSignals(False)

        self.chk_grid.blockSignals(True)
        self.chk_grid.setChecked(plot_style.get_grid_enabled())
        self.chk_grid.blockSignals(False)

        self.chk_box.blockSignals(True)
        self.chk_box.setChecked(plot_style.get_box_enabled())
        self.chk_box.blockSignals(False)

        self.spin_axes_lw.blockSignals(True)
        self.spin_axes_lw.setValue(plot_style.get_axes_linewidth())
        self.spin_axes_lw.blockSignals(False)

        bg = plot_style.get_bg_color()
        self.rb_dark.blockSignals(True)
        self.rb_light_gray.blockSignals(True)
        self.rb_white.blockSignals(True)
        if bg == '#ffffff':
            self.rb_white.setChecked(True)
        elif bg == '#f5f5f5':
            self.rb_light_gray.setChecked(True)
        else:
            self.rb_dark.setChecked(True)
        self.rb_dark.blockSignals(False)
        self.rb_light_gray.blockSignals(False)
        self.rb_white.blockSignals(False)

        self._redraw_preview()

    def _push_to_style(self):
        """Write widget values into plot_style._current."""
        plot_style.set_curve_palette(self.combo_palette.currentText())
        plot_style.set_surface_cmap(self.combo_cmap.currentText())
        plot_style.set_phase_color('binodal', self.btn_binodal.color())
        plot_style.set_phase_color('spinodal', self.btn_spinodal.color())
        plot_style.set_phase_color('llcp', self.btn_llcp.color())
        plot_style.set_line_width(self.spin_lw.value())
        plot_style.set_phase_line_width(self.spin_phase_lw.value())
        plot_style.set_font_size(self.spin_font.value())
        plot_style.set_export_dpi(self.spin_dpi.value())
        plot_style.set_grid_enabled(self.chk_grid.isChecked())
        plot_style.set_box_enabled(self.chk_box.isChecked())
        plot_style.set_axes_linewidth(self.spin_axes_lw.value())
        if self.rb_white.isChecked():
            bg = '#ffffff'
        elif self.rb_light_gray.isChecked():
            bg = '#f5f5f5'
        else:
            bg = '#1a1a1a'
        plot_style.set_bg_color(bg)
        # Re-apply rcParams so the preview (and future plots) pick up changes
        plot_style.apply_watereos_style()

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    def _on_theme_change(self, checked=None):
        if not checked:
            return  # Only act on the button being checked, not unchecked
        theme = 'light' if self.rb_theme_light.isChecked() else 'dark'
        plot_style.set_app_theme(theme)
        # Switch plot background to match app theme
        if theme == 'light':
            plot_style.set_bg_color('#f5f5f5')
            self.rb_light_gray.blockSignals(True)
            self.rb_light_gray.setChecked(True)
            self.rb_light_gray.blockSignals(False)
        else:
            plot_style.set_bg_color('#1a1a1a')
            self.rb_dark.blockSignals(True)
            self.rb_dark.setChecked(True)
            self.rb_dark.blockSignals(False)
        plot_style.apply_watereos_style()
        app = QApplication.instance()
        if app:
            app.setStyleSheet(plot_style.get_app_stylesheet())
        self._redraw_preview()

    def _on_any_change(self, _=None):
        self._push_to_style()
        self._redraw_preview()

    def _on_reset(self):
        plot_style.reset_defaults()
        plot_style.apply_watereos_style()
        app = QApplication.instance()
        if app:
            app.setStyleSheet(plot_style.get_app_stylesheet())
        self._sync_from_style()

    # -----------------------------------------------------------------
    # Preview
    # -----------------------------------------------------------------

    def _redraw_preview(self):
        """Draw a small sample plot reflecting the current settings."""
        fig = self.canvas_widget.fig
        fig.clear()

        ax = fig.add_subplot(111)
        plot_style.style_axes(ax)

        # Apply current background / grid from _current
        ax.set_facecolor(plot_style.get_bg_color())
        ax.grid(plot_style.get_grid_enabled())

        # Sample curves using active palette
        palette = plot_style.get_curve_palette()
        x = np.linspace(0, 10, 200)
        for i in range(3):
            ax.plot(x, np.sin(x + i * 1.2) * (3 - 0.5 * i),
                    color=palette[i % len(palette)],
                    linewidth=plot_style.get_line_width(),
                    label=f'Curve {i+1}')

        # Phase boundary sample lines
        ps = plot_style.get_phase_styles()
        bs = ps['binodal']
        ax.axhline(2.0, color=bs['color'], linestyle=bs['linestyle'],
                    linewidth=bs['linewidth'], label='Binodal')
        ss = ps['spinodal']
        ax.axhline(-2.0, color=ss['color'], linestyle=ss['linestyle'],
                    linewidth=ss['linewidth'], label='Spinodal')
        ls = ps['LLCP']
        ax.plot(5, 0, color=ls['color'], marker=ls['marker'],
                markersize=ls['markersize'], linestyle='none', label='LLCP')

        ax.set_xlabel('X label')
        ax.set_ylabel('Y label')
        ax.set_title('Preview')
        ax.legend(fontsize=max(7, plot_style.get_font_size() - 3))

        fig.set_tight_layout(True)
        self.canvas_widget.redraw()
