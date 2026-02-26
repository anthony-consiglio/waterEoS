"""
Tab 1: Property Explorer — plot isobars, isotherms, 2D colormap, or 3D surface.
"""

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QRadioButton, QGroupBox, QSplitter, QComboBox,
)

from watereos_gui.widgets.model_selector import ModelSelector
from watereos_gui.widgets.property_dropdown import PropertyDropdown
from watereos_gui.widgets.pt_range_controls import PTRangeControls
from watereos_gui.widgets.mpl_canvas import MplCanvasWidget
from watereos_gui.widgets.export_toolbar import ExportToolbar
from watereos_gui.utils.model_registry import MODEL_REGISTRY, get_display_label


class PropertyExplorerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._pd_worker = None      # for phase boundary computation
        self._plot_data = None
        self._surface_data = None
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

        # Model selector (single)
        self.model_selector = ModelSelector(title='Model', multi=False)
        self.model_selector.set_selected(['duska2020'])
        sb_layout.addWidget(self.model_selector)

        # Property dropdown
        lbl = QLabel('Property:')
        lbl.setStyleSheet('font-weight: bold;')
        sb_layout.addWidget(lbl)
        self.prop_dropdown = PropertyDropdown()
        self.prop_dropdown.update_for_models(['duska2020'])
        sb_layout.addWidget(self.prop_dropdown)

        # T/P range controls
        self.pt_controls = PTRangeControls()
        sb_layout.addWidget(self.pt_controls)

        # Display mode
        display_group = QGroupBox('Display')
        display_layout = QVBoxLayout(display_group)
        self.rb_curves = QRadioButton('Curves (isobars / isotherms)')
        self.rb_surface_2d = QRadioButton('2D Surface (colormap)')
        self.rb_surface_3d = QRadioButton('3D Surface (interactive)')
        self.rb_curves.setChecked(True)
        display_layout.addWidget(self.rb_curves)
        display_layout.addWidget(self.rb_surface_2d)
        display_layout.addWidget(self.rb_surface_3d)

        # Axis arrangement for surface modes
        self.lbl_z_axis = QLabel('Z / Color axis:')
        display_layout.addWidget(self.lbl_z_axis)
        self.combo_z_axis = QComboBox()
        self.combo_z_axis.addItem('Property', 'property')
        self.combo_z_axis.addItem('Temperature', 'T')
        self.combo_z_axis.addItem('Pressure', 'P')
        display_layout.addWidget(self.combo_z_axis)

        sb_layout.addWidget(display_group)

        # Phase boundary checkbox (for two-state models)
        self.chk_phase_boundary = QCheckBox('Show phase boundaries')
        self.chk_phase_boundary.setToolTip(
            'Overlay spinodal/binodal on plots (two-state models only)'
        )
        sb_layout.addWidget(self.chk_phase_boundary)

        # Update button
        self.btn_update = QPushButton('Update Plot')
        self.btn_update.setStyleSheet('QPushButton { padding: 8px; font-weight: bold; }')
        sb_layout.addWidget(self.btn_update)

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
        self.model_selector.selection_changed.connect(self._on_model_changed)
        self.btn_update.clicked.connect(self._on_update)
        self.export_bar.btn_export_fig.clicked.connect(self._export_figure)
        self.export_bar.btn_export_data.clicked.connect(self._export_data)
        self.rb_curves.toggled.connect(self._on_display_mode_changed)
        self.rb_surface_2d.toggled.connect(self._on_display_mode_changed)
        self.rb_surface_3d.toggled.connect(self._on_display_mode_changed)

        # Initial state
        self._on_display_mode_changed()

    # -- display mode ---------------------------------------------------------

    def _is_surface_mode(self):
        return self.rb_surface_2d.isChecked() or self.rb_surface_3d.isChecked()

    def _on_display_mode_changed(self, _checked=None):
        is_curves = self.rb_curves.isChecked()
        is_surface = self._is_surface_mode()
        # Curve-specific controls
        self.pt_controls.rb_isobars.setVisible(is_curves)
        self.pt_controls.rb_isotherms.setVisible(is_curves)
        self.pt_controls.n_curves.setEnabled(is_curves)
        # Surface-specific controls
        self.lbl_z_axis.setVisible(is_surface)
        self.combo_z_axis.setVisible(is_surface)

    # -- model changed --------------------------------------------------------

    def _on_model_changed(self, models):
        self.prop_dropdown.update_for_models(models)
        if models:
            info = MODEL_REGISTRY[models[0]]
            self.pt_controls.set_ranges_from_model(info)
            self.chk_phase_boundary.setEnabled(info.has_phase_diagram)

    # -- update ---------------------------------------------------------------

    def _on_update(self):
        models = self.model_selector.selected_models()
        if not models:
            return
        prop = self.prop_dropdown.current_property()
        if not prop:
            return

        self.btn_update.setEnabled(False)

        if self._is_surface_mode():
            self._launch_surface_worker(models[0], prop)
        else:
            self._launch_curve_worker(models[0], prop)

    def _launch_curve_worker(self, model_key, prop_key):
        from watereos_gui.backend.workers import PropertyComputeWorker
        self._worker = PropertyComputeWorker(
            model_key=model_key,
            prop_key=prop_key,
            T_range=self.pt_controls.get_T_range(),
            P_range=self.pt_controls.get_P_range(),
            n_curves=self.pt_controls.get_n_curves(),
            n_points=self.pt_controls.get_n_points(),
            isobar_mode=self.pt_controls.is_isobar_mode(),
        )
        self._worker.result_ready.connect(self._on_curve_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _launch_surface_worker(self, model_key, prop_key):
        from watereos_gui.backend.workers import SurfaceComputeWorker
        self._worker = SurfaceComputeWorker(
            model_key=model_key,
            prop_key=prop_key,
            T_range=self.pt_controls.get_T_range(),
            P_range=self.pt_controls.get_P_range(),
            n_points=self.pt_controls.get_n_points(),
        )
        self._worker.result_ready.connect(self._on_surface_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    # -- results --------------------------------------------------------------

    def _on_curve_result(self, data):
        self.btn_update.setEnabled(True)
        self._plot_data = data
        self._plot_curves(data)
        if self.chk_phase_boundary.isChecked():
            self._request_phase_boundaries(data['model_key'], mode='curves')

    def _on_surface_result(self, data):
        self.btn_update.setEnabled(True)
        self._surface_data = data
        self._plot_surface(data)
        if self.chk_phase_boundary.isChecked():
            self._request_phase_boundaries(data['model_key'], mode='surface')

    def _on_error(self, msg):
        self.btn_update.setEnabled(True)
        fig = self.canvas_widget.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error:\n{msg[:300]}', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='red')
        self.canvas_widget.redraw()

    # -- phase boundary overlay -----------------------------------------------

    def _request_phase_boundaries(self, model_key, mode):
        """Fetch phase diagram from cache or compute, then overlay."""
        info = MODEL_REGISTRY.get(model_key)
        if not info or not info.has_phase_diagram:
            return

        from watereos_gui.backend.cache import phase_cache
        cached = phase_cache.get(model_key)
        if cached is not None:
            self._overlay_phase_boundaries(cached, mode)
        else:
            # Compute in background
            from watereos_gui.backend.workers import PhaseDiagramWorker
            self._pd_worker = PhaseDiagramWorker(model_key)
            self._pd_worker.result_ready.connect(
                lambda data, m=mode: self._overlay_phase_boundaries(data, m)
            )
            self._pd_worker.error_occurred.connect(
                lambda msg: print(f'Phase diagram error: {msg}')
            )
            self._pd_worker.start()

    def _overlay_phase_boundaries(self, pd_data, mode):
        """Draw spinodal/binodal on the current plot."""
        if mode == 'surface':
            self._overlay_phase_on_surface(pd_data)
        else:
            self._overlay_phase_on_curves(pd_data)

    def _overlay_phase_on_surface(self, pd_data):
        """Overlay spinodal/binodal on the 2D colormap or 3D surface."""
        from watereos_gui.utils.plot_style import get_phase_styles
        from watereos_gui.backend.computation import compute_property_at_forced_x

        fig = self.canvas_widget.fig
        if not fig.axes:
            return
        ax = fig.axes[0]

        z_choice = self.combo_z_axis.currentData()
        is_3d = self.rb_surface_3d.isChecked()
        model_key = self._surface_data['model_key'] if self._surface_data else None
        prop_key = self.prop_dropdown.current_property()
        info = MODEL_REGISTRY.get(model_key) if model_key else None

        phase_styles = get_phase_styles()

        # --- Spinodal ---
        if 'spinodal' in pd_data:
            sp = pd_data['spinodal']
            s = phase_styles['spinodal']
            if is_3d and model_key:
                p_arr = np.asarray(sp['p_array'])
                branch_cfg = [
                    ('T_upper', 'x_hi_upper', True),
                    ('T_lower', 'x_lo_lower', False),
                ]
                for branch_key, x_key, show_label in branch_cfg:
                    T_b = np.asarray(sp[branch_key])
                    valid = np.isfinite(T_b)
                    if info:
                        valid &= (T_b >= info.T_min - 10) & (T_b <= info.T_max + 10)
                    if not np.any(valid):
                        continue
                    Tv = T_b[valid]
                    Pv = p_arr[valid]

                    x_sp = np.asarray(sp.get(x_key, []))
                    prop_vals = None
                    if x_sp.size == p_arr.size and not prop_key.endswith(('_A', '_B')):
                        prop_vals = compute_property_at_forced_x(
                            model_key, prop_key, Tv, Pv, x_sp[valid])
                    if prop_vals is None or np.all(np.isnan(prop_vals)):
                        prop_vals = self._compute_along_curve(
                            model_key, prop_key, Tv, Pv)

                    Tv, Pv, prop_vals = self._extend_to_llcp(
                        Tv, Pv, prop_vals, pd_data, model_key, prop_key)

                    mask = np.isfinite(prop_vals)
                    xp, yp, zp = self._map_axes_3d(
                        Tv[mask], Pv[mask], prop_vals[mask], z_choice)
                    label = s['label'] if show_label else None
                    ax.plot(xp, yp, zp, color=s['color'], linestyle=s['linestyle'],
                            linewidth=s['linewidth'] + 0.5, label=label)
            else:
                T = np.asarray(sp['T_K'])
                P = np.asarray(sp['p_MPa'])
                ax.plot(T, P, color=s['color'], linestyle=s['linestyle'],
                        linewidth=s['linewidth'], label=s['label'])

        # --- Binodal dome ---
        if 'binodal' in pd_data:
            bn = pd_data['binodal']
            s = phase_styles['binodal']
            if is_3d and model_key:
                p_arr = np.asarray(bn['p_array'])
                T_bn = np.asarray(bn['T_binodal'])
                valid = np.isfinite(T_bn)
                if info:
                    valid &= (T_bn >= info.T_min - 10) & (T_bn <= info.T_max + 10)

                if np.any(valid):
                    T_b = T_bn[valid]
                    P_b = p_arr[valid]
                    prop_lo, prop_hi = self._compute_binodal_dome(
                        model_key, prop_key, T_b, P_b, None, None)

                    if prop_lo is not None and prop_hi is not None:
                        for branch, lbl in [(prop_lo, s['label']), (prop_hi, None)]:
                            Te, Pe, br = self._extend_to_llcp(
                                T_b, P_b, branch, pd_data, model_key, prop_key)
                            mask = np.isfinite(br)
                            xp, yp, zp = self._map_axes_3d(
                                Te[mask], Pe[mask], br[mask], z_choice)
                            ax.plot(xp, yp, zp, color=s['color'],
                                    linestyle=s['linestyle'],
                                    linewidth=s['linewidth'] + 0.5, label=lbl)
            else:
                T = np.asarray(bn['T_K'])
                P = np.asarray(bn['p_MPa'])
                ax.plot(T, P, color=s['color'], linestyle=s['linestyle'],
                        linewidth=s['linewidth'], label=s['label'])

        # --- LLCP ---
        if 'LLCP' in pd_data:
            llcp = pd_data['LLCP']
            s = phase_styles['LLCP']
            T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])

            if is_3d and model_key:
                prop_c = self._compute_along_curve(
                    model_key, prop_key,
                    np.array([T_c]), np.array([P_c]))
                if np.isfinite(prop_c[0]):
                    x, y, z = self._map_axes_3d(
                        np.array([T_c]), np.array([P_c]),
                        prop_c, z_choice)
                    ax.plot(x, y, z, color=s['color'], marker=s['marker'],
                            markersize=s['markersize'], linestyle='none',
                            label=f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)")
            else:
                ax.plot(T_c, P_c, color=s['color'], marker=s['marker'],
                        markersize=s['markersize'], linestyle='none',
                        label=f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)")

        ax.legend(fontsize=8)
        self.canvas_widget.redraw()

    def _map_axes_2d(self, T, P, z_choice):
        """Map T, P arrays to x, y for 2D plot based on z_choice."""
        if z_choice == 'property':
            return T, P        # x=T, y=P
        elif z_choice == 'T':
            return P, T        # x=P, y=... (phase boundaries don't have property values)
        else:  # 'P'
            return T, P        # x=T, y=... same issue
        # For non-default z_choice, the T-P phase boundary overlays
        # still plot in the two physical axes that are shown

    def _map_axes_3d(self, T, P, Z, z_choice):
        """Map T, P, Z to x, y, z for 3D plot based on z_choice."""
        if z_choice == 'property':
            return T, P, Z
        elif z_choice == 'T':
            return P, Z, T
        else:  # 'P'
            return T, Z, P

    def _compute_binodal_dome(self, model_key, prop_key, T_bn, P_bn, x_lo, x_hi):
        """
        Compute property values for both coexisting phases along the binodal.

        Uses a T±ε offset: at the first-order transition the model picks
        different phases on each side.  T+ε → HDL-rich, T−ε → LDL-rich.

        Returns (prop_above, prop_below) — arrays for the two coexisting
        phases.  Returns (None, None) when dome cannot be computed.
        """
        # State-specific properties (_A / _B) don't form a dome
        if prop_key.endswith('_A') or prop_key.endswith('_B'):
            return None, None

        eps = 0.1  # K — small enough to stay near binodal
        prop_above = self._compute_along_curve(
            model_key, prop_key, T_bn + eps, P_bn)
        prop_below = self._compute_along_curve(
            model_key, prop_key, T_bn - eps, P_bn)
        return prop_above, prop_below

    def _extend_to_llcp(self, T_arr, P_arr, prop_arr, pd_data, model_key, prop_key):
        """Prepend LLCP point so curves converge at the critical point."""
        if 'LLCP' not in pd_data:
            return T_arr, P_arr, prop_arr
        llcp = pd_data['LLCP']
        T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])
        prop_c = self._compute_along_curve(
            model_key, prop_key, np.array([T_c]), np.array([P_c]))
        if not np.isfinite(prop_c[0]):
            return T_arr, P_arr, prop_arr
        return (np.concatenate([[T_c], T_arr]),
                np.concatenate([[P_c], P_arr]),
                np.concatenate([prop_c, prop_arr]))

    def _overlay_phase_on_curves(self, pd_data):
        """Overlay binodal/spinodal dome and LLCP on property-vs-T/P curves."""
        from watereos_gui.utils.plot_style import get_phase_styles
        from watereos_gui.backend.computation import compute_property_at_forced_x

        if not self._plot_data:
            return

        fig = self.canvas_widget.fig
        if not fig.axes:
            return
        ax = fig.axes[0]

        model_key = self._plot_data['model_key']
        prop_key = self.prop_dropdown.current_property()
        isobar_mode = self.pt_controls.is_isobar_mode()
        info = MODEL_REGISTRY[model_key]
        phase_styles = get_phase_styles()

        # --- Spinodal: use metastable-phase x to put curves inside dome ---
        if 'spinodal' in pd_data:
            sp = pd_data['spinodal']
            p_arr = np.asarray(sp['p_array'])
            style = phase_styles['spinodal']

            # T_upper: metastable LDL disappears → use x_hi_upper
            # T_lower: metastable HDL disappears → use x_lo_lower
            branch_cfg = [
                ('T_upper', 'x_hi_upper', True),
                ('T_lower', 'x_lo_lower', False),
            ]
            for branch_key, x_key, show_label in branch_cfg:
                T_branch = np.asarray(sp[branch_key])
                valid = (np.isfinite(T_branch)
                         & (T_branch >= info.T_min - 10)
                         & (T_branch <= info.T_max + 10))
                if not np.any(valid):
                    continue
                T_b = T_branch[valid]
                P_b = p_arr[valid]

                # Try forced-x for correct metastable density
                x_sp = np.asarray(sp.get(x_key, []))
                prop_vals = None
                if x_sp.size == p_arr.size and not prop_key.endswith(('_A', '_B')):
                    x_b = x_sp[valid]
                    prop_vals = compute_property_at_forced_x(
                        model_key, prop_key, T_b, P_b, x_b)
                # Fallback: equilibrium property
                if prop_vals is None or np.all(np.isnan(prop_vals)):
                    prop_vals = self._compute_along_curve(
                        model_key, prop_key, T_b, P_b)

                # Extend to LLCP
                T_b, P_b, prop_vals = self._extend_to_llcp(
                    T_b, P_b, prop_vals, pd_data, model_key, prop_key)

                x_vals = T_b if isobar_mode else P_b
                mask = np.isfinite(prop_vals)
                label = style['label'] if show_label else None
                ax.plot(x_vals[mask], prop_vals[mask],
                        color=style['color'], linestyle=style['linestyle'],
                        linewidth=style['linewidth'], label=label)

        # --- Binodal dome: two coexisting phase branches ---
        if 'binodal' in pd_data:
            bn = pd_data['binodal']
            p_arr = np.asarray(bn['p_array'])
            T_bn = np.asarray(bn['T_binodal'])
            style = phase_styles['binodal']

            valid = (np.isfinite(T_bn)
                     & (T_bn >= info.T_min - 10)
                     & (T_bn <= info.T_max + 10))

            if np.any(valid):
                T_b = T_bn[valid]
                P_b = p_arr[valid]

                prop_lo, prop_hi = self._compute_binodal_dome(
                    model_key, prop_key, T_b, P_b, None, None)

                if prop_lo is not None and prop_hi is not None:
                    for branch, lbl in [(prop_lo, style['label']), (prop_hi, None)]:
                        T_ext, P_ext, br_ext = self._extend_to_llcp(
                            T_b, P_b, branch, pd_data, model_key, prop_key)
                        x_vals = T_ext if isobar_mode else P_ext
                        mask = np.isfinite(br_ext)
                        ax.plot(x_vals[mask], br_ext[mask],
                                color=style['color'], linestyle=style['linestyle'],
                                linewidth=style['linewidth'], label=lbl)

        # --- LLCP marker ---
        if 'LLCP' in pd_data:
            llcp = pd_data['LLCP']
            T_c, P_c = float(llcp['T_K']), float(llcp['p_MPa'])
            prop_c = self._compute_along_curve(
                model_key, prop_key,
                np.array([T_c]), np.array([P_c]))
            style = phase_styles['LLCP']
            x_c = T_c if isobar_mode else P_c
            if np.isfinite(prop_c[0]):
                ax.plot(x_c, prop_c[0],
                        color=style['color'], marker=style['marker'],
                        markersize=style['markersize'], linestyle='none',
                        label=f"LLCP ({T_c:.1f} K, {P_c:.1f} MPa)")

        ax.legend(fontsize=8)
        self.canvas_widget.redraw()

    def _compute_along_curve(self, model_key, prop_key, T_arr, P_arr):
        """Compute property values along a (T, P) curve using grid diagonal."""
        from watereos import getProp
        PT = np.array([P_arr, T_arr], dtype=object)
        result = getProp(PT, model_key)
        Z = np.asarray(getattr(result, prop_key), dtype=float)
        return np.diag(Z)

    # -- plotting -------------------------------------------------------------

    def _plot_curves(self, data):
        from watereos_gui.utils.plot_style import get_curve_palette, style_axes

        fig = self.canvas_widget.fig
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        self.canvas_widget.canvas.ax = ax

        colors = get_curve_palette()
        for i, (x, y, label) in enumerate(zip(
                data['x_values'], data['y_values'], data['curve_labels'])):
            mask = np.isfinite(y)
            ax.plot(x[mask], y[mask], color=colors[i % len(colors)], label=label)

        ax.set_xlabel(data['x_label'])
        ax.set_ylabel(data['y_label'])
        ax.set_title(data['title'])
        if data['curve_labels']:
            ax.legend(fontsize=9)
        fig.set_tight_layout(True)
        self.canvas_widget.redraw()

    def _plot_surface(self, data):
        if self.rb_surface_3d.isChecked():
            self._plot_surface_3d(data)
        else:
            self._plot_surface_2d(data)

    def _plot_surface_2d(self, data):
        from watereos_gui.utils.plot_style import style_axes, get_surface_cmap

        fig = self.canvas_widget.fig
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        self.canvas_widget.canvas.ax = ax

        T_grid = data['T_grid']
        P_grid = data['P_grid']
        Z = data['Z']
        Z_masked = np.ma.masked_invalid(Z)

        z_choice = self.combo_z_axis.currentData()
        prop_label = get_display_label(data['prop_key'])
        info = MODEL_REGISTRY[data['model_key']]

        if z_choice == 'property':
            x, y, c = T_grid, P_grid, Z_masked
            xlabel, ylabel, clabel = 'Temperature [K]', 'Pressure [MPa]', prop_label
        elif z_choice == 'T':
            # x=P, y=Property, color=T → scatter needed (irregular grid in y)
            x, y, c = P_grid.ravel(), Z_masked.data.ravel(), T_grid.ravel()
            xlabel, ylabel, clabel = 'Pressure [MPa]', prop_label, 'Temperature [K]'
        else:  # 'P'
            x, y, c = T_grid.ravel(), Z_masked.data.ravel(), P_grid.ravel()
            xlabel, ylabel, clabel = 'Temperature [K]', prop_label, 'Pressure [MPa]'

        cmap = get_surface_cmap()
        if z_choice == 'property':
            im = ax.pcolormesh(x, y, c, shading='gouraud', cmap=cmap)
            try:
                cs = ax.contour(x, y, c, levels=12,
                                colors='white', linewidths=0.6, alpha=0.6)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.4g')
            except Exception:
                pass
        else:
            # Scatter plot for non-standard axis arrangements
            mask = np.isfinite(c) & np.isfinite(y)
            im = ax.scatter(x[mask], y[mask], c=c[mask], s=1.5,
                            cmap=cmap, edgecolors='none')

        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(clabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{info.display_name} — {prop_label}')

        fig.set_tight_layout(True)
        self.canvas_widget.redraw()

    def _plot_surface_3d(self, data):
        # Import to register 3D projection
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from watereos_gui.utils.plot_style import get_surface_cmap
        import matplotlib.cm as cm

        fig = self.canvas_widget.fig
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        self.canvas_widget.canvas.ax = ax

        T_grid = data['T_grid']
        P_grid = data['P_grid']
        Z = data['Z']
        Z_masked = np.ma.masked_invalid(Z)

        z_choice = self.combo_z_axis.currentData()
        prop_label = get_display_label(data['prop_key'])
        info = MODEL_REGISTRY[data['model_key']]

        if z_choice == 'property':
            X, Y, ZZ = T_grid, P_grid, Z_masked
            xlabel, ylabel, zlabel = 'Temperature [K]', 'Pressure [MPa]', prop_label
        elif z_choice == 'T':
            X, Y, ZZ = P_grid, Z_masked, T_grid
            xlabel, ylabel, zlabel = 'Pressure [MPa]', prop_label, 'Temperature [K]'
        else:  # 'P'
            X, Y, ZZ = T_grid, Z_masked, P_grid
            xlabel, ylabel, zlabel = 'Temperature [K]', prop_label, 'Pressure [MPa]'

        # Color by the z variable
        norm = None
        if hasattr(ZZ, 'data'):
            z_for_color = ZZ.data
        else:
            z_for_color = ZZ
        z_finite = z_for_color[np.isfinite(z_for_color)]
        if z_finite.size > 0:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=np.nanmin(z_finite), vmax=np.nanmax(z_finite))
        cmap_name = get_surface_cmap()
        cmap_obj = cm.get_cmap(cmap_name)
        facecolors = cmap_obj(norm(z_for_color)) if norm else None

        ax.plot_surface(X, Y, ZZ, facecolors=facecolors,
                        rstride=2, cstride=2, alpha=0.85,
                        linewidth=0, antialiased=True)

        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_zlabel(zlabel, labelpad=10)
        ax.set_title(f'{info.display_name} — {prop_label}')

        # Add colorbar via a ScalarMappable
        if norm is not None:
            sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            cb.set_label(zlabel)

        self.canvas_widget.redraw()

    # -- export ---------------------------------------------------------------

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
            export_property_curves_csv(path, self._plot_data)
