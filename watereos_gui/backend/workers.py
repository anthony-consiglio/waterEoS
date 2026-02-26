"""
QThread workers for background computation.
"""

import traceback
from PyQt6.QtCore import QThread, pyqtSignal

from watereos_gui.backend.computation import (
    compute_property_curves,
    compute_property_surface,
    compute_multi_model_curves,
    compute_phase_diagram_data,
    compute_point_properties,
)
from watereos_gui.backend.cache import phase_cache


class PropertyComputeWorker(QThread):
    """Compute property curves for a single model."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_key, prop_key, T_range, P_range,
                 n_curves, n_points, isobar_mode):
        super().__init__()
        self._args = (model_key, prop_key, T_range, P_range,
                      n_curves, n_points, isobar_mode)

    def run(self):
        try:
            data = compute_property_curves(*self._args)
            self.result_ready.emit(data)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class SurfaceComputeWorker(QThread):
    """Compute a 2-D property surface over T and P."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_key, prop_key, T_range, P_range, n_points):
        super().__init__()
        self._args = (model_key, prop_key, T_range, P_range, n_points)

    def run(self):
        try:
            data = compute_property_surface(*self._args)
            self.result_ready.emit(data)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class MultiModelPropertyWorker(QThread):
    """Compute property curves for multiple models."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_keys, prop_key, T_range, P_range,
                 n_curves, n_points, isobar_mode):
        super().__init__()
        self._args = (model_keys, prop_key, T_range, P_range,
                      n_curves, n_points, isobar_mode)

    def run(self):
        try:
            data = compute_multi_model_curves(*self._args)
            self.result_ready.emit(data)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class PhaseDiagramWorker(QThread):
    """Compute phase diagram with caching."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_key, n_pressures=150):
        super().__init__()
        self._model_key = model_key
        self._n_pressures = n_pressures

    def run(self):
        try:
            cached = phase_cache.get(self._model_key, self._n_pressures)
            if cached is not None:
                self.result_ready.emit(cached)
                return
            data = compute_phase_diagram_data(self._model_key, self._n_pressures)
            phase_cache.put(self._model_key, self._n_pressures, data)
            self.result_ready.emit(data)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class PointCalcWorker(QThread):
    """Compute all properties at a single (T, P) for multiple models."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_keys, T_K, P_MPa):
        super().__init__()
        self._args = (model_keys, T_K, P_MPa)

    def run(self):
        try:
            data = compute_point_properties(*self._args)
            self.result_ready.emit(data)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())
