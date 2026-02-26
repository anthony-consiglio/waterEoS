"""
Simple dict cache for expensive phase diagram computations.
"""


class PhaseDiagramCache:
    """Caches phase diagram results keyed by (model_key, n_pressures)."""

    def __init__(self):
        self._store = {}

    def get(self, model_key, n_pressures=150):
        return self._store.get((model_key, n_pressures))

    def put(self, model_key, n_pressures, data):
        self._store[(model_key, n_pressures)] = data

    def has(self, model_key, n_pressures=150):
        return (model_key, n_pressures) in self._store

    def clear(self, model_key=None):
        if model_key is None:
            self._store.clear()
        else:
            self._store = {k: v for k, v in self._store.items() if k[0] != model_key}


# Global instance shared across tabs
phase_cache = PhaseDiagramCache()
