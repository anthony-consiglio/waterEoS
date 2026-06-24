"""
Simple dict cache for expensive phase diagram computations.

Phase diagrams (spinodal, binodal, TMD, etc.) are costly to compute
and rarely change, so results are cached in memory keyed by
``(model_key, n_pressures)``.
"""


class PhaseDiagramCache:
    """In-memory cache for phase diagram results.

    Keys are ``(model_key, n_pressures)`` tuples.  Values are the
    dict returned by ``compute_phase_diagram_data()``.
    """

    def __init__(self):
        self._store = {}

    def get(self, model_key, n_pressures=150):
        """Return cached data, or ``None`` if not present."""
        return self._store.get((model_key, n_pressures))

    def put(self, model_key, n_pressures, data):
        """Store *data* under the given key."""
        self._store[(model_key, n_pressures)] = data

    def has(self, model_key, n_pressures=150):
        """Return ``True`` if a result is cached for this key."""
        return (model_key, n_pressures) in self._store

    def clear(self, model_key=None):
        """Clear cached entries.

        If *model_key* is ``None``, clear everything.  Otherwise remove
        only entries for that model.
        """
        if model_key is None:
            self._store.clear()
        else:
            self._store = {k: v for k, v in self._store.items() if k[0] != model_key}


# Global instance shared across tabs
phase_cache = PhaseDiagramCache()
