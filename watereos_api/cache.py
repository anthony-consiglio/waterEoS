"""Process-local memoization for slow computes."""
import json
import threading

_LOCK = threading.Lock()
_STORE: dict = {}


def memoize(key_parts, producer):
    key = json.dumps(key_parts, sort_keys=True, default=str)
    with _LOCK:
        if key in _STORE:
            return _STORE[key]
    value = producer()
    with _LOCK:
        _STORE.setdefault(key, value)
        return _STORE[key]
