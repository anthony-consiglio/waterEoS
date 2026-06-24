"""Convert a Plotly figure to a strictly-JSON-safe dict (NaN/Inf -> null).

Uses Figure.to_dict() (not plotly's JSON engine) to avoid Plotly's binary
base64 array encoding; numpy arrays/scalars are converted recursively and
non-finite floats become None so json.dumps produces strictly valid JSON.

Plotly 6+ encodes numpy arrays as {"dtype", "bdata"[, "shape"]} dicts even
inside to_dict().  _sanitize decodes these via numpy.frombuffer (more robust
than manual struct unpacking) and then applies NaN/Inf -> None replacement.
"""
import base64
import math

import numpy as np

# Plotly dtype codes -> numpy dtype strings (explicit little-endian prefix)
# Plotly bdata is always little-endian; bare dtype strings use native byte order
# which is silently wrong on big-endian servers.
_DTYPE_MAP = {
    "f4": "<f4", "f8": "<f8",
    "i1": "<i1", "i2": "<i2", "i4": "<i4", "i8": "<i8",
    "u1": "<u1", "u2": "<u2", "u4": "<u4", "u8": "<u8",
}


def _is_bdata(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "dtype" in obj
        and "bdata" in obj
        and obj["dtype"] in _DTYPE_MAP
    )


def _decode_bdata(obj: dict):
    """Decode a Plotly binary-encoded array dict to a nested Python list."""
    raw = base64.b64decode(obj["bdata"])
    arr = np.frombuffer(raw, dtype=_DTYPE_MAP[obj["dtype"]])
    shape_str = obj.get("shape", "")
    if shape_str:
        shape = tuple(int(s) for s in shape_str.split(","))
        arr = arr.reshape(shape)
    return _sanitize(arr)


def _sanitize(obj):
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):  # numpy scalar (float64, int64, bool_, ...)
        return _sanitize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if _is_bdata(obj):
        return _decode_bdata(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def figure_to_jsonable(fig) -> dict:
    """Return a dict json.dumps can serialize with no NaN/Infinity tokens."""
    return _sanitize(fig.to_dict())
