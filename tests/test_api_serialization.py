import json
import math
import numpy as np
import plotly.graph_objects as go
from watereos_api.serialization import figure_to_jsonable


def test_nan_and_inf_become_null_and_valid_json():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1.0, np.nan, np.inf]))
    obj = figure_to_jsonable(fig)
    s = json.dumps(obj)            # must not raise
    parsed = json.loads(s)         # round-trips as valid JSON
    ys = parsed["data"][0]["y"]
    assert ys[0] == 1.0
    assert ys[1] is None           # NaN -> null
    assert ys[2] is None           # Inf -> null
    assert "NaN" not in s and "Infinity" not in s


def test_numpy_arrays_serialize():
    fig = go.Figure(go.Scatter(x=np.array([1, 2]), y=np.array([3.0, 4.0])))
    obj = figure_to_jsonable(fig)
    json.dumps(obj)
    assert list(obj["data"][0]["y"]) == [3.0, 4.0]


def test_2d_surface_bdata_shape_and_nonfinite():
    import json
    fig = go.Figure(go.Surface(z=np.array([[1.0, np.nan],
                                           [np.inf, 4.0]])))
    obj = figure_to_jsonable(fig)
    s = json.dumps(obj)
    assert "NaN" not in s and "Infinity" not in s
    z = obj["data"][0]["z"]
    assert z[0][0] == 1.0 and z[0][1] is None
    assert z[1][0] is None and z[1][1] == 4.0
