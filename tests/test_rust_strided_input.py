"""
The Rust entry points must accept 1-D arrays of any memory layout.

A column of a 2-D array (`path[:, 1]`), a stepped slice and a reversed
view are all non-contiguous. Reading one through `as_slice()` or
`to_vec()` fails, and in the phase-diagram functions that failure was a
panic, which aborts the interpreter because the release profile sets
`panic = "abort"`. Each test here passes such views and requires the
result to equal the same call on `np.ascontiguousarray` copies.

Skipped automatically if the Rust module is unavailable on the host.
"""

from __future__ import annotations

import numpy as np
import pytest

watereos_rs = pytest.importorskip(
    "watereos_rs",
    reason="watereos_rs (Rust backend) not built; skipping strided-input tests",
)

N_PRESSURES = 40


def _strided_views(x):
    """Non-contiguous views holding the values of `x`, keyed by layout."""
    x = np.asarray(x, dtype=float)
    views = {
        "column": np.column_stack([np.zeros_like(x), x])[:, 1],
        "every_other": np.repeat(x, 2)[::2],
        "reversed": x[::-1].copy()[::-1],
    }
    for name, view in views.items():
        assert not view.flags.c_contiguous, name
        np.testing.assert_array_equal(view, x)
    return views


def _assert_same_result(strided, contiguous):
    if isinstance(contiguous, dict):
        assert list(strided) == list(contiguous)
        for key in contiguous:
            np.testing.assert_array_equal(strided[key], contiguous[key],
                                          err_msg=key)
    else:
        np.testing.assert_array_equal(strided, contiguous)


def _check(call, *arrays):
    """Call with each strided layout of every array, against contiguous."""
    expected = call(*(np.ascontiguousarray(a, dtype=float) for a in arrays))
    per_array = [_strided_views(a) for a in arrays]
    for layout in per_array[0]:
        got = call(*(views[layout] for views in per_array))
        _assert_same_result(got, expected)


# ── Liquid-liquid spinodal and binodal ──────────────────────────────────


def _llcp(model):
    if model in ("caupin", "caupin_kim"):
        import caupin_eos.params_kim as params_kim
        from caupin_eos import phase_diagram
        pset = params_kim if model == "caupin_kim" else phase_diagram._DEFAULT_PARAMS
        llcp = phase_diagram.find_LLCP(pset)
    elif model == "holten":
        from holten_eos import phase_diagram
        llcp = phase_diagram.find_LLCP()
    else:
        from duska_eos import phase_diagram
        llcp = phase_diagram.find_LLCP()
    return llcp["T_K"], llcp["p_MPa"]


PHASE_MODELS = ["caupin", "caupin_kim", "holten", "duska"]


@pytest.mark.parametrize("model", PHASE_MODELS)
def test_spinodal_accepts_strided_pressures(model):
    from watereos.two_state_phase import _default_pressure_grid

    spinodal = getattr(watereos_rs, f"compute_spinodal_{model}")
    t_llcp, p_llcp = _llcp(model)
    p = _default_pressure_grid(p_llcp, N_PRESSURES)
    result = spinodal(np.ascontiguousarray(p), t_llcp, p_llcp)
    assert np.isfinite(result["T_upper"]).any()
    _check(lambda pp: spinodal(pp, t_llcp, p_llcp), p)


@pytest.mark.parametrize("model", PHASE_MODELS)
def test_binodal_accepts_strided_spinodal(model):
    from watereos.two_state_phase import _default_pressure_grid

    spinodal = getattr(watereos_rs, f"compute_spinodal_{model}")
    binodal = getattr(watereos_rs, f"compute_binodal_{model}")
    t_llcp, p_llcp = _llcp(model)
    spin = spinodal(_default_pressure_grid(p_llcp, N_PRESSURES), t_llcp, p_llcp)
    result = binodal(spin["p_array"], spin["T_upper"], spin["T_lower"],
                     t_llcp, p_llcp)
    assert np.isfinite(result["T_K"]).any()
    _check(lambda sp, tu, tl: binodal(sp, tu, tl, t_llcp, p_llcp),
           spin["p_array"], spin["T_upper"], spin["T_lower"])


@pytest.mark.parametrize("module", ["caupin_eos", "holten_eos", "duska_eos"])
def test_public_phase_diagram_accepts_strided_pressures(module):
    """The public wrappers hand their pressure array to Rust unchanged."""
    import importlib

    phase_diagram = importlib.import_module(f"{module}.phase_diagram")
    p = np.linspace(80.0, 180.0, N_PRESSURES)
    _check(phase_diagram.compute_spinodal_curve, p)
    _check(phase_diagram.compute_binodal_curve, p)


# ── Property batches ────────────────────────────────────────────────────


def _tp_points(n=200, t_range=(200.0, 320.0), p_range=(0.1, 300.0)):
    rng = np.random.default_rng(7)
    return rng.uniform(*t_range, n), rng.uniform(*p_range, n)


@pytest.mark.parametrize("name", ["caupin", "caupin_kim", "holten", "duska",
                                  "bridge29", "local26", "bridge37", "shi_tanaka",
                                  "vapor"])
def test_batch_accepts_strided_input(name):
    batch = getattr(watereos_rs, f"compute_batch_{name}")
    t, p = _tp_points()
    _check(batch, t, p)


def test_vapor_grid_accepts_strided_input():
    t = np.linspace(280.0, 600.0, 30)
    p = np.linspace(0.001, 0.5, 20)
    _check(watereos_rs.compute_grid_vapor, t, p)


def test_seafreeze_accepts_strided_input():
    t, p = _tp_points(t_range=(240.0, 270.0), p_range=(0.1, 200.0))
    _check(lambda pp, tt: watereos_rs.sf_getprop_scatter("Ih", pp, tt), p, t)
    _check(lambda pp, tt: watereos_rs.sf_getprop_grid("Ih", pp, tt),
           p[:30], t[:20])
    for grid in (False, True):
        pp, tt = (p[:30], t[:20]) if grid else (p, t)
        _check(lambda a, b: watereos_rs.sf_eval_raw("Ih", a, b, 1, 0, grid),
               pp, tt)
