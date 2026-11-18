"""
Arrays that must pair up point by point must have the same length.

The Rust entry points used to index past the end of the shorter array, or
`assert_eq!` on the lengths. Either one panics, and a panic aborts the
interpreter because the release profile sets `panic = "abort"`. Each test
here requires a ValueError instead.

Skipped automatically if the Rust module is unavailable on the host.
"""

from __future__ import annotations

import numpy as np
import pytest

watereos_rs = pytest.importorskip(
    "watereos_rs",
    reason="watereos_rs (Rust backend) not built; skipping length tests",
)

SHORT, LONG = 2, 3


def _pairs():
    """(first, second) argument arrays of unequal length, both ways round."""
    return [(np.full(LONG, 250.0), np.full(SHORT, 100.0)),
            (np.full(SHORT, 250.0), np.full(LONG, 100.0))]


@pytest.mark.parametrize("name", ["caupin", "caupin_kim", "holten", "duska",
                                  "bridge29", "local26", "shi_tanaka",
                                  "vapor"])
def test_batch_rejects_unequal_lengths(name):
    batch = getattr(watereos_rs, f"compute_batch_{name}")
    for t, p in _pairs():
        with pytest.raises(ValueError, match="length"):
            batch(t, p)


def test_seafreeze_scatter_rejects_unequal_lengths():
    for t, p in _pairs():
        with pytest.raises(ValueError, match="length"):
            watereos_rs.sf_getprop_scatter("Ih", p, t)
        with pytest.raises(ValueError, match="length"):
            watereos_rs.sf_eval_raw("Ih", p, t, 0, 0, False)


def test_seafreeze_grid_takes_any_lengths():
    """On a grid, P and T are axes, so their lengths are independent."""
    p, t = np.linspace(0.1, 200.0, LONG), np.linspace(250.0, 270.0, SHORT)
    props = watereos_rs.sf_getprop_grid("Ih", p, t)
    assert props["rho"].shape == (LONG, SHORT)
    assert watereos_rs.sf_eval_raw("Ih", p, t, 0, 0, True).shape == (LONG, SHORT)


@pytest.mark.parametrize("model", ["caupin", "caupin_kim", "holten", "duska"])
@pytest.mark.parametrize("short", ["spin_p", "spin_t_upper", "spin_t_lower"])
def test_binodal_rejects_unequal_lengths(model, short):
    binodal = getattr(watereos_rs, f"compute_binodal_{model}")
    arrays = {
        "spin_p": np.linspace(100.0, 150.0, LONG),
        "spin_t_upper": np.full(LONG, 230.0),
        "spin_t_lower": np.full(LONG, 200.0),
    }
    arrays[short] = arrays[short][:SHORT]
    with pytest.raises(ValueError, match="length"):
        binodal(arrays["spin_p"], arrays["spin_t_upper"],
                arrays["spin_t_lower"], 220.0, 50.0)
