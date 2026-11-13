"""
The Rust batch entry points evaluate a small batch on the calling thread
and split a large one across the thread pool (src/par.rs). Which of the
two runs must not change a single bit of the result: every point is
evaluated by the same function either way.

Each test evaluates one large batch in a single call (the thread pool)
and again in pieces of 64 points (the calling thread), and requires the
two to be identical, NaN for NaN.

Skipped automatically if the Rust module is unavailable on the host.
"""

from __future__ import annotations

import numpy as np
import pytest

watereos_rs = pytest.importorskip(
    "watereos_rs",
    reason="watereos_rs (Rust backend) not built; skipping scheduling tests",
)

# Far above the size at which a batch is handed to the thread pool.
N_LARGE = 100_000
PIECE = 64


def _points(seed, t_range, p_range, n=N_LARGE):
    rng = np.random.default_rng(seed)
    return rng.uniform(*t_range, n), rng.uniform(*p_range, n)


def _in_pieces(call, t, p):
    """Evaluate in PIECE-sized calls and join each output array."""
    parts = [call(t[i:i + PIECE], p[i:i + PIECE])
             for i in range(0, len(t), PIECE)]
    return {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}


def _assert_identical(whole, pieces):
    assert list(whole) == list(pieces)
    for key in whole:
        a, b = np.asarray(whole[key]), np.asarray(pieces[key])
        assert a.dtype == b.dtype, key
        assert a.shape == b.shape, key
        # Bytes, not values: NaN must match NaN and -0.0 must not match 0.0.
        assert a.tobytes() == b.tobytes(), key


# The ranges run past each model's domain on purpose, so that the NaN
# results are compared as well as the finite ones.
@pytest.mark.parametrize("name", ["holten", "caupin", "caupin_kim", "duska"])
@pytest.mark.parametrize("branch", ["stable", "hdl", "ldl"])
def test_two_state_batch_is_independent_of_batch_size(name, branch):
    fn = getattr(watereos_rs, f"compute_batch_{name}")
    t, p = _points(11, (150.0, 330.0), (-150.0, 450.0))
    whole = fn(t, p, branch)
    # A third of these points still have an LDL branch; most have the others.
    assert np.isfinite(whole["rho"]).sum() > N_LARGE // 10
    _assert_identical(whole, _in_pieces(lambda tt, pp: fn(tt, pp, branch), t, p))


def test_shi_tanaka_batch_is_independent_of_batch_size():
    fn = watereos_rs.compute_batch_shi_tanaka
    t, p = _points(12, (150.0, 330.0), (-150.0, 450.0))
    _assert_identical(fn(t, p), _in_pieces(fn, t, p))


@pytest.mark.parametrize("phase", sorted(watereos_rs.sf_phases()))
def test_seafreeze_scatter_is_independent_of_batch_size(phase):
    t, p = _points(13, (150.0, 520.0), (-50.0, 2500.0))

    def call(tt, pp):
        return watereos_rs.sf_getprop_scatter(phase, pp, tt)

    _assert_identical(call(t, p), _in_pieces(call, t, p))


@pytest.mark.parametrize("phase", ["Ih", "water1"])
def test_seafreeze_grid_matches_scatter(phase):
    """A large grid (thread pool) against its rows as small scatter calls."""
    t = np.linspace(245.0, 270.0, 50)
    p = np.linspace(0.1, 200.0, 400)
    grid = watereos_rs.sf_getprop_grid(phase, p, t)
    for key, arr in grid.items():
        assert arr.shape == (p.size, t.size), key
    for i in (0, 1, 199, 399):
        row = watereos_rs.sf_getprop_scatter(phase, np.full(t.size, p[i]), t)
        assert list(row) == list(grid)
        for key in grid:
            assert grid[key][i].tobytes() == row[key].tobytes(), (key, i)


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("deriv", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)])
def test_seafreeze_raw_derivative_is_independent_of_batch_size(grid, deriv):
    d_p, d_t = deriv
    if grid:
        t = np.linspace(245.0, 270.0, 50)
        p = np.linspace(0.1, 200.0, 400)
        whole = watereos_rs.sf_eval_raw("Ih", p, t, d_p, d_t, True)
        assert whole.shape == (p.size, t.size)
        rows = np.stack([
            watereos_rs.sf_eval_raw("Ih", p[i:i + 1], t, d_p, d_t, True)[0]
            for i in range(p.size)
        ])
        assert whole.tobytes() == rows.tobytes()
    else:
        t, p = _points(14, (245.0, 270.0), (0.1, 200.0), n=20_000)
        whole = watereos_rs.sf_eval_raw("Ih", p, t, d_p, d_t, False)
        pieces = np.concatenate([
            watereos_rs.sf_eval_raw("Ih", p[i:i + PIECE], t[i:i + PIECE],
                                    d_p, d_t, False)
            for i in range(0, t.size, PIECE)
        ])
        assert whole.tobytes() == pieces.tobytes()
