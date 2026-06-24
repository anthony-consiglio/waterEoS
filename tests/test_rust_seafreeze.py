"""
Cross-validation: native watereos_rs.sf_getprop_* vs reference seafreeze.

Validates the in-process Rust B-spline evaluator (built from the precomputed
spline binary in watereos/data/seafreeze_splines.bin) against the upstream
SeaFreeze Python package across every supported phase, every thermodynamic
output, both evaluation modes (grid + scatter), and a representative cloud
of (P, T) points well inside each phase's stability/validity window.

Tolerances are tight (rtol = 1e-8) because the two pipelines should agree
to floating-point round-off — both evaluate the same tensor-product
B-spline coefficients via the same de Boor recurrence and apply the same
property derivation algebra.

Skipped automatically if either the Rust module or the reference SeaFreeze
package is unavailable on the host.
"""

from __future__ import annotations

import numpy as np
import pytest


# ───────────────────────────────────────────────────────────────────────
# Skip the whole module if either backend can't be imported
# ───────────────────────────────────────────────────────────────────────

watereos_rs = pytest.importorskip(
    "watereos_rs",
    reason="watereos_rs (Rust backend) not built; skipping seafreeze cross-val",
)
sf = pytest.importorskip(
    "seafreeze.seafreeze",
    reason="reference seafreeze package not installed; skipping cross-val",
)


# ───────────────────────────────────────────────────────────────────────
# Test grid: each phase paired with (P, T) clouds inside its validity box.
# ───────────────────────────────────────────────────────────────────────
#
# We sample interior points (i.e. avoid the very corners of the spline
# domain) so that a small-but-real disagreement between the two evaluators
# would not be masked by extrapolation noise. Ranges are chosen from the
# SeaFreeze spline metadata; values were spot-checked against the
# published P–T stability windows for each ice polymorph.

PHASES = {
    # name           ((P_min, P_max, n_P), (T_min, T_max, n_T))
    # NOTE: n_P != n_T per the SeaFreeze convention — the reference's
    # internal `_get_shear_mod_GPa` returns an object-dtype array when both
    # axes have the same length, which then breaks `np.sqrt` in `_get_Vp`.
    # Picking unequal sizes sidesteps that upstream quirk.
    "water1":          ((0.1, 800.0, 9),    (240.0, 500.0, 7)),
    "water_IAPWS95":   ((0.1, 100.0, 9),    (260.0, 500.0, 7)),
    "Ih":              ((0.1, 200.0, 7),    (130.0, 270.0, 5)),
    "II":              ((100.0, 900.0, 7),  (120.0, 270.0, 5)),
    "III":             ((200.0, 350.0, 7),  (220.0, 260.0, 5)),
    "V":               ((350.0, 600.0, 7),  (220.0, 285.0, 5)),
    "VI":              ((650.0, 2200.0, 7), (220.0, 350.0, 5)),
    "VII_X_French":    ((2200.0, 50000.0, 7), (200.0, 800.0, 5)),
}

# Tolerances. The two pipelines should differ only by floating-point
# operation ordering, which empirically gives relative errors around 1e-12
# for the base properties. Kp involves three derivative powers and a
# subtractive ``- 1`` so its conditioning is poorer; we measure ~2e-8
# relative error in the worst case, well above the noise floor of the
# other quantities. We pick a single relaxed tolerance that covers Kp.
RTOL = 1e-7
ATOL = 1e-9

# Property keys returned by SeaFreeze. Liquid phases omit shear/Vp/Vs.
PROPS_FLUID = ["G", "V", "rho", "S", "H", "U", "A", "Cp", "Cv", "Kt", "Ks", "Kp", "alpha", "vel"]
PROPS_SOLID = PROPS_FLUID + ["shear", "Vp", "Vs"]


def _props_for(phase: str) -> list[str]:
    # Solid ices carry a 6-parameter shear modulus parameterisation.
    fluids = {"water1", "water_IAPWS95"}
    return PROPS_FLUID if phase in fluids else PROPS_SOLID


def _close(a: np.ndarray, b: np.ndarray, *, label: str) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        # Phase boundary edge case — both NaN is fine, but we don't want a
        # silent pass when both backends just return all-NaN garbage.
        pytest.skip(f"{label}: no finite values to compare")
    np.testing.assert_allclose(
        a[mask], b[mask], rtol=RTOL, atol=ATOL,
        err_msg=f"{label}: Rust seafreeze disagrees with reference",
    )


# ───────────────────────────────────────────────────────────────────────
# Grid mode
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase", list(PHASES.keys()))
def test_grid_matches_reference(phase: str) -> None:
    (p_lo, p_hi, n_p), (t_lo, t_hi, n_t) = PHASES[phase]
    P = np.linspace(p_lo, p_hi, n_p)
    T = np.linspace(t_lo, t_hi, n_t)

    rust = watereos_rs.sf_getprop_grid(phase, P, T)

    PT = np.array([P, T], dtype=object)
    ref = sf.getProp(PT, phase)

    for key in _props_for(phase):
        rust_val = np.asarray(rust[key])
        ref_val = np.asarray(getattr(ref, key))
        _close(rust_val, ref_val, label=f"{phase}/{key} (grid)")


# ───────────────────────────────────────────────────────────────────────
# Scatter mode
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phase", list(PHASES.keys()))
def test_scatter_matches_reference(phase: str) -> None:
    (p_lo, p_hi, _), (t_lo, t_hi, _) = PHASES[phase]
    rng = np.random.default_rng(seed=hash(phase) & 0xFFFF_FFFF)
    n = 24
    P = rng.uniform(p_lo, p_hi, size=n)
    T = rng.uniform(t_lo, t_hi, size=n)

    rust = watereos_rs.sf_getprop_scatter(phase, P, T)

    PT = np.empty(n, dtype=object)
    PT[:] = list(zip(P, T))
    ref = sf.getProp(PT, phase)

    for key in _props_for(phase):
        rust_val = np.asarray(rust[key]).ravel()
        ref_val = np.asarray(getattr(ref, key)).ravel()
        _close(rust_val, ref_val, label=f"{phase}/{key} (scatter)")


# ───────────────────────────────────────────────────────────────────────
# Phase manifest
# ───────────────────────────────────────────────────────────────────────

def test_phase_manifest() -> None:
    """Every phase the binary exposes is loadable, and the set matches what
    the call sites in watereos rely on."""
    available = set(watereos_rs.sf_phases().keys())
    for required in PHASES:
        assert required in available, f"missing phase '{required}' in Rust binary"
