"""
Regression tests: Caupin and Duska two-state EoSs MUST return NaN below the
liquid-vapor spinodal — not plausible-but-unphysical clamped values.

Previously the spinodal Gibbs contribution G^sigma = A(T) * (P_hat - Ps_hat)^(3/2)
silently clamped its radicand to 1e-30 (Caupin) or 0 (Duska), returning finite-
looking densities far below the spinodal. That misleads exactly the stretched-
water researchers these models target.

After the fix, below-spinodal queries propagate NaN through the entire derivation
chain; above-spinodal queries inside the documented validity range remain finite.
"""
import numpy as np
import pytest
from watereos import getProp

# Points well below the liquid-vapor spinodal (outside the documented validity
# range of the respective EoS). These are stress-test points — the model is
# undefined here and must return NaN.
CAUPIN_BELOW_SPINODAL = [
    (-300.0, 290.0),   # well into the unphysical region
    (-500.0, 250.0),
    (-1000.0, 220.0),
]
DUSKA_BELOW_SPINODAL = [
    (-400.0, 250.0),   # just below Duska's spinodal Ps(250) ~ -319 MPa
    (-600.0, 250.0),
    (-2000.0, 220.0),
]

# Points just above the spinodal but at negative pressure: model should still
# give finite, physical (positive) density.
CAUPIN_ABOVE_SPINODAL = [
    (0.1,    273.15),
    (-100.0, 250.0),
    (50.0,   230.0),
]
DUSKA_ABOVE_SPINODAL = [
    (0.1,    273.15),
    (-200.0, 250.0),
    (-100.0, 300.0),
]


@pytest.mark.parametrize("P,T", CAUPIN_BELOW_SPINODAL)
def test_caupin_below_spinodal_is_nan(P, T):
    PT = np.array([[P], [T]], dtype=object)
    out = getProp(PT, "caupin2019")
    rho = float(np.ravel(out.rho)[0])
    assert np.isnan(rho), f"expected NaN below spinodal at (P={P},T={T}), got {rho}"


@pytest.mark.parametrize("P,T", DUSKA_BELOW_SPINODAL)
def test_duska_below_spinodal_is_nan(P, T):
    PT = np.array([[P], [T]], dtype=object)
    out = getProp(PT, "duska2020")
    rho = float(np.ravel(out.rho)[0])
    assert np.isnan(rho), f"expected NaN below spinodal at (P={P},T={T}), got {rho}"


@pytest.mark.parametrize("P,T", CAUPIN_ABOVE_SPINODAL)
def test_caupin_above_spinodal_is_finite(P, T):
    PT = np.array([[P], [T]], dtype=object)
    out = getProp(PT, "caupin2019")
    rho = float(np.ravel(out.rho)[0])
    assert np.isfinite(rho) and rho > 0, f"unexpected non-physical rho at (P={P},T={T}): {rho}"


@pytest.mark.parametrize("P,T", DUSKA_ABOVE_SPINODAL)
def test_duska_above_spinodal_is_finite(P, T):
    PT = np.array([[P], [T]], dtype=object)
    out = getProp(PT, "duska2020")
    rho = float(np.ravel(out.rho)[0])
    assert np.isfinite(rho) and rho > 0, f"unexpected non-physical rho at (P={P},T={T}): {rho}"


def test_caupin_kim_below_spinodal_is_nan():
    """The with-Kim variant shares the same spinodal model; should also NaN."""
    PT = np.array([[-500.0], [250.0]], dtype=object)
    out = getProp(PT, "caupin2019_kim")
    rho = float(np.ravel(out.rho)[0])
    assert np.isnan(rho)
