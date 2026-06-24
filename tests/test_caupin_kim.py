"""
Tests for the Caupin (2019) with-Kim (Table II) variant: 'caupin2019_kim'.

Covers:
  * Rust vs numpy backend agreement for the Table II parameter set.
  * Reproduction of the paper's published signatures that distinguish the
    with-Kim fit from the without-Kim fit (Table II vs Table III):
      - LLCP at 219.47 K / 58.74 MPa (rho_c = 961.12 kg/m^3)
      - a sharp isothermal-compressibility peak (~1.0 GPa^-1 near ~226-231 K
        at saturated vapor pressure), which the without-Kim fit lacks.
  * IAPWS-95 reference-state alignment at (273.15 K, 0.1 MPa).
  * The unified API exposes the model and its phase diagram.
"""

import numpy as np
import pytest

from caupin_eos import core
from caupin_eos import params_kim as PK


# ── Rust vs numpy parity ───────────────────────────────────────────────────

def test_rust_matches_numpy_kim():
    """Rust compute_batch_caupin_kim must match the numpy Table II core."""
    wrs = pytest.importorskip("watereos_rs")
    T = np.linspace(205.0, 295.0, 60)
    P = np.linspace(-100.0, 350.0, 60)
    rust = wrs.compute_batch_caupin_kim(np.ascontiguousarray(T), np.ascontiguousarray(P))
    npy = core.compute_batch(T, P, pset=PK)
    for key in ("rho", "S", "G", "H", "Cp", "Cv", "Kt", "Ks", "alpha", "vel", "x"):
        a = np.asarray(rust[key]); b = np.asarray(npy[key])
        mask = np.isfinite(a) & np.isfinite(b)
        assert mask.any(), f"{key}: no finite values"
        rel = np.abs(a[mask] - b[mask]) / (np.abs(b[mask]) + 1e-12)
        assert rel.max() < 1e-10, f"{key}: Rust/numpy disagree, max rel={rel.max():.2e}"


def test_kim_differs_from_default():
    """The with-Kim fit must not be numerically identical to the default."""
    T = np.full(5, 250.0)
    P = np.linspace(0.1, 200.0, 5)
    kim = core.compute_batch(T, P, pset=PK)
    nokim = core.compute_batch(T, P)  # default Table III
    # The two fits have different LLCPs, so the LDL fraction x differs by a
    # clearly resolvable margin (rho agrees better since both fit rho data).
    assert np.max(np.abs(kim["x"] - nokim["x"])) > 0.02


# ── Paper signatures ───────────────────────────────────────────────────────

def test_llcp_location():
    """Table II LLCP: 219.47 K, 58.74 MPa, rho_c = 961.12 kg/m^3."""
    from caupin_eos.phase_diagram import find_LLCP
    llcp = find_LLCP(PK)
    assert llcp["T_K"] == pytest.approx(219.47, abs=0.05)
    assert llcp["p_MPa"] == pytest.approx(58.74, abs=0.05)
    rho_c = PK.M_H2O / PK.Vc
    assert rho_c == pytest.approx(961.12, abs=0.1)


def test_compressibility_peak_signature():
    """With-Kim fit shows a sharp kT peak (~1 GPa^-1) near ~226-231 K at svp;
    the without-Kim fit is markedly smaller and smoother there."""
    T = np.linspace(200.0, 300.0, 401)
    P = np.full_like(T, 0.1)
    kim = core.compute_batch(T, P, pset=PK)
    kt_GPa = 1.0 / kim["Kt"] * 1e3  # Kt is in MPa -> kappa_T in 1/GPa
    i = int(np.nanargmax(kt_GPa))
    assert 220.0 < T[i] < 235.0, f"kT peak at unexpected T={T[i]:.1f} K"
    assert kt_GPa[i] > 0.9, f"kT peak too small: {kt_GPa[i]:.3f} GPa^-1"

    nokim = core.compute_batch(T, P)
    kt3 = 1.0 / nokim["Kt"] * 1e3
    # Without-Kim peak is much weaker.
    assert np.nanmax(kt3) < kt_GPa[i]


# ── Reference alignment + unified API ──────────────────────────────────────

def test_iapws_alignment():
    """S, G at (273.15 K, 0.1 MPa) match IAPWS-95 (same target as default)."""
    r = core.compute_batch(np.array([273.15]), np.array([0.1]), pset=PK)
    assert float(r["S"][0]) == pytest.approx(-0.147737, abs=1e-4)
    assert float(r["G"][0]) == pytest.approx(100.017518, abs=1e-2)


def test_unified_api_dispatch():
    from watereos import getProp, list_models
    assert "caupin2019_kim" in list_models()
    PT = np.array([[0.1, 100.0], [250.0, 270.0]], dtype=object)
    out = getProp(PT, "caupin2019_kim")
    assert out.rho.shape == (2, 2)
    assert np.all(np.isfinite(out.rho))
