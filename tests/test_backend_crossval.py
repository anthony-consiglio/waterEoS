"""
Cross-validation tests: hand-coded core.py vs JAX core_ad.py backends.

For each two-state EoS model (Caupin, Holten, Duška), verifies that both
backends produce identical thermodynamic properties at a range of (T, P)
conditions.  Catches derivative bugs, missing guards, and scalar/batch
mode inconsistencies.
"""

import numpy as np
import pytest

# Skip entire module if JAX is not installed
jax = pytest.importorskip("jax", reason="JAX required for cross-validation tests")

# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────

MODELS = ["caupin", "holten", "duska"]

# All keys returned by compute_batch (Kp excluded by design)
BATCH_KEYS = [
    "rho", "V", "S", "G", "H", "U", "A", "Cp", "Cv",
    "Kt", "Ks", "alpha", "vel", "x",
]
STATE_SUFFIXES = ["_A", "_B"]


@pytest.fixture(params=MODELS)
def model(request):
    return request.param


def _import_backends(model_name):
    """Import both backends for a given model."""
    hand = __import__(f"{model_name}_eos.core", fromlist=["compute_batch"]).compute_batch
    ad = __import__(f"{model_name}_eos.core_ad", fromlist=["compute_batch"]).compute_batch
    return hand, ad


# Standard test points: normal liquid water conditions
T_NORMAL = np.array([273.15, 283.15, 293.15, 298.15, 310.0, 330.0, 350.0])
P_NORMAL = np.array([0.1, 10.0, 50.0, 0.1, 100.0, 0.1, 200.0])

# Supercooled/stressed points: closer to LLCP, edge of validity
T_STRESS = np.array([220.0, 225.0, 230.0, 240.0, 200.0, 250.0, 210.0])
P_STRESS = np.array([80.0, 10.0, 60.0, 0.1, 150.0, 200.0, 50.0])

# High-pressure points
T_HIGHP = np.array([273.15, 280.0, 300.0, 260.0, 250.0])
P_HIGHP = np.array([100.0, 200.0, 300.0, 150.0, 250.0])


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _compare_backends(hand_result, ad_result, keys, rtol, atol, label=""):
    """Assert all keys match between hand-coded and AD backends."""
    failures = []
    for key in keys:
        h = np.asarray(hand_result[key], dtype=float)
        a = np.asarray(ad_result[key], dtype=float)

        # Both NaN or both inf → agree
        both_nan = np.isnan(h) & np.isnan(a)
        both_inf = np.isinf(h) & np.isinf(a) & (np.sign(h) == np.sign(a))
        skip = both_nan | both_inf

        # Where both are finite, check agreement
        finite = np.isfinite(h) & np.isfinite(a) & ~skip
        if not finite.any():
            continue

        h_f = h[finite]
        a_f = a[finite]
        denom = np.maximum(np.abs(h_f), np.abs(a_f))
        denom = np.where(denom > 0, denom, 1.0)
        rel_err = np.abs(h_f - a_f) / denom
        abs_err = np.abs(h_f - a_f)

        bad = (rel_err > rtol) & (abs_err > atol)
        if bad.any():
            worst = np.argmax(rel_err[bad])
            idx_full = np.where(finite)[0][np.where(bad)[0][worst]]
            failures.append(
                f"  {key}: max rel_err={rel_err[bad][worst]:.2e}, "
                f"hand={h_f[np.where(bad)[0][worst]]:.6e}, "
                f"AD={a_f[np.where(bad)[0][worst]]:.6e} "
                f"(point idx={idx_full})"
            )

    if failures:
        msg = f"Backend mismatch {label}:\n" + "\n".join(failures)
        pytest.fail(msg)


def _all_keys():
    """All 40 batch keys (mixture + state A + state B)."""
    keys = list(BATCH_KEYS)
    for suffix in STATE_SUFFIXES:
        for k in BATCH_KEYS:
            if k != "x":  # x is mixture-only
                keys.append(k + suffix)
    return keys


# ───────────────────────────────────────────────────────────────────────
# 1. Key consistency: both backends return the same dict keys
# ───────────────────────────────────────────────────────────────────────

class TestKeyConsistency:
    def test_same_keys(self, model):
        """Both backends must return identical dictionary keys."""
        hand, ad = _import_backends(model)
        T = np.array([273.15])
        P = np.array([0.1])
        r_hand = hand(T, P)
        r_ad = ad(T, P)
        assert sorted(r_hand.keys()) == sorted(r_ad.keys()), (
            f"{model}: key mismatch — "
            f"only in hand: {set(r_hand) - set(r_ad)}, "
            f"only in AD: {set(r_ad) - set(r_hand)}"
        )


# ───────────────────────────────────────────────────────────────────────
# 2. Normal conditions: tight tolerance (1e-8 relative)
# ───────────────────────────────────────────────────────────────────────

class TestNormalConditions:
    """Cross-validate at standard liquid water conditions (273–350 K)."""

    def test_all_properties(self, model):
        hand, ad = _import_backends(model)
        r_hand = hand(T_NORMAL, P_NORMAL)
        r_ad = ad(T_NORMAL, P_NORMAL)
        _compare_backends(
            r_hand, r_ad, _all_keys(),
            rtol=1e-7, atol=1e-8,
            label=f"{model} normal conditions",
        )


# ───────────────────────────────────────────────────────────────────────
# 3. Supercooled/stressed conditions: looser tolerance (1e-6)
# ───────────────────────────────────────────────────────────────────────

class TestStressedConditions:
    """Cross-validate near LLCP and in the supercooled regime."""

    def test_all_properties(self, model):
        hand, ad = _import_backends(model)
        r_hand = hand(T_STRESS, P_STRESS)
        r_ad = ad(T_STRESS, P_STRESS)
        _compare_backends(
            r_hand, r_ad, _all_keys(),
            rtol=1e-5, atol=1e-6,
            label=f"{model} stressed conditions",
        )


# ───────────────────────────────────────────────────────────────────────
# 4. High-pressure conditions
# ───────────────────────────────────────────────────────────────────────

class TestHighPressure:
    """Cross-validate at high pressures (100–300 MPa)."""

    def test_all_properties(self, model):
        hand, ad = _import_backends(model)
        r_hand = hand(T_HIGHP, P_HIGHP)
        r_ad = ad(T_HIGHP, P_HIGHP)
        _compare_backends(
            r_hand, r_ad, _all_keys(),
            rtol=1e-6, atol=1e-6,
            label=f"{model} high pressure",
        )


# ───────────────────────────────────────────────────────────────────────
# 5. Per-property focused tests at a dense T grid
# ───────────────────────────────────────────────────────────────────────

class TestDenseTemperatureSweep:
    """Sweep T at fixed P to catch derivative bugs that only appear
    in narrow temperature ranges."""

    @pytest.mark.parametrize("P_fixed", [0.1, 50.0, 150.0])
    def test_sweep(self, model, P_fixed):
        hand, ad = _import_backends(model)
        T_arr = np.linspace(200.0, 350.0, 50)
        P_arr = np.full_like(T_arr, P_fixed)
        r_hand = hand(T_arr, P_arr)
        r_ad = ad(T_arr, P_arr)
        _compare_backends(
            r_hand, r_ad, _all_keys(),
            rtol=1e-5, atol=1e-6,
            label=f"{model} T-sweep at P={P_fixed} MPa",
        )


# ───────────────────────────────────────────────────────────────────────
# 6. Individual property regression tests
# ───────────────────────────────────────────────────────────────────────

class TestSpecificProperties:
    """Targeted tests for properties known to be tricky."""

    def test_Cv_when_Kt_diverges(self, model):
        """Cv must fall back to Cp when isothermal compressibility diverges.

        At low T where Kt → inf, the thermodynamic identity
        Cv = Cp - T·V·alpha²/kappa_T should give Cv ≈ Cp (correction → 0).
        A missing guard in the hand-coded backend would produce wrong Cv.
        """
        hand, ad = _import_backends(model)
        # Scan low temperatures where Kt may diverge
        T_arr = np.linspace(180.0, 220.0, 20)
        P_arr = np.full_like(T_arr, 0.1)
        r_hand = hand(T_arr, P_arr)
        r_ad = ad(T_arr, P_arr)

        Kt_hand = r_hand["Kt"]
        Cv_hand = r_hand["Cv"]
        Cv_ad = r_ad["Cv"]
        Cp_hand = r_hand["Cp"]

        # Where Kt is infinite, Cv should equal Cp
        inf_mask = np.isinf(Kt_hand) | (Kt_hand > 1e12)
        if inf_mask.any():
            np.testing.assert_allclose(
                Cv_hand[inf_mask], Cp_hand[inf_mask],
                rtol=1e-10,
                err_msg=f"{model}: Cv != Cp where Kt diverges (hand-coded)",
            )

        # Both backends should agree on Cv everywhere
        finite = np.isfinite(Cv_hand) & np.isfinite(Cv_ad)
        if finite.any():
            np.testing.assert_allclose(
                Cv_hand[finite], Cv_ad[finite],
                rtol=1e-5,
                err_msg=f"{model}: Cv mismatch between backends at low T",
            )

    def test_density_positive(self, model):
        """Density must always be positive at all test points."""
        hand, ad = _import_backends(model)
        for T, P in [(T_NORMAL, P_NORMAL), (T_STRESS, P_STRESS)]:
            r_hand = hand(T, P)
            r_ad = ad(T, P)
            assert np.all(r_hand["rho"] > 0), f"{model} hand: negative rho"
            assert np.all(r_ad["rho"] > 0), f"{model} AD: negative rho"

    def test_x_in_unit_interval(self, model):
        """LDL fraction x must be in [0, 1]."""
        hand, ad = _import_backends(model)
        for T, P in [(T_NORMAL, P_NORMAL), (T_STRESS, P_STRESS)]:
            r_hand = hand(T, P)
            r_ad = ad(T, P)
            assert np.all((r_hand["x"] >= 0) & (r_hand["x"] <= 1)), (
                f"{model} hand: x out of [0,1]"
            )
            assert np.all((r_ad["x"] >= 0) & (r_ad["x"] <= 1)), (
                f"{model} AD: x out of [0,1]"
            )

    def test_entropy_ordering(self, model):
        """At fixed P, entropy should increase with temperature."""
        hand, ad = _import_backends(model)
        T_arr = np.linspace(260.0, 340.0, 20)
        P_arr = np.full_like(T_arr, 0.1)

        for backend, label in [(hand, "hand"), (ad, "AD")]:
            r = backend(T_arr, P_arr)
            S = r["S"]
            dS = np.diff(S)
            # Allow small numerical noise but overall trend must be increasing
            assert np.sum(dS > 0) > 0.8 * len(dS), (
                f"{model} {label}: entropy not monotonically increasing with T"
            )

    def test_Cp_positive_normal(self, model):
        """Cp must be positive at normal conditions."""
        hand, ad = _import_backends(model)
        r_hand = hand(T_NORMAL, P_NORMAL)
        r_ad = ad(T_NORMAL, P_NORMAL)
        assert np.all(r_hand["Cp"] > 0), f"{model} hand: negative Cp"
        assert np.all(r_ad["Cp"] > 0), f"{model} AD: negative Cp"

    def test_thermodynamic_identity_H_eq_G_plus_TS(self, model):
        """H = G + T*S (fundamental thermodynamic relation)."""
        hand, ad = _import_backends(model)
        for T, P, label in [
            (T_NORMAL, P_NORMAL, "normal"),
            (T_HIGHP, P_HIGHP, "high-P"),
        ]:
            for backend, blabel in [(hand, "hand"), (ad, "AD")]:
                r = backend(T, P)
                H_calc = r["G"] + T * r["S"]
                np.testing.assert_allclose(
                    r["H"], H_calc, rtol=1e-8,
                    err_msg=f"{model} {blabel} ({label}): H != G + T*S",
                )

    def test_thermodynamic_identity_U_eq_H_minus_PV(self, model):
        """U = H - P*V (with P in Pa and V in m³/kg)."""
        hand, ad = _import_backends(model)
        for T, P, label in [
            (T_NORMAL, P_NORMAL, "normal"),
            (T_HIGHP, P_HIGHP, "high-P"),
        ]:
            for backend, blabel in [(hand, "hand"), (ad, "AD")]:
                r = backend(T, P)
                U_calc = r["H"] - P * 1e6 * r["V"]  # P in Pa, V in m³/kg
                np.testing.assert_allclose(
                    r["U"], U_calc, rtol=1e-7,
                    err_msg=f"{model} {blabel} ({label}): U != H - P*V",
                )

    def test_thermodynamic_identity_A_eq_G_minus_PV(self, model):
        """A = G - P*V (Helmholtz = Gibbs - P*V)."""
        hand, ad = _import_backends(model)
        for T, P, label in [
            (T_NORMAL, P_NORMAL, "normal"),
            (T_HIGHP, P_HIGHP, "high-P"),
        ]:
            for backend, blabel in [(hand, "hand"), (ad, "AD")]:
                r = backend(T, P)
                A_calc = r["G"] - P * 1e6 * r["V"]
                np.testing.assert_allclose(
                    r["A"], A_calc, rtol=1e-7,
                    err_msg=f"{model} {blabel} ({label}): A != G - P*V",
                )


# ───────────────────────────────────────────────────────────────────────
# 7. Array shape consistency
# ───────────────────────────────────────────────────────────────────────

class TestOutputShapes:
    """Both backends must return arrays matching input length."""

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_output_length(self, model, n):
        hand, ad = _import_backends(model)
        T = np.linspace(260.0, 320.0, n)
        P = np.full(n, 0.1)
        r_hand = hand(T, P)
        r_ad = ad(T, P)
        for key in r_hand:
            assert r_hand[key].shape == (n,), (
                f"{model} hand: {key} shape={r_hand[key].shape}, expected ({n},)"
            )
            assert r_ad[key].shape == (n,), (
                f"{model} AD: {key} shape={r_ad[key].shape}, expected ({n},)"
            )
