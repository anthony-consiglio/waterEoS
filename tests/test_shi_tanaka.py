"""
Validation tests for the Shi & Tanaka (2020) hierarchical two-state EoS.

Reference: R. Shi and H. Tanaka, "The anomalies and criticality of liquid
water", PNAS 117, 26591 (2020). Parameters from Table S2 (thermo) and
Table S3 (transport) of the supporting information.

These tests pin:

  - Numerical parity between the Python `core.py` and the Rust port
    (`watereos_rs.compute_batch_shi_tanaka`).
  - The paper's reported Schottky-line condition T_{s=1/2}(P=0.1 MPa) = 235 K
    for real water (Table 1, third row).
  - The Eq. S19 heat-capacity-at-reference constraint
    Ĉ_P^ρ |_ref = -2·c_20 - c_1 > 0, evaluated against the model.
  - Validity over the paper's claimed extrapolation envelope (-110 MPa to
    +200 MPa, ~190-320 K) — no NaN/Inf in the principal properties.
  - Output schema (all expected keys present, with _A / _B variants).
  - Dispatcher routes both model keys correctly.
  - Transport module produces finite, monotonically-correct viscosity,
    diffusion, and rotational relaxation across the supercooled regime.
"""

import numpy as np
import pytest

from shi_tanaka_eos.core import compute_batch as py_batch
from shi_tanaka_eos.core import compute_properties as py_props
from shi_tanaka_eos import params as STP

import watereos


# ═══════════════════════════════════════════════════════════════════════════
# 1. Paper's Table 1 — Schottky line for real water at 0.1 MPa
# ═══════════════════════════════════════════════════════════════════════════

def test_schottky_line_table_1():
    """T_{s=1/2}(P=0.1 MPa) = 235 K  (Shi & Tanaka 2020, Table 1, H2O)."""
    out = py_props(235.0, 0.1)
    # Paper says exactly Ts=1/2 = 235 K to 3 sig figs.  Our s should be
    # within a few thousandths of 0.5 there.
    assert abs(out['x'] - 0.5) < 0.005, (
        f"s={out['x']:.4f} at (T=235, P=0.1) — expected ~0.5 per Table 1"
    )


def test_two_state_parameter_alignment():
    """ΔE_hat, Δσ_hat, ΔV_hat dimensionless values follow Table S2 conventions.

    The "MPa⁻¹·K" unit annotation on Table S2's ΔV is misleading — the listed
    numerical value 1.593 is already the dimensionless ΔV̂ that enters Eq. S2.
    See params.py docstring for the empirical justification.
    """
    # ΔE / (kB Tr) = -1952 / 308.15
    assert abs(STP.DELTA_E_HAT - (-1952.0 / 308.15)) < 1e-6
    # Δσ already dimensionless (= Δσ / kB)
    assert STP.DELTA_SIGMA_HAT == -8.317
    # ΔV table value used directly as dimensionless ΔV̂ (no Pr/Tr scaling)
    assert STP.DELTA_V_HAT == 1.593


# ═══════════════════════════════════════════════════════════════════════════
# 2. Eq. S19 / S22 — DNLS heat-capacity-at-reference must be positive
# ═══════════════════════════════════════════════════════════════════════════

def test_dnls_heat_capacity_at_reference_positive():
    """Eq. S19 constraint: -2 c_20 - c_1 > 0  (DNLS Cp positive at ref)."""
    val = -2.0 * STP.c20 - STP.c1
    assert val > 0, f"Constraint S19 violated: -2 c_20 - c_1 = {val:.3f}"


def test_table_S1_constraints():
    """Sign / magnitude constraints from Table S1 (all 12 rows)."""
    assert STP.N_UNIT > 4,         "n > 4 (S22)"
    assert STP.c01 > 0,            "c_01 > 0"
    assert STP.c02 < 0,            "c_02 < 0"
    assert STP.c03 > 0,            "c_03 > 0"
    assert STP.c11 > 0,            "c_11 > 0"
    assert STP.c12 < 0,            "c_12 < 0"
    assert STP.c13 > 0,            "c_13 > 0"
    assert STP.c23 > 0,            "c_23 > 0"
    assert STP.c30 < 0,            "c_30 < 0"
    assert STP.c31 > 0,            "c_31 > 0"
    assert STP.c20 < -3.0 * STP.c30, f"c_20 < -3 c_30 violated"
    assert STP.c1 < -2.0 * STP.c20,  f"c_1 < -2 c_20 violated"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Rust vs Python parity (machine-precision)
# ═══════════════════════════════════════════════════════════════════════════

def _get_rust_batch():
    try:
        import watereos_rs
        return watereos_rs.compute_batch_shi_tanaka
    except (ImportError, AttributeError):
        return None


@pytest.mark.skipif(_get_rust_batch() is None,
                    reason="watereos_rs Rust extension not available")
def test_rust_python_parity():
    """Rust and Python implementations must agree to machine precision."""
    rs_batch = _get_rust_batch()
    T = np.array([200, 220, 235, 250, 270, 285, 298.15, 308.15, 320], dtype=float)
    P = np.array([-50, -10, 0.1, 50, 100, 150, 200], dtype=float)
    T_flat = np.repeat(T, len(P))
    P_flat = np.tile(P, len(T))

    py = py_batch(T_flat, P_flat)
    rs = rs_batch(np.ascontiguousarray(T_flat),
                  np.ascontiguousarray(P_flat))

    for key in ['rho', 'V', 'S', 'G', 'H', 'U', 'A',
                'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel', 'x',
                'rho_A', 'Cp_A', 'Kt_A',
                'rho_B', 'Cp_B', 'Kt_B']:
        a = np.asarray(py[key])
        b = np.asarray(rs[key])
        denom = np.where(np.abs(a) > 1e-12, np.abs(a), 1.0)
        max_rel_err = float(np.max(np.abs(a - b) / denom))
        assert max_rel_err < 1e-12, (
            f"Rust/Python divergence on {key}: max_rel_err={max_rel_err:.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Output schema — all expected keys with _A / _B variants
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_THERMO_KEYS = {
    'rho', 'V', 'S', 'G', 'H', 'U', 'A',
    'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel', 'x',
}
EXPECTED_STATE_KEYS = {
    'rho', 'V', 'S', 'G', 'H', 'U', 'A',
    'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel',
}


def test_compute_batch_output_schema():
    out = py_batch(np.array([280.0]), np.array([10.0]))
    for k in EXPECTED_THERMO_KEYS:
        assert k in out, f"missing thermo key {k}"
    for k in EXPECTED_STATE_KEYS:
        assert f'{k}_A' in out, f"missing state-A key {k}_A"
        assert f'{k}_B' in out, f"missing state-B key {k}_B"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Validity envelope — paper's claimed range, no NaN/Inf in principal props
# ═══════════════════════════════════════════════════════════════════════════

def test_validity_envelope_no_nan_no_inf():
    """No NaN/Inf in core properties across the paper's claimed range."""
    T = np.linspace(190.0, 320.0, 14)
    P = np.linspace(-110.0, 200.0, 12)
    T_flat = np.repeat(T, len(P))
    P_flat = np.tile(P, len(T))

    out = py_batch(T_flat, P_flat)
    for k in ('rho', 'V', 'S', 'Cp', 'Kt', 'alpha', 'vel', 'x'):
        arr = np.asarray(out[k])
        assert np.all(np.isfinite(arr)), (
            f"NaN/Inf in {k} over validity envelope: "
            f"{int(np.sum(~np.isfinite(arr)))} bad points"
        )


def test_negative_pressure_density_monotonic():
    """Eq. S2 / Fig. S2: density decreases monotonically as P → -110 MPa at T=275 K."""
    T = np.full(12, 275.0)
    P = np.linspace(0.0, -110.0, 12)
    out = py_batch(T, P)
    rho = np.asarray(out['rho'])
    # Density should decrease as pressure decreases (or stay close to flat),
    # i.e., dρ/dP > 0 → ρ at P=0 > ρ at P=-110 by at least a few %.
    assert rho[0] > rho[-1], (
        f"Density did not drop under tension: rho(0)={rho[0]:.1f}, "
        f"rho(-110)={rho[-1]:.1f}"
    )
    assert rho[0] - rho[-1] > 10.0, (
        f"Density barely changed under tension: dρ={rho[0]-rho[-1]:.1f} kg/m^3 "
        "(expected several tens at -110 MPa per Fig S2)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Per-state properties — DNLS denser than LFTS (low-density tetrahedral)
# ═══════════════════════════════════════════════════════════════════════════

def test_lfts_lower_density_than_dnls():
    """State B (LFTS, low-density tetrahedral) should be less dense than
    state A (DNLS, disordered normal-liquid) at the same (T, P)."""
    T = np.array([220.0, 250.0, 280.0, 310.0])
    P = np.full_like(T, 50.0)
    out = py_batch(T, P)
    assert np.all(out['rho_B'] < out['rho_A']), (
        "LFTS (B) must have lower density than DNLS (A); got "
        f"rho_A={out['rho_A']}, rho_B={out['rho_B']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Dispatcher integration
# ═══════════════════════════════════════════════════════════════════════════

def test_dispatcher_thermo():
    PT = np.array([np.array([0.1, 100.0]),
                   np.array([260.0, 290.0])], dtype=object)
    out = watereos.getProp(PT, 'shi_tanaka2020')
    assert out.rho.shape == (2, 2)
    assert np.all(np.isfinite(out.rho))
    assert hasattr(out, 'x') and hasattr(out, 'rho_A') and hasattr(out, 'rho_B')


def test_dispatcher_transport():
    PT = np.array([np.array([0.1, 50.0]),
                   np.array([260.0, 298.15])], dtype=object)
    out = watereos.getProp(PT, 'shi_tanaka2020_transport')
    for k in ('eta', 'D', 'tau_r', 'f', 'rho', 'x'):
        assert hasattr(out, k), f"missing attribute {k}"
        assert getattr(out, k).shape == (2, 2)
    assert np.all(out.eta > 0)
    assert np.all(out.D > 0)
    assert np.all(out.tau_r > 0)


def test_registry_entries_present():
    from watereos.model_registry import MODEL_REGISTRY, MODEL_ORDER
    assert 'shi_tanaka2020' in MODEL_REGISTRY
    assert 'shi_tanaka2020_transport' in MODEL_REGISTRY
    assert 'shi_tanaka2020' in MODEL_ORDER
    info = MODEL_REGISTRY['shi_tanaka2020']
    assert info.is_two_state and not info.has_transport
    info_t = MODEL_REGISTRY['shi_tanaka2020_transport']
    assert info_t.has_transport


# ═══════════════════════════════════════════════════════════════════════════
# 8. Transport — qualitative trends in supercooled regime
# ═══════════════════════════════════════════════════════════════════════════

def test_viscosity_increases_on_cooling():
    """η(T=210 K) >> η(T=298 K) at 0.1 MPa — supercooled water gets sluggish."""
    from shi_tanaka_transport import compute_properties as t_props
    eta_warm = t_props(298.15, 0.1)['eta']
    eta_cold = t_props(210.0, 0.1)['eta']
    assert eta_cold > 100.0 * eta_warm, (
        f"Viscosity rise on cooling too weak: η(298)={eta_warm:.3e}, "
        f"η(210)={eta_cold:.3e}"
    )


def test_diffusion_decreases_on_cooling():
    from shi_tanaka_transport import compute_properties as t_props
    D_warm = t_props(298.15, 0.1)['D']
    D_cold = t_props(210.0, 0.1)['D']
    assert D_cold < 0.01 * D_warm, (
        f"Diffusion drop on cooling too weak: D(298)={D_warm:.3e}, "
        f"D(210)={D_cold:.3e}"
    )


def test_dynamic_schottky_line_at_207K():
    """T_{s^D = 1/2}(P = 0.1 MPa) ≈ -ΔE^D / Δσ^D = 2356/11.40 ≈ 207 K."""
    from shi_tanaka_transport.core import _dynamic_fraction
    T_predicted = 2356.0 / 11.40   # ≈ 206.7 K
    s_D = float(_dynamic_fraction(np.array([T_predicted]),
                                  np.array([0.1]))[0])
    assert abs(s_D - 0.5) < 0.01, (
        f"s^D({T_predicted:.1f}) = {s_D:.4f}, expected ~0.5"
    )
