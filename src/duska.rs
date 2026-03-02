//! Duška (2020) EOS-VaT two-state EoS — Rust implementation.
//!
//! Direct port of `duska_eos/core.py:compute_batch` with per-point loop
//! fusion, 16-point Gauss-Legendre quadrature, and finite-difference dSA/dTh.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════════════════
// Constants from duska_eos/params.py
// ═══════════════════════════════════════════════════════════════════════════

const T_VLCP: f64 = 647.096;
const P_VLCP: f64 = 22.064;
const R_SPEC: f64 = 461.523;

// DeltaG coefficients (a0..a6)
const A_DG: [f64; 7] = [
    -4.3743227e-1, -1.3836753e-2, 1.8525106e-2, 4.3306058e-1,
    2.1944047e+0, -1.6301740e-5, 7.6204693e-6,
];

// phi coefficients (b0..b4)
const B_PHI: [f64; 5] = [
    2.6732998e+1, -1.0405443e+2, 2.1364435e+2, -2.3582144e+2, 1.0783316e+2,
];

// VS coefficients (c0..c4)
const C_VS: [f64; 5] = [
    7.3009898e-2, -8.9096098e-3, 5.7261662e-2, -1.3084560e-2, 7.7905108e-3,
];

// pS coefficients (d1..d3)
const D_PS: [f64; 3] = [1.2756957e+1, -2.6960321e+0, 2.8548221e+1];

// omega coefficients (w0..w3)
const W_OM: [f64; 4] = [4.1420925e-1, 3.6615174e-2, 1.6181775e+0, 7.1477190e-3];

// entropy coefficients (s0..s3)
const S_ENT: [f64; 4] = [-6.3674996e+0, 8.7732559e+1, -1.7214704e+2, 1.1210116e+2];

const S_OFFSET: f64 = -13463.3332601599;
const H_OFFSET: f64 = -9540603.2122842800;

// 16-point Gauss-Legendre nodes and weights on [-1, 1]
const GL_N: usize = 16;
const GL_NODES: [f64; GL_N] = [
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572274,
     0.6178762444026438,  0.7554044083550030,  0.8656312023878318,
     0.9445750230732326,  0.9894009349916499,
];
const GL_WEIGHTS: [f64; GL_N] = [
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
    0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
    0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
    0.1495959888165767, 0.1246289712555339, 0.0951585116824928,
    0.0622535239386479, 0.0271524594117541,
];

// ═══════════════════════════════════════════════════════════════════════════
// Spinodal polynomials
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn spinodal_props(th: f64) -> (f64, f64, f64, f64, f64, f64) {
    let dth = th - 1.0;
    let dth2 = dth * dth;
    let dth3 = dth2 * dth;
    let th2 = th * th;
    let th3 = th2 * th;
    let th4 = th3 * th;

    let ps = 1.0 + D_PS[0]*dth + D_PS[1]*dth2 + D_PS[2]*dth3;
    let dps = D_PS[0] + 2.0*D_PS[1]*dth + 3.0*D_PS[2]*dth2;

    let vs = C_VS[0] + C_VS[1]*th + C_VS[2]*th2 + C_VS[3]*th3 + C_VS[4]*th4;
    let dvs = C_VS[1] + 2.0*C_VS[2]*th + 3.0*C_VS[3]*th2 + 4.0*C_VS[4]*th3;

    let phi = B_PHI[0] + B_PHI[1]*th + B_PHI[2]*th2 + B_PHI[3]*th3 + B_PHI[4]*th4;
    let dphi = B_PHI[1] + 2.0*B_PHI[2]*th + 3.0*B_PHI[3]*th2 + 4.0*B_PHI[4]*th3;

    (ps, vs, phi, dps, dvs, dphi)
}

// ═══════════════════════════════════════════════════════════════════════════
// B parameter
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn compute_b(ps: f64, vs: f64, phi: f64, dps: f64, dvs: f64, dphi: f64) -> (f64, f64) {
    let arg = -phi / (2.0 * ps * vs * vs);
    let b = arg.abs().sqrt();

    let num = -dphi * ps * vs * vs
              + phi * dps * vs * vs
              + 2.0 * phi * ps * vs * dvs;
    let denom = 2.0 * ps * ps * vs * vs * vs * vs;
    let darg = num / denom;
    let db = if b > 0.0 { darg / (2.0 * b) } else { 0.0 };

    (b, db)
}

// ═══════════════════════════════════════════════════════════════════════════
// Volume of state A
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn compute_va(ph: f64, ps: f64, vs: f64, b: f64) -> (f64, f64, f64) {
    let u_arg = (1.0 - ph / ps).max(0.0);
    let u = u_arg.sqrt();
    let denom = u + b;
    let va = vs * b / denom;
    let dva_dp = if u > 1e-30 {
        vs * b / (2.0 * ps * denom * denom * u)
    } else { f64::INFINITY };
    (va, dva_dp, u)
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropy of state A
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn compute_ss(th: f64) -> f64 {
    S_ENT[0] * th.ln()
        + S_ENT[1] * th
        + S_ENT[2] / 2.0 * th * th
        + S_ENT[3] / 3.0 * th * th * th
}

#[inline(always)]
fn compute_sa(va: f64, vs: f64, th: f64, dps: f64, b: f64, db: f64,
              ps: f64, dvs: f64, _dphi: f64) -> f64 {
    let a_coeff = dps * b * b + 2.0 * ps * b * db;
    let c_coeff = 2.0 * ps * b * b * dvs;
    let ss = compute_ss(th);
    let r = va / vs;
    let ln_r = if r > 0.0 { r.ln() } else { 0.0 };

    dps * (va - vs)
        + a_coeff * (vs * vs / va + 2.0 * vs * ln_r - va)
        + c_coeff * (vs / va + ln_r - 1.0)
        + ss
}

/// Evaluate VA and SA at given (ph, th) — used for finite-difference dSA/dTh.
#[inline(always)]
fn state_a_at(ph: f64, th: f64) -> (f64, f64) {
    let (ps, vs, phi, dps, dvs, dphi) = spinodal_props(th);
    let (b, db) = compute_b(ps, vs, phi, dps, dvs, dphi);
    let (va, _, _) = compute_va(ph, ps, vs, b);
    let sa = compute_sa(va, vs, th, dps, b, db, ps, dvs, dphi);
    (va, sa)
}

// ═══════════════════════════════════════════════════════════════════════════
// DeltaG and omega
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn compute_delta_g(ph: f64, th: f64) -> (f64, f64, f64, f64, f64, f64) {
    let dg = A_DG[0] + A_DG[1]*ph*th + A_DG[2]*ph + A_DG[3]*th
             + A_DG[4]*th*th + A_DG[5]*ph*ph + A_DG[6]*ph*ph*ph;
    let ddg_dp = A_DG[1]*th + A_DG[2] + 2.0*A_DG[5]*ph + 3.0*A_DG[6]*ph*ph;
    let ddg_dt = A_DG[1]*ph + A_DG[3] + 2.0*A_DG[4]*th;
    let d2dg_dp2 = 2.0*A_DG[5] + 6.0*A_DG[6]*ph;
    let d2dg_dt2 = 2.0*A_DG[4];
    let d2dg_dpdt = A_DG[1];
    (dg, ddg_dp, ddg_dt, d2dg_dp2, d2dg_dt2, d2dg_dpdt)
}

#[inline(always)]
fn compute_omega(ph: f64, th: f64) -> (f64, f64, f64) {
    let om = W_OM[0] * (1.0 + W_OM[1]*ph + W_OM[2]*th + W_OM[3]*th*ph);
    let dom_dp = W_OM[0] * (W_OM[1] + W_OM[3]*th);
    let dom_dt = W_OM[0] * (W_OM[2] + W_OM[3]*ph);
    (om, dom_dp, dom_dt)
}

// ═══════════════════════════════════════════════════════════════════════════
// Newton solver (two starts, pick lower g)
// ═══════════════════════════════════════════════════════════════════════════

const NEWTON_EPS: f64 = 1e-15;

#[inline(always)]
fn newton_solve(dg: f64, th: f64, om: f64, x0: f64) -> f64 {
    let mut x = x0;
    for _ in 0..50 {
        x = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
        let lnrat = (x / (1.0 - x)).ln();
        let f_val = dg + th * lnrat + om * (1.0 - 2.0 * x);
        let fx = th / (x * (1.0 - x)) - 2.0 * om;
        if fx.abs() < 1e-30 { break; }
        let dx = -f_val / fx;
        if x + dx < NEWTON_EPS {
            x /= 2.0;
        } else if x + dx > 1.0 - NEWTON_EPS {
            x = (x + 1.0 - NEWTON_EPS) / 2.0;
        } else {
            x += dx;
        }
        if f_val.abs() < 1e-12 { break; }
    }
    x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS)
}

#[inline(always)]
fn solve_x_stable(dg: f64, th: f64, om: f64) -> f64 {
    let x_lo = newton_solve(dg, th, om, 0.05);
    let x_hi = newton_solve(dg, th, om, 0.95);

    let g_mix = |x: f64| -> f64 {
        let xc = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
        let me = xc * xc.ln() + (1.0 - xc) * (1.0 - xc).ln();
        xc * dg + th * me + om * xc * (1.0 - xc)
    };

    if g_mix(x_lo) <= g_mix(x_hi) { x_lo } else { x_hi }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gibbs energy of state A via 2-leg GL quadrature
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn compute_g_a_integral(ph: f64, th: f64) -> f64 {
    // Leg 1: T-integral from Th=1 to Th at ph=0
    let mut g_t_leg = 0.0;
    let dth_val = th - 1.0;
    if dth_val.abs() > 1e-30 {
        let half_dth = dth_val / 2.0;
        for j in 0..GL_N {
            let th_i = half_dth * (GL_NODES[j] + 1.0) + 1.0;
            let (_, sa_i) = state_a_at(0.0, th_i);
            g_t_leg += GL_WEIGHTS[j] * (-sa_i);
        }
        g_t_leg *= half_dth;
    }

    // Leg 2: P-integral from ph=0 to ph at fixed Th
    let mut g_p_leg = 0.0;
    if ph.abs() > 1e-30 {
        let (ps, vs, phi, dps, dvs, dphi) = spinodal_props(th);
        let (b, _db) = compute_b(ps, vs, phi, dps, dvs, dphi);
        let half_ph = ph / 2.0;
        for j in 0..GL_N {
            let ph_i = half_ph * (GL_NODES[j] + 1.0);
            let (va_i, _, _) = compute_va(ph_i, ps, vs, b);
            g_p_leg += GL_WEIGHTS[j] * va_i;
        }
        g_p_leg *= half_ph;
    }

    g_t_leg + g_p_leg
}

// ═══════════════════════════════════════════════════════════════════════════
// Physical property conversion (reduced → SI, per-kg units)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn to_physical(
    vh: f64, sh: f64, dvh_dp: f64, dvh_dt: f64, dsh_dt: f64,
    th: f64, _ph: f64, g_red: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let t = th * T_VLCP;
    let v_phys = vh * R_SPEC * T_VLCP / (P_VLCP * 1e6);
    let rho = if v_phys > 0.0 { 1.0 / v_phys } else { f64::INFINITY };
    let s_phys = sh * R_SPEC;
    let cp = R_SPEC * th * dsh_dt;

    let kappa_t_red = if dvh_dp.abs() > 1e-30 {
        -(1.0 / vh) * dvh_dp
    } else { f64::INFINITY };
    let kappa_t = kappa_t_red / (P_VLCP * 1e6);
    let kt = if kappa_t.abs() > 1e-30 && kappa_t.is_finite() {
        1.0 / kappa_t / 1e6
    } else { 0.0 };

    let alpha = if vh > 0.0 { (1.0 / vh) * dvh_dt / T_VLCP } else { 0.0 };

    let cv = if kappa_t > 0.0 && kappa_t.is_finite() {
        cp - t * v_phys * alpha * alpha / kappa_t
    } else { cp };

    let kappa_s = if cp > 0.0 {
        kappa_t - t * v_phys * alpha * alpha / cp
    } else { kappa_t };

    let ks = if kappa_s > 0.0 { 1.0 / kappa_s / 1e6 } else { f64::INFINITY };
    let vel = if rho > 0.0 && kappa_s > 0.0 {
        (1.0 / (rho * kappa_s)).sqrt()
    } else { f64::NAN };

    let g_val = R_SPEC * T_VLCP * g_red;

    (rho, v_phys, s_phys, g_val, cp, cv, kt, ks, alpha, vel)
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-point compute
// ═══════════════════════════════════════════════════════════════════════════

struct PointResult {
    rho: f64, v: f64, s: f64, g: f64, h: f64, u: f64, a: f64,
    cp: f64, cv: f64, kt: f64, ks: f64, alpha: f64, vel: f64, x: f64,
    rho_a: f64, v_a: f64, s_a: f64, g_a: f64, h_a: f64, u_a: f64, a_a: f64,
    cp_a: f64, cv_a: f64, kt_a: f64, ks_a: f64, alpha_a: f64, vel_a: f64,
    rho_b: f64, v_b: f64, s_b: f64, g_b: f64, h_b: f64, u_b: f64, a_b: f64,
    cp_b: f64, cv_b: f64, kt_b: f64, ks_b: f64, alpha_b: f64, vel_b: f64,
}

#[inline]
fn compute_point(t_k: f64, p_mpa: f64) -> PointResult {
    let th = t_k / T_VLCP;
    let ph = p_mpa / P_VLCP;

    // Spinodal
    let (ps, vs, phi, dps, dvs, dphi) = spinodal_props(th);
    let (b, db) = compute_b(ps, vs, phi, dps, dvs, dphi);
    let (va, dva_dp, u) = compute_va(ph, ps, vs, b);
    let sa = compute_sa(va, vs, th, dps, b, db, ps, dvs, dphi);

    // DeltaG and omega
    let (dg, ddg_dp, ddg_dt, d2dg_dp2, d2dg_dt2, d2dg_dpdt) = compute_delta_g(ph, th);
    let (om, dom_dp, dom_dt) = compute_omega(ph, th);

    // Solve for x
    let x = solve_x_stable(dg, th, om);

    // ── Mixture volume and entropy ───────────────────────────────────
    let delta_v = ddg_dp;
    let vh = va + x * delta_v + dom_dp * x * (1.0 - x);

    let x_c = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
    let mix_entropy = x_c * x_c.ln() + (1.0 - x_c) * (1.0 - x_c).ln();
    let sh = sa - x * ddg_dt - mix_entropy - dom_dt * x * (1.0 - x);

    // ── dx/dTh and dx/dph ────────────────────────────────────────────
    let fx = th / (x_c * (1.0 - x_c)) - 2.0 * om;
    let lnrat = (x_c / (1.0 - x_c)).ln();
    let f_th = ddg_dt + lnrat + dom_dt * (1.0 - 2.0 * x);
    let f_ph = delta_v + dom_dp * (1.0 - 2.0 * x);
    let fx_safe = if fx.abs() > 1e-30 { fx } else { 1e30 };
    let dx_dth = -f_th / fx_safe;
    let dx_dph = -f_ph / fx_safe;

    // ── dVh/dph ──────────────────────────────────────────────────────
    let dvh_dp_x = dva_dp + x * d2dg_dp2;
    let dvhdx = delta_v + dom_dp * (1.0 - 2.0 * x);
    let dvh_dp = dvh_dp_x + dvhdx * dx_dph;

    // ── Analytical dVA/dTh ───────────────────────────────────────────
    let denom_va = u + b;
    let dva_dth = if u > 1e-30 {
        let du_dth = ph * dps / (2.0 * ps * ps * u);
        let dn_dth = dvs * b + vs * db;
        let dd_dth = du_dth + db;
        (dn_dth * denom_va - vs * b * dd_dth) / (denom_va * denom_va)
    } else { 0.0 };

    // ── dVh/dTh ──────────────────────────────────────────────────────
    let dvh_dt_x = dva_dth + x * d2dg_dpdt + W_OM[0] * W_OM[3] * x * (1.0 - x);
    let dvh_dt = dvh_dt_x + dvhdx * dx_dth;

    // ── dSA/dTh via central finite difference ────────────────────────
    let delta = 1e-7;
    let (_, sa_p) = state_a_at(ph, th + delta);
    let (_, sa_m) = state_a_at(ph, th - delta);
    let dsa_dth = (sa_p - sa_m) / (2.0 * delta);

    // ── dSh/dTh ──────────────────────────────────────────────────────
    let dsh_dt_x = dsa_dth - x * d2dg_dt2;
    let dshdx = if x > NEWTON_EPS && x < 1.0 - NEWTON_EPS {
        -ddg_dt + ((1.0 - x_c) / x_c).ln() - dom_dt * (1.0 - 2.0 * x)
    } else { 0.0 };
    let dsh_dt = dsh_dt_x + dshdx * dx_dth;

    // ── Gibbs energy via GL integral ─────────────────────────────────
    let g_a = compute_g_a_integral(ph, th);
    let g_mix = g_a + x * dg + th * mix_entropy + om * x * (1.0 - x);
    let g_b = g_a + dg;

    // ── Mixture physical properties ──────────────────────────────────
    let (rho, v_spec, s_phys, mut g_phys, cp, cv, kt, ks, alpha, vel) =
        to_physical(vh, sh, dvh_dp, dvh_dt, dsh_dt, th, ph, g_mix);

    // ── State A properties ───────────────────────────────────────────
    let (rho_a, v_a, s_a_phys, mut g_a_phys, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a) =
        to_physical(va, sa, dva_dp, dva_dth, dsa_dth, th, ph, g_a);

    // ── State B properties ───────────────────────────────────────────
    let vb = va + delta_v;
    let sb = sa - ddg_dt;
    let dvb_dp = dva_dp + d2dg_dp2;
    let dvb_dth = dva_dth + d2dg_dpdt;
    let dsb_dth = dsa_dth - d2dg_dt2;
    let (rho_b, v_b, s_b_phys, mut g_b_phys, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b) =
        to_physical(vb, sb, dvb_dp, dvb_dth, dsb_dth, th, ph, g_b);

    // ── IAPWS-95 reference state alignment ───────────────────────────
    let s_val = s_phys + S_OFFSET;
    g_phys += H_OFFSET - t_k * S_OFFSET;
    let s_a_val = s_a_phys + S_OFFSET;
    g_a_phys += H_OFFSET - t_k * S_OFFSET;
    let s_b_val = s_b_phys + S_OFFSET;
    g_b_phys += H_OFFSET - t_k * S_OFFSET;

    // ── Derived potentials ───────────────────────────────────────────
    let p_pa = p_mpa * 1e6;
    let h = g_phys + t_k * s_val;
    let u_pot = h - p_pa * v_spec;
    let a_pot = g_phys - p_pa * v_spec;

    let h_a = g_a_phys + t_k * s_a_val;
    let u_a = h_a - p_pa * v_a;
    let a_a = g_a_phys - p_pa * v_a;

    let h_b = g_b_phys + t_k * s_b_val;
    let u_b = h_b - p_pa * v_b;
    let a_b = g_b_phys - p_pa * v_b;

    PointResult {
        rho, v: v_spec, s: s_val, g: g_phys, h, u: u_pot, a: a_pot,
        cp, cv, kt, ks, alpha, vel, x,
        rho_a, v_a, s_a: s_a_val, g_a: g_a_phys, h_a, u_a, a_a,
        cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a,
        rho_b, v_b, s_b: s_b_val, g_b: g_b_phys, h_b, u_b, a_b,
        cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 entry point
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_batch_duska<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let t_vec: Vec<f64>;
    let p_vec: Vec<f64>;
    let t_data: &[f64] = match t_k.as_slice() {
        Ok(s) => s,
        Err(_) => { t_vec = t_k.to_vec()?; &t_vec }
    };
    let p_data: &[f64] = match p_mpa.as_slice() {
        Ok(s) => s,
        Err(_) => { p_vec = p_mpa.to_vec()?; &p_vec }
    };
    let n = t_data.len();

    let mut rho = Array1::<f64>::zeros(n); let mut v = Array1::<f64>::zeros(n);
    let mut s = Array1::<f64>::zeros(n); let mut g = Array1::<f64>::zeros(n);
    let mut h = Array1::<f64>::zeros(n); let mut u = Array1::<f64>::zeros(n);
    let mut a = Array1::<f64>::zeros(n); let mut cp = Array1::<f64>::zeros(n);
    let mut cv = Array1::<f64>::zeros(n); let mut kt = Array1::<f64>::zeros(n);
    let mut ks = Array1::<f64>::zeros(n); let mut alpha = Array1::<f64>::zeros(n);
    let mut vel = Array1::<f64>::zeros(n); let mut x_out = Array1::<f64>::zeros(n);

    let mut rho_a = Array1::<f64>::zeros(n); let mut v_a = Array1::<f64>::zeros(n);
    let mut s_a = Array1::<f64>::zeros(n); let mut g_a = Array1::<f64>::zeros(n);
    let mut h_a = Array1::<f64>::zeros(n); let mut u_a = Array1::<f64>::zeros(n);
    let mut a_a = Array1::<f64>::zeros(n); let mut cp_a = Array1::<f64>::zeros(n);
    let mut cv_a = Array1::<f64>::zeros(n); let mut kt_a = Array1::<f64>::zeros(n);
    let mut ks_a = Array1::<f64>::zeros(n); let mut alpha_a = Array1::<f64>::zeros(n);
    let mut vel_a = Array1::<f64>::zeros(n);

    let mut rho_b = Array1::<f64>::zeros(n); let mut v_b = Array1::<f64>::zeros(n);
    let mut s_b = Array1::<f64>::zeros(n); let mut g_b = Array1::<f64>::zeros(n);
    let mut h_b = Array1::<f64>::zeros(n); let mut u_b = Array1::<f64>::zeros(n);
    let mut a_b = Array1::<f64>::zeros(n); let mut cp_b = Array1::<f64>::zeros(n);
    let mut cv_b = Array1::<f64>::zeros(n); let mut kt_b = Array1::<f64>::zeros(n);
    let mut ks_b = Array1::<f64>::zeros(n); let mut alpha_b = Array1::<f64>::zeros(n);
    let mut vel_b = Array1::<f64>::zeros(n);

    for i in 0..n {
        let r = compute_point(t_data[i], p_data[i]);
        rho[i] = r.rho; v[i] = r.v; s[i] = r.s; g[i] = r.g;
        h[i] = r.h; u[i] = r.u; a[i] = r.a;
        cp[i] = r.cp; cv[i] = r.cv; kt[i] = r.kt; ks[i] = r.ks;
        alpha[i] = r.alpha; vel[i] = r.vel; x_out[i] = r.x;
        rho_a[i] = r.rho_a; v_a[i] = r.v_a; s_a[i] = r.s_a; g_a[i] = r.g_a;
        h_a[i] = r.h_a; u_a[i] = r.u_a; a_a[i] = r.a_a;
        cp_a[i] = r.cp_a; cv_a[i] = r.cv_a; kt_a[i] = r.kt_a; ks_a[i] = r.ks_a;
        alpha_a[i] = r.alpha_a; vel_a[i] = r.vel_a;
        rho_b[i] = r.rho_b; v_b[i] = r.v_b; s_b[i] = r.s_b; g_b[i] = r.g_b;
        h_b[i] = r.h_b; u_b[i] = r.u_b; a_b[i] = r.a_b;
        cp_b[i] = r.cp_b; cv_b[i] = r.cv_b; kt_b[i] = r.kt_b; ks_b[i] = r.ks_b;
        alpha_b[i] = r.alpha_b; vel_b[i] = r.vel_b;
    }

    let dict = PyDict::new(py);
    dict.set_item("rho", rho.into_pyarray(py))?;
    dict.set_item("V", v.into_pyarray(py))?;
    dict.set_item("S", s.into_pyarray(py))?;
    dict.set_item("G", g.into_pyarray(py))?;
    dict.set_item("H", h.into_pyarray(py))?;
    dict.set_item("U", u.into_pyarray(py))?;
    dict.set_item("A", a.into_pyarray(py))?;
    dict.set_item("Cp", cp.into_pyarray(py))?;
    dict.set_item("Cv", cv.into_pyarray(py))?;
    dict.set_item("Kt", kt.into_pyarray(py))?;
    dict.set_item("Ks", ks.into_pyarray(py))?;
    dict.set_item("alpha", alpha.into_pyarray(py))?;
    dict.set_item("vel", vel.into_pyarray(py))?;
    dict.set_item("x", x_out.into_pyarray(py))?;
    dict.set_item("rho_A", rho_a.into_pyarray(py))?;
    dict.set_item("V_A", v_a.into_pyarray(py))?;
    dict.set_item("S_A", s_a.into_pyarray(py))?;
    dict.set_item("G_A", g_a.into_pyarray(py))?;
    dict.set_item("H_A", h_a.into_pyarray(py))?;
    dict.set_item("U_A", u_a.into_pyarray(py))?;
    dict.set_item("A_A", a_a.into_pyarray(py))?;
    dict.set_item("Cp_A", cp_a.into_pyarray(py))?;
    dict.set_item("Cv_A", cv_a.into_pyarray(py))?;
    dict.set_item("Kt_A", kt_a.into_pyarray(py))?;
    dict.set_item("Ks_A", ks_a.into_pyarray(py))?;
    dict.set_item("alpha_A", alpha_a.into_pyarray(py))?;
    dict.set_item("vel_A", vel_a.into_pyarray(py))?;
    dict.set_item("rho_B", rho_b.into_pyarray(py))?;
    dict.set_item("V_B", v_b.into_pyarray(py))?;
    dict.set_item("S_B", s_b.into_pyarray(py))?;
    dict.set_item("G_B", g_b.into_pyarray(py))?;
    dict.set_item("H_B", h_b.into_pyarray(py))?;
    dict.set_item("U_B", u_b.into_pyarray(py))?;
    dict.set_item("A_B", a_b.into_pyarray(py))?;
    dict.set_item("Cp_B", cp_b.into_pyarray(py))?;
    dict.set_item("Cv_B", cv_b.into_pyarray(py))?;
    dict.set_item("Kt_B", kt_b.into_pyarray(py))?;
    dict.set_item("Ks_B", ks_b.into_pyarray(py))?;
    dict.set_item("alpha_B", alpha_b.into_pyarray(py))?;
    dict.set_item("vel_B", vel_b.into_pyarray(py))?;
    Ok(dict)
}
