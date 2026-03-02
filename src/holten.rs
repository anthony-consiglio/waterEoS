//! Holten, Sengers & Anisimov (2014) two-state EoS — Rust implementation.
//!
//! Direct port of `holten_eos/core.py:compute_batch` with per-point loop
//! fusion and early-exit solvers.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════════════════
// Constants from holten_eos/params.py
// ═══════════════════════════════════════════════════════════════════════════

const R: f64 = 461.523087;          // J/(kg·K)
const TC: f64 = 228.2;             // K
const PC: f64 = 0.0;               // MPa
const RHO0: f64 = 1081.6482;       // kg/m³
const P0: f64 = -300.0;            // MPa

const P_SCALE_PA: f64 = RHO0 * R * TC;  // ~1.1383e8 Pa
const OMEGA0: f64 = 0.52122690;
const L0: f64 = 0.76317954;
const K0: f64 = 0.072158686;
const K1: f64 = -0.31569232;
const K2: f64 = 5.2992608;

const S_OFFSET: f64 = -0.0000995240;
const H_OFFSET: f64 = -0.0272789916;

// Background coefficients (20 terms)
const C_BG: [f64; 20] = [
    -8.1570681381655, 1.2875032e+000, 7.0901673598012,
    -3.2779161e-002, 7.3703949e-001, -2.1628622e-001, -5.1782479e+000,
    4.2293517e-004, 2.3592109e-002, 4.3773754e+000, -2.9967770e-003,
    -9.6558018e-001, 3.7595286e+000, 1.2632441e+000, 2.8542697e-001,
    -8.5994947e-001, -3.2916153e-001, 9.0019616e-002, 8.1149726e-002,
    -3.2788213e+000,
];

const A_BG: [f64; 20] = [
    0.0, 0.0, 1.0, -0.2555, 1.5762, 1.64, 3.6385, -0.3828,
    1.6219, 4.3287, 3.4763, 5.1556, -0.3593, 5.0361, 2.9786, 6.2373,
    4.046, 5.3558, 9.0157, 1.2194,
];

const B_BG: [f64; 20] = [
    0.0, 1.0, 0.0, 2.1051, 1.1422, 0.951, 0.0, 3.6402,
    2.076, -0.0016, 2.2769, 0.0008, 0.3706, -0.3975, 2.973, -0.318,
    2.9805, 2.9265, 0.4456, 0.1298,
];

const D_BG: [f64; 20] = [
    0.0, 0.0, 0.0, -0.0016, 0.6894, 0.013, 0.0002, 0.0435,
    0.05, 0.0004, 0.0528, 0.0147, 0.8584, 0.9924, 1.0041, 1.0961,
    1.0228, 1.0303, 1.618, 0.5213,
];

// ═══════════════════════════════════════════════════════════════════════════
// Background B(tau, pi) and derivatives (20-term sum, Eq. 12)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn b_all(tau: f64, pi: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut b_val = 0.0;
    let mut bp = 0.0;
    let mut bt = 0.0;
    let mut bpp = 0.0;
    let mut btp = 0.0;
    let mut btt = 0.0;
    let inv_pi = 1.0 / pi;
    let inv_tau = 1.0 / tau;

    for i in 0..20 {
        let ci = C_BG[i];
        let ai = A_BG[i];
        let bi = B_BG[i];
        let di = D_BG[i];
        let base = ci * tau.powf(ai) * pi.powf(bi) * (-di * pi).exp();
        let bdp = bi - di * pi;
        b_val += base;
        bp += base * bdp * inv_pi;
        bt += base * ai * inv_tau;
        bpp += base * (bdp * bdp - bi) * inv_pi * inv_pi;
        btp += base * ai * bdp * inv_tau * inv_pi;
        btt += base * ai * (ai - 1.0) * inv_tau * inv_tau;
    }
    (b_val, bp, bt, bpp, btp, btt)
}

// ═══════════════════════════════════════════════════════════════════════════
// Hyperbolic field L and derivatives (Eq. 14)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn compute_l(t: f64, p_red: f64) -> (f64, f64, f64, f64, f64, f64) {
    let arg = p_red - K2 * t;
    let inner = 1.0 + K0 * K2 + K1 * arg;
    let k1_val = (inner * inner - 4.0 * K0 * K1 * K2 * arg).sqrt();
    let k3 = k1_val * k1_val * k1_val;
    let k2_val = (1.0 + K2 * K2).sqrt();

    let l = L0 * k2_val * (1.0 - k1_val + K0 * K2 + K1 * (p_red + K2 * t))
        / (2.0 * K1 * K2);
    let lt = L0 * 0.5 * k2_val
        * (1.0 + (1.0 - K0 * K2 + K1 * (p_red - K2 * t)) / k1_val);
    let lp = L0 * k2_val
        * (k1_val + K0 * K2 - K1 * p_red + K1 * K2 * t - 1.0)
        / (2.0 * K2 * k1_val);
    let ltt = -2.0 * L0 * k2_val * K0 * K1 * K2 * K2 / k3;
    let ltp = 2.0 * L0 * k2_val * K0 * K1 * K2 / k3;
    let lpp = -2.0 * L0 * k2_val * K0 * K1 / k3;

    (l, lt, lp, ltt, ltp, lpp)
}

// ═══════════════════════════════════════════════════════════════════════════
// Equilibrium solver with flip trick + bisection + Newton fallback
// ═══════════════════════════════════════════════════════════════════════════

const EPS: f64 = 1e-15;

#[inline(always)]
fn findxe(l_in: f64, omega: f64) -> f64 {
    let flip = l_in < 0.0;
    let l = if flip { -l_in } else { l_in };

    // Smart bracket selection (from MATLAB)
    let (mut x0, mut x1);
    if omega < 1.1111111 * (2.944439 - l) {
        x0 = 0.049;
        x1 = 0.5;
    } else if omega < 1.0204081 * (4.595119 - l) {
        x0 = 0.0099;
        x1 = 0.051;
    } else {
        x0 = 0.99 * (-1.0204081 * l - omega).exp();
        x1 = 1.01 * 1.087 * (-l - omega).exp();
        if x1 > 0.0101 {
            x1 = 0.0101;
        }
    }

    x0 = x0.max(EPS);
    x1 = x1.min(1.0 - EPS);

    #[inline(always)]
    fn f_eq(x: f64, l: f64, omega: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 { return f64::INFINITY; }
        l + (x / (1.0 - x)).ln() + omega * (1.0 - 2.0 * x)
    }

    let mut f0 = f_eq(x0, l, omega);
    let f1 = f_eq(x1, l, omega);

    // If bracket doesn't straddle, try wider
    let (mut lo, mut hi, mut flo);
    if f0 * f1 > 0.0 {
        x0 = EPS;
        x1 = 0.5;
        f0 = f_eq(x0, l, omega);
        let f1b = f_eq(x1, l, omega);
        if f0 * f1b > 0.0 {
            // Newton fallback from 0.05
            let xe = newton_xe(l, omega, 0.05);
            return if flip { 1.0 - xe } else { xe };
        }
        lo = x0;
        hi = x1;
        flo = f0;
    } else {
        lo = x0;
        hi = x1;
        flo = f0;
    }

    // Bisection
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let fm = f_eq(mid, l, omega);
        if fm * flo < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
        if hi - lo < 1e-14 {
            break;
        }
    }
    let xe = (lo + hi) / 2.0;
    if flip { 1.0 - xe } else { xe }
}

#[inline(always)]
fn newton_xe(l: f64, omega: f64, x0: f64) -> f64 {
    let mut x = x0;
    for _ in 0..200 {
        x = x.max(EPS).min(1.0 - EPS);
        let lnrat = (x / (1.0 - x)).ln();
        let f_val = l + lnrat + omega * (1.0 - 2.0 * x);
        let fx = 1.0 / (x * (1.0 - x)) - 2.0 * omega;
        if fx.abs() < 1e-30 { break; }
        let dx = -f_val / fx;
        if x + dx < EPS {
            x /= 2.0;
        } else if x + dx > 1.0 - EPS {
            x = (x + 1.0 - EPS) / 2.0;
        } else {
            x += dx;
        }
        if f_val.abs() < 1e-13 { break; }
    }
    x
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-point property computation
// ═══════════════════════════════════════════════════════════════════════════

struct PointResult {
    rho: f64, v: f64, s: f64, g: f64, h: f64, u: f64, a: f64,
    cp: f64, cv: f64, kt: f64, ks: f64, alpha: f64, vel: f64, x: f64,
    rho_a: f64, v_a: f64, s_a: f64, g_a: f64, h_a: f64, u_a: f64, a_a: f64,
    cp_a: f64, cv_a: f64, kt_a: f64, ks_a: f64, alpha_a: f64, vel_a: f64,
    rho_b: f64, v_b: f64, s_b: f64, g_b: f64, h_b: f64, u_b: f64, a_b: f64,
    cp_b: f64, cv_b: f64, kt_b: f64, ks_b: f64, alpha_b: f64, vel_b: f64,
}

/// Convert reduced (v, s, kap, alp, cp) to physical SI units.
/// Returns (rho, V_spec, S, G, Cp, Cv, Kt, Ks, alpha, vel).
#[inline(always)]
fn to_physical(
    v_red: f64, s_red: f64, kap_red: f64, alp_red: f64, cp_red: f64,
    t_k: f64, g_red: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let rho = RHO0 / v_red;
    let v_spec = 1.0 / rho;
    let s_val = R * s_red;
    let kappa_t = kap_red / (RHO0 * R * TC); // Pa^-1
    let alp_val = alp_red / TC;               // K^-1
    let cp_val = R * cp_red;                  // J/(kg·K)

    let cv_val = if kappa_t > 0.0 && kappa_t.is_finite() {
        cp_val - t_k * alp_val * alp_val / (rho * kappa_t)
    } else { cp_val };

    let kappa_s = if cp_val > 0.0 {
        kappa_t - t_k * v_spec * alp_val * alp_val / cp_val
    } else { kappa_t };

    let kt = if kappa_t > 0.0 && kappa_t.is_finite() {
        1.0 / kappa_t / 1e6
    } else { f64::INFINITY };
    let ks = if kappa_s > 0.0 { 1.0 / kappa_s / 1e6 } else { f64::INFINITY };
    let vel = if rho > 0.0 && kappa_s > 0.0 {
        (1.0 / (rho * kappa_s)).sqrt()
    } else { f64::NAN };

    let g_val = R * TC * g_red;

    (rho, v_spec, s_val, g_val, cp_val, cv_val, kt, ks, alp_val, vel)
}

#[inline]
fn compute_point(t_k: f64, p_mpa: f64) -> PointResult {
    let p_pa = p_mpa * 1e6;
    let tau = t_k / TC;
    let pi = (p_pa - P0 * 1e6) / P_SCALE_PA;
    let t = tau - 1.0;
    let p_red = (p_pa - PC * 1e6) / P_SCALE_PA;

    // Background
    let (b_val, bp, bt, bpp, btp, btt) = b_all(tau, pi);

    // Field L
    let (l, lt, lp, ltt, ltp, lpp) = compute_l(t, p_red);

    // Omega
    let omega = 2.0 + OMEGA0 * p_red;

    // Equilibrium x
    let x = findxe(l, omega);

    // ── Mixture (phi/chi formulation, MATLAB lines 104-127) ──────────
    let f = 2.0 * x - 1.0;
    let f2 = f * f;
    let chi = if (1.0 - f2).abs() > 1e-30 {
        1.0 / (2.0 / (1.0 - f2) - omega)
    } else { 0.0 };

    let eps_log = 1e-300_f64;
    let g0 = if x > eps_log && x < 1.0 - eps_log {
        x * l + x * x.ln() + (1.0 - x) * (1.0 - x).ln() + omega * x * (1.0 - x)
    } else if x <= eps_log { 0.0 } else { l };

    let s_mix = -0.5 * (f + 1.0) * lt * tau - g0 - bt;
    let v_mix = 0.5 * tau * (OMEGA0 / 2.0 * (1.0 - f2) + lp * (f + 1.0)) + bp;
    let kap_mix = (1.0 / v_mix)
        * (tau / 2.0 * (chi * (lp - OMEGA0 * f).powi(2)
                        - (f + 1.0) * lpp)
           - bpp);
    let alp_mix = (1.0 / v_mix)
        * (ltp / 2.0 * tau * (f + 1.0)
           + (OMEGA0 / 2.0 * (1.0 - f2) + lp * (f + 1.0)) / 2.0
           - tau * lt / 2.0 * chi * (lp - OMEGA0 * f)
           + btp);
    let cp_mix = tau * (-lt * (f + 1.0)
                        + tau * (lt * lt * chi - ltt * (f + 1.0)) / 2.0
                        - btt);

    let g_red_mix = b_val + tau * g0;

    let (rho, v_spec, s_phys, mut g_phys, cp, cv, kt, ks, alpha, vel) =
        to_physical(v_mix, s_mix, kap_mix, alp_mix, cp_mix, t_k, g_red_mix);

    // ── State A (x=0, background only) ──────────────────────────────
    let v_a_red = bp;
    let s_a_red = -bt;
    let kap_a_red = -bpp / v_a_red;
    let alp_a_red = btp / v_a_red;
    let cp_a_red = -tau * btt;
    let g_red_a = b_val;

    let (rho_a, v_a_spec, s_a_phys, mut g_a_phys, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a) =
        to_physical(v_a_red, s_a_red, kap_a_red, alp_a_red, cp_a_red, t_k, g_red_a);

    // ── State B (x=1, f=+1) ─────────────────────────────────────────
    let s_b_red = -lt * tau - l - bt;
    let v_b_red = tau * lp + bp;
    let kap_b_red = (1.0 / v_b_red) * (-tau * lpp - bpp);
    let alp_b_red = (1.0 / v_b_red) * (ltp * tau + lp + btp);
    let cp_b_red = tau * (-2.0 * lt - tau * ltt - btt);
    let g_red_b = b_val + tau * l;

    let (rho_b, v_b_spec, s_b_phys, mut g_b_phys, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b) =
        to_physical(v_b_red, s_b_red, kap_b_red, alp_b_red, cp_b_red, t_k, g_red_b);

    // ── IAPWS-95 reference state alignment ───────────────────────────
    let s_val = s_phys + S_OFFSET;
    g_phys += H_OFFSET - t_k * S_OFFSET;
    let s_a_val = s_a_phys + S_OFFSET;
    g_a_phys += H_OFFSET - t_k * S_OFFSET;
    let s_b_val = s_b_phys + S_OFFSET;
    g_b_phys += H_OFFSET - t_k * S_OFFSET;

    // ── Derived potentials ───────────────────────────────────────────
    let p_pa_f = p_mpa * 1e6;
    let h = g_phys + t_k * s_val;
    let u_pot = h - p_pa_f * v_spec;
    let a_pot = g_phys - p_pa_f * v_spec;

    let h_a = g_a_phys + t_k * s_a_val;
    let u_a = h_a - p_pa_f * v_a_spec;
    let a_a = g_a_phys - p_pa_f * v_a_spec;

    let h_b = g_b_phys + t_k * s_b_val;
    let u_b = h_b - p_pa_f * v_b_spec;
    let a_b = g_b_phys - p_pa_f * v_b_spec;

    PointResult {
        rho, v: v_spec, s: s_val, g: g_phys, h, u: u_pot, a: a_pot,
        cp, cv, kt, ks, alpha, vel, x,
        rho_a, v_a: v_a_spec, s_a: s_a_val, g_a: g_a_phys, h_a, u_a, a_a,
        cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a,
        rho_b, v_b: v_b_spec, s_b: s_b_val, g_b: g_b_phys, h_b, u_b, a_b,
        cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 entry point
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_batch_holten<'py>(
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
