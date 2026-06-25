//! Shi & Tanaka (2020) hierarchical two-state EoS — Rust implementation.
//!
//! See `shi_tanaka_eos/core.py` for the full mathematical derivation.
//! Parameters mirror `shi_tanaka_eos/params.py` from Table S2 of the SI.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════════════════
// Constants  (must stay in sync with shi_tanaka_eos/params.py)
// ═══════════════════════════════════════════════════════════════════════════

const R: f64 = 8.314462;
const KB: f64 = 1.380649e-23;
const NA: f64 = 6.02214076e23;
const M_H2O: f64 = 0.018015268;

const TR: f64 = 308.15;
const PR_MPA: f64 = 200.0;
const PR_PA: f64 = PR_MPA * 1e6;
// Vr = kB * Tr / Pr_Pa  ≈ 2.127e-29 m³ per molecule
const VR: f64 = KB * TR / PR_PA;

// Two-state parameters (Table S2, H₂O column)
const DELTA_E_K: f64 = -1952.0;
const DELTA_SIGMA: f64 = -8.317;
const DELTA_V_MK: f64 = 1.593;
const N_UNIT: f64 = 7.888;

const DELTA_E_HAT: f64 = DELTA_E_K / TR;             // ≈ -6.334
const DELTA_SIGMA_HAT: f64 = DELTA_SIGMA;            // already /kB
// Table S2's "ΔV [MPa^-1 K] = 1.593" annotation is misleading: the listed
// value is already the dimensionless ΔV̂ entering Eq. S2 directly.
// Verified empirically vs IAPWS-95 + Fig. 2. See shi_tanaka_eos/params.py.
const DELTA_V_HAT: f64 = DELTA_V_MK;                 // = 1.593

// Polynomial background coefficients (Table S2)
const C01: f64 = 10.34;
const C02: f64 = -0.2629;
const C03: f64 = 0.03432;
const C11: f64 = 1.309;
const C12: f64 = -0.3383;
const C13: f64 = 0.1090;
const C20: f64 = -13.73;
const C21: f64 = -0.7274;
const C23: f64 = 0.7443;
const C30: f64 = -0.3602;
const C31: f64 = 2.062;
const C1: f64 = -39.91;

// IAPWS-95 alignment offsets (placeholders — zero until calibrated)
const S_OFFSET: f64 = 0.0;
const H_OFFSET: f64 = 0.0;

// ═══════════════════════════════════════════════════════════════════════════
// Background G_ρ polynomial + derivatives  (scalar)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn g_rho(d_t: f64, d_p: f64, t_hat: f64) -> (f64, f64, f64, f64, f64, f64) {
    let dt2 = d_t * d_t;
    let dt3 = dt2 * d_t;
    let dp2 = d_p * d_p;
    let dp3 = dp2 * d_p;

    let log_t = t_hat.ln();

    let val = C01*d_p + C02*dp2 + C03*dp3
        + C11*d_t*d_p + C12*d_t*dp2 + C13*d_t*dp3
        + C20*dt2 + C21*dt2*d_p + C23*dt2*dp3
        + C30*dt3 + C31*dt3*d_p
        + C1 * t_hat * (log_t - 1.0);

    let dval_dp = C01 + 2.0*C02*d_p + 3.0*C03*dp2
        + C11*d_t + 2.0*C12*d_t*d_p + 3.0*C13*d_t*dp2
        + C21*dt2 + 3.0*C23*dt2*dp2
        + C31*dt3;

    let dval_dt = C11*d_p + C12*dp2 + C13*dp3
        + 2.0*C20*d_t + 2.0*C21*d_t*d_p + 2.0*C23*d_t*dp3
        + 3.0*C30*dt2 + 3.0*C31*dt2*d_p
        + C1 * log_t;

    let d2val_dp2 = 2.0*C02 + 6.0*C03*d_p
        + 2.0*C12*d_t + 6.0*C13*d_t*d_p
        + 6.0*C23*dt2*d_p;

    let d2val_dt2 = 2.0*C20 + 2.0*C21*d_p + 2.0*C23*dp3
        + 6.0*C30*d_t + 6.0*C31*d_t*d_p
        + C1 / t_hat;

    let d2val_dpdt = C11 + 2.0*C12*d_p + 3.0*C13*dp2
        + 2.0*C21*d_t + 6.0*C23*d_t*dp2
        + 3.0*C31*dt2;

    (val, dval_dp, dval_dt, d2val_dp2, d2val_dt2, d2val_dpdt)
}

// ═══════════════════════════════════════════════════════════════════════════
// Reduced → physical conversion
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn physical(
    vh: f64, sh: f64, d2g_dp2: f64, d2g_dt2: f64, d2g_dpdt: f64,
    g_hat: f64, t_k: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let t_hat = t_k / TR;

    let v_unit = VR * vh;                                   // m³/unit
    let v_spec = v_unit * NA / (N_UNIT * M_H2O);            // m³/kg
    let rho = if v_spec > 0.0 { 1.0 / v_spec } else { f64::INFINITY };

    let s_spec = sh * R / (N_UNIT * M_H2O);                 // J/(kg·K)
    let cp = -(t_hat / N_UNIT) * d2g_dt2 * R / M_H2O;       // J/(kg·K)

    let kappa_t_hat = if vh.abs() > 1e-30 { -d2g_dp2 / vh } else { f64::INFINITY };
    let kappa_t = kappa_t_hat / PR_PA;                      // Pa⁻¹
    let kt = if kappa_t.is_finite() && kappa_t.abs() > 1e-30 {
        1.0 / kappa_t / 1e6
    } else { 0.0 };

    let alpha = if vh.abs() > 1e-30 { (d2g_dpdt / vh) / TR } else { 0.0 };

    let cv = if kappa_t > 0.0 && kappa_t.is_finite() {
        cp - t_k * v_spec * alpha * alpha / kappa_t
    } else { cp };

    let kappa_s = if cp > 0.0 {
        kappa_t - t_k * v_spec * alpha * alpha / cp
    } else { kappa_t };
    let ks = if kappa_s > 0.0 { 1.0 / kappa_s / 1e6 } else { f64::INFINITY };
    let vel = if rho > 0.0 && kappa_s > 0.0 {
        (1.0 / (rho * kappa_s)).sqrt()
    } else { f64::NAN };

    let g_val = g_hat * R * TR / (N_UNIT * M_H2O);

    (rho, v_spec, s_spec, g_val, cp, cv, kt, ks, alpha, vel)
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
    let t_hat = t_k / TR;
    let p_hat = p_mpa / PR_MPA;
    let d_t = t_hat - 1.0;
    let d_p = p_hat - 1.0;

    // Negative/zero T̂ would make the log term diverge — propagate NaN.
    let t_hat_safe = if t_hat > 0.0 { t_hat } else { f64::NAN };

    let (grho, dgrho_dp, dgrho_dt, d2grho_dp2, d2grho_dt2, d2grho_dpdt) =
        g_rho(d_t, d_p, t_hat_safe);

    // Two-state piece
    let de = DELTA_E_HAT;
    let ds = DELTA_SIGMA_HAT;
    let dv = DELTA_V_HAT;

    let delta_g = de - t_hat * ds + p_hat * dv;
    let arg = (delta_g / t_hat_safe).clamp(-700.0, 700.0);
    let s = 1.0 / (1.0 + arg.exp());

    // First derivatives (envelope theorem)
    let vh = dgrho_dp + s * dv;
    let eps = 1e-15;
    let s_c = s.max(eps).min(1.0 - eps);
    let mix_ent = s_c * s_c.ln() + (1.0 - s_c) * (1.0 - s_c).ln();
    let sh = -(dgrho_dt - s * ds + mix_ent);

    // Second derivatives with fluctuation contribution
    let delta_h = de + p_hat * dv;
    let sfluc = s * (1.0 - s);
    let d2g_dp2 = d2grho_dp2 - sfluc * dv * dv / t_hat_safe;
    let d2g_dpdt = d2grho_dpdt + sfluc * dv * delta_h / (t_hat_safe * t_hat_safe);
    let d2g_dt2 = d2grho_dt2 - sfluc * delta_h * delta_h / (t_hat_safe * t_hat_safe * t_hat_safe);

    // Reduced Gibbs energies for the three "phases"
    let g_hat_mix = grho + s * delta_g + t_hat * mix_ent;
    let g_hat_a = grho;
    let g_hat_b = grho + (de - t_hat * ds + p_hat * dv);

    // Mixture
    let (rho, v, mut s_val, mut g, cp, cv, kt, ks, alpha, vel) =
        physical(vh, sh, d2g_dp2, d2g_dt2, d2g_dpdt, g_hat_mix, t_k);

    // State A = DNLS (s = 0)
    let vh_a = dgrho_dp;
    let sh_a = -dgrho_dt;
    let (rho_a, v_a, mut s_a, mut g_a, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a) =
        physical(vh_a, sh_a, d2grho_dp2, d2grho_dt2, d2grho_dpdt, g_hat_a, t_k);

    // State B = LFTS (s = 1)
    let vh_b = dgrho_dp + dv;
    let sh_b = -(dgrho_dt - ds);
    let (rho_b, v_b, mut s_b, mut g_b, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b) =
        physical(vh_b, sh_b, d2grho_dp2, d2grho_dt2, d2grho_dpdt, g_hat_b, t_k);

    // IAPWS-95 alignment (no-op until calibrated)
    s_val += S_OFFSET; g += H_OFFSET - t_k * S_OFFSET;
    s_a += S_OFFSET; g_a += H_OFFSET - t_k * S_OFFSET;
    s_b += S_OFFSET; g_b += H_OFFSET - t_k * S_OFFSET;

    let p_pa = p_mpa * 1e6;
    let h = g + t_k * s_val; let u_pot = h - p_pa * v; let a_pot = g - p_pa * v;
    let h_a = g_a + t_k * s_a; let u_a = h_a - p_pa * v_a; let a_a = g_a - p_pa * v_a;
    let h_b = g_b + t_k * s_b; let u_b = h_b - p_pa * v_b; let a_b = g_b - p_pa * v_b;

    PointResult {
        rho, v, s: s_val, g, h, u: u_pot, a: a_pot, cp, cv, kt, ks, alpha, vel, x: s,
        rho_a, v_a, s_a, g_a, h_a, u_a, a_a, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a,
        rho_b, v_b, s_b, g_b, h_b, u_b, a_b, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 entry point
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_batch_shi_tanaka<'py>(
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
