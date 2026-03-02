//! Caupin & Anisimov (2019) two-state EoS — Rust implementation.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════════════════
// Constants from caupin_eos/params.py
// ═══════════════════════════════════════════════════════════════════════════

const R: f64 = 8.314462;
const M_H2O: f64 = 0.018015268;

const TC: f64 = 218.1348;
const PC: f64 = 71.94655;
const VC: f64 = 18.22426e-6;

const P_SCALE_PA: f64 = R * TC / VC;
const P_SCALE_MPA: f64 = P_SCALE_PA / 1e6;

const OMEGA0: f64 = 0.1854443;

const LAM: f64 = 1.653737;
const A_PARAM: f64 = 0.1030250;
const B_PARAM: f64 = -0.0392417;
const D_PARAM: f64 = -0.01039947;
const F_PARAM: f64 = 1.021512;

const A0: f64 = -0.08118730;
const A1: f64 = 0.05070641;

const C01: f64 = 1.126869;
const C02: f64 = 0.01005341;
const C11: f64 = -0.2092770;
const C20: f64 = -2.520114;
const C03: f64 = -0.001149367;
const C12: f64 = -0.008992042;
const C21: f64 = 0.2118502;
const C30: f64 = 0.1087670;
const C04: f64 = 0.00007573062;
const C13: f64 = 0.002393927;
const C22: f64 = -0.01831198;
const C40: f64 = 0.02803712;
const C14: f64 = -0.0001641608;

const PS_A: f64 = -462.0;
const PS_B: f64 = 2.61;
const PS_C: f64 = -0.0065;
const PS_T0: f64 = 182.0;

const S_OFFSET: f64 = -146.1582559570;
const H_OFFSET: f64 = 112500.2342686583;

// ═══════════════════════════════════════════════════════════════════════════
// Per-point Gibbs energy components
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn spinodal_pressure(t_k: f64) -> (f64, f64, f64) {
    let dt = t_k - PS_T0;
    let ps = PS_A + PS_B * dt + PS_C * dt * dt;
    let dps_dt = PS_B + 2.0 * PS_C * dt;
    let d2ps_dt2 = 2.0 * PS_C;
    (ps, dps_dt, d2ps_dt2)
}

#[inline(always)]
fn compute_gsigma(d_th: f64, d_ph: f64, t_k: f64) -> (f64, f64, f64, f64, f64, f64) {
    let ah = A0 + A1 * d_th;
    let dah_dt = A1;
    let (ps, dps_dt, d2ps_dt2) = spinodal_pressure(t_k);
    let dphs = (PC - ps) / P_SCALE_MPA;
    let u = d_ph + dphs;
    let dphs_dth = dps_dt * TC / P_SCALE_MPA;
    let d2phs_dth2 = d2ps_dt2 * TC * TC / P_SCALE_MPA;
    let du_dth = -dphs_dth;
    let d2u_dth2 = -d2phs_dth2;
    let u_safe = u.max(1e-30);
    let sqrt_u = u_safe.sqrt();
    let u32 = u_safe * sqrt_u;
    let u12 = sqrt_u;
    let u_m12 = if u_safe > 1e-30 { 1.0 / sqrt_u } else { 0.0 };
    let gs = ah * u32;
    let dgs_dp = 1.5 * ah * u12;
    let d2gs_dp2 = 0.75 * ah * u_m12;
    let dgs_dt = dah_dt * u32 + 1.5 * ah * u12 * du_dth;
    let d2gs_dpdt = 1.5 * dah_dt * u12 + 0.75 * ah * u_m12 * du_dth;
    let d2gs_dt2 =
        3.0 * dah_dt * u12 * du_dth + 0.75 * ah * u_m12 * du_dth * du_dth + 1.5 * ah * u12 * d2u_dth2;
    (gs, dgs_dp, dgs_dt, d2gs_dp2, d2gs_dt2, d2gs_dpdt)
}

#[inline(always)]
fn compute_ga_poly(d_th: f64, d_ph: f64) -> (f64, f64, f64, f64, f64, f64) {
    let t = d_th;
    let q = d_ph;
    let t2 = t * t; let t3 = t2 * t; let t4 = t3 * t;
    let q2 = q * q; let q3 = q2 * q; let q4 = q3 * q;
    let val = C01*q + C02*q2 + C11*t*q + C20*t2
        + C03*q3 + C12*t*q2 + C21*t2*q + C30*t3
        + C04*q4 + C13*t*q3 + C22*t2*q2 + C40*t4 + C14*t*q4;
    let dval_dp = C01 + 2.0*C02*q + C11*t
        + 3.0*C03*q2 + 2.0*C12*t*q + C21*t2
        + 4.0*C04*q3 + 3.0*C13*t*q2 + 2.0*C22*t2*q + 4.0*C14*t*q3;
    let dval_dt = C11*q + 2.0*C20*t
        + C12*q2 + 2.0*C21*t*q + 3.0*C30*t2
        + C13*q3 + 2.0*C22*t*q2 + 4.0*C40*t3 + C14*q4;
    let d2val_dp2 = 2.0*C02 + 6.0*C03*q + 2.0*C12*t
        + 12.0*C04*q2 + 6.0*C13*t*q + 2.0*C22*t2 + 12.0*C14*t*q2;
    let d2val_dt2 = 2.0*C20 + 2.0*C21*q + 6.0*C30*t + 2.0*C22*q2 + 12.0*C40*t2;
    let d2val_dpdt = C11 + 2.0*C12*q + 2.0*C21*t
        + 3.0*C13*q2 + 4.0*C22*t*q + 4.0*C14*q3;
    (val, dval_dp, dval_dt, d2val_dp2, d2val_dt2, d2val_dpdt)
}

#[inline(always)]
fn compute_gba(d_th: f64, d_ph: f64) -> (f64, f64, f64, f64, f64, f64) {
    let gba = LAM * (d_th + A_PARAM*d_ph + B_PARAM*d_th*d_ph + D_PARAM*d_ph*d_ph + F_PARAM*d_th*d_th);
    let dgba_dp = LAM * (A_PARAM + B_PARAM*d_th + 2.0*D_PARAM*d_ph);
    let dgba_dt = LAM * (1.0 + B_PARAM*d_ph + 2.0*F_PARAM*d_th);
    let d2gba_dp2 = 2.0 * LAM * D_PARAM;
    let d2gba_dt2 = 2.0 * LAM * F_PARAM;
    let d2gba_dpdt = LAM * B_PARAM;
    (gba, dgba_dp, dgba_dt, d2gba_dp2, d2gba_dt2, d2gba_dpdt)
}

#[inline(always)]
fn compute_omega(d_th: f64, d_ph: f64) -> (f64, f64, f64, f64, f64, f64) {
    let th = 1.0 + d_th;
    let num = 2.0 + OMEGA0 * d_ph;
    let om = num / th;
    let dom_dp = OMEGA0 / th;
    let dom_dt = -num / (th * th);
    let d2om_dp2 = 0.0;
    let d2om_dt2 = 2.0 * num / (th * th * th);
    let d2om_dpdt = -OMEGA0 / (th * th);
    (om, dom_dp, dom_dt, d2om_dp2, d2om_dt2, d2om_dpdt)
}

// ═══════════════════════════════════════════════════════════════════════════
// Newton solver
// ═══════════════════════════════════════════════════════════════════════════

const NEWTON_EPS: f64 = 1e-15;
const NEWTON_TOL: f64 = 1e-12;
const NEWTON_MAX_ITER: usize = 50;

#[inline(always)]
fn newton_solve(gba: f64, th: f64, om: f64, x0: f64) -> f64 {
    let mut x = x0;
    for _ in 0..NEWTON_MAX_ITER {
        x = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
        let lnrat = (x / (1.0 - x)).ln();
        let f_val = gba + th * (lnrat + om * (1.0 - 2.0 * x));
        let fx = th * (1.0 / (x * (1.0 - x)) - 2.0 * om);
        if fx.abs() < 1e-30 { break; }
        let dx = -f_val / fx;
        if x + dx < NEWTON_EPS {
            x /= 2.0;
        } else if x + dx > 1.0 - NEWTON_EPS {
            x = (x + 1.0 - NEWTON_EPS) / 2.0;
        } else {
            x += dx;
        }
        if f_val.abs() < NEWTON_TOL { break; }
    }
    x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS)
}

#[inline(always)]
fn gibbs_mixing(x: f64, gba: f64, th: f64, om: f64) -> f64 {
    let xc = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
    let mix_ent = xc * xc.ln() + (1.0 - xc) * (1.0 - xc).ln();
    xc * gba + th * (mix_ent + om * xc * (1.0 - xc))
}

#[inline(always)]
fn solve_x_stable(gba: f64, th: f64, om: f64) -> f64 {
    let x_lo = newton_solve(gba, th, om, 0.05);
    let x_hi = newton_solve(gba, th, om, 0.95);
    let g_lo = gibbs_mixing(x_lo, gba, th, om);
    let g_hi = gibbs_mixing(x_hi, gba, th, om);
    if g_lo <= g_hi { x_lo } else { x_hi }
}

// ═══════════════════════════════════════════════════════════════════════════
// Physical properties
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn physical_props(
    vh: f64, sh_red: f64, d2g_dp2: f64, d2g_dt2: f64, d2g_dpdt: f64,
    t_k: f64, g_hat: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let th = t_k / TC;
    let v_molar = VC * vh;
    let v_spec = v_molar / M_H2O;
    let rho = if v_spec > 0.0 { 1.0 / v_spec } else { f64::INFINITY };
    let s_molar = R * sh_red;
    let s_spec = s_molar / M_H2O;
    let cp_molar = -R * th * d2g_dt2;
    let cp = cp_molar / M_H2O;
    let kappa_t = if vh.abs() > 1e-30 {
        -(VC / (R * TC)) * d2g_dp2 / vh
    } else { f64::INFINITY };
    let kt = if kappa_t.abs() > 1e-30 && kappa_t.is_finite() {
        1.0 / kappa_t / 1e6
    } else { 0.0 };
    let alpha = if vh.abs() > 1e-30 {
        (1.0 / TC) * d2g_dpdt / vh
    } else { 0.0 };
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
    let g_val = R * TC * g_hat / M_H2O;
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
    let d_th = (t_k - TC) / TC;
    let d_ph = (p_mpa - PC) / P_SCALE_MPA;
    let th = 1.0 + d_th;

    let (gs, dgs_dp, dgs_dt, d2gs_dp2, d2gs_dt2, d2gs_dpdt) = compute_gsigma(d_th, d_ph, t_k);
    let (gp, dgp_dp, dgp_dt, d2gp_dp2, d2gp_dt2, d2gp_dpdt) = compute_ga_poly(d_th, d_ph);

    let ga = gs + gp;
    let dga_dp = dgs_dp + dgp_dp;
    let dga_dt = dgs_dt + dgp_dt;
    let d2ga_dp2 = d2gs_dp2 + d2gp_dp2;
    let d2ga_dt2 = d2gs_dt2 + d2gp_dt2;
    let d2ga_dpdt = d2gs_dpdt + d2gp_dpdt;

    let (gba, dgba_dp, dgba_dt, d2gba_dp2, d2gba_dt2, d2gba_dpdt) = compute_gba(d_th, d_ph);
    let (om, _dom_dp, _dom_dt, _d2om_dp2, _d2om_dt2, _d2om_dpdt) = compute_omega(d_th, d_ph);

    let x = solve_x_stable(gba, th, om);

    let big_om = th * om;
    let dom_dp_big = OMEGA0;

    let vh = dga_dp + x * dgba_dp + dom_dp_big * x * (1.0 - x);

    let x_c = x.max(NEWTON_EPS).min(1.0 - NEWTON_EPS);
    let mix_ent = x_c * x_c.ln() + (1.0 - x_c) * (1.0 - x_c).ln();
    let sh_red = -(dga_dt + x * dgba_dt + mix_ent);

    let lnrat = (x_c / (1.0 - x_c)).ln();
    let fx = th * (1.0 / (x_c * (1.0 - x_c)) - 2.0 * om);
    let fx_safe = if fx.abs() > 1e-30 { fx } else { 1e30 };

    let f_dt = dgba_dt + lnrat;
    let f_dp = dgba_dp + dom_dp_big * (1.0 - 2.0 * x);

    let dx_dt = -f_dt / fx_safe;
    let dx_dp = -f_dp / fx_safe;

    let d2g_dp2_x = d2ga_dp2 + x * d2gba_dp2;
    let d2g_dt2_x = d2ga_dt2 + x * d2gba_dt2;
    let d2g_dpdt_x = d2ga_dpdt + x * d2gba_dpdt;

    let d2g_dp2_total = d2g_dp2_x + f_dp * dx_dp;
    let d2g_dt2_total = d2g_dt2_x + f_dt * dx_dt;
    let d2g_dpdt_total = d2g_dpdt_x + f_dp * dx_dt;

    let g_hat_mix = ga + x * gba + th * mix_ent + big_om * x * (1.0 - x);
    let g_hat_a = ga;
    let g_hat_b = ga + gba;

    let (rho, v, mut s, mut g, cp, cv, kt, ks, alpha, vel) =
        physical_props(vh, sh_red, d2g_dp2_total, d2g_dt2_total, d2g_dpdt_total, t_k, g_hat_mix);
    let (rho_a, v_a, mut s_a, mut g_a, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a) =
        physical_props(dga_dp, -dga_dt, d2ga_dp2, d2ga_dt2, d2ga_dpdt, t_k, g_hat_a);

    let vhb = dga_dp + dgba_dp;
    let shb_red = -(dga_dt + dgba_dt);
    let d2gb_dp2 = d2ga_dp2 + d2gba_dp2;
    let d2gb_dt2 = d2ga_dt2 + d2gba_dt2;
    let d2gb_dpdt = d2ga_dpdt + d2gba_dpdt;
    let (rho_b, v_b, mut s_b, mut g_b, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b) =
        physical_props(vhb, shb_red, d2gb_dp2, d2gb_dt2, d2gb_dpdt, t_k, g_hat_b);

    s += S_OFFSET; g += H_OFFSET - t_k * S_OFFSET;
    s_a += S_OFFSET; g_a += H_OFFSET - t_k * S_OFFSET;
    s_b += S_OFFSET; g_b += H_OFFSET - t_k * S_OFFSET;

    let p_pa = p_mpa * 1e6;
    let h = g + t_k * s; let u_pot = h - p_pa * v; let a_pot = g - p_pa * v;
    let h_a = g_a + t_k * s_a; let u_a = h_a - p_pa * v_a; let a_a = g_a - p_pa * v_a;
    let h_b = g_b + t_k * s_b; let u_b = h_b - p_pa * v_b; let a_b = g_b - p_pa * v_b;

    PointResult {
        rho, v, s, g, h, u: u_pot, a: a_pot, cp, cv, kt, ks, alpha, vel, x,
        rho_a, v_a, s_a, g_a, h_a, u_a, a_a, cp_a, cv_a, kt_a, ks_a, alpha_a, vel_a,
        rho_b, v_b, s_b, g_b, h_b, u_b, a_b, cp_b, cv_b, kt_b, ks_b, alpha_b, vel_b,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 entry point
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_batch_caupin<'py>(
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
