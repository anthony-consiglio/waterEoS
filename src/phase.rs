//! Generic phase diagram algorithms for two-state water EoS models.
//!
//! Implements spinodal and binodal computation using trait-based dispatch,
//! monomorphized by the compiler for zero-cost abstraction.
//!
//! All three models share identical mixing algebra:
//!   F_eq(x) = field + th * ln(x/(1-x)) + coop * (1-2x)
//!   g_mix(x) = x*field + th*[x*ln(x)+(1-x)*ln(1-x)] + coop*x*(1-x)
//!   disc = 1 - 2*th/coop

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════════════════
// Trait: model-specific mixing parameters
// ═══════════════════════════════════════════════════════════════════════════

trait PhaseMath {
    fn mixing_params(t_k: f64, p_mpa: f64) -> (f64, f64, f64);
}

// ═══════════════════════════════════════════════════════════════════════════
// Caupin implementation
// ═══════════════════════════════════════════════════════════════════════════

mod caupin_consts {
    pub const TC: f64 = 218.1348;
    pub const PC: f64 = 71.94655;
    const R: f64 = 8.314462;
    const VC: f64 = 18.22426e-6;
    pub const P_SCALE_MPA: f64 = R * TC / VC / 1e6;
    pub const OMEGA0: f64 = 0.1854443;
    pub const LAM: f64 = 1.653737;
    pub const A: f64 = 0.1030250;
    pub const B: f64 = -0.0392417;
    pub const D: f64 = -0.01039947;
    pub const F: f64 = 1.021512;
}

struct CaupinPhase;

impl PhaseMath for CaupinPhase {
    #[inline]
    fn mixing_params(t_k: f64, p_mpa: f64) -> (f64, f64, f64) {
        use caupin_consts::*;
        let d_th = (t_k - TC) / TC;
        let d_ph = (p_mpa - PC) / P_SCALE_MPA;
        let gba = LAM
            * (d_th + A * d_ph + B * d_th * d_ph + D * d_ph * d_ph + F * d_th * d_th);
        let th = 1.0 + d_th;
        let big_omega = 2.0 + OMEGA0 * d_ph;
        (gba, th, big_omega)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Holten implementation
// ═══════════════════════════════════════════════════════════════════════════

mod holten_consts {
    pub const TC: f64 = 228.2;
    pub const PC_MPA: f64 = 0.0;
    const R: f64 = 461.523087;
    const RHO0: f64 = 1081.6482;
    pub const P_SCALE_PA: f64 = RHO0 * R * TC;
    pub const OMEGA0: f64 = 0.52122690;
    pub const L0: f64 = 0.76317954;
    pub const K0: f64 = 0.072158686;
    pub const K1: f64 = -0.31569232;
    pub const K2: f64 = 5.2992608;
}

struct HoltenPhase;

impl HoltenPhase {
    #[inline]
    fn compute_l(t: f64, p_red: f64) -> f64 {
        use holten_consts::*;
        let arg = p_red - K2 * t;
        let inner = 1.0 + K0 * K2 + K1 * arg;
        let k1_arg = inner * inner - 4.0 * K0 * K1 * K2 * arg;
        let k1_val = if k1_arg > 0.0 { k1_arg.sqrt() } else { 0.0 };
        let k2_val = (1.0 + K2 * K2).sqrt();
        L0 * k2_val * (1.0 - k1_val + K0 * K2 + K1 * (p_red + K2 * t))
            / (2.0 * K1 * K2)
    }
}

impl PhaseMath for HoltenPhase {
    #[inline]
    fn mixing_params(t_k: f64, p_mpa: f64) -> (f64, f64, f64) {
        use holten_consts::*;
        let t = (t_k - TC) / TC;
        let p_red = (p_mpa * 1e6 - PC_MPA * 1e6) / P_SCALE_PA;
        let l = HoltenPhase::compute_l(t, p_red);
        let omega = 2.0 + OMEGA0 * p_red;
        (l, 1.0, omega)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Duska implementation
// ═══════════════════════════════════════════════════════════════════════════

mod duska_consts {
    pub const T_VLCP: f64 = 647.096;
    pub const P_VLCP: f64 = 22.064;
    pub const A_DG: [f64; 7] = [
        -4.3743227e-1, -1.3836753e-2, 1.8525106e-2, 4.3306058e-1,
        2.1944047e+0, -1.6301740e-5, 7.6204693e-6,
    ];
    pub const W_OM: [f64; 4] = [
        4.1420925e-1, 3.6615174e-2, 1.6181775e+0, 7.1477190e-3,
    ];
}

struct DuskaPhase;

impl PhaseMath for DuskaPhase {
    #[inline]
    fn mixing_params(t_k: f64, p_mpa: f64) -> (f64, f64, f64) {
        use duska_consts::*;
        let th = t_k / T_VLCP;
        let ph = p_mpa / P_VLCP;
        let dg = A_DG[0]
            + A_DG[1] * ph * th
            + A_DG[2] * ph
            + A_DG[3] * th
            + A_DG[4] * th * th
            + A_DG[5] * ph * ph
            + A_DG[6] * ph * ph * ph;
        let omega = W_OM[0] * (1.0 + W_OM[1] * ph + W_OM[2] * th + W_OM[3] * th * ph);
        (dg, th, omega)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic mixing functions
// ═══════════════════════════════════════════════════════════════════════════

const CLIP_EPS: f64 = 1e-15;
const EVAL_EPS: f64 = 1e-12;

#[inline]
fn f_eq(x: f64, field: f64, th: f64, coop: f64) -> f64 {
    let sx = x.clamp(CLIP_EPS, 1.0 - CLIP_EPS);
    field + th * (sx / (1.0 - sx)).ln() + coop * (1.0 - 2.0 * sx)
}

#[inline]
fn g_mix(x: f64, field: f64, th: f64, coop: f64) -> f64 {
    let sx = x.clamp(CLIP_EPS, 1.0 - CLIP_EPS);
    sx * field
        + th * (sx * sx.ln() + (1.0 - sx) * (1.0 - sx).ln())
        + coop * sx * (1.0 - sx)
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic phase diagram helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether F(x)=0 has 3 roots at (T_K, P_MPa).
#[inline]
fn has_three_roots<M: PhaseMath>(t_k: f64, p_mpa: f64) -> bool {
    let (field, th, coop) = M::mixing_params(t_k, p_mpa);
    has_three_roots_params(field, th, coop)
}

#[inline]
fn has_three_roots_params(field: f64, th: f64, coop: f64) -> bool {
    if coop <= 0.0 {
        return false;
    }
    let disc = 1.0 - 2.0 * th / coop;
    if disc <= 0.0 {
        return false;
    }
    let sqrt_disc = disc.sqrt();
    let x_lo = (1.0 - sqrt_disc) / 2.0;
    let x_hi = (1.0 + sqrt_disc) / 2.0;

    let f1 = f_eq(EVAL_EPS, field, th, coop);
    let f2 = f_eq(x_lo, field, th, coop);
    let f3 = f_eq(x_hi, field, th, coop);
    let f4 = f_eq(1.0 - EVAL_EPS, field, th, coop);

    (f1 * f2 < 0.0) && (f2 * f3 < 0.0) && (f3 * f4 < 0.0)
}

/// Find roots x1 (near 0) and x3 (near 1) of F(x)=0.
/// Returns (x1, x3) or (NaN, NaN) if invalid.
#[inline]
fn find_roots_from_params(field: f64, th: f64, coop: f64) -> (f64, f64) {
    if coop <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let disc = 1.0 - 2.0 * th / coop;
    if disc <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let sqrt_disc = disc.sqrt();
    let x_infl_lo = (1.0 - sqrt_disc) / 2.0;
    let x_infl_hi = (1.0 + sqrt_disc) / 2.0;

    // Root 1: bisect x in (EVAL_EPS, x_infl_lo)
    let mut lo1 = EVAL_EPS;
    let mut hi1 = x_infl_lo;
    let mut f_a1 = f_eq(lo1, field, th, coop);
    for _ in 0..35 {
        let mid = (lo1 + hi1) / 2.0;
        let f_mid = f_eq(mid, field, th, coop);
        if f_a1 * f_mid > 0.0 {
            lo1 = mid;
            f_a1 = f_mid;
        } else {
            hi1 = mid;
        }
    }
    let x1 = (lo1 + hi1) / 2.0;

    // Root 3: bisect x in (x_infl_hi, 1-EVAL_EPS)
    let mut lo3 = x_infl_hi;
    let mut hi3 = 1.0 - EVAL_EPS;
    let mut f_a3 = f_eq(lo3, field, th, coop);
    for _ in 0..35 {
        let mid = (lo3 + hi3) / 2.0;
        let f_mid = f_eq(mid, field, th, coop);
        if f_a3 * f_mid > 0.0 {
            lo3 = mid;
            f_a3 = f_mid;
        } else {
            hi3 = mid;
        }
    }
    let x3 = (lo3 + hi3) / 2.0;

    (x1, x3)
}

// ═══════════════════════════════════════════════════════════════════════════
// Spinodal result
// ═══════════════════════════════════════════════════════════════════════════

struct SpinodalResult {
    t_k: Vec<f64>,
    p_mpa: Vec<f64>,
    t_upper: Vec<f64>,
    t_lower: Vec<f64>,
    x_lo_upper: Vec<f64>,
    x_hi_upper: Vec<f64>,
    x_lo_lower: Vec<f64>,
    x_hi_lower: Vec<f64>,
    p_array: Vec<f64>,
}

impl SpinodalResult {
    fn empty(t_llcp: f64, p_llcp: f64) -> Self {
        SpinodalResult {
            t_k: vec![t_llcp],
            p_mpa: vec![p_llcp],
            t_upper: vec![],
            t_lower: vec![],
            x_lo_upper: vec![],
            x_hi_upper: vec![],
            x_lo_lower: vec![],
            x_hi_lower: vec![],
            p_array: vec![],
        }
    }

    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("T_K", Array1::from(self.t_k.clone()).into_pyarray(py))?;
        dict.set_item("p_MPa", Array1::from(self.p_mpa.clone()).into_pyarray(py))?;
        dict.set_item("T_upper", Array1::from(self.t_upper.clone()).into_pyarray(py))?;
        dict.set_item("T_lower", Array1::from(self.t_lower.clone()).into_pyarray(py))?;
        dict.set_item(
            "x_lo_upper",
            Array1::from(self.x_lo_upper.clone()).into_pyarray(py),
        )?;
        dict.set_item(
            "x_hi_upper",
            Array1::from(self.x_hi_upper.clone()).into_pyarray(py),
        )?;
        dict.set_item(
            "x_lo_lower",
            Array1::from(self.x_lo_lower.clone()).into_pyarray(py),
        )?;
        dict.set_item(
            "x_hi_lower",
            Array1::from(self.x_hi_lower.clone()).into_pyarray(py),
        )?;
        dict.set_item("p_array", Array1::from(self.p_array.clone()).into_pyarray(py))?;
        Ok(dict)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spinodal algorithm (port of fast_phase_diagram.compute_spinodal_fast)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_spinodal_impl<M: PhaseMath>(
    p_arr: &[f64],
    t_llcp: f64,
    p_llcp: f64,
) -> SpinodalResult {
    let n = p_arr.len();

    // ── Step 1: Find a temperature inside the 3-root region ──────────────
    // Build scan temperatures: fine near LLCP, then coarser
    let mut scan_temps = Vec::new();
    {
        let mut t = t_llcp - 0.005;
        while t > t_llcp - 1.5 {
            scan_temps.push(t);
            t -= 0.005;
        }
        t = t_llcp - 1.5;
        while t > t_llcp - 5.0 {
            scan_temps.push(t);
            t -= 0.5;
        }
        t = t_llcp - 5.0;
        while t > 50.0 {
            scan_temps.push(t);
            t -= 2.0;
        }
    }

    let mut t_inside = vec![f64::NAN; n];
    let mut found = vec![false; n];

    for &t_test in &scan_temps {
        let mut all_found = true;
        for i in 0..n {
            if !found[i] {
                if has_three_roots::<M>(t_test, p_arr[i]) {
                    t_inside[i] = t_test;
                    found[i] = true;
                } else {
                    all_found = false;
                }
            }
        }
        if all_found {
            break;
        }
    }

    let valid = found;
    if !valid.iter().any(|&v| v) {
        return SpinodalResult::empty(t_llcp, p_llcp);
    }

    // ── Step 2: Upper spinodal — bisect [T_inside, T_LLCP + 5] ──────────
    let mut t_lo = vec![0.0f64; n];
    let mut t_hi = vec![t_llcp + 5.0; n];
    for i in 0..n {
        t_lo[i] = if valid[i] { t_inside[i] } else { t_llcp };
    }

    for _ in 0..45 {
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let t_mid = (t_lo[i] + t_hi[i]) / 2.0;
            if has_three_roots::<M>(t_mid, p_arr[i]) {
                t_lo[i] = t_mid;
            } else {
                t_hi[i] = t_mid;
            }
        }
    }

    let t_upper_all: Vec<f64> = (0..n).map(|i| (t_lo[i] + t_hi[i]) / 2.0).collect();

    // ── Step 3: Lower spinodal ───────────────────────────────────────────
    // Exponential scan downward from T_inside, then bisect.
    let mut t_hi_bnd = vec![0.0f64; n];
    let mut t_lo_bnd = vec![5.0f64; n];
    let mut bracketed = vec![false; n];
    let mut t_cursor = t_inside.clone();
    let mut step = vec![0.01f64; n];

    for i in 0..n {
        t_hi_bnd[i] = if valid[i] { t_inside[i] } else { t_llcp };
    }

    for _ in 0..40 {
        let mut all_valid_bracketed = true;
        for i in 0..n {
            if !valid[i] || bracketed[i] {
                continue;
            }
            let t_test = (t_cursor[i] - step[i]).max(5.0);
            let has = has_three_roots::<M>(t_test, p_arr[i]);

            if has {
                // Still has roots: advance cursor, widen step
                t_cursor[i] = t_test;
                t_hi_bnd[i] = t_test;
                step[i] *= 2.0;
            }

            if !has {
                // Gap found — record bracket
                t_lo_bnd[i] = t_test;
                bracketed[i] = true;
                continue;
            }

            // has == true, check if at temperature floor
            if t_test <= 5.0 {
                t_lo_bnd[i] = 5.0;
                bracketed[i] = true;
                continue;
            }

            all_valid_bracketed = false;
        }
        if all_valid_bracketed {
            break;
        }
    }

    // Bisect to refine the local lower boundary
    for _ in 0..50 {
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let t_mid = (t_lo_bnd[i] + t_hi_bnd[i]) / 2.0;
            if has_three_roots::<M>(t_mid, p_arr[i]) {
                t_hi_bnd[i] = t_mid;
            } else {
                t_lo_bnd[i] = t_mid;
            }
        }
    }

    let t_lower_all: Vec<f64> = (0..n).map(|i| (t_lo_bnd[i] + t_hi_bnd[i]) / 2.0).collect();

    // ── Filter disconnected regions ──────────────────────────────────────
    // Detect via large jumps in T_upper scanning from high P to low P.
    let mut idx: Vec<usize> = (0..n).filter(|&i| valid[i]).collect();

    if idx.len() > 1 {
        let mut connected = vec![true; idx.len()];
        for i in (0..idx.len() - 1).rev() {
            if t_upper_all[idx[i + 1]] - t_upper_all[idx[i]] > 30.0 {
                connected[i] = false;
            } else if !connected[i + 1] {
                connected[i] = false;
            }
        }
        let new_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .filter(|&(j, _)| connected[j])
            .map(|(_, &ii)| ii)
            .collect();
        idx = new_idx;
    }

    if idx.is_empty() {
        return SpinodalResult::empty(t_llcp, p_llcp);
    }

    // ── Extract valid pressures ──────────────────────────────────────────
    let t_upper: Vec<f64> = idx.iter().map(|&i| t_upper_all[i]).collect();
    let t_lower: Vec<f64> = idx.iter().map(|&i| t_lower_all[i]).collect();
    let p_valid: Vec<f64> = idx.iter().map(|&i| p_arr[i]).collect();

    // ── Get spinodal compositions ────────────────────────────────────────
    let delta = 0.001;
    let nv = idx.len();

    let mut x_lo_upper = vec![0.0f64; nv];
    let mut x_hi_upper = vec![0.0f64; nv];
    let mut x_lo_lower = vec![0.0f64; nv];
    let mut x_hi_lower = vec![0.0f64; nv];

    for j in 0..nv {
        let (field, th, coop) = M::mixing_params(t_upper[j] - delta, p_valid[j]);
        let (x1, x3) = find_roots_from_params(field, th, coop);
        x_lo_upper[j] = x1;
        x_hi_upper[j] = x3;

        let (field, th, coop) = M::mixing_params(t_lower[j] + delta, p_valid[j]);
        let (x1, x3) = find_roots_from_params(field, th, coop);
        x_lo_lower[j] = x1;
        x_hi_lower[j] = x3;
    }

    // ── Build closed curve ───────────────────────────────────────────────
    let mut t_curve = Vec::with_capacity(2 + 2 * nv);
    let mut p_curve = Vec::with_capacity(2 + 2 * nv);

    t_curve.push(t_llcp);
    p_curve.push(p_llcp);
    for j in 0..nv {
        t_curve.push(t_upper[j]);
        p_curve.push(p_valid[j]);
    }
    for j in (0..nv).rev() {
        t_curve.push(t_lower[j]);
        p_curve.push(p_valid[j]);
    }
    t_curve.push(t_llcp);
    p_curve.push(p_llcp);

    SpinodalResult {
        t_k: t_curve,
        p_mpa: p_curve,
        t_upper,
        t_lower,
        x_lo_upper,
        x_hi_upper,
        x_lo_lower,
        x_hi_lower,
        p_array: p_valid,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Binodal result
// ═══════════════════════════════════════════════════════════════════════════

struct BinodalResult {
    t_k: Vec<f64>,
    p_mpa: Vec<f64>,
    x: Vec<f64>,
    t_binodal: Vec<f64>,
    x_lo: Vec<f64>,
    x_hi: Vec<f64>,
    p_array: Vec<f64>,
}

impl BinodalResult {
    fn empty(t_llcp: f64, p_llcp: f64) -> Self {
        BinodalResult {
            t_k: vec![t_llcp],
            p_mpa: vec![p_llcp],
            x: vec![0.5],
            t_binodal: vec![],
            x_lo: vec![],
            x_hi: vec![],
            p_array: vec![],
        }
    }

    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("T_K", Array1::from(self.t_k.clone()).into_pyarray(py))?;
        dict.set_item("p_MPa", Array1::from(self.p_mpa.clone()).into_pyarray(py))?;
        dict.set_item("x", Array1::from(self.x.clone()).into_pyarray(py))?;
        dict.set_item(
            "T_binodal",
            Array1::from(self.t_binodal.clone()).into_pyarray(py),
        )?;
        dict.set_item("x_lo", Array1::from(self.x_lo.clone()).into_pyarray(py))?;
        dict.set_item("x_hi", Array1::from(self.x_hi.clone()).into_pyarray(py))?;
        dict.set_item("p_array", Array1::from(self.p_array.clone()).into_pyarray(py))?;
        Ok(dict)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Binodal algorithm (port of fast_phase_diagram.compute_binodal_fast)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_binodal_impl<M: PhaseMath>(
    spin_p: &[f64],
    spin_t_upper: &[f64],
    spin_t_lower: &[f64],
    t_llcp: f64,
    p_llcp: f64,
) -> BinodalResult {
    let ns = spin_p.len();
    if ns == 0 {
        return BinodalResult::empty(t_llcp, p_llcp);
    }

    // T bounds (slightly inside spinodal)
    let t_lo_bound: Vec<f64> = spin_t_lower.iter().map(|&t| t + 0.0001).collect();
    let t_hi_bound: Vec<f64> = spin_t_upper.iter().map(|&t| t - 0.0001).collect();
    let scan_valid: Vec<bool> = (0..ns).map(|i| t_hi_bound[i] > t_lo_bound[i]).collect();

    // ── Scan phase: evaluate delta_g at multiple temperatures ────────────
    let n_scan: usize = 30;
    let dome_width: Vec<f64> = (0..ns).map(|i| t_hi_bound[i] - t_lo_bound[i]).collect();

    // Build scan offsets per pressure (geometric spacing)
    let min_off: f64 = 0.0001;
    let mut scan_offsets = vec![vec![0.0f64; ns]; n_scan];
    for j in 0..n_scan {
        let frac = j as f64 / (n_scan - 1) as f64;
        for i in 0..ns {
            let max_off = dome_width[i].max(0.001);
            scan_offsets[j][i] = min_off * (max_off / min_off).powf(frac);
        }
    }

    let mut dg_grid = vec![vec![f64::NAN; ns]; n_scan];
    let mut t_scan_grid = vec![vec![f64::NAN; ns]; n_scan];

    for j in 0..n_scan {
        for i in 0..ns {
            if !scan_valid[i] {
                t_scan_grid[j][i] = t_llcp;
                continue;
            }
            let ts = t_hi_bound[i] - scan_offsets[j][i];
            let t_scan = if ts > t_lo_bound[i] { ts } else { t_llcp };
            t_scan_grid[j][i] = t_scan;

            if t_scan == t_llcp {
                continue;
            }

            // Compute mixing params once, find roots, evaluate g_mix
            let (field, th, coop) = M::mixing_params(t_scan, spin_p[i]);
            let (x1, x3) = find_roots_from_params(field, th, coop);
            if !x1.is_nan() && !x3.is_nan() {
                let g1 = g_mix(x1, field, th, coop);
                let g3 = g_mix(x3, field, th, coop);
                dg_grid[j][i] = g3 - g1;
            }
        }
    }

    // ── Find sign changes in delta_g ─────────────────────────────────────
    let mut t_bracket_lo = vec![f64::NAN; ns];
    let mut t_bracket_hi = vec![f64::NAN; ns];
    let mut bracket_found = vec![false; ns];

    for j in 0..n_scan - 1 {
        for i in 0..ns {
            if bracket_found[i] || !scan_valid[i] {
                continue;
            }
            let dg_j = dg_grid[j][i];
            let dg_j1 = dg_grid[j + 1][i];
            if !dg_j.is_nan() && !dg_j1.is_nan() && dg_j * dg_j1 < 0.0 {
                let t_j = t_scan_grid[j][i];
                let t_j1 = t_scan_grid[j + 1][i];
                t_bracket_hi[i] = t_j.max(t_j1);
                t_bracket_lo[i] = t_j.min(t_j1);
                bracket_found[i] = true;
            }
        }
    }

    if !bracket_found.iter().any(|&b| b) {
        return BinodalResult::empty(t_llcp, p_llcp);
    }

    // ── Refine with bisection on T ───────────────────────────────────────
    let bix: Vec<usize> = (0..ns).filter(|&i| bracket_found[i]).collect();
    let nb = bix.len();
    let p_bix: Vec<f64> = bix.iter().map(|&i| spin_p[i]).collect();
    let mut t_lo_b: Vec<f64> = bix.iter().map(|&i| t_bracket_lo[i]).collect();
    let mut t_hi_b: Vec<f64> = bix.iter().map(|&i| t_bracket_hi[i]).collect();

    // Evaluate dg at T_lo to establish reference sign
    let mut dg_lo = vec![0.0f64; nb];
    for k in 0..nb {
        let (field, th, coop) = M::mixing_params(t_lo_b[k], p_bix[k]);
        let (x1, x3) = find_roots_from_params(field, th, coop);
        let g1 = g_mix(x1, field, th, coop);
        let g3 = g_mix(x3, field, th, coop);
        dg_lo[k] = g3 - g1;
    }

    for _ in 0..25 {
        for k in 0..nb {
            let t_mid = (t_lo_b[k] + t_hi_b[k]) / 2.0;
            let (field, th, coop) = M::mixing_params(t_mid, p_bix[k]);
            let (x1, x3) = find_roots_from_params(field, th, coop);
            let g1 = g_mix(x1, field, th, coop);
            let g3 = g_mix(x3, field, th, coop);
            let dg_mid = g3 - g1;

            if dg_lo[k] * dg_mid > 0.0 {
                t_lo_b[k] = t_mid;
                dg_lo[k] = dg_mid;
            } else {
                t_hi_b[k] = t_mid;
            }
        }
    }

    let t_eq: Vec<f64> = (0..nb).map(|k| (t_lo_b[k] + t_hi_b[k]) / 2.0).collect();

    // Get final x values at binodal
    let mut x1_final = vec![0.0f64; nb];
    let mut x3_final = vec![0.0f64; nb];
    for k in 0..nb {
        let (field, th, coop) = M::mixing_params(t_eq[k], p_bix[k]);
        let (x1, x3) = find_roots_from_params(field, th, coop);
        x1_final[k] = x1;
        x3_final[k] = x3;
    }

    // Filter NaN
    let good: Vec<bool> = (0..nb)
        .map(|k| !t_eq[k].is_nan() && !x1_final[k].is_nan() && !x3_final[k].is_nan())
        .collect();

    let t_binodal: Vec<f64> = (0..nb).filter(|&k| good[k]).map(|k| t_eq[k]).collect();
    let x_lo: Vec<f64> = (0..nb).filter(|&k| good[k]).map(|k| x1_final[k]).collect();
    let x_hi: Vec<f64> = (0..nb).filter(|&k| good[k]).map(|k| x3_final[k]).collect();
    let p_valid: Vec<f64> = (0..nb).filter(|&k| good[k]).map(|k| p_bix[k]).collect();

    if t_binodal.is_empty() {
        return BinodalResult::empty(t_llcp, p_llcp);
    }

    BinodalResult {
        t_k: t_binodal.clone(),
        p_mpa: p_valid.clone(),
        x: x_lo.clone(),
        t_binodal,
        x_lo,
        x_hi,
        p_array: p_valid,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 wrappers — Spinodal
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_spinodal_caupin<'py>(
    py: Python<'py>,
    p_arr: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = p_arr.to_vec().unwrap_or_else(|_| p_arr.as_slice().unwrap().to_vec());
    let result = compute_spinodal_impl::<CaupinPhase>(&p, t_llcp, p_llcp);
    result.to_py_dict(py)
}

#[pyfunction]
pub fn compute_spinodal_holten<'py>(
    py: Python<'py>,
    p_arr: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = p_arr.to_vec().unwrap_or_else(|_| p_arr.as_slice().unwrap().to_vec());
    let result = compute_spinodal_impl::<HoltenPhase>(&p, t_llcp, p_llcp);
    result.to_py_dict(py)
}

#[pyfunction]
pub fn compute_spinodal_duska<'py>(
    py: Python<'py>,
    p_arr: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = p_arr.to_vec().unwrap_or_else(|_| p_arr.as_slice().unwrap().to_vec());
    let result = compute_spinodal_impl::<DuskaPhase>(&p, t_llcp, p_llcp);
    result.to_py_dict(py)
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 wrappers — Binodal
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
pub fn compute_binodal_caupin<'py>(
    py: Python<'py>,
    spin_p: PyReadonlyArray1<'py, f64>,
    spin_t_upper: PyReadonlyArray1<'py, f64>,
    spin_t_lower: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = spin_p.to_vec().unwrap_or_else(|_| spin_p.as_slice().unwrap().to_vec());
    let tu = spin_t_upper
        .to_vec()
        .unwrap_or_else(|_| spin_t_upper.as_slice().unwrap().to_vec());
    let tl = spin_t_lower
        .to_vec()
        .unwrap_or_else(|_| spin_t_lower.as_slice().unwrap().to_vec());
    let result = compute_binodal_impl::<CaupinPhase>(&sp, &tu, &tl, t_llcp, p_llcp);
    result.to_py_dict(py)
}

#[pyfunction]
pub fn compute_binodal_holten<'py>(
    py: Python<'py>,
    spin_p: PyReadonlyArray1<'py, f64>,
    spin_t_upper: PyReadonlyArray1<'py, f64>,
    spin_t_lower: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = spin_p.to_vec().unwrap_or_else(|_| spin_p.as_slice().unwrap().to_vec());
    let tu = spin_t_upper
        .to_vec()
        .unwrap_or_else(|_| spin_t_upper.as_slice().unwrap().to_vec());
    let tl = spin_t_lower
        .to_vec()
        .unwrap_or_else(|_| spin_t_lower.as_slice().unwrap().to_vec());
    let result = compute_binodal_impl::<HoltenPhase>(&sp, &tu, &tl, t_llcp, p_llcp);
    result.to_py_dict(py)
}

#[pyfunction]
pub fn compute_binodal_duska<'py>(
    py: Python<'py>,
    spin_p: PyReadonlyArray1<'py, f64>,
    spin_t_upper: PyReadonlyArray1<'py, f64>,
    spin_t_lower: PyReadonlyArray1<'py, f64>,
    t_llcp: f64,
    p_llcp: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = spin_p.to_vec().unwrap_or_else(|_| spin_p.as_slice().unwrap().to_vec());
    let tu = spin_t_upper
        .to_vec()
        .unwrap_or_else(|_| spin_t_upper.as_slice().unwrap().to_vec());
    let tl = spin_t_lower
        .to_vec()
        .unwrap_or_else(|_| spin_t_lower.as_slice().unwrap().to_vec());
    let result = compute_binodal_impl::<DuskaPhase>(&sp, &tu, &tl, t_llcp, p_llcp);
    result.to_py_dict(py)
}
