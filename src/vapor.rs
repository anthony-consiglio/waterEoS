//! Water vapor as a Gibbs function of temperature and pressure.
//!
//! The low-density limit of IAPWS-95: its ideal-gas part plus the second
//! and third virial coefficients of its residual part, with the density
//! series reverted to pressure so that nothing has to be iterated:
//!
//!     g / (R T) = 1 + phi0(tau, q) + b1 q + (b2 - b1^2 / 2) q^2,
//!     q = p / (rho_c R T),   tau = T_c / T.
//!
//! Mirrors vapor_eos/core.py step for step; see that module for the
//! derivation and vapor_eos/params.py for the provenance of the
//! coefficients. Units follow SeaFreeze (T in K, p in MPa, SI otherwise,
//! bulk moduli in MPa).

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

// IAPWS-95 reducing parameters
const T_C: f64 = 647.096;
const RHO_C: f64 = 322.0;
const R: f64 = 461.51805;
const P_STAR: f64 = RHO_C * R * T_C;

// Ideal-gas part
const N0_1: f64 = -8.3204464837497;
const N0_2: f64 = 6.6832105275932;
const N0_3: f64 = 3.00632;
const N0_EXP: [f64; 5] = [0.012436, 0.97315, 1.27950, 0.96956, 0.24873];
const GAMMA0: [f64; 5] =
    [1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105];

// b1(tau) = sum n tau^t: the delta^1 coefficient of the residual part,
// as (n, t). Terms 1-3, 8-10 and 23 of IAPWS-95.
const B1_TERMS: [(f64, f64); 7] = [
    (0.12533547935523e-1, -0.5),
    (0.78957634722828e1, 0.875),
    (-0.87803203303561e1, 1.0),
    (-0.66856572307965, 4.0),
    (0.20433810950965, 6.0),
    (-0.66212605039687e-4, 12.0),
    (-0.10793600908932, 7.0),
];
// b2(tau): the delta^2 coefficient. Terms 4, 5, 11, 12, 24-26, and terms
// 8-10 with the opposite sign (second order of delta exp(-delta)).
const B2_TERMS: [(f64, f64); 10] = [
    (0.31802509345418, 0.5),
    (-0.26145533859358, 0.75),
    (0.66856572307965, 4.0),
    (-0.20433810950965, 6.0),
    (0.66212605039687e-4, 12.0),
    (-0.19232721156002, 1.0),
    (-0.25709043003438, 5.0),
    (0.17611491008752e-1, 1.0),
    (0.22132295167546, 9.0),
    (-0.40247669763528, 10.0),
];

/// Number of output properties, in the order of `KEYS`.
const N_PROPS: usize = 14;
const KEYS: [&str; N_PROPS] = [
    "G", "V", "rho", "S", "H", "U", "A", "Cp", "Cv",
    "Kt", "Ks", "Kp", "alpha", "vel",
];

/// Below this many points the evaluation stays on the calling thread:
/// a point costs tens of nanoseconds, so waking the thread pool would
/// dominate small calls.
const PAR_MIN: usize = 8192;
const CHUNK: usize = 4096;

/// The temperature-only parts of the Gibbs function.
#[derive(Clone, Copy)]
struct TauTerms {
    t: f64,
    tau: f64,
    /// F = phi0 - ln(delta), with tau F' and tau^2 F''
    f: f64,
    tf: f64,
    ttf: f64,
    /// b1, tau b1', tau^2 b1''
    b1: f64,
    tb1: f64,
    ttb1: f64,
    /// k = b2 - b1^2 / 2, tau k', tau^2 k''
    k: f64,
    tk: f64,
    ttk: f64,
}

/// f, tau f' and tau^2 f'' of f = sum n tau^t, given the powers tau^t.
#[inline]
fn series(terms: &[(f64, f64)], powers: &[f64]) -> (f64, f64, f64) {
    let (mut f, mut d1, mut d2) = (0.0, 0.0, 0.0);
    for (&(n, t), &pw) in terms.iter().zip(powers) {
        let u = n * pw;
        f += u;
        d1 += t * u;
        d2 += t * (t - 1.0) * u;
    }
    (f, d1, d2)
}

#[inline]
fn tau_terms(t: f64) -> TauTerms {
    let tau = T_C / t;

    let (mut se, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for i in 0..5 {
        let x = GAMMA0[i] * tau;
        let ex = (-x).exp();
        let om = 1.0 - ex;
        se += N0_EXP[i] * (-ex).ln_1p();
        s1 += N0_EXP[i] * x * ex / om;
        s2 += N0_EXP[i] * x * x * ex / (om * om);
    }
    let f = N0_1 + N0_2 * tau + N0_3 * tau.ln() + se;
    let tf = N0_2 * tau + N0_3 + s1;
    let ttf = -N0_3 - s2;

    let s = tau.sqrt(); // tau^(1/2)
    let r4 = s.sqrt(); // tau^(1/4)
    let r8 = r4.sqrt(); // tau^(1/8)
    let t2 = tau * tau;
    let t4 = t2 * t2;
    let t5 = t4 * tau;
    let t6 = t4 * t2;
    let t12 = t6 * t6;
    // powers in the order of B1_TERMS and B2_TERMS
    let p1 = [1.0 / s, tau / r8, tau, t4, t6, t12, t6 * tau];
    let p2 = [s, s * r4, t4, t6, t12, tau, t5, tau, t5 * t4, t5 * t5];
    let (b1, tb1, ttb1) = series(&B1_TERMS, &p1);
    let (b2, tb2, ttb2) = series(&B2_TERMS, &p2);

    TauTerms {
        t,
        tau,
        f,
        tf,
        ttf,
        b1,
        tb1,
        ttb1,
        k: b2 - 0.5 * b1 * b1,
        tk: tb2 - b1 * tb1,
        ttk: ttb2 - tb1 * tb1 - b1 * ttb1,
    }
}

/// All properties at one pressure, in the order of `KEYS`.
#[inline]
fn point(tt: &TauTerms, p_mpa: f64) -> [f64; N_PROPS] {
    let t = tt.t;
    let p_pa = p_mpa * 1.0e6;
    let q = tt.tau * (p_pa / P_STAR);
    let q2 = q * q;

    let gam = 1.0 + q.ln() + tt.f + tt.b1 * q + tt.k * q2;
    let tgam = 1.0 + tt.tf + (tt.tb1 + tt.b1) * q + (tt.tk + 2.0 * tt.k) * q2;
    let ttgam = -1.0
        + tt.ttf
        + (tt.ttb1 + 2.0 * tt.tb1) * q
        + (tt.ttk + 4.0 * tt.tk + 2.0 * tt.k) * q2;

    // compressibility factor, and the two combinations that carry the
    // pressure and temperature derivatives of the volume
    let z = 1.0 + tt.b1 * q + 2.0 * tt.k * q2;
    let e = 1.0 - 2.0 * tt.k * q2;
    let a = 1.0 - tt.tb1 * q - 2.0 * (tt.tk + tt.k) * q2;

    let rt = R * t;
    let g = rt * gam;
    let h = rt * tgam;
    let s = R * (tgam - gam);
    let cp = -R * ttgam;
    let cv = cp - R * a * a / e;

    // Outside the meaning of the truncated series (and for non-positive
    // or NaN input) every property is NaN.
    if !(t > 0.0 && p_mpa > 0.0 && z > 0.0 && e > 0.0 && cv > 0.0) {
        return [f64::NAN; N_PROPS];
    }

    let v = z * rt / p_pa;
    let pv = z * rt;
    let kt = p_mpa * z / e;
    let ks = kt * cp / cv;
    [
        g,
        v,
        1.0 / v,
        s,
        h,
        h - pv,
        g - pv,
        cp,
        cv,
        kt,
        ks,
        2.0 * z / (e * e) - 1.0,
        a / (t * z),
        (ks * 1.0e6 * v).sqrt(),
    ]
}

/// Evaluate `f(i)` for i in 0..n into one column per property.
fn fill<F>(n: usize, f: F) -> Vec<Vec<f64>>
where
    F: Fn(usize) -> [f64; N_PROPS] + Sync,
{
    let mut cols: Vec<Vec<f64>> = (0..N_PROPS).map(|_| vec![0.0; n]).collect();
    if n < PAR_MIN {
        for i in 0..n {
            let pr = f(i);
            for (col, &val) in cols.iter_mut().zip(pr.iter()) {
                col[i] = val;
            }
        }
    } else {
        // Hand each task the same index range of every column.
        let mut chunks: Vec<Vec<&mut [f64]>> =
            (0..n.div_ceil(CHUNK)).map(|_| Vec::with_capacity(N_PROPS)).collect();
        for col in cols.iter_mut() {
            for (slot, part) in chunks.iter_mut().zip(col.chunks_mut(CHUNK)) {
                slot.push(part);
            }
        }
        chunks.into_par_iter().enumerate().for_each(|(ci, mut parts)| {
            let base = ci * CHUNK;
            for j in 0..parts[0].len() {
                let pr = f(base + j);
                for (part, &val) in parts.iter_mut().zip(pr.iter()) {
                    part[j] = val;
                }
            }
        });
    }
    cols
}

/// Copy of a 1-D array of any memory layout (strided views included).
fn to_vec(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

/// Vapor properties at paired points.
///
/// `t_k` and `p_mpa` are 1-D arrays of equal length; returns a dict of
/// 1-D arrays (G, V, rho, S, H, U, A, Cp, Cv, Kt, Ks, Kp, alpha, vel).
#[pyfunction]
pub fn compute_batch_vapor<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let t_vec: Vec<f64>;
    let p_vec: Vec<f64>;
    // Contiguous input is read in place; a strided view is copied.
    let t: &[f64] = match t_k.as_slice() {
        Ok(s) => s,
        Err(_) => {
            t_vec = to_vec(&t_k);
            &t_vec
        }
    };
    let p: &[f64] = match p_mpa.as_slice() {
        Ok(s) => s,
        Err(_) => {
            p_vec = to_vec(&p_mpa);
            &p_vec
        }
    };
    if t.len() != p.len() {
        return Err(PyValueError::new_err(
            "compute_batch_vapor needs T and P of equal length",
        ));
    }
    let n = t.len();

    let cols = if n < PAR_MIN {
        fill(n, |i| point(&tau_terms(t[i]), p[i]))
    } else {
        py.allow_threads(|| fill(n, |i| point(&tau_terms(t[i]), p[i])))
    };

    let dict = PyDict::new(py);
    for (key, col) in KEYS.iter().zip(cols) {
        dict.set_item(*key, Array1::from(col).into_pyarray(py))?;
    }
    Ok(dict)
}

/// Vapor properties on the grid of every pressure with every temperature.
///
/// Returns a dict of arrays of shape (len(p_mpa), len(t_k)). The
/// temperature-only terms are evaluated once per temperature.
#[pyfunction]
pub fn compute_grid_vapor<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let t = to_vec(&t_k);
    let p = to_vec(&p_mpa);
    let n_t = t.len();
    let n_p = p.len();
    let n = n_p * n_t;

    let cols = py.allow_threads(|| {
        let terms: Vec<TauTerms> = t.iter().map(|&ti| tau_terms(ti)).collect();
        fill(n, |i| point(&terms[i % n_t], p[i / n_t]))
    });

    let dict = PyDict::new(py);
    for (key, col) in KEYS.iter().zip(cols) {
        let arr = Array2::from_shape_vec((n_p, n_t), col)
            .expect("grid shape mismatch");
        dict.set_item(*key, arr.into_pyarray(py))?;
    }
    Ok(dict)
}
