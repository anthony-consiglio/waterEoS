//! Native Rust evaluator for SeaFreeze GLBF tensor-product B-splines.
//!
//! Replaces seafreeze.getProp -> lbftd.evalGibbs -> mlbspline.eval ->
//! scipy.splev for the eight pure-water phases we use.
//!
//! Algorithm:
//!   * 1-D B-spline value: De Boor's algorithm (Piegl & Tiller, "The
//!     NURBS Book", §2.4).
//!   * 1-D B-spline derivative: the textbook recurrence
//!         d_i = p * (c_{i+1} - c_i) / (t_{i+p+1} - t_{i+1})
//!     with the knot vector trimmed by one entry at each end after every
//!     differentiation step. In code we sidestep the explicit trim by
//!     noting that the right-side knot index in the denominator stays
//!     pinned at (i + original_order) while only the left-side index
//!     walks right by one per iteration.
//!   * 2-D tensor-product: apply the 1-D pipeline along T first, then P,
//!     mirroring mlbspline.evalMultivarSpline's evaluation order so the
//!     floating-point accumulation matches SeaFreeze's reference.
//!
//! Property derivations: see compute_properties().

use std::collections::HashMap;
use std::sync::OnceLock;

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const SPLINE_BIN: &[u8] =
    include_bytes!("../watereos/data/seafreeze_splines.bin");

/// Working buffer size must be >= max(order_p, order_t). SeaFreeze tops
/// out at order 8 (Ice V T-direction); leave headroom.
const MAX_ORDER: usize = 12;

pub struct Spline2D {
    pub knots_p: Vec<f64>,
    pub knots_t: Vec<f64>,
    /// Row-major (n_basis_p, n_basis_t).
    pub coefs: Vec<f64>,
    pub order_p: usize,
    pub order_t: usize,
    pub n_basis_p: usize,
    pub n_basis_t: usize,
    pub shear_mod: Option<[f64; 6]>,
}

static PHASES: OnceLock<HashMap<String, Spline2D>> = OnceLock::new();

fn read_u32_le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}
fn read_u16_le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn read_f64_le(b: &[u8], i: usize) -> f64 {
    f64::from_le_bytes([
        b[i], b[i + 1], b[i + 2], b[i + 3],
        b[i + 4], b[i + 5], b[i + 6], b[i + 7],
    ])
}
fn read_f64_vec(b: &[u8], i: usize, n: usize) -> Vec<f64> {
    (0..n).map(|k| read_f64_le(b, i + k * 8)).collect()
}

fn parse_splines() -> HashMap<String, Spline2D> {
    let mut c = 0usize;
    assert_eq!(&SPLINE_BIN[c..c + 4], b"WSF1", "bad spline file magic");
    c += 4;
    let n_phases = read_u16_le(SPLINE_BIN, c) as usize;
    c += 2;

    let mut phases = HashMap::with_capacity(n_phases);
    for _ in 0..n_phases {
        let name_len = SPLINE_BIN[c] as usize;
        c += 1;
        let name = std::str::from_utf8(&SPLINE_BIN[c..c + name_len])
            .expect("invalid utf-8 phase name")
            .to_string();
        c += name_len;
        let order_p = SPLINE_BIN[c] as usize;
        c += 1;
        let order_t = SPLINE_BIN[c] as usize;
        c += 1;
        let n_knots_p = read_u32_le(SPLINE_BIN, c) as usize;
        c += 4;
        let n_knots_t = read_u32_le(SPLINE_BIN, c) as usize;
        c += 4;
        let n_basis_p = read_u32_le(SPLINE_BIN, c) as usize;
        c += 4;
        let n_basis_t = read_u32_le(SPLINE_BIN, c) as usize;
        c += 4;
        let has_shear = SPLINE_BIN[c];
        c += 1;
        let knots_p = read_f64_vec(SPLINE_BIN, c, n_knots_p);
        c += n_knots_p * 8;
        let knots_t = read_f64_vec(SPLINE_BIN, c, n_knots_t);
        c += n_knots_t * 8;
        let coefs = read_f64_vec(SPLINE_BIN, c, n_basis_p * n_basis_t);
        c += n_basis_p * n_basis_t * 8;
        let shear_mod = if has_shear != 0 {
            let s = read_f64_vec(SPLINE_BIN, c, 6);
            c += 48;
            Some([s[0], s[1], s[2], s[3], s[4], s[5]])
        } else {
            None
        };

        phases.insert(
            name,
            Spline2D {
                knots_p, knots_t, coefs, order_p, order_t,
                n_basis_p, n_basis_t, shear_mod,
            },
        );
    }
    phases
}

pub fn get_phase(name: &str) -> Option<&'static Spline2D> {
    PHASES.get_or_init(parse_splines).get(name)
}

// ─────────────────────────────────────────────────────────────────────
// Knot-span lookup
// ─────────────────────────────────────────────────────────────────────

fn find_knot_span(knots: &[f64], order: usize, x: f64) -> usize {
    let n_basis = knots.len() - order;
    let lo = order - 1;        // first span with full left-side basis support
    let hi = n_basis - 1;      // last span with full right-side basis support
    if x <= knots[lo] {
        return lo;
    }
    if x >= knots[hi + 1] {
        return hi;
    }
    let mut a = lo;
    let mut b = hi + 1;
    while b - a > 1 {
        let m = (a + b) / 2;
        if x >= knots[m] {
            a = m;
        } else {
            b = m;
        }
    }
    a
}

// ─────────────────────────────────────────────────────────────────────
// 1-D De Boor (value of an order-`order` B-spline at x in span `j`,
// given the `order` active coefficients in local[0..order])
// ─────────────────────────────────────────────────────────────────────

fn de_boor_1d(
    knots: &[f64],
    local: &[f64; MAX_ORDER],
    order: usize,
    j: usize,
    x: f64,
) -> f64 {
    let p = order - 1;
    let mut d = *local;
    // Standard De Boor recursion (Wikipedia / Piegl & Tiller §2.4):
    //   for r = 1..=p:
    //     for i = p..=r (descending):
    //       t_left  = knots[i + j - p]
    //       t_right = knots[i + j - r + 1]
    //       alpha   = (x - t_left) / (t_right - t_left)
    //       d[i]    = (1 - alpha) * d[i-1] + alpha * d[i]
    //   return d[p]
    for r in 1..=p {
        for i in (r..=p).rev() {
            let t_left = knots[i + j - p];
            let t_right = knots[i + j - r + 1];
            let denom = t_right - t_left;
            if denom > 0.0 {
                let alpha = (x - t_left) / denom;
                d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i];
            }
        }
    }
    d[p]
}

/// Apply one differentiation step in place. Reads/writes local[0..size].
///
/// * `iter_s`           1-indexed iteration count (1 for first derivative)
/// * `original_order`   the spline's original order (= scipy degree + 1)
/// * `window_start`     the GLOBAL coef index that local[0] originally was
///
/// After this call, local[0..size_after] contains the differentiated
/// coefficients where size_after = original_order - iter_s. Using the
/// "trim-equivalent" indexing: the denominator's right index stays at
/// (i + original_order); only the left index walks right with `iter_s`.
/// This sidesteps maintaining explicit trimmed knot slices.
fn differentiate_step(
    knots: &[f64],
    local: &mut [f64; MAX_ORDER],
    iter_s: usize,
    original_order: usize,
    window_start: usize,
) {
    let size_after = original_order - iter_s;
    // Current degree BEFORE this step = original_degree - (iter_s - 1)
    //                                 = (original_order - 1) - iter_s + 1
    //                                 = original_order - iter_s.
    let mult = (original_order - iter_s) as f64;
    for loc in 0..size_after {
        let global_i = window_start + loc;
        let denom = knots[global_i + original_order] - knots[global_i + iter_s];
        local[loc] = if denom > 0.0 {
            mult * (local[loc + 1] - local[loc]) / denom
        } else {
            0.0
        };
    }
    // local[size_after..] is stale.
}

// ─────────────────────────────────────────────────────────────────────
// 2-D tensor-product evaluation at a single (P, T) point
// ─────────────────────────────────────────────────────────────────────

fn eval_point(
    sp: &Spline2D,
    p_val: f64,
    t_val: f64,
    deriv_p: usize,
    deriv_t: usize,
) -> f64 {
    let order_p = sp.order_p;
    let order_t = sp.order_t;

    if deriv_p >= order_p || deriv_t >= order_t {
        return 0.0;
    }

    let j_p = find_knot_span(&sp.knots_p, order_p, p_val);
    let j_t = find_knot_span(&sp.knots_t, order_t, t_val);

    // Active windows: same start as the no-derivative case.
    let p_window_start = j_p + 1 - order_p;
    let t_window_start = j_t + 1 - order_t;

    // Step 1: for each of the order_p active P rows, evaluate the
    // T-direction value (with deriv_t T-differentiations).
    let mut row_vals = [0.0f64; MAX_ORDER];
    for a in 0..order_p {
        let row_idx = p_window_start + a;
        let row_base = row_idx * sp.n_basis_t;
        let mut local = [0.0f64; MAX_ORDER];
        for b in 0..order_t {
            local[b] = sp.coefs[row_base + t_window_start + b];
        }
        // Differentiate deriv_t times.
        for s in 1..=deriv_t {
            differentiate_step(&sp.knots_t, &mut local, s, order_t, t_window_start);
        }
        let eff_order_t = order_t - deriv_t;
        row_vals[a] = de_boor_1d(&sp.knots_t, &local, eff_order_t, j_t, t_val);
    }

    // Step 2: De Boor in P on row_vals (after optional differentiation).
    for s in 1..=deriv_p {
        differentiate_step(&sp.knots_p, &mut row_vals, s, order_p, p_window_start);
    }
    let eff_order_p = order_p - deriv_p;
    de_boor_1d(&sp.knots_p, &row_vals, eff_order_p, j_p, p_val)
}

// ─────────────────────────────────────────────────────────────────────
// Property derivations (Gibbs energy and its derivatives -> TDVs)
// ─────────────────────────────────────────────────────────────────────
//
// Unit conventions (matching SeaFreeze / lbftd):
//   G   J/kg
//   P   MPa
//   T   K
//   V   m^3/kg
//   rho kg/m^3
//   S   J/(kg K)
//   Cp  J/(kg K)
//   Kt  Pa
//   Ks  Pa
//   vel m/s
//
// Sign-and-unit notes:
//   * SeaFreeze evaluates G with P in MPa, so a "spline d/dP" is the
//     derivative w.r.t. P_MPa. To get SI V = dG/dP_SI we multiply by
//     1e-6 (1 MPa = 1e6 Pa). Two MPa-derivative powers in d2G/dP2 give
//     a 1e-12 factor; one each in d2G/dPdT gives 1e-6.

/// One Gibbs derivative directive: (deriv_p, deriv_t).
type Dir = (usize, usize);

/// Standard set of derivatives needed for the full property list.
const DIR_G: Dir = (0, 0);
const DIR_G_P: Dir = (1, 0);
const DIR_G_T: Dir = (0, 1);
const DIR_G_PP: Dir = (2, 0);
const DIR_G_TT: Dir = (0, 2);
const DIR_G_PT: Dir = (1, 1);
const DIR_G_PPP: Dir = (3, 0);  // for Kp = dKt/dP

/// All 12 properties (mass-specific water/ice) derived from G + derivs.
/// Solid ices additionally get shear/Vp/Vs from the empirical shear_mod
/// parameter set.
#[derive(Default, Clone, Copy)]
pub struct Props {
    pub g: f64,        // J/kg
    pub v: f64,        // m^3/kg
    pub rho: f64,      // kg/m^3
    pub s: f64,        // J/(kg K)
    pub h: f64,        // J/kg
    pub u: f64,        // J/kg
    pub a: f64,        // J/kg (Helmholtz)
    pub cp: f64,       // J/(kg K)
    pub cv: f64,       // J/(kg K)
    pub kt: f64,       // Pa
    pub ks: f64,       // Pa
    pub kp: f64,       // dimensionless (dKt/dP)
    pub alpha: f64,    // 1/K
    pub vel: f64,      // m/s
    pub shear: f64,    // MPa (only meaningful for solid phases)
    pub vp: f64,       // m/s
    pub vs: f64,       // m/s
}

/// SeaFreeze's shear-modulus parametrisation (matches
/// seafreeze.seafreeze._get_shear_mod_GPa):
///   sm[0] + sm[1]*(rho - sm[4]) + sm[2]*(rho - sm[4])^2 + sm[3]*(T - sm[5])
/// Returns shear modulus in GPa.
fn shear_mod_gpa(parms: &[f64; 6], rho: f64, t_k: f64) -> f64 {
    let drho = rho - parms[4];
    let dt = t_k - parms[5];
    parms[0] + parms[1] * drho + parms[2] * drho * drho + parms[3] * dt
}

fn compute_props(sp: &Spline2D, p_mpa: f64, t_k: f64) -> Props {
    let g = eval_point(sp, p_mpa, t_k, DIR_G.0, DIR_G.1);
    let g_p = eval_point(sp, p_mpa, t_k, DIR_G_P.0, DIR_G_P.1);
    let g_t = eval_point(sp, p_mpa, t_k, DIR_G_T.0, DIR_G_T.1);
    let g_pp = eval_point(sp, p_mpa, t_k, DIR_G_PP.0, DIR_G_PP.1);
    let g_tt = eval_point(sp, p_mpa, t_k, DIR_G_TT.0, DIR_G_TT.1);
    let g_pt = eval_point(sp, p_mpa, t_k, DIR_G_PT.0, DIR_G_PT.1);
    let g_ppp = eval_point(sp, p_mpa, t_k, DIR_G_PPP.0, DIR_G_PPP.1);

    // V = dG/dP_SI = dG/dP_MPa * 1e-6
    let v = g_p * 1.0e-6;
    let rho = if v != 0.0 { 1.0 / v } else { f64::NAN };

    // S = -dG/dT
    let s = -g_t;

    // alpha = (d2G/dPdT_SI) / V = (g_pt * 1e-6) / v = g_pt / g_p  (MPa cancels)
    let alpha = if g_p != 0.0 { g_pt / g_p } else { f64::NAN };

    // Kt_SI = -V / (d2G/dP_SI^2) = -V / (g_pp * 1e-12); substitute
    //         V = g_p * 1e-6  =>  Kt_SI = -(g_p / g_pp) * 1e6 Pa
    // SeaFreeze returns Kt in MPa (lbftd convention), so divide by 1e6.
    let kt_si = if g_pp != 0.0 { -g_p / g_pp * 1.0e6 } else { f64::NAN };
    let kt = kt_si / 1.0e6;  // MPa

    // Cp = -T * d2G/dT^2
    let cp = -t_k * g_tt;

    // Cp - Cv = T * V * alpha^2 * Kt_SI (all SI)
    let cp_minus_cv = t_k * v * alpha * alpha * kt_si;
    let cv = cp - cp_minus_cv;

    // Ks_SI = Kt_SI * Cp/Cv; report Ks in MPa to match SeaFreeze.
    let ks_si = if cv != 0.0 { kt_si * cp / cv } else { f64::NAN };
    let ks = ks_si / 1.0e6;  // MPa

    // Kp = pressure derivative of the isothermal bulk modulus (dimensionless).
    // From lbftd.statevars.evalPDerivIsothermalBulkModulus:
    //   Kp = d1P * d3P / d2P^2  -  1
    // where d_iP = d^i G / dP^i evaluated with P in MPa (same convention as
    // the spline). The MPa/SI scaling cancels in the ratio.
    let kp = if g_pp != 0.0 { g_p * g_ppp / (g_pp * g_pp) - 1.0 } else { f64::NAN };

    // vel = sqrt(Ks_SI / rho)   (m/s)
    let vel = if ks_si > 0.0 && rho > 0.0 {
        (ks_si / rho).sqrt()
    } else {
        f64::NAN
    };

    // Thermodynamic combinations (all SI internally)
    let p_pa = p_mpa * 1.0e6;
    let pv = p_pa * v;
    let h = g + t_k * s;
    let u = h - pv;
    let a = g - pv;

    let mut props = Props {
        g, v, rho, s, h, u, a, cp, cv, kt, ks, kp, alpha, vel,
        shear: f64::NAN, vp: f64::NAN, vs: f64::NAN,
    };

    if let Some(parms) = sp.shear_mod.as_ref() {
        let smg = shear_mod_gpa(parms, rho, t_k);   // GPa
        let ks_gpa = ks / 1.0e3;                     // ks already in MPa => GPa
        let rho_g_cm3 = rho / 1000.0;
        let vp = if rho_g_cm3 > 0.0 && (ks_gpa + 4.0 / 3.0 * smg) > 0.0 {
            1000.0 * ((ks_gpa + 4.0 / 3.0 * smg) / rho_g_cm3).sqrt()
        } else { f64::NAN };
        let vs = if rho_g_cm3 > 0.0 && smg > 0.0 {
            1000.0 * (smg / rho_g_cm3).sqrt()
        } else { f64::NAN };
        props.shear = 1.0e3 * smg;  // MPa, matching seafreeze convention
        props.vp = vp;
        props.vs = vs;
    }

    props
}

// ─────────────────────────────────────────────────────────────────────
// Python entry points
// ─────────────────────────────────────────────────────────────────────

/// All output arrays returned by `props_arrays`. Boxed into a struct
/// instead of an N-tuple so adding new fields doesn't ripple through
/// every call site.
pub struct PropsArrays {
    pub g: Vec<f64>,
    pub v: Vec<f64>,
    pub rho: Vec<f64>,
    pub s: Vec<f64>,
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub a: Vec<f64>,
    pub cp: Vec<f64>,
    pub cv: Vec<f64>,
    pub kt: Vec<f64>,
    pub ks: Vec<f64>,
    pub kp: Vec<f64>,
    pub alpha: Vec<f64>,
    pub vel: Vec<f64>,
    pub shear: Vec<f64>,
    pub vp: Vec<f64>,
    pub vs: Vec<f64>,
}

fn props_arrays<I>(sp: &Spline2D, pairs: I) -> PropsArrays
where
    I: IndexedParallelIterator<Item = (f64, f64)>,
{
    let n = pairs.len();
    let mut out = PropsArrays {
        g: vec![0.0; n], v: vec![0.0; n], rho: vec![0.0; n],
        s: vec![0.0; n], h: vec![0.0; n], u: vec![0.0; n], a: vec![0.0; n],
        cp: vec![0.0; n], cv: vec![0.0; n], kt: vec![0.0; n], ks: vec![0.0; n],
        kp: vec![0.0; n], alpha: vec![0.0; n], vel: vec![0.0; n],
        shear: vec![0.0; n], vp: vec![0.0; n], vs: vec![0.0; n],
    };

    let computed: Vec<Props> = pairs.map(|(p, t)| compute_props(sp, p, t)).collect();
    for (i, pr) in computed.iter().enumerate() {
        out.g[i] = pr.g; out.v[i] = pr.v; out.rho[i] = pr.rho;
        out.s[i] = pr.s; out.h[i] = pr.h; out.u[i] = pr.u; out.a[i] = pr.a;
        out.cp[i] = pr.cp; out.cv[i] = pr.cv;
        out.kt[i] = pr.kt; out.ks[i] = pr.ks; out.kp[i] = pr.kp;
        out.alpha[i] = pr.alpha; out.vel[i] = pr.vel;
        out.shear[i] = pr.shear; out.vp[i] = pr.vp; out.vs[i] = pr.vs;
    }
    out
}

fn fill_dict<'py>(
    py: Python<'py>,
    sp: &Spline2D,
    arrays: PropsArrays,
    shape: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let to_py = |v: Vec<f64>| -> Bound<'py, PyAny> {
        if let Some((nr, nc)) = shape {
            Array2::from_shape_vec((nr, nc), v)
                .expect("shape mismatch")
                .into_pyarray(py)
                .into_any()
        } else {
            Array1::from(v).into_pyarray(py).into_any()
        }
    };

    dict.set_item("G", to_py(arrays.g))?;
    dict.set_item("V", to_py(arrays.v))?;
    dict.set_item("rho", to_py(arrays.rho))?;
    dict.set_item("S", to_py(arrays.s))?;
    dict.set_item("H", to_py(arrays.h))?;
    dict.set_item("U", to_py(arrays.u))?;
    dict.set_item("A", to_py(arrays.a))?;
    dict.set_item("Cp", to_py(arrays.cp))?;
    dict.set_item("Cv", to_py(arrays.cv))?;
    dict.set_item("Kt", to_py(arrays.kt))?;
    dict.set_item("Ks", to_py(arrays.ks))?;
    dict.set_item("Kp", to_py(arrays.kp))?;
    dict.set_item("alpha", to_py(arrays.alpha))?;
    dict.set_item("vel", to_py(arrays.vel))?;
    if sp.shear_mod.is_some() {
        dict.set_item("shear", to_py(arrays.shear))?;
        dict.set_item("Vp", to_py(arrays.vp))?;
        dict.set_item("Vs", to_py(arrays.vs))?;
    }
    Ok(dict)
}

/// Native equivalent of seafreeze.getProp(...) for grid input.
///
/// PT is an ndarray of [P_arr, T_arr]; output arrays are shape (n_P, n_T).
#[pyfunction]
pub fn sf_getprop_grid<'py>(
    py: Python<'py>,
    phase: &str,
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = get_phase(phase).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown phase '{phase}'"))
    })?;
    let p_arr = p.as_slice()?.to_vec();
    let t_arr = t.as_slice()?.to_vec();
    let n_p = p_arr.len();
    let n_t = t_arr.len();

    let arrays = py.allow_threads(|| {
        // Build (P, T) pairs in row-major (P-fast outer, T-fast inner) order.
        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_p * n_t);
        for i in 0..n_p {
            for j in 0..n_t {
                pairs.push((p_arr[i], t_arr[j]));
            }
        }
        props_arrays(sp, pairs.into_par_iter())
    });
    fill_dict(py, sp, arrays, Some((n_p, n_t)))
}

/// Native equivalent of seafreeze.getProp(...) for scatter input
/// (matched P and T arrays; output is 1D of length n).
#[pyfunction]
pub fn sf_getprop_scatter<'py>(
    py: Python<'py>,
    phase: &str,
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let sp = get_phase(phase).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown phase '{phase}'"))
    })?;
    let p_arr = p.as_slice()?.to_vec();
    let t_arr = t.as_slice()?.to_vec();
    assert_eq!(p_arr.len(), t_arr.len(), "scatter mode needs equal-length P and T");
    let n = p_arr.len();

    let arrays = py.allow_threads(|| {
        let pairs: Vec<(f64, f64)> = (0..n).map(|i| (p_arr[i], t_arr[i])).collect();
        props_arrays(sp, pairs.into_par_iter())
    });
    fill_dict(py, sp, arrays, None)
}

/// Debug entry: raw spline derivative at a grid of points.
#[pyfunction]
pub fn sf_eval_raw<'py>(
    py: Python<'py>,
    phase: &str,
    p: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    deriv_p: usize,
    deriv_t: usize,
    grid: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let sp = get_phase(phase).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown phase '{phase}'"))
    })?;
    let p_arr = p.as_slice()?.to_vec();
    let t_arr = t.as_slice()?.to_vec();
    if grid {
        let n_p = p_arr.len();
        let n_t = t_arr.len();
        let flat: Vec<f64> = py.allow_threads(|| {
            let p_ref = &p_arr;
            let t_ref = &t_arr;
            (0..n_p)
                .into_par_iter()
                .flat_map(|i| {
                    (0..n_t)
                        .map(move |j| eval_point(sp, p_ref[i], t_ref[j], deriv_p, deriv_t))
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        let arr = Array2::from_shape_vec((n_p, n_t), flat)
            .expect("shape mismatch in raw grid eval");
        Ok(arr.into_pyarray(py).into_any())
    } else {
        assert_eq!(p_arr.len(), t_arr.len());
        let vals: Vec<f64> = py.allow_threads(|| {
            (0..p_arr.len())
                .into_par_iter()
                .map(|i| eval_point(sp, p_arr[i], t_arr[i], deriv_p, deriv_t))
                .collect()
        });
        Ok(Array1::from(vals).into_pyarray(py).into_any())
    }
}

#[pyfunction]
pub fn sf_phases(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for name in PHASES.get_or_init(parse_splines).keys() {
        d.set_item(name, true)?;
    }
    Ok(d)
}
