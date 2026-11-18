//! Bridge29 equation of state and transport model — Rust implementation.
//!
//! Port of `watereos/bridge29/core.py` (Gibbs energy of the two states,
//! regular-mixture equilibrium, finite-barrier transport) and
//! `watereos/bridge29/ice.py` (IAPWS-06 ice Ih reference of the
//! low-density state). The parameters are those of
//! `watereos/data/bridge29.json`; `tests/test_bridge29_rust.py` compares
//! this backend with the NumPy one.
//!
//! The NumPy core takes the second derivatives of the high-density state
//! by a complex step through its analytic first derivatives. Here the same
//! first derivatives are evaluated on dual numbers carrying d/dT and d/dp
//! (dual.rs), which is the same derivative without the step; both are free
//! of subtractive cancellation, so the backends agree to rounding.
//!
//! Pure-state Gibbs derivatives are molar (J/mol, K, MPa). The returned
//! properties use the specific SI units of watereos, with bulk moduli in
//! MPa.

use std::ops::{Add, Div, Mul, Sub};

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::branch::{select_branch, Branch};
use crate::dual::Dual;
use crate::par::map_indexed;

/// Smallest task handed to the thread pool (see par.rs). A point costs
/// about 0.8 us, so a batch of fewer than 128 points stays on the calling
/// thread.
const PAR_GRAIN: usize = 64;

// ═══════════════════════════════════════════════════════════════════════════
// Parameters (watereos/data/bridge29.json)
// ═══════════════════════════════════════════════════════════════════════════

// Low-density state: ice Ih plus an Einstein oscillator and a quadratic
// volume.
const H_L: f64 = 1377.3715209469667;
const S_L: f64 = 1.586288424208464;
const N_L: f64 = 0.2506041387887622;
const TH_L: f64 = 455.7686039688983;
const V_L: f64 = 0.4017001993517383;
const K_L: f64 = 0.0036547291640662864;

// High-density state: spinodal backbone, oscillators and a two-level
// excitation. The terms n_H1 and n_H4 of the JSON file are zero and have
// no oscillator temperature in the NumPy core; they are left out.
const H_H: f64 = -15647.111715885247;
const S_H: f64 = -62.085975698493286;
const N_H0: f64 = 2.1914175864177485;
const N_H2: f64 = 0.394998044816933;
const N_H3: f64 = 7.806688870520927;

// Chebyshev coefficients, in t = (T_eff - 200 K) / 100 K, of the reduced
// spinodal pressure, spinodal volume, curvature and cubic coefficient.
const PS_CHEB: [f64; 3] = [-16.476275297992373, 4.908190530393981, 0.06838132520267964];
const VS_CHEB: [f64; 3] = [0.09490898148840093, -0.0046046164216466245, 0.0010524037848080204];
const PH_CHEB: [f64; 3] = [2.145175311983132, 0.17398741013997943, -0.045411054417580005];
const CS_CHEB: [f64; 3] = [1.7821543152798973, 4.111094943821056, 1.1649134148860336];

const E_X: f64 = 6213.2180148850275;
const V_X: f64 = 0.3185366889545743;
const LG_X: f64 = 2.0689886436420397;

// Interaction W = w0 + w1 p and cooperative unit N_c.
const W0: f64 = 1267.21223944121;
const W1: f64 = 1.9129529997859995;
const N_C: f64 = 2.72187158781023;

// Shape of the transport crossover f(T, p).
const T50_K: f64 = 194.3160582773401;
const WIDTH_K: f64 = 19.312214633533802;
const SLOPE_K_PER_MPA: f64 = -0.13545670620144795;
const CURV_K_PER_100MPA2: f64 = 2.503813584844022;

/// Arrhenius coefficients of one transport property: intercept, barrier of
/// the high-density limit and its increment (both in units of 1000 K), and
/// the pressure dependence of the barrier (K/MPa).
struct Barrier {
    intercept: f64,
    high_e: f64,
    increment_e: f64,
    pressure_e: f64,
}

const ETA: Barrier = Barrier {
    intercept: -13.284539631092649,
    high_e: 1.7779814610169826,
    increment_e: 3.4187922255664653,
    pressure_e: 0.21294087776077886,
};
const DIFF: Barrier = Barrier {
    intercept: -13.731437697071756,
    high_e: 1.784315877723145,
    increment_e: 2.7015103950801422,
    pressure_e: 0.156398099578009,
};
const TAU_R: Barrier = Barrier {
    intercept: -33.134869040133495,
    high_e: 1.7283905696754644,
    increment_e: 3.386043386738633,
    pressure_e: 2.6103916289053146e-37,
};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const R: f64 = 8.314462;
const RM: f64 = R / N_C;
const M: f64 = 0.018015268; // kg/mol
const P_SCALE: f64 = 22.064;
const G_SCALE: f64 = 461.523 * 647.096 * M;

/// Power-series coefficients of a degree-2 Chebyshev series
/// (T0 = 1, T1 = t, T2 = 2t^2 - 1), as `Chebyshev(c).convert(kind=Polynomial)`.
const fn cheb_to_poly(c: [f64; 3]) -> [f64; 3] {
    [c[0] - c[2], c[1], 2.0 * c[2]]
}

const fn poly_deriv(p: [f64; 3]) -> [f64; 2] {
    [p[1], 2.0 * p[2]]
}

const PS: [f64; 3] = cheb_to_poly(PS_CHEB);
const VS: [f64; 3] = cheb_to_poly(VS_CHEB);
const PHI: [f64; 3] = cheb_to_poly(PH_CHEB);
const CUBIC: [f64; 3] = cheb_to_poly(CS_CHEB);
const PS_T: [f64; 2] = poly_deriv(PS);
const VS_T: [f64; 2] = poly_deriv(VS);
const PHI_T: [f64; 2] = poly_deriv(PHI);
const CUBIC_T: [f64; 2] = poly_deriv(CUBIC);

/// Integral of VS * dPS/dt, zero at t = 0, as power-series coefficients.
const IVP: [f64; 5] = [
    0.0,
    VS[0] * PS_T[0],
    (VS[0] * PS_T[1] + VS[1] * PS_T[0]) / 2.0,
    (VS[1] * PS_T[1] + VS[2] * PS_T[0]) / 3.0,
    VS[2] * PS_T[1] / 4.0,
];

/// Power series c[0] + c[1] t + ... by Horner's rule.
#[inline(always)]
fn poly<const N: usize, const M: usize>(c: &[f64; N], t: Dual<M>) -> Dual<M> {
    let mut r = Dual::cst(c[N - 1]);
    for k in (0..N - 1).rev() {
        r = r * t + c[k];
    }
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex numbers for the IAPWS-06 ice Ih reference
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }

    #[inline(always)]
    fn ln(self) -> Cx {
        Cx::new(self.re.hypot(self.im).ln(), self.im.atan2(self.re))
    }

    #[inline(always)]
    fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }

    #[inline(always)]
    fn unscale(self, s: f64) -> Cx {
        Cx::new(self.re / s, self.im / s)
    }
}

impl Add for Cx {
    type Output = Cx;
    #[inline(always)]
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Cx {
    type Output = Cx;
    #[inline(always)]
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Cx {
    type Output = Cx;
    #[inline(always)]
    fn mul(self, o: Cx) -> Cx {
        Cx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl Div for Cx {
    type Output = Cx;
    /// Smith's algorithm, as NumPy divides complex arrays.
    #[inline(always)]
    fn div(self, o: Cx) -> Cx {
        if o.re.abs() >= o.im.abs() {
            let rat = o.im / o.re;
            let scl = 1.0 / (o.re + o.im * rat);
            Cx::new((self.re + self.im * rat) * scl, (self.im - self.re * rat) * scl)
        } else {
            let rat = o.re / o.im;
            let scl = 1.0 / (o.im + o.re * rat);
            Cx::new((self.re * rat + self.im) * scl, (self.im * rat - self.re) * scl)
        }
    }
}

// IAPWS-06 (watereos/bridge29/ice.py)
const TT: f64 = 273.16; // K
const PT: f64 = 611.657e-6; // MPa
const P0: f64 = 101325e-6 / PT;
const S0: f64 = -0.332733756492168e4 * 1e-3; // kJ/(kg K)
const GOK: [f64; 5] = [
    -0.632020233335886e6,
    0.655022213658955,
    -0.189369929326131e-7,
    0.339746123271053e-14,
    -0.556464869058991e-21,
];
const R2K: [Cx; 3] = [
    Cx::new(-0.725974574329220e2 * 1e-3, -0.781008427112870e2 * 1e-3),
    Cx::new(-0.557107698030123e-4 * 1e-3, 0.464578634580806e-4 * 1e-3),
    Cx::new(0.234801409215913e-10 * 1e-3, -0.285651142904972e-10 * 1e-3),
];
const T1: Cx = Cx::new(0.368017112855051e-1, 0.510878114959572e-1);
const T2: Cx = Cx::new(0.337315741065416, 0.335449415919309);
const R1: Cx = Cx::new(0.447050716285388e2 * 1e-3, 0.656876847463481e2 * 1e-3);

/// Molar Gibbs energy of ice Ih and its derivatives
/// (g, g_T, g_p, g_TT, g_Tp, g_pp) in J/mol, K and MPa.
#[inline]
fn ice(t: f64, p: f64) -> [f64; 6] {
    let tr_re = t * (1.0 / TT);
    let tr = Cx::new(tr_re, 0.0);
    let dp = p / PT - P0;
    let dpk = [1.0, dp, dp * dp, dp.powf(3.0), dp.powf(4.0)];

    let mut go = 0.0;
    for k in 0..5 {
        go += GOK[k] * 1e-3 * dpk[k];
    }
    let mut gop = 0.0;
    for k in 1..5 {
        gop += GOK[k] * 1e-3 * k as f64 / PT * dpk[k - 1];
    }
    let mut gopp = 0.0;
    for k in 2..5 {
        gopp += GOK[k] * 1e-3 * k as f64 * (k - 1) as f64 / (PT * PT) * dpk[k - 2];
    }
    let mut r2 = Cx::new(0.0, 0.0);
    for k in 0..3 {
        r2 = r2 + R2K[k].scale(dpk[k]);
    }
    let mut r2p = Cx::new(0.0, 0.0);
    for k in 1..3 {
        r2p = r2p + R2K[k].scale(k as f64).unscale(PT).scale(dpk[k - 1]);
    }

    let two = Cx::new(2.0, 0.0);
    let ell = |tk: Cx| -> Cx {
        let lm = tk - tr;
        let lp = tk + tr;
        lm * lm.ln() + lp * lp.ln() - two * tk * tk.ln() - tr * tr / tk
    };
    let ell_t = |tk: Cx| -> Cx {
        Cx::new(0.0, 0.0) - (tk - tr).ln() + (tk + tr).ln() - two * tr / tk
    };
    let ell_tt = |tk: Cx| -> Cx {
        let one = Cx::new(1.0, 0.0);
        one / (tk - tr) + one / (tk + tr) - two / tk
    };

    let l2 = ell(T2);
    let lt2 = ell_t(T2);
    let c = R1 * ell(T1) + r2 * l2;
    let ct = R1 * ell_t(T1) + r2 * lt2;
    let ctt = R1 * ell_tt(T1) + r2 * ell_tt(T2);
    let cp = r2p * l2;
    let ctp = r2p * lt2;

    let g = go - S0 * TT * tr_re + TT * c.re; // kJ/kg
    let gt = -S0 + ct.re; // kJ/(kg K)
    let gp = gop + TT * cp.re; // 1e-3 m3/kg
    let gtt = ctt.re / TT;
    let gtp = ctp.re;
    let gpp = gopp + TT * (R2K[2].scale(2.0).unscale(PT * PT) * l2).re;
    [g, gt, gp, gtt, gtp, gpp].map(|v| v * 1e3 * M)
}

// ═══════════════════════════════════════════════════════════════════════════
// The two pure states
// ═══════════════════════════════════════════════════════════════════════════

/// Einstein oscillator of characteristic temperature `theta`:
/// (A, dA/dT, d2A/dT2) per mole of oscillators.
#[inline(always)]
fn oscillator<const N: usize>(t: Dual<N>, theta: f64) -> (Dual<N>, Dual<N>, Dual<N>) {
    let u = theta / t;
    let log_partition = (-(-u).exp_m1()).ln();
    let population = u.exp_m1().recip();
    let a = R * t * log_partition;
    let at = R * (log_partition - u * population);
    let att = -R / t * u.square() * population * (1.0 + population);
    (a, at, att)
}

/// Low-density state: (g, g_T, g_p, g_TT, g_Tp, g_pp).
#[inline]
fn low(t: f64, p: f64) -> [f64; 6] {
    let [g, gt, gp, gtt, gtp, gpp] = ice(t, p);
    let (a, at, att) = oscillator(Dual::<0>::cst(t), TH_L);
    [
        g + H_L - t * S_L + N_L * a.v + p * (V_L - 0.5 * K_L * p),
        gt - S_L + N_L * at.v,
        gp + V_L - K_L * p,
        gtt + N_L * att.v,
        gtp,
        gpp - K_L,
    ]
}

/// High-density state: G, G_T and G_p, each with its T and p derivatives.
/// NaN beyond the liquid-vapor spinodal of the state.
#[inline]
fn high(t: f64, p: f64) -> [Dual<2>; 3] {
    const NAN: Dual<2> = Dual { v: f64::NAN, d: [f64::NAN; 2] };
    let tk = Dual::<2>::var(t, 0);
    let pm = Dual::<2>::var(p, 1);

    let t2 = tk * tk;
    let teff = (t2 * t2 + 100.0_f64.powi(4)).quarter_power();
    let x = (teff - 200.0) / 100.0;
    let r = tk / teff;
    let dt = r * (r * r) / 100.0;
    let ps = poly(&PS, x);
    let vs = poly(&VS, x);
    let phi = poly(&PHI, x);
    let c = poly(&CUBIC, x);
    let ps_t = poly(&PS_T, x);
    let vs_t = poly(&VS_T, x);
    let c_t = poly(&CUBIC_T, x);
    let b = -phi / (2.0 * ps * vs.square());
    let b_t = b * (poly(&PHI_T, x) / phi - ps_t / ps - 2.0 * vs_t / vs);
    let y = 1.0 - pm / (P_SCALE * ps);
    let valid = ps.v < 0.0
        && vs.v > 0.0
        && b.v > 0.0
        && y.v > 0.0
        && (c.v >= 0.0 || 27.0 * (c.v * c.v) * y.v < 4.0 * b.v.powf(3.0));
    if !valid {
        return [NAN; 3];
    }
    // Physical root of b d^2 + c d^3 = y, continuous from d = 0 at the
    // spinodal.
    let mut d = (y / b).sqrt();
    for _ in 0..30 {
        let correction = (d * d * (b + c * d) - y) / (d * (2.0 * b + 3.0 * c * d));
        d = d - correction;
        if correction.v.abs() <= 3e-15 * (1.0 + d.v.abs()) {
            break;
        }
    }
    let d_t = ((1.0 - y) * ps_t / ps - d * d * (b_t + c_t * d)) / (d * (2.0 * b + 3.0 * c * d));
    let logv = d.ln_1p();
    let j2 = 2.0 * (d - logv);
    let j3 = 3.0 * (0.5 * d * d - d + logv);
    let j = b * j2 + c * j3;
    let j_t = b_t * j2 + c_t * j3 + d * (2.0 * b + 3.0 * c * d) / (1.0 + d) * d_t;
    let mut g = G_SCALE * (poly(&IVP, x) - ps * vs * j) + H_H - tk * S_H;
    let mut gt = G_SCALE * dt * (vs * ps_t - (ps_t * vs + ps * vs_t) * j - ps * vs * j_t) - S_H;
    let mut gp = G_SCALE / P_SCALE * vs / (1.0 + d);
    for (theta, n) in [(100.0, N_H0), (500.0, N_H2), (1000.0, N_H3)] {
        let (a, at, _) = oscillator(tk, theta);
        g = g + n * a;
        gt = gt + n * at;
    }
    let gap = E_X + pm * V_X;
    let excitation = (LG_X - gap / (R * tk)).exp();
    let population = excitation / (1.0 + excitation);
    let log_partition = excitation.ln_1p();
    g = g - 2.0 * R * tk * log_partition;
    gt = gt - 2.0 * (R * log_partition + gap / tk * population);
    gp = gp + 2.0 * V_X * population;
    [g, gt, gp]
}

// ═══════════════════════════════════════════════════════════════════════════
// Properties and transport
// ═══════════════════════════════════════════════════════════════════════════

/// Specific properties from molar Gibbs derivatives, in the order of
/// `STATE_KEYS`.
#[inline]
fn properties(t: f64, p: f64, d: [f64; 6]) -> [f64; 13] {
    let [g, gt, gp, gtt, gtp, gpp] = d;
    let v = gp * 1e-6 / M;
    let cp = -t * gtt / M;
    let cv = cp + t / M * (gtp * gtp) / gpp;
    let kt = -gp / gpp;
    let ks = kt * cp / cv;
    let gs = g / M;
    let s = -gt / M;
    let h = gs + t * s;
    let pv = p * 1e6 * v;
    [
        gs, s, h, h - pv, gs - pv, v, 1.0 / v,
        cp, cv, kt, ks, gtp / gp, (1e6 * ks * v).sqrt(),
    ]
}

const STATE_KEYS: [&str; 13] = [
    "G", "S", "H", "U", "A", "V", "rho", "Cp", "Cv", "Kt", "Ks", "alpha", "vel",
];

/// Crossover fraction f, then eta, D and tau_r.
#[inline]
fn transport(t: f64, p: f64) -> [f64; 4] {
    let dp = p - 0.1;
    let midpoint = T50_K + SLOPE_K_PER_MPA * dp + CURV_K_PER_100MPA2 * ((dp / 100.0) * (dp / 100.0));
    let z = (midpoint - t) * 220.0 / (t * WIDTH_K);
    let f = 1.0 / (1.0 + (-z).exp());
    let rate = |c: &Barrier, sign: f64| -> f64 {
        let barrier = 1000.0 * (c.high_e + c.increment_e * f) + dp * c.pressure_e;
        (c.intercept + sign * barrier / t).exp()
    };
    [f, t / 273.15 * rate(&ETA, 1.0), rate(&DIFF, -1.0), rate(&TAU_R, 1.0)]
}

const TRANSPORT_KEYS: [&str; 4] = ["f", "eta", "D", "tau_r"];

#[inline(always)]
fn xlogx(x: f64) -> f64 {
    // scipy.special.xlogy(x, x): 0 at x = 0.
    if x == 0.0 { 0.0 } else { x * x.ln() }
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-point compute
// ═══════════════════════════════════════════════════════════════════════════

struct PointResult {
    mix: [f64; 13],
    x: f64,
    transport: [f64; 4],
    state_a: [f64; 13],
    state_b: [f64; 13],
    metastable: bool,
}

const NAN13: [f64; 13] = [f64::NAN; 13];

#[inline]
fn compute_point(t: f64, p: f64, branch: Branch) -> PointResult {
    let missing = PointResult {
        mix: NAN13,
        x: f64::NAN,
        transport: [f64::NAN; 4],
        state_a: NAN13,
        state_b: NAN13,
        metastable: false,
    };
    if !(t > 0.0) {
        return missing;
    }
    let [hg, hgt, hgp] = high(t, p);
    let hi = [hg.v, hgt.v, hgp.v, hgt.d[0], hgp.d[0], hgp.d[1]];
    let lo = low(t, p);
    if !(hi[0].is_finite() && lo[0].is_finite()) {
        return missing;
    }

    let delta = lo[0] - hi[0];
    let delta_t = lo[1] - hi[1];
    let delta_p = lo[2] - hi[2];
    let w = W0 + p * W1;
    let (x, metastable) = select_branch(delta / (RM * t), w / (RM * t), branch);

    let xx = x * (1.0 - x);
    let entropy_term = xlogx(x) + xlogx(1.0 - x);
    let curvature = RM * t / xx - 2.0 * w;
    // A response at a spinodal is singular. First-derivative properties
    // remain defined.
    let inv = if curvature.abs() < 1e-9 { f64::NAN } else { 1.0 / curvature };
    let ft = delta_t + RM * (x.clamp(1e-300, 1.0) / (1.0 - x).clamp(1e-300, 1.0)).ln();
    let fp = delta_p + W1 * (1.0 - 2.0 * x);
    let avg: [f64; 6] = std::array::from_fn(|k| (1.0 - x) * hi[k] + x * lo[k]);
    let mix = properties(t, p, [
        avg[0] + RM * t * entropy_term + w * xx,
        avg[1] + RM * entropy_term,
        avg[2] + W1 * xx,
        avg[3] - ft * ft * inv,
        avg[4] - ft * fp * inv,
        avg[5] - fp * fp * inv,
    ]);
    let mut r = PointResult {
        mix,
        x,
        transport: transport(t, p),
        state_a: properties(t, p, hi),
        state_b: properties(t, p, lo),
        metastable,
    };
    // Where the requested liquid does not exist, everything but the pure
    // states is NaN (two_state_branch.mask_mixture).
    if branch != Branch::Stable && !x.is_finite() {
        r.mix = NAN13;
        r.transport = [f64::NAN; 4];
    }
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Bridge29 properties at points (T [K], p [MPa]): the keys of
/// `watereos.bridge29.core.compute_batch`. `branch` is "stable" (default),
/// "hdl" or "ldl"; see branch.rs. With a named branch the result also has
/// `metastable`.
#[pyfunction]
#[pyo3(signature = (t_k, p_mpa, branch="stable"))]
pub fn compute_batch_bridge29<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
    branch: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let branch = Branch::parse(branch)?;
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
    if t_data.len() != p_data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "t_k and p_mpa must have the same length ({} != {})",
            t_data.len(),
            p_data.len()
        )));
    }
    let n = t_data.len();

    let points: Vec<PointResult> = py.allow_threads(|| {
        map_indexed(n, PAR_GRAIN, |i| compute_point(t_data[i], p_data[i], branch))
    });

    let dict = PyDict::new(py);
    let column = |f: &dyn Fn(&PointResult) -> f64| -> Array1<f64> {
        points.iter().map(f).collect()
    };
    for (k, key) in STATE_KEYS.iter().enumerate() {
        dict.set_item(*key, column(&|r| r.mix[k]).into_pyarray(py))?;
    }
    dict.set_item("x", column(&|r| r.x).into_pyarray(py))?;
    for (k, key) in TRANSPORT_KEYS.iter().enumerate() {
        dict.set_item(*key, column(&|r| r.transport[k]).into_pyarray(py))?;
    }
    for (k, key) in STATE_KEYS.iter().enumerate() {
        dict.set_item(format!("{key}_A"), column(&|r| r.state_a[k]).into_pyarray(py))?;
    }
    for (k, key) in STATE_KEYS.iter().enumerate() {
        dict.set_item(format!("{key}_B"), column(&|r| r.state_b[k]).into_pyarray(py))?;
    }
    if branch != Branch::Stable {
        let meta: Array1<bool> = points.iter().map(|r| r.metastable).collect();
        dict.set_item("metastable", meta.into_pyarray(py))?;
    }
    Ok(dict)
}
