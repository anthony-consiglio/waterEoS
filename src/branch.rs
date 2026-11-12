//! Branch selection for the regular-mixture two-state models.
//!
//! Holten (2014), Caupin (2019) and Duska (2020) share one equilibrium
//! condition for the fraction x of the low-density state once it is divided
//! by the model's temperature factor:
//!
//!     F(x) = L + ln(x / (1 - x)) + omega * (1 - 2x) = 0
//!
//! with the mixing function
//!
//!     g(x) = x L + x ln x + (1 - x) ln(1 - x) + omega x (1 - x).
//!
//! For omega <= 2, g is convex and F has one root, below x = 1/2 when
//! L > 0 and above it when L < 0.  For omega > 2, g has inflection points
//! x_minus < 1/2 < x_plus and up to two minima: a low-x one in
//! (0, x_minus) and a high-x one in (x_plus, 1).
//!
//! `Stable` keeps each model's original solver (lowest Gibbs energy).
//! `Hdl` returns the minimum with x <= 1/2 (high-density liquid) and `Ldl`
//! the minimum with x >= 1/2 (low-density liquid), including the
//! metastable range between the liquid-liquid transition and the
//! spinodal, and NaN where that liquid does not exist.
//!
//! This mirrors `watereos/two_state_branch.py` operation for operation, so
//! the Rust and NumPy backends agree to rounding.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Branch {
    Stable,
    Hdl,
    Ldl,
}

impl Branch {
    pub fn parse(name: &str) -> PyResult<Branch> {
        match name.trim().to_ascii_lowercase().as_str() {
            "stable" => Ok(Branch::Stable),
            "hdl" => Ok(Branch::Hdl),
            "ldl" => Ok(Branch::Ldl),
            _ => Err(PyValueError::new_err(format!(
                "Unknown branch '{}'. Choose from: stable, hdl, ldl",
                name
            ))),
        }
    }
}

const EPS: f64 = 1e-15;
const MAX_ITER: usize = 100;
const X_TOL: f64 = 1e-15;

#[inline(always)]
fn f_eq(x: f64, l: f64, omega: f64) -> f64 {
    l + (x / (1.0 - x)).ln() + omega * (1.0 - 2.0 * x)
}

#[inline(always)]
fn g_mix(x: f64, l: f64, omega: f64) -> f64 {
    x * l + x * x.ln() + (1.0 - x) * (1.0 - x).ln() + omega * x * (1.0 - x)
}

/// Root of F in [a, b], where F is increasing and F(a) <= 0 <= F(b).
///
/// Newton iteration safeguarded by bisection: a step that leaves the current
/// bracket is replaced by its midpoint, so the iteration also converges at a
/// spinodal, where dF/dx vanishes at the root.
#[inline]
fn root_in_bracket(l: f64, omega: f64, a0: f64, b0: f64) -> f64 {
    let (mut a, mut b) = (a0, b0);
    let mut x = 0.5 * (a + b);
    for _ in 0..MAX_ITER {
        let f = f_eq(x, l, omega);
        if f > 0.0 {
            b = x;
        } else {
            a = x;
        }
        let fx = 1.0 / (x * (1.0 - x)) - 2.0 * omega;
        let mut xn = x - f / fx;
        if !(xn > a && xn < b) {
            xn = 0.5 * (a + b);
        }
        let done = (xn - x).abs() <= X_TOL;
        x = xn;
        if done {
            break;
        }
    }
    x
}

/// Minimum of g with x <= 1/2 and minimum with x >= 1/2.  NaN where that
/// minimum does not exist or the inputs are not finite.
#[inline]
pub fn solve_branches(l: f64, omega: f64) -> (f64, f64) {
    if !(l.is_finite() && omega.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    // F increases on (0, x_m] and on [x_p, 1).  For omega > 2 these are the
    // inflection points of g (the extrema of F); otherwise F increases
    // everywhere and x = 1/2, where F = L, separates the two liquids.
    let disc = if omega > 2.0 { (1.0 - 2.0 / omega).sqrt() } else { 0.0 };
    let x_m = (0.5 * (1.0 - disc)).max(EPS).min(0.5);
    let x_p = (0.5 * (1.0 + disc)).max(0.5).min(1.0 - EPS);
    let x_lo = if f_eq(x_m, l, omega) >= 0.0 {
        root_in_bracket(l, omega, EPS, x_m)
    } else {
        f64::NAN
    };
    let x_hi = if f_eq(x_p, l, omega) <= 0.0 {
        root_in_bracket(l, omega, x_p, 1.0 - EPS)
    } else {
        f64::NAN
    };
    (x_lo, x_hi)
}

/// Fraction x on the requested branch, and whether that state is metastable
/// (a local minimum of g that is not the lowest one).
#[inline]
pub fn select_branch(l: f64, omega: f64, branch: Branch) -> (f64, bool) {
    let (x_lo, x_hi) = solve_branches(l, omega);
    let both = x_lo.is_finite() && x_hi.is_finite();
    let lo_wins = if both {
        g_mix(x_lo, l, omega) <= g_mix(x_hi, l, omega)
    } else {
        true
    };
    match branch {
        Branch::Hdl => (x_lo, both && !lo_wins),
        Branch::Ldl => (x_hi, both && lo_wins && x_hi != x_lo),
        Branch::Stable => {
            let x = if both {
                if lo_wins { x_lo } else { x_hi }
            } else if x_lo.is_finite() {
                x_lo
            } else {
                x_hi
            };
            (x, false)
        }
    }
}

/// Set the mixture fields of a per-point result to NaN.  Used where the
/// requested branch does not exist; the pure-state fields do not depend on
/// x and are kept.
macro_rules! mask_mixture {
    ($r:expr) => {{
        $r.rho = f64::NAN;
        $r.v = f64::NAN;
        $r.s = f64::NAN;
        $r.g = f64::NAN;
        $r.h = f64::NAN;
        $r.u = f64::NAN;
        $r.a = f64::NAN;
        $r.cp = f64::NAN;
        $r.cv = f64::NAN;
        $r.kt = f64::NAN;
        $r.ks = f64::NAN;
        $r.alpha = f64::NAN;
        $r.vel = f64::NAN;
    }};
}
pub(crate) use mask_mixture;
