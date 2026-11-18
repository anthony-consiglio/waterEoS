//! Forward-mode dual numbers: a value and its derivatives along `N`
//! directions, propagated through arithmetic by the chain rule.
//!
//! The NumPy cores take the same derivatives by a complex step (evaluate at
//! x + i h with h = 1e-20 and keep Im/h). Both are exact to rounding, so a
//! derivative evaluated here agrees with its NumPy counterpart to rounding.
//! Division by a real number multiplies by its reciprocal, as NumPy divides
//! a complex array by a real one.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant: all derivatives zero.
    #[inline(always)]
    pub const fn cst(v: f64) -> Self {
        Dual { v, d: [0.0; N] }
    }

    /// The independent variable of direction `k`.
    #[inline(always)]
    pub fn var(v: f64, k: usize) -> Self {
        let mut d = [0.0; N];
        d[k] = 1.0;
        Dual { v, d }
    }

    /// f(self), given f and f' at self.v.
    #[inline(always)]
    fn chain(self, f: f64, df: f64) -> Self {
        Dual { v: f, d: self.d.map(|x| df * x) }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }

    #[inline(always)]
    pub fn ln_1p(self) -> Self {
        self.chain(self.v.ln_1p(), 1.0 / (1.0 + self.v))
    }

    #[inline(always)]
    pub fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }

    #[inline(always)]
    pub fn exp_m1(self) -> Self {
        self.chain(self.v.exp_m1(), self.v.exp())
    }

    /// x^(1/4) as exp(log(x) / 4), the complex power NumPy evaluates.
    #[inline(always)]
    pub fn quarter_power(self) -> Self {
        let r = (0.25 * self.v.ln()).exp();
        self.chain(r, 0.25 * r / self.v)
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        self * self
    }

    #[inline(always)]
    pub fn recip(self) -> Self {
        Dual::cst(1.0) / self
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Dual { v: self.v + o.v, d: std::array::from_fn(|k| self.d[k] + o.d[k]) }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Dual { v: self.v - o.v, d: std::array::from_fn(|k| self.d[k] - o.d[k]) }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Dual {
            v: self.v * o.v,
            d: std::array::from_fn(|k| self.v * o.d[k] + self.d[k] * o.v),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let v = self.v / o.v;
        Dual { v, d: std::array::from_fn(|k| (self.d[k] - v * o.d[k]) / o.v) }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual { v: -self.v, d: self.d.map(|x| -x) }
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: f64) -> Self {
        Dual { v: self.v + o, d: self.d }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: f64) -> Self {
        Dual { v: self.v - o, d: self.d }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: f64) -> Self {
        Dual { v: self.v * o, d: self.d.map(|x| x * o) }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    /// Multiplication by the reciprocal, as NumPy divides a complex array
    /// by a real number.
    #[inline(always)]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl<const N: usize> Add<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline(always)]
    fn add(self, o: Dual<N>) -> Dual<N> {
        Dual { v: self + o.v, d: o.d }
    }
}

impl<const N: usize> Sub<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline(always)]
    fn sub(self, o: Dual<N>) -> Dual<N> {
        Dual { v: self - o.v, d: o.d.map(|x| -x) }
    }
}

impl<const N: usize> Mul<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline(always)]
    fn mul(self, o: Dual<N>) -> Dual<N> {
        Dual { v: self * o.v, d: o.d.map(|x| self * x) }
    }
}

impl<const N: usize> Div<Dual<N>> for f64 {
    type Output = Dual<N>;
    #[inline(always)]
    fn div(self, o: Dual<N>) -> Dual<N> {
        Dual::cst(self) / o
    }
}
