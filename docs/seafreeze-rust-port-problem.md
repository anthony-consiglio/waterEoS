# Native B-spline evaluator for SeaFreeze: problem statement

**Context.** This is a write-up of a stalled side-project in the
[waterEoS](https://github.com/anthony-consiglio/waterEoS) package. The goal
was to port SeaFreeze's tensor-product B-spline evaluator into native code
(originally Rust, but the language doesn't matter for the puzzle below) so
that we can drop the `seafreeze` / `lbftd` / `mlbspline` / `hdf5storage`
Python dependency chain and get a meaningful speedup on the H₂O phase
diagram and Kauzmann curve code paths. The **value** evaluator works; the
**derivative** evaluator is what's stuck. The blocker isn't fundamental
B-spline math — it's reproducing scipy's specific algorithmic choices
bit-exactly. I'd like a second pair of eyes from someone with FITPACK /
B-spline experience.

The current production code still calls `seafreeze.getProp(...)` for every
ice/water evaluation. That works fine; the optimisation question is just
how much we can shave by going native.

---

## 1. What SeaFreeze actually is

SeaFreeze ([Journaux et al., UW](https://github.com/Bjournaux/SeaFreeze))
ships a single `.mat` file containing, for each ice polymorph (Ih, II,
III, V, VI, VII/X) and for liquid water (`water1`, `water_IAPWS95`), a
**tensor-product B-spline parametrisation of Gibbs free energy
$G(P, T)$** with $P$ in MPa and $T$ in K, $G$ in J/kg. Splines are
quintic-by-quintic (order 6 in both dimensions) for most phases.

Every thermodynamic property comes from $G$ via standard relations:

| Quantity | Formula                                                          |
|----------|------------------------------------------------------------------|
| $V$      | $\partial G / \partial P$ (× $10^{-6}$ for SI m³/kg from MPa)    |
| $\rho$   | $1/V$                                                            |
| $S$      | $-\partial G / \partial T$                                       |
| $H$      | $G + TS$                                                         |
| $C_p$    | $-T\,\partial^2 G / \partial T^2$                                |
| $\alpha$ | $(\partial^2 G/\partial P\partial T) / (\partial G/\partial P)$  |
| $K_T$    | $-V / (\partial^2 G/\partial P^2)$                               |
| $v_{snd}$| $\sqrt{K_S/\rho}$ etc.                                           |

So a native EoS evaluator needs to compute $G$ **and** all first/second
partial derivatives w.r.t. $P$ and $T$ at arbitrary $(P, T)$ — i.e. six
distinct spline evaluations per call: $G$, $G_P$, $G_T$, $G_{PP}$,
$G_{TT}$, $G_{PT}$.

## 2. SeaFreeze's call chain (what we'd replace)

```
seafreeze.getProp(PT, phase, defpath, *tdvSpec)
  └─ mlbspline.load.loadSpline           (reads the .mat each call)
  └─ lbftd.evalGibbs._evalInternal       (dispatches on requested TDVs)
       └─ lbftd.evalGibbs.getDerivatives (one call per distinct derivative)
            └─ mlbspline.eval.evalMultivarSpline  (recursive 1-D eval)
                 └─ scipy.interpolate.splev       (FITPACK, FORTRAN)
```

The interesting bit is the bottom layer: `scipy.interpolate.splev` evaluates
a 1-D B-spline (or, with `der=k`, its $k$-th derivative) using
[Dierckx's FITPACK](https://en.wikipedia.org/wiki/FITPACK) FORTRAN library.

## 3. What I got working

I extracted the per-phase spline coefficients from SeaFreeze's `.mat`
file into a flat binary format (a few-line Python script using
`mlbspline.load.loadSpline`) and wrote a tensor-product 2-D B-spline
**value** evaluator in Rust using De Boor's algorithm
(Piegl & Tiller, "The NURBS Book", §2.4):

```python
# Sample comparison: water1 G at (P=10 MPa, T=260 K)
scipy reference: 8636.015234
rust evaluator:  8636.015234   # matches to 1e-12 relative
```

Same agreement across all 8 phases at hundreds of test points, including
near-boundary cases. The value path is solid.

## 4. Where it goes off the rails: derivatives

I implemented the standard B-spline derivative formula
([Piegl & Tiller §3.3, eq. 3.7](https://en.wikipedia.org/wiki/B-spline#Derivative_expressions)):

$$c'_i = (k-1)\,\frac{c_{i+1} - c_i}{t_{i+k} - t_{i+1}}$$

then evaluated the resulting reduced-order spline via De Boor.

The Rust output is **systematically wrong** by factors that depend on
both phase and derivative order — anywhere from ~1% off (first $P$
derivative of `water1`) to ~100% off (mixed $P, T$ derivative of ice
VII/X). Errors that big aren't roundoff; some structural choice is
different from what scipy does internally.

## 5. The puzzle: scipy itself disagrees with scipy

While trying to pin down the convention I should match, I noticed scipy's
own two paths for taking a B-spline derivative don't give the same answer.

**Minimal reproducer** (`scipy` 1.16.x, but I think this is version-stable):

```python
import numpy as np
from scipy.interpolate import splev, splder

# A cubic B-spline with clamped end knots
t = np.array([0., 0., 0., 0., 1., 2., 3., 3., 3., 3.])
c = np.array([0., 1., 2., 4., 3., 5.])
k = 3   # scipy uses degree, not order

x = 1.5

# Path A: ask splev for the derivative directly
print(splev(x, (t, c, k), der=1))
# -> 1.500000

# Path B: build the derivative spline, then splev it
t_d, c_d, k_d = splder((t, c, k), n=1)
# t_d = [0., 0., 0., 1., 2., 3., 3., 3.]  (len 8 = len(t) - 2)
# c_d = [3., 1.5, 1., 1.5, 3., 0., 0., 0.]  (len 8; only first 5 are meaningful?)
# k_d = 2
n_eff = len(t_d) - k_d - 1   # = 5
print(splev(x, (t_d, c_d[:n_eff], k_d)))
# -> 1.125000
```

`1.500 / 1.125 = 4/3 = k/(k-1)`. The factor is suspicious — it's
exactly the ratio of original degree to reduced degree.

That ratio holds at every test point I tried, for both first and second
derivatives, on splines of different orders. It's not random; it's
structural.

**SeaFreeze, via mlbspline, uses Path A** (calls `splev(...,der=...)`
directly). So whatever Path A is doing is the "ground truth" we need to
reproduce. Whatever `splder` is doing — if it's a bug, an unconventional
normalisation, or something I'm misusing — that path is not what
SeaFreeze relies on.

## 6. What I'd like help with

Three questions, in order of usefulness:

1. **Why do Path A and Path B disagree?** Is `splder` doing something
   I'm reading wrong (e.g. returning a Bernstein-like
   normalised representation rather than a B-spline that splev can
   evaluate normally)? Or is one of them buggy? The
   [scipy issue tracker](https://github.com/scipy/scipy/issues) has a
   few open threads about `splder` but none I found that explain this
   factor.

2. **What recurrence does `splev(der=k)` actually use?** Reading FITPACK's
   `splev.f` and `splder.f` directly is possible but tedious — if anyone
   already knows the recurrence in modern notation, that'd shortcut the
   whole exercise. Specifically: what's the exact formula for the
   derivative coefficient sequence such that, when fed back to the
   same De Boor evaluation as for a value, gives `splev(x, ..., der=k)`?

3. **Is there a known-good Rust crate** whose derivative convention has
   already been validated against scipy? I looked at
   [`splines`](https://crates.io/crates/splines),
   [`bspline`](https://crates.io/crates/bspline),
   [`splr`](https://crates.io/crates/splr) but didn't find one that
   advertises bit-compatibility.

## 7. Escape hatches if (1) and (2) take too long

These are paths I'd take if the convention question proves to be too
deep a hole. None of them are blocked on the scipy mystery.

- **(A) Pre-compute derivative splines at build time.** For each phase,
  use `scipy.splder` (or, more carefully, just call `splev(..., der=k)`
  on a dense grid and refit) to produce six splines per phase: $G$,
  $G_P$, $G_T$, $G_{PP}$, $G_{TT}$, $G_{PT}$. Bundle all six as raw
  data in the native binary. The native code then only ever needs the
  value evaluator (which we know works). Storage: ~6× the original 1.2
  MB → ~7 MB. Cheap.

- **(B) Pre-compute dense lookup tables.** Skip splines at runtime
  entirely: evaluate every needed thermodynamic property on a fine
  $(P, T)$ grid offline, bundle the tables, do bilinear/bicubic interp
  at runtime. Largest binary, simplest code. Accuracy is a tunable knob
  (grid density).

- **(C) Vendor a Rust B-spline crate** as in question 3. Cheapest in
  lines of code if a crate matches, but needs verification.

## 8. Repository pointers

- waterEoS source: <https://github.com/anthony-consiglio/waterEoS>
- The four SeaFreeze callsites we'd want to replace:
  - `watereos/computation.py::_compute_kauzmann_curve` (uses `.S` from ice Ih)
  - `watereos/kauzmann.py::compute_kauzmann_temperature` (same)
  - `watereos/tv_phase_diagram.py::_batch_evaluate_phases` (uses `.G` and `.rho` from all phases)
  - `watereos/watereos.py::getProp` (general dispatch for `water1` / `IAPWS95`)
- SeaFreeze: <https://github.com/Bjournaux/SeaFreeze/tree/master/Python>
- FITPACK (the FORTRAN underneath scipy.splev): <https://netlib.org/dierckx/>
- Piegl & Tiller, "The NURBS Book", 2nd ed. (1997), Springer.

## 9. State of the in-progress branch

I reverted the broken Rust derivative implementation when I gave up. The
**value evaluator** (which is correct) is not committed but is small;
I can resurrect it from this conversation's git history if helpful.
The spline-extraction Python script and the binary blob are also
recoverable from git history. Happy to share either on request.

---

*Last updated 2026-05-23.*

---

## 10. Resolution: the mismatch is coefficient padding

The minimal reproducer above is not showing two valid scipy derivative
conventions. It is feeding the legacy tuple form of `splder` a compact
coefficient vector.

For the example,

```python
len(t) == 10
k == 3
n = len(t) - k - 1 == 6
len(c) == 6
```

`splev(x, (t, c, k), der=1)` accepts this compact `c`. But the legacy
`splder((t, c, k), n=1)` implementation assumes the FITPACK padded
coefficient convention used by `splrep`, where `len(c) == len(t)` and the
extra entries are trailing zeroes. It does not pad or reject the compact
input. Instead, scipy's numpy slicing broadcasts a one-element numerator
against a five-element denominator, silently producing the wrong derivative
coefficients:

```python
# Wrong for compact c:
t_d, c_d, k_d = splder((t, c, k), n=1)
# c_d[:5] == [3.0, 1.5, 1.0, 1.5, 3.0]
```

Pad first and the two paths agree:

```python
c_pad = np.r_[c, np.zeros(len(t) - len(c))]
t_d, c_d, k_d = splder((t, c_pad, k), n=1)

print(splev(x, (t, c, k), der=1))
print(splev(x, (t_d, c_d, k_d)))
# both 1.5
```

Equivalently, use the object API:

```python
BSpline(t, c, k).derivative(1)(x)
```

`BSpline.derivative` pads the coefficient vector before delegating to the
same implementation.

### Recurrence to port

Use scipy's `k` as the spline degree `p`, not MATLAB's order. SeaFreeze /
mlbspline stores MATLAB order, so for each dimension:

```python
p = order - 1
n = len(t) - p - 1
```

For one derivative step, with compact coefficients:

```text
d_i = p * (c_{i+1} - c_i) / (t_{i+p+1} - t_{i+1}),  i = 0..n-2
t'  = t[1:-1]
p'  = p - 1
n'  = n - 1
```

Repeat that recurrence for second derivatives, decrementing `p` and trimming
the knots after each step. The denominator must be computed from the current
trimmed knot vector at each step. If an interior repeated knot makes the
denominator zero, scipy raises for `splder`; FITPACK derivative evaluation is
not defined there in the usual smooth-spline sense.

For a tensor-product SeaFreeze spline, apply this recurrence along the
requested axis only. For mixed derivatives, apply it along both axes. If the
goal is numerical equality, pre-deriving both axes and then using the value
evaluator is fine. If the goal is closest-to-bitwise agreement with
`mlbspline.evalMultivarSpline`, mirror mlbspline's evaluation order:

1. Start with `spd["coefs"]`.
2. Evaluate dimensions from last to first.
3. For each dimension, move that dimension into scipy's coefficient axis
   exactly as `_getNextSpline` does.
4. Apply the derivative recurrence for that dimension's derivative order.
5. Run the same value evaluator.

This reproduces `splev(..., der=m)` because FITPACK's derivative path is
equivalent to evaluating the reduced-degree spline generated by the recurrence
above.

### Porting implication

The Rust derivative implementation should not use `splder`'s unpadded output
as a reference unless the original coefficients are padded first. The native
formula should use the degree factor `p = order - 1`:

```text
factor = degree
denom  = knots[i + degree + 1] - knots[i + 1]
```

An error with a factor like `degree / (degree - 1)` is exactly what appears
when the derivative factor is accidentally taken from the reduced degree, or
when the unpadded `splder` tuple path is used as the expected answer.
