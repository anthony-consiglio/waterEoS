"""
Branch selection for the regular-mixture two-state models.

Holten (2014), Caupin (2019) and Duska (2020) write the Gibbs energy of
mixing of the two interconvertible states as a regular solution.  After
dividing by the model's temperature factor, the equilibrium condition for
the fraction x of the low-density state B is the same in all three:

    F(x) = L + ln(x / (1 - x)) + omega * (1 - 2x) = 0

with the mixing function

    g(x) = x * L + x ln x + (1 - x) ln(1 - x) + omega * x * (1 - x).

    model      L                 omega
    holten     L                 omega
    caupin     GBA / T_hat       omega_hat
    duska      DeltaG / T_hat    omega / T_hat

For omega <= 2, g(x) is convex and F has a single root, which lies below
x = 1/2 when L > 0 and above it when L < 0.  For omega > 2, g(x) has two
inflection points x_minus < 1/2 < x_plus and up to two minima: a low-x one
in (0, x_minus) and a high-x one in (x_plus, 1).  Below the liquid-liquid
critical point both can exist at the same (T, p); the one with the higher
g is metastable and ends at its spinodal.

By default the models return the minimum with the lowest Gibbs energy,
which jumps from one minimum to the other at the liquid-liquid transition.
The functions here return a named liquid instead, so that it can be
followed smoothly through the transition into its metastable range:

    'stable'  lowest Gibbs energy (the default behavior)
    'hdl'     high-density liquid: the minimum with x <= 1/2
    'ldl'     low-density liquid:  the minimum with x >= 1/2

Each name returns NaN where its liquid does not exist.  The high-density
liquid exists on the high-density side of the liquid-liquid transition and
of its continuation beyond the critical point (the line L = 0, where the
single state has x = 1/2), and as a metastable state from the transition
down to the HDL spinodal.  The low-density liquid is the mirror image.
Wherever exactly one of the two exists it is also the stable state.
"""

import numpy as np

BRANCHES = ('stable', 'hdl', 'ldl')

_EPS = 1e-15
_MAX_ITER = 100
_X_TOL = 1e-15


def normalize_branch(branch):
    """Return the canonical branch name; raise ValueError if unknown."""
    if branch is None:
        return 'stable'
    key = str(branch).strip().lower()
    if key not in BRANCHES:
        raise ValueError(
            f"Unknown branch {branch!r}. Choose from: {', '.join(BRANCHES)}"
        )
    return key


def _F(x, L, omega):
    return L + np.log(x / (1.0 - x)) + omega * (1.0 - 2.0 * x)


def _g(x, L, omega):
    return (x * L + x * np.log(x) + (1.0 - x) * np.log(1.0 - x)
            + omega * x * (1.0 - x))


def _root_in_bracket(L, omega, a, b):
    """Root of F in [a, b], where F is increasing and F(a) <= 0 <= F(b).

    Newton iteration safeguarded by bisection: a step that leaves the
    current bracket is replaced by its midpoint, so the iteration also
    converges at a spinodal, where dF/dx vanishes at the root.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    x = 0.5 * (a + b)
    for _ in range(_MAX_ITER):
        f = _F(x, L, omega)
        pos = f > 0.0
        b = np.where(pos, x, b)
        a = np.where(pos, a, x)
        fx = 1.0 / (x * (1.0 - x)) - 2.0 * omega
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = x - f / fx
        inside = (xn > a) & (xn < b)
        xn = np.where(inside, xn, 0.5 * (a + b))
        done = np.abs(xn - x) <= _X_TOL
        x = xn
        if np.all(done):
            break
    return x


def solve_branches(L, omega):
    """Fraction x of the high-density and of the low-density liquid.

    Parameters
    ----------
    L, omega : 1-D arrays
        Field and interaction parameter in the normalized form above.

    Returns
    -------
    x_hdl, x_ldl : 1-D arrays
        Minimum of g with x <= 1/2 and minimum with x >= 1/2.  NaN where
        that minimum does not exist and where the inputs are not finite.
    """
    L = np.asarray(L, dtype=float)
    omega = np.asarray(omega, dtype=float)
    n = L.shape[0]
    finite = np.isfinite(L) & np.isfinite(omega)
    Ls = np.where(finite, L, 0.0)
    oms = np.where(finite, omega, 0.0)

    # F increases on (0, x_m] and on [x_p, 1).  For omega > 2 these are the
    # inflection points of g (the extrema of F); otherwise F increases
    # everywhere and x = 1/2, where F = L, separates the two liquids.
    double = oms > 2.0
    disc = np.sqrt(np.where(double, 1.0 - 2.0 / np.where(double, oms, 4.0), 0.0))
    x_m = np.clip(0.5 * (1.0 - disc), _EPS, 0.5)
    x_p = np.clip(0.5 * (1.0 + disc), 0.5, 1.0 - _EPS)

    has_lo = _F(x_m, Ls, oms) >= 0.0
    has_hi = _F(x_p, Ls, oms) <= 0.0

    x_lo = _root_in_bracket(Ls, oms, np.full(n, _EPS), x_m)
    x_hi = _root_in_bracket(Ls, oms, x_p, np.full(n, 1.0 - _EPS))

    x_lo = np.where(finite & has_lo, x_lo, np.nan)
    x_hi = np.where(finite & has_hi, x_hi, np.nan)
    return x_lo, x_hi


def select_branch(L, omega, branch):
    """Fraction x on a named branch.

    Parameters
    ----------
    L, omega : 1-D arrays
        Field and interaction parameter in the normalized form above.
    branch : {'stable', 'hdl', 'ldl'}

    Returns
    -------
    x : 1-D array
        Fraction on the requested branch; NaN where it does not exist.
    metastable : 1-D bool array
        True where the returned state is a local minimum but the other
        minimum has the lower Gibbs energy.
    """
    branch = normalize_branch(branch)
    x_lo, x_hi = solve_branches(L, omega)
    both = np.isfinite(x_lo) & np.isfinite(x_hi)
    with np.errstate(invalid='ignore'):
        lo_wins = np.where(both, _g(np.where(both, x_lo, 0.5), L, omega)
                           <= _g(np.where(both, x_hi, 0.5), L, omega), True)
    if branch == 'hdl':
        return x_lo, both & ~lo_wins
    if branch == 'ldl':
        return x_hi, both & lo_wins & (x_hi != x_lo)
    x = np.where(both, np.where(lo_wins, x_lo, x_hi),
                 np.where(np.isfinite(x_lo), x_lo, x_hi))
    return x, np.zeros(x.shape, dtype=bool)


def mask_mixture(result, x):
    """Set the mixture properties to NaN where the branch does not exist.

    The pure-state properties (``_A`` / ``_B`` suffixes) do not depend on x
    and are left untouched.
    """
    missing = ~np.isfinite(x)
    if not missing.any():
        return result
    for key, val in result.items():
        if key.endswith('_A') or key.endswith('_B') or key == 'metastable':
            continue
        result[key] = np.where(missing, np.nan, val)
    return result
