"""Liquid–liquid boundaries of the symmetric Bridge29 Gibbs energy."""
import numpy as np
from scipy.optimize import brentq

from .core import PS, P_SCALE, Q, RM, _high, _low


def vapor_spinodal(T):
    t = ((np.asarray(T)**4 + 100.**4)**0.25 - 200.) / 100.
    return P_SCALE * PS(t)


@np.errstate(all="ignore")
def _field(T, P, side=0):
    delta = _low(T, P)[0] - _high(T, P)[0]
    if not side:
        return delta
    w = Q["w0"] + Q["w1"]*P
    # Continuous extension through the cusp is used only while bracketing.
    x = (1 + side*np.sqrt(np.maximum(0., 1 - 2*RM*T/w))) / 2
    return delta + RM*T*np.log(x/(1-x)) + w*(1-2*x)


def find_LLCP():
    # At criticality x=1/2, delta G=0 and W=2 R_mix T.
    def temperature(p):
        return (Q["w0"] + Q["w1"]*p)/(2*RM)
    pc = brentq(lambda p: _field(temperature(p), p), 0., 100.)
    return {"T_K": temperature(pc), "p_MPa": pc, "x": 0.5}


def _bisect(fn, left, right):
    left_sign = np.sign(fn(left))
    for _ in range(42):
        mid = (left+right)/2
        same = np.sign(fn(mid)) == left_sign
        left, right = np.where(same, mid, left), np.where(same, right, mid)
    return (left+right)/2


def _transition_temperature(P, side):
    """Warmest subcritical root along each isobar in the plotting range."""
    T = np.linspace(100., find_LLCP()["T_K"], 300)
    values = _field(T, P[:, None], side)
    crossings = values[:, :-1] * values[:, 1:] <= 0
    index = len(T)-2 - np.argmax(crossings[:, ::-1], axis=1)
    root = _bisect(lambda t: _field(t, P, side), T[index], T[index+1])
    physical = Q["w0"] + Q["w1"]*P >= 2*RM*root
    return np.where(crossings.any(axis=1) & physical, root, np.nan)


def compute_spinodal_curve(p_range_MPa):
    p = np.atleast_1d(np.asarray(p_range_MPa, float))
    return {"p_array": p, "T_lower": _transition_temperature(p, -1),
            "T_upper": _transition_temperature(p, 1)}


def compute_binodal_curve(p_range_MPa, spinodal=None):
    p = np.atleast_1d(np.asarray(p_range_MPa, float))
    return {"p_MPa": p, "T_K": _transition_temperature(p, 0)}


def liquid_lines(T_min, P_max):
    """Trace in temperature so turning points in pressure stay on the same limb."""
    cp = find_LLCP()
    tc, pc = cp["T_K"], cp["p_MPa"]
    t = np.linspace(tc, T_min, 501)[1:]
    lines = {}
    # A coarse scan selects the pressure root connected to the critical point.
    p_scan = np.arange(-400., 1001., 5.)
    for key, side in (("binodal", 0), ("hdl_limit", -1), ("ldl_limit", 1)):
        values = _field(t[:, None], p_scan, side)
        changes = values[:, :-1] * values[:, 1:] <= 0
        left = np.full_like(t, np.nan)
        last = pc
        for i, row in enumerate(changes):
            candidates = p_scan[:-1][row]
            if candidates.size:
                last = left[i] = candidates[np.argmin(abs(candidates-last))]
        p = _bisect(lambda pressure: _field(t, pressure, side), left, left+5.)
        path = np.vstack(([tc, pc], np.column_stack((t, p))))
        # A limb can leave the pressure window while the other two remain inside.
        outside = np.flatnonzero(path[:, 1] > P_max)
        if outside.size:
            i = outside[0]
            crossing = brentq(lambda temp: _field(temp, P_max, side), path[i, 0], path[i-1, 0])
            path = np.vstack((path[:i], [crossing, P_max]))
        lines[key] = path if side == 0 else [path]
    return lines
