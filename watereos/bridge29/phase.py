"""Liquid–liquid boundaries of the symmetric Bridge29 Gibbs energy.

``PhaseDiagram`` computes them for any coefficient set of the equation; the
module-level functions are those of Bridge29.
"""
import numpy as np
from scipy.optimize import brentq

from .core import EQUATION


def _bisect(fn, left, right):
    left_sign = np.sign(fn(left))
    for _ in range(42):
        mid = (left+right)/2
        same = np.sign(fn(mid)) == left_sign
        left, right = np.where(same, mid, left), np.where(same, right, mid)
    return (left+right)/2


class PhaseDiagram:
    """Critical point, binodal and spinodals of one ``equation.Equation``."""

    def __init__(self, equation):
        self.equation = equation
        self.vapor_spinodal = equation.vapor_spinodal

    @np.errstate(all="ignore")
    def _field(self, T, P, side=0):
        eq = self.equation
        delta = eq.low(T, P)[0] - eq.high(T, P)[0]
        if not side:
            return delta
        w = eq.Q["w0"] + eq.Q["w1"]*P
        # Continuous extension through the cusp is used only while bracketing.
        x = (1 + side*np.sqrt(np.maximum(0., 1 - 2*eq.RM*T/w))) / 2
        return delta + eq.RM*T*np.log(x/(1-x)) + w*(1-2*x)

    def _critical_temperature(self, p):
        # At criticality x=1/2, delta G=0 and W=2 R_mix T.
        return (self.equation.Q["w0"] + self.equation.Q["w1"]*p)/(2*self.equation.RM)

    def find_LLCP(self):
        pc = brentq(lambda p: self._field(self._critical_temperature(p), p), 0., 100.)
        return {"T_K": self._critical_temperature(pc), "p_MPa": pc, "x": 0.5}

    def _transition_temperature(self, P, side):
        """Warmest subcritical root along each isobar in the plotting range."""
        T = np.linspace(100., self.find_LLCP()["T_K"], 300)
        values = self._field(T, P[:, None], side)
        crossings = values[:, :-1] * values[:, 1:] <= 0
        index = len(T)-2 - np.argmax(crossings[:, ::-1], axis=1)
        root = _bisect(lambda t: self._field(t, P, side), T[index], T[index+1])
        physical = self.equation.Q["w0"] + self.equation.Q["w1"]*P >= 2*self.equation.RM*root
        return np.where(crossings.any(axis=1) & physical, root, np.nan)

    def compute_spinodal_curve(self, p_range_MPa):
        p = np.atleast_1d(np.asarray(p_range_MPa, float))
        return {"p_array": p, "T_lower": self._transition_temperature(p, -1),
                "T_upper": self._transition_temperature(p, 1)}

    def compute_binodal_curve(self, p_range_MPa, spinodal=None):
        p = np.atleast_1d(np.asarray(p_range_MPa, float))
        return {"p_MPa": p, "T_K": self._transition_temperature(p, 0)}

    def liquid_lines(self, T_min, P_max):
        """Trace in temperature so turning points in pressure stay on the same limb."""
        cp = self.find_LLCP()
        tc, pc = cp["T_K"], cp["p_MPa"]
        t = np.linspace(tc, T_min, 501)[1:]
        lines = {}
        # A coarse scan selects the pressure root connected to the critical point.
        p_scan = np.arange(-400., 1001., 5.)
        for key, side in (("binodal", 0), ("hdl_limit", -1), ("ldl_limit", 1)):
            values = self._field(t[:, None], p_scan, side)
            changes = values[:, :-1] * values[:, 1:] <= 0
            left = np.full_like(t, np.nan)
            last = pc
            for i, row in enumerate(changes):
                candidates = p_scan[:-1][row]
                if candidates.size:
                    last = left[i] = candidates[np.argmin(abs(candidates-last))]
            p = _bisect(lambda pressure: self._field(t, pressure, side), left, left+5.)
            # A line ends where one of the liquids stops existing (Local26's
            # high-density state at the fold of its expansion).
            end = np.flatnonzero(~np.isfinite(p))
            end = end[0] if end.size else len(p)
            path = np.vstack(([tc, pc], np.column_stack((t[:end], p[:end]))))
            # A limb can leave the pressure window while the other two remain inside.
            outside = np.flatnonzero(path[:, 1] > P_max)
            if outside.size:
                i = outside[0]
                crossing = brentq(lambda temp: self._field(temp, P_max, side),
                                  path[i, 0], path[i-1, 0])
                path = np.vstack((path[:i], [crossing, P_max]))
            lines[key] = path if side == 0 else [path]
        return lines


_BRIDGE29 = PhaseDiagram(EQUATION)
vapor_spinodal = _BRIDGE29.vapor_spinodal
find_LLCP = _BRIDGE29.find_LLCP
compute_spinodal_curve = _BRIDGE29.compute_spinodal_curve
compute_binodal_curve = _BRIDGE29.compute_binodal_curve
liquid_lines = _BRIDGE29.liquid_lines
