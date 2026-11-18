"""
Temperature-pressure phase diagram of a two-state water equation of state.

One panel, 100-375 K and -400 to 420 MPa, with:

* the liquid-liquid coexistence line, its critical point and the stability
  limits of the high- and low-density liquids;
* the Widom line, where the two liquid fractions are equal above the critical
  point;
* the liquid-vapor stability limit (where the model has one) and the
  liquid-vapor coexistence of the model liquid;
* the homogeneous nucleation line of Holten, Sengers & Anisimov (2014);
* the melting lines of ices Ih, III and V predicted by the model, where the
  liquid Gibbs energy equals that of the ice;
* the maxima and minima of density, isothermal compressibility and sound
  speed along isobars, and of isobaric heat capacity along isotherms;
* measured reference points: melting points of ices Ih, III and V, density
  maxima, warm compressibility minima and sound-speed maxima, and Troncoso's
  heat-capacity minima (bundled in ``watereos/data/phase_diagram_reference.json``).

Curves reach down to 100 K and up to 400 MPa wherever the model has them.
Lines whose temperature is not single valued in pressure (the reentrant
liquid-liquid lines of some models at low temperature) are traced as contours
on a temperature-pressure grid and refined by bisection.

Public functions
----------------
compute_eos_phase_diagram(model_key)   curves as numpy arrays
liquid_liquid_lines(model_key)         coexistence line and stability limits only
liquid_fraction(model_key, T, p, ...)  fraction x of one liquid, also on its stability limit
reference_points()                     the measured points
plot_eos_phase_diagram(model_key)      Plotly figure in the visualizer's style (needs plotly)
eos_phase_diagram_to_json(data)        JSON-ready copy of a result

Ice Ih uses the IAPWS-06 Gibbs function (IAPWS R10-06, revised 2009) without
its range check, so it extends to negative pressure and to the metastable
melting points above 210 MPa. Ices III and V use SeaFreeze (Journaux et al.
2020). Every model Gibbs energy in waterEoS shares the IAPWS-95 reference
state, as do both ice functions.
"""

import functools
import importlib.resources
import json
import warnings

import numpy as np

# Plot window. Curves are computed up to 400 MPa; the axis leaves a margin.
T_RANGE = (100.0, 375.0)
P_RANGE = (-400.0, 400.0)
PLOT_P_RANGE = (-400.0, 420.0)
ATM_PRESSURE = 0.101325  # MPa

#: model key -> (figure title, note on the fitted range)
MODELS = {
    'bridge29': ('Bridge29 (Tony Consiglio)',
                 'The low-temperature liquid branches are continuations constrained by amorphous-water data.'),
    'local26': ('Local26 (Tony Consiglio)',
                'Bridge29 refitted to liquid water at 175–300 K and −50 to 250 MPa. '
                'Curves outside that window are extrapolations.'),
    'caupin2019': (
        'Caupin and Anisimov (2019), corrected Table III',
        'The published fit uses data up to 300 K. Curves outside its fitted range are extrapolations.'),
    'caupin2019_kim': (
        'Caupin and Anisimov (2019), Table II (with Kim et al. 2017)',
        'The published fit uses data up to 300 K. Curves outside its fitted range are extrapolations.'),
    'duska2020': (
        'Duška (2020), EOS-VaT',
        'Duška targets positive pressures up to 100 MPa. The wider comparison shows extrapolations.'),
    'holten2014': (
        'Holten, Sengers, and Anisimov (2014)',
        'Negative pressures and temperatures above 300 K are extrapolated. '
        'The EOS has no liquid-vapor spinodal.'),
}

#: (key, compute_batch property, derivative axis). Axis 1 is temperature
#: (extrema along isobars), axis 0 is pressure (along isotherms), following
#: Caupin & Anisimov (2019), Fig. 12.
EXTREMA = (
    ('density', 'rho', 1),
    ('compressibility', 'kT', 1),
    ('sound_speed', 'vel', 1),
    ('heat_capacity', 'Cp', 0),
)

ICES = ('Ih', 'III', 'V')
#: A point on each measured melting line. The melting line is the Gibbs
#: equality branch that passes nearest to it (ice Ih-liquid-vapor, ice
#: Ih-III-liquid and ice III-V-liquid triple points, IAPWS R14-08).
MELTING_ANCHOR = {'Ih': (273.16, 0.0), 'III': (251.165, 209.9), 'V': (256.164, 350.1)}

# Grid spacing for the contour tracing (K, MPa).
_DT, _DP = 0.5, 1.0
# Neighboring grid cells whose liquid fraction differs by more than this lie
# on opposite sides of the liquid-liquid transition.
_FRACTION_JUMP = 0.12
# Width (MPa) above the critical point where the exact per-pressure solution
# replaces the grid tracing, which cannot resolve the thin cusp there.
_CRITICAL_WINDOW = 12.0

_cache = {}


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------

def _model(model_key):
    """Return the model's batch evaluator, phase-diagram functions and liquid-vapor spinodal."""
    if model_key not in MODELS:
        raise ValueError(f'No EoS phase diagram for model {model_key!r}; '
                         f'available: {", ".join(MODELS)}')
    if model_key in ('caupin2019', 'caupin2019_kim'):
        from caupin_eos import core, phase_diagram
        if model_key == 'caupin2019_kim':
            from caupin_eos import params_kim as pset
        else:
            from caupin_eos import params as pset
        return dict(
            batch=functools.partial(core.compute_batch, pset=pset),
            find_llcp=functools.partial(phase_diagram.find_LLCP, pset),
            spinodal=functools.partial(phase_diagram.compute_spinodal_curve, pset=pset),
            binodal=functools.partial(phase_diagram.compute_binodal_curve, pset=pset),
            lv_spinodal=lambda T: core.spinodal_pressure(T, pset=pset)[0],
            p_min=P_RANGE[0])
    if model_key in ('bridge29', 'local26'):
        core = importlib.import_module(f'watereos.{model_key}.core')
        phase = importlib.import_module(f'watereos.{model_key}.phase')
        return dict(batch=core.compute_batch, find_llcp=phase.find_LLCP,
                    spinodal=phase.compute_spinodal_curve, binodal=phase.compute_binodal_curve,
                    liquid_lines=functools.partial(phase.liquid_lines, T_RANGE[0], P_RANGE[1]),
                    lv_spinodal=phase.vapor_spinodal, p_min=P_RANGE[0])
    if model_key == 'duska2020':
        from duska_eos import core, params, phase_diagram
        return dict(
            batch=core.compute_batch,
            find_llcp=phase_diagram.find_LLCP,
            spinodal=phase_diagram.compute_spinodal_curve,
            binodal=phase_diagram.compute_binodal_curve,
            lv_spinodal=lambda T: params.p_VLCP * core.spinodal_props(T / params.T_VLCP)[0],
            p_min=P_RANGE[0])
    from holten_eos import core, phase_diagram
    # Holten (2014) has no liquid-vapor spinodal; its fit reaches -140 MPa.
    return dict(
        batch=core.compute_batch,
        find_llcp=phase_diagram.find_LLCP,
        spinodal=phase_diagram.compute_spinodal_curve,
        binodal=phase_diagram.compute_binodal_curve,
        lv_spinodal=None,
        p_min=-140.0)


def _evaluate(batch, T, P, branch='stable'):
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return batch(np.asarray(T, float), np.asarray(P, float), branch=branch)


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def _contour_zero(T, P, field, level=0.0):
    """Zero contour of field[P, T] as a list of (n, 2) arrays of (T, P)."""
    import contourpy
    lines = contourpy.contour_generator(x=T, y=P, z=np.ma.masked_invalid(field)).lines(level)
    return [np.asarray(line, float) for line in lines if len(line) > 1]


def _extrema_curves(T, P, values, axis):
    """Zero first derivatives of values[P, T] along `axis`, split by the sign of the second."""
    from scipy.interpolate import RegularGridInterpolator
    coordinates = (P, T)[axis]
    derivative = np.gradient(values, coordinates, axis=axis)
    derivative = np.where(np.isfinite(values), derivative, np.nan)
    curvature = np.gradient(derivative, coordinates, axis=axis)
    interpolate = RegularGridInterpolator((P, T), curvature, bounds_error=False, fill_value=np.nan)
    curves = {'max': [], 'min': []}
    for path in _contour_zero(T, P, derivative):
        mid = 0.5 * (path[:-1] + path[1:])
        sign = np.sign(interpolate(mid[:, ::-1]))
        starts = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1]) + 1]
        for left, right in zip(starts, np.r_[starts[1:], len(sign)]):
            if sign[left] == -1.0:
                curves['max'].append(path[left:right + 1])
            elif sign[left] == 1.0:
                curves['min'].append(path[left:right + 1])
    return curves


# ---------------------------------------------------------------------------
# Liquid-liquid lines
# ---------------------------------------------------------------------------

def _two_liquids(batch, T, P):
    """Where both liquids exist and are distinct, which ones exist, and their Gibbs energies."""
    hdl = _evaluate(batch, T, P, 'hdl')
    ldl = _evaluate(batch, T, P, 'ldl')
    has_h, has_l = np.isfinite(hdl['x']), np.isfinite(ldl['x'])
    distinct = np.abs(np.where(has_l, ldl['x'], 0.0) - np.where(has_h, hdl['x'], 1.0)) > 1e-3
    return has_h & has_l & distinct, has_h, has_l, hdl['G'], ldl['G']


def _refine_edges(batch, T, P, path, iterations=32):
    """Move each vertex of a contour of the two-liquid indicator onto the region's edge.

    A contour of a 0/1 field puts every vertex at the middle of a grid edge;
    bisection along that edge finds the boundary. Returns the refined path and
    which liquid ceases to exist just outside each vertex ('hdl' or 'ldl').
    """
    dT, dP = T[1] - T[0], P[1] - P[0]
    t, p = path[:, 0].copy(), path[:, 1].copy()
    on_t = np.abs((t - T[0]) / dT - np.round((t - T[0]) / dT)) < 1e-6
    on_p = np.abs((p - P[0]) / dP - np.round((p - P[0]) / dP)) < 1e-6
    vertical, horizontal = on_t & ~on_p, on_p & ~on_t
    a_t, a_p, b_t, b_p = t.copy(), p.copy(), t.copy(), p.copy()
    a_p[vertical] = P[0] + np.floor((p[vertical] - P[0]) / dP) * dP
    b_p[vertical] = a_p[vertical] + dP
    a_t[horizontal] = T[0] + np.floor((t[horizontal] - T[0]) / dT) * dT
    b_t[horizontal] = a_t[horizontal] + dT
    movable = vertical | horizontal
    # Put the endpoint inside the region first.
    outside = ~_two_liquids(batch, a_t, a_p)[0]
    a_t, b_t = np.where(outside, b_t, a_t), np.where(outside, a_t, b_t)
    a_p, b_p = np.where(outside, b_p, a_p), np.where(outside, a_p, b_p)
    for _ in range(iterations):
        m_t, m_p = 0.5 * (a_t + b_t), 0.5 * (a_p + b_p)
        inside = _two_liquids(batch, m_t, m_p)[0]
        a_t, a_p = np.where(inside, m_t, a_t), np.where(inside, m_p, a_p)
        b_t, b_p = np.where(inside, b_t, m_t), np.where(inside, b_p, m_p)
    refined = np.column_stack((np.where(movable, 0.5 * (a_t + b_t), t),
                               np.where(movable, 0.5 * (a_p + b_p), p)))
    _, has_h, has_l, _, _ = _two_liquids(batch, b_t, b_p)
    kind = np.where(~has_h, 'hdl', np.where(~has_l, 'ldl', ''))
    return refined, kind


def _runs(path, kind):
    """Split a path into contiguous runs of one kind."""
    edges = np.r_[0, np.flatnonzero(kind[1:] != kind[:-1]) + 1, len(kind)]
    return [(kind[a], path[a:b]) for a, b in zip(edges[:-1], edges[1:]) if kind[a] and b - a > 1]


def _liquid_liquid_lines(model, tc, pc, T, P):
    """Coexistence line and the stability limits of both liquids.

    Both liquids exist where the 'hdl' and 'ldl' branches of compute_batch are
    distinct. The stability limits are the edges of that region, and
    coexistence is where the two Gibbs energies are equal. Within
    ``_CRITICAL_WINDOW`` MPa above the critical point the model's own
    per-pressure solution replaces the grid, which cannot resolve the cusp.
    """
    batch = model['batch']
    T = T[T <= tc + 2.0]
    tt, pp = np.meshgrid(T, P)
    both, _, _, g_h, g_l = _two_liquids(batch, tt.ravel(), pp.ravel())
    both = both.reshape(tt.shape)
    scale = np.array([T[1] - T[0], P[1] - P[0]])

    difference = np.where(both & (tt < tc), (g_l - g_h).reshape(tt.shape), np.nan)
    paths = _contour_zero(T, P, difference)
    binodal = max(paths, key=len) if paths else np.empty((0, 2))

    limits = {'hdl': [], 'ldl': []}
    for path in _contour_zero(T, P, both.astype(float), level=0.5):
        refined, kind = _refine_edges(batch, T, P, path)
        for key, run in _runs(refined, kind):
            limits[key].append(run)

    # Exact solution next to the critical point.
    p_tip = pc + _CRITICAL_WINDOW * np.linspace(0.0, 1.0, 121)[1:] ** 2
    spinodal = model['spinodal'](p_tip)
    exact_binodal = model['binodal'](p_tip, spinodal=spinodal)
    exact = {'hdl': np.column_stack((spinodal['T_lower'], spinodal['p_array'])),
             'ldl': np.column_stack((spinodal['T_upper'], spinodal['p_array'])),
             'binodal': np.column_stack((exact_binodal['T_K'], exact_binodal['p_MPa']))}

    def near_tip(path):
        return (path[:, 1] < pc + _CRITICAL_WINDOW) & (path[:, 0] > tc - 15.0)

    def join(path, key):
        tip = exact[key][np.isfinite(exact[key]).all(axis=1)]
        tip = tip[(tip[:, 1] > pc) & (tip[:, 1] <= pc + _CRITICAL_WINDOW)]
        tip = tip[np.argsort(tip[:, 1])]
        kept = path[~near_tip(path)] if len(path) else path
        if not len(kept):
            return np.vstack(([tc, pc], tip))
        if len(tip) and np.hypot(*((kept[-1] - tip[-1]) / scale)) < np.hypot(*((kept[0] - tip[-1]) / scale)):
            kept = kept[::-1]
        return np.vstack(([tc, pc], tip, kept))

    out = {'binodal': join(binodal, 'binodal')}
    for key, paths in limits.items():
        joined = []
        for path in paths:
            if near_tip(path).all() and len(path) < 20:
                continue  # fragments of the unresolved cusp
            joined.append(join(path, key) if near_tip(path).any() else path)
        if not any(np.allclose(path[0], (tc, pc)) for path in joined):
            joined.append(join(np.empty((0, 2)), key))
        out[f'{key}_limit'] = joined
    return out


def _grid(model):
    """Temperature and pressure axes of the tracing grid."""
    T = np.arange(T_RANGE[0], T_RANGE[1] + _DT / 2, _DT)
    P = np.arange(model['p_min'], P_RANGE[1] + _DP / 2, _DP)
    return T, P


def liquid_liquid_lines(model_key):
    """Liquid-liquid critical point, coexistence line and stability limits of a two-state model.

    The lines are traced as contours on the diagram's temperature-pressure
    grid (100 K up, 400 MPa at most) and refined by bisection, so they follow
    the reentrant branches some models have below about 150 K, where the
    per-pressure solvers return one temperature per pressure and can jump to
    a spurious root.

    Parameters
    ----------
    model_key : str
        One of ``MODELS``.

    Returns
    -------
    dict
        ``llcp``: {'T_K', 'p_MPa'};
        ``binodal``: (n, 2) array of (T, p), starting at the critical point;
        ``hdl_limit``: list of (n, 2) arrays, the stability limit of the
        high-density liquid, below which (on cooling) it no longer exists;
        ``ldl_limit``: list of (n, 2) arrays, the stability limit of the
        low-density liquid, above which (on heating) it no longer exists.
        The piece of each limit that joins the critical point starts there.
        Temperatures in K, pressures in MPa. The result is cached; treat it
        as read-only.
    """
    key = ('liquid_liquid', model_key)
    if key not in _cache:
        model = _model(model_key)
        llcp = model['find_llcp']()
        tc, pc = float(llcp['T_K']), float(llcp['p_MPa'])
        lines = (model['liquid_lines']() if 'liquid_lines' in model else
                 _liquid_liquid_lines(model, tc, pc, *_grid(model)))
        _cache[key] = {'llcp': {'T_K': tc, 'p_MPa': pc}, **lines}
    return _cache[key]


#: Steps (K or MPa) that liquid_fraction tries, smallest first, around a point
#: just outside the region where the requested liquid exists.
_FRACTION_STEPS = (1e-6, 1e-4, 1e-2)


def liquid_fraction(model_key, T_K, p_MPa, branch, inset=0.0):
    """Fraction x that the model gives to its 'hdl' or 'ldl' liquid at each point.

    Parameters
    ----------
    model_key : str
        One of ``MODELS``.
    T_K, p_MPa : array_like
        Points, in K and MPa.
    branch : {'hdl', 'ldl'}
        Liquid whose fraction is wanted (the ``branch`` of the model's
        ``compute_batch``).
    inset : float
        For points on that liquid's own stability limit (``hdl_limit`` for
        'hdl', ``ldl_limit`` for 'ldl'). The response functions of a liquid
        (heat capacity, compressibility, expansivity) diverge on its
        stability limit, so properties evaluated with x from the limit
        itself are arbitrarily large. With ``inset`` > 0, x is taken
        ``inset`` K or MPa inside the region where the liquid exists: at
        whichever of the four points ``inset`` away along T or p has the
        liquid and gives the x farthest from its value at the limit. The
        per-pressure solvers took x 1e-3 K inside the limit for the same
        reason.

    Returns
    -------
    numpy.ndarray
        x at each point, NaN where the liquid does not exist. A point traced
        on the liquid's limit lies within about 1e-9 K or MPa of it and may
        fall just outside; there x is taken at the first of a few points
        1e-6 to 1e-2 K or MPa away along T or p that has the liquid.
    """
    if branch not in ('hdl', 'ldl'):
        raise ValueError(f"branch must be 'hdl' or 'ldl', not {branch!r}")
    batch = _model(model_key)['batch']
    T = np.atleast_1d(np.asarray(T_K, float)).ravel()
    p = np.atleast_1d(np.asarray(p_MPa, float)).ravel()
    directions = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))

    x = np.asarray(_evaluate(batch, T, p, branch)['x'], float).copy()
    for step in _FRACTION_STEPS:
        for dT, dp in directions:
            missing = ~np.isfinite(x) & np.isfinite(T) & np.isfinite(p)
            if not missing.any():
                break
            x[missing] = _evaluate(batch, T[missing] + step * dT, p[missing] + step * dp, branch)['x']
    if not inset:
        return x

    inside = np.full(x.shape, np.nan)
    depth = np.full(x.shape, -1.0)
    for dT, dp in directions:
        candidate = np.asarray(_evaluate(batch, T + inset * dT, p + inset * dp, branch)['x'], float)
        distance = np.where(np.isfinite(x), np.abs(candidate - x), 0.0)
        deeper = np.isfinite(candidate) & (distance > depth)
        inside = np.where(deeper, candidate, inside)
        depth = np.where(deeper, distance, depth)
    return inside


# ---------------------------------------------------------------------------
# Ices, nucleation, vapor
# ---------------------------------------------------------------------------

_IH_TT, _IH_PT = 273.16, 611.657e-6
_IH_P0 = 101325e-6 / _IH_PT
_IH_S0 = -0.332733756492168e4 * 1e-3
_IH_G0 = (-0.632020233335886e6, 0.655022213658955, -0.189369929326131e-7,
          0.339746123271053e-14, -0.556464869058991e-21)
_IH_R2 = (complex(-0.725974574329220e2, -0.781008427112870e2) * 1e-3,
          complex(-0.557107698030123e-4, 0.464578634580806e-4) * 1e-3,
          complex(0.234801409215913e-10, -0.285651142904972e-10) * 1e-3)
_IH_T1 = complex(0.368017112855051e-1, 0.510878114959572e-1)
_IH_T2 = complex(0.337315741065416, 0.335449415919309)
_IH_R1 = complex(0.447050716285388e2, 0.656876847463481e2) * 1e-3


def ice_ih_gibbs(T_K, p_MPa):
    """Specific Gibbs energy of ice Ih in J/kg, IAPWS R10-06 (2009), without a range check."""
    T, p = np.broadcast_arrays(np.asarray(T_K, float), np.asarray(p_MPa, float))
    tau = T.astype(complex) / _IH_TT
    dp = p / _IH_PT - _IH_P0
    g0 = sum(_IH_G0[k] * 1e-3 * dp ** k for k in range(5))
    r2 = sum(_IH_R2[k] * dp ** k for k in range(3))

    def L(t):
        return (t - tau) * np.log(t - tau) + (t + tau) * np.log(t + tau) - 2 * t * np.log(t) - tau ** 2 / t

    c = _IH_R1 * L(_IH_T1) + r2 * L(_IH_T2)
    return (g0 - _IH_S0 * _IH_TT * tau.real + _IH_TT * c.real) * 1e3


def nucleation_temperature(p_MPa):
    """Homogeneous ice nucleation temperature (K), Holten, Sengers & Anisimov (2014), Appendix A.

    Eq. A1 (ice Ih) up to 200 MPa, extended to negative pressure by the same
    equation, and Eq. A2 (ice III) above.
    """
    from scipy.optimize import brentq
    p_arr = np.atleast_1d(np.asarray(p_MPa, float))
    out = np.full(p_arr.shape, np.nan)
    for i, p in enumerate(p_arr):
        if p > 200.0:
            out[i] = 172.82 + 0.03718 * p + 3.403e-5 * p ** 2 - 1.573e-8 * p ** 3
        else:
            def a1(T, p=p):
                th = T / 235.15
                return 0.1 * (1 + 2282.7 * (1 - th ** 6.243) + 157.24 * (1 - th ** 79.81)) - p
            out[i] = brentq(a1, 140.0, 320.0)
    return out


def _melting_lines(T, P, g_liquid, valid):
    """Gibbs-equality lines of the liquid with ices Ih, III and V, one branch each."""
    from watereos._seafreeze_native import getProp_grid
    tt, pp = np.meshgrid(T, P)
    lines = {}
    for name in ICES:
        if name == 'Ih':
            g_ice = ice_ih_gibbs(tt, pp)
            use = valid
        else:
            # SeaFreeze is not trusted at negative pressure or above 280 K for these ices.
            t_win, p_win = T[T <= 280.0], P[P >= 0.0]
            g_ice = np.full(tt.shape, np.nan)
            g_ice[np.ix_(np.searchsorted(P, p_win), np.searchsorted(T, t_win))] = \
                np.asarray(getProp_grid(name, p_win, t_win).G, float)
            use = valid & (tt <= 280.0) & (pp >= 0.0)
        paths = _contour_zero(T, P, np.where(use, g_liquid - g_ice, np.nan))
        if paths:
            t0, p0 = MELTING_ANCHOR[name]
            paths = [min(paths, key=lambda c: np.min(np.hypot(c[:, 0] - t0, c[:, 1] - p0)))]
        lines[name] = paths
    return lines


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_eos_phase_diagram(model_key):
    """Curves of the EoS phase diagram of a two-state model.

    Parameters
    ----------
    model_key : str
        One of ``MODELS``: 'caupin2019', 'caupin2019_kim', 'duska2020' or
        'holten2014'.

    Returns
    -------
    dict
        ``model``, ``title``, ``note``;
        ``llcp``: {'T_K', 'p_MPa'};
        ``binodal``: (n, 2) array of (T, p), starting at the critical point;
        ``hdl_limit``, ``ldl_limit``, ``widom``: lists of (n, 2) arrays;
        ``lv_spinodal``: (n, 2) array or None;
        ``vapor_liquid``, ``nucleation``: (n, 2) arrays;
        ``melting``: {'Ih', 'III', 'V'} -> list of (n, 2) arrays;
        ``extrema``: {'density', 'compressibility', 'sound_speed',
        'heat_capacity'} -> {'max', 'min'} -> list of (n, 2) arrays.
        Temperatures in K, pressures in MPa. The result is cached; treat it
        as read-only.
    """
    if model_key in _cache:
        return _cache[model_key]
    model = _model(model_key)
    batch = model['batch']
    liquid_liquid = liquid_liquid_lines(model_key)
    tc, pc = liquid_liquid['llcp']['T_K'], liquid_liquid['llcp']['p_MPa']

    T, P = _grid(model)
    tt, pp = np.meshgrid(T, P)
    valid = np.ones(tt.shape, bool)
    lv_spinodal = None
    if model['lv_spinodal'] is not None:
        p_lv = np.asarray(model['lv_spinodal'](T), float)
        valid &= pp > p_lv
        keep = p_lv >= PLOT_P_RANGE[0] - 50.0
        lv_spinodal = np.column_stack((T[keep], p_lv[keep]))

    # The plot needs seven fields from a batch of over 400,000 states. Evaluate
    # in blocks so the other mixture and pure-state derivatives stay temporary.
    keys = ('G', 'x', 'rho', 'Kt', 'Ks', 'Cp', 'vel')
    props = {k: np.empty(tt.shape) for k in keys}
    pressures = np.where(valid, pp, 0.).ravel()
    for start in range(0, tt.size, 16384):
        sl = slice(start, start + 16384)
        block = _evaluate(batch, tt.ravel()[sl], pressures[sl])
        for k in keys:
            props[k].ravel()[sl] = block[k]
    for k in ('rho', 'Kt', 'Ks', 'Cp', 'vel'):
        valid &= np.isfinite(props[k]) & (props[k] > 0)
    props['kT'] = 1.0 / props['Kt']

    # Exclude grid cells on either side of the liquid-liquid transition.
    x = props['x']
    continuous = valid.copy()
    with np.errstate(invalid='ignore'):
        across_t = (tt[:, :-1] < tc) & (np.abs(np.diff(x, axis=1)) > _FRACTION_JUMP)
        across_p = (tt[:-1] < tc) & (np.abs(np.diff(x, axis=0)) > _FRACTION_JUMP)
    continuous[:, :-1] &= ~across_t
    continuous[:, 1:] &= ~across_t
    continuous[:-1] &= ~across_p
    continuous[1:] &= ~across_p

    extrema = {key: _extrema_curves(T, P, np.where(continuous, props[prop], np.nan), axis)
               for key, prop, axis in EXTREMA}
    widom = _contour_zero(T, P, np.where(continuous & (tt >= tc), x - 0.5, np.nan))

    from vapor_eos.vapor_eos import vapor_pressure
    t_vl = np.arange(T_RANGE[0], T_RANGE[1] + 0.5, 1.0)
    vapor_liquid = np.column_stack((t_vl, vapor_pressure(t_vl, model_key)))
    p_nuc = np.arange(-140.0, P_RANGE[1] + 2.5, 5.0)
    nucleation = np.column_stack((nucleation_temperature(p_nuc), p_nuc))

    title, note = MODELS[model_key]
    result = {
        'model': model_key, 'title': title, 'note': note,
        'llcp': {'T_K': tc, 'p_MPa': pc},
        'binodal': liquid_liquid['binodal'],
        'hdl_limit': liquid_liquid['hdl_limit'],
        'ldl_limit': liquid_liquid['ldl_limit'],
        'widom': widom,
        'lv_spinodal': lv_spinodal,
        'vapor_liquid': vapor_liquid,
        'nucleation': nucleation,
        'melting': _melting_lines(T, P, props['G'], valid),
        'extrema': extrema,
    }
    _cache[model_key] = result
    return result


def reference_points():
    """Measured points drawn on the phase diagram (bundled JSON, see its 'about' entry)."""
    if 'reference' not in _cache:
        path = importlib.resources.files('watereos') / 'data' / 'phase_diagram_reference.json'
        _cache['reference'] = json.loads(path.read_text())
    return _cache['reference']


def _rounded(path, digits=3):
    return {'T_K': np.round(path[:, 0], digits).tolist(), 'p_MPa': np.round(path[:, 1], digits).tolist()}


def eos_phase_diagram_to_json(data):
    """JSON-ready copy of a compute_eos_phase_diagram result: each path becomes {'T_K': [...], 'p_MPa': [...]}."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return _rounded(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    return convert(data)




# ---------------------------------------------------------------------------
# Plotly figure in the visualizer's style
# ---------------------------------------------------------------------------

#: Layers of the figure, in drawing order: key -> legend label.
LAYERS = {
    'binodal': 'Liquid–liquid coexistence',
    'LLCP': 'Liquid–liquid critical point',
    'hdl_spinodal': 'HDL stability limit',
    'ldl_spinodal': 'LDL stability limit',
    'widom': 'Widom line (equal fractions)',
    'lv_spinodal': 'Liquid–vapor stability limit',
    'vapor_liquid': 'Vapor–liquid coexistence',
    'nucleation': 'Homogeneous nucleation',
    'kauzmann': 'Kauzmann line (S liquid = S ice Ih)',
    'ice_ih': 'Ice Ih melting (model and data)',
    'ice_iii': 'Ice III melting (model and data)',
    'ice_v': 'Ice V melting (model and data)',
    'density': 'Density extrema (isobars)',
    'tmd_data': 'Measured density maxima',
    'compressibility': 'Compressibility extrema (isobars)',
    'sound_speed': 'Sound speed extrema (isobars)',
    'heat_capacity': 'Heat capacity extrema (isotherms)',
    'warm_extrema': 'Measured warm extrema',
    'cp_minima': 'Troncoso heat-capacity minima',
}
#: Layers drawn by default: all but the Kauzmann line.
DEFAULT_LAYERS = tuple(key for key in LAYERS if key != 'kauzmann')
#: Older layer names that map onto current ones.
LAYER_ALIASES = {'tmd': 'density', 'nuc_ih': 'nucleation', 'nuc_iii': 'nucleation'}

# Line and marker colors for each visualizer theme. The light colors are those
# of the EoS-fitting project's figures; the dark ones are lighter versions that
# keep the same hues readable on the dark background.
_PALETTES = {
    'light': {
        'liquid_liquid': '#16324f', 'lv_spinodal': '#595959', 'vapor_liquid': '#5773a4',
        'nucleation': '#b0b0b0', 'kauzmann': '#b04a5a', 'ice_ih_data': '#a6a6a6',
        'ice_ih': '#5b8fb0', 'ice_iii': '#137f85', 'ice_v': '#a77b13',
        'density': '#946438', 'compressibility': '#327844', 'sound_speed': '#ca6a1c',
        'heat_capacity': '#903f91',
    },
    'dark': {
        'liquid_liquid': '#a9c4ea', 'lv_spinodal': '#a3a3ab', 'vapor_liquid': '#8ea8d8',
        'nucleation': '#686c74', 'kauzmann': '#e07a8a', 'ice_ih_data': '#7a7e86',
        'ice_ih': '#7fb2d6', 'ice_iii': '#2fb3b9', 'ice_v': '#d1a531',
        'density': '#c4915e', 'compressibility': '#5fb27a', 'sound_speed': '#ec8a3c',
        'heat_capacity': '#c27ac4',
    },
}
_ICE_LAYER = {'Ih': 'ice_ih', 'III': 'ice_iii', 'V': 'ice_v'}
_ICE_SYMBOL = {'III': 'square', 'V': 'triangle-up'}


def _normalize_layers(show):
    if show is None:
        return list(DEFAULT_LAYERS)
    keys = [LAYER_ALIASES.get(key, key) for key in show]
    return [key for key in LAYERS if key in keys]


def _joined(paths, digits=2):
    """x and y lists of several paths in one trace, separated by gaps."""
    xs, ys = [], []
    for path in paths:
        if len(path):
            xs += np.round(path[:, 0], digits).tolist() + [None]
            ys += np.round(path[:, 1], digits).tolist() + [None]
    return xs[:-1], ys[:-1]


def plot_eos_phase_diagram(model_key, show=None, theme='light', T_range=None, P_range=None):
    """EoS phase diagram as a Plotly figure in the waterEoS visualizer's style.

    Parameters
    ----------
    model_key : str
        A key of ``MODELS``.
    show : iterable of str, optional
        Layers to draw, keys of ``LAYERS`` (older names in ``LAYER_ALIASES``
        are accepted). Defaults to ``DEFAULT_LAYERS``. Each layer is one
        legend entry; clicking it in the legend hides the whole layer.
    theme : {'light', 'dark'}
        Visualizer theme (``watereos.plot_theme``).
    T_range, P_range : (min, max), optional
        Axis ranges; default 100-375 K and -400 to 420 MPa.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from watereos.plot_theme import THEMES, apply_theme

    if theme not in THEMES:
        raise ValueError(f'unknown theme: {theme!r}')
    layers = _normalize_layers(show)
    data = compute_eos_phase_diagram(model_key)
    ref = reference_points()
    colors, tokens = _PALETTES[theme], THEMES[theme]
    fill = tokens['bg_elev']
    fig = go.Figure()

    def lines(key, paths, dash='solid', width=1.6, name=None, legend=True, color=None):
        x, y = _joined(paths)
        label = name or LAYERS[key]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', name=label, legendgroup=key, showlegend=legend,
            line=dict(color=color or colors[key], dash=dash, width=width),
            hovertemplate=f'{label}<br>%{{x:.1f}} K, %{{y:.1f}} MPa<extra></extra>'))

    def markers(key, T, p, symbol, size, color, sources=None, name=None, legend=False, filled=False):
        label = name or LAYERS[key]
        hover = '%{customdata}<br>' if sources is not None else ''
        fig.add_trace(go.Scatter(
            x=list(T), y=list(p), mode='markers', name=label, legendgroup=key, showlegend=legend,
            customdata=None if sources is None else list(sources),
            marker=dict(symbol=symbol, size=size, color=color if filled else fill,
                        line=dict(color=color, width=1.2)),
            hovertemplate=f'{label}<br>{hover}%{{x:.2f}} K, %{{y:.1f}} MPa<extra></extra>'))

    tc, pc = data['llcp']['T_K'], data['llcp']['p_MPa']
    for key in layers:
        if key == 'binodal':
            lines(key, [data['binodal']], width=2.6, color=colors['liquid_liquid'])
        elif key == 'LLCP':
            c = colors['liquid_liquid']
            fig.add_trace(go.Scatter(
                x=[tc], y=[pc], mode='markers', name=LAYERS[key], legendgroup=key,
                marker=dict(color=c, size=10, line=dict(color=fill, width=1)),
                hovertemplate=f'{LAYERS[key]}<br>%{{x:.2f}} K, %{{y:.2f}} MPa<extra></extra>'))
            fig.add_annotation(x=tc, y=pc, ax=125.0, ay=-90.0, axref='x', ayref='y',
                               text=f'critical point<br>{tc:.1f} K, {pc:.0f} MPa',
                               showarrow=True, arrowhead=0, arrowwidth=0.8, arrowcolor=c,
                               font=dict(color=c, size=11), align='left')
        elif key in ('hdl_spinodal', 'ldl_spinodal'):
            lines(key, data[key.replace('spinodal', 'limit')], dash='dot', width=1.8,
                  color=colors['liquid_liquid'])
        elif key == 'widom':
            lines(key, data['widom'], dash='dashdot', width=1.5, color=colors['liquid_liquid'])
        elif key == 'lv_spinodal':
            if data['lv_spinodal'] is not None:
                lines(key, [data['lv_spinodal']], width=1.4)
        elif key == 'vapor_liquid':
            lines(key, [data['vapor_liquid']], dash='dashdot', width=1.4)
        elif key == 'nucleation':
            lines(key, [data['nucleation']], width=1.4)
        elif key == 'kauzmann':
            from watereos.computation import compute_phase_diagram_data
            curve = compute_phase_diagram_data(model_key).get('kauzmann')
            if curve is not None:
                lines(key, [np.column_stack((curve['T_K'], curve['p_MPa']))], dash='longdash', width=1.6)
        elif key.startswith('ice_'):
            name = {v: k for k, v in _ICE_LAYER.items()}[key]
            points = ref['melting'][name]
            lines(key, data['melting'][name], width=1.8)
            if name == 'Ih':
                markers(key, points['T_K'], points['p_MPa'], 'circle', 3.5, colors['ice_ih_data'],
                        points['source_key'], filled=True)
            else:
                markers(key, points['T_K'], points['p_MPa'], _ICE_SYMBOL[name], 6, colors[key],
                        points['source_key'])
        elif key in data['extrema']:
            curves = data['extrema'][key]
            lines(key, curves['max'], width=1.6, legend=bool(curves['max']))
            lines(key, curves['min'], dash='dash', width=1.6, legend=not curves['max'])
        elif key == 'tmd_data':
            points = ref['density_maxima']
            markers(key, points['T_K'], points['p_MPa'], 'square', 5, colors['density'],
                    points['source_key'], legend=True)
        elif key == 'warm_extrema':
            first = True
            for entry in ref['warm_extrema'].values():
                markers(key, entry['T_K'], entry['p_MPa'], 'circle', 8, colors[entry['property']],
                        legend=first)
                first = False
            for entry in ref['ambient_extrema'].values():
                markers(key, [entry['T_K']], [entry['p_MPa']], 'circle', 11, colors[entry['property']])
        elif key == 'cp_minima':
            cp = ref['heat_capacity_minima']
            inside = np.asarray(cp['in_measured_range'], bool)
            T, p = np.asarray(cp['T_K']), np.asarray(cp['p_MPa'])
            markers(key, T[inside], p[inside], 'diamond', 8, colors['heat_capacity'], legend=True, filled=True)
            markers(key, T[~inside], p[~inside], 'diamond', 8, colors['heat_capacity'],
                    name=f'{LAYERS[key]} (below the measured range)')

    if any(key in data['extrema'] for key in layers):
        for label, dash in (('Response maximum', 'solid'), ('Response minimum', 'dash')):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name=label, legendgroup=f'key_{dash}',
                                     line=dict(color=tokens['text_muted'], dash=dash, width=1.6),
                                     hoverinfo='skip'))

    apply_theme(fig, theme)
    fig.update_layout(
        title=dict(text=f"{data['title']}<br><span style='font-size:11px;color:{tokens['text_muted']}'>"
                        f"{data['note']}</span>", x=0.0, xanchor='left', y=0.98, yanchor='top'),
        # Legend below the plot in three columns, as in the EoS-fitting figures.
        margin=dict(l=64, r=24, t=72, b=48),
        legend=dict(orientation='h', x=0.0, xanchor='left', y=-0.13, yanchor='top',
                    entrywidth=0.3, entrywidthmode='fraction', tracegroupgap=0,
                    font=dict(size=11)),
        xaxis_title='Temperature [K]', yaxis_title='Pressure [MPa]',
        hovermode='closest',
    )
    fig.update_xaxes(range=list(T_range or T_RANGE), ticks='inside', mirror='ticks', showline=True)
    fig.update_yaxes(range=list(P_range or PLOT_P_RANGE), ticks='inside', mirror='ticks', showline=True)
    return fig
