"""
Temperature-Volume (T-V) phase diagram for water and ice phases.

Reproduces and extends Fig. 1c from Powell-Palm (2022), RSC Advances, 12,
4458-4469.  Uses isothermal 2D convex hulls in Helmholtz free energy (A) vs
specific volume (V) space to determine phase equilibria.

Algorithm
---------
For each temperature T:
1. Sample A(V) for each phase via SeaFreeze across a range of pressures
2. Collect all (V, A) points, compute the 2D convex hull
3. Extract the lower envelope (minimum A at each V) = stable states
4. Contiguous same-phase points on the lower hull = single-phase regions
5. Gaps between phases on the lower hull = two-phase coexistence
6. Coexistence pressure P = -dA/dV (slope of common tangent)
7. Track topology changes between temperatures for three-phase invariants

References
----------
- Powell-Palm (2022), RSC Advances, 12, 4458-4469
- Verwiebe (1939), Am. J. Phys., 7, 187-194
- Journaux et al. (2020), J. Geophys. Res.: Planets, 125, e2019JE006176
"""

import io
import warnings
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------
PHASE_CONFIGS = {
    'water1': {
        'label': 'Liquid',
        'color': '#2196F3',
        'P_min': 0.1,
        'P_max': 2300.0,
        'T_min': 200.0,
        'T_max': 500.0,
        'n_P': 600,
    },
    'Ih': {
        'label': 'Ice Ih',
        'color': '#90CAF9',
        'P_min': 0.1,
        'P_max': 400.0,
        'T_min': 0.0,
        'T_max': 280.0,
        'n_P': 400,
    },
    'II': {
        'label': 'Ice II',
        'color': '#FF9800',
        'P_min': 50.0,
        'P_max': 900.0,
        'T_min': 0.0,
        'T_max': 350.0,
        'n_P': 400,
    },
    'III': {
        'label': 'Ice III',
        'color': '#4CAF50',
        'P_min': 200.0,
        'P_max': 500.0,
        'T_min': 0.0,
        'T_max': 280.0,
        'n_P': 400,
    },
    'V': {
        'label': 'Ice V',
        'color': '#F44336',
        'P_min': 300.0,
        'P_max': 900.0,
        'T_min': 0.0,
        'T_max': 350.0,
        'n_P': 400,
    },
    'VI': {
        'label': 'Ice VI',
        'color': '#9C27B0',
        'P_min': 600.0,
        'P_max': 2200.0,
        'T_min': 0.0,
        'T_max': 370.0,
        'n_P': 500,
    },
    'vapor': {
        'label': 'Vapor',
        'color': '#E0E0E0',
    },
}

DEFAULT_PHASES = ['water1', 'Ih', 'II', 'III', 'V', 'VI']

# ---------------------------------------------------------------------------
# SeaFreeze interface
# ---------------------------------------------------------------------------
_sf_func = None


def _get_sf():
    """Lazy import of SeaFreeze getProp."""
    global _sf_func
    if _sf_func is None:
        from seafreeze.seafreeze import getProp
        _sf_func = getProp
    return _sf_func


from contextlib import contextmanager

@contextmanager
def _suppress():
    """Context manager to suppress SeaFreeze stdout/stderr/warnings."""
    with warnings.catch_warnings(), \
         redirect_stdout(io.StringIO()), \
         redirect_stderr(io.StringIO()):
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# IAPWS-95 saturation and low-pressure liquid
# ---------------------------------------------------------------------------
def _compute_saturation_curve(T_min=273.16, T_max=500.0, n_T=200):
    """Compute liquid-vapor saturation properties via IAPWS-95.

    Returns dict with arrays: T (K), P_sat (MPa), V_liq (m^3/kg),
    V_vap (m^3/kg).
    """
    from iapws import IAPWS95

    T_arr = np.linspace(T_min, min(T_max, 646.0), n_T)
    T_valid, P_sat, V_liq, V_vap = [], [], [], []

    for T in T_arr:
        try:
            sat_liq = IAPWS95(T=T, x=0)
            sat_vap = IAPWS95(T=T, x=1)
            if (sat_liq.v is not None and sat_vap.v is not None
                    and np.isfinite(sat_liq.v) and np.isfinite(sat_vap.v)):
                T_valid.append(T)
                P_sat.append(sat_liq.P)
                V_liq.append(sat_liq.v)
                V_vap.append(sat_vap.v)
        except (NotImplementedError, ValueError, RuntimeError):
            continue

    return {
        'T': np.array(T_valid),
        'P_sat': np.array(P_sat),
        'V_liq': np.array(V_liq),
        'V_vap': np.array(V_vap),
    }


def _sample_iapws_liquid(sat_data, P_max_fill=0.1, n_P=10, T_stride=3):
    """Sample liquid V(T,P) via IAPWS-95 from P_sat up to P_max_fill.

    Fills the gap between the saturation pressure and SeaFreeze's P_min.

    Parameters
    ----------
    sat_data : dict
        Output of _compute_saturation_curve.
    P_max_fill : float
        Upper pressure limit (MPa), normally SeaFreeze P_min.
    n_P : int
        Pressure samples per temperature.
    T_stride : int
        Use every T_stride-th temperature to limit computation time.

    Returns dict with arrays: V (m^3/kg), T (K), P (MPa).
    """
    from iapws import IAPWS95

    V_pts, T_pts, P_pts = [], [], []

    for T, P_sat in zip(sat_data['T'][::T_stride],
                        sat_data['P_sat'][::T_stride]):
        if P_sat >= P_max_fill:
            continue
        # Offset slightly above P_sat to avoid two-phase ambiguity
        P_arr = np.linspace(P_sat + 1e-6, P_max_fill, n_P)
        for P in P_arr:
            try:
                st = IAPWS95(T=T, P=P)
                if (st.v is not None and np.isfinite(st.v)
                        and st.v < 2e-3):  # liquid only
                    V_pts.append(st.v)
                    T_pts.append(T)
                    P_pts.append(P)
            except (NotImplementedError, ValueError, RuntimeError):
                continue

    return {
        'V': np.array(V_pts) if V_pts else np.array([]),
        'T': np.array(T_pts) if T_pts else np.array([]),
        'P': np.array(P_pts) if P_pts else np.array([]),
    }


def sample_phase_at_T(phase_name, T, config=None):
    """Evaluate SeaFreeze for one phase at one temperature.

    Parameters
    ----------
    phase_name : str
        SeaFreeze phase name ('water1', 'Ih', 'II', 'III', 'V', 'VI').
    T : float
        Temperature in K.
    config : dict, optional
        Phase configuration dict. Defaults to PHASE_CONFIGS[phase_name].

    Returns
    -------
    V : ndarray
        Specific volume in m^3/kg.
    A : ndarray
        Helmholtz free energy in J/kg.
    P : ndarray
        Pressure in MPa.
    """
    if config is None:
        config = PHASE_CONFIGS[phase_name]

    if T < config['T_min'] or T > config['T_max']:
        return np.array([]), np.array([]), np.array([])

    sf = _get_sf()
    P_arr = np.linspace(config['P_min'], config['P_max'], config['n_P'])
    PT = np.array([P_arr, np.array([T])], dtype=object)

    with _suppress():
        try:
            out = sf(PT, phase_name)
        except (ValueError, TypeError, RuntimeError) as exc:
            warnings.warn(
                f"SeaFreeze failed for phase '{phase_name}' at T={T:.1f} K: "
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            return np.array([]), np.array([]), np.array([])

    G = np.asarray(out.G).squeeze()
    rho = np.asarray(out.rho).squeeze()

    with np.errstate(divide='ignore', invalid='ignore'):
        V = 1.0 / rho
        A = G - P_arr * 1e6 / rho

    valid = np.isfinite(V) & np.isfinite(A) & (rho > 0)
    return V[valid], A[valid], P_arr[valid]


# ---------------------------------------------------------------------------
# Batch SeaFreeze evaluation (one call per phase for ALL temperatures)
# ---------------------------------------------------------------------------
def _batch_evaluate_phases(phase_list, temperatures):
    """Evaluate SeaFreeze for every phase across all temperatures at once.

    Returns
    -------
    phase_data : dict
        phase_name -> {
            'V': (n_P, n_T_valid) array,
            'A': (n_P, n_T_valid) array,
            'P': (n_P,) array,
            'T_mask': boolean array of length n_T (which temps are valid),
            'col_map': int array mapping global T index -> column index,
        }
        or None if phase returns no data.
    """
    sf = _get_sf()
    phase_data = {}

    for phase in phase_list:
        cfg = PHASE_CONFIGS[phase]
        T_mask = (temperatures >= cfg['T_min']) & (temperatures <= cfg['T_max'])
        T_valid = temperatures[T_mask]

        if len(T_valid) == 0:
            phase_data[phase] = None
            continue

        P_arr = np.linspace(cfg['P_min'], cfg['P_max'], cfg['n_P'])
        PT = np.array([P_arr, T_valid], dtype=object)

        with _suppress():
            try:
                out = sf(PT, phase)
            except (ValueError, TypeError, RuntimeError) as exc:
                warnings.warn(
                    f"SeaFreeze batch failed for phase '{phase}': "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                phase_data[phase] = None
                continue

        G = np.asarray(out.G, dtype=np.float64)    # (n_P, n_T_valid)
        rho = np.asarray(out.rho, dtype=np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            V = 1.0 / rho
            A = G - P_arr[:, None] * 1e6 / rho

        invalid = ~np.isfinite(V) | ~np.isfinite(A) | (rho <= 0)
        V[invalid] = np.nan
        A[invalid] = np.nan

        # Build column-index map: col_map[i] gives the column in V/A for
        # global temperature index i (only meaningful when T_mask[i] is True).
        col_map = np.cumsum(T_mask) - 1

        phase_data[phase] = {
            'V': V,
            'A': A,
            'P': P_arr,
            'T_mask': T_mask,
            'col_map': col_map,
        }

    return phase_data


# ---------------------------------------------------------------------------
# Lower convex hull
# ---------------------------------------------------------------------------
def _trace_lower_hull(V_all, A_all, hull):
    """Return ordered vertex indices along the lower hull, left to right."""
    adj = defaultdict(set)
    for simplex, eq in zip(hull.simplices, hull.equations):
        if eq[1] < 0:  # outward normal has negative A-component -> lower hull
            i, j = simplex
            adj[i].add(j)
            adj[j].add(i)

    if not adj:
        return []

    start = min(adj.keys(), key=lambda i: V_all[i])

    path = [start]
    prev = None
    current = start
    while True:
        neighbors = adj[current] - ({prev} if prev is not None else set())
        if not neighbors:
            break
        nxt = neighbors.pop()
        path.append(nxt)
        prev = current
        current = nxt

    return path


def _compute_hull_from_parts(V_parts, A_parts, P_parts, label_parts):
    """Compute the lower convex hull from pre-sampled phase data.

    Parameters
    ----------
    V_parts, A_parts, P_parts : list of 1-D arrays
    label_parts : list of 1-D string arrays

    Returns
    -------
    dict  (same structure as compute_hull_at_T)
    """
    empty = {
        'single_phase': [], 'two_phase': [], 'stable_phases': set(),
        'hull_path': {'V': np.array([]), 'A': np.array([]),
                      'P': np.array([]), 'phase': np.array([])},
        'raw': {'V': np.array([]), 'A': np.array([]),
                'P': np.array([]), 'labels': np.array([])},
    }
    if not V_parts:
        return empty

    V_all = np.concatenate(V_parts)
    A_all = np.concatenate(A_parts)
    P_all = np.concatenate(P_parts)
    labels = np.concatenate(label_parts)

    if len(V_all) < 3:
        return empty

    # Scale for numerical stability
    V_lo, V_hi = V_all.min(), V_all.max()
    A_lo, A_hi = A_all.min(), A_all.max()
    V_range = V_hi - V_lo if V_hi > V_lo else 1.0
    A_range = A_hi - A_lo if A_hi > A_lo else 1.0
    V_norm = (V_all - V_lo) / V_range
    A_norm = (A_all - A_lo) / A_range

    points = np.column_stack([V_norm, A_norm])

    try:
        hull = ConvexHull(points)
    except QhullError:
        return empty

    path = _trace_lower_hull(V_norm, A_norm, hull)
    if len(path) < 2:
        return empty

    hp_V = V_all[path]
    hp_A = A_all[path]
    hp_P = P_all[path]
    hp_phase = labels[path]

    # Group consecutive same-phase vertices
    groups = []
    cur_phase = hp_phase[0]
    cur_indices = [0]
    for k in range(1, len(path)):
        if hp_phase[k] == cur_phase:
            cur_indices.append(k)
        else:
            groups.append((cur_phase, cur_indices))
            cur_phase = hp_phase[k]
            cur_indices = [k]
    groups.append((cur_phase, cur_indices))

    # Single-phase regions
    single_phase = []
    for phase, idx_list in groups:
        Vs = hp_V[idx_list]
        Ps = hp_P[idx_list]
        single_phase.append({
            'phase': phase,
            'V_min': float(Vs.min()),
            'V_max': float(Vs.max()),
            'P_min': float(Ps.min()),
            'P_max': float(Ps.max()),
        })

    # Two-phase coexistence
    two_phase = []
    for g in range(len(groups) - 1):
        _, idx_L = groups[g]
        _, idx_R = groups[g + 1]
        iL = idx_L[-1]
        iR = idx_R[0]
        V_L, A_L = hp_V[iL], hp_A[iL]
        V_R, A_R = hp_V[iR], hp_A[iR]
        dV = V_R - V_L
        if abs(dV) > 0:
            P_coex = -(A_R - A_L) / dV / 1e6
        else:
            P_coex = float(hp_P[iL])
        two_phase.append({
            'phases': (groups[g][0], groups[g + 1][0]),
            'V_left': float(V_L),
            'V_right': float(V_R),
            'A_left': float(A_L),
            'A_right': float(A_R),
            'P_coex': float(P_coex),
        })

    stable = set(phase for phase, _ in groups)

    return {
        'single_phase': single_phase,
        'two_phase': two_phase,
        'stable_phases': stable,
        'hull_path': {
            'V': hp_V, 'A': hp_A, 'P': hp_P,
            'phase': np.array(hp_phase),
        },
        'raw': {'V': V_all, 'A': A_all, 'P': P_all, 'labels': labels},
    }


def compute_hull_at_T(T, phase_list=None):
    """Compute the lower convex hull at a single temperature.

    This is a convenience wrapper that samples SeaFreeze per-phase.  For batch
    computation across many temperatures, use compute_tv_phase_diagram instead.

    Returns
    -------
    dict with keys: 'single_phase', 'two_phase', 'stable_phases',
    'hull_path', 'raw'.
    """
    if phase_list is None:
        phase_list = DEFAULT_PHASES

    V_parts, A_parts, P_parts, label_parts = [], [], [], []
    for phase in phase_list:
        V, A, P = sample_phase_at_T(phase, T)
        if len(V) > 0:
            V_parts.append(V)
            A_parts.append(A)
            P_parts.append(P)
            label_parts.append(np.full(len(V), phase))

    return _compute_hull_from_parts(V_parts, A_parts, P_parts, label_parts)


# ---------------------------------------------------------------------------
# Three-phase invariant detection
# ---------------------------------------------------------------------------
def _detect_invariants(temperatures, slices):
    """Find temperatures where the topology of stable phases changes."""
    invariants = []
    prev_ordered = None
    prev_T = None
    prev_slice = None

    for T, s in zip(temperatures, slices):
        ordered = tuple(sp['phase'] for sp in s['single_phase'])
        if not ordered:
            prev_ordered = ordered
            prev_T = T
            prev_slice = s
            continue

        if prev_ordered is not None and ordered != prev_ordered:
            T_inv = (prev_T + T) / 2.0
            prev_set = set(prev_ordered)
            cur_set = set(ordered)
            disappeared = prev_set - cur_set
            appeared = cur_set - prev_set

            prev_list = list(prev_ordered)
            for phase in disappeared:
                idx = prev_list.index(phase)
                left = prev_list[idx - 1] if idx > 0 else None
                right = prev_list[idx + 1] if idx < len(prev_list) - 1 else None
                trio = tuple(sorted(p for p in [left, phase, right]
                                    if p is not None))
                if len(trio) >= 3:
                    P_inv = None
                    use_slice = prev_slice if prev_slice else s
                    for tp in use_slice['two_phase']:
                        if phase in tp['phases']:
                            P_inv = tp['P_coex']
                            break
                    V_bounds = []
                    for sp in use_slice['single_phase']:
                        if sp['phase'] in trio:
                            V_bounds.extend([sp['V_min'], sp['V_max']])
                    invariants.append({
                        'T': T_inv, 'phases': trio, 'P_coex': P_inv,
                        'V_min': min(V_bounds) if V_bounds else None,
                        'V_max': max(V_bounds) if V_bounds else None,
                    })

            cur_list = list(ordered)
            for phase in appeared:
                idx = cur_list.index(phase)
                left = cur_list[idx - 1] if idx > 0 else None
                right = cur_list[idx + 1] if idx < len(cur_list) - 1 else None
                trio = tuple(sorted(p for p in [left, phase, right]
                                    if p is not None))
                if len(trio) >= 3:
                    if any(inv['phases'] == trio
                           and abs(inv['T'] - T_inv) < 0.1
                           for inv in invariants):
                        continue
                    P_inv = None
                    for tp in s['two_phase']:
                        if phase in tp['phases']:
                            P_inv = tp['P_coex']
                            break
                    V_bounds = []
                    for sp in s['single_phase']:
                        if sp['phase'] in trio:
                            V_bounds.extend([sp['V_min'], sp['V_max']])
                    invariants.append({
                        'T': T_inv, 'phases': trio, 'P_coex': P_inv,
                        'V_min': min(V_bounds) if V_bounds else None,
                        'V_max': max(V_bounds) if V_bounds else None,
                    })

        prev_ordered = ordered
        prev_T = T
        prev_slice = s

    if len(invariants) < 2:
        return invariants
    invariants.sort(key=lambda x: (x['phases'], x['T']))
    merged = [invariants[0]]
    for inv in invariants[1:]:
        m = merged[-1]
        if inv['phases'] == m['phases'] and abs(inv['T'] - m['T']) < 5.0:
            m['T'] = (m['T'] + inv['T']) / 2.0
            if inv['V_min'] is not None and m['V_min'] is not None:
                m['V_min'] = min(m['V_min'], inv['V_min'])
                m['V_max'] = max(m['V_max'], inv['V_max'])
        else:
            merged.append(inv)
    merged.sort(key=lambda x: x['T'])
    return merged


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_tv_phase_diagram(T_min=190.0, T_max=353.5, dT=1.0,
                             phase_list=None, verbose=True,
                             include_vapor=True):
    """Compute the T-V phase diagram by isothermal convex-hull sweeps.

    Parameters
    ----------
    T_min, T_max : float
        Temperature range in K.
    dT : float
        Temperature step in K.  Use 1.0 for quick preview (~3 s),
        0.0654 for publication quality (~40 s).
    phase_list : list of str, optional
        Phases to include. Default: all six.
    verbose : bool
        Print progress.

    Returns
    -------
    diagram : dict
        Keys: 'temperatures', 'slices', 'phase_fields', 'two_phase_bounds',
        'invariants', 'phase_list'.
    """
    if phase_list is None:
        phase_list = list(DEFAULT_PHASES)

    temperatures = np.arange(T_min, T_max + dT / 2, dT)
    n_T = len(temperatures)

    # --- Batch-evaluate SeaFreeze: one call per phase for all T's ----------
    if verbose:
        print("  Evaluating SeaFreeze for all phases...", end='', flush=True)
    phase_data = _batch_evaluate_phases(phase_list, temperatures)
    if verbose:
        print(" done.")

    # --- Hull sweep --------------------------------------------------------
    slices = []
    pf = defaultdict(lambda: {'T': [], 'V_min': [], 'V_max': []})
    tp = defaultdict(lambda: {'T': [], 'V_left': [], 'V_right': [], 'P_coex': []})

    for i, T in enumerate(temperatures):
        if verbose and (i % max(1, n_T // 20) == 0 or i == n_T - 1):
            print(f"\r  Convex hulls: {i + 1}/{n_T}  (T = {T:.1f} K)",
                  end='', flush=True)

        # Collect data for this temperature from pre-computed batch arrays
        V_parts, A_parts, P_parts, label_parts = [], [], [], []
        for phase in phase_list:
            pd = phase_data[phase]
            if pd is None or not pd['T_mask'][i]:
                continue
            col = pd['col_map'][i]
            V_col = pd['V'][:, col]
            A_col = pd['A'][:, col]
            P_col = pd['P']
            valid = np.isfinite(V_col) & np.isfinite(A_col)
            if valid.sum() > 0:
                V_parts.append(V_col[valid])
                A_parts.append(A_col[valid])
                P_parts.append(P_col[valid])
                label_parts.append(np.full(int(valid.sum()), phase))

        s = _compute_hull_from_parts(V_parts, A_parts, P_parts, label_parts)
        slices.append(s)

        for sp in s['single_phase']:
            pf[sp['phase']]['T'].append(T)
            pf[sp['phase']]['V_min'].append(sp['V_min'])
            pf[sp['phase']]['V_max'].append(sp['V_max'])
        for tp_entry in s['two_phase']:
            key = tp_entry['phases']
            tp[key]['T'].append(T)
            tp[key]['V_left'].append(tp_entry['V_left'])
            tp[key]['V_right'].append(tp_entry['V_right'])
            tp[key]['P_coex'].append(tp_entry['P_coex'])

    if verbose:
        print()

    phase_fields = {phase: {k: np.array(v) for k, v in data.items()}
                    for phase, data in pf.items()}
    two_phase_bounds = {key: {k: np.array(v) for k, v in data.items()}
                        for key, data in tp.items()}
    invariants = _detect_invariants(temperatures, slices)

    diagram = {
        'temperatures': temperatures,
        'slices': slices,
        'phase_fields': phase_fields,
        'two_phase_bounds': two_phase_bounds,
        'invariants': invariants,
        'phase_list': phase_list,
    }

    if verbose:
        print(f"  Found {len(phase_fields)} phase fields, "
              f"{len(two_phase_bounds)} two-phase regions, "
              f"{len(invariants)} invariants.")
        for inv in invariants:
            print(f"    Invariant at T = {inv['T']:.1f} K: "
                  f"{' / '.join(inv['phases'])}")

    # --- IAPWS saturation and low-pressure liquid --------------------------
    if include_vapor:
        if verbose:
            print("  Computing IAPWS saturation curve...", end='', flush=True)
        sat_data = _compute_saturation_curve(
            T_min=max(273.16, T_min), T_max=T_max)
        if verbose:
            print(f" {len(sat_data['T'])} points.")
            print("  Sampling IAPWS low-pressure liquid...", end='',
                  flush=True)
        iapws_liq = _sample_iapws_liquid(sat_data)
        if verbose:
            print(f" {len(iapws_liq['V'])} points.")

        diagram['saturation'] = sat_data
        diagram['iapws_liquid'] = iapws_liq

    return diagram


# ---------------------------------------------------------------------------
# Isochore computation
# ---------------------------------------------------------------------------
def compute_isochore(diagram, V_system):
    """Trace a vertical line at constant specific volume through the T-V diagram.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    V_system : float
        System specific volume in m^3/kg.

    Returns
    -------
    result : dict
        Keys: 'T', 'P', 'phase', 'region', 'phase_fraction'.
    """
    T_out, P_out, phase_out, region_out, frac_out = [], [], [], [], []

    for T, s in zip(diagram['temperatures'], diagram['slices']):
        found = False
        for sp in s['single_phase']:
            if sp['V_min'] <= V_system <= sp['V_max']:
                hp = s['hull_path']
                mask = hp['phase'] == sp['phase']
                Vs = hp['V'][mask]
                Ps = hp['P'][mask]
                if len(Vs) >= 2:
                    P_interp = float(np.interp(V_system, Vs, Ps))
                elif len(Vs) == 1:
                    P_interp = float(Ps[0])
                else:
                    continue
                T_out.append(T)
                P_out.append(P_interp)
                phase_out.append(sp['phase'])
                region_out.append('single')
                frac_out.append(0.0)
                found = True
                break
        if found:
            continue
        for tp in s['two_phase']:
            V_L, V_R = tp['V_left'], tp['V_right']
            if V_L <= V_system <= V_R:
                f = (V_system - V_L) / (V_R - V_L) if V_R > V_L else 0.5
                T_out.append(T)
                P_out.append(tp['P_coex'])
                phase_out.append(tp['phases'])
                region_out.append('two_phase')
                frac_out.append(float(f))
                found = True
                break

    return {
        'T': np.array(T_out),
        'P': np.array(P_out),
        'phase': phase_out,
        'region': region_out,
        'phase_fraction': np.array(frac_out),
        'V_system': V_system,
    }


# ---------------------------------------------------------------------------
# Plotting: 2D T-V phase diagram
# ---------------------------------------------------------------------------
def plot_tv_phase_diagram(diagram, ax=None, show_labels=True,
                          show_invariants=True):
    """Plot the T-V phase diagram (reproduces Powell-Palm 2022 Fig. 1c).

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    ax : matplotlib Axes, optional
    show_labels : bool
        Label phase regions.
    show_invariants : bool
        Draw horizontal lines at three-phase invariants.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Fill single-phase regions
    plotted_labels = set()
    for phase in diagram['phase_list']:
        if phase not in diagram['phase_fields']:
            continue
        pf = diagram['phase_fields'][phase]
        cfg = PHASE_CONFIGS.get(phase, {})
        color = cfg.get('color', '#888888')
        label = cfg.get('label', phase)

        lbl = label if phase not in plotted_labels else None
        plotted_labels.add(phase)

        ax.fill_betweenx(pf['T'], pf['V_min'], pf['V_max'],
                         color=color, alpha=0.7, label=lbl)

        if show_labels and len(pf['T']) > 0:
            mid_idx = len(pf['T']) // 2
            V_mid = (pf['V_min'][mid_idx] + pf['V_max'][mid_idx]) / 2
            T_mid = pf['T'][mid_idx]
            ax.text(V_mid, T_mid, label, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              alpha=0.7, ec='none'))

    # Two-phase boundary curves
    for key, bounds in diagram['two_phase_bounds'].items():
        ax.plot(bounds['V_left'], bounds['T'], 'k-', lw=0.5)
        ax.plot(bounds['V_right'], bounds['T'], 'k-', lw=0.5)

    # Saturation boundary (liquid side) from IAPWS
    if 'saturation' in diagram:
        sat = diagram['saturation']
        if len(sat['T']) > 0:
            ax.plot(sat['V_liq'], sat['T'], 'k-', lw=1.5,
                    label='Saturation')

    # Three-phase invariant lines (solid horizontal segments).
    # At a three-phase invariant, three phases coexist at one (T, P).
    # The invariant line must lie ONLY in the two-phase gaps between
    # the three involved phases — it must NOT cross through any
    # single-phase region.  So it spans from V_max of the leftmost
    # trio phase (its right edge) to V_min of the rightmost trio phase
    # (its left edge).
    if show_invariants:
        for inv in diagram['invariants']:
            if inv['V_min'] is None or inv['V_max'] is None:
                continue
            trio = set(inv['phases'])

            # Find the nearest slice where the trio phases exist
            temps = diagram['temperatures']
            idx = int(np.argmin(np.abs(temps - inv['T'])))
            best_sps = []
            for di in [0, -1, 1]:
                j = idx + di
                if j < 0 or j >= len(temps):
                    continue
                s = diagram['slices'][j]
                sps = [sp for sp in s['single_phase']
                       if sp['phase'] in trio]
                if len(sps) > len(best_sps):
                    best_sps = sps

            if len(best_sps) >= 2:
                best_sps.sort(key=lambda sp: sp['V_min'])
                # Right edge of leftmost → left edge of rightmost
                V_lo = best_sps[0]['V_max']
                V_hi = best_sps[-1]['V_min']
            else:
                V_lo, V_hi = inv['V_min'], inv['V_max']

            ax.plot([V_lo, V_hi], [inv['T'], inv['T']],
                    'k-', lw=1.5, zorder=4)

            P_label = (f"{inv['P_coex']:.0f} MPa"
                       if inv['P_coex'] is not None else '')
            phase_names = [PHASE_CONFIGS.get(p, {}).get('label', p)
                           for p in inv['phases']]
            phase_label = ' / '.join(phase_names)
            ax.annotate(f"{phase_label}  {P_label}",
                        xy=(V_lo, inv['T']),
                        xytext=(-5, 4), textcoords='offset points',
                        fontsize=6, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  fc='white', alpha=0.8, ec='none'))

    # Triple point invariant (Liquid / Ice Ih / Vapor) from IAPWS
    if (show_invariants and 'saturation' in diagram
            and 'Ih' in diagram.get('phase_list', [])):
        sat = diagram['saturation']
        temps = diagram['temperatures']
        T_triple = 273.16
        if len(sat['T']) > 0 and temps[0] <= T_triple <= temps[-1]:
            idx_tp = int(np.argmin(np.abs(sat['T'] - T_triple)))
            V_liq_tp = sat['V_liq'][idx_tp]
            P_triple = sat['P_sat'][idx_tp]
            # Line extends rightward from V_liq through Ih toward vapor
            ax.plot([V_liq_tp, 2e-3], [T_triple, T_triple],
                    'k-', lw=1.5, zorder=4)
            ax.annotate(
                f"Liquid / Ice Ih / Vapor  {P_triple:.4f} MPa",
                xy=(V_liq_tp, T_triple),
                xytext=(-5, 4), textcoords='offset points',
                fontsize=6, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white', alpha=0.8, ec='none'))

    ax.set_xlabel('Specific volume (m$^3$/kg)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('T-V Phase Diagram of H$_2$O')
    ax.legend(loc='upper right', fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Plotting: A-V convex hull at a single temperature
# ---------------------------------------------------------------------------
def plot_av_hull(T, phase_list=None, ax=None):
    """Plot A(V) curves and the convex hull at one temperature."""
    import matplotlib.pyplot as plt

    if phase_list is None:
        phase_list = DEFAULT_PHASES
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    result = compute_hull_at_T(T, phase_list)

    raw = result['raw']
    for phase in phase_list:
        mask = raw['labels'] == phase
        if not mask.any():
            continue
        cfg = PHASE_CONFIGS.get(phase, {})
        ax.plot(raw['V'][mask], raw['A'][mask], '.', markersize=1,
                color=cfg.get('color', '#888'), alpha=0.4)
        ax.plot([], [], 'o', color=cfg.get('color', '#888'),
                label=cfg.get('label', phase), markersize=5)

    hp = result['hull_path']
    if len(hp['V']) > 0:
        ax.plot(hp['V'], hp['A'], 'k-', lw=2, label='Lower hull', zorder=5)
        ax.plot(hp['V'], hp['A'], 'ko', markersize=3, zorder=6)

    for tp in result['two_phase']:
        ax.plot([tp['V_left'], tp['V_right']],
                [tp['A_left'], tp['A_right']],
                'r--', lw=1.5, zorder=4)
        ax.text((tp['V_left'] + tp['V_right']) / 2,
                (tp['A_left'] + tp['A_right']) / 2,
                f"  P={tp['P_coex']:.0f} MPa",
                fontsize=7, color='red', va='bottom')

    ax.set_xlabel('Specific volume (m$^3$/kg)')
    ax.set_ylabel('Helmholtz free energy A (J/kg)')
    ax.set_title(f'A-V Convex Hull at T = {T:.1f} K')
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Plotting: T-P phase diagram (projection)
# ---------------------------------------------------------------------------
def plot_tp_phase_diagram(diagram, ax=None):
    """Plot the T-P phase diagram derived from coexistence pressures."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for key, bounds in diagram['two_phase_bounds'].items():
        label = ' / '.join(PHASE_CONFIGS.get(p, {}).get('label', p)
                           for p in key)
        ax.plot(bounds['T'], bounds['P_coex'], '-', lw=1.5, label=label)

    # Boiling curve from IAPWS saturation
    if 'saturation' in diagram:
        sat = diagram['saturation']
        if len(sat['T']) > 0:
            ax.plot(sat['T'], sat['P_sat'], 'k--', lw=2.0,
                    label='Boiling curve')

    for i, inv in enumerate(diagram['invariants']):
        if inv['P_coex'] is not None:
            ax.plot(inv['T'], inv['P_coex'], 'ko', markersize=5)
            phase_names = [PHASE_CONFIGS.get(p, {}).get('label', p)
                           for p in inv['phases']]
            y_offset = 8 + (i % 3) * 12
            ax.annotate(' / '.join(phase_names),
                        (inv['T'], inv['P_coex']),
                        fontsize=6, textcoords='offset points',
                        xytext=(5, y_offset),
                        arrowprops=dict(arrowstyle='-', color='gray',
                                        lw=0.5),
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  alpha=0.8, ec='none'))

    # Use log scale when pressure spans many orders of magnitude
    if 'saturation' in diagram:
        ax.set_yscale('log')

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_title('T-P Phase Diagram (from T-V computation)')
    ax.legend(fontsize=7, loc='upper left')
    return ax


# ---------------------------------------------------------------------------
# Plotting: Isochore
# ---------------------------------------------------------------------------
def plot_isochore(isochore_data, diagram=None, axes=None):
    """Plot isochore results: P-T trajectory and phase fraction vs T."""
    import matplotlib.pyplot as plt

    if not isinstance(isochore_data, list):
        isochore_data = [isochore_data]

    if axes is None:
        fig, (ax_PT, ax_frac) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax_PT, ax_frac = axes

    if diagram is not None:
        for key, bounds in diagram['two_phase_bounds'].items():
            ax_PT.plot(bounds['T'], bounds['P_coex'], 'k-', lw=0.5,
                       alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(isochore_data)))
    for iso, clr in zip(isochore_data, colors):
        V_label = (f"V = {iso['V_system']*1e4:.2f}"
                   + r"$\times 10^{-4}$ m$^3$/kg")
        ax_PT.plot(iso['T'], iso['P'], '-', color=clr, lw=1.5, label=V_label)
        frac = iso['phase_fraction']
        region = iso['region']
        two_mask = np.array([r == 'two_phase' for r in region])
        if two_mask.any():
            ax_frac.plot(iso['T'][two_mask], frac[two_mask], '-',
                         color=clr, lw=1.5, label=V_label)

    ax_PT.set_xlabel('Temperature (K)')
    ax_PT.set_ylabel('Pressure (MPa)')
    ax_PT.set_title('Isochoric P-T Paths')
    ax_PT.legend(fontsize=7)
    ax_frac.set_xlabel('Temperature (K)')
    ax_frac.set_ylabel('Phase fraction (right-hand phase)')
    ax_frac.set_title('Phase Fraction Along Isochores')
    ax_frac.legend(fontsize=7)
    ax_frac.set_ylim(-0.05, 1.05)
    return (ax_PT, ax_frac)


# ---------------------------------------------------------------------------
# Plotting: 3D P-T-V phase diagram (contiguous surface)
# ---------------------------------------------------------------------------
def plot_ptv_phase_diagram(diagram, phase_list=None, elev=25, azim=-60,
                           T_stride=2, n_pts_per_phase=25, n_tie=5,
                           V_min=6e-4, V_max=1.2e-3, P_max=1000.0):
    """Plot the 3D P-T-V surface as one contiguous surface.

    Single-phase regions are connected by two-phase ruled surfaces.  The
    surface is parameterised as P(V, T) — single-valued everywhere — and
    triangulated in the (V, T) projection plane.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    phase_list : list of str, optional
    elev, azim : float
        Camera elevation and azimuth angles.
    T_stride : int
        Use every T_stride-th temperature slice (controls density).
    n_pts_per_phase : int
        Max points sampled per phase per temperature.
    n_tie : int
        Points along each two-phase tie line.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.tri import Triangulation

    if phase_list is None:
        phase_list = diagram.get('phase_list', DEFAULT_PHASES)

    temperatures = diagram['temperatures']
    slices_data = diagram['slices']

    # Collect (V, T, P, phase) from the raw slice data already in memory
    V_pts, T_pts, P_pts, ph_pts = [], [], [], []

    for idx in range(0, len(temperatures), T_stride):
        T = float(temperatures[idx])
        s = slices_data[idx]
        raw = s['raw']

        if not s['single_phase'] or len(raw['V']) == 0:
            continue

        # --- single-phase surface points (from raw sampling data) ----------
        for sp in s['single_phase']:
            phase = sp['phase']
            mask = ((raw['labels'] == phase)
                    & (raw['V'] >= sp['V_min'])
                    & (raw['V'] <= sp['V_max'])
                    & np.isfinite(raw['V'])
                    & np.isfinite(raw['P']))
            V_ph = raw['V'][mask]
            P_ph = raw['P'][mask]
            if len(V_ph) == 0:
                continue
            # Sort by V and decimate
            order = np.argsort(V_ph)
            step = max(1, len(order) // n_pts_per_phase)
            sel = order[::step]
            # Always include first and last for boundary continuity
            if sel[0] != order[0]:
                sel = np.concatenate([[order[0]], sel])
            if sel[-1] != order[-1]:
                sel = np.concatenate([sel, [order[-1]]])
            V_pts.extend(V_ph[sel].tolist())
            T_pts.extend([T] * len(sel))
            P_pts.extend(P_ph[sel].tolist())
            ph_pts.extend([phase] * len(sel))

        # --- two-phase tie-line points ------------------------------------
        for tp in s['two_phase']:
            P_coex = tp['P_coex']
            if P_coex < 0:
                continue
            V_tie = np.linspace(tp['V_left'], tp['V_right'], n_tie)
            V_pts.extend(V_tie.tolist())
            T_pts.extend([T] * n_tie)
            P_pts.extend([P_coex] * n_tie)
            ph_pts.extend(['two_phase'] * n_tie)

    # --- IAPWS low-pressure liquid surface --------------------------------
    if 'iapws_liquid' in diagram:
        iapws = diagram['iapws_liquid']
        if len(iapws['V']) > 0:
            V_pts.extend(iapws['V'].tolist())
            T_pts.extend(iapws['T'].tolist())
            P_pts.extend(iapws['P'].tolist())
            ph_pts.extend(['water1'] * len(iapws['V']))

    # --- Saturation line points (liquid side at P_sat) --------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        if len(sat['T']) > 0:
            V_pts.extend(sat['V_liq'].tolist())
            T_pts.extend(sat['T'].tolist())
            P_pts.extend(sat['P_sat'].tolist())
            ph_pts.extend(['water1'] * len(sat['T']))

    V_arr = np.array(V_pts)
    T_arr = np.array(T_pts)
    P_arr = np.array(P_pts)
    ph_arr = np.array(ph_pts)

    # Remove any invalid entries
    ok = np.isfinite(V_arr) & np.isfinite(T_arr) & np.isfinite(P_arr)
    V_arr, T_arr, P_arr, ph_arr = (
        V_arr[ok], T_arr[ok], P_arr[ok], ph_arr[ok])

    # Filter to V and P ranges
    in_range = ((V_arr >= V_min) & (V_arr <= V_max)
                & (P_arr <= P_max) & (P_arr > 0))
    V_arr = V_arr[in_range]
    T_arr = T_arr[in_range]
    P_arr = P_arr[in_range]
    ph_arr = ph_arr[in_range]

    if len(V_arr) < 4:
        raise RuntimeError("Too few valid points for the 3-D surface.")

    dT_step = (float(temperatures[1] - temperatures[0])
               if len(temperatures) > 1 else 1.0)

    # --- plot per-phase surfaces ------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Small P offset to lift lines above the surface so they're visible
    dP_lift = P_max * 0.012

    # Render each single-phase region as a separate surface
    # so boundaries between phases are crisp, not jagged.
    total_V_range = np.ptp(V_arr)

    for phase in set(ph_arr):
        if phase == 'two_phase':
            continue  # rendered separately below as ruled surfaces
        mask = ph_arr == phase
        if mask.sum() < 3:
            continue
        V_ph = V_arr[mask]
        T_ph = T_arr[mask]
        P_ph = P_arr[mask]

        V_range = np.ptp(V_ph)
        T_range = np.ptp(T_ph)
        if V_range < 1e-12 or T_range < 1e-12:
            continue

        V_n = (V_ph - V_ph.min()) / V_range
        T_n = (T_ph - T_ph.min()) / T_range
        try:
            tri_ph = Triangulation(V_n, T_n)
        except (ValueError, RuntimeError):
            continue

        # Prune badly shaped triangles (use global ranges for thresholds)
        v0 = V_ph[tri_ph.triangles[:, 0]]
        v1 = V_ph[tri_ph.triangles[:, 1]]
        v2 = V_ph[tri_ph.triangles[:, 2]]
        tri_Vs = (np.max(np.column_stack([v0, v1, v2]), axis=1)
                  - np.min(np.column_stack([v0, v1, v2]), axis=1))
        t0 = T_ph[tri_ph.triangles[:, 0]]
        t1 = T_ph[tri_ph.triangles[:, 1]]
        t2 = T_ph[tri_ph.triangles[:, 2]]
        tri_Ts = (np.max(np.column_stack([t0, t1, t2]), axis=1)
                  - np.min(np.column_stack([t0, t1, t2]), axis=1))
        bad = (tri_Vs > 0.30 * total_V_range) | (tri_Ts > dT_step * 3)
        tri_ph.set_mask(bad)

        good = tri_ph.get_masked_triangles()
        if len(good) == 0:
            continue

        color = PHASE_CONFIGS.get(phase, {}).get('color', '#888888')
        ax.plot_trisurf(V_ph, T_ph, P_ph, triangles=good,
                        color=color, alpha=0.7,
                        edgecolor='none', antialiased=True, shade=False)

    # --- Two-phase ruled surfaces (structured triangles) ------------------
    n_across = 5  # points across each tie-line strip
    for key, bounds in diagram['two_phase_bounds'].items():
        T_b = bounds['T']
        V_l = bounds['V_left']
        V_r = bounds['V_right']
        P_b = bounds['P_coex']

        # Keep rows where the tie-line overlaps our V window
        keep = ((V_r >= V_min) & (V_l <= V_max)
                & (P_b <= P_max) & (P_b > 0))
        T_b = T_b[keep]
        V_l = V_l[keep]
        V_r = V_r[keep]
        P_b = P_b[keep]
        if len(T_b) < 2:
            continue

        # Clip V endpoints to the display range
        V_l = np.clip(V_l, V_min, V_max)
        V_r = np.clip(V_r, V_min, V_max)

        n = len(T_b)
        # Build a structured grid: n rows x n_across columns
        V_grid, T_grid, P_grid = [], [], []
        for i in range(n):
            for j in range(n_across):
                frac = j / (n_across - 1)
                V_grid.append(V_l[i] + frac * (V_r[i] - V_l[i]))
                T_grid.append(T_b[i])
                P_grid.append(P_b[i])
        V_grid = np.array(V_grid)
        T_grid = np.array(T_grid)
        P_grid = np.array(P_grid)

        # Build explicit quad→triangle connectivity
        tris = []
        for i in range(n - 1):
            for j in range(n_across - 1):
                i00 = i * n_across + j
                i01 = i00 + 1
                i10 = (i + 1) * n_across + j
                i11 = i10 + 1
                tris.append([i00, i01, i10])
                tris.append([i01, i11, i10])
        tris = np.array(tris)

        ax.plot_trisurf(V_grid, T_grid, P_grid, triangles=tris,
                        color='#bbbbbb', alpha=0.7,
                        edgecolor='none', antialiased=True, shade=False)

    # --- Two-phase boundary curves in 3D ----------------------------------
    for key, bounds in diagram['two_phase_bounds'].items():
        for V_side in ('V_left', 'V_right'):
            V_b = bounds[V_side]
            T_b = bounds['T']
            P_b = bounds['P_coex']
            keep = ((V_b >= V_min) & (V_b <= V_max)
                    & (P_b <= P_max) & (P_b > 0))
            if keep.any():
                ax.plot(V_b[keep], T_b[keep], P_b[keep] + dP_lift,
                        'k-', lw=1.5)

    # --- Saturation curve in 3D -------------------------------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        mask = ((sat['V_liq'] >= V_min) & (sat['V_liq'] <= V_max)
                & (sat['P_sat'] <= P_max))
        if mask.any():
            ax.plot(sat['V_liq'][mask], sat['T'][mask],
                    sat['P_sat'][mask] + dP_lift, 'k-', lw=1.5)

    # --- Three-phase invariant lines in 3D --------------------------------
    for inv in diagram.get('invariants', []):
        if inv['P_coex'] is None or inv['V_min'] is None:
            continue
        P_inv = inv['P_coex']
        if P_inv > P_max or P_inv <= 0:
            continue
        trio = set(inv['phases'])
        idx = int(np.argmin(np.abs(temperatures - inv['T'])))
        best_sps = []
        for di in [0, -1, 1]:
            j = idx + di
            if j < 0 or j >= len(temperatures):
                continue
            s = slices_data[j]
            sps = [sp for sp in s['single_phase'] if sp['phase'] in trio]
            if len(sps) > len(best_sps):
                best_sps = sps
        if len(best_sps) >= 2:
            best_sps.sort(key=lambda sp: sp['V_min'])
            V_lo = max(best_sps[0]['V_max'], V_min)
            V_hi = min(best_sps[-1]['V_min'], V_max)
        else:
            V_lo = max(inv['V_min'], V_min)
            V_hi = min(inv['V_max'], V_max)
        if V_lo < V_hi:
            ax.plot([V_lo, V_hi], [inv['T'], inv['T']],
                    [P_inv + dP_lift, P_inv + dP_lift], 'k-', lw=1.5)

    # Triple point invariant (Liquid / Ih / Vapor)
    if ('saturation' in diagram
            and 'Ih' in diagram.get('phase_list', [])):
        sat = diagram['saturation']
        T_triple = 273.16
        temps = diagram['temperatures']
        if len(sat['T']) > 0 and temps[0] <= T_triple <= temps[-1]:
            idx_tp = int(np.argmin(np.abs(sat['T'] - T_triple)))
            V_liq_tp = sat['V_liq'][idx_tp]
            P_triple = sat['P_sat'][idx_tp]
            ax.plot([V_liq_tp, V_max], [T_triple, T_triple],
                    [P_triple + dP_lift, P_triple + dP_lift],
                    'k-', lw=1.5)

    ax.set_xlim(V_min, V_max)
    ax.set_zlim(0, P_max)
    ax.set_xlabel('Specific volume V (m$^3$/kg)')
    ax.set_ylabel('Temperature T (K)')
    ax.set_zlabel('Pressure P (MPa)')
    ax.set_title('P-T-V Phase Diagram of H$_2$O')
    ax.view_init(elev=elev, azim=azim)

    return fig


# ---------------------------------------------------------------------------
# Plotting: 3D P-T-V phase diagram (Plotly)
# ---------------------------------------------------------------------------
def plot_ptv_phase_diagram_plotly(diagram, phase_list=None,
                                  T_stride=1, n_pts_per_phase=150, n_tie=5,
                                  V_min=7e-4, V_max=1.1e-3, P_max=1000.0):
    """Plot the 3D P-T-V surface using Plotly for smooth interactive rendering.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    phase_list : list of str, optional
    T_stride : int
        Use every T_stride-th temperature slice (controls density).
    n_pts_per_phase : int
        Max points sampled per phase per temperature.
    n_tie : int
        Points along each two-phase tie line.
    V_min, V_max : float
        Specific volume display range (m^3/kg).
    P_max : float
        Maximum pressure to display (MPa).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from scipy.spatial import Delaunay

    if phase_list is None:
        phase_list = diagram.get('phase_list', DEFAULT_PHASES)

    temperatures = diagram['temperatures']
    slices_data = diagram['slices']
    dT_step = (float(temperatures[1] - temperatures[0])
               if len(temperatures) > 1 else 1.0)
    dT_eff = dT_step * T_stride  # effective spacing after striding

    # --- Collect points per phase -----------------------------------------
    from collections import defaultdict as _dfl
    phase_pts = _dfl(lambda: {'V': [], 'T': [], 'P': []})

    for idx in range(0, len(temperatures), T_stride):
        T = float(temperatures[idx])
        s = slices_data[idx]
        raw = s['raw']
        if not s['single_phase'] or len(raw['V']) == 0:
            continue

        for sp in s['single_phase']:
            phase = sp['phase']
            mask = ((raw['labels'] == phase)
                    & (raw['V'] >= sp['V_min'])
                    & (raw['V'] <= sp['V_max'])
                    & np.isfinite(raw['V'])
                    & np.isfinite(raw['P']))
            V_ph = raw['V'][mask]
            P_ph = raw['P'][mask]
            if len(V_ph) == 0:
                continue
            order = np.argsort(V_ph)
            step = max(1, len(order) // n_pts_per_phase)
            sel = order[::step]
            if sel[0] != order[0]:
                sel = np.concatenate([[order[0]], sel])
            if sel[-1] != order[-1]:
                sel = np.concatenate([sel, [order[-1]]])
            phase_pts[phase]['V'].extend(V_ph[sel].tolist())
            phase_pts[phase]['T'].extend([T] * len(sel))
            phase_pts[phase]['P'].extend(P_ph[sel].tolist())

    # IAPWS liquid fill
    if 'iapws_liquid' in diagram:
        iapws_liq = diagram['iapws_liquid']
        if len(iapws_liq['V']) > 0:
            phase_pts['water1']['V'].extend(iapws_liq['V'].tolist())
            phase_pts['water1']['T'].extend(iapws_liq['T'].tolist())
            phase_pts['water1']['P'].extend(iapws_liq['P'].tolist())

    # Saturation line points -> liquid surface
    if 'saturation' in diagram:
        sat = diagram['saturation']
        if len(sat['T']) > 0:
            phase_pts['water1']['V'].extend(sat['V_liq'].tolist())
            phase_pts['water1']['T'].extend(sat['T'].tolist())
            phase_pts['water1']['P'].extend(sat['P_sat'].tolist())

    # Add boundary points to each bordering phase (fixes alignment)
    for key, bounds in diagram['two_phase_bounds'].items():
        phase_left, phase_right = key
        T_b = bounds['T']
        V_l, V_r = bounds['V_left'], bounds['V_right']
        P_b = bounds['P_coex']
        keep = (P_b > 0) & (P_b <= P_max)
        if keep.any():
            phase_pts[phase_left]['V'].extend(V_l[keep].tolist())
            phase_pts[phase_left]['T'].extend(T_b[keep].tolist())
            phase_pts[phase_left]['P'].extend(P_b[keep].tolist())
            phase_pts[phase_right]['V'].extend(V_r[keep].tolist())
            phase_pts[phase_right]['T'].extend(T_b[keep].tolist())
            phase_pts[phase_right]['P'].extend(P_b[keep].tolist())

    # Convert to arrays and filter to display range
    phase_arrays = {}
    for phase, pts in phase_pts.items():
        V = np.array(pts['V'])
        T = np.array(pts['T'])
        P = np.array(pts['P'])
        ok = (np.isfinite(V) & np.isfinite(T) & np.isfinite(P)
              & (V >= V_min) & (V <= V_max)
              & (P <= P_max) & (P > 0))
        if ok.sum() >= 3:
            phase_arrays[phase] = {'V': V[ok], 'T': T[ok], 'P': P[ok]}

    display_V_range = V_max - V_min
    traces = []

    # --- Helper: Delaunay triangulation with relaxed pruning ---------------
    def _triangulate(V, T, P):
        """Return (i, j, k) triangle index arrays or None."""
        if len(V) < 3:
            return None
        V_range = np.ptp(V)
        T_range = np.ptp(T)
        if V_range < 1e-12 or T_range < 1e-12:
            return None
        V_n = (V - V.min()) / V_range
        T_n = (T - T.min()) / T_range
        pts2d = np.column_stack([V_n, T_n])
        try:
            tri = Delaunay(pts2d)
        except QhullError:
            return None
        simplices = tri.simplices
        # Prune oversized convex-hull triangles (relaxed thresholds)
        v0, v1, v2 = V[simplices[:, 0]], V[simplices[:, 1]], V[simplices[:, 2]]
        t0, t1, t2 = T[simplices[:, 0]], T[simplices[:, 1]], T[simplices[:, 2]]
        tri_Vs = (np.max(np.column_stack([v0, v1, v2]), axis=1)
                  - np.min(np.column_stack([v0, v1, v2]), axis=1))
        tri_Ts = (np.max(np.column_stack([t0, t1, t2]), axis=1)
                  - np.min(np.column_stack([t0, t1, t2]), axis=1))
        V_thresh = max(0.50 * V_range, 0.20 * display_V_range)
        T_thresh = max(dT_eff * 5, T_range * 0.10)
        good = (tri_Vs <= V_thresh) & (tri_Ts <= T_thresh)
        s = simplices[good]
        if len(s) == 0:
            return None
        return s[:, 0], s[:, 1], s[:, 2]

    # --- Per-phase surfaces -----------------------------------------------
    phase_centroids = {}
    for phase, arr in phase_arrays.items():
        V_ph, T_ph, P_ph = arr['V'], arr['T'], arr['P']
        result = _triangulate(V_ph, T_ph, P_ph)
        if result is None:
            continue
        ii, jj, kk = result
        color = PHASE_CONFIGS.get(phase, {}).get('color', '#888888')
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        traces.append(go.Mesh3d(
            x=V_ph, y=T_ph, z=P_ph,
            i=ii, j=jj, k=kk,
            color=color, opacity=0.7,
            name=label, showlegend=True,
            flatshading=True,
            hovertemplate=(
                f'<b>{label}</b><br>'
                'V=%{x:.4e} m³/kg<br>'
                'T=%{y:.1f} K<br>'
                'P=%{z:.1f} MPa<extra></extra>'),
        ))
        phase_centroids[phase] = {
            'V': float(np.mean(V_ph)),
            'T': float(np.mean(T_ph)),
            'P': float(np.mean(P_ph)),
            'label': label,
        }

    # --- Two-phase ruled surfaces -----------------------------------------
    n_across = 5
    two_phase_centroids = {}
    shown_tp_legend = False
    for key, bounds in diagram['two_phase_bounds'].items():
        phase_left, phase_right = key
        T_b = bounds['T']
        V_l, V_r = bounds['V_left'], bounds['V_right']
        P_b = bounds['P_coex']
        keep = ((V_r >= V_min) & (V_l <= V_max)
                & (P_b <= P_max) & (P_b > 0))
        T_b, V_l, V_r, P_b = T_b[keep], V_l[keep], V_r[keep], P_b[keep]
        if len(T_b) < 2:
            continue
        V_l = np.clip(V_l, V_min, V_max)
        V_r = np.clip(V_r, V_min, V_max)
        n = len(T_b)
        V_grid, T_grid, P_grid = [], [], []
        for i in range(n):
            for j in range(n_across):
                frac = j / (n_across - 1)
                V_grid.append(V_l[i] + frac * (V_r[i] - V_l[i]))
                T_grid.append(T_b[i])
                P_grid.append(P_b[i])
        ii, jj, kk = [], [], []
        for i in range(n - 1):
            for j in range(n_across - 1):
                i00 = i * n_across + j
                ii.extend([i00, i00 + 1])
                jj.extend([i00 + 1, (i + 1) * n_across + j + 1])
                kk.extend([(i + 1) * n_across + j,
                           (i + 1) * n_across + j])
        label_l = PHASE_CONFIGS.get(phase_left, {}).get('label', phase_left)
        label_r = PHASE_CONFIGS.get(phase_right, {}).get('label', phase_right)
        tp_label = f'{label_l} + {label_r}'
        traces.append(go.Mesh3d(
            x=np.array(V_grid), y=np.array(T_grid), z=np.array(P_grid),
            i=np.array(ii), j=np.array(jj), k=np.array(kk),
            color='#bbbbbb', opacity=0.5,
            name='2-phase coexistence',
            legendgroup='two_phase',
            showlegend=(not shown_tp_legend),
            flatshading=True,
            hovertemplate=(
                f'<b>2-phase: {tp_label}</b><br>'
                'V=%{x:.4e} m³/kg<br>'
                'T=%{y:.1f} K<br>'
                'P=%{z:.1f} MPa<extra></extra>'),
        ))
        shown_tp_legend = True
        two_phase_centroids[key] = {
            'V': float(np.mean(V_grid)),
            'T': float(np.mean(T_grid)),
            'P': float(np.mean(P_grid)),
            'label': tp_label,
        }

    # --- Boundary curves (no legend) --------------------------------------
    for key, bounds in diagram['two_phase_bounds'].items():
        phase_left, phase_right = key
        label_l = PHASE_CONFIGS.get(phase_left, {}).get('label', phase_left)
        label_r = PHASE_CONFIGS.get(phase_right, {}).get('label', phase_right)
        for V_side in ('V_left', 'V_right'):
            V_b, T_b, P_b = bounds[V_side], bounds['T'], bounds['P_coex']
            keep = ((V_b >= V_min) & (V_b <= V_max)
                    & (P_b <= P_max) & (P_b > 0))
            if not keep.any():
                continue
            traces.append(go.Scatter3d(
                x=V_b[keep], y=T_b[keep], z=P_b[keep],
                mode='lines', line=dict(color='black', width=3),
                showlegend=False,
                hovertemplate=(
                    f'<b>2-phase: {label_l}/{label_r}</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'T=%{y:.1f} K<br>'
                    'P=%{z:.1f} MPa<extra></extra>'),
            ))

    # --- Saturation curve (no legend) -------------------------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        mask = ((sat['V_liq'] >= V_min) & (sat['V_liq'] <= V_max)
                & (sat['P_sat'] <= P_max))
        if mask.any():
            traces.append(go.Scatter3d(
                x=sat['V_liq'][mask], y=sat['T'][mask],
                z=sat['P_sat'][mask],
                mode='lines', line=dict(color='black', width=3),
                showlegend=False,
                hovertemplate=(
                    '<b>Saturation (Liquid/Vapor)</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'T=%{y:.1f} K<br>'
                    'P=%{z:.4f} MPa<extra></extra>'),
            ))

    # --- Three-phase invariant lines --------------------------------------
    # Derive invariant lines from the two_phase_bounds data so they sit
    # exactly on the boundary curves / mesh vertices.
    shown_3phase = False
    for inv in diagram.get('invariants', []):
        if inv['P_coex'] is None or inv['V_min'] is None:
            continue
        trio = set(inv['phases'])
        T_inv = inv['T']
        # Collect V endpoints and P from the two-phase boundaries
        # that participate in this invariant (both phases in trio).
        V_endpoints = []
        P_values = []
        T_used = None
        for key, bounds in diagram['two_phase_bounds'].items():
            if not set(key).issubset(trio):
                continue
            idx_b = int(np.argmin(np.abs(bounds['T'] - T_inv)))
            if abs(float(bounds['T'][idx_b]) - T_inv) > dT_step * 2:
                continue
            T_used = float(bounds['T'][idx_b])
            V_endpoints.append(float(bounds['V_left'][idx_b]))
            V_endpoints.append(float(bounds['V_right'][idx_b]))
            P_values.append(float(bounds['P_coex'][idx_b]))
        if not V_endpoints or T_used is None:
            continue
        V_lo = max(min(V_endpoints), V_min)
        V_hi = min(max(V_endpoints), V_max)
        P_inv = float(np.mean(P_values))
        if V_lo >= V_hi or P_inv > P_max or P_inv <= 0:
            continue
        phase_names = [PHASE_CONFIGS.get(p, {}).get('label', p)
                       for p in inv['phases']]
        inv_label = ' / '.join(phase_names)
        traces.append(go.Scatter3d(
            x=[V_lo, V_hi], y=[T_used, T_used],
            z=[P_inv, P_inv],
            mode='lines', line=dict(color='red', width=5),
            name='3-phase coexistence',
            legendgroup='3phase_boundary',
            showlegend=(not shown_3phase),
            hovertemplate=(
                '<b>3-phase: ' + inv_label + '</b><br>'
                'T=' + f'{T_used:.1f}' + ' K<br>'
                'P=' + f'{P_inv:.1f}' + ' MPa<extra></extra>'),
        ))
        shown_3phase = True

    # Triple point invariant — use saturation data at nearest slice T
    if ('saturation' in diagram
            and 'Ih' in diagram.get('phase_list', [])):
        sat = diagram['saturation']
        T_triple = 273.16
        temps = diagram['temperatures']
        if len(sat['T']) > 0 and temps[0] <= T_triple <= temps[-1]:
            # Use the nearest actual slice temperature
            idx_snap = int(np.argmin(np.abs(temps - T_triple)))
            T_snap = float(temps[idx_snap])
            # Get saturation V/P at this T
            idx_sat = int(np.argmin(np.abs(sat['T'] - T_snap)))
            V_liq_tp = sat['V_liq'][idx_sat]
            P_triple = sat['P_sat'][idx_sat]
            traces.append(go.Scatter3d(
                x=[V_liq_tp, V_max], y=[T_snap, T_snap],
                z=[P_triple, P_triple],
                mode='lines', line=dict(color='red', width=5),
                name='3-phase coexistence',
                legendgroup='3phase_boundary',
                showlegend=(not shown_3phase),
                hovertemplate=(
                    '<b>3-phase: Water / Ice Ih / Vapor</b><br>'
                    'T=' + f'{T_snap:.1f}' + ' K<br>'
                    'P=' + f'{P_triple:.4f}' + ' MPa<extra></extra>'),
            ))
            shown_3phase = True

    # --- Region text annotations ------------------------------------------
    # Place labels at actual surface points (median T, median V at that T)
    # so they sit on the surface rather than floating.
    ann_traces = []
    for phase, arr in phase_arrays.items():
        if phase not in phase_centroids:
            continue
        label = phase_centroids[phase]['label']
        V_ph, T_ph, P_ph = arr['V'], arr['T'], arr['P']
        # Find point at median T, median V
        T_med = float(np.median(T_ph))
        near_T = np.abs(T_ph - T_med) <= dT_eff
        if not near_T.any():
            near_T = np.ones(len(T_ph), dtype=bool)
        V_sub, P_sub, T_sub = V_ph[near_T], P_ph[near_T], T_ph[near_T]
        V_med = float(np.median(V_sub))
        closest = int(np.argmin(np.abs(V_sub - V_med)))
        ann_traces.append(go.Scatter3d(
            x=[V_sub[closest]], y=[T_sub[closest]], z=[P_sub[closest]],
            mode='text', text=[label],
            textfont=dict(size=11, color='black', family='Arial Black'),
            textposition='middle center',
            showlegend=False, hoverinfo='skip',
        ))
    for key, c in two_phase_centroids.items():
        ann_traces.append(go.Scatter3d(
            x=[c['V']], y=[c['T']], z=[c['P']],
            mode='text', text=[c['label']],
            textfont=dict(size=9, color='#444444', family='Arial Black'),
            textposition='middle center',
            showlegend=False, hoverinfo='skip',
        ))
    traces.extend(ann_traces)

    # --- Assemble figure --------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='P-T-V Phase Diagram of H₂O',
        scene=dict(
            xaxis_title='Specific volume V (m³/kg)',
            yaxis_title='Temperature T (K)',
            zaxis_title='Pressure P (MPa)',
            xaxis=dict(range=[V_min, V_max]),
            zaxis=dict(range=[0, P_max]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=1000, height=800,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: collect P bounds for each phase at each temperature
# ---------------------------------------------------------------------------
def _collect_phase_TP_bounds(diagram, P_cap=2500.0):
    """For each single phase at each T, collect P_upper and P_lower bounds.

    The slice ordering is: sp[0] (lowest V, highest P) | tp[0] | sp[1] | …
    So sp[i] has:
      - P_upper = tp[i-1].P_coex  (if i > 0, else P_cap)
      - P_lower = tp[i].P_coex    (if i < len(tp), else saturation or 0)

    Returns dict  {phase: {T, P_upper, P_lower, V_min, V_max}} with list values.
    """
    from collections import defaultdict
    out = defaultdict(lambda: {'T': [], 'P_upper': [], 'P_lower': [],
                                'V_min': [], 'V_max': []})
    sat = diagram.get('saturation')

    for T, s in zip(diagram['temperatures'], diagram['slices']):
        sps = s['single_phase']
        tps = s['two_phase']
        for i, sp in enumerate(sps):
            phase = sp['phase']
            P_up = tps[i - 1]['P_coex'] if i > 0 else P_cap
            if i < len(tps):
                P_lo = tps[i]['P_coex']
            else:
                # Lowest-P phase: use saturation if available
                if sat is not None and len(sat['T']) > 0 and T >= sat['T'][0]:
                    idx_s = int(np.argmin(np.abs(sat['T'] - T)))
                    P_lo = float(sat['P_sat'][idx_s])
                else:
                    P_lo = 0.0
            if P_up <= 0:
                P_up = 0.0
            out[phase]['T'].append(float(T))
            out[phase]['P_upper'].append(float(P_up))
            out[phase]['P_lower'].append(float(P_lo))
            out[phase]['V_min'].append(sp['V_min'])
            out[phase]['V_max'].append(sp['V_max'])

    # Convert to arrays
    return {ph: {k: np.array(v) for k, v in d.items()}
            for ph, d in out.items()}


# ---------------------------------------------------------------------------
# Helper: invariant lines derived from two_phase_bounds
# ---------------------------------------------------------------------------
def _smooth_segmented(arr, window=9, polyorder=2,
                      jump_threshold_factor=10):
    """Smooth an array in segments separated by discontinuous jumps.

    Uses Savitzky-Golay filter within each contiguous segment so that
    smoothing does not blur across large discontinuities (e.g. invariant
    transitions in phase boundaries).

    Parameters
    ----------
    arr : array-like
        1-D data to smooth.
    window : int
        Savitzky-Golay window length (reduced automatically for short
        segments).
    polyorder : int
        Polynomial order for Savitzky-Golay filter.
    jump_threshold_factor : float
        A step |V[i+1]-V[i]| exceeding this many times the median step
        size is treated as a discontinuity.

    Returns
    -------
    np.ndarray
        Smoothed copy of *arr* with the same length.
    """
    from scipy.signal import savgol_filter

    V = np.asarray(arr, dtype=float)
    if len(V) < 4:
        return V.copy()

    dV = np.abs(np.diff(V))
    median_dV = float(np.median(dV)) if len(dV) > 0 else 0.0

    if median_dV <= 0:
        return V.copy()

    threshold = jump_threshold_factor * median_dV
    jump_indices = np.where(dV > threshold)[0]

    # Segment boundaries: [0, j1+1, j2+1, ..., len]
    bounds = [0] + (jump_indices + 1).tolist() + [len(V)]

    V_smooth = V.copy()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg_len = e - s
        if seg_len < 4:
            continue
        win = min(window, seg_len)
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            V_smooth[s:e] = savgol_filter(V[s:e], win, polyorder=polyorder)

    return V_smooth


def _invariant_coords(diagram, V_min=None, V_max=None, P_max=None):
    """Compute snapped (T, P, V_lo, V_hi) for each invariant.

    Returns list of dicts with keys: T, P, V_lo, V_hi, label, phases.
    """
    temperatures = diagram['temperatures']
    dT_step = (float(temperatures[1] - temperatures[0])
               if len(temperatures) > 1 else 1.0)
    results = []

    for inv in diagram.get('invariants', []):
        if inv['P_coex'] is None or inv['V_min'] is None:
            continue
        trio = set(inv['phases'])
        T_inv = inv['T']
        V_endpoints, P_values, T_used = [], [], None
        for key, bounds in diagram['two_phase_bounds'].items():
            if not set(key).issubset(trio):
                continue
            idx_b = int(np.argmin(np.abs(bounds['T'] - T_inv)))
            if abs(float(bounds['T'][idx_b]) - T_inv) > dT_step * 2:
                continue
            T_used = float(bounds['T'][idx_b])
            V_endpoints.append(float(bounds['V_left'][idx_b]))
            V_endpoints.append(float(bounds['V_right'][idx_b]))
            P_values.append(float(bounds['P_coex'][idx_b]))
        if not V_endpoints or T_used is None:
            continue
        V_lo = min(V_endpoints)
        V_hi = max(V_endpoints)
        if V_min is not None:
            V_lo = max(V_lo, V_min)
        if V_max is not None:
            V_hi = min(V_hi, V_max)
        P_inv = float(np.mean(P_values))
        if P_max is not None and P_inv > P_max:
            continue
        if P_inv <= 0 or V_lo >= V_hi:
            continue
        phase_names = [PHASE_CONFIGS.get(p, {}).get('label', p)
                       for p in inv['phases']]
        results.append({
            'T': T_used, 'P': P_inv,
            'V_lo': V_lo, 'V_hi': V_hi,
            'label': ' / '.join(phase_names),
            'phases': inv['phases'],
        })

    # Triple point
    sat = diagram.get('saturation')
    if (sat is not None and 'Ih' in diagram.get('phase_list', [])
            and len(sat['T']) > 0):
        temps = diagram['temperatures']
        T_triple = 273.16
        if temps[0] <= T_triple <= temps[-1]:
            idx_snap = int(np.argmin(np.abs(temps - T_triple)))
            T_snap = float(temps[idx_snap])
            idx_sat = int(np.argmin(np.abs(sat['T'] - T_snap)))
            V_liq_tp = float(sat['V_liq'][idx_sat])
            P_triple = float(sat['P_sat'][idx_sat])
            V_hi_tp = V_max if V_max is not None else 1.2e-3
            results.append({
                'T': T_snap, 'P': P_triple,
                'V_lo': V_liq_tp, 'V_hi': V_hi_tp,
                'label': 'Water / Ice Ih / Vapor',
                'phases': ('water1', 'Ih', 'vapor'),
            })

    return results


# ---------------------------------------------------------------------------
# Plotting: 2D T-V phase diagram (Plotly)
# ---------------------------------------------------------------------------
def plot_tv_phase_diagram_plotly(diagram,
                                  V_min=7e-4, V_max=1.1e-3,
                                  T_min=None, T_max=None):
    """Interactive Plotly T-V phase diagram with filled phase regions.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    V_min, V_max : float
        Specific volume display range (m^3/kg).
    T_min, T_max : float or None
        Temperature display range (K). Defaults to diagram extent.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    temps = diagram['temperatures']
    if T_min is None:
        T_min = float(temps[0])
    if T_max is None:
        T_max = float(temps[-1])

    traces = []

    # --- Filled single-phase regions --------------------------------------
    for phase, pf in diagram['phase_fields'].items():
        if len(pf['T']) == 0:
            continue
        color = PHASE_CONFIGS.get(phase, {}).get('color', '#888888')
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        T_arr = pf['T']
        V_lo = np.clip(_smooth_segmented(pf['V_min']), V_min, V_max)
        V_hi = np.clip(_smooth_segmented(pf['V_max']), V_min, V_max)
        # Polygon: V_min forward, V_max reversed
        poly_x = np.concatenate([V_lo, V_hi[::-1]])
        poly_y = np.concatenate([T_arr, T_arr[::-1]])
        traces.append(go.Scatter(
            x=poly_x, y=poly_y,
            fill='toself', fillcolor=color,
            opacity=0.5, line=dict(width=0),
            name=label, showlegend=False,
            hovertemplate=(
                f'<b>{label}</b><br>'
                'V=%{x:.4e} m³/kg<br>'
                'T=%{y:.1f} K<extra></extra>'),
        ))

    # --- Two-phase boundary lines -----------------------------------------
    for key, bounds in diagram['two_phase_bounds'].items():
        ph_l, ph_r = key
        label_l = PHASE_CONFIGS.get(ph_l, {}).get('label', ph_l)
        label_r = PHASE_CONFIGS.get(ph_r, {}).get('label', ph_r)
        for V_side in ('V_left', 'V_right'):
            V_b = _smooth_segmented(bounds[V_side])
            T_b = bounds['T']
            mask = (V_b >= V_min) & (V_b <= V_max)
            if not mask.any():
                continue
            traces.append(go.Scatter(
                x=V_b[mask], y=T_b[mask],
                mode='lines', line=dict(color='black', width=1),
                showlegend=False,
                hovertemplate=(
                    f'<b>{label_l}/{label_r}</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'T=%{y:.1f} K<extra></extra>'),
            ))

    # --- Saturation curve -------------------------------------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        mask = (sat['V_liq'] >= V_min) & (sat['V_liq'] <= V_max)
        if mask.any():
            traces.append(go.Scatter(
                x=sat['V_liq'][mask], y=sat['T'][mask],
                mode='lines', line=dict(color='black', width=1.5),
                showlegend=False,
                hovertemplate=(
                    '<b>Saturation (L/V)</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'T=%{y:.1f} K<extra></extra>'),
            ))

    # --- Three-phase invariant lines (red) --------------------------------
    invs = _invariant_coords(diagram, V_min=V_min, V_max=V_max)
    shown_3ph = False
    for inv in invs:
        traces.append(go.Scatter(
            x=[inv['V_lo'], inv['V_hi']],
            y=[inv['T'], inv['T']],
            mode='lines', line=dict(color='red', width=2.5),
            name='3-phase coexistence',
            legendgroup='3phase',
            showlegend=(not shown_3ph),
            hovertemplate=(
                '<b>3-phase: ' + inv['label'] + '</b><br>'
                'T=' + f'{inv["T"]:.1f}' + ' K<br>'
                'P=' + f'{inv["P"]:.2f}' + ' MPa<extra></extra>'),
        ))
        shown_3ph = True

    # --- Region labels ----------------------------------------------------
    for phase, pf in diagram['phase_fields'].items():
        if len(pf['T']) == 0:
            continue
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        T_mid = float(np.median(pf['T']))
        idx = int(np.argmin(np.abs(pf['T'] - T_mid)))
        V_mid = (pf['V_min'][idx] + pf['V_max'][idx]) / 2.0
        if V_min <= V_mid <= V_max:
            traces.append(go.Scatter(
                x=[V_mid], y=[T_mid],
                mode='text', text=[label],
                textfont=dict(size=12, color='black', family='Arial Black'),
                textposition='middle center',
                showlegend=False, hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='T-V Phase Diagram of H₂O',
        xaxis_title='Specific volume V (m³/kg)',
        yaxis_title='Temperature T (K)',
        xaxis=dict(range=[V_min, V_max]),
        yaxis=dict(range=[T_min, T_max]),
        width=900, height=700,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotting: 2D T-P phase diagram (Plotly)
# ---------------------------------------------------------------------------
def plot_tp_phase_diagram_plotly(diagram,
                                  T_min=None, T_max=None,
                                  P_min=1e-4, P_max=2500.0):
    """Interactive Plotly T-P phase diagram with filled phase regions.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    T_min, T_max : float or None
        Temperature display range (K). Defaults to diagram extent.
    P_min, P_max : float
        Pressure display range (MPa).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    temps = diagram['temperatures']
    if T_min is None:
        T_min = float(temps[0])
    if T_max is None:
        T_max = float(temps[-1])

    bounds_data = _collect_phase_TP_bounds(diagram, P_cap=P_max)
    traces = []

    # --- Filled single-phase regions --------------------------------------
    for phase, bd in bounds_data.items():
        color = PHASE_CONFIGS.get(phase, {}).get('color', '#888888')
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        T_arr = bd['T']
        P_up = np.clip(_smooth_segmented(bd['P_upper']), P_min, P_max)
        P_lo = np.clip(_smooth_segmented(bd['P_lower']), P_min, P_max)
        # Polygon: (T, P_upper) forward, (T, P_lower) reversed
        poly_x = np.concatenate([T_arr, T_arr[::-1]])
        poly_y = np.concatenate([P_up, P_lo[::-1]])
        traces.append(go.Scatter(
            x=poly_x, y=poly_y,
            fill='toself', fillcolor=color,
            opacity=0.5, line=dict(width=0),
            name=label, showlegend=False,
            hovertemplate=(
                f'<b>{label}</b><br>'
                'T=%{x:.1f} K<br>'
                'P=%{y:.2f} MPa<extra></extra>'),
        ))

    # --- Two-phase coexistence curves -------------------------------------
    for key, bounds in diagram['two_phase_bounds'].items():
        ph_l, ph_r = key
        label_l = PHASE_CONFIGS.get(ph_l, {}).get('label', ph_l)
        label_r = PHASE_CONFIGS.get(ph_r, {}).get('label', ph_r)
        T_b = bounds['T']
        P_b = bounds['P_coex']
        mask = (P_b > 0) & (P_b <= P_max)
        if not mask.any():
            continue
        traces.append(go.Scatter(
            x=T_b[mask], y=P_b[mask],
            mode='lines', line=dict(color='black', width=1.5),
            showlegend=False,
            hovertemplate=(
                f'<b>{label_l}/{label_r}</b><br>'
                'T=%{x:.1f} K<br>'
                'P=%{y:.2f} MPa<extra></extra>'),
        ))

    # --- Boiling curve from saturation ------------------------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        mask = (sat['P_sat'] <= P_max) & (sat['P_sat'] >= P_min)
        if mask.any():
            traces.append(go.Scatter(
                x=sat['T'][mask], y=sat['P_sat'][mask],
                mode='lines', line=dict(color='black', width=1.5, dash='dash'),
                showlegend=False,
                hovertemplate=(
                    '<b>Boiling curve (L/V)</b><br>'
                    'T=%{x:.1f} K<br>'
                    'P=%{y:.4f} MPa<extra></extra>'),
            ))

    # --- Three-phase invariant points (red markers) -----------------------
    invs = _invariant_coords(diagram, P_max=P_max)
    shown_3ph = False
    for inv in invs:
        traces.append(go.Scatter(
            x=[inv['T']], y=[inv['P']],
            mode='markers', marker=dict(color='red', size=8, symbol='circle'),
            name='3-phase coexistence',
            legendgroup='3phase',
            showlegend=(not shown_3ph),
            hovertemplate=(
                '<b>3-phase: ' + inv['label'] + '</b><br>'
                'T=' + f'{inv["T"]:.1f}' + ' K<br>'
                'P=' + f'{inv["P"]:.2f}' + ' MPa<extra></extra>'),
        ))
        shown_3ph = True

    # --- Region labels ----------------------------------------------------
    for phase, bd in bounds_data.items():
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        T_mid = float(np.median(bd['T']))
        idx = int(np.argmin(np.abs(bd['T'] - T_mid)))
        P_mid = (bd['P_upper'][idx] + bd['P_lower'][idx]) / 2.0
        # Use geometric mean for log-scale friendliness
        if bd['P_upper'][idx] > 0 and bd['P_lower'][idx] > 0:
            P_mid = np.sqrt(bd['P_upper'][idx] * bd['P_lower'][idx])
        if P_min <= P_mid <= P_max and T_min <= T_mid <= T_max:
            traces.append(go.Scatter(
                x=[T_mid], y=[P_mid],
                mode='text', text=[label],
                textfont=dict(size=12, color='black', family='Arial Black'),
                textposition='middle center',
                showlegend=False, hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='T-P Phase Diagram of H₂O',
        xaxis_title='Temperature T (K)',
        yaxis_title='Pressure P (MPa)',
        xaxis=dict(range=[T_min, T_max]),
        yaxis=dict(range=[P_min, P_max]),
        width=900, height=700,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotting: 2D P-V phase diagram (Plotly)
# ---------------------------------------------------------------------------
def plot_pv_phase_diagram_plotly(diagram,
                                  V_min=7e-4, V_max=1.1e-3,
                                  P_min=1e-4, P_max=2500.0):
    """Interactive Plotly P-V phase diagram with filled phase regions.

    Parameters
    ----------
    diagram : dict
        Output of compute_tv_phase_diagram.
    V_min, V_max : float
        Specific volume display range (m^3/kg).
    P_min, P_max : float
        Pressure display range (MPa).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    bounds_data = _collect_phase_TP_bounds(diagram, P_cap=P_max)
    traces = []

    # --- Filled single-phase regions --------------------------------------
    # For each phase, the P-V envelope is:
    #   Forward:  (V_min[t], P_upper[t]) for t going forward
    #   Backward: (V_max[t], P_lower[t]) for t going backward
    for phase, bd in bounds_data.items():
        color = PHASE_CONFIGS.get(phase, {}).get('color', '#888888')
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        V_lo = np.clip(_smooth_segmented(bd['V_min']), V_min, V_max)
        V_hi = np.clip(_smooth_segmented(bd['V_max']), V_min, V_max)
        P_up = np.clip(_smooth_segmented(bd['P_upper']), P_min, P_max)
        P_lo = np.clip(_smooth_segmented(bd['P_lower']), P_min, P_max)
        # Left edge (V_min, P_upper) → right edge reversed (V_max, P_lower)
        poly_x = np.concatenate([V_lo, V_hi[::-1]])
        poly_y = np.concatenate([P_up, P_lo[::-1]])
        traces.append(go.Scatter(
            x=poly_x, y=poly_y,
            fill='toself', fillcolor=color,
            opacity=0.5, line=dict(width=0),
            name=label, showlegend=False,
            hovertemplate=(
                f'<b>{label}</b><br>'
                'V=%{x:.4e} m³/kg<br>'
                'P=%{y:.2f} MPa<extra></extra>'),
        ))

    # --- Two-phase boundary lines -----------------------------------------
    for key, bounds in diagram['two_phase_bounds'].items():
        ph_l, ph_r = key
        label_l = PHASE_CONFIGS.get(ph_l, {}).get('label', ph_l)
        label_r = PHASE_CONFIGS.get(ph_r, {}).get('label', ph_r)
        P_b = bounds['P_coex']
        mask = (P_b > 0) & (P_b <= P_max)
        if not mask.any():
            continue
        for V_side in ('V_left', 'V_right'):
            V_b = _smooth_segmented(bounds[V_side])
            P_b_s = _smooth_segmented(P_b)
            v_mask = mask & (V_b >= V_min) & (V_b <= V_max)
            if not v_mask.any():
                continue
            traces.append(go.Scatter(
                x=V_b[v_mask], y=P_b_s[v_mask],
                mode='lines', line=dict(color='black', width=1),
                showlegend=False,
                hovertemplate=(
                    f'<b>{label_l}/{label_r}</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'P=%{y:.2f} MPa<extra></extra>'),
            ))

    # --- Saturation curve -------------------------------------------------
    if 'saturation' in diagram:
        sat = diagram['saturation']
        mask = ((sat['V_liq'] >= V_min) & (sat['V_liq'] <= V_max)
                & (sat['P_sat'] >= P_min) & (sat['P_sat'] <= P_max))
        if mask.any():
            traces.append(go.Scatter(
                x=sat['V_liq'][mask], y=sat['P_sat'][mask],
                mode='lines', line=dict(color='black', width=1.5),
                showlegend=False,
                hovertemplate=(
                    '<b>Saturation (L/V)</b><br>'
                    'V=%{x:.4e} m³/kg<br>'
                    'P=%{y:.4f} MPa<extra></extra>'),
            ))

    # --- Three-phase invariant lines (red, horizontal) --------------------
    invs = _invariant_coords(diagram, V_min=V_min, V_max=V_max, P_max=P_max)
    shown_3ph = False
    for inv in invs:
        traces.append(go.Scatter(
            x=[inv['V_lo'], inv['V_hi']],
            y=[inv['P'], inv['P']],
            mode='lines', line=dict(color='red', width=2.5),
            name='3-phase coexistence',
            legendgroup='3phase',
            showlegend=(not shown_3ph),
            hovertemplate=(
                '<b>3-phase: ' + inv['label'] + '</b><br>'
                'T=' + f'{inv["T"]:.1f}' + ' K<br>'
                'P=' + f'{inv["P"]:.2f}' + ' MPa<extra></extra>'),
        ))
        shown_3ph = True

    # --- Region labels ----------------------------------------------------
    for phase, bd in bounds_data.items():
        label = PHASE_CONFIGS.get(phase, {}).get('label', phase)
        T_mid_idx = len(bd['T']) // 2
        V_mid = (bd['V_min'][T_mid_idx] + bd['V_max'][T_mid_idx]) / 2.0
        P_up = bd['P_upper'][T_mid_idx]
        P_lo = bd['P_lower'][T_mid_idx]
        P_mid = np.sqrt(max(P_up, P_min) * max(P_lo, P_min))
        if V_min <= V_mid <= V_max and P_min <= P_mid <= P_max:
            traces.append(go.Scatter(
                x=[V_mid], y=[P_mid],
                mode='text', text=[label],
                textfont=dict(size=12, color='black', family='Arial Black'),
                textposition='middle center',
                showlegend=False, hoverinfo='skip',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='P-V Phase Diagram of H₂O',
        xaxis_title='Specific volume V (m³/kg)',
        yaxis_title='Pressure P (MPa)',
        xaxis=dict(range=[V_min, V_max]),
        yaxis=dict(type='log', range=[np.log10(max(P_min, 1e-6)),
                                       np.log10(P_max)]),
        width=900, height=700,
        legend=dict(x=0.02, y=0.98),
    )
    return fig
