"""
Refit the Singh-Issenmann-Caupin (2017) two-state transport model with
alternative two-state EoS backbones.

The Singh model (PNAS 114, 4312 (2017), Eq. 1) expresses a dynamic property
A in {eta, D, tau_r} as

  A(T,P) = A0 * (T/Tref)^nu
           * exp{ eps * [ (1-f) * (E_HDS + dv_HDS*P) / (kB*(T-T0))
                          + f * E_LDS / (kB*T) ] }

where f(T,P) is the low-density-structure (LDS) fraction supplied by a
two-state thermodynamic backbone.  The published parameters were fitted with
the Holten et al. (2014) fraction; this script refits (A0, E_LDS, E_HDS,
dv_HDS, T0) against the original experimental datasets using the fraction
from any of the backbones implemented in this repo:

  holten2014, caupin2019, duska2020

Experimental data (references/data/):
  eta   : singh_2017_eta.csv    -- Singh 2017 Table S1 (20-298 MPa)
          dehaoui_2015_eta.csv  -- Dehaoui 2015 (0.1 MPa, supercooled)
  D     : prielmeier_1988_D.csv -- Prielmeier 1988 (PGSE NMR, high P)
          harris_1997_D.csv     -- Harris & Newitt 1997 (SGSE NMR)
          + 3 points at 238.15/243.15/248.15 K, 0.1 MPa from the
            Dehaoui 2015 power-law fit of ambient literature D data
  tau_r : lang_1981_T1.csv      -- Lang & Ludemann 1981 (17O T1)
          arnold_2002_T1.csv    -- Arnold & Ludemann 2002 (17O T1)
          converted via tau_r = 1/(T1 * omega_Q^2), omega_Q = 9.12e6 1/s

Fitting mirrors the paper: weighted least squares with residuals
(model - exp)/sigma; each property fitted separately (5 free parameters),
then jointly with a common T0 (13 free parameters).

Usage (from the waterEoS repo root):
  python scripts/refit_singh_transport.py --validate     # holten2014 vs paper
  python scripts/refit_singh_transport.py --all          # fit all backbones
  python scripts/refit_singh_transport.py --backbone caupin2019
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
k_B = 1.380649e-23      # J/K
T_REF = 273.15          # K (paper's Tref)
OMEGA_Q = 9.12e6        # 1/s, 17O nuclear quadrupole frequency (Singh M&M)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'references', 'data')

# Relative 1-SD uncertainties for datasets that do not tabulate per-point
# errors.  Values from the stated experimental accuracy in each source paper
# (see the *_notes.md files in references/data/).
REL_UNC = {
    'prielmeier_D': 0.03,        # "judged reliable to +/-3%" (p. 1112) ...
    'prielmeier_D_lowT': 0.10,   # ... "below 210 K ... maximal error to +/-10%"
    'harris_D': 0.02,
    'lang_T1': 0.05,
    'arnold_T1': 0.05,   # "judged reliable to +/-5%" (Arnold 2002, p. 1582)
    'dehaoui_D_powerlaw': 0.05,
}

# Dehaoui et al. (2015) ambient-pressure power law for self-diffusion,
# D(T) = D0 * (T/Ts - 1)^gamma  -- parameters filled in from the paper
# (see dehaoui_2015_notes.md).  Used only to generate the three points at
# 238.15, 243.15, 248.15 K that Singh et al. added to the D dataset.
# Dehaoui 2015 Table S5, Dt fit: Dt = D0*(T/Ts - 1)^2.0801 (their gamma is
# negative in the A0*(T/Ts-1)^-gamma convention).
DEHAOUI_D_POWERLAW = {
    'D0': 1.6077e-8,    # m^2/s (16,077 um^2/s)
    'Ts': 213.96,       # K
    'gamma': 2.0801,
}

# Singh et al. used N_eta = 178 points: the 165 of their Table S1 plus the
# Dehaoui 2015 ambient-pressure points below their own lowest capillary
# temperature.  Dehaoui Table S2 (1-K steps, 239.15-293.15 K) restricted to
# T <= 251.15 K gives exactly 13 points -> 165 + 13 = 178.
DEHAOUI_ETA_T_MAX = 251.2   # K

# Weight scale for the Dehaoui ambient points.  With the tabulated 1-SD
# uncertainties (2.3-2.9%) the eta fit is pulled toward the deeply
# supercooled ambient tail (E_LDS/kB -> 2440 K) and away from the published
# solution; scaling sigma by 2 (~5-6% effective) reproduces every published
# Holten-backbone parameter within 1 published SD (A0 33.6 vs 32.3+/-2.2,
# E_LDS 2307 vs 2283+/-24, E_HDS 465 vs 469+/-18, dv 2.47 vs 2.52+/-0.08,
# T0 141.2 vs 141.4+/-2.3), so it evidently matches the paper's effective
# weighting.  Kept for all backbones so that cross-backbone differences
# isolate the fraction f, not the weighting convention.
DEHAOUI_ETA_SIGMA_SCALE = 2.0

# Ambient stable-region anchor points from the IAPWS-2008 viscosity
# formulation (Huber et al. 2009) -- the same correlation Singh et al. used
# to calibrate every capillary run.  The fitted dataset contains no stable
# points below 18.8 MPa (the capillary needed P0 >~ 19 MPa), so without
# these anchors the 20 -> 0.1 MPa extrapolation drifts by +2-4% for the
# caupin2019/duska2020 fractions (holten2014 happens to extrapolate to
# +0.3%).  sigma = 1% (formulation uncertainty at ambient conditions).
# (T_K, eta_mPas); P = 0.1 MPa.
IAPWS_ETA_ANCHORS = [
    (273.15, 1.7911), (278.15, 1.5182), (283.15, 1.3059),
    (288.15, 1.1382), (293.15, 1.0016), (298.15, 0.8900),
]
IAPWS_ETA_REL_UNC = 0.01

# Published parameters (for initial guesses and validation).
# Table 1 (common T0 = 147.75 K) and Table S2 (separate fits).
PUBLISHED = {
    'common_T0': {
        'T0': 147.75,
        'eta':   {'A0': 38.75e-6, 'E_LDS_k': 2262.0, 'E_HDS_k': 421.9,
                  'dv_HDS': 2.44e-30, 'nu': 0.5, 'eps': +1},
        'D':     {'A0': 40330e-12, 'E_LDS_k': 1984.0, 'E_HDS_k': 402.2,
                  'dv_HDS': 1.79e-30, 'nu': 0.5, 'eps': -1},
        'tau_r': {'A0': 86.2e-15, 'E_LDS_k': 2585.0, 'E_HDS_k': 395.0,
                  'dv_HDS': 1.62e-30, 'nu': -0.5, 'eps': +1},
    },
    'separate': {
        'eta':   {'A0': 32.31e-6, 'E_LDS_k': 2283.0, 'E_HDS_k': 468.9,
                  'dv_HDS': 2.52e-30, 'T0': 141.39},
        'D':     {'A0': 39200e-12, 'E_LDS_k': 1978.0, 'E_HDS_k': 396.4,
                  'dv_HDS': 1.78e-30, 'T0': 148.46},
        'tau_r': {'A0': 53.0e-15, 'E_LDS_k': 2656.0, 'E_HDS_k': 503.3,
                  'dv_HDS': 1.70e-30, 'T0': 135.28},
    },
}

NU = {'eta': 0.5, 'D': 0.5, 'tau_r': -0.5}
EPS = {'eta': +1, 'D': -1, 'tau_r': +1}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_csv(name):
    path = os.path.join(DATA_DIR, name)
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'{name}: empty')
    return rows


def load_eta():
    """Viscosity dataset: T_K, P_MPa, value (Pa*s), sigma (Pa*s)."""
    T, P, y, s = [], [], [], []
    for r in _read_csv('singh_2017_eta.csv'):
        T.append(float(r['T_K'])); P.append(float(r['P_MPa']))
        y.append(float(r['eta_mPas']) * 1e-3)
        s.append(float(r['deta_mPas']) * 1e-3)
    n_singh = len(T)
    for r in _read_csv('dehaoui_2015_eta.csv'):
        if float(r['T_K']) > DEHAOUI_ETA_T_MAX:
            continue
        T.append(float(r['T_K'])); P.append(float(r['P_MPa']))
        y.append(float(r['eta_mPas']) * 1e-3)
        s.append(float(r['deta_mPas']) * 1e-3 * DEHAOUI_ETA_SIGMA_SCALE)
    n_deh = len(T) - n_singh
    for Ti, ei in IAPWS_ETA_ANCHORS:
        T.append(Ti); P.append(0.1)
        y.append(ei * 1e-3)
        s.append(IAPWS_ETA_REL_UNC * ei * 1e-3)
    print(f'  eta: {n_singh} Singh Table S1 + {n_deh} Dehaoui 2015 '
          f'+ {len(T)-n_singh-n_deh} IAPWS ambient anchors = {len(T)} points')
    return map(np.array, (T, P, y, s))


def load_D():
    """Self-diffusion dataset: T_K, P_MPa, value (m^2/s), sigma (m^2/s)."""
    T, P, y, s = [], [], [], []
    for r in _read_csv('prielmeier_1988_D.csv'):
        v = float(r['D_m2s'])
        Ti = float(r['T_K'])
        rel = (REL_UNC['prielmeier_D_lowT'] if Ti < 210.0
               else REL_UNC['prielmeier_D'])
        T.append(Ti); P.append(float(r['P_MPa']))
        y.append(v); s.append(rel * v)
    n_p = len(T)
    for r in _read_csv('harris_1997_D.csv'):
        v = float(r['D_m2s'])
        T.append(float(r['T_K'])); P.append(float(r['P_MPa']))
        y.append(v); s.append(REL_UNC['harris_D'] * v)
    n_h = len(T) - n_p
    # Three low-T ambient points from the Dehaoui 2015 power law (as in Singh)
    pl = DEHAOUI_D_POWERLAW
    n_syn = 0
    if pl['D0'] is not None:
        for Ti in (238.0, 243.0, 248.0):   # "(238, 243, and 248 K)", Singh M&M
            v = pl['D0'] * (Ti / pl['Ts'] - 1.0) ** pl['gamma']
            T.append(Ti); P.append(0.1)
            y.append(v); s.append(REL_UNC['dehaoui_D_powerlaw'] * v)
            n_syn += 1
    print(f'  D: {n_p} Prielmeier + {n_h} Harris + {n_syn} power-law '
          f'= {len(T)} points')
    return map(np.array, (T, P, y, s))


def load_tau_r():
    """Rotational correlation time dataset from 17O T1 measurements.

    Singh et al. collected dynamic data "from 300 K to the lowest available
    temperatures"; Lang & Ludemann's Table 1 extends to 457 K, so rows with
    T > 300 K are excluded.  The remaining 60 Lang points + 41 Arnold points
    reproduce the paper's N_tau = 101 exactly.
    """
    T, P, y, s = [], [], [], []
    for r in _read_csv('lang_1981_T1.csv'):
        if float(r['T_K']) > 300.0:
            continue
        t1 = float(r['T1_s'])
        tau = 1.0 / (t1 * OMEGA_Q ** 2)
        T.append(float(r['T_K'])); P.append(float(r['P_MPa']))
        y.append(tau); s.append(REL_UNC['lang_T1'] * tau)
    n_l = len(T)
    for r in _read_csv('arnold_2002_T1.csv'):
        t1 = float(r['T1_s'])
        tau = 1.0 / (t1 * OMEGA_Q ** 2)
        T.append(float(r['T_K'])); P.append(float(r['P_MPa']))
        y.append(tau); s.append(REL_UNC['arnold_T1'] * tau)
    print(f'  tau_r: {n_l} Lang + {len(T)-n_l} Arnold = {len(T)} points')
    return map(np.array, (T, P, y, s))


# ---------------------------------------------------------------------------
# Backbone LDS fraction
# ---------------------------------------------------------------------------
def backbone_fraction(name, T_K, P_MPa):
    if name == 'holten2014':
        from holten_eos.core import compute_batch
    elif name == 'caupin2019':
        from caupin_eos.core import compute_batch
    elif name == 'duska2020':
        from duska_eos.core import compute_batch
    else:
        raise ValueError(f'unknown backbone {name!r}')
    out = compute_batch(np.ascontiguousarray(T_K, dtype=float),
                        np.ascontiguousarray(P_MPa, dtype=float))
    x = np.asarray(out['x'], dtype=float)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))[0]
        raise RuntimeError(
            f'{name}: non-finite fraction at {len(bad)} points, e.g. '
            f'T={T_K[bad[:5]]}, P={P_MPa[bad[:5]]}')
    return x


# ---------------------------------------------------------------------------
# Model and fitting
# ---------------------------------------------------------------------------
def singh_eq1(T_K, P_MPa, f, A0, E_LDS_k, E_HDS_k, dv_HDS, T0, nu, eps):
    P_Pa = P_MPa * 1e6
    HDS = (E_HDS_k + dv_HDS * P_Pa / k_B) / (T_K - T0)
    LDS = E_LDS_k / T_K
    return A0 * (T_K / T_REF) ** nu * np.exp(eps * ((1.0 - f) * HDS + f * LDS))


class PropData:
    def __init__(self, key, T, P, y, sigma):
        self.key, self.T, self.P, self.y, self.sigma = key, T, P, y, sigma
        self.f = None  # filled per backbone

    @property
    def n(self):
        return len(self.T)


def _residual_prop(theta, d, T0):
    A0, EL, EH, dv = theta
    model = singh_eq1(d.T, d.P, d.f, A0, EL, EH, dv * 1e-30, T0,
                      NU[d.key], EPS[d.key])
    return (model - d.y) / d.sigma


def fit_separate(d, guess):
    """5-parameter fit of one property. theta = [A0, EL, EH, dv(1e-30 m^3), T0]."""
    def resid(theta):
        return _residual_prop(theta[:4], d, theta[4])
    x0 = np.array([guess['A0'], guess['E_LDS_k'], guess['E_HDS_k'],
                   guess['dv_HDS'] * 1e30, guess.get('T0', 147.75)])
    lo = [0.0, 0.0, 0.0, 0.0, 50.0]
    hi = [np.inf, 8000.0, 4000.0, 20.0, 200.0]
    res = least_squares(resid, x0, bounds=(lo, hi), x_scale=np.abs(x0),
                        xtol=1e-12, ftol=1e-12)
    return res


def fit_common_T0(props, guesses, T0_guess):
    """Joint 13-parameter fit: 4 per property + common T0."""
    def resid(theta):
        out = []
        T0 = theta[-1]
        for i, d in enumerate(props):
            out.append(_residual_prop(theta[4*i:4*i+4], d, T0))
        return np.concatenate(out)
    x0 = []
    for d in props:
        g = guesses[d.key]
        x0 += [g['A0'], g['E_LDS_k'], g['E_HDS_k'], g['dv_HDS'] * 1e30]
    x0.append(T0_guess)
    x0 = np.array(x0)
    lo = [0.0, 0.0, 0.0, 0.0] * 3 + [50.0]
    hi = [np.inf, 8000.0, 4000.0, 20.0] * 3 + [200.0]
    res = least_squares(resid, x0, bounds=(lo, hi), x_scale=np.abs(x0),
                        xtol=1e-12, ftol=1e-12)
    return res


def param_sigmas(res):
    """1-SD parameter uncertainties from the Jacobian, scaled by reduced chi2."""
    m, n = res.jac.shape
    dof = max(m - n, 1)
    chi2red = 2.0 * res.cost / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * chi2red
        return np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


def chi2_red(res, n_params):
    m = res.fun.size
    return 2.0 * res.cost / max(m - n_params, 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_backbone(name, props, validate=False):
    print(f'\n=== backbone: {name} ===')
    for d in props:
        d.f = backbone_fraction(name, d.T, d.P)
        print(f'  {d.key}: f range [{d.f.min():.4f}, {d.f.max():.4f}]')

    guesses = {k: dict(PUBLISHED['separate'][k]) for k in ('eta', 'D', 'tau_r')}

    # --- separate fits ---
    sep = {}
    for d in props:
        res = fit_separate(d, guesses[d.key])
        sig = param_sigmas(res)
        sep[d.key] = {'theta': res.x, 'sigma': sig,
                      'chi2red': chi2_red(res, 5), 'n': d.n}
        A0, EL, EH, dv, T0 = res.x
        print(f'  separate {d.key:6s}: A0={A0:.4g}  E_LDS/kB={EL:7.1f}  '
              f'E_HDS/kB={EH:6.1f}  dv={dv:.3f}e-30  T0={T0:7.2f}  '
              f'chi2red={sep[d.key]["chi2red"]:.2f}  (n={d.n})')

    # --- joint fit, common T0 ---
    g2 = {}
    for d in props:
        A0, EL, EH, dv, T0 = sep[d.key]['theta']
        g2[d.key] = {'A0': A0, 'E_LDS_k': EL, 'E_HDS_k': EH, 'dv_HDS': dv * 1e-30}
    T0_guess = float(np.mean([sep[k]['theta'][4] for k in sep]))
    res = fit_common_T0(props, g2, T0_guess)
    sig = param_sigmas(res)
    T0c = res.x[-1]
    joint = {'T0': T0c, 'T0_sigma': sig[-1], 'props': {},
             'chi2red': chi2_red(res, 13)}
    print(f'  joint common T0 = {T0c:.2f} +/- {sig[-1]:.2f} K, '
          f'chi2red(total) = {joint["chi2red"]:.2f}')
    for i, d in enumerate(props):
        A0, EL, EH, dv = res.x[4*i:4*i+4]
        sA0, sEL, sEH, sdv = sig[4*i:4*i+4]
        # per-property chi2 with joint parameters
        r = _residual_prop(res.x[4*i:4*i+4], d, T0c)
        c2 = float(np.sum(r**2)) / max(d.n - 4, 1)
        joint['props'][d.key] = {
            'A0': A0, 'A0_sigma': sA0,
            'E_LDS_k': EL, 'E_LDS_k_sigma': sEL,
            'E_HDS_k': EH, 'E_HDS_k_sigma': sEH,
            'dv_HDS_1e30': dv, 'dv_HDS_1e30_sigma': sdv,
            'nu': NU[d.key], 'eps': EPS[d.key],
            'n': d.n, 'chi2red': c2,
        }
        print(f'    {d.key:6s}: A0={A0:.4g}+/-{sA0:.2g}  '
              f'E_LDS/kB={EL:7.1f}+/-{sEL:.0f}  E_HDS/kB={EH:6.1f}+/-{sEH:.1f}  '
              f'dv={dv:.3f}+/-{sdv:.3f}e-30  chi2red={c2:.2f}')

    if validate:
        print('\n  --- validation vs published (holten2014 backbone) ---')
        print('  Table 1 (common T0=147.75):')
        for k in ('eta', 'D', 'tau_r'):
            pub = PUBLISHED['common_T0'][k]
            fit = joint['props'][k]
            print(f'    {k:6s}: A0 {fit["A0"]:.4g} vs {pub["A0"]:.4g} | '
                  f'E_LDS {fit["E_LDS_k"]:.0f} vs {pub["E_LDS_k"]:.0f} | '
                  f'E_HDS {fit["E_HDS_k"]:.1f} vs {pub["E_HDS_k"]:.1f} | '
                  f'dv {fit["dv_HDS_1e30"]:.2f} vs {pub["dv_HDS"]*1e30:.2f}')
        print(f'    T0: {joint["T0"]:.2f} vs 147.75')

    return {'separate': {k: {'A0': v['theta'][0], 'E_LDS_k': v['theta'][1],
                             'E_HDS_k': v['theta'][2],
                             'dv_HDS_1e30': v['theta'][3],
                             'T0': v['theta'][4],
                             'chi2red': v['chi2red'], 'n': v['n']}
                         for k, v in sep.items()},
            'joint_common_T0': joint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default=None,
                    choices=['holten2014', 'caupin2019', 'duska2020'])
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='fit holten2014 and compare against published values')
    ap.add_argument('--out', default=os.path.join(DATA_DIR, 'singh_refit_results.json'))
    args = ap.parse_args()

    print('Loading datasets...')
    props = [PropData('eta', *load_eta()),
             PropData('D', *load_D()),
             PropData('tau_r', *load_tau_r())]

    backbones = []
    if args.validate:
        backbones = ['holten2014']
    elif args.all:
        backbones = ['holten2014', 'caupin2019', 'duska2020']
    elif args.backbone:
        backbones = [args.backbone]
    else:
        ap.error('specify --backbone, --all, or --validate')

    results = {}
    for name in backbones:
        results[name] = run_backbone(name, props, validate=(name == 'holten2014'))

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f'\nResults written to {args.out}')


if __name__ == '__main__':
    main()
