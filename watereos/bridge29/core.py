"""Bridge29 equilibrium Gibbs equation and finite-barrier transport model.

Pure-state Gibbs derivatives use molar units, K and MPa. The public results
use the specific SI units of watereos, with bulk moduli in MPa.
"""
import json
from importlib.resources import files

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import expit, xlogy

from watereos.two_state_branch import mask_mixture, select_branch

from .ice import M, ice

with (files("watereos") / "data" / "bridge29.json").open() as stream:
    _data = json.load(stream)
Q = _data["thermodynamics"]
R = 8.314462
RM = R / Q["N_c"]
P_SCALE = 22.064
G_SCALE = 461.523 * 647.096 * M


def _series(key):
    return Chebyshev([Q[f"{key}{i}"] for i in range(3)]).convert(kind=Polynomial)


PS, VS, PHI, CUBIC = map(_series, ("ps", "vs", "ph", "cs"))
PS_T, VS_T, PHI_T, CUBIC_T = (s.deriv() for s in (PS, VS, PHI, CUBIC))
# Integral of the spinodal volume times the pressure differential.
IVP = (VS * PS_T).integ()


def _oscillator(T, theta):
    u = theta / T
    log_partition = np.log(-np.expm1(-u))
    population = 1 / np.expm1(u)
    return (R*T*log_partition, R*(log_partition - u*population),
            -R/T * u**2 * population * (1 + population))


def _low(T, P):
    g, gt, gp, gtt, gtp, gpp = ice(T, P)
    a, at, att = _oscillator(T, Q["th_L"])
    n, k, v = Q["n_L"], Q["k_L"], Q["v_L"]
    return (g + Q["h_L"] - T*Q["s_L"] + n*a + P*(v - 0.5*k*P),
            gt - Q["s_L"] + n*at, gp + v - k*P, gtt + n*att, gtp, gpp-k)


def _high(T, P):
    """Return G, G_T and G_P. Analytic continuation permits complex-step derivatives."""
    teff = (T**4 + 100.**4)**0.25
    t = (teff - 200.) / 100.
    dt = (T / teff)**3 / 100.
    ps, vs, phi, c = PS(t), VS(t), PHI(t), CUBIC(t)
    ps_t, vs_t, c_t = PS_T(t), VS_T(t), CUBIC_T(t)
    b = -phi / (2 * ps * vs**2)
    b_t = b * (PHI_T(t)/phi - ps_t/ps - 2*vs_t/vs)
    y = 1 - P/(P_SCALE*ps)
    valid = ((ps.real < 0) & (vs.real > 0) & (b.real > 0) & (y.real > 0)
             & ((c.real >= 0) | (27 * c.real**2 * y.real < 4*b.real**3)))
    d = np.sqrt(np.where(valid, y/b, np.nan))
    # Physical root of b*d² + c*d³ = y, continuous from d=0 at the spinodal.
    for _ in range(30):
        correction = (d*d*(b+c*d)-y) / (d*(2*b+3*c*d))
        d = d - correction
        if np.all(~valid | (abs(correction) <= 3e-15*(1+abs(d)))):
            break
    d_t = ((1-y)*ps_t/ps - d*d*(b_t+c_t*d)) / (d*(2*b+3*c*d))
    logv = np.log1p(d)
    j2, j3 = 2*(d-logv), 3*(0.5*d*d-d+logv)
    j = b*j2 + c*j3
    j_t = b_t*j2 + c_t*j3 + d*(2*b+3*c*d)/(1+d)*d_t
    g = G_SCALE*(IVP(t) - ps*vs*j) + Q["h_H"] - T*Q["s_H"]
    gt = G_SCALE*dt*(vs*ps_t - (ps_t*vs+ps*vs_t)*j - ps*vs*j_t) - Q["s_H"]
    gp = G_SCALE/P_SCALE * vs/(1+d)
    for theta, n in ((100., Q["n_H0"]), (500., Q["n_H2"]), (1000., Q["n_H3"])):
        a, at, _ = _oscillator(T, theta)
        g, gt = g + n*a, gt + n*at
    gap = Q["e_x"] + P*Q["v_x"]
    excitation = np.exp(Q["lg_x"] - gap/(R*T))
    population = excitation/(1+excitation)
    log_partition = np.log1p(excitation)
    return (g - 2*R*T*log_partition, gt - 2*(R*log_partition + gap/T*population),
            gp + 2*Q["v_x"]*population)


def _properties(T, P, g, gt, gp, gtt, gtp, gpp):
    V = gp * 1e-6 / M
    cp = -T*gtt / M
    cv = cp + T/M * gtp**2/gpp
    kt = -gp/gpp
    ks = kt*cp/cv
    G, S = g/M, -gt/M
    H, pv = G + T*S, P*1e6*V
    return dict(G=G, S=S, H=H, U=H-pv, A=G-pv, V=V, rho=1/V,
                Cp=cp, Cv=cv, Kt=kt, Ks=ks, alpha=gtp/gp, vel=np.sqrt(1e6*ks*V))


def _transport(T, P):
    shape, dp = _data["shape"], P-0.1
    midpoint = (shape["T50_K"] + shape["slope_K_per_MPa"]*dp
                + shape["curvature_K_per_100_MPa_squared"]*(dp/100.)**2)
    f = expit((midpoint-T)*220/(T*shape["width_K"]))
    result = {"f": f}
    for key, c in _data["transport"].items():
        barrier = (1000*(c["high_E_1000K"] + c["increment_E_1000K"]*f)
                   + dp*c["pressure_E_K_per_MPa"])
        sign = -1 if key == "D" else 1
        prefactor = T/273.15 if key == "eta" else 1
        result[key] = prefactor*np.exp(c["intercept"] + sign*barrier/T)
    return result


@np.errstate(all="ignore")
def compute_batch(T, P, branch="stable", *, x=None):
    T, P = np.broadcast_arrays(np.asarray(T, float), np.asarray(P, float))
    T = np.where(T > 0, T, np.nan)
    # Differentiate the analytic first derivatives without subtractive cancellation.
    step = 1e-20
    g, gt, gp = _high(T + 1j*step, P)
    _, _, gp_p = _high(T, P + 1j*step)
    high = (g.real, gt.real, gp.real, gt.imag/step, gp.imag/step, gp_p.imag/step)
    low = _low(T, P)
    delta, dt, dp = (l-h for l, h in zip(low[:3], high[:3]))
    W = Q["w0"] + P*Q["w1"]
    if x is None:
        x, metastable = select_branch((delta/(RM*T)).ravel(), (W/(RM*T)).ravel(), branch)
        x = x.reshape(T.shape)
    xx = x*(1-x)
    entropy_term = xlogy(x, x) + xlogy(1-x, 1-x)
    curvature = RM*T/xx - 2*W
    # A response at a spinodal is singular. First-derivative properties remain defined.
    inv = np.where(abs(curvature) < 1e-9, np.nan, 1/curvature)
    ft = dt + RM*np.log(np.clip(x, 1e-300, 1)/np.clip(1-x, 1e-300, 1))
    fp = dp + Q["w1"]*(1-2*x)
    avg = [(1-x)*h + x*l for h, l in zip(high, low)]
    result = _properties(T, P, avg[0]+RM*T*entropy_term+W*xx, avg[1]+RM*entropy_term,
                         avg[2]+Q["w1"]*xx, avg[3]-ft*ft*inv,
                         avg[4]-ft*fp*inv, avg[5]-fp*fp*inv)
    result.update(x=x, **_transport(T, P))
    for suffix, state in (("_A", high), ("_B", low)):
        result.update({k+suffix: v for k, v in _properties(T, P, *state).items()})
    valid = np.isfinite(high[0]) & np.isfinite(low[0])
    result = {k: np.where(valid, v, np.nan) for k, v in result.items()}
    if branch != "stable":
        result["metastable"] = metastable.reshape(T.shape) & valid
        return mask_mixture(result, result["x"])
    return result
