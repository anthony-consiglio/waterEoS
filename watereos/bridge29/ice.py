"""IAPWS-06 ice Ih reference and molar Gibbs derivatives for Bridge29."""
import numpy as np

M = 0.018015268          # kg/mol
TT = 273.16              # K
PT = 611.657e-6          # MPa
P0 = 101325e-6 / PT
S0 = -0.332733756492168e4 * 1e-3      # kJ/(kg K)
GOK = [-0.632020233335886e6, 0.655022213658955, -0.189369929326131e-7, 0.339746123271053e-14,
       -0.556464869058991e-21]
R2K = [complex(-0.725974574329220e2, -0.781008427112870e2) * 1e-3,
       complex(-0.557107698030123e-4, 0.464578634580806e-4) * 1e-3,
       complex(0.234801409215913e-10, -0.285651142904972e-10) * 1e-3]
T1 = complex(0.368017112855051e-1, 0.510878114959572e-1)
T2 = complex(0.337315741065416, 0.335449415919309)
R1 = complex(0.447050716285388e2, 0.656876847463481e2) * 1e-3


def ice(T, P):
    T, P = np.broadcast_arrays(np.asarray(T, float), np.asarray(P, float))
    Tr = T.astype(complex) / TT
    dp = P / PT - P0
    go = sum(GOK[k] * 1e-3 * dp ** k for k in range(5))
    gop = sum(GOK[k] * 1e-3 * k / PT * dp ** (k - 1) for k in range(1, 5))
    r2 = sum(R2K[k] * dp ** k for k in range(3))
    r2p = sum(R2K[k] * k / PT * dp ** (k - 1) for k in range(1, 3))

    def L(t):
        return (t - Tr) * np.log(t - Tr) + (t + Tr) * np.log(t + Tr) - 2 * t * np.log(t) - Tr ** 2 / t

    def Lt(t):
        return -np.log(t - Tr) + np.log(t + Tr) - 2 * Tr / t

    def Ltt(t):
        return 1 / (t - Tr) + 1 / (t + Tr) - 2 / t

    c = R1 * L(T1) + r2 * L(T2)
    ct = R1 * Lt(T1) + r2 * Lt(T2)
    ctt = R1 * Ltt(T1) + r2 * Ltt(T2)
    cp_ = r2p * L(T2)
    ctp = r2p * Lt(T2)
    g = go - S0 * TT * Tr.real + TT * c.real          # kJ/kg
    gt = -S0 + ct.real                                  # kJ/(kg K)
    gp = gop + TT * cp_.real                            # kJ/(kg MPa) = 1e-3 m3/kg
    gtt = ctt.real / TT
    gtp = ctp.real
    gopp = sum(GOK[k] * 1e-3 * k * (k-1) / PT**2 * dp**(k-2) for k in range(2, 5))
    gpp = gopp + TT * (2 * R2K[2] / PT**2 * L(T2)).real
    return tuple(v * 1e3 * M for v in (g, gt, gp, gtt, gtp, gpp))
