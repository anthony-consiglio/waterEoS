"""
Transport properties from Singh, Issenmann & Caupin (2017).

The singh2017 model predicts dynamic viscosity (eta), self-diffusion
coefficient (D), and rotational correlation time (tau_r) for supercooled
water.  It uses Holten (2014) as its thermodynamic backbone, so all
Holten thermodynamic properties are also available.
"""
import numpy as np
from watereos import compute

print("Singh (2017) transport properties at 0.1 MPa:\n")
print(f"  {'T (K)':>6s}  {'eta (mPa·s)':>12s}  {'D (m²/s)':>12s}  {'tau_r (ps)':>12s}")
print(f"  {'-'*48}")

for T in [240, 245, 250, 255, 260, 270, 280, 298]:
    out = compute(T_K=T, P_MPa=0.1, model='singh2017')
    eta = out.eta[0, 0] * 1e3     # Pa·s -> mPa·s
    D = out.D[0, 0]               # m²/s
    tau = out.tau_r[0, 0] * 1e12  # s -> ps
    print(f"  {T:6d}  {eta:12.4f}  {D:12.4e}  {tau:12.2f}")

# Pressure dependence at 250 K
print("\nPressure dependence at 250 K:\n")
print(f"  {'P (MPa)':>8s}  {'eta (mPa·s)':>12s}  {'D (m²/s)':>12s}")
print(f"  {'-'*36}")

for P in [0.1, 50, 100, 150, 200]:
    out = compute(T_K=250, P_MPa=P, model='singh2017')
    eta = out.eta[0, 0] * 1e3
    D = out.D[0, 0]
    print(f"  {P:8.1f}  {eta:12.4f}  {D:12.4e}")
