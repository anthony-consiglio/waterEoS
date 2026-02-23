"""
Basic usage of waterEoS: single point, grid mode, and scatter mode.
"""
import numpy as np
from watereos import getProp, list_models

# --- Available models ---
print("Available models:", list_models())

# --- Single point ---
PT = np.array([[0.1], [273.15]], dtype=object)
out = getProp(PT, 'duska2020')
print(f"\nDuska (2020) at 0.1 MPa, 273.15 K:")
print(f"  rho   = {out.rho[0,0]:.2f} kg/m³")
print(f"  Cp    = {out.Cp[0,0]:.1f} J/(kg·K)")
print(f"  vel   = {out.vel[0,0]:.1f} m/s")
print(f"  x     = {out.x[0,0]:.4f}")

# --- Grid mode ---
P = np.arange(0.1, 200, 50)   # 4 pressures (MPa)
T = np.arange(250, 310, 10)   # 6 temperatures (K)
PT = np.array([P, T], dtype=object)
out = getProp(PT, 'holten2014')
print(f"\nHolten (2014) grid: {out.rho.shape[0]} pressures x {out.rho.shape[1]} temperatures")
print(f"  rho range: {out.rho.min():.1f} - {out.rho.max():.1f} kg/m³")

# --- Scatter mode ---
PT = np.empty(3, dtype=object)
PT[0] = (0.1, 273.15)
PT[1] = (0.1, 298.15)
PT[2] = (100.0, 250.0)
out = getProp(PT, 'caupin2019')
print(f"\nCaupin (2019) scatter mode:")
for i in range(3):
    print(f"  Point {i}: rho = {out.rho[i]:.2f} kg/m³, Cp = {out.Cp[i]:.1f} J/(kg·K)")
