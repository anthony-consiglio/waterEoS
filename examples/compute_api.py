"""
Using the compute() API — the simplest way to call waterEoS.

compute() takes plain scalars or lists instead of the SeaFreeze-style
numpy array format.  It returns the same ThermodynamicStates object.
"""
from watereos import compute

# --- Single point ---
out = compute(T_K=273.15, P_MPa=0.1, model='duska2020')
print("Duska (2020) at 273.15 K, 0.1 MPa:")
print(f"  Density:      {out.rho[0,0]:.4f} kg/m³")
print(f"  Heat capacity:{out.Cp[0,0]:.1f} J/(kg·K)")
print(f"  Gibbs energy: {out.G[0,0]:.2f} J/kg")

# --- Multiple temperatures at one pressure ---
out = compute(T_K=[250, 260, 270, 280], P_MPa=0.1, model='holten2014')
print("\nHolten (2014) at 0.1 MPa, varying T:")
for i, T in enumerate([250, 260, 270, 280]):
    print(f"  T = {T} K  ->  rho = {out.rho[0, i]:.2f} kg/m³")

# --- Full pressure x temperature grid ---
out = compute(T_K=[250, 275, 300], P_MPa=[0.1, 50, 100], model='caupin2019')
print(f"\nCaupin (2019) grid shape: {out.rho.shape}")  # (3 pressures, 3 temperatures)
print(f"  rho at 50 MPa, 275 K = {out.rho[1, 1]:.2f} kg/m³")
