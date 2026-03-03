"""
Accessing per-state (HDL/LDL) properties from two-state models.

Two-state models treat water as a mixture of two local structures:
  - State A (HDL): high-density, disordered hydrogen-bond network
  - State B (LDL): low-density, tetrahedral ice-like network

The equilibrium fraction of LDL is called x (ranges 0 to 1).
Each thermodynamic property is available for the mixture (no suffix),
State A (_A suffix), and State B (_B suffix).
"""
from watereos import compute

# Evaluate at a supercooled state point where the two-state
# decomposition is significant
out = compute(T_K=230, P_MPa=0.1, model='duska2020')

x = out.x[0, 0]
print(f"Duska (2020) at 230 K, 0.1 MPa:")
print(f"  LDL fraction x = {x:.4f}")
print()

# Compare mixture vs. individual state densities
print(f"  {'':15s} {'Mixture':>10s} {'State A':>10s} {'State B':>10s}")
print(f"  {'':15s} {'(HDL+LDL)':>10s} {'(HDL)':>10s} {'(LDL)':>10s}")
print(f"  {'-'*47}")

for label, key in [
    ('Density',  'rho'),
    ('Volume',   'V'),
    ('Entropy',  'S'),
    ('Gibbs',    'G'),
    ('Cp',       'Cp'),
    ('Kt',       'Kt'),
]:
    mix = getattr(out, key)[0, 0]
    a = getattr(out, f'{key}_A')[0, 0]
    b = getattr(out, f'{key}_B')[0, 0]
    print(f"  {label:15s} {mix:>10.4g} {a:>10.4g} {b:>10.4g}")

# Show how x changes with temperature
print("\nLDL fraction vs. temperature at 0.1 MPa:")
for T in [200, 210, 220, 230, 240, 250, 260, 273]:
    out = compute(T_K=T, P_MPa=0.1, model='duska2020')
    print(f"  T = {T:3d} K  ->  x = {out.x[0,0]:.4f}")
