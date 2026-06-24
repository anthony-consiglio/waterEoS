"""
Compute and display the liquid-liquid phase diagram from each two-state model.
"""
from duska_eos import find_LLCP, compute_phase_diagram

# Find the liquid-liquid critical point
llcp = find_LLCP()
print(f"Duska (2020) LLCP: T = {llcp['T_K']:.2f} K, P = {llcp['p_MPa']:.2f} MPa")

# Compute full phase diagram (spinodal + binodal)
result = compute_phase_diagram()
print(f"Spinodal: {len(result['spinodal']['T_K'])} points")
print(f"Binodal:  {len(result['binodal']['T_K'])} points")

# Same for Caupin
from caupin_eos import find_LLCP as find_LLCP_c
llcp = find_LLCP_c()
print(f"\nCaupin (2019) LLCP: T = {llcp['T_K']:.2f} K, P = {llcp['p_MPa']:.2f} MPa")

# Same for Holten
from holten_eos import find_LLCP as find_LLCP_h
llcp = find_LLCP_h()
print(f"Holten (2014) LLCP: T = {llcp['T_K']:.2f} K, P = {llcp['p_MPa']:.2f} MPa")
