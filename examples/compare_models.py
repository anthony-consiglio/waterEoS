"""
Compare density predictions from all five models at ambient conditions.
"""
import numpy as np
from watereos import getProp, list_models

PT = np.array([[0.1], [273.15]], dtype=object)

print("Density at 0.1 MPa, 273.15 K:")
print(f"{'Model':<15} {'rho (kg/m³)':>12}")
print("-" * 28)

for model in list_models():
    out = getProp(PT, model)
    rho = out.rho.flat[0]
    print(f"{model:<15} {rho:>12.4f}")
