"""
T-V phase diagram example.

Reproduces figures from Powell-Palm (2022), RSC Advances:
  1. A-V convex hull at a sample temperature (Fig. 1a/b)
  2. 2D T-V phase diagram (Fig. 1c)
  3. T-P projection (standard phase diagram)
  4. Isochoric paths (Fig. 2)
  5. 3D P-T-V phase diagram (a la Verwiebe 1939)

Plots are saved to ../figures/ and also displayed interactively.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from watereos.tv_phase_diagram import (
    compute_tv_phase_diagram,
    compute_isochore,
    plot_tv_phase_diagram,
    plot_av_hull,
    plot_tp_phase_diagram,
    plot_isochore,
    plot_ptv_phase_diagram,
    plot_ptv_phase_diagram_plotly,
    plot_tv_phase_diagram_plotly,
    plot_tp_phase_diagram_plotly,
    plot_pv_phase_diagram_plotly,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DPI = 150


def savefig(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved {path}")


# ---- Compute the T-V diagram (dT=0.5 for smooth boundaries) ----
print("Computing T-V phase diagram...")
diagram = compute_tv_phase_diagram(T_min=190.0, T_max=300.0, dT=0.5,
                                   verbose=True)

# ---- 1. A-V convex hull at a sample temperature ----
print("\n1/5  A-V convex hull at T = 253 K")
fig1, ax1 = plt.subplots(figsize=(10, 6))
plot_av_hull(253.0, ax=ax1)
fig1.tight_layout()
savefig(fig1, "av_hull_253K.png")

# ---- 2. 2D T-V phase diagram ----
print("2/5  2D T-V phase diagram")
fig2, ax2 = plt.subplots(figsize=(10, 10))
plot_tv_phase_diagram(diagram, ax=ax2)
ax2.set_xlim(7e-4, 1.1e-3)
ax2.set_ylim(190, 300)
fig2.tight_layout()
savefig(fig2, "tv_phase_diagram.png")

# ---- 3. T-P projection ----
print("3/5  T-P phase diagram")
fig3, ax3 = plt.subplots(figsize=(9, 6))
plot_tp_phase_diagram(diagram, ax=ax3)
ax3.set_xlim(190, 300)
ax3.set_ylim(5e-4, 3000)
fig3.tight_layout()
savefig(fig3, "tp_phase_diagram.png")

# ---- 4. Isochoric paths ----
print("4/5  Isochoric paths")
V_values = [10.0e-4, 9.75e-4, 9.50e-4, 9.25e-4]  # m^3/kg
isochores = [compute_isochore(diagram, V) for V in V_values]
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
plot_isochore(isochores, diagram=diagram, axes=axes4)
fig4.tight_layout()
savefig(fig4, "isochoric_paths.png")

# ---- 5. 3D P-T-V phase diagram (matplotlib) ----
print("5/6  3D P-T-V phase diagram (matplotlib)")
fig5 = plot_ptv_phase_diagram(diagram, V_min=7e-4, V_max=1.1e-3, P_max=1000.0,
                              T_stride=1, n_pts_per_phase=50)
fig5.tight_layout()
savefig(fig5, "ptv_phase_diagram_3d.png")

# ---- 6. 3D P-T-V phase diagram (Plotly, interactive) ----
print("6/9  3D P-T-V phase diagram (Plotly)")
fig6 = plot_ptv_phase_diagram_plotly(diagram)
fig6.write_html(str(FIGURES_DIR / "ptv_phase_diagram_3d.html"))
print(f"  Saved {FIGURES_DIR / 'ptv_phase_diagram_3d.html'}")

# ---- 7. 2D T-V phase diagram (Plotly) ----
print("7/9  2D T-V phase diagram (Plotly)")
fig7 = plot_tv_phase_diagram_plotly(diagram, V_min=7e-4, V_max=1.1e-3,
                                    T_min=190, T_max=300)
fig7.write_html(str(FIGURES_DIR / "tv_phase_diagram.html"))
print(f"  Saved {FIGURES_DIR / 'tv_phase_diagram.html'}")

# ---- 8. 2D T-P phase diagram (Plotly) ----
print("8/9  2D T-P phase diagram (Plotly)")
fig8 = plot_tp_phase_diagram_plotly(diagram, T_min=190, T_max=300,
                                    P_min=1e-4, P_max=1000)
fig8.write_html(str(FIGURES_DIR / "tp_phase_diagram.html"))
print(f"  Saved {FIGURES_DIR / 'tp_phase_diagram.html'}")

# ---- 9. 2D P-V phase diagram (Plotly) ----
print("9/9  2D P-V phase diagram (Plotly)")
fig9 = plot_pv_phase_diagram_plotly(diagram, V_min=7e-4, V_max=1.1e-3,
                                    P_min=1e-4, P_max=2500)
fig9.write_html(str(FIGURES_DIR / "pv_phase_diagram.html"))
print(f"  Saved {FIGURES_DIR / 'pv_phase_diagram.html'}")

fig7.show()
fig8.show()
fig9.show()

plt.show()
print("\nDone.")
