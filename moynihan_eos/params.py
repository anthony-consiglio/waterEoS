"""
Moynihan (1996) two-species / nonideal-solution model: parameters.

Reference: C. T. Moynihan, "Two species/nonideal solution model for
amorphous/amorphous phase transitions", MRS Proceedings 455, 411-425 (1996),
doi:10.1557/PROC-455-411.  Page numbers refer to the proceedings.

Species (p. 412, Eq. 1): A is the low-density, LDA-like species and B the
high-density, HDA-like species, written so that A -> B is endothermic.  The
paper's X is the mole fraction of B.  The package returns x = 1 - X, the
fraction of the low-density species, like its other two-state models.
"""

R = 8.314462618           # J/(mol K)
M_H2O = 0.018015268       # kg/mol; the species formula is H2O (p. 417)
ATM = 101325.0            # Pa

# Standard state: pure species at P0 = 1 atm (p. 412, after Eq. 3)
P0 = ATM                  # Pa

# Fitted constants for the species formula H2O
DV0 = -4.5e-6             # m3/mol, Eq. 22a (p. 417): Delta V of A -> B
W = 3610.0                # J/mol, Eq. 22c (p. 419): regular-solution
                          # interaction, W = 2 R Tc
DS0 = 0.98 * R            # J/(mol K), Eq. 25 (p. 421): 0.98 R = 8.1
DH0 = 1940.0              # J/mol, Eq. 26 (p. 421), from Eq. 23 with
                          # Delta S0 / R = 0.98

# Background used only for the T-v phase diagram (p. 421, Fig. 7):
# expansivity and compressibility of both species, independent of T and P.
ALPHA_BKGR = 0.46e-3              # 1/K
KAPPA_BKGR = 1.62e-5 / ATM        # 1/Pa (1.62e-5 atm^-1)
# Critical specific volume read off Fig. 7 and stated in the text (p. 421):
# "The Tc and vc values - 217 K and 1.01 cm3/g - at the critical point".
V_C = 1.01e-3                     # m3/kg

# Values printed in the paper, used as checks
TC_PRINTED = 217.0                # K, Eq. 22c (217 +- 10 K)
PC_PRINTED = 380.0 * ATM          # Pa, Eq. 27 (380 atm)
T_AA = 135.0                      # K, Mishima's transformation temperature
P_AA_PRINTED = 1850.0 * ATM       # Pa, Eq. 20: coexistence at 135 K
P_SPIN_LDA_TO_HDA = 3200.0 * ATM  # Pa, p. 417: LDA spinodal at 135 K
P_SPIN_HDA_TO_LDA = 500.0 * ATM   # Pa, p. 417: HDA spinodal at 135 K
LNK_135 = -0.75                   # Eq. 22b

# Ranges the constants were fitted to (pp. 417-421): the transformation at
# 135 K up to 1850 atm, and Cp, alpha, kappa of the supercooled liquid
# between 236 and 277 K at 0.1-100 MPa.
T_MIN, T_MAX = 135.0, 277.0       # K
P_MIN, P_MAX = 0.1, 330.0         # MPa
