"""MODEL WATER, NOT REAL WATER. Classical and quantum q-TIP4P/F model water.

Eltareb, Lopez and Giovambattista, Sci. Rep. 12, 6004 (2022),
doi:10.1038/s41598-022-09525-x. The three variants use the middle (325 K
upper limit) columns of Tables S1-S3 (SI pp. 14-16), as used in the main
article. SI Sec. V gives a shared pressure fit interval of -50 to 350 MPa
and temperature intervals of 180-325 K (classical and quantum H2O) and
190-325 K (quantum D2O). All three liquid-liquid critical points are
extrapolations below the fitted temperatures.

SI Sec. V cites Gartner et al. (2020), whose Eqs. 2-5 give

    G = G_A + x G_BA + R T [x ln x + (1 - x) ln(1 - x)] + W x (1 - x)
    G_A = R Tc sum c_jk u^j v^k
    G_BA = R T lambda (u + a v + b u v)
    W = R Tc (2 + w0 v)

with u = (T - Tc)/Tc, v = (P - Pc)/Pc and x the low-density fraction.
The native Gibbs fields are in J/mol, differentiated in K and Pa. Batch
inputs are K and MPa and outputs use the package's mass-specific units.
Each fit keeps its own printed energy and entropy reference, with the
rotated c00 providing the energy offset and no c10 term.

The rendered tables have a systematic row-label mismatch. In all three,
the first background value (labeled c00) is used for c01, the remaining
values advance by one label, and the final value (labeled c33) is c00.
This preserves every printed value. It is an inferred correction: the
literal labels yield critical densities of roughly 7200-9000 kg/m3 on
the fit's mass basis. The rotated values reproduce the critical and
in-range densities. Tc, Pc, lambda, a, b and w0 use their printed labels.

The package H2O molar mass (18.015268 g/mol) is the effective conversion
for all three fits. For S3 the physical D2O mass is 20.027508 g/mol, but
substituting it in the EOS gives 1258.74 kg/m3 at the critical point,
against 1130 +/- 10 kg/m3 in the table. The common mass basis reproduces
the reported D2O density and the numerical scale of the caloric plots
when converting back to the table's native energy basis. This is an
inference from the data, not a documented convention in the supplement.
Conversion of those mass-specific results to a physical mole of D2O
scales the native caloric values by 20.027508/18.015268. For example the
0.1 MPa Cp peak is 106.9 J/(mol K) on that basis, while Fig. S7b shows
about 96 J/(mol K). Keeping one conversion for all Gibbs derivatives
preserves thermodynamic consistency.

Table S1-S3 critical points (K, MPa, computed kg/m3): classical H2O
(175, 203, 1028.41), quantum H2O (159, 167, 1018.14), quantum D2O
(177, 176, 1132.27). The reported densities are 1030, 1020 and 1130,
each +/- 10 kg/m3. At 300 K and 0.1 MPa, Figs. 1c, 1d and S4 give
approximately 1002, 995 and 1110 kg/m3, versus 1001.61, 995.04 and
1107.86 kg/m3. At 0.1 MPa the quantum H2O fit peaks in Cp at 211.87 K
versus 216 K in the main text (p. 8). SI Sec. II notes the sensitivity
of the peak to the enthalpy fit. The alternate 300 K and 350 K upper-limit
columns give 211.19 K and 208.93 K. The printed peak is an expected failure.
"""

from dataclasses import dataclass

from watereos.two_state_general import (
    M_H2O, R, compute_two_state, poly_field, times_T,
)

H2O_CLASSICAL = 'eltareb2022_qtip4pf_h2o_classical'
H2O_QUANTUM = 'eltareb2022_qtip4pf_h2o_quantum'
D2O_QUANTUM = 'eltareb2022_qtip4pf_d2o_quantum'


@dataclass(frozen=True)
class _Fit:
    Tc: float
    Pc: float  # Pa
    lam: float
    a: float
    b: float
    w0: float
    background: dict

    def terms(self, T, P_Pa):
        u = (T - self.Tc) / self.Tc
        v = (P_Pa - self.Pc) / self.Pc
        dT, dP = 1.0 / self.Tc, 1.0 / self.Pc
        GA = poly_field(self.background, u, v, dT, dP, R * self.Tc)
        inner = poly_field({(1, 0): 1.0, (0, 1): self.a, (1, 1): self.b},
                           u, v, dT, dP, R * self.lam)
        W = poly_field({(0, 0): 2.0, (0, 1): self.w0}, u, v, dT, dP, R * self.Tc)
        return GA, times_T(inner, T), W

    def compute_batch(self, T_K, P_MPa, branch='stable'):
        """Evaluate this q-TIP4P/F fit on flat arrays in K and MPa."""
        return compute_two_state(T_K, P_MPa, self.terms, M=M_H2O, branch=branch)


# Tables S1, S2 and S3, middle columns. Keys are the inferred (j, k)
# after rotating the row labels. Each coefficient stays at printed precision.
_CLASSICAL = _Fit(Tc=175.0, Pc=203e6, lam=1.470, a=0.417, b=-0.299, w0=0.206,
                  background={
                      (0, 0): -29.0,
                      (0, 1): 2.086,
                      (0, 2): -0.008,
                      (1, 1): 0.112,
                      (2, 0): -7.328,
                      (1, 2): -0.051,
                      (2, 1): 0.174,
                      (3, 0): 2.485,
                      (2, 2): -0.031,
                      (3, 1): -0.036,
                      (4, 0): -0.449,
                      (2, 3): 0.037,
                      (3, 2): 0.024,
                      (3, 3): -0.026,
                  })
_QUANTUM_H2O = _Fit(Tc=159.0, Pc=167e6, lam=1.058, a=0.542, b=-0.264, w0=0.074,
                    background={
                        (0, 0): 6.397,
                        (0, 1): 1.930,
                        (0, 2): -0.023,
                        (1, 1): -0.027,
                        (2, 0): -4.922,
                        (1, 2): 0.042,
                        (2, 1): 0.214,
                        (3, 0): 1.387,
                        (2, 2): -0.095,
                        (3, 1): -0.052,
                        (4, 0): -0.236,
                        (2, 3): 0.001,
                        (3, 2): 0.034,
                        (3, 3): 0.005,
                    })
_QUANTUM_D2O = _Fit(Tc=177.0, Pc=176e6, lam=1.084, a=0.430, b=-0.277, w0=0.103,
                    background={
                        (0, 0): -3.576,
                        (0, 1): 1.644,
                        (0, 2): -0.020,
                        (1, 1): -0.015,
                        (2, 0): -4.416,
                        (1, 2): 0.031,
                        (2, 1): 0.219,
                        (3, 0): 0.947,
                        (2, 2): -0.118,
                        (3, 1): -0.044,
                        (4, 0): -0.107,
                        (2, 3): 0.017,
                        (3, 2): 0.062,
                        (3, 3): -0.009,
                    })

MODELS = {
    H2O_CLASSICAL: _CLASSICAL.compute_batch,
    H2O_QUANTUM: _QUANTUM_H2O.compute_batch,
    D2O_QUANTUM: _QUANTUM_D2O.compute_batch,
}
BRANCH_MODELS = tuple(MODELS)
