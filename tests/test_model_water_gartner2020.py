"""
Gartner et al. (2020), SCAN-DNN model water: checks against the paper.

- Liquid-liquid critical point (p. 3, Table 1): Tc = 224 K, Pc = 2687 bar,
  rho_c = 1072 +- 16 kg/m3.
- Fig. 2A (read at 400 dpi): the 3000-bar isobar jumps at about 214 K from
  about 1134 to 1033 kg/m3; the 1-bar isobar is near 963 kg/m3 at 240 and
  250 K; the 3000-bar isobar near 1143 kg/m3 at 250 K.
- Fig. 2B: internal energy near -1545 kJ/mol at 330 K and about
  -1556 kJ/mol at 240 K, low pressure (configurational plus kinetic, with
  the DFT zero of energy carried by c00).

The text (p. 3) also quotes coexisting densities of 1102 and 1047 kg/m3 at
T/Tc = 0.965.  Table 1 does not give these: at T/Tc = 0.965 it gives 1129 and
1038 kg/m3, consistent with the paper's own Fig. 2A.  That check is kept as
a strict xfail.
"""

import numpy as np
import pytest

from model_water_eos import gartner2020 as g


def test_critical_point():
    c = g.critical_point()
    assert c['T_K'] == 224.0
    assert c['p_MPa'] == pytest.approx(268.7)
    assert c['rho'] == pytest.approx(1072.0, abs=16.0)
    assert c['rho'] == pytest.approx(1073.2, abs=0.1)


def _coexistence(T):
    dT = T / g.Tc - 1.0
    dP = -dT / (g.A + g.B * dT)          # G_BA = 0
    P = g.Pc * (1.0 + dP) / 1e6
    h = g.compute_batch([T], [P], branch='hdl')
    lo = g.compute_batch([T], [P], branch='ldl')
    return P, h, lo


def test_jump_on_3000_bar_isobar_fig2a():
    P = 300.0
    dP = P / (g.Pc / 1e6) - 1.0
    T = g.Tc * (1.0 - g.A * dP / (1.0 + g.B * dP))
    assert T == pytest.approx(214.0, abs=1.0)
    h = g.compute_batch([T], [P], branch='hdl')
    lo = g.compute_batch([T], [P], branch='ldl')
    np.testing.assert_allclose(h['G'], lo['G'], rtol=0, atol=1e-6)
    assert h['rho'][0] == pytest.approx(1134.0, abs=6.0)
    assert lo['rho'][0] == pytest.approx(1033.0, abs=6.0)


@pytest.mark.parametrize('T, P, rho_fig', [
    (240.0, 0.1, 963.0), (250.0, 0.1, 963.0), (250.0, 300.0, 1143.0),
])
def test_isobars_fig2a(T, P, rho_fig):
    assert g.compute_batch([T], [P])['rho'][0] == pytest.approx(rho_fig, abs=5.0)


@pytest.mark.parametrize('T, U_fig', [(330.0, -1545.0), (240.0, -1556.0)])
def test_internal_energy_fig2b(T, U_fig):
    U = g.compute_batch([T], [0.1])['U'][0] * g.M / 1e3       # kJ/mol
    assert U == pytest.approx(U_fig, abs=4.0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    'Text (p. 3) quotes 1102 / 1047 kg/m3 at T/Tc = 0.965; Table 1 and '
    'Fig. 2A give 1129 / 1038 kg/m3'))
def test_coexisting_densities_quoted_in_text():
    _, h, lo = _coexistence(0.965 * g.Tc)
    assert h['rho'][0] == pytest.approx(1102.0, abs=5.0)
    assert lo['rho'][0] == pytest.approx(1047.0, abs=5.0)
