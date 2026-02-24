"""
Validation tests for waterEoS package.

Reference values from IAPWS-95 at 273.15 K, 0.1 MPa:
  rho = 999.84 kg/m³, Cp = 4218 J/(kg·K), vel = 1403 m/s
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def PT_single():
    """Single point: 0.1 MPa, 273.15 K."""
    return np.array([[0.1], [273.15]], dtype=object)

@pytest.fixture
def PT_grid():
    """Small 3x4 grid."""
    P = np.array([0.1, 50.0, 100.0])
    T = np.array([273.15, 283.15, 293.15, 303.15])
    return np.array([P, T], dtype=object)

@pytest.fixture
def PT_scatter():
    """Three scatter points."""
    PT = np.empty(3, dtype=object)
    PT[0] = (0.1, 273.15)
    PT[1] = (0.1, 298.15)
    PT[2] = (100.0, 280.0)
    return PT

# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

def test_list_models():
    from watereos import list_models
    models = list_models()
    assert isinstance(models, list)
    assert len(models) == 7
    assert 'duska2020' in models
    assert 'caupin2019' in models
    assert 'holten2014' in models
    assert 'grenke2025' in models
    assert 'singh2017' in models
    assert 'water1' in models
    assert 'IAPWS95' in models

def test_invalid_model():
    from watereos import getProp
    PT = np.array([[0.1], [300.0]], dtype=object)
    with pytest.raises(ValueError, match="Unknown model"):
        getProp(PT, 'nonexistent')

def test_case_insensitive():
    from watereos import getProp
    PT = np.array([[0.1], [300.0]], dtype=object)
    out1 = getProp(PT, 'Duska2020')
    out2 = getProp(PT, 'DUSKA2020')
    np.testing.assert_allclose(out1.rho, out2.rho)

# ---------------------------------------------------------------------------
# Model accuracy: rho at 273.15 K, 0.1 MPa (IAPWS-95: 999.84 kg/m³)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, tol", [
    ('holten2014', 0.1),
    ('caupin2019', 0.1),
    ('duska2020', 0.1),
    ('grenke2025', 0.1),
])
def test_density_at_ambient(PT_single, model, tol):
    from watereos import getProp
    out = getProp(PT_single, model)
    assert abs(out.rho.flat[0] - 999.84) < tol, (
        f"{model}: rho={out.rho.flat[0]:.4f}, expected ~999.84"
    )

# ---------------------------------------------------------------------------
# Grid mode: shape checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ['holten2014', 'caupin2019', 'duska2020'])
def test_grid_shape(PT_grid, model):
    from watereos import getProp
    out = getProp(PT_grid, model)
    assert out.rho.shape == (3, 4)
    assert out.Cp.shape == (3, 4)
    assert out.x.shape == (3, 4)

# ---------------------------------------------------------------------------
# Scatter mode: shape checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ['holten2014', 'caupin2019', 'duska2020'])
def test_scatter_shape(PT_scatter, model):
    from watereos import getProp
    out = getProp(PT_scatter, model)
    assert out.rho.shape == (3,)
    assert out.Cp.shape == (3,)
    assert out.x.shape == (3,)

# ---------------------------------------------------------------------------
# Grid/scatter mode: Grenke (no x attribute)
# ---------------------------------------------------------------------------

def test_grenke_grid_shape(PT_grid):
    from watereos import getProp
    out = getProp(PT_grid, 'grenke2025')
    assert out.rho.shape == (3, 4)
    assert out.Cp.shape == (3, 4)

def test_grenke_scatter_shape(PT_scatter):
    from watereos import getProp
    out = getProp(PT_scatter, 'grenke2025')
    assert out.rho.shape == (3,)
    assert out.Cp.shape == (3,)

def test_grenke_output_attributes(PT_single):
    from watereos import getProp
    out = getProp(PT_single, 'grenke2025')
    for attr in ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
                 'Kt', 'Ks', 'Kp', 'alpha', 'vel']:
        assert hasattr(out, attr), f"Missing attribute: {attr}"
    # Grenke has no x or _A/_B properties
    assert not hasattr(out, 'x')
    assert not hasattr(out, 'rho_A')

# ---------------------------------------------------------------------------
# Output attributes: check all expected properties exist
# ---------------------------------------------------------------------------

def test_output_attributes(PT_single):
    from watereos import getProp
    out = getProp(PT_single, 'duska2020')
    # Mixture properties
    for attr in ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
                 'Kt', 'Ks', 'Kp', 'alpha', 'vel', 'x']:
        assert hasattr(out, attr), f"Missing attribute: {attr}"
    # State A and B properties
    for suffix in ('_A', '_B'):
        for attr in ['rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
                     'Kt', 'Ks', 'Kp', 'alpha', 'vel']:
            assert hasattr(out, attr + suffix), f"Missing attribute: {attr}{suffix}"

# ---------------------------------------------------------------------------
# Phase diagram functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module, T_exp, p_exp, T_tol, p_tol", [
    ('duska_eos', 220.9, 54.2, 0.5, 0.5),
    ('caupin_eos', 218.1, 72.0, 0.5, 0.5),
    ('holten_eos', 228.2, 0.0, 0.5, 0.5),
])
def test_find_LLCP(module, T_exp, p_exp, T_tol, p_tol):
    mod = __import__(module)
    llcp = mod.find_LLCP()
    T_c = llcp['T_K']
    p_c = llcp['p_MPa']
    assert abs(T_c - T_exp) < T_tol, f"{module}: T_LLCP={T_c:.2f}, expected ~{T_exp}"
    assert abs(p_c - p_exp) < p_tol, f"{module}: p_LLCP={p_c:.2f}, expected ~{p_exp}"

def test_compute_phase_diagram_runs():
    from duska_eos import compute_phase_diagram
    result = compute_phase_diagram()
    assert 'spinodal' in result
    assert 'binodal' in result
    assert len(result['spinodal']['T_K']) > 10

# ---------------------------------------------------------------------------
# SeaFreeze pass-through
# ---------------------------------------------------------------------------

def test_seafreeze_IAPWS95(PT_single):
    from watereos import getProp
    out = getProp(PT_single, 'IAPWS95')
    assert abs(out.rho.flat[0] - 999.84) < 0.02

def test_seafreeze_water1(PT_single):
    from watereos import getProp
    out = getProp(PT_single, 'water1')
    assert out.rho.flat[0] > 990  # reasonable density

# ---------------------------------------------------------------------------
# Singh (2017) transport model tests
# ---------------------------------------------------------------------------

def test_singh_viscosity_at_0C(PT_single):
    """eta at 273.15 K, 0.1 MPa should be ~1.79 mPa*s (IAPWS)."""
    from watereos import getProp
    out = getProp(PT_single, 'singh2017')
    eta_mPas = out.eta.flat[0] * 1e3
    assert abs(eta_mPas - 1.793) < 0.05, f"eta={eta_mPas:.4f} mPa*s, expected ~1.793"

def test_singh_viscosity_at_25C():
    """eta at 298.15 K, 0.1 MPa should be ~0.89 mPa*s (IAPWS)."""
    from watereos import getProp
    PT = np.array([[0.1], [298.15]], dtype=object)
    out = getProp(PT, 'singh2017')
    eta_mPas = out.eta.flat[0] * 1e3
    assert abs(eta_mPas - 0.890) < 0.01, f"eta={eta_mPas:.4f} mPa*s, expected ~0.890"

def test_singh_grid_shape(PT_grid):
    from watereos import getProp
    out = getProp(PT_grid, 'singh2017')
    assert out.eta.shape == (3, 4)
    assert out.D.shape == (3, 4)
    assert out.tau_r.shape == (3, 4)
    assert out.rho.shape == (3, 4)  # thermodynamic pass-through

def test_singh_scatter_shape(PT_scatter):
    from watereos import getProp
    out = getProp(PT_scatter, 'singh2017')
    assert out.eta.shape == (3,)
    assert out.D.shape == (3,)
    assert out.tau_r.shape == (3,)

def test_singh_output_attributes(PT_single):
    from watereos import getProp
    out = getProp(PT_single, 'singh2017')
    # Transport properties
    for attr in ['eta', 'D', 'tau_r', 'f']:
        assert hasattr(out, attr), f"Missing attribute: {attr}"
    # Thermodynamic pass-through from Holten backbone
    for attr in ['rho', 'V', 'S', 'Cp', 'Cv', 'Kt', 'Ks', 'alpha', 'vel', 'x']:
        assert hasattr(out, attr), f"Missing attribute: {attr}"

def test_singh_pressure_anomaly():
    """At 250 K, viscosity should decrease with increasing pressure."""
    from watereos import getProp
    PT_lo = np.array([[0.1], [250.0]], dtype=object)
    PT_hi = np.array([[100.0], [250.0]], dtype=object)
    eta_lo = getProp(PT_lo, 'singh2017').eta.flat[0]
    eta_hi = getProp(PT_hi, 'singh2017').eta.flat[0]
    assert eta_hi < eta_lo, (
        f"Anomaly not observed: eta(0.1 MPa)={eta_lo:.6e}, eta(100 MPa)={eta_hi:.6e}"
    )
