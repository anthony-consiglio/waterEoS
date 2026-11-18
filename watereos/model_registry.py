"""
Model registry — single source of truth for waterEoS model metadata.
"""

from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """Metadata for a single equation-of-state model.

    Attributes
    ----------
    display_name : str
        Human-readable name shown in the UI (e.g. ``'Holten (2014)'``).
    model_key : str
        Key passed to ``watereos.getProp(PT, model_key)``.
    is_two_state : bool
        Whether the model uses a two-state (HDL/LDL) mixing framework.
        Two-state models can expose per-state outputs (``_A`` / ``_B`` suffixes)
        and a mixing fraction ``x`` (LDL fraction), depending on the fit.
    has_phase_diagram : bool
        Whether the visualization layer supports a phase diagram for this model.
    has_transport : bool
        Whether the model provides transport properties (``eta``, ``D``, ``tau_r``).
    T_min, T_max : float
        Suggested temperature range (K) where results are physically meaningful.
    P_min, P_max : float
        Suggested pressure range (MPa).
    properties : list[str]
        Output property keys this model supports (e.g. ``['rho', 'V', ...]``).
    water_model : str
        Empty for real water.  For an equation fitted to simulations of a
        model water, the name of the water potential (e.g. ``'TIP4P/2005'``);
        such a model describes that model water, not real water.
    """
    display_name: str
    model_key: str
    is_two_state: bool
    has_phase_diagram: bool
    has_transport: bool
    T_min: float
    T_max: float
    P_min: float
    P_max: float
    properties: list = field(default_factory=list)
    water_model: str = ''

    @property
    def is_model_water(self):
        """True for equations fitted to a model water, not real water."""
        return bool(self.water_model)


# --- Property definitions ---------------------------------------------------

_MIX_KEYS = [
    'rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
    'Kt', 'Ks', 'Kp', 'alpha', 'vel', 'x',
]

_STATE_KEYS = [
    'rho', 'V', 'S', 'G', 'H', 'U', 'A', 'Cp', 'Cv',
    'Kt', 'Ks', 'Kp', 'alpha', 'vel',
]

_TRANSPORT_KEYS = ['eta', 'D', 'tau_r', 'f']

TWO_STATE_PROPS = _MIX_KEYS + [f'{k}_A' for k in _STATE_KEYS] + [f'{k}_B' for k in _STATE_KEYS]
EMPIRICAL_PROPS = [k for k in _MIX_KEYS if k != 'x']
TRANSPORT_PROPS = _TRANSPORT_KEYS + TWO_STATE_PROPS
_VOLUME_RESPONSE_PROPS = ['rho', 'V', 'Kt', 'alpha']
_YU_PROPS = _VOLUME_RESPONSE_PROPS + [
    f'{prop}_{state}' for state in ('A', 'B') for prop in _VOLUME_RESPONSE_PROPS
] + ['x', 'D', 'tau_r', 'f']
# Moynihan (1996): fraction, volume built on a constant background, and the
# excess (conformational) heat capacity; no Gibbs energy or total Cp
MOYNIHAN_PROPS = ['rho', 'V', 'Kt', 'alpha', 'x', 'Cp_ex', 'V_alpha_ex', 'V_kappa_ex']

PROPERTY_LABELS = {
    'rho': 'Density',
    'V': 'Specific Volume',
    'S': 'Entropy',
    'G': 'Gibbs Energy',
    'H': 'Enthalpy',
    'U': 'Internal Energy',
    'A': 'Helmholtz Energy',
    'Cp': 'Isobaric Heat Capacity',
    'Cv': 'Isochoric Heat Capacity',
    'Kt': 'Isothermal Bulk Modulus',
    'Ks': 'Adiabatic Bulk Modulus',
    'Kp': "dKt/dP",
    'alpha': 'Thermal Expansivity',
    'vel': 'Speed of Sound',
    'x': 'LDL Fraction (x)',
    'eta': 'Dynamic Viscosity',
    'D': 'Self-Diffusion Coefficient',
    'tau_r': 'Rotational Correlation Time',
    'f': 'LDS Fraction (f)',
    'Cp_ex': 'Excess Isobaric Heat Capacity',
    'V_alpha_ex': 'Excess Volume-Weighted Expansivity',
    'V_kappa_ex': 'Excess Volume-Weighted Compressibility',
}

PROPERTY_UNITS = {
    'rho': 'kg/m³',
    'V': 'm³/kg',
    'S': 'J/(kg·K)',
    'G': 'J/kg',
    'H': 'J/kg',
    'U': 'J/kg',
    'A': 'J/kg',
    'Cp': 'J/(kg·K)',
    'Cv': 'J/(kg·K)',
    'Kt': 'MPa',
    'Ks': 'MPa',
    'Kp': 'dimensionless',
    'alpha': '1/K',
    'vel': 'm/s',
    'x': '',
    'eta': 'Pa·s',
    'D': 'm²/s',
    'tau_r': 's',
    'f': '',
    'Cp_ex': 'J/(kg·K)',
    'V_alpha_ex': 'm³/(kg·K)',
    'V_kappa_ex': 'm³/(kg·MPa)',
}

# Extend labels/units for _A and _B variants
for k in _STATE_KEYS:
    for suffix, state_name in [('_A', ' (State A)'), ('_B', ' (State B)')]:
        key = k + suffix
        PROPERTY_LABELS[key] = PROPERTY_LABELS[k] + state_name
        PROPERTY_UNITS[key] = PROPERTY_UNITS[k]


# --- Model registry ---------------------------------------------------------

MODEL_REGISTRY = {
    'water1': ModelInfo(
        display_name='SeaFreeze water1',
        model_key='water1',
        is_two_state=False,
        has_phase_diagram=False,
        has_transport=False,
        T_min=240, T_max=500,
        P_min=0.1, P_max=2300,
        properties=EMPIRICAL_PROPS,
    ),
    'IAPWS95': ModelInfo(
        display_name='IAPWS-95',
        model_key='IAPWS95',
        is_two_state=False,
        has_phase_diagram=False,
        has_transport=False,
        T_min=240, T_max=500,
        P_min=0.1, P_max=2300,
        properties=EMPIRICAL_PROPS,
    ),
    'holten2014': ModelInfo(
        display_name='Holten (2014)',
        model_key='holten2014',
        is_two_state=True,
        has_phase_diagram=True,
        has_transport=False,
        T_min=200, T_max=300,
        P_min=0.0, P_max=400,
        properties=TWO_STATE_PROPS,
    ),
    'caupin2019': ModelInfo(
        display_name='Caupin (2019)',
        model_key='caupin2019',
        is_two_state=True,
        has_phase_diagram=True,
        has_transport=False,
        T_min=200, T_max=300,
        P_min=-140, P_max=400,
        properties=TWO_STATE_PROPS,
    ),
    'bridge29': ModelInfo(
        display_name='Bridge29 (Consiglio)',
        model_key='bridge29', is_two_state=True, has_phase_diagram=True, has_transport=True,
        T_min=100, T_max=400, P_min=-200, P_max=400,
        properties=[k for k in TRANSPORT_PROPS if k.split('_')[0] != 'Kp'],
    ),
    'duska2020': ModelInfo(
        display_name='Duska (2020)',
        model_key='duska2020',
        is_two_state=True,
        has_phase_diagram=True,
        has_transport=False,
        T_min=200, T_max=370,
        # Formulated above the liquid spinodal, so it extends into the
        # stretched (negative-pressure) regime. Duska (2020) reports
        # accurate agreement with data to ~100 MPa, degrading gradually at
        # negative pressure; -150 MPa is a practical lower bound well above
        # the model's own spinodal at all T >= 200 K.
        P_min=-150, P_max=400,
        properties=TWO_STATE_PROPS,
    ),
    'caupin2019_kim': ModelInfo(
        display_name='Caupin (2019, with Kim)',
        model_key='caupin2019_kim',
        is_two_state=True,
        has_phase_diagram=True,
        has_transport=False,
        T_min=200, T_max=300,
        # Same fitted data window as the without-Kim fit (Caupin 2019:
        # below 300 K, -140 to 400 MPa). LLCP is at 219.47 K / 58.74 MPa.
        P_min=-140, P_max=400,
        properties=TWO_STATE_PROPS,
    ),
    'grenke2025': ModelInfo(
        display_name='Grenke (2025)',
        model_key='grenke2025',
        is_two_state=False,
        has_phase_diagram=False,
        has_transport=False,
        T_min=200, T_max=300,
        P_min=0.1, P_max=400,
        properties=EMPIRICAL_PROPS,
    ),
    'singh2017': ModelInfo(
        display_name='Singh (2017)',
        model_key='singh2017',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=True,
        T_min=200, T_max=300,
        P_min=0.0, P_max=400,
        properties=TRANSPORT_PROPS,
    ),
    'singh2017_caupin2019': ModelInfo(
        display_name='Singh (2017) / Caupin backbone',
        model_key='singh2017_caupin2019',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=True,
        # Same experimental transport data envelope as singh2017; the
        # LDS fraction comes from the Caupin (2019) EoS with parameters
        # refitted in this repo (scripts/refit_singh_transport.py).
        T_min=200, T_max=300,
        P_min=0.0, P_max=400,
        properties=TRANSPORT_PROPS,
    ),
    'singh2017_duska2020': ModelInfo(
        display_name='Singh (2017) / Duska backbone',
        model_key='singh2017_duska2020',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=True,
        # Same experimental transport data envelope as singh2017; the
        # LDS fraction comes from the Duska (2020) EoS with parameters
        # refitted in this repo (scripts/refit_singh_transport.py).
        T_min=200, T_max=300,
        P_min=0.0, P_max=400,
        properties=TRANSPORT_PROPS,
    ),
    'shi_tanaka2020': ModelInfo(
        display_name='Shi & Tanaka (2020)',
        model_key='shi_tanaka2020',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=False,
        # Fitted to scattering, density, kappaT, alphaP, Cp data over
        # roughly 240-320 K at 0.1 MPa, validated against negative-P
        # density to -110 MPa. The model puts the LLCP at 184 K / 173 MPa
        # but the paper argues it has negligible impact in the
        # experimentally accessible regime.
        T_min=180, T_max=320,
        P_min=-110, P_max=200,
        properties=TWO_STATE_PROPS,
    ),
    'shi_tanaka2020_transport': ModelInfo(
        display_name='Shi & Tanaka (2020) transport',
        model_key='shi_tanaka2020_transport',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=True,
        # Transport range matches the thermodynamic envelope above; the
        # dynamic Schottky line s^D = 1/2 falls at T ~ 207 K at 0.1 MPa,
        # close to homogeneous nucleation, so transport extrapolation
        # below ~205 K becomes very sensitive.
        T_min=180, T_max=320,
        P_min=-110, P_max=200,
        properties=TRANSPORT_PROPS,
    ),
    'vapor': ModelInfo(
        display_name='Water vapor',
        model_key='vapor',
        is_two_state=False,
        has_phase_diagram=False,
        has_transport=False,
        # IAPWS-95 ideal gas with the second and third virial coefficients
        # of its residual part. 130 K is the limit of the ideal-gas part;
        # by 400 K the truncated virial series is within 2e-4 in volume of
        # the full formulation at saturation (0.246 MPa) and loses accuracy
        # quickly beyond. Pressure must be positive.
        T_min=130, T_max=400,
        P_min=0.0, P_max=0.25,
        properties=EMPIRICAL_PROPS,
    ),
    'mishima2023equation': ModelInfo(
        display_name='Mishima & Sumita (2023), volume',
        model_key='mishima2023equation',
        is_two_state=False,
        has_phase_diagram=False,
        has_transport=False,
        T_min=135, T_max=500,
        P_min=-100, P_max=1000,
        properties=_VOLUME_RESPONSE_PROPS,
    ),
    'moynihan1996species': ModelInfo(
        display_name='Moynihan (1996)',
        model_key='moynihan1996species',
        is_two_state=True,
        has_phase_diagram=False,
        has_transport=False,
        # Four constants fitted to Mishima's LDA <-> HDA transformation at
        # 135 K (spinodals 500 and 3200 atm) and to Cp, alpha and kappa_T of
        # supercooled water at 236-277 K, 0.1-100 MPa. Not a full equation
        # of state: x, the excess heat capacity, and a volume built on the
        # paper's constant background. LLCP 217 K, 38 MPa.
        T_min=135, T_max=277,
        P_min=0.1, P_max=324.24,
        properties=MOYNIHAN_PROPS,
    ),
}

# Pure-water SAFT variants use the single dense-fluid branch.
_PCSAFT_NAMES = {
    'pcsaft_water': 'PC-SAFT water (Diamantonis–Economou)',
    'novak2024_pcsaft_ts_tv': 'Novak (2024), PC-SAFT-TS-TV',
    'novak2024_pcsaft_ts_t': 'Novak (2024), PC-SAFT-TS-T',
    'novak2025_pcsaft_ts_caf': 'Novak (2025), PC-SAFT-TS-CAF',
}
_PCSAFT_PROPS = [p for p in EMPIRICAL_PROPS if p != 'Kp']
for key, name in _PCSAFT_NAMES.items():
    two_state = key != 'pcsaft_water'
    MODEL_REGISTRY[key] = ModelInfo(
        display_name=name,
        model_key=key,
        is_two_state=two_state,
        has_phase_diagram=False,
        has_transport=False,
        T_min=273.16, T_max=703.16,
        P_min=0.1, P_max=400.0,
        properties=_PCSAFT_PROPS + (['x'] if two_state else []),
    )


def pcsaft_keys():
    """Selectable pure-water PC-SAFT models."""
    return list(_PCSAFT_NAMES)


# --- Equations fitted to simulations of model waters ------------------------
# These describe the named water potential, not real water. Each display
# name says so, and ``water_model`` names the potential. Ranges are the
# fitted ranges stated in each paper.

def _model_water(key, display, potential, T_range, P_range,
                 properties=TWO_STATE_PROPS):
    """Registry entry for a fit to the named simulation potential."""
    return ModelInfo(
        display_name=f'{display}, {potential} model water',
        model_key=key,
        is_two_state='x' in properties,
        has_phase_diagram=False,
        has_transport=any(k in properties for k in ('eta', 'D', 'tau_r')),
        T_min=T_range[0], T_max=T_range[1],
        P_min=P_range[0], P_max=P_range[1],
        properties=list(properties),
        water_model=potential,
    )


_MODEL_WATERS = {
    info.model_key: info for info in (
        _model_water('singh2016_tip4p2005', 'Singh (2016)', 'TIP4P/2005',
                     (180, 300), (-80, 400)),
        _model_water('biddle2017_tip4p2005', 'Biddle (2017), corrected 2018', 'TIP4P/2005',
                     (180, 320), (-275, 400)),
        _model_water('holten2014_st2_i_mf', 'Holten (2014), ST2(I) mean-field', 'ST2',
                     (240, 322), (100, 250)),
        _model_water('holten2014_st2_ii_mf', 'Holten (2014), ST2(II) mean-field', 'ST2',
                     (240, 315), (100, 300)),
        _model_water('holten2014_st2_i_crossover', 'Holten (2014), ST2(I) crossover', 'ST2',
                     (240, 322), (100, 250)),
        _model_water('holten2014_st2_ii_crossover', 'Holten (2014), ST2(II) crossover', 'ST2',
                     (240, 315), (100, 300)),
        _model_water('holten2013_mw', 'Holten (2013), unclustered', 'mW',
                     (170, 280), (0.1, 1013)),
        _model_water('holten2013_mw_n6', 'Holten (2013), clustered', 'mW',
                     (170, 280), (0.1, 1013)),
        _model_water('holten2013_mw_weak_crystallization',
                     'Holten (2013), weak crystallization', 'mW',
                     (170, 280), (0.1, 1013), EMPIRICAL_PROPS),
        _model_water('gartner2020_scan', 'Gartner (2020)', 'SCAN-DNN',
                     (240, 330), (0.1, 300)),
        _model_water('weis2022_wail', 'Weis (2022), printed coefficients', 'WAIL',
                     (215, 260), (10, 80)),
        _model_water('eltareb2022_qtip4pf_h2o_classical', 'Eltareb (2022), classical H₂O',
                     'q-TIP4P/F', (180, 325), (-50, 350)),
        _model_water('eltareb2022_qtip4pf_h2o_quantum', 'Eltareb (2022), quantum H₂O',
                     'q-TIP4P/F', (180, 325), (-50, 350)),
        _model_water('eltareb2022_qtip4pf_d2o_quantum', 'Eltareb (2022), quantum D₂O',
                     'q-TIP4P/F', (190, 325), (-50, 350)),
        _model_water('zorzi2024_mlbop', 'Zorzi (2024)', 'ML-BOP',
                     (100, 270), (0.1, 700)),
        _model_water('yu2023_tip4p2005', 'Yu (2023)', 'TIP4P/2005',
                     (170, 350), (0.1, 180), _YU_PROPS),
        _model_water('handle2018_tip4p2005', 'Handle (2018), landscape', 'TIP4P/2005',
                     (200, 270), (-142, 2411), EMPIRICAL_PROPS),
        _model_water('eltareb2024_qtip4pf_classical',
                     'Eltareb (2024), classical landscape, printed coefficients',
                     'q-TIP4P/F', (200, 280), (-161, 2579), EMPIRICAL_PROPS),
    )
}

MODEL_REGISTRY.update(_MODEL_WATERS)

# Ordered list for consistent UI display: real water first, model waters last
MODEL_ORDER = [
    'duska2020', 'bridge29', 'holten2014', 'caupin2019', 'caupin2019_kim',
    'shi_tanaka2020', 'shi_tanaka2020_transport',
    'grenke2025', 'singh2017', 'singh2017_caupin2019', 'singh2017_duska2020',
    'moynihan1996species', 'mishima2023equation',
    'water1', 'IAPWS95', 'vapor',
] + pcsaft_keys() + list(_MODEL_WATERS)


def model_water_keys():
    """Keys of the equations fitted to model waters (not real water)."""
    return [k for k in MODEL_ORDER if MODEL_REGISTRY[k].is_model_water]


def get_common_properties(model_keys):
    """Return the intersection of properties supported by all given models."""
    if not model_keys:
        return []
    sets = [set(MODEL_REGISTRY[k].properties) for k in model_keys if k in MODEL_REGISTRY]
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    # Preserve a canonical order
    all_keys = (_MIX_KEYS + [f'{k}_A' for k in _STATE_KEYS]
                + [f'{k}_B' for k in _STATE_KEYS] + _TRANSPORT_KEYS
                + ['Cp_ex'])
    return [k for k in all_keys if k in common]


def get_display_label(prop_key):
    """Return 'Label (unit)' string for a property key."""
    label = PROPERTY_LABELS.get(prop_key, prop_key)
    unit = PROPERTY_UNITS.get(prop_key, '')
    if unit:
        return f'{label} [{unit}]'
    return label


def models_with_phase_diagram():
    """Return list of model keys that support phase diagram computation."""
    return [k for k in MODEL_ORDER if MODEL_REGISTRY[k].has_phase_diagram]
