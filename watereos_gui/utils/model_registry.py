"""
Model registry — single source of truth for waterEoS model metadata.
"""

from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    display_name: str
    model_key: str          # key passed to watereos.getProp(PT, model)
    is_two_state: bool
    has_phase_diagram: bool
    has_transport: bool
    T_min: float            # suggested minimum T (K)
    T_max: float            # suggested maximum T (K)
    P_min: float            # suggested minimum P (MPa)
    P_max: float            # suggested maximum P (MPa)
    properties: list = field(default_factory=list)


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
    'duska2020': ModelInfo(
        display_name='Duska (2020)',
        model_key='duska2020',
        is_two_state=True,
        has_phase_diagram=True,
        has_transport=False,
        T_min=200, T_max=370,
        P_min=0.1, P_max=200,
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
}

# Ordered list for consistent UI display
MODEL_ORDER = [
    'duska2020', 'holten2014', 'caupin2019',
    'grenke2025', 'singh2017',
    'water1', 'IAPWS95',
]


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
    all_keys = _MIX_KEYS + [f'{k}_A' for k in _STATE_KEYS] + [f'{k}_B' for k in _STATE_KEYS] + _TRANSPORT_KEYS
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
