"""
Unit conversion module for waterEoS-Visualizer.

Converts EoS output values from native SI-like units to user-selected
display units.  All conversions are pure multiplication factors applied
at render time — computation always runs in native units.
"""

from watereos.model_registry import PROPERTY_LABELS, PROPERTY_UNITS

# Molar mass of water (kg/mol)
MW_WATER = 0.01801528

# ---------------------------------------------------------------------------
# Property → category mapping
# ---------------------------------------------------------------------------

_BASE_CATEGORY = {
    'rho': 'density',
    'V': 'volume',
    'S': 'entropy', 'Cp': 'entropy', 'Cv': 'entropy',
    'G': 'energy', 'H': 'energy', 'U': 'energy', 'A': 'energy',
    'Kt': 'bulk_modulus', 'Ks': 'bulk_modulus',
    'Kp': None, 'alpha': None, 'vel': None, 'x': None,
    'eta': 'viscosity', 'D': None, 'tau_r': None, 'f': None,
}

PROP_CATEGORY = dict(_BASE_CATEGORY)
for _k, _cat in list(_BASE_CATEGORY.items()):
    if _cat is not None and _k not in ('eta', 'D', 'tau_r', 'f', 'x'):
        PROP_CATEGORY[_k + '_A'] = _cat
        PROP_CATEGORY[_k + '_B'] = _cat

# ---------------------------------------------------------------------------
# Default unit per category (matches native SI output)
# ---------------------------------------------------------------------------

UNIT_DEFAULTS = {
    'unit_density': 'kg/m\u00b3',
    'unit_volume': 'm\u00b3/kg',
    'unit_energy': 'J/kg',
    'unit_entropy': 'J/(kg\u00b7K)',
    'unit_bulk_modulus': 'MPa',
    'unit_viscosity': 'Pa\u00b7s',
}

# ---------------------------------------------------------------------------
# Dropdown options shown in Settings
# ---------------------------------------------------------------------------

UNIT_OPTIONS = {
    'unit_density': [
        {'label': 'kg/m\u00b3', 'value': 'kg/m\u00b3'},
        {'label': 'g/cm\u00b3', 'value': 'g/cm\u00b3'},
    ],
    'unit_volume': [
        {'label': 'm\u00b3/kg', 'value': 'm\u00b3/kg'},
        {'label': 'cm\u00b3/g', 'value': 'cm\u00b3/g'},
    ],
    'unit_energy': [
        {'label': 'J/kg', 'value': 'J/kg'},
        {'label': 'kJ/kg', 'value': 'kJ/kg'},
        {'label': 'J/mol', 'value': 'J/mol'},
        {'label': 'kJ/mol', 'value': 'kJ/mol'},
    ],
    'unit_entropy': [
        {'label': 'J/(kg\u00b7K)', 'value': 'J/(kg\u00b7K)'},
        {'label': 'kJ/(kg\u00b7K)', 'value': 'kJ/(kg\u00b7K)'},
        {'label': 'J/(mol\u00b7K)', 'value': 'J/(mol\u00b7K)'},
        {'label': 'kJ/(mol\u00b7K)', 'value': 'kJ/(mol\u00b7K)'},
    ],
    'unit_bulk_modulus': [
        {'label': 'MPa', 'value': 'MPa'},
        {'label': 'GPa', 'value': 'GPa'},
    ],
    'unit_viscosity': [
        {'label': 'Pa\u00b7s', 'value': 'Pa\u00b7s'},
        {'label': 'mPa\u00b7s', 'value': 'mPa\u00b7s'},
    ],
}

# ---------------------------------------------------------------------------
# Conversion factors:  native_value * factor = display_value
# ---------------------------------------------------------------------------

_FACTORS = {
    'density': {
        'kg/m\u00b3': 1.0,
        'g/cm\u00b3': 1e-3,
    },
    'volume': {
        'm\u00b3/kg': 1.0,
        'cm\u00b3/g': 1e3,
    },
    'energy': {
        'J/kg': 1.0,
        'kJ/kg': 1e-3,
        'J/mol': MW_WATER,
        'kJ/mol': MW_WATER * 1e-3,
    },
    'entropy': {
        'J/(kg\u00b7K)': 1.0,
        'kJ/(kg\u00b7K)': 1e-3,
        'J/(mol\u00b7K)': MW_WATER,
        'kJ/(mol\u00b7K)': MW_WATER * 1e-3,
    },
    'bulk_modulus': {
        'MPa': 1.0,
        'GPa': 1e-3,
    },
    'viscosity': {
        'Pa\u00b7s': 1.0,
        'mPa\u00b7s': 1e3,
    },
}

# Display labels for each category (used in Settings UI)
CATEGORY_LABELS = {
    'unit_density': 'Density',
    'unit_volume': 'Volume',
    'unit_energy': 'Energy (G, H, U, A)',
    'unit_entropy': 'Entropy / Heat cap.',
    'unit_bulk_modulus': 'Bulk modulus (Kt, Ks)',
    'unit_viscosity': 'Viscosity (\u03b7)',
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _base_key(prop_key):
    """Strip _A / _B suffix to get the base property key."""
    if prop_key.endswith('_A') or prop_key.endswith('_B'):
        return prop_key[:-2]
    return prop_key


def get_factor(prop_key, settings):
    """Return multiplication factor to convert native SI → display unit."""
    cat = PROP_CATEGORY.get(_base_key(prop_key))
    if cat is None:
        return 1.0
    setting_key = f'unit_{cat}'
    target = (settings or {}).get(setting_key, UNIT_DEFAULTS[setting_key])
    return _FACTORS[cat].get(target, 1.0)


def get_unit_string(prop_key, settings):
    """Return the display unit string for a property under current settings."""
    cat = PROP_CATEGORY.get(_base_key(prop_key))
    if cat is None:
        return PROPERTY_UNITS.get(prop_key, '')
    setting_key = f'unit_{cat}'
    return (settings or {}).get(setting_key, UNIT_DEFAULTS[setting_key])


def convert_array(prop_key, arr, settings):
    """Multiply a list (or numpy array) of values by the conversion factor."""
    factor = get_factor(prop_key, settings)
    if factor == 1.0:
        return arr
    return [v * factor for v in arr]


def display_label(prop_key, settings):
    """Return 'Label [display_unit]' for a property key."""
    label = PROPERTY_LABELS.get(prop_key, prop_key)
    unit = get_unit_string(prop_key, settings)
    if unit:
        return f'{label} [{unit}]'
    return label
