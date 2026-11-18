"""Layout of the published measured-data database.

One place for the column list, the SQL table definitions and the display
vocabulary, shared by the loader (``load.py``) and the API (``store.py``,
``routes/measured.py``).
"""
from __future__ import annotations

SCHEMA_VERSION = 1

# Columns of every fitting_data table, in file order (fitting_data/SCHEMA.md).
# The loader requires all of them; a table with extra columns loads with a
# warning, and the extra columns are not published.
SOURCE_COLUMNS = [
    'id', 'table', 'property', 'definition', 'value', 'unit', 'T_K', 'P_MPa',
    'u', 'u_type', 'u_k', 'u_T_K', 'u_P_MPa', 'u_source', 'u_digitization',
    'u_digitization_T_K', 'u_digitization_P_MPa', 'substance', 'state',
    'state_final', 'ice_phase', 'sample', 'sample_detail', 'method', 'rate',
    'rate_unit', 'data_type', 'source_key', 'location', 'via', 'extraction',
    'verification', 'corrections_applied', 'used_in_fits',
    'used_in_fits_level', 'compared_in', 'qc_flag', 'qc_note', 'value_orig',
    'unit_orig', 'T_orig', 'T_unit_orig', 'T_scale_orig', 'P_orig',
    'P_unit_orig', 'notes',
]

REAL_SOURCE_COLUMNS = {
    'value', 'T_K', 'P_MPa', 'u', 'u_k', 'u_T_K', 'u_P_MPa', 'u_digitization',
    'u_digitization_T_K', 'u_digitization_P_MPa',
}

# Floating-point columns. PostgreSQL's REAL is single precision, which would
# round the published numbers; DOUBLE PRECISION is 8 bytes in both databases
# (SQLite gives it REAL affinity).
REAL = 'DOUBLE PRECISION'

# Columns added by the loader.
#   tbl          the fitting_data table key (``table`` is a reserved SQL word)
#   table_label  the CSV path without extension, e.g. ``liquid/density``
#   grp          the folder: liquid, phase_equilibrium, amorphous, kinetic,
#                dynamics, structure
#   u_std        absolute standard uncertainty (k = 1) of ``value``
#   rate_K_min   heating or cooling rate in K/min, when given in K/min, K/s or K/h
#   year         publication year, from sources.csv
DERIVED_COLUMNS = [
    ('tbl', 'TEXT'), ('table_label', 'TEXT'), ('grp', 'TEXT'),
    ('u_std', REAL), ('rate_K_min', REAL), ('year', 'INTEGER'),
]

# Published columns, in order: the derived ones first, then the source
# columns except ``table`` (stored as ``tbl``).
ROW_COLUMNS = [c for c, _ in DERIVED_COLUMNS] + [c for c in SOURCE_COLUMNS if c != 'table']


def _row_column_type(col):
    for c, t in DERIVED_COLUMNS:
        if c == col:
            return t
    return REAL if col in REAL_SOURCE_COLUMNS else 'TEXT'


ROW_COLUMN_TYPES = {c: _row_column_type(c) for c in ROW_COLUMNS}

SOURCE_TABLE_COLUMNS = [
    ('source_key', 'TEXT PRIMARY KEY'), ('citation', 'TEXT'), ('cite_short', 'TEXT'),
    ('year', 'INTEGER'), ('doi', 'TEXT'), ('n_rows', 'INTEGER'), ('tables', 'TEXT'),
    ('extraction', 'TEXT'), ('used_in_fits', 'TEXT'),
]

RELEASE_COLUMNS = [
    ('release_id', 'TEXT PRIMARY KEY'), ('created_at', 'TEXT'), ('source_modified_at', 'TEXT'),
    ('n_rows', 'INTEGER'), ('n_sources', 'INTEGER'), ('n_tables', 'INTEGER'),
    ('inputs_sha256', 'TEXT'), ('schema_version', 'INTEGER'),
]

# PostgreSQL folds unquoted identifiers to lower case (T_K comes back as t_k);
# the API maps result column names back through this table.
CANONICAL_NAMES = {c.lower(): c for c in ROW_COLUMNS + [c for c, _ in SOURCE_TABLE_COLUMNS]
                   + [c for c, _ in RELEASE_COLUMNS]}

TABLE_ROWS = 'measured_rows'
TABLE_SOURCES = 'measured_sources'
TABLE_RELEASE = 'measured_release'


def ddl(prefix=''):
    """CREATE TABLE statements, valid in SQLite and PostgreSQL."""
    def cols(spec):
        return ',\n  '.join(f'{c} {t}' for c, t in spec)

    rows_spec = [(c, ROW_COLUMN_TYPES[c] + (' PRIMARY KEY' if c == 'id' else '')) for c in ROW_COLUMNS]
    return [
        f'CREATE TABLE {prefix}{TABLE_ROWS} (\n  {cols(rows_spec)}\n)',
        f'CREATE INDEX {prefix}{TABLE_ROWS}_tbl_property ON {prefix}{TABLE_ROWS} (tbl, property)',
        f'CREATE TABLE {prefix}{TABLE_SOURCES} (\n  {cols(SOURCE_TABLE_COLUMNS)}\n)',
        f'CREATE TABLE {prefix}{TABLE_RELEASE} (\n  {cols(RELEASE_COLUMNS)}\n)',
    ]


# ---------------------------------------------------------------------------
# Display vocabulary
# ---------------------------------------------------------------------------
GROUPS = [
    # key, label, thermodynamic
    ('liquid', 'Liquid', True),
    ('phase_equilibrium', 'Phase equilibrium', True),
    ('amorphous', 'Amorphous', True),
    ('kinetic', 'Kinetic (not thermodynamic)', False),
    ('dynamics', 'Dynamics (transport, not thermodynamic)', False),
    ('structure', 'Structure (derived fractions)', False),
]
GROUP_KEYS = [g[0] for g in GROUPS]

PROPERTY_LABELS = {
    'density': 'Density ρ',
    'isothermal_compressibility': 'Isothermal compressibility κ_T',
    'isobaric_expansivity': 'Isobaric expansivity α_P',
    'isobaric_heat_capacity': 'Isobaric heat capacity c_p',
    'speed_of_sound': 'Speed of sound w',
    'temperature_of_maximum_density': 'Temperature of maximum density',
    'melting_point': 'Melting point',
    'triple_point': 'Triple point',
    'vapor_pressure': 'Vapor pressure',
    'vapor_pressure_difference': 'Vapor-pressure difference, liquid − ice',
    'enthalpy_change': 'Enthalpy change ΔH',
    'entropy_difference': 'Entropy difference ΔS',
    'gibbs_energy_difference': 'Gibbs energy difference ΔG',
    'enthalpy_difference': 'Enthalpy difference ΔH',
    'glass_transition_temperature': 'Glass transition temperature T_g',
    'heat_capacity_step': 'Heat-capacity step ΔC_p',
    'glass_transition_width': 'Glass transition width',
    'volume_change': 'Volume change ΔV',
    'relative_volume_change': 'Relative volume change',
    'homogeneous_nucleation_temperature': 'Homogeneous nucleation temperature T_H',
    'nucleation_rate': 'Nucleation rate J',
    'crystallization_temperature': 'Crystallization temperature T_x',
    'crystallization_time': 'Crystallization time',
    'crystal_growth_rate': 'Crystal growth rate',
    'transformation_pressure': 'Transformation pressure',
    'transformation_temperature': 'Transformation temperature',
    'equilibrium_window': 'Equilibrium window',
    'cavitation_pressure': 'Cavitation pressure',
    'self_diffusion_coefficient': 'Self-diffusion coefficient D',
    'shear_viscosity': 'Shear viscosity η',
    'dielectric_relaxation_time': 'Dielectric relaxation time τ',
    'rotational_correlation_time': 'Rotational correlation time τ',
    'spin_lattice_relaxation_time': 'NMR spin-lattice relaxation time T₁',
    'structural_relaxation_time': 'Structural relaxation time τ',
    'tetrahedral_fraction': 'Tetrahedral fraction',
    'tetrahedral_fraction_proxy': 'X-ray tetrahedral-fraction proxy (1 − g_OO)',
}

# Loci: the value repeats T_K (LOCUS_T) or P_MPa (LOCUS_P); they are drawn in
# the T-P plane rather than against T or P.
LOCUS_T = {
    'temperature_of_maximum_density', 'melting_point', 'triple_point',
    'glass_transition_temperature', 'homogeneous_nucleation_temperature',
    'crystallization_temperature', 'transformation_temperature', 'equilibrium_window',
}
LOCUS_P = {'transformation_pressure', 'vapor_pressure', 'cavitation_pressure'}

# Properties whose value axis starts logarithmic (others switch automatically
# when their positive values span three decades or more).
LOG_DEFAULT = {'vapor_pressure', 'nucleation_rate', 'crystallization_time'}

# Model overlays: data property -> (waterEoS output attribute, invert). Only
# liquid and dynamics tables get overlays; the models describe the liquid.
OVERLAY_PROPERTIES = {
    'density': ('rho', False),
    'isothermal_compressibility': ('Kt', True),   # waterEoS gives the bulk modulus K_T in MPa
    'isobaric_expansivity': ('alpha', False),
    'isobaric_heat_capacity': ('Cp', False),
    'speed_of_sound': ('vel', False),
    'shear_viscosity': ('eta', False),
    'self_diffusion_coefficient': ('D', False),
    'rotational_correlation_time': ('tau_r', False),
}
OVERLAY_GROUPS = {'liquid', 'dynamics'}
# Registry models never offered as liquid overlays.
OVERLAY_EXCLUDED_MODELS = {'vapor'}


def property_label(prop):
    return PROPERTY_LABELS.get(prop, prop.replace('_', ' ').capitalize())


def locus_kind(prop):
    if prop in LOCUS_T:
        return 'T'
    if prop in LOCUS_P:
        return 'P'
    return None


def overlay_attr(group, prop):
    """(attribute, invert) when the models can be drawn over this property, else None."""
    if group not in OVERLAY_GROUPS:
        return None
    return OVERLAY_PROPERTIES.get(prop)


# What the site and the download say about the data and how to cite them.
NOTICE = (
    'These values were retrieved or transcribed from published studies; each row names the study it '
    'comes from (source_key, citation, DOI, and the table, page or figure in "location"). The waterEoS '
    'project asserts no rights over the values and attaches no license to them. Values taken from '
    "supplementary files remain subject to the terms of the publisher of each study. When you use these "
    'data, cite the original studies, and cite waterEoS for the compilation.'
)
CITE_WATEREOS = (
    'A. Consiglio, waterEoS: Thermodynamic equations of state for supercooled water, '
    'https://github.com/anthony-consiglio/waterEoS (measured-data compilation, release {release_id}).'
)
