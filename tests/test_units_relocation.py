def test_units_importable_from_watereos():
    from watereos.units import (
        get_factor, get_unit_string, convert_array, display_label,
        UNIT_DEFAULTS, UNIT_OPTIONS, CATEGORY_LABELS,
    )
    assert get_factor("rho", {"unit_density": "g/cm³"}) == 1e-3
    assert get_factor("rho", None) == 1.0
    assert "unit_energy" in UNIT_DEFAULTS
    assert display_label("rho", None) == "Density [kg/m³]"

# The previous `test_legacy_shim_still_works` checked that
# `watereos_visualizer.units` re-exported `get_factor` from `watereos.units`.
# That entire package was retired when the front-end migrated to FastAPI +
# Vite, so the shim no longer exists and the test was removed.
