"""
Model registry validation tests.

Verifies that MODEL_REGISTRY declarations match what each model's
getProp() actually returns — catches drift between the registry and
the real model implementations.
"""

import numpy as np
import pytest
from watereos.model_registry import (
    MODEL_REGISTRY,
    MODEL_ORDER,
    PROPERTY_LABELS,
    PROPERTY_UNITS,
    get_common_properties,
    get_display_label,
    models_with_phase_diagram,
)
from watereos import getProp, list_models


# A safe (T, P) point that all models can evaluate
PT_PROBE = np.array([[0.1], [273.15]], dtype=object)


# ───────────────────────────────────────────────────────────────────────
# 1. Registry completeness: every dispatcher model is in the registry
# ───────────────────────────────────────────────────────────────────────

class TestRegistryCompleteness:
    def test_all_dispatcher_models_in_registry(self):
        """Every model known to watereos.list_models() must have a registry entry."""
        dispatcher_models = set(list_models())
        registry_models = set(MODEL_REGISTRY.keys())
        missing = dispatcher_models - registry_models
        assert not missing, (
            f"Models in dispatcher but not in MODEL_REGISTRY: {missing}"
        )

    def test_no_stale_registry_entries(self):
        """Every registry entry must correspond to a real dispatcher model."""
        dispatcher_models = set(list_models())
        registry_models = set(MODEL_REGISTRY.keys())
        stale = registry_models - dispatcher_models
        assert not stale, (
            f"Models in MODEL_REGISTRY but not in dispatcher: {stale}"
        )

    def test_model_order_covers_non_seafreeze(self):
        """MODEL_ORDER should include all non-SeaFreeze models."""
        non_sf = {k for k in MODEL_REGISTRY if k not in ('water1', 'IAPWS95')}
        ordered = set(MODEL_ORDER)
        missing = non_sf - ordered
        assert not missing, f"Models missing from MODEL_ORDER: {missing}"


# ───────────────────────────────────────────────────────────────────────
# 2. Property declarations match actual getProp() output
# ───────────────────────────────────────────────────────────────────────

class TestPropertySync:
    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_declared_properties_exist(self, model_key):
        """Every property declared in the registry must exist on getProp output."""
        out = getProp(PT_PROBE, model_key)
        info = MODEL_REGISTRY[model_key]
        missing = []
        for prop in info.properties:
            if not hasattr(out, prop):
                missing.append(prop)
        assert not missing, (
            f"{model_key}: registry declares properties not found on output: {missing}"
        )

    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_no_undeclared_properties(self, model_key):
        """Warn about properties on the output that aren't in the registry."""
        out = getProp(PT_PROBE, model_key)
        info = MODEL_REGISTRY[model_key]
        declared = set(info.properties)

        # Collect all public numeric attributes from the output
        undeclared = []
        for attr in dir(out):
            if attr.startswith('_'):
                continue
            val = getattr(out, attr, None)
            if isinstance(val, np.ndarray) and attr not in declared:
                undeclared.append(attr)

        # This is a warning, not a hard failure — SeaFreeze pass-throughs
        # may have extra attributes (like 'shear', 'Vp', 'Vs')
        if undeclared:
            import warnings
            warnings.warn(
                f"{model_key}: output has undeclared properties: {undeclared}",
                stacklevel=2,
            )


# ───────────────────────────────────────────────────────────────────────
# 3. Property values are finite at the probe point
# ───────────────────────────────────────────────────────────────────────

# Properties that should be finite at 273.15 K, 0.1 MPa for all models
ALWAYS_FINITE = ['rho', 'V', 'S', 'G', 'H', 'Cp', 'alpha']


class TestPropertyValues:
    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_core_properties_finite(self, model_key):
        """Core properties must be finite at the standard probe point."""
        out = getProp(PT_PROBE, model_key)
        for prop in ALWAYS_FINITE:
            if hasattr(out, prop):
                val = np.asarray(getattr(out, prop)).ravel()
                assert np.all(np.isfinite(val)), (
                    f"{model_key}.{prop} is not finite at 273.15 K, 0.1 MPa: {val}"
                )


# ───────────────────────────────────────────────────────────────────────
# 4. Flags consistency
# ───────────────────────────────────────────────────────────────────────

class TestFlags:
    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_two_state_has_x(self, model_key):
        """Two-state models must have 'x' in their properties list."""
        info = MODEL_REGISTRY[model_key]
        if info.is_two_state:
            assert 'x' in info.properties, (
                f"{model_key}: is_two_state=True but 'x' not in properties"
            )

    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_non_two_state_lacks_x(self, model_key):
        """Non-two-state models must NOT have 'x' in their properties list."""
        info = MODEL_REGISTRY[model_key]
        if not info.is_two_state:
            assert 'x' not in info.properties, (
                f"{model_key}: is_two_state=False but 'x' in properties"
            )

    @pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
    def test_transport_flag_matches_properties(self, model_key):
        """has_transport=True iff transport properties (eta, D) are declared."""
        info = MODEL_REGISTRY[model_key]
        has_eta = 'eta' in info.properties
        assert info.has_transport == has_eta, (
            f"{model_key}: has_transport={info.has_transport} but "
            f"'eta' {'in' if has_eta else 'not in'} properties"
        )

    def test_phase_diagram_models_have_functions(self):
        """Models with has_phase_diagram=True must export find_LLCP."""
        for model_key in models_with_phase_diagram():
            # Map registry key to module name
            module_map = {
                'holten2014': 'holten_eos',
                'caupin2019': 'caupin_eos',
                'duska2020': 'duska_eos',
            }
            if model_key in module_map:
                mod = __import__(module_map[model_key])
                assert hasattr(mod, 'find_LLCP'), (
                    f"{model_key}: has_phase_diagram=True but module lacks find_LLCP"
                )


# ───────────────────────────────────────────────────────────────────────
# 5. Labels and units coverage
# ───────────────────────────────────────────────────────────────────────

class TestLabelsAndUnits:
    def test_all_properties_have_labels(self):
        """Every property appearing in any model must have a label."""
        all_props = set()
        for info in MODEL_REGISTRY.values():
            all_props.update(info.properties)
        missing = [p for p in all_props if p not in PROPERTY_LABELS]
        assert not missing, f"Properties without labels: {missing}"

    def test_all_properties_have_units(self):
        """Every property appearing in any model must have a unit entry."""
        all_props = set()
        for info in MODEL_REGISTRY.values():
            all_props.update(info.properties)
        missing = [p for p in all_props if p not in PROPERTY_UNITS]
        assert not missing, f"Properties without units: {missing}"

    def test_display_label_format(self):
        """get_display_label should return 'Label [unit]' for properties with units."""
        label = get_display_label('rho')
        assert 'Density' in label
        assert 'kg/m' in label

        label_x = get_display_label('x')
        assert 'LDL' in label_x
        assert '[' not in label_x  # dimensionless, no unit bracket


# ───────────────────────────────────────────────────────────────────────
# 6. Utility function tests
# ───────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_common_properties_two_state(self):
        """Common properties of two two-state models should include x."""
        common = get_common_properties(['holten2014', 'caupin2019'])
        assert 'x' in common
        assert 'rho' in common

    def test_common_properties_mixed(self):
        """Common properties between two-state and empirical should exclude x."""
        common = get_common_properties(['holten2014', 'grenke2025'])
        assert 'x' not in common
        assert 'rho' in common

    def test_common_properties_empty_input(self):
        assert get_common_properties([]) == []

    def test_models_with_phase_diagram(self):
        pd_models = models_with_phase_diagram()
        assert 'duska2020' in pd_models
        assert 'caupin2019' in pd_models
        assert 'holten2014' in pd_models
        assert 'grenke2025' not in pd_models
        assert 'singh2017' not in pd_models
