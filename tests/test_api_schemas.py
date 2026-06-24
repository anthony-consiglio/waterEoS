import pytest
from pydantic import ValidationError
from watereos_api.schemas import (
    CurvesRequest, SurfaceRequest, CompareRequest, PointRequest,
    EosPhaseRequest, H2OPhaseRequest,
)


def test_curves_request_defaults_and_validation():
    req = CurvesRequest(model="duska2020", property="rho",
                         T_range=[200, 300], P_range=[0.1, 200])
    assert req.n_curves == 5 and req.n_points == 200
    assert req.isobar_mode is True and req.theme == "dark"
    with pytest.raises(ValidationError):
        CurvesRequest(model="duska2020", property="rho",
                      T_range=[200], P_range=[0.1, 200])  # bad tuple len
    with pytest.raises(ValidationError):
        CurvesRequest(model="duska2020", property="rho", T_range=[200, 300],
                      P_range=[0.1, 200], theme="neon")   # bad theme


def test_point_request():
    r = PointRequest(model_keys=["duska2020", "holten2014"], T_K=273.15, P_MPa=0.1)
    assert r.units is None


def test_h2o_request_projection_enum():
    H2OPhaseRequest(projection="tv")
    with pytest.raises(ValidationError):
        H2OPhaseRequest(projection="xyz")


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        CurvesRequest(model="duska2020", property="rho",
                      T_range=[200, 300], P_range=[0.1, 200],
                      moodel="typo")          # typo'd field must 422
    with pytest.raises(ValidationError):
        PointRequest(model_keys=["duska2020"], T_K=273.15, P_MPa=0.1,
                      extra_field=1)
    with pytest.raises(ValidationError):
        H2OPhaseRequest(projection="tv", units={"unit_density": "g/cm³"})  # H2O has no units field
