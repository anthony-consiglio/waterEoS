//! Bridge37, Bridge29's equation of state with quartic spinodal functions,
//! refitted to liquid water at 100-400 K and -140 to 400 MPa together with
//! Bridge29's amorphous-water and ice constraints — Rust implementation.
//!
//! The equation is that of bridge29.rs, with the high-density state's
//! spinodal functions in the Bernstein basis; the coefficients are those of
//! `watereos/data/bridge37.json`. Bridge37 has no transport model.
//! `tests/test_bridge29_rust.py` compares this backend with the NumPy one
//! (`watereos/bridge37/core.py`).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bridge29::{compute_batch, Equation, Model, Spinodal, Thermodynamics};

/// watereos/data/bridge37.json. The 250 K and 2000 K high-density
/// oscillators (n_H1, n_H4) are fixed at zero.
const BRIDGE37: Model = Model {
    equation: Equation::new(Thermodynamics {
        h_l: 1344.177333040231,
        s_l: 1.136192067988273,
        n_l: 0.5456009380416265,
        th_l: 500.0000000000049,
        v_l: 0.5491464102883448,
        k_l: 0.003577664291305694,
        h_h: -16011.020763306493,
        s_h: -56.55624648210805,
        n_h0: 1.019932303276408,
        n_h2: 5.417237156861294,
        n_h3: 3.746425734049098,
        spinodal: Spinodal::Bernstein {
            ps: [-18.478074024190228, -20.581983245707416, -13.809654123497772, -11.088752037101928, -6.548337450105645],
            vs: [0.11066190985605527, 0.10222364640679169, 0.09186132813147882, 0.09724842508660207, 0.09917692887253707],
            ph: [0.5612255253771332, 1.1474086929421532, 2.3201143190797846, 1.1938683425554466, 1.7076303527985663],
            cs: [1.5993925446894108, 1.8559562744517084, -0.2036775419493801, 12.554151182891703, 14.074032077009226],
        },
        e_x: 4997.53073519984,
        v_x: 0.14768245742278743,
        lg_x: 1.5596009442875611,
        w0: 1044.5315560679383,
        w1: 1.5805500048657677,
        n_c: 3.08064240021824,
    }),
    transport: None,
};

/// Bridge37 properties at points (T [K], p [MPa]): the keys of
/// `watereos.bridge37.core.compute_batch`. `branch` is "stable" (default),
/// "hdl" or "ldl"; see branch.rs. With a named branch the result also has
/// `metastable`.
#[pyfunction]
#[pyo3(signature = (t_k, p_mpa, branch="stable"))]
pub fn compute_batch_bridge37<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
    branch: &str,
) -> PyResult<Bound<'py, PyDict>> {
    compute_batch(py, &BRIDGE37, t_k, p_mpa, branch)
}
