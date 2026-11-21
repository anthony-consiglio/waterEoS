//! Local26, Bridge29's equation of state refitted to liquid water at
//! 175-300 K and -50 to 250 MPa — Rust implementation.
//!
//! The equation is that of bridge29.rs; the coefficients are those of
//! `watereos/data/local26.json`. Local26 has no transport model.
//! `tests/test_bridge29_rust.py` compares this backend with the NumPy one
//! (`watereos/local26/core.py`).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bridge29::{compute_batch, Equation, Model, Spinodal, Thermodynamics};

/// watereos/data/local26.json. The low-density oscillator (n_L) and the
/// 500 K high-density oscillator (n_H2) are fixed at zero.
const LOCAL26: Model = Model {
    equation: Equation::new(Thermodynamics {
        h_l: 1721.35948099646,
        s_l: 3.479692342625391,
        n_l: 0.0,
        th_l: 82.43975538767864,
        v_l: 0.24343696328646988,
        k_l: 0.0025695475191271602,
        h_h: -18507.2556497202,
        s_h: -68.6364558733257,
        n_h0: 1.9844386652707942,
        n_h2: 0.0,
        n_h3: 11.244722205965632,
        spinodal: Spinodal::Chebyshev {
            ps: [-22.430957302895475, 8.39848165168045, -0.7241029777904431],
            vs: [0.11150299806339095, 0.00627605596647058, 0.001970415582952441],
            ph: [1.872737285122753, -2.102826114384249, 0.2638401163436146],
            cs: [-1.12155933280279, 5.384518995048788, -0.8680318890521439],
        },
        e_x: 4558.513223040699,
        v_x: 0.22273837392716844,
        lg_x: 2.5521790995741447,
        w0: 822.9372709220128,
        w1: 1.9547227758982992,
        n_c: 3.895472015205659,
    }),
    transport: None,
};

/// Local26 properties at points (T [K], p [MPa]): the keys of
/// `watereos.local26.core.compute_batch`. `branch` is "stable" (default),
/// "hdl" or "ldl"; see branch.rs. With a named branch the result also has
/// `metastable`.
#[pyfunction]
#[pyo3(signature = (t_k, p_mpa, branch="stable"))]
pub fn compute_batch_local26<'py>(
    py: Python<'py>,
    t_k: PyReadonlyArray1<'py, f64>,
    p_mpa: PyReadonlyArray1<'py, f64>,
    branch: &str,
) -> PyResult<Bound<'py, PyDict>> {
    compute_batch(py, &LOCAL26, t_k, p_mpa, branch)
}
