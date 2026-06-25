//! watereos_rs: Rust accelerators for three two-state water EoS models.

use pyo3::prelude::*;

mod caupin;
mod holten;
mod duska;
mod shi_tanaka;
mod phase;
mod seafreeze;
mod hull;
mod hull_v2;

#[pymodule]
fn watereos_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Per-point EoS batch computation
    m.add_function(wrap_pyfunction!(caupin::compute_batch_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(caupin::compute_batch_caupin_kim, m)?)?;
    m.add_function(wrap_pyfunction!(holten::compute_batch_holten, m)?)?;
    m.add_function(wrap_pyfunction!(duska::compute_batch_duska, m)?)?;
    m.add_function(wrap_pyfunction!(shi_tanaka::compute_batch_shi_tanaka, m)?)?;
    // Phase diagram (spinodal + binodal)
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_caupin_kim, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_holten, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_duska, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_caupin_kim, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_holten, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_duska, m)?)?;
    // Native SeaFreeze evaluator (B-spline values + thermodynamic
    // property derivations from G derivatives)
    m.add_function(wrap_pyfunction!(seafreeze::sf_getprop_grid, m)?)?;
    m.add_function(wrap_pyfunction!(seafreeze::sf_getprop_scatter, m)?)?;
    m.add_function(wrap_pyfunction!(seafreeze::sf_eval_raw, m)?)?;
    m.add_function(wrap_pyfunction!(seafreeze::sf_phases, m)?)?;
    // Binary phase diagram hull sweep
    m.add_function(wrap_pyfunction!(hull::hull_sweep, m)?)?;
    // pTvx per-x hull sweep v2
    m.add_function(wrap_pyfunction!(hull_v2::hull_sweep_v2, m)?)?;
    // Lower hull (exposed to Python)
    m.add_function(wrap_pyfunction!(hull_v2::lower_hull_rs, m)?)?;
    // Batch bisection hull
    m.add_function(wrap_pyfunction!(hull_v2::hull_bisect, m)?)?;
    Ok(())
}
