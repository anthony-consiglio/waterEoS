//! watereos_rs: Rust accelerators for three two-state water EoS models.

use pyo3::prelude::*;

mod caupin;
mod holten;
mod duska;
mod phase;

#[pymodule]
fn watereos_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Per-point EoS batch computation
    m.add_function(wrap_pyfunction!(caupin::compute_batch_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(holten::compute_batch_holten, m)?)?;
    m.add_function(wrap_pyfunction!(duska::compute_batch_duska, m)?)?;
    // Phase diagram (spinodal + binodal)
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_holten, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_spinodal_duska, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_caupin, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_holten, m)?)?;
    m.add_function(wrap_pyfunction!(phase::compute_binodal_duska, m)?)?;
    Ok(())
}
