//! Scheduling of per-point batch evaluation.
//!
//! Every batch entry point evaluates one independent function at each
//! of `n` points. Handing that work to the rayon pool is not free: the
//! calling thread parks, a worker wakes, and every task that is split
//! off wakes another worker. Measured on a 30-core machine, that costs
//! 20-30 us with a pool of four threads and typically 150-250 us with
//! the default pool of one thread per core, while a point costs
//! 0.1-1 us. A batch is therefore split into tasks of at least `grain`
//! points, and a batch too small for two such tasks is evaluated on the
//! calling thread without touching the pool.
//!
//! The grain is set per evaluator so that a task carries about 60 us of
//! work: enough to pay for waking a thread, and the smallest size at
//! which splitting was not slower than the calling thread for pools of
//! 2, 4, 8 and 30 threads.
//!
//! Which path runs has no effect on the results: both call the same
//! function at the same points and store the values in index order.

use rayon::prelude::*;

/// Evaluate `f(i)` for `i` in `0..n` and return the values in index
/// order.
///
/// A batch of fewer than `2 * grain` points runs on the calling thread,
/// as does every batch when the pool has a single thread. A larger one
/// is split across the rayon pool into tasks of at least `grain`
/// points.
pub fn map_indexed<T, F>(n: usize, grain: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if n < 2 * grain || rayon::current_num_threads() < 2 {
        (0..n).map(f).collect()
    } else {
        (0..n).into_par_iter().with_min_len(grain).map(f).collect()
    }
}
