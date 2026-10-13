//! Per-test counters for tracking operation counts (clones, drops, etc.)
//! without relying on process-global statics.
//!
//! Each counter is backed by an [`AtomicUsize`] and shared via [`Arc`] so that
//! multiple handles (e.g. every element in a collection) observe and mutate
//! the *same* counter safely across threads. Construct one per test to avoid
//! cross-test interference under parallel execution.

extern crate std;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A per-test clone counter. Each test constructs its own instance so there is
/// no cross-test interference from parallel execution. Backed by an atomic so a
/// shared [`Arc<CloneCounter>`] can be observed from multiple threads.
#[derive(Debug, Default)]
pub struct CloneCounter {
    count: AtomicUsize,
}

impl CloneCounter {
    /// Returns the current clone count.
    pub fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Increments the counter (call from a `TryClone` impl via the shared `Arc`).
    pub fn record_clone(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Convenience: creates an [`Arc`] handle suitable for embedding in
    /// collection elements that share the counter.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }
}
