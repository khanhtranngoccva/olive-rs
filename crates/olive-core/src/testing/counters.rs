//! Per-test counters for aggregating operation counts (clones, drops, etc.).
//!
//! A counter is an *aggregate*: it answers "how many times did this happen",
//! with no identity attached to any individual event. It holds no per-event
//! state — just a single tally. For per-id tracking ("what happened to element
//! 3"), use [`Ledger`](crate::testing::ledger::Ledger) instead.

extern crate std;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A per-test aggregate counter. Each test constructs its own instance so there
/// is no cross-test interference from parallel execution. Backed by an atomic so
/// a shared [`Arc<Counter>`] can be observed from multiple threads.
#[derive(Debug, Default)]
pub struct Counter {
    count: AtomicUsize,
}

impl Counter {
    /// Creates a counter at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one occurrence of the event being counted. Call from a `Drop` or
    /// `TryClone` impl via the shared [`Arc`].
    pub fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current count.
    pub fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Convenience: creates an [`Arc`] handle suitable for embedding in
    /// collection elements that share the counter.
    #[must_use]
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }
}
