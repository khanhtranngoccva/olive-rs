//! Newtypes whose trait operations blow up on command via *panics*, for tests that
//! must break an algorithm mid-flight.

extern crate std;

pub mod panicky_drop;
pub mod panicky_ord;
pub use panicky_drop::PanickyDrop;
pub use panicky_ord::PanickyOrd;

use std::sync::atomic::{AtomicBool, Ordering};

/// A per-test panic armer. Starts disarmed; the test arms it before triggering
/// the code path under test and can disarm it afterwards. This armer does not
/// aim to synchronize anything else.
#[derive(Debug, Default)]
pub struct PanicArmer {
    armed: AtomicBool,
}

impl PanicArmer {
    /// Creates an armer in the disarmed state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Arms the armer, so comparisons touching it will panic.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Disarms the armer, so comparisons touching it stop panicking.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    /// Whether the armer is currently armed.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Atomically disarms the armer, returning whether it was armed.
    /// This helps prevent the double-panic scenario.
    pub fn fetch_and_disarm(&self) -> bool {
        self.armed.swap(false, Ordering::Relaxed)
    }
}
