//! Newtypes whose trait operations *fail on command* — returning `Err` rather
//! than panicking — for tests that must break an algorithm mid-flight and check
//! the recovery path.
//!
//! This is different from [`panicky_ops`](crate::panicky_ops), whose operations blow up
//! with a panic.
//!
//! The failure triggers come in three shapes:
//! - *command-triggered*, set off at will through a [`FailArmer`]
//!   ([`MaybeUnclonable`]),
//! - *unconditional* (like [`NoDefault`]),
//! - *delayed*, landing after a set number of clones ([`FlakyClone`] with a
//!   per-instance counter, [`CloneBudget`] with one budget shared by every
//!   clone).

extern crate std;

use std::sync::atomic::{AtomicBool, Ordering};

pub mod clone_budget;
pub mod flaky_clone;
pub mod maybe_unclonable;
pub mod no_default;
pub use clone_budget::CloneBudget;
pub use flaky_clone::FlakyClone;
pub use maybe_unclonable::MaybeUnclonable;
pub use no_default::NoDefault;

/// An atomically-toggled switch controlling whether the elements holding
/// it will fail their fallible operation. This armer does not
/// aim to synchronize anything else.
///
/// To make it shared, wrap it in an Arc.
#[derive(Debug, Default)]
pub struct FailArmer {
    armed: AtomicBool,
}

impl FailArmer {
    /// Creates an armer in the disarmed state (the operation is allowed).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an armer in the armed state (the operation fails).
    #[must_use]
    pub fn armed() -> Self {
        let armer = Self::default();
        armer.arm();
        armer
    }

    /// Arms the armer, so the operation on every element holding it fails.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Disarms the armer, so the operation on every element holding it succeeds.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    /// Whether the armer is currently armed.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }
}
