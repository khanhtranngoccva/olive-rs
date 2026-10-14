//! A shared clone budget: every clone in the test draws from one counter.

extern crate std;

use core::sync::atomic::{AtomicU32, Ordering};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use std::sync::Arc;

/// A per-test clone budget whose counter is shared by *every* clone: when the
/// budget is exhausted, all clones fail, no matter which value they were
/// cloned from.
///
/// This is the [`FlakyClone`](super::FlakyClone) poison pill with a shared
/// counter instead of a unique one.
#[derive(Debug)]
pub struct CloneBudget {
    remaining: Arc<AtomicU32>,
}

impl CloneBudget {
    /// Creates a budget allowing `remaining` successful clones before failing.
    pub fn new(remaining: u32) -> Self {
        Self {
            remaining: Arc::new(AtomicU32::new(remaining)),
        }
    }

    /// Attempts to consume one unit of budget. Returns `true` if a clone is
    /// allowed, `false` if the budget is exhausted. Atomic so concurrent
    /// consumers can't overdraw the shared budget.
    pub fn try_consume(&self) -> bool {
        // `fetch_update` retries on contention; the closure rejects only when
        // the counter is already zero, so the sole failure value is `Err(0)`.
        self.remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                if cur > 0 { Some(cur - 1) } else { None }
            })
            .is_ok()
    }

    /// Obtains a new handle sharing this budget's counter *without* drawing
    /// from it. Used to seed collection elements at construction time, where
    /// creating the payload must not count against the clone budget — only
    /// actual [`TryClone`]s do.
    #[inline]
    pub fn share(&self) -> Self {
        Self {
            remaining: self.remaining.clone(),
        }
    }
}

// Cloning it draws one unit from the shared counter and hands back a handle
// that observes the same remaining budget. When the budget is exhausted,
// the clone deterministically fails.
impl TryClone for CloneBudget {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if !self.try_consume() {
            return Err(TryCloneError::Other("budget exhausted"));
        }
        Ok(Self {
            remaining: self.remaining.clone(),
        })
    }
}
