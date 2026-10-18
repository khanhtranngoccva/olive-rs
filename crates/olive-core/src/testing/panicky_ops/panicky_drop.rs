//! A drop guard whose destructor blows up on command via *panics*, for tests
//! that must break a container mid-teardown.

extern crate std;

use super::PanicArmer;
use std::sync::Arc;

/// A drop guard that panics when the shared [`PanicArmer`] is armed.
///
/// Firing atomically *auto-disarms* the armer with a single fetch-and-set swap by
/// the time the panic starts. This prevents the double-drop scenario where the unwind
/// continues the drop and triggers the armer a second time.
///
/// This object does not aim to synchronize anything else.
#[derive(Debug)]
pub struct PanickyDrop {
    armer: Arc<PanicArmer>,
}

impl PanickyDrop {
    /// Creates a drop guard wired to `armer`.
    pub fn new(armer: Arc<PanicArmer>) -> Self {
        Self { armer }
    }
}

impl Drop for PanickyDrop {
    fn drop(&mut self) {
        // Disarm *before* panicking: drops that run during the unwind must
        // observe the disarmed state, or the process aborts (double panic).
        #[allow(
            clippy::manual_assert,
            reason = "this is not an assertion but intentional testing behavior"
        )]
        if self.armer.fetch_and_disarm() {
            panic!("PanickyDrop fired");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// An armed `PanickyDrop` panics exactly once when dropped, and the
    /// shared armer is disarmed afterwards so sibling drops stay silent.
    #[test]
    fn fires_once_per_arming() {
        let armer = Arc::new(PanicArmer::new());
        let first = PanickyDrop::new(armer.clone());
        let second = PanickyDrop::new(armer.clone());

        armer.arm();
        let panicked = catch_unwind(AssertUnwindSafe(|| drop(first)));
        assert!(panicked.is_err(), "armed PanickyDrop must panic on drop");
        assert!(!armer.is_armed(), "firing must auto-disarm the armer");

        // The sibling drop runs during the unwind in real use; here it just
        // must not panic, or the process would abort.
        drop(second);
    }

    /// A disarmed `PanickyDrop` drops silently and leaves the armer alone.
    #[test]
    fn silent_when_disarmed() {
        let armer = Arc::new(PanicArmer::new());
        drop(PanickyDrop::new(armer.clone()));
        assert!(!armer.is_armed());
    }

    /// Re-arming after a firing makes the next drop panic again: the
    /// auto-disarm consumed exactly the arming that fired.
    #[test]
    fn rearming_fires_again() {
        let armer = Arc::new(PanicArmer::new());
        let a = PanickyDrop::new(armer.clone());
        let b = PanickyDrop::new(armer.clone());

        armer.arm();
        assert!(catch_unwind(AssertUnwindSafe(|| drop(a))).is_err());

        armer.arm();
        assert!(catch_unwind(AssertUnwindSafe(|| drop(b))).is_err());
        assert!(!armer.is_armed());
    }
}
