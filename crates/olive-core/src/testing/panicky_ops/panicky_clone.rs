//! The [`PanickyClone`] newtype: a payload whose clone blows up on command via
//! panics, for tests that must break a cloning algorithm mid-flight.

extern crate std;

use super::PanicArmer;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use std::sync::Arc;

/// A payload whose [`TryClone`] panics while its attached [`PanicArmer`] is
/// armed.
///
/// Firing atomically *auto-disarms* the armer with a single fetch-and-set swap
/// by the time the panic starts, so drops that run during the unwind observe
/// the disarmed state instead of aborting the process with a second panic.
#[derive(Debug)]
pub struct PanickyClone {
    armer: Arc<PanicArmer>,
}

impl PanickyClone {
    /// Creates an instance wired to its own armer, which starts disarmed.
    #[must_use]
    pub fn new() -> Self {
        Self::wired_to(Arc::new(PanicArmer::new()))
    }

    /// Creates an instance wired to `armer`.
    pub fn wired_to(armer: Arc<PanicArmer>) -> Self {
        Self { armer }
    }

    /// The shared armer, for arming or inspecting it from the test side.
    #[must_use]
    pub fn armer(&self) -> &Arc<PanicArmer> {
        &self.armer
    }

    /// Arms the item, so its next clone panics.
    pub fn arm(&self) {
        self.armer.arm();
    }

    /// Disarms the item, so clones stop panicking.
    pub fn disarm(&self) {
        self.armer.disarm();
    }

    /// Whether the next clone of this item would panic.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armer.is_armed()
    }
}

impl Default for PanickyClone {
    fn default() -> Self {
        Self::new()
    }
}

impl TryClone for PanickyClone {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Disarm *before* panicking: drops that run during the unwind must
        // observe the disarmed state, or the process aborts (double panic).
        #[allow(
            clippy::manual_assert,
            reason = "this is not an assertion but intentional testing behavior"
        )]
        if self.armer.fetch_and_disarm() {
            panic!("clone requested a panic");
        }
        Ok(Self {
            armer: Arc::clone(&self.armer),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// An armed `PanickyClone` panics exactly once on clone and the armer is
    /// disarmed afterwards; the disarmed instance clones silently again.
    #[test]
    fn armed_clone_panics_once_and_auto_disarms() {
        let src = PanickyClone::new();
        src.arm();

        let res = catch_unwind(AssertUnwindSafe(|| src.try_clone()));
        assert!(res.is_err(), "armed clone must panic");
        assert!(!src.is_armed(), "firing must auto-disarm the armer");

        let ok = src.try_clone().expect("disarmed clone must succeed");
        assert!(!ok.is_armed());
    }

    /// A never-armed `PanickyClone` clones silently, and clones share the
    /// source's armer: arming through one handle makes the other fire too.
    #[test]
    fn disarmed_clone_is_silent_and_shares_armer() {
        let src = PanickyClone::new();
        let cloned = src.try_clone().expect("disarmed clone must succeed");
        assert!(!cloned.is_armed());

        src.arm();
        let res = catch_unwind(AssertUnwindSafe(|| cloned.try_clone()));
        assert!(res.is_err(), "the shared armer must fire on any holder");
        assert!(!src.is_armed());
        assert!(!cloned.is_armed());
    }

    /// Two instances wired to one armer: the first clone to run fires and
    /// auto-disarms, so the second stays silent. The arming is consumed
    /// exactly once, and re-arming fires again.
    #[test]
    fn shared_armer_fires_once_per_arming() {
        let armer = Arc::new(PanicArmer::new());
        let a = PanickyClone::wired_to(armer.clone());
        let b = PanickyClone::wired_to(armer.clone());

        armer.arm();
        let res = catch_unwind(AssertUnwindSafe(|| a.try_clone()));
        assert!(res.is_err(), "armed clone must panic");
        assert!(!armer.is_armed(), "firing must auto-disarm the armer");
        let _ = b.try_clone().expect("disarmed clone must succeed");

        armer.arm();
        let res = catch_unwind(AssertUnwindSafe(|| b.try_clone()));
        assert!(res.is_err(), "re-armed clone must panic again");
        assert!(!armer.is_armed());
    }
}
