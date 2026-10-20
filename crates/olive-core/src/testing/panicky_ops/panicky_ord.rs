//! The [`PanickyOrd`] newtype: an ordering that panics while armed.

extern crate std;

use super::PanicArmer;
use core::ops::{Deref, DerefMut};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use std::sync::Arc;

/// An ordered newtype over `T` whose comparisons panic while an attached
/// [`PanicArmer`] is armed.
///
/// Firing atomically *auto-disarms* the armer with a single fetch-and-set swap by
/// the time the panic starts, so drops that run during the unwind observe the
/// disarmed state instead of aborting the process with a second panic.
#[derive(Debug)]
pub struct PanickyOrd<T> {
    /// The wrapped value, ordered by delegation.
    pub value: T,
    armer: Option<Arc<PanicArmer>>,
}

impl<T> PanickyOrd<T> {
    /// Wraps `value` with no armer: its comparisons never panic.
    pub const fn calm(value: T) -> Self {
        Self { value, armer: None }
    }

    /// Wraps `value` with `armer`: while that armer is armed, every
    /// comparison touching this value panics.
    pub fn new(value: T, armer: Arc<PanicArmer>) -> Self {
        Self {
            value,
            armer: Some(armer),
        }
    }
}

impl<T: Ord> Ord for PanickyOrd<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Disarm *before* panicking: drops that run during the unwind must
        // observe the disarmed state, or the process aborts (double panic).
        let self_fires = self
            .armer
            .as_deref()
            .is_some_and(PanicArmer::fetch_and_disarm);
        // We deliberately disarm both armers to avoid aborting.
        let other_fires = other
            .armer
            .as_deref()
            .is_some_and(PanicArmer::fetch_and_disarm);
        #[allow(
            clippy::manual_assert,
            reason = "this is not an assertion but intentional testing behavior"
        )]
        if self_fires || other_fires {
            panic!("comparison requested a panic");
        }
        self.value.cmp(&other.value)
    }
}

impl<T: Ord> PartialOrd for PanickyOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialEq> PartialEq for PanickyOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Eq> Eq for PanickyOrd<T> {}

impl<T> Deref for PanickyOrd<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for PanickyOrd<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: TryClone> TryClone for PanickyOrd<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self {
            value: self.value.try_clone()?,
            armer: self.armer.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// A firing comparison auto-disarms the armer, so drops running during
    /// the unwind cannot fire it a second time (which would abort). The
    /// disarmed comparison is silent again, and re-arming fires once more.
    #[test]
    fn firing_auto_disarms() {
        let armer = Arc::new(PanicArmer::new());
        let a = PanickyOrd::new(1, Arc::clone(&armer));
        let b = PanickyOrd::calm(1);

        armer.arm();
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = a.cmp(&b);
        }));
        assert!(res.is_err(), "armed comparison must panic");
        assert!(!armer.is_armed(), "firing must auto-disarm the armer");

        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);

        armer.arm();
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = b.cmp(&a);
        }));
        assert!(
            res.is_err(),
            "the armed element in the other operand position must fire a panic"
        );
        assert!(!armer.is_armed());
    }
}
