//! The [`PanickyOrd`] newtype: an ordering that panics while armed.

extern crate std;

use crate::test_helpers::PanicArmer;
use core::ops::{Deref, DerefMut};
use std::sync::Arc;

/// An ordered newtype over `T` whose comparisons panic while an attached
/// [`PanicArmer`] is armed.
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

    /// Whether a comparison involving this value would panic right now.
    fn is_armed(&self) -> bool {
        self.armer.as_deref().is_some_and(PanicArmer::is_armed)
    }
}

impl<T: Ord> Ord for PanickyOrd<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        if self.is_armed() || other.is_armed() {
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
