//! A payload whose `try_clone` can be switched on and off from outside.

extern crate std;

use super::FailArmer;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use std::sync::Arc;

/// A wrapper around a value whose `try_clone` fails while its [`FailArmer`] is
/// armed.
#[derive(Debug, Clone)]
pub struct MaybeUnclonable<T> {
    /// The wrapped payload.
    pub inner: T,
    /// The shared switch controlling whether `try_clone` succeeds.
    pub armer: Arc<FailArmer>,
}

impl<T> MaybeUnclonable<T> {
    /// A value whose `try_clone` is controlled by `armer`.
    pub fn new(inner: T, armer: Arc<FailArmer>) -> Self {
        Self { inner, armer }
    }

    /// A value whose `try_clone` always fails: it carries its own armed
    /// [`FailArmer`], unshared with anything else.
    pub fn unclonable(inner: T) -> Self {
        Self::new(inner, Arc::new(FailArmer::armed()))
    }

    /// A value whose `try_clone` always succeeds: it carries its own disarmed
    /// [`FailArmer`], unshared with anything else.
    pub fn clonable(inner: T) -> Self {
        Self::new(inner, Arc::new(FailArmer::new()))
    }
}

impl<T: PartialEq> PartialEq for MaybeUnclonable<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Eq> Eq for MaybeUnclonable<T> {}

impl<T: Ord> PartialOrd for MaybeUnclonable<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for MaybeUnclonable<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl<T: TryClone> TryClone for MaybeUnclonable<T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.armer.is_armed() {
            return Err(TryCloneError::Other("unclonable"));
        }
        Ok(Self {
            inner: self.inner.try_clone()?,
            // The clone shares the switch, so arming reaches both.
            armer: self.armer.clone(),
        })
    }
}
