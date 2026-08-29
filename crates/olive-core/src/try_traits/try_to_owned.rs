//! [`TryToOwned`]: a fallible analogue of [`core::borrow::ToOwned`].

use crate::alloc_errors::{AllocError, TryReserveError};
use core::{borrow::Borrow, fmt};

/// Error returned by [`TryToOwned::try_to_owned`].
#[derive(Clone, PartialEq, Eq)]
pub enum TryToOwnedError {
    /// A capacity reservation on a collection failed (overflow or OOM).
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. a leaf
    /// allocation such as a `Box`, `Arc`, or `Rc` node).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryToOwnedError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryToOwnedError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryToOwnedError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "to_owned failed: {e}"),
            Self::Alloc(_) => write!(f, "to_owned failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "to_owned failed: {msg}"),
        }
    }
}

impl core::error::Error for TryToOwnedError {}

impl From<TryReserveError> for TryToOwnedError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryToOwnedError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// A fallible analogue of [`core::borrow::ToOwned`].
///
/// Types implementing this trait guarantee that constructing their owned variant
/// will not panic on allocation failure.
pub trait TryToOwned: Sized {
    /// The owned type produced by [`Self::try_to_owned`].
    type Owned: Borrow<Self>;

    /// Construct the owned version of `self`, falling back to an error on
    /// allocation failure rather than panicking.
    ///
    /// # Errors
    ///
    /// Returns [`TryToOwnedError`] if a capacity reservation or allocation fails.
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError>;
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn try_to_owned_error_conversions() {
        let e: TryToOwnedError = TryReserveError::new_capacity_overflow().into();
        assert!(matches!(e, TryToOwnedError::Reserve(_)));
        let e: TryToOwnedError = AllocError.into();
        assert!(matches!(e, TryToOwnedError::Alloc(_)));
    }
}
