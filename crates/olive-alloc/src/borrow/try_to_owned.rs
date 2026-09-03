//! [`TryToOwned`]: a fallible analogue of [`alloc::borrow::ToOwned`].
//!
//! This modules declares the [`TryToOwned`] trait, which fallibly converts
//! borrowed references to owned types.

extern crate alloc;
use core::{borrow::Borrow, fmt};

use olive_core::alloc_errors::{AllocError, TryReserveError};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

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

/// Lift a [`TryCloneError`] into the identically-shaped [`TryToOwnedError`].
///
/// The two error enums carry the same variants (`Reserve`, `Alloc`, `Other`);
/// this lifts a clone failure into a to-owned failure without coupling the two
/// unrelated traits through a shared `From` impl. All payload types are `Copy`,
/// so the mapping reads through a reference.
fn clone_err_to_owned(err: &TryCloneError) -> TryToOwnedError {
    match err {
        TryCloneError::Reserve(e) => TryToOwnedError::Reserve(*e),
        TryCloneError::Alloc(e) => TryToOwnedError::Alloc(*e),
        TryCloneError::Other(msg) => TryToOwnedError::Other(msg),
    }
}

/// A fallible analogue of [`core::borrow::ToOwned`].
///
/// Types implementing this trait guarantee that constructing their owned variant
/// will not panic on allocation failure.
pub trait TryToOwned {
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

/// Blanket impl mirroring std's `impl<T: Clone> ToOwned for T`: any type that can
/// be fallibly cloned can be fallibly turned into its owned form, which is
/// itself. The `Owned = Self` associated value satisfies the `Borrow<Self>` bound
/// via std's reflexive `impl<T: ?Sized> Borrow<T> for T`.
impl<T: TryClone> TryToOwned for T {
    type Owned = Self;

    #[inline]
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        self.try_clone().map_err(|e| clone_err_to_owned(&e))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::vec::Vec;

    #[test]
    fn try_to_owned_error_conversions() {
        let e: TryToOwnedError = TryReserveError::new_capacity_overflow().into();
        assert!(matches!(e, TryToOwnedError::Reserve(_)));
        let e: TryToOwnedError = AllocError.into();
        assert!(matches!(e, TryToOwnedError::Alloc(_)));
    }

    #[test]
    fn blanket_impl_covers_try_clone_types() {
        // Primitives implement `TryClone` in olive-core, so they inherit
        // `TryToOwned` through the blanket impl defined above.
        let x: u32 = 42;
        let owned: u32 = x.try_to_owned().unwrap();
        assert_eq!(owned, 42);

        // A heap-backed collection (`Vec`) likewise inherits it via its own
        // `TryClone` impl.
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        v.try_push(3).unwrap();
        let owned: Vec<i32> = v.try_to_owned().unwrap();
        assert_eq!(owned.as_slice(), &[1, 2, 3]);
    }
}
