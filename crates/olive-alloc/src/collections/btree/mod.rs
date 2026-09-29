//! A fallible port of `alloc::collections::BTreeMap`.

use core::fmt;
use olive_core::alloc::AllocError;
use olive_core::try_traits::try_clone::TryCloneError;

mod borrow;
mod construction;
mod entry;
mod extract_if;
mod fix;
mod insertion;
#[cfg(test)]
mod invariant;
mod iter;
mod map;
mod mem;
mod mutation;
mod navigate;
mod node;
mod query;
mod remove;
mod scratch;
mod search;
mod set_val;
mod split;
mod traits;

pub use extract_if::ExtractIf;
pub use iter::{IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys, Values, ValuesMut};
pub use map::BTreeMap;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible BTreeMap operations that may both allocate and
/// clone elements.
///
/// Covers [`TryExtendFromSlice`](olive_core::try_traits::try_extend::TryExtendFromSlice)
/// for `BTreeMap` — any operation whose failure modes are limited to an
/// allocation ([`AllocError`](crate::alloc::AllocError)) or a key/value clone
/// failure ([`TryCloneError`]).
#[derive(Clone, PartialEq, Eq)]
pub enum TryBTreeMapWithCloneError {
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
    /// A key or value clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryBTreeMapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeMapWithCloneError::Alloc")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryBTreeMapWithCloneError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBTreeMapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alloc(_) => write!(f, "BTreeMap operation failed: memory allocation failed"),
            Self::Clone(e) => write!(f, "BTreeMap operation failed: {e}"),
        }
    }
}

impl core::error::Error for TryBTreeMapWithCloneError {}

impl From<AllocError> for TryBTreeMapWithCloneError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

impl From<TryCloneError> for TryBTreeMapWithCloneError {
    #[inline]
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}
