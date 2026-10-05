//! A fallible port of `alloc::collections::BTreeMap`.

use core::fmt;
use olive_core::alloc::AllocError;
use olive_core::try_traits::try_clone::TryCloneError;
use olive_core::try_traits::try_default::TryDefaultError;

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
mod set;
mod set_val;
mod split;
mod traits;
mod merge_iter;

pub use entry::Entry;
pub use extract_if::ExtractIf;
pub use iter::{IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys, Values, ValuesMut};
pub use map::BTreeMap;
pub use set::{
    BTreeSet, Entry as BTreeSetEntry, ExtractIf as BTreeSetExtractIf, IntoIter as BTreeSetIntoIter,
    Iter as BTreeSetIter, OccupiedEntry as BTreeSetOccupiedEntry,
    VacantEntry as BTreeSetVacantEntry,
};

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

/// Error returned by [`Entry::or_try_default`](entry::Entry::or_try_default).
///
/// The default value may fail to be constructed, or the subsequent insertion
/// may fail due to an allocation error.
#[derive(Clone, PartialEq, Eq)]
pub enum TryBTreeMapEntryWithDefaultError {
    /// The default value failed to be constructed.
    Default(TryDefaultError),
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
}

impl fmt::Debug for TryBTreeMapEntryWithDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default(e) => f
                .debug_tuple("TryBTreeMapEntryWithDefaultError::Default")
                .field(e)
                .finish(),
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeMapEntryWithDefaultError::Alloc")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBTreeMapEntryWithDefaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default(e) => write!(f, "BTreeMap entry operation failed: {e}"),
            Self::Alloc(_) => write!(
                f,
                "BTreeMap entry operation failed: memory allocation failed"
            ),
        }
    }
}

impl core::error::Error for TryBTreeMapEntryWithDefaultError {}

impl From<TryDefaultError> for TryBTreeMapEntryWithDefaultError {
    #[inline]
    fn from(err: TryDefaultError) -> Self {
        Self::Default(err)
    }
}

impl From<AllocError> for TryBTreeMapEntryWithDefaultError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// Error returned by [`Entry::or_try_insert_with`](entry::Entry::or_try_insert_with)
/// and [`Entry::or_try_insert_with_key`](entry::Entry::or_try_insert_with_key).
///
/// The closure may fail, or the subsequent insertion may fail due to an
/// allocation error.
#[derive(Clone, PartialEq, Eq)]
pub enum TryBTreeMapEntryWithError<E> {
    /// The closure failed with this error.
    Closure(E),
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
}

impl<E: fmt::Debug> fmt::Debug for TryBTreeMapEntryWithError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closure(e) => f
                .debug_tuple("TryBTreeMapEntryWithError::Closure")
                .field(e)
                .finish(),
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeMapEntryWithError::Alloc")
                .field(e)
                .finish(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for TryBTreeMapEntryWithError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closure(e) => write!(f, "BTreeMap entry operation failed: {e}"),
            Self::Alloc(_) => write!(
                f,
                "BTreeMap entry operation failed: memory allocation failed"
            ),
        }
    }
}

impl<E: core::error::Error> core::error::Error for TryBTreeMapEntryWithError<E> {}

impl<E> From<AllocError> for TryBTreeMapEntryWithError<E> {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

/// Error returned by fallible [`BTreeMap`](map::BTreeMap) operations that
/// enforce a uniqueness check before inserting (including but not limited to
/// insertion).
///
/// There are two ways these operations can fail:
/// - the key is already present in the map ([`Self::KeyExists`]),
/// - or an allocation fails while restructuring the tree to insert a
///   previously-absent key ([`Self::Alloc`]).
#[derive(Clone)]
pub enum TryBTreeMapUniqueError {
    /// The key was already present in the map, so nothing was inserted.
    KeyExists,
    /// An allocation during tree restructuring failed.
    Alloc(AllocError),
}

impl fmt::Debug for TryBTreeMapUniqueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyExists => f.debug_tuple("TryBTreeMapUniqueError::KeyExists").finish(),
            Self::Alloc(e) => f
                .debug_tuple("TryBTreeMapUniqueError::Alloc")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBTreeMapUniqueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyExists => write!(f, "BTreeMap unique operation failed: key already exists"),
            Self::Alloc(_) => write!(
                f,
                "BTreeMap unique operation failed: memory allocation failed"
            ),
        }
    }
}

impl core::error::Error for TryBTreeMapUniqueError {}

impl From<AllocError> for TryBTreeMapUniqueError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}
