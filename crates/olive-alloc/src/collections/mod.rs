//! Fallible ports of the `alloc` crate's collection types.
mod btree;
pub mod vec_deque;

/// A fallible port of `alloc::collections::BTreeMap`.
pub mod btree_map {
    pub use super::btree::{
        BTreeMap, Entry, ExtractIf, IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys,
        TryBTreeMapEntryWithDefaultError, TryBTreeMapEntryWithError, TryBTreeMapUniqueError,
        TryBTreeMapWithCloneError, Values, ValuesMut,
    };
}

/// A fallible port of `alloc::collections::BTreeSet`.
pub mod btree_set {
    pub use super::btree::{
        BTreeSet, BTreeSetEntry as Entry, BTreeSetExtractIf as ExtractIf,
        BTreeSetIntoIter as IntoIter, BTreeSetIter as Iter, BTreeSetOccupiedEntry as OccupiedEntry,
        BTreeSetVacantEntry as VacantEntry,
    };
}
