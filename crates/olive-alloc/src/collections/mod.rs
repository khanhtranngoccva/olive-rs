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
        BTreeSet, BTreeSetDifference as Difference, BTreeSetEntry as Entry,
        BTreeSetExtractIf as ExtractIf, BTreeSetIntersection as Intersection,
        BTreeSetIntoIter as IntoIter, BTreeSetIter as Iter,
        BTreeSetOccupiedEntry as OccupiedEntry, BTreeSetSymmetricDifference as SymmetricDifference,
        BTreeSetUnion as Union, BTreeSetVacantEntry as VacantEntry,
    };
}
