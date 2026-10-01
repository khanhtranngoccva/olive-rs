//! Fallible ports of the `alloc` crate's collection types.
mod btree;
pub mod vec_deque;

/// A fallible port of `alloc::collections::BTreeMap`.
pub mod btree_map {
    pub use super::btree::{
        BTreeMap, Entry, ExtractIf, IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys,
        TryBTreeMapEntryWithDefaultError, TryBTreeMapEntryWithError, TryBTreeMapWithCloneError,
        Values, ValuesMut,
    };
}
