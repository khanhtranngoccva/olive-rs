//! A fallible port of `alloc::collections::BTreeMap`.
//!
//! This implementation uses a reserve-and-commit architecture for insertion.
use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::borrow::Borrow;
use crate::boxed::Box;
use crate::collections::btree::node::InternalNode;
use crate::vec::Vec;
use core::marker::PhantomData;

use super::borrow::DormantMutRef;
use super::entry::{Entry, OccupiedEntry, VacantEntry};
use super::node::Root;
use super::search::SearchResult;

/// Converts a `TryCloneError` to an `AllocError`.
pub(super) fn try_clone_err_to_alloc_error(
    _e: olive_core::try_traits::try_clone::TryCloneError,
) -> AllocError {
    AllocError
}

/// A B-tree based implementation of a ordered map, similar to std's BTreeMap.
pub struct BTreeMap<K, V, A: AllocatorTryClone = Global> {
    pub(super) root: Option<Root<K, V>>,
    pub(super) length: usize,
    pub(super) alloc: A,
    /// Stack of reserved internal nodes awaiting commitment.
    /// During the reserve phase, newly allocated internal nodes are pushed here.
    /// The commit phase pops them as it climbs the tree.
    #[allow(clippy::type_complexity)]
    pub(super) reserve_stack: Option<Vec<Box<InternalNode<K, V>, A>, A>>,
}

impl<K, V, A: AllocatorTryClone> Drop for BTreeMap<K, V, A> {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            let alloc = self
                .alloc
                .try_clone()
                .expect("allocator clone must succeed during drop");
            root.into_dying().drop_tree(alloc);
        }
        // reserve_stack is dropped normally (Vec handles freeing its elements).
    }
}

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Attempts to create an empty `BTreeMap` with the given allocator.
    pub fn new_in(alloc: A) -> Result<Self, AllocError> {
        let alloc_clone = alloc.try_clone().map_err(try_clone_err_to_alloc_error).ok();
        Ok(Self {
            root: None,
            length: 0,
            alloc,
            // Lazily clones the alloc for one more chance.
            reserve_stack: alloc_clone.map(|alloc_clone| Vec::new_in(alloc_clone)),
        })
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Inserts a key-value pair into the map, attempting allocation as needed.
    ///
    /// If the key already existed, the old value is returned.
    /// Otherwise, `None` is returned.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, AllocError> {
        match self.entry(key) {
            Entry::Occupied(mut occ) => Ok(Some(occ.insert(value))),
            Entry::Vacant(vac) => vac
                .try_insert_entry(value)
                .map(|_| None)
                .map_err(|(_, _, e)| e),
        }
    }

    /// Gets an [`Entry`] to a single entry in the map, which may either be
    /// occupied or vacant.
    ///
    /// This is the standard entry API, mirroring `std::collections::BTreeMap::entry`.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, A> {
        let (map, dormant_map) = DormantMutRef::new(self);
        match map.root {
            None => Entry::Vacant(VacantEntry {
                key,
                handle: None,
                dormant_map,
                _marker: PhantomData,
            }),
            Some(ref mut root) => match root.borrow_mut().search_tree(&key) {
                SearchResult::Found(handle) => Entry::Occupied(OccupiedEntry {
                    handle,
                    dormant_map,
                    _marker: PhantomData,
                }),
                SearchResult::GoDown(handle) => Entry::Vacant(VacantEntry {
                    key,
                    handle: Some(handle),
                    dormant_map,
                    _marker: PhantomData,
                }),
            },
        }
    }
    /// Gets the mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.root.as_mut()?.borrow_valmut();
        match node.search_tree(key) {
            SearchResult::Found(kv) => Some(kv.into_kv_valmut().1),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Gets the immutable reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let node = self.root.as_ref()?.reborrow();
        match node.search_tree(key) {
            SearchResult::Found(kv) => Some(kv.into_kv().1),
            SearchResult::GoDown(_) => None,
        }
    }

    /// Removes a key from the map, returning the value if present.
    pub fn remove<Q>(&mut self, _key: &Q) -> Option<V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // TODO: Implement removal logic
        unimplemented!("remove")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_single() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        assert!(map.is_empty());

        map.try_insert(1, "one").unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn insert_multiple_no_split() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        // CAPACITY is 11, so 11 inserts fit in one leaf without splitting.
        for i in 0..11 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 11);
        for i in 0..11 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_triggers_leaf_split() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        // 12th insert should trigger a leaf split.
        for i in 0..12 {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 12);
        let mut missing_count = 0;
        let mut first_missing = 0;
        for i in 0..12 {
            if map.get(&i).is_none() {
                missing_count += 1;
                if missing_count == 1 {
                    first_missing = i;
                }
            }
        }
        let h = map.root.as_ref().map_or(0, |r| r.height());
        assert_eq!(
            missing_count, 0,
            "Missing {} keys after leaf split, first missing={}, height={}",
            missing_count, first_missing, h
        );
    }

    #[test]
    fn insert_triggers_root_growth() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        // With CAPACITY=11, the root (internal) splits after ~12 leaf splits.
        // Each leaf holds ~6 keys on average, so ~72 inserts fills the root.
        // Insert 80 to guarantee root growth.
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert_eq!(map.len(), 80);
        for i in 0..80u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)), "missing key {}", i);
        }
    }

    #[test]
    fn insert_many_triggers_multi_level_splits() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        // Enough inserts to force multiple levels of splits.
        for i in 0..120u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert_eq!(map.len(), 120);
        for i in 0..120u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)), "missing key {}", i);
        }
    }

    #[test]
    fn insert_overwrite_existing_key() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        map.try_insert(1, "first").unwrap();
        let old = map.try_insert(1, "second").unwrap();
        assert_eq!(old, Some("first"));
        assert_eq!(map.get(&1), Some(&"second"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn get_mut_returns_correct_value() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        map.try_insert(5, 50).unwrap();
        map.try_insert(10, 100).unwrap();

        if let Some(v) = map.get_mut(&5) {
            *v = 55;
        }
        assert_eq!(map.get(&5), Some(&55));
        assert_eq!(map.get(&10), Some(&100));
    }

    #[test]
    fn reverse_order_insertion() {
        let mut map = BTreeMap::new_in(crate::alloc::Global).unwrap();
        for i in (0..100).rev() {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }
}
