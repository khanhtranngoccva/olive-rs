//! A fallible port of `alloc::collections::BTreeMap`.
//!
//! This implementation uses a reserve-and-commit architecture for insertion.
use olive_core::marker::PhantomData;
use olive_core::mem::ManuallyDrop;
use olive_core::ptr;

use super::borrow::DormantMutRef;
use super::entry::{Entry, OccupiedEntry};
use super::node::{self, Root};
use super::search::SearchResult;
use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::borrow::Borrow;
use crate::boxed::Box;
use crate::collections::btree::node::InternalNode;
use crate::vec::Vec;

/// Minimum number of key-value pairs a non-root node must retain after removal.
/// A node with fewer than this is underfull and needs rebalancing.
pub(super) const MIN_LEN: usize = node::MIN_LEN_AFTER_SPLIT;

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
    pub(super) alloc: ManuallyDrop<A>,
    /// Stack of reserved internal nodes awaiting commitment.
    /// During the reserve phase, newly allocated internal nodes are pushed here.
    /// The commit phase pops them as it climbs the tree.
    #[allow(clippy::type_complexity)]
    pub(super) reserve_stack: ManuallyDrop<Option<Vec<Box<InternalNode<K, V>, A>, A>>>,
}

impl<K, V, A: AllocatorTryClone> Drop for BTreeMap<K, V, A> {
    fn drop(&mut self) {
        // SAFETY: Mirrors std.
        // All fields are either trivially copyable or are stored in ManuallyDrop.
        drop(unsafe { ptr::read(self) }.into_iter())
    }
}

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Attempts to create an empty `BTreeMap` with the given allocator.
    pub fn new_in(alloc: A) -> Self {
        let alloc_clone = alloc.try_clone().map_err(try_clone_err_to_alloc_error).ok();
        Self {
            root: None,
            length: 0,
            alloc: ManuallyDrop::new(alloc),
            // Lazily clones the alloc for one more chance.
            reserve_stack: ManuallyDrop::new(
                alloc_clone.map(|alloc_clone| Vec::new_in(alloc_clone)),
            ),
        }
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Like [`entry`](Self::entry) but accepts a borrowed key via `Borrow`.
    /// Only useful for lookups that don't need to insert (e.g. `remove`).
    fn entry_ref<Q>(&mut self, key: &Q) -> Result<Entry<'_, K, V, A>, ()>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        match map.root {
            None => Err(()), // empty map, nothing to remove
            Some(ref mut root) => match root.borrow_mut().search_tree(key) {
                SearchResult::Found(handle) => Ok(Entry::Occupied(OccupiedEntry {
                    handle,
                    dormant_map,
                    _marker: PhantomData,
                })),
                SearchResult::GoDown(_) => Err(()), // not found
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
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.entry_ref(key) {
            Ok(Entry::Occupied(occupied)) => Some(occupied.remove_entry().1),
            _ => None,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_single() {
        let mut map = BTreeMap::new_in(Global);
        assert!(map.is_empty());

        map.try_insert(1, "one").unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn insert_multiple_no_split() {
        let mut map = BTreeMap::new_in(Global);
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
        let mut map = BTreeMap::new_in(Global);
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
        let mut map = BTreeMap::new_in(Global);
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
        let mut map = BTreeMap::new_in(Global);
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
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, "first").unwrap();
        let old = map.try_insert(1, "second").unwrap();
        assert_eq!(old, Some("first"));
        assert_eq!(map.get(&1), Some(&"second"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn get_mut_returns_correct_value() {
        let mut map = BTreeMap::new_in(Global);
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
        let mut map = BTreeMap::new_in(Global);
        for i in (0..100).rev() {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    // ── Deletion tests ────────────────────────────────────────────────────────

    #[test]
    fn remove_single_element() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.remove(&1), None);
    }

    #[test]
    fn remove_nonexistent_key() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        assert_eq!(map.remove(&999), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn remove_from_leaf_no_rebalance() {
        let mut map = BTreeMap::new_in(Global);
        // Fill a leaf to capacity (11 keys) then remove one — still ≥ MIN_LEN.
        for i in 0..11 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.remove(&5), Some(50));
        assert_eq!(map.len(), 10);
        for i in 0..11 {
            if i != 5 {
                assert_eq!(map.get(&i), Some(&(i * 10)), "missing key {}", i);
            } else {
                assert_eq!(map.get(&i), None);
            }
        }
    }

    #[test]
    fn remove_triggers_steal() {
        let mut map = BTreeMap::new_in(Global);
        // Build a tree with multiple leaves, then remove enough from one leaf
        // to force a steal from a sibling.
        for i in 0..30u32 {
            map.try_insert(i, i).unwrap();
        }
        // Remove the middle key of the leftmost leaf region to trigger rebalance.
        assert_eq!(map.remove(&2), Some(2));
        assert_eq!(map.len(), 29);
        for i in 0..30u32 {
            if i != 2 {
                assert_eq!(map.get(&i), Some(&i), "missing key {}", i);
            }
        }
    }

    #[test]
    fn remove_triggers_merge_and_root_shrink() {
        let mut map = BTreeMap::new_in(Global);
        // Build a multi-level tree, then drain it down to empty.
        for i in 0..80u32 {
            map.try_insert(i, i).unwrap();
        }
        let initial_height = map.root.as_ref().map_or(0, |r| r.height());
        assert!(initial_height >= 1, "expected multi-level tree");

        // Remove all elements one by one.
        for i in 0..80u32 {
            assert_eq!(map.remove(&i), Some(i), "failed to remove key {}", i);
        }
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn remove_all_keys_random_order() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 200;
        for i in 0..N {
            map.try_insert(i, i * 7).unwrap();
        }
        // Remove in descending order (worst-case for B-tree: always hits rightmost leaf).
        for i in (0..N).rev() {
            assert_eq!(map.remove(&i), Some(i * 7), "failed to remove key {}", i);
            assert_eq!(
                map.len(),
                i as usize,
                "length mismatch after removing key {}",
                i
            );
        }
        assert!(map.is_empty());
    }

    #[test]
    fn remove_interleaved_with_insert() {
        let mut map = BTreeMap::new_in(Global);
        // Interleave inserts and removes to stress the rebalancing logic.
        for round in 0..5u32 {
            for i in 0..20 {
                map.try_insert(round * 100 + i, i).unwrap();
            }
            for i in (0..20).step_by(2) {
                map.remove(&(round * 100 + i));
            }
        }
        // Verify remaining keys are correct.
        for round in 0..5u32 {
            for i in 0..20 {
                if i % 2 == 1 {
                    assert_eq!(map.get(&(round * 100 + i)), Some(&i));
                } else {
                    assert_eq!(map.get(&(round * 100 + i)), None);
                }
            }
        }
    }

    #[test]
    fn remove_preserves_sorted_invariant() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            map.try_insert(i, i).unwrap();
        }
        // Remove every third key.
        for i in (0..50).step_by(3) {
            map.remove(&i);
        }
        // Walk the map via get and verify sorted order is maintained.
        let mut prev = None;
        for i in 0..50u32 {
            if i % 3 != 0 {
                assert_eq!(map.get(&i), Some(&i));
                if let Some(p) = prev {
                    assert!(p < i, "sorted invariant violated: {} !< {}", p, i);
                }
                prev = Some(i);
            }
        }
    }
}
