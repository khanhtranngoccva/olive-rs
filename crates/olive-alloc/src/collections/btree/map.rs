//! A fallible port of `alloc::collections::BTreeMap`.
//!
//! This implementation uses a reserve-and-commit architecture for insertion:
//! 1. **Probe**: walk the tree to determine which nodes will split (reads only).
//! 2. **Reserve**: allocate all needed nodes in one batch. On failure, roll back.
//! 3. **Commit**: perform the splits and promotions using reserved nodes (infallible).
//!
//! The three-phase machinery lives in [`super::entry`]; this module is the
//! public map surface and routes inserts through it.

use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::borrow::Borrow;
use crate::boxed::Box;
use crate::collections::btree::node::InternalNode;
use crate::vec::Vec;
use core::marker::PhantomData;
use core::mem;
use core::ptr;

use super::borrow::DormantMutRef;
use super::entry::{OccupiedEntry, VacantEntry};
use super::node::{Handle, LeafNode, NodeRef, Root, marker};

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
    pub(super) reserve_stack: Option<Vec<Box<InternalNode<K, V>, A>, A>>,
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
        // Fast path: check if key exists first.
        if let Some(existing) = self.get_mut(&key) {
            return Ok(Some(mem::replace(existing, value)));
        }

        // Empty map: create root leaf directly.
        if self.root.is_none() {
            self.insert_into_empty_map(key, value)?;
            return Ok(None);
        }

        // Probe the tree for the vacant position, then insert.
        let vacant = try_probe(self, key).expect("key absence was verified by get_mut above");
        vacant.insert(value)?;
        Ok(None)
    }
    /// Gets the mutable reference to the value corresponding to the key.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut node = self.root.as_mut()?.borrow_valmut();
        loop {
            if node.height() == 0 {
                // At leaf level - search for exact match.
                let leaf_ptr = NodeRef::as_leaf_ptr(&node);
                let edge_idx = find_edge_index_generic(leaf_ptr, key);
                if edge_idx < node.len() {
                    let keys = node.keys();
                    if keys[edge_idx].borrow() == key {
                        return Some(unsafe { node.into_key_val_mut_at(edge_idx).1 });
                    }
                }
                return None;
            }

            // Internal node: check if the key matches a separator stored here.
            let leaf_ptr = NodeRef::as_leaf_ptr(&node);
            let edge_idx = find_edge_index_generic(leaf_ptr, key);
            if edge_idx < node.len() {
                let keys = node.keys();
                if keys[edge_idx].borrow() == key {
                    return Some(unsafe { node.into_key_val_mut_at(edge_idx).1 });
                }
            }
            // Descend to the appropriate child.
            let internal_node = unsafe { node.cast_to_internal_unchecked() };
            let edge_handle = unsafe { Handle::new_edge(internal_node, edge_idx) };
            node = edge_handle.descend();
        }
    }

    /// Gets the immutable reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut node = self.root.as_ref()?.reborrow();
        loop {
            if node.height() == 0 {
                // At leaf level - search for exact match.
                let leaf_ptr = NodeRef::as_leaf_ptr(&node);
                let edge_idx = find_edge_index_generic(leaf_ptr, key);
                if edge_idx < node.len() {
                    let keys = node.keys();
                    if keys[edge_idx].borrow() == key {
                        return Some(unsafe { node.into_key_val_at(edge_idx).1 });
                    }
                }
                return None;
            }

            // Internal node: check if the key matches a separator stored here.
            let leaf_ptr = NodeRef::as_leaf_ptr(&node);
            let edge_idx = find_edge_index_generic(leaf_ptr, key);
            if edge_idx < node.len() {
                let keys = node.keys();
                if keys[edge_idx].borrow() == key {
                    return Some(unsafe { node.into_key_val_at(edge_idx).1 });
                }
            }
            // Descend to the appropriate child.
            let internal_node = unsafe { node.cast_to_internal_unchecked() };
            let edge_handle = unsafe { Handle::new_edge(internal_node, edge_idx) };
            node = edge_handle.descend();
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

// ── Vacant-entry probing ──────────────────────────────────────────────────────

/// Walks the tree to locate the vacant leaf edge where `key` would be inserted,
/// packaging it into a [`VacantEntry`] backed by a dormant borrow of `map`.
///
/// Returns `None` if the key is already present (callers that have verified
/// absence via `get_mut` should treat this as unreachable).
fn try_probe<'a, K: 'a + Ord, V: 'a, A: AllocatorTryClone>(
    map: &'a mut BTreeMap<K, V, A>,
    key: K,
) -> Option<VacantEntry<'a, K, V, A>> {
    let root = map.root.as_mut()?;
    // Capture a unique borrow of the map, then immediately reborrow it so we can
    // hand out mutable node references tied to `'a` while keeping the original
    // borrow dormant for the entry's lifetime.
    let (map_ref, dormant_map) = DormantMutRef::new(map);
    let mut node = map_ref.root.as_mut()?.borrow_mut();
    loop {
        let leaf_ptr = NodeRef::as_leaf_ptr(&node);
        let edge_idx = find_edge_index_generic(leaf_ptr, &key);
        // If the key matches an existing separator, the slot is occupied.
        if edge_idx < node.len() {
            let keys = node.keys();
            if keys[edge_idx].borrow() == key {
                return None;
            }
        }
        if node.height() == 0 {
            // At the leaf: record the vacant edge position.
            let handle = unsafe { Handle::new_edge(node, edge_idx) };
            return Some(VacantEntry {
                key,
                handle: Some(handle),
                dormant_map,
                alloc: map_ref.alloc.clone(),
                _marker: PhantomData,
            });
        }
        // Descend to the appropriate child.
        let internal_node = unsafe { node.cast_to_internal_unchecked() };
        let edge_handle = unsafe { Handle::new_edge(internal_node, edge_idx) };
        node = edge_handle.descend();
    }
}

impl<'a, K: 'a + Ord, V: 'a, A: AllocatorTryClone> VacantEntry<'a, K, V, A> {
    /// Inserts `value` at the vacant slot, returning the newly occupied entry.
    pub(super) fn insert(self, value: V) -> Result<OccupiedEntry<'a, K, V, A>, AllocError> {
        self.try_insert_entry(value).map_err(|(_, _, e)| e)
    }
}

// ── Private helper methods ────────────────────────────────────────────────────

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Inserts a key-value pair when the map is empty by creating a new leaf root.
    fn insert_into_empty_map(&mut self, key: K, value: V) -> Result<(), AllocError> {
        let owned_root = NodeRef::<marker::Owned, K, V, marker::Leaf>::new_leaf(
            self.alloc
                .try_clone()
                .map_err(try_clone_err_to_alloc_error)?,
        )?;

        let leaf_ptr = owned_root.node.as_ptr();
        unsafe {
            (*leaf_ptr).keys[0].write(key);
            (*leaf_ptr).vals[0].write(value);
            (*leaf_ptr).len = 1;
        }

        self.root = Some(unsafe {
            NodeRef::<marker::Owned, K, V, marker::LeafOrInternal> {
                height: 0,
                node: ptr::read(&owned_root.node),
                _marker: PhantomData,
            }
        });
        self.length = 1;

        Ok(())
    }
}

// ── Edge-index helpers ────────────────────────────────────────────────────────

/// Finds the edge index in a node (by raw pointer) where the given key should be inserted.
pub(super) fn find_edge_index_generic<K, V, Q>(leaf_ptr: *mut LeafNode<K, V>, key: &Q) -> usize
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    let len = unsafe { (*leaf_ptr).len as usize };
    // SAFETY: the first `len` elements of `keys` are initialized.
    let keys = unsafe { core::slice::from_raw_parts((*leaf_ptr).keys.as_ptr().cast::<K>(), len) };
    keys.partition_point(|k| k.borrow() < key)
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
