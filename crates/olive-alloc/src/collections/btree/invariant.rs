//! Test-only structural invariant checks for the B-tree.
//!
//! These functions walk the tree via immutable borrows and assert properties
//! that must hold after every mutation completes. They are intentionally
//! decoupled from the rebalancing implementation: they observe the resulting
//! structure, not the mechanism that produced it.

use core::fmt::Debug;

use super::map::BTreeMap;
use crate::alloc::AllocatorTryClone;

/// Asserts structural invariants on the entire tree rooted at `map`:
/// - Min-length: the root is checked against a relaxed bound (1 if internal,
///   0 if leaf); all non-root nodes are checked against `MIN_LEN`.
/// - Length consistency: the cached `map.length` field matches the actual
///   number of key-value pairs counted by walking every node.
///
/// Panics on the first violation with a descriptive message.
pub(crate) fn check_tree_invariant<K, V, A: AllocatorTryClone>(map: &BTreeMap<K, V, A>) {
    if let Some(root) = map.root.as_ref() {
        let min_len = if root.height() > 0 { 1 } else { 0 };
        root.reborrow().assert_min_len(min_len);

        let actual = root.reborrow().calc_length();
        assert_eq!(
            actual, map.length,
            "length field ({}) does not match actual entry count ({})",
            map.length, actual
        );
    } else {
        assert_eq!(
            map.length, 0,
            "empty map must have length 0, found {}",
            map.length
        );
    }
}

/// Asserts that all keys in the tree appear in strictly ascending order by
/// walking the public iterator. Panics on the first out-of-order pair.
pub(crate) fn check_ascending_keys<K, V, A: AllocatorTryClone>(map: &BTreeMap<K, V, A>)
where
    K: Ord + Debug,
{
    let iter = map.iter();
    let mut last_key: Option<&K> = None;
    for (key, _) in iter {
        if let Some(prev) = last_key {
            assert!(
                prev < key,
                "ascending order violated: {:?} >= {:?}",
                prev,
                key
            );
        }
        last_key = Some(key);
    }
}

