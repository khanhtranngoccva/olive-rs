//! Construction methods for [`BTreeMap`].

use olive_core::mem::ManuallyDrop;

use super::map::BTreeMap;
use crate::alloc::{AllocatorTryClone, Global};

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Attempts to create an empty `BTreeMap` with the given allocator.
    pub fn new_in(alloc: A) -> Self {
        Self {
            root: None,
            length: 0,
            alloc: ManuallyDrop::new(alloc),
        }
    }
}

impl<K: Ord, V> BTreeMap<K, V, Global> {
    /// Creates an empty `BTreeMap` backed by the global allocator.
    ///
    /// This is a convenience wrapper around [`Self::new_in`] that uses
    /// [`Global`], mirroring std's `BTreeMap::new()`.
    #[inline]
    pub fn new() -> Self {
        Self::new_in(Global)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn new_in_creates_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn new_and_new_in_are_equivalent() {
        let a: BTreeMap<u32, u32> = BTreeMap::new();
        let b: BTreeMap<u32, u32> = BTreeMap::new_in(Global);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.is_empty(), b.is_empty());
    }

    #[test]
    fn new_map_accepts_inserts_immediately() {
        let mut map = BTreeMap::new();
        for i in 0..5u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 5);
        for i in 0..5u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn new_in_map_accepts_inserts_immediately() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 5);
        for i in 0..5u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn new_map_allocator_is_global() {
        let map: BTreeMap<i32, i32> = BTreeMap::new();
        let _alloc: &Global = map.allocator();
    }

    #[test]
    fn new_in_map_allocator_is_provided() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let _alloc: &Global = map.allocator();
    }

    #[test]
    fn new_map_drop_is_clean() {
        // Dropping a freshly constructed empty map must not leak or panic.
        let map: BTreeMap<i32, i32> = BTreeMap::new();
        drop(map);
    }

    #[test]
    fn new_in_map_drop_is_clean() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        drop(map);
    }

    #[test]
    fn new_map_grow_to_multilevel_then_drop() {
        // Exercise the full lifecycle: construct empty, grow past a single
        // leaf into a multi-level tree, verify integrity, then drop.
        let mut map = BTreeMap::new();
        for i in 0..120u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        assert_eq!(map.len(), 120);
        for i in 0..120u32 {
            assert_eq!(map.get(&i), Some(&(i * 3)), "missing key {}", i);
        }
        drop(map);
    }
}
