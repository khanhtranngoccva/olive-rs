//! Mutation methods for `BTreeMap`: clearing, popping from ends, and removing by key.

use core::borrow::Borrow;
use core::mem::ManuallyDrop;

use super::entry::Entry;
use super::map::BTreeMap;
use crate::alloc::AllocatorTryClone;
use crate::vec::Vec;

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Removes all entries in this map.
    pub fn clear(&mut self) {
        if self.root.is_some() {
            // Bootstrap a throwaway map that owns the old root and length, and holds a
            // reference to our allocator (`&A` is itself an allocator). Dropping it runs
            // the normal `into_iter()` teardown path, freeing every node through us.
            let alloc = &*self.alloc;
            let ephemeral = BTreeMap::<K, V, &A> {
                root: self.root.take(),
                length: self.length,
                alloc: ManuallyDrop::new(alloc),
                reserve_stack: ManuallyDrop::new(Some(Vec::new_in(alloc))),
            };
            drop(ephemeral);
        }
        self.length = 0;
    }

    /// Pops the first key-value pair out of the map.
    ///
    /// The keys are ordered by their ord comparison.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        match self.first_entry()? {
            Entry::Occupied(occupied) => Some(occupied.remove_entry()),
            _ => None,
        }
    }

    /// Pops the last key-value pair out of the map.
    ///
    /// The keys are ordered by their ord comparison.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        match self.last_entry()? {
            Entry::Occupied(occupied) => Some(occupied.remove_entry()),
            _ => None,
        }
    }

    /// Removes a key from the map, returning the value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove_entry(key).map(|(_k, v)| v)
    }

    /// Removes a key from the map, returning the key-value pair if present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Ord + Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.entry_ref(key)? {
            Entry::Occupied(occupied) => Some(occupied.remove_entry()),
            _ => None,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use std::sync::Arc;

    use super::*;
    use crate::alloc::Global;
    use crate::borrow::Borrow;
    use crate::test_helpers::{Ledger, TrackedItem};

    /// Lets a tracked key be looked up by its inner `u32` value, so tests can
    /// probe with cheap raw values instead of minting throwaway payloads that
    /// would themselves register/unregister ledger ids. Matches on `inner`,
    /// which is precisely the ordering the map uses.
    impl Borrow<u32> for TrackedItem<u32> {
        fn borrow(&self) -> &u32 {
            &self.inner
        }
    }

    /// Inserts a tracked key + tracked value pair into the map, registering both
    /// ids with the ledger up front so drops are observable.
    fn insert_tracked_pair(
        map: &mut BTreeMap<TrackedItem<u32>, TrackedItem<u32>>,
        key_val: u32,
        val_val: u32,
        ledger: &Arc<Ledger>,
    ) {
        let kid = ledger.allocate();
        ledger.register(kid);
        let vid = ledger.allocate();
        ledger.register(vid);
        map.try_insert(
            TrackedItem {
                id: kid,
                ledger: ledger.clone(),
                inner: key_val,
            },
            TrackedItem {
                id: vid,
                ledger: ledger.clone(),
                inner: val_val,
            },
        )
        .unwrap();
    }

    #[test]
    fn clear_empty_map() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn clear_single_element() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn clear_multiple_elements() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 20);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        // Every previously inserted key must be gone.
        for i in 0..20 {
            assert_eq!(map.get(&i), None);
        }
    }

    #[test]
    fn clear_drops_every_key_and_value_once() {
        // The ledger proves each cleared payload is dropped exactly once — no
        // leaks, no double-frees — which is the real correctness bar for clear.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        assert_eq!(map.len(), 50);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        // 50 keys + 50 values, each dropped exactly once.
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(ledger.total_allocated(), 100);
        assert!(ledger.all_dropped_once(0..100));
    }

    #[test]
    fn clear_multilevel_tree() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert!(map.root.as_ref().is_some_and(|r| r.height() >= 1));
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        // Spot-check a spread of keys across the former range are all absent.
        for i in [0u32, 1, 7, 39, 79, 80] {
            assert_eq!(map.get(&i), None);
        }
    }

    #[test]
    fn clear_multilevel_drops_all_payloads_once() {
        // Multi-level tree: clearing must free every node and payload without
        // leaking or double-freeing anything.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            insert_tracked_pair(&mut map, i, i * 2, &ledger);
        }
        assert!(map.root.as_ref().is_some_and(|r| r.height() >= 1));
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(ledger.total_allocated(), 400);
        assert!(ledger.all_dropped_once(0..400));
    }

    #[test]
    fn clear_then_reinsert() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i).unwrap();
        }
        map.clear();
        assert!(map.is_empty());
        map.try_insert(5, 50).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&5), Some(&50));
    }

    #[test]
    fn clear_drops_old_entries_but_keeps_new_ones_live() {
        // After clear, the original payloads are all dead, but a pair re-inserted
        // afterwards stays live until the map itself is dropped — proving clear
        // frees exactly what it emptied and nothing more.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..30u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        // Ids 0..60 belong to the original 30 pairs.
        map.clear();
        assert!(map.is_empty());
        // All originals are gone from the live set.
        assert!(
            ledger.live_ids().is_empty(),
            "cleared ids still live: {:?}",
            ledger.live_ids()
        );
        // Re-insert one fresh pair; its two new ids must now be live.
        insert_tracked_pair(&mut map, 99, 990, &ledger);
        let survivors = ledger.live_ids();
        assert_eq!(
            survivors.len(),
            2,
            "expected exactly 2 live ids, got {:?}",
            survivors
        );
        assert_eq!(ledger.total_allocated(), 62);
        // No double-free so far.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        // Dropping the map finally drops the surviving pair.
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..62));
    }

    #[test]
    fn pop_first_empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert_eq!(map.pop_first(), None);
    }

    #[test]
    fn pop_first_single() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(42, 420).unwrap();
        assert_eq!(map.pop_first(), Some((42, 420)));
        assert!(map.is_empty());
        assert_eq!(map.get(&42), None);
    }

    #[test]
    fn pop_first_ordered() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i * 10).unwrap();
        }
        for i in 0..3u32 {
            assert_eq!(map.pop_first(), Some((i, i * 10)));
            // The popped key is gone immediately: a second pop-front miss isn't
            // possible (it would return the next key), so verify via get.
            assert_eq!(map.get(&i), None, "get({i}) after pop should miss");
        }
        assert_eq!(map.len(), 7);
        assert_eq!(map.first_key_value(), Some((&3, &30)));
    }

    #[test]
    fn pop_first_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        for i in 0..2u32 {
            assert_eq!(map.pop_first(), Some((i, i * 2)));
            // The popped key is gone immediately.
            assert_eq!(map.get(&i), None, "get({i}) after pop should miss");
        }
        assert_eq!(map.len(), 78);
        assert_eq!(map.first_key_value(), Some((&2, &4)));
    }

    #[test]
    fn pop_first_drops_popped_pairs_once_no_leak() {
        // Popping from the front must free each popped key/value exactly once,
        // with no leak or double-free, while survivors stay live until the map
        // itself drops.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        // Pop the first 10 pairs (ids 0..20).
        for i in 0..10u32 {
            let (k, v) = map.pop_first().expect("expected a pair");
            assert_eq!((k.inner, v.inner), (i, i * 10));
            // The popped key is gone immediately.
            assert_eq!(map.get(&k), None, "get({i}) after pop should miss");
        }
        assert_eq!(map.len(), 40);
        // The 20 popped payloads are all dead; no double-free yet.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        for id in 0..20u32 {
            assert_eq!(
                ledger.drop_count(id),
                1,
                "popped id {id} not dropped exactly once"
            );
        }
        // Survivors (ids 20..100) are still live.
        let live = ledger.live_ids();
        assert_eq!(live.len(), 80, "unexpected live ids: {:?}", live);
        assert_eq!(live.first().copied(), Some(20));
        assert_eq!(live.last().copied(), Some(99));
        // Dropping the map frees the rest; everything dies exactly once.
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..100));
    }

    #[test]
    fn pop_last_empty() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert_eq!(map.pop_last(), None);
    }

    #[test]
    fn pop_last_single() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(42, 420).unwrap();
        assert_eq!(map.pop_last(), Some((42, 420)));
        assert!(map.is_empty());
        assert_eq!(map.get(&42), None);
    }

    #[test]
    fn pop_last_ordered() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i * 10).unwrap();
        }
        for i in (7..10).rev() {
            assert_eq!(map.pop_last(), Some((i, i * 10)));
            // The popped key is gone immediately.
            assert_eq!(map.get(&i), None, "get({i}) after pop should miss");
        }
        assert_eq!(map.len(), 7);
        assert_eq!(map.last_key_value(), Some((&6, &60)));
    }

    #[test]
    fn pop_last_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        for i in (78..80).rev() {
            assert_eq!(map.pop_last(), Some((i, i * 2)));
            // The popped key is gone immediately.
            assert_eq!(map.get(&i), None, "get({i}) after pop should miss");
        }
        assert_eq!(map.len(), 78);
        assert_eq!(map.last_key_value(), Some((&77, &154)));
    }

    #[test]
    fn pop_last_drops_popped_pairs_once_no_leak() {
        // Popping from the back must free each popped key/value exactly once,
        // with no leak or double-free, while survivors stay live until the map
        // itself drops.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        // Pop the last 10 pairs (ids 80..100).
        for i in (40..50u32).rev() {
            let (k, v) = map.pop_last().expect("expected a pair");
            assert_eq!((k.inner, v.inner), (i, i * 10));
            assert!(map.get(&k).is_none(), "get {i} should miss");
        }
        assert_eq!(map.len(), 40);
        // The 20 popped payloads are all dead; no double-free yet.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        for id in 80..100u32 {
            assert_eq!(
                ledger.drop_count(id),
                1,
                "popped id {id} not dropped exactly once"
            );
        }
        // Survivors (ids 0..80) are still live.
        let live = ledger.live_ids();
        assert_eq!(live.len(), 80, "unexpected live ids: {:?}", live);
        assert_eq!(live.first().copied(), Some(0));
        assert_eq!(live.last().copied(), Some(79));
        // Dropping the map frees the rest; everything dies exactly once.
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..100));
    }

    #[test]
    fn pop_first_and_last_alternating() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i).unwrap();
        }
        for i in 0..10 {
            assert_eq!(map.pop_first(), Some((i, i)));
            assert!(map.get(&i).is_none(), "get {i} should miss");
            assert_eq!(map.pop_last(), Some((19 - i, 19 - i)));
            assert!(map.get(&(19 - i)).is_none(), "get {} should miss", 19 - i);
        }
        assert!(map.is_empty());
    }

    #[test]
    fn remove_entry_found() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        map.try_insert(2, 20).unwrap();
        map.try_insert(3, 30).unwrap();
        assert_eq!(map.remove_entry(&2), Some((2, 20)));
        // The removed key is gone: a second remove and a get both miss.
        assert!(map.remove_entry(&2).is_none());
        assert_eq!(map.get(&2), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn remove_entry_missing() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        assert_eq!(map.remove_entry(&99), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_entry_drains_to_empty() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..30u32 {
            map.try_insert(i, i * 7).unwrap();
        }
        for i in 0..30u32 {
            assert_eq!(map.remove_entry(&i), Some((i, i * 7)));
            // The key is gone immediately: a second remove and a get both miss.
            assert!(
                map.remove_entry(&i).is_none(),
                "second removal of {i} should miss"
            );
            assert!(map.get(&i).is_none(), "get({i}) after removal should miss");
        }
        assert!(map.is_empty());
    }

    #[test]
    fn remove_entry_custom_order_scrambled() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..40u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        // A fixed pseudo-random-looking permutation of 0..40 (stride 17 is
        // coprime to 40, so this visits every key exactly once).
        let order: std::vec::Vec<u32> = (0..40u32).map(|n| (n * 17 + 5) % 40).collect();
        // Sanity: it's a genuine permutation.
        let mut seen = [false; 40];
        for &k in &order {
            assert!(!seen[k as usize], "duplicate in removal order");
            seen[k as usize] = true;
        }
        for &k in &order {
            assert_eq!(
                map.remove_entry(&k),
                Some((k, k * 3)),
                "removal of {k} failed"
            );
            // The key is gone immediately: a second remove and a get both miss.
            assert!(
                map.remove_entry(&k).is_none(),
                "second removal of {k} should miss"
            );
            assert!(map.get(&k).is_none(), "get({k}) after removal should miss");
        }
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn remove_entry_custom_order_multilevel_schedule() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i.wrapping_mul(7)).unwrap();
        }
        assert!(map.root.as_ref().is_some_and(|r| r.height() >= 1));
        // Deterministic scrambled order: stride 31 is coprime to 80, hitting
        // interior keys alongside the extremes across all levels.
        let order: std::vec::Vec<u32> = (0..80u32).map(|n| (n * 31 + 3) % 80).collect();
        let mut seen = [false; 80];
        for &k in &order {
            assert!(!seen[k as usize], "duplicate in removal schedule");
            seen[k as usize] = true;
            assert_eq!(
                map.remove_entry(&k),
                Some((k, k.wrapping_mul(7))),
                "removal of {k} failed"
            );
            // The key is gone immediately: a second remove and a get both miss.
            assert!(
                map.remove_entry(&k).is_none(),
                "second removal of {k} should miss"
            );
            assert!(map.get(&k).is_none(), "get({k}) after removal should miss");
        }
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    /// Removing the same key twice returns the value once and `None` the second
    /// time, without corrupting the rest of the map.
    #[test]
    fn remove_entry_duplicate_key_returns_none_second_time() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.remove_entry(&5), Some((5, 5)));
        assert_eq!(map.remove_entry(&5), None);
        assert_eq!(map.len(), 9);
        assert_eq!(map.get(&4), Some(&4));
        assert_eq!(map.get(&6), Some(&6));
    }

    #[test]
    fn remove_entry_drops_removed_pairs_once_no_leak() {
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        // Remove an interleaved subset: evens among 0..40. Probe with a raw u32
        // (TrackedItem<u32>: Borrow<u32>) so no real payload is minted or dropped.
        let mut removed: std::vec::Vec<u32> = std::vec::Vec::new();
        for i in (0..40u32).step_by(2) {
            let (k, v) = map.remove_entry(&i).expect("expected the key to exist");
            assert_eq!((k.inner, v.inner), (i, i * 10));
            removed.push(i);
        }
        assert_eq!(removed.len(), 20);
        assert_eq!(map.len(), 30);
        // Each removed key must be gone: a second remove and a get both miss.
        for &i in &removed {
            assert!(
                map.remove_entry(&i).is_none(),
                "second removal of {i} should miss"
            );
            assert!(map.get(&i).is_none(), "get({i}) after removal should miss");
        }
        // No double-free so far.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        // Exactly 40 payloads (20 keys + 20 values) dropped, each once.
        let counts = ledger.drop_counts();
        assert_eq!(counts.values().sum::<usize>(), 40);
        assert!(
            counts.values().all(|&c| c == 1),
            "some payload dropped != once: {counts:?}"
        );
        // 100 total payloads minus the 40 dropped leaves 60 live: the 20 odd
        // inners as keys plus the 40 values whose keys survived (odds among 0..40
        // and all of 40..50).
        assert_eq!(
            ledger.live_ids().len(),
            60,
            "unexpected live ids: {:?}",
            ledger.live_ids()
        );
        // Dropping the map frees the rest; everything dies exactly once.
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..100));
    }

    #[test]
    fn remove_entry_drain_scrambled_ledger_clean() {
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..60u32 {
            insert_tracked_pair(&mut map, i, i * 2, &ledger);
        }
        // Scrambled removal order over all inners; probe with raw u32 keys.
        let order: std::vec::Vec<u32> = (0..60u32).map(|n| (n * 23 + 7) % 60).collect();
        for &inner in &order {
            let (k, v) = map.remove_entry(&inner).expect("key should exist");
            assert_eq!((k.inner, v.inner), (inner, inner * 2));
            // The key is gone immediately: a second remove misses.
            assert!(
                map.remove_entry(&inner).is_none(),
                "second removal of {inner} should miss"
            );
            assert!(
                map.get(&inner).is_none(),
                "get({inner}) after removal should miss"
            );
        }
        assert!(map.is_empty());
        // Every payload is now dead — no survivors remain live.
        assert!(
            ledger.live_ids().is_empty(),
            "survivors still live after full drain: {:?}",
            ledger.live_ids()
        );
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert!(ledger.all_dropped_once(0..120));
    }
}
