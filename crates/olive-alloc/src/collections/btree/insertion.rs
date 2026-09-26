//! Public insertion API for BTreeMap.
//!
//! Thin facade exposing the user-facing insertion variants. All actual
//! insertion logic lives in [`entry`](super::entry) and
//! [`node`](super::node).

use crate::alloc::{AllocError, AllocatorTryClone};

use super::entry::Entry;
use super::map::BTreeMap;

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Inserts a key-value pair into the map, attempting allocation as needed.
    ///
    /// If the key already existed, the old value is replaced and returned.
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

    /// Attempts to insert a key-value pair, returning the key and value back
    /// on allocation failure so the caller can retry or handle the error.
    ///
    /// On success returns `Ok(Some(old_value))` if the key was already present,
    /// or `Ok(None)` if it was newly inserted.
    ///
    /// # Errors
    ///
    /// Returns `Err((key, value))` if memory allocation fails. The tree is
    /// left unmodified on failure.
    pub fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Result<Option<V>, ()>, (K, V)> {
        match self.entry(key) {
            Entry::Occupied(mut occ) => Ok(Ok(Some(occ.insert(value)))),
            Entry::Vacant(vac) => match vac.try_insert_entry(value) {
                Ok(_) => Ok(Ok(None)),
                Err((k, v, _)) => Err((k, v)),
            },
        }
    }

    /// Inserts a key-value pair only if the key does not already exist.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was
    /// already present (in which case the existing value is unchanged).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails.
    pub fn try_insert_unique(&mut self, key: K, value: V) -> Result<bool, AllocError> {
        match self.entry(key) {
            Entry::Occupied(_) => Ok(false),
            Entry::Vacant(vac) => vac
                .try_insert_entry(value)
                .map(|_| true)
                .map_err(|(_, _, e)| e),
        }
    }

    /// Like [`Self::try_insert_unique`], but returns the key and value back
    /// on allocation failure.
    ///
    /// Returns `Ok(true)` if the key was newly inserted, `Ok(false)` if it was
    /// already present.
    ///
    /// # Errors
    ///
    /// Returns `Err((key, value))` if memory allocation fails. The tree is
    /// left unmodified on failure.
    pub fn try_insert_unique_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Result<bool, ()>, (K, V)> {
        match self.entry(key) {
            Entry::Occupied(_) => Ok(Ok(false)),
            Entry::Vacant(vac) => match vac.try_insert_entry(value) {
                Ok(_) => Ok(Ok(true)),
                Err((k, v, _)) => Err((k, v)),
            },
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a key from `other` is already present in `self`, the respective
    /// value from `self` will be overwritten with the respective value from
    /// `other`. Similar to [`try_insert`](Self::try_insert), though, the key
    /// is not overwritten, which matters for types that can be `==` without
    /// being identical.
    ///
    /// Uses the "slow" approach: iterates over `other`'s entries and re-inserts
    /// them into `self` one by one. Each individual insertion is atomic — on
    /// allocation failure, `self` remains in a valid state instead of being
    /// completely destroyed.
    ///
    /// # Irreversibility on failure
    ///
    /// This method is **not** transactional. Once an insertion succeeds, its
    /// effects cannot be undone: in particular, when a colliding key is
    /// overwritten, the displaced `self` value is dropped at that moment and
    /// is unrecoverable afterwards. So if the append later fails, `self` is
    /// left in a half-merged state where some collisions have permanently
    /// resolved to `other`'s values.
    ///
    /// There is no cheap way to make this operation all-or-nothing, because doing
    /// so would require snapshotting every evicted value up front - as costly
    /// as copying the whole map.
    ///
    /// Callers who need true all-or-nothing semantics should build the result
    /// in a fresh map and commit only on success.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during one of the
    /// individual insertions. On failure the first un-inserted pair is dropped
    /// (see [`try_append_give_back`](Self::try_append_give_back) to recover it);
    /// entries inserted before the failure remain in `self`, and `other` retains
    /// only the tail of entries not yet drained.
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), AllocError> {
        match self.try_append_give_back(other) {
            Ok(()) => Ok(()),
            Err((_key, _value)) => Err(AllocError),
        }
    }

    /// Like [`try_append`](Self::try_append), but returns the first un-inserted
    /// `(key, value)` back to the caller on allocation failure instead of
    /// dropping it, so it can be retried or otherwise handled.
    ///
    /// Semantics are identical to [`try_append`](Self::try_append): overlapping
    /// keys are overwritten by `other`'s value (keeping `self`'s key), and each
    /// insertion is atomic. See [`try_append`](Self::try_append)'s
    ///
    /// Like [`try_append`](Self::try_append), overriding old values is irreversible.
    ///
    /// # Errors
    ///
    /// Returns `Err((key, value), AllocError)` - a tuple carrying the current un-inserted
    /// pair and an [`AllocError`] memory allocation fails during one of the individual
    /// insertions.
    pub fn try_append_give_back(&mut self, other: &mut Self) -> Result<(), (K, V)> {
        while let Some((k, v)) = other.pop_first() {
            match self.try_insert_give_back(k, v) {
                Ok(_) => {}
                Err(stranded) => return Err(stranded),
            }
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::super::map::BTreeMap;
    use crate::alloc::Global;

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
        // Measured for ascending sequential insertion: the first leaf split
        // (height 0 -> 1) happens at len 12, and the internal root itself
        // splits (height 1 -> 2, i.e. genuine root growth) at len 89. So we
        // insert past 89 to actually exercise the taller tree.
        for i in 0..95u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert_eq!(map.len(), 95);
        // Root growth means the tree is now at least two levels deep.
        let h = map.root.as_ref().map_or(0, |r| r.height());
        assert!(
            h >= 2,
            "expected multi-level tree (root grew), got height {}",
            h
        );
        for i in 0..95u32 {
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

    // ── try_append tests ─────────────────────────────────────────────────────

    #[test]
    fn append_empty_to_empty() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_append(&mut b).unwrap();
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_to_empty() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        for i in 0..5 {
            b.try_insert(i, i * 10).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        for i in 0..5 {
            assert_eq!(a.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn append_nonempty_to_nonempty() {
        let mut a = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        for i in 0..5 {
            a.try_insert(i, i).unwrap();
        }
        for i in 5..10 {
            b.try_insert(i, i * 100).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 10);
        assert!(b.is_empty());
        for i in 0..5 {
            assert_eq!(a.get(&i), Some(&i));
        }
        for i in 5..10 {
            assert_eq!(a.get(&i), Some(&(i * 100)));
        }
    }

    #[test]
    fn append_multilevel() {
        let mut a = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            a.try_insert(i, i).unwrap();
        }
        for i in 50..100u32 {
            b.try_insert(i, i * 2).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 100);
        assert!(b.is_empty());
        for i in 0..100u32 {
            let expected = if i < 50 { i } else { i * 2 };
            assert_eq!(a.get(&i), Some(&expected), "missing key {}", i);
        }
    }

    #[test]
    fn append_clears_other() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        for i in 0..20 {
            b.try_insert(i, i).unwrap();
        }
        assert_eq!(b.len(), 20);
        a.try_append(&mut b).unwrap();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    /// Inserts a tracked key/value pair into `map`, registering both IDs with
    /// `ledger` so their eventual drop can be verified. Returns the allocated
    /// key ID.
    fn insert_tracked_pair(
        map: &mut BTreeMap<crate::test_helpers::TrackedItem<u32>, crate::test_helpers::TrackedItem<u32>>,
        key_val: u32,
        val_val: u32,
        ledger: &std::sync::Arc<crate::test_helpers::Ledger>,
    ) -> u32 {
        use crate::test_helpers::TrackedItem;
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
        kid
    }

    #[test]
    fn append_overlapping_keys_other_value_wins() {
        // Mirrors std's `append`: when a key exists in both maps, `other`'s
        // value overwrites `self`'s, and `self`'s key is kept.
        let mut a = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        a.try_insert(1, "a").unwrap();
        a.try_insert(2, "b").unwrap();
        a.try_insert(3, "c").unwrap(); // key 3 also present in b
        b.try_insert(3, "d").unwrap(); // key 3 also present in a
        b.try_insert(4, "e").unwrap();
        b.try_insert(5, "f").unwrap();

        a.try_append(&mut b).unwrap();

        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        assert_eq!(a.get(&1), Some(&"a"));
        assert_eq!(a.get(&2), Some(&"b"));
        assert_eq!(a.get(&3), Some(&"d")); // "c" was overwritten by b's "d"
        assert_eq!(a.get(&4), Some(&"e"));
        assert_eq!(a.get(&5), Some(&"f"));
    }

    #[test]
    fn append_overlapping_keys_keeps_self_key_identity() {
        // Identity version of the overlap test above: two *distinct* keys that
        // compare equal (both wrapping `inner == 3`) live in `a` and `b`. The
        // appended entry must resolve to `a`'s existing key — keeping its
        // original ID — while `b`'s duplicate key and both values are dropped.
        // If the merge ever substituted `b`'s key for `a`'s, the surviving
        // entry would carry `b`'s ID and this assertion would fail.
        use crate::test_helpers::{Ledger, TrackedItem};
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);

        let a_k3 = insert_tracked_pair(&mut a, 3, 30, &ledger); // self's key 3
        let b_k3 = insert_tracked_pair(&mut b, 3, 31, &ledger); // other's dup key 3
        assert_ne!(a_k3, b_k3);
        insert_tracked_pair(&mut a, 1, 10, &ledger);
        insert_tracked_pair(&mut a, 2, 20, &ledger);
        insert_tracked_pair(&mut b, 4, 40, &ledger);
        insert_tracked_pair(&mut b, 5, 50, &ledger);

        a.try_append(&mut b).unwrap();

        assert_eq!(a.len(), 5);
        assert!(b.is_empty());

        // The merged entry at key 3 keeps SELF's key object (original ID)…
        let (k, v) = a.iter().find(|(k, _)| k.inner == 3).expect("key 3 survives");
        assert_eq!(k.id, a_k3, "surviving key must be self's original key, not other's");
        // …and carries OTHER's value payload (its old value was displaced).
        assert_eq!(v.inner, 31);

        // Drop both maps so every tracked item dies, then verify each of the 12
        // allocated ids (3 pairs per map) was dropped exactly once.
        drop(a);
        drop(b);
        assert!(ledger.leaked_ids().is_empty(), "leaked ids: {:?}", ledger.leaked_ids());
        assert!(ledger.double_dropped().is_empty(), "double-dropped: {:?}", ledger.double_dropped());
        assert_eq!(ledger.total_allocated(), 12);
    }

    #[test]
    fn append_disjoint_but_interleaved_ranges() {
        // No overlap, but `other`'s keys interleave below/above `self`'s keys.
        // There is no ordering precondition; entries simply land in sorted order.
        let mut a = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        a.try_insert(10, 100).unwrap();
        a.try_insert(20, 200).unwrap();
        b.try_insert(5, 50).unwrap(); // less than some of a's keys
        b.try_insert(15, 150).unwrap(); // interleaves between a's keys
        b.try_insert(30, 300).unwrap(); // greater than all of a's keys

        a.try_append(&mut b).unwrap();

        assert_eq!(a.len(), 5);
        assert!(b.is_empty());
        // Ascending iteration must yield sorted keys regardless of origin.
        let mut it = a.iter();
        assert_eq!(it.next(), Some((&5, &50)));
        assert_eq!(it.next(), Some((&10, &100)));
        assert_eq!(it.next(), Some((&15, &150)));
        assert_eq!(it.next(), Some((&20, &200)));
        assert_eq!(it.next(), Some((&30, &300)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn append_single_element() {
        let mut a = BTreeMap::new_in(Global);
        let mut b = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(2, 20).unwrap();
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a.get(&1), Some(&10));
        assert_eq!(a.get(&2), Some(&20));
        assert!(b.is_empty());
    }

    /// Inserts a tracked pair into a `BudgetedAlloc`-backed map, registering
    /// both IDs with `ledger`. Used by the failed-append tests below so the
    /// ledger can prove no element is dropped twice.
    fn insert_tracked_pair_budgeted(
        map: &mut BTreeMap<crate::test_helpers::TrackedItem<u32>, crate::test_helpers::TrackedItem<u32>, crate::test_helpers::BudgetedAlloc>,
        key_val: u32,
        val_val: u32,
        ledger: &std::sync::Arc<crate::test_helpers::Ledger>,
    ) {
        use crate::test_helpers::TrackedItem;
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
    fn append_give_back_hands_stranded_pair_and_partitions_union() {
        use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};
        use std::collections::HashSet;

        // Build both maps under a generous shared budget so construction
        // succeeds deterministically.
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        let mut b: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        for i in 0..40u32 {
            if i % 2 == 0 {
                insert_tracked_pair_budgeted(&mut a, i, i * 100, &ledger);
            } else {
                insert_tracked_pair_budgeted(&mut b, i, i * 100, &ledger);
            }
        }
        let total = 40;

        // Snapshot the original union as a key -> value map so we can verify
        // the partition contract after a forced failure.
        let mut expected: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for (k, v) in a.iter() {
            expected.insert(k.inner, v.inner);
        }
        for (k, v) in b.iter() {
            expected.insert(k.inner, v.inner);
        }

        // Drain the budget so the next node allocation during re-insertion
        // fails. The append then strands exactly one pair.
        alloc.drain();

        let result = a.try_append_give_back(&mut b);
        assert!(result.is_err(), "expected an allocation failure mid-append");
        let (sk, sv) = result.unwrap_err();

        // The stranded pair must be present in neither map.
        assert!(a.get(&sk).is_none(), "stranded key must not be in self");
        assert!(b.get(&sk).is_none(), "stranded key must not be in other");

        // Partition contract: self ∪ other ∪ {stranded} == original union,
        // with no key appearing twice. This proves nothing was silently lost
        // and nothing was duplicated.
        let mut seen: HashSet<u32> = HashSet::new();
        for (k, v) in a.iter() {
            assert!(
                seen.insert(k.inner),
                "duplicate key in self after failure: {}",
                k.inner
            );
            assert_eq!(v.inner, expected[&k.inner], "value mismatch in self for key {}", k.inner);
        }
        for (k, v) in b.iter() {
            assert!(
                seen.insert(k.inner),
                "duplicate key in other after failure: {}",
                k.inner
            );
            assert_eq!(v.inner, expected[&k.inner], "value mismatch in other for key {}", k.inner);
        }
        assert!(seen.insert(sk.inner), "stranded key duplicates an existing key");
        assert_eq!(sv.inner, expected[&sk.inner], "stranded value mismatches original");
        assert_eq!(
            seen.len(),
            total,
            "partition must cover exactly the original union"
        );

        // Drop everything — both maps plus the stranded pair — and verify each
        // of the 80 allocated ids died exactly once. A double-drop anywhere in
        // the failure path (e.g. the stranded pair being dropped again inside
        // `other`) would show up here.
        drop(a);
        drop(b);
        drop((sk, sv));
        assert!(ledger.leaked_ids().is_empty(), "leaked ids: {:?}", ledger.leaked_ids());
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(ledger.total_allocated(), 80);
    }

    #[test]
    fn append_delegates_to_give_back_and_drops_stranded_pair() {
        use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};

        // Same setup as above; here we only confirm that `try_append` reports
        // the same failure and leaves `self` in a valid (non-corrupt) state,
        // i.e. it faithfully discards the give-back pair without panicking.
        // Tracked items let us additionally prove the discarded pair was
        // dropped exactly once — not leaked, not dropped twice.
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        let mut b: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        for i in 0..40u32 {
            if i % 2 == 0 {
                insert_tracked_pair_budgeted(&mut a, i, i * 100, &ledger);
            } else {
                insert_tracked_pair_budgeted(&mut b, i, i * 100, &ledger);
            }
        }
        alloc.drain();

        let result = a.try_append(&mut b);
        assert!(result.is_err(), "expected try_append to surface the OOM");
        // Whatever subset made it in, `a` must still be a valid, sorted map.
        let mut last_key = None;
        for (k, _) in a.iter() {
            if let Some(prev) = last_key {
                assert!(prev < k.inner, "self must remain sorted after failed append");
            }
            last_key = Some(k.inner);
        }

        // The stranded pair was handed back and then dropped by `try_append`;
        // dropping the maps completes the picture. Every one of the 80 ids
        // must have been dropped exactly once.
        drop(a);
        drop(b);
        assert!(ledger.leaked_ids().is_empty(), "leaked ids: {:?}", ledger.leaked_ids());
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(ledger.total_allocated(), 80);
    }
}
