//! Public insertion API for [`BTreeSet`].

use super::super::set_val::SetValZST;
use super::BTreeSet;
use crate::alloc::{AllocError, Allocator};

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Inserts a value into the set.
    ///
    /// If the value was already present, it is left unchanged and `false` is
    /// returned. Otherwise the new value is inserted and `true` is returned.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during insertion of a
    /// previously-absent value. The probed value is dropped on failure and the
    /// set is left unmodified.
    pub fn try_insert(&mut self, value: T) -> Result<bool, AllocError> {
        // Inserting the ZST marker under `value` commits the value into the
        // set. `was_new` reports whether this particular slot was empty before.
        let was_new = self.map.try_insert(value, SetValZST)?.is_none();
        Ok(was_new)
    }

    /// Inserts a value into the set, returning ownership of it back on
    /// allocation failure.
    ///
    /// If the value was already present, it is left unchanged and `false` is
    /// returned. Otherwise the new value is inserted and `true` is returned.
    ///
    /// # Errors
    ///
    /// Returns `(T, AllocError)` if memory allocation fails during insertion
    /// of a previously-absent value; the tuple carries the original value back
    /// to the caller.
    pub fn try_insert_give_back(&mut self, value: T) -> Result<bool, (T, AllocError)> {
        // The map hands back `(key, marker, error)` on failure; the marker is
        // a ZST with no data, so we discard it and keep the key plus the error.
        match self.map.try_insert_give_back(value, SetValZST) {
            // `None` → slot was empty, we just inserted  → true
            // `Some(_)` → slot was occupied, value unchanged → false
            Ok(opt) => Ok(opt.is_none()),
            Err((k, _marker, e)) => Err((k, e)),
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// If a value from `other` is already present in `self`, it is simply
    /// skipped (sets cannot hold duplicates, so there is nothing to overwrite).
    ///
    /// Uses the "slow" approach: iterates over `other`'s entries and re-inserts
    /// them into `self` one by one. Each individual insertion is atomic — on
    /// allocation failure, `self` remains in a valid state instead of being
    /// completely destroyed.
    ///
    /// # Irreversibility on failure
    ///
    /// This method is **not** transactional. Once an insertion succeeds, its
    /// effects cannot be undone. So if the append later fails, `self` is left
    /// in a half-merged state containing some of `other`'s values, and `other`
    /// retains only the tail of entries not yet drained.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during one of the
    /// individual insertions. On failure the first un-inserted value is dropped
    /// (see [`try_append_give_back`](Self::try_append_give_back) to recover it);
    /// entries inserted before the failure remain in `self`, and `other` retains
    /// only the tail of entries not yet drained.
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), AllocError> {
        self.map.try_append(&mut other.map)
    }

    /// Like [`try_append`](Self::try_append), but returns the first un-inserted
    /// value back to the caller on allocation failure instead of dropping it,
    /// so it can be retried or otherwise handled.
    ///
    /// Semantics are identical to [`try_append`](Self::try_append): overlapping
    /// values are skipped (the set already holds them), and each insertion is
    /// atomic.
    ///
    /// # Errors
    ///
    /// Returns `(T, AllocError)` if memory allocation fails during one of the
    /// individual insertions. The tuple carries the current un-inserted value
    /// back to the caller.
    pub fn try_append_give_back(&mut self, other: &mut Self) -> Result<(), (T, AllocError)> {
        match self.map.try_append_give_back(&mut other.map) {
            Ok(()) => Ok(()),
            Err((key, _marker, e)) => Err((key, e)),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::collections::btree::invariant::{check_ascending_keys, check_tree_invariant};
    use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};
    use std::vec::Vec;

    #[test]
    fn insert_single_returns_true_and_is_visible() {
        let mut set = BTreeSet::new();
        assert!(set.is_empty());
        assert_eq!(set.try_insert(1), Ok(true));
        assert_eq!(set.len(), 1);
        assert!(set.contains(&1));
    }

    #[test]
    fn insert_duplicate_returns_false_and_keeps_len() {
        let mut set = BTreeSet::new();
        set.try_insert(5).unwrap();
        assert_eq!(set.try_insert(5), Ok(false));
        assert_eq!(set.len(), 1);
        assert!(set.contains(&5));
    }

    #[test]
    fn insert_many_sorted_and_complete() {
        let mut set = BTreeSet::new();
        for i in 0..20u32 {
            assert_eq!(set.try_insert(i), Ok(true));
        }
        assert_eq!(set.len(), 20);
        for i in 0..20u32 {
            assert!(set.contains(&i));
        }
        assert!(!set.contains(&99));
    }

    #[test]
    fn insert_reverse_order_yields_same_set() {
        let mut set = BTreeSet::new();
        for i in (0..15u32).rev() {
            assert_eq!(set.try_insert(i), Ok(true));
        }
        assert_eq!(set.len(), 15);
        for i in 0..15u32 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn insert_triggers_leaf_split_still_complete() {
        // CAPACITY is 11, so the 12th insert forces a leaf split.
        let mut set = BTreeSet::new();
        for i in 0..12u32 {
            assert_eq!(set.try_insert(i), Ok(true));
        }
        assert_eq!(set.len(), 12);
        for i in 0..12u32 {
            assert!(set.contains(&i), "missing {} after split", i);
        }
    }

    #[test]
    fn insert_grows_multi_level_tree() {
        let mut set = BTreeSet::new();
        for i in 0..200u32 {
            assert_eq!(set.try_insert(i), Ok(true));
        }
        assert_eq!(set.len(), 200);
        for i in 0..200u32 {
            assert!(set.contains(&i), "missing {}", i);
        }
    }

    #[test]
    fn insert_mixed_duplicates_only_counts_once() {
        let mut set = BTreeSet::new();
        // Interleave repeats; each distinct value must count exactly once.
        for _ in 0..3 {
            for v in [7, 3, 9, 1, 5] {
                set.try_insert(v).unwrap();
            }
        }
        assert_eq!(set.len(), 5);
        for v in [7, 3, 9, 1, 5] {
            assert!(set.contains(&v));
        }
    }

    #[test]
    fn insert_fails_with_alloc_error_and_no_leak() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        let item = TrackedItem::construct(&ledger, 7);
        alloc.drain();

        match set.try_insert(item) {
            Err(e) => assert!(matches!(e, AllocError)),
            Ok(_) => panic!("expected allocation failure"),
        }
        // The probed value was dropped on failure — nothing leaked, and the
        // set is untouched.
        assert!(set.is_empty());
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn occupied_path_never_allocates_and_succeeds_despite_drained_budget() {
        // Re-inserting an already-present value short-circuits without
        // touching the allocator, so it succeeds even after the budget drains.
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        for i in 0..11 {
            set.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        alloc.drain();

        let dup = TrackedItem::construct(&ledger, 10);
        assert_eq!(set.try_insert(dup), Ok(false));
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn give_back_success_matches_plain_try_insert() {
        let mut set = BTreeSet::new();
        // First insert: slot was empty → Ok(true).
        assert_eq!(set.try_insert_give_back(3), Ok(true));
        // Duplicate: slot occupied → Ok(false).
        assert_eq!(set.try_insert_give_back(3), Ok(false));
        assert_eq!(set.len(), 1);
        assert!(set.contains(&3));
    }

    #[test]
    fn give_back_failure_returns_value_intact_and_no_leak() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        let item = TrackedItem::construct(&ledger, 42);
        alloc.drain();

        match set.try_insert_give_back(item) {
            Err((returned, _err)) => {
                // The value must be handed back with its original identity.
                assert_eq!(returned.inner, 42);
                // Drop the returned value so it deregisters from the ledger.
                drop(returned);
            }
            Ok(_) => panic!("expected allocation failure"),
        }
        // Value was recovered and dropped by us — nothing leaked.
        assert!(set.is_empty());
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    // ── try_append tests ─────────────────────────────────────────────────────

    #[test]
    fn append_empty_to_empty() {
        let mut a = BTreeSet::<i32>::new();
        let mut b = BTreeSet::new();
        a.try_append(&mut b).unwrap();
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn append_disjoint_sets_yields_union() {
        let mut a = BTreeSet::new();
        let mut b = BTreeSet::new();
        for i in 0..5u32 {
            a.try_insert(i).unwrap();
        }
        for i in 5..10u32 {
            b.try_insert(i).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 10);
        assert!(b.is_empty());
        for i in 0..10u32 {
            assert!(a.contains(&i), "missing {}", i);
        }
    }

    #[test]
    fn append_overlapping_values_collapses_without_duplication() {
        let mut a = BTreeSet::new();
        let mut b = BTreeSet::new();
        for i in [1, 3, 5, 7] {
            a.try_insert(i).unwrap();
        }
        for i in [3, 5, 9, 11] {
            b.try_insert(i).unwrap();
        }
        a.try_append(&mut b).unwrap();
        // Union is {1, 3, 5, 7, 9, 11} — overlaps at 3 and 5 collapse.
        assert_eq!(a.len(), 6);
        assert!(b.is_empty());
        for i in [1, 3, 5, 7, 9, 11] {
            assert!(a.contains(&i));
        }
    }

    #[test]
    fn append_interleaved_ranges_preserves_ordering() {
        let mut a = BTreeSet::new();
        let mut b = BTreeSet::new();
        for i in [10, 20, 40] {
            a.try_insert(i).unwrap();
        }
        for i in [5, 15, 30, 50] {
            b.try_insert(i).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 7);
        assert!(b.is_empty());
        let collected: Vec<u32> = a.iter().copied().collect();
        assert_eq!(collected, std::vec![5, 10, 15, 20, 30, 40, 50]);
    }

    #[test]
    fn append_multilevel_trees() {
        let mut a = BTreeSet::new();
        let mut b = BTreeSet::new();
        for i in 0..200u32 {
            a.try_insert(i).unwrap();
        }
        for i in 100..300u32 {
            b.try_insert(i).unwrap();
        }
        // Both trees should be multilevel at this size (CAPACITY is 11).
        assert!(
            a.map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        assert!(
            b.map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );

        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 300);
        assert!(b.is_empty());
        check_tree_invariant(&a.map);
        check_ascending_keys(&a.map);
        for i in 0..300u32 {
            assert!(a.contains(&i), "missing {}", i);
        }
    }

    #[test]
    fn append_clears_other() {
        let mut a = BTreeSet::new();
        let mut b = BTreeSet::new();
        for i in 0..20u32 {
            b.try_insert(i).unwrap();
        }
        assert_eq!(b.len(), 20);
        a.try_append(&mut b).unwrap();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(a.len(), 20);
    }

    #[test]
    fn append_drops_other_elements_once_no_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeSet<TrackedItem<u32>> = BTreeSet::new();
        let mut b: BTreeSet<TrackedItem<u32>> = BTreeSet::new();
        for i in 0..10u32 {
            a.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        for i in 10..20u32 {
            b.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        a.try_append(&mut b).unwrap();
        assert_eq!(a.len(), 20);
        assert!(b.is_empty());
        drop(a);
        drop(b);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn append_overlap_drops_duplicate_from_other_once() {
        // When both sets contain the same value, `other`'s copy must be
        // dropped exactly once (the set already holds it; nothing is stored).
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeSet<TrackedItem<u32>> = BTreeSet::new();
        let mut b: BTreeSet<TrackedItem<u32>> = BTreeSet::new();
        // Overlap on inner values 5 and 10.
        for i in [3, 5, 7, 10] {
            a.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        for i in [5, 10, 12, 14] {
            b.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        let id_of_five = a.get(&5).unwrap().id;
        a.try_append(&mut b).unwrap();
        let new_id_of_five = a.get(&5).unwrap().id;
        assert_eq!(
            new_id_of_five, id_of_five,
            "old item should not be pushed out"
        );
        // Union: {3, 5, 7, 10, 12, 14}
        assert_eq!(a.len(), 6);
        assert!(b.is_empty());
        drop(a);
        drop(b);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn append_give_back_hands_stranded_value_and_partitions_union() {
        use std::collections::HashSet;

        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        let mut b: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        for i in 0..40u32 {
            if i % 2 == 0 {
                a.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
            } else {
                b.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
            }
        }

        // Snapshot the original union.
        let expected: HashSet<u32> = a.iter().chain(b.iter()).map(|item| item.inner).collect();

        // Drain the budget so the next node allocation during re-insertion fails.
        alloc.drain();

        let result = a.try_append_give_back(&mut b);
        assert!(result.is_err(), "expected an allocation failure mid-append");
        let (stranded, _err) = result.unwrap_err();

        // The stranded value must be present in neither set.
        assert!(!a.contains(&stranded.inner));
        assert!(!b.contains(&stranded.inner));

        // Partition contract: self ∪ other ∪ {stranded} == original union.
        let mut seen: HashSet<u32> = HashSet::new();
        for item in a.iter() {
            assert!(seen.insert(item.inner), "duplicate in self: {}", item.inner);
        }
        for item in b.iter() {
            assert!(
                seen.insert(item.inner),
                "duplicate in other: {}",
                item.inner
            );
        }
        assert!(seen.insert(stranded.inner), "stranded value duplicated");
        assert_eq!(seen, expected, "partition does not cover original union");

        // Drop everything; no leaks, no double-frees.
        drop(stranded);
        drop(a);
        drop(b);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn give_back_occupied_path_succeeds_despite_drained_budget() {
        // Re-inserting an already-present value short-circuits without
        // touching the allocator, so it succeeds even after the budget drains.
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set: BTreeSet<TrackedItem<u32>, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        let first = TrackedItem::construct(&ledger, 7);
        set.try_insert(first).expect("seed insert should succeed");
        alloc.drain();

        let dup = TrackedItem::construct(&ledger, 7);
        // Occupied path short-circuits without allocating → Ok(false).
        assert_eq!(set.try_insert_give_back(dup), Ok(false));
        assert_eq!(set.len(), 1);
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }
}
