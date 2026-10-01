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
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};

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
