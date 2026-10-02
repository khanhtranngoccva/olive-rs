//! Mutation methods for [`BTreeSet`]: bulk removal, end popping, and keyed
//! removal.
//!
//! These are thin facades over the underlying [`BTreeMap`] mutation surface;
//! each hides the zero-sized value marker so callers only ever see values.

use super::BTreeSet;
use crate::alloc::Allocator;
use core::borrow::Borrow;

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Removes all values from the set, returning nothing.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Removes the first (lowest-keyed) value from the set and returns it, or
    /// `None` if the set is empty.
    pub fn pop_first(&mut self) -> Option<T> {
        self.map.pop_first().map(|(k, _)| k)
    }

    /// Removes the last (highest-keyed) value from the set and returns it, or
    /// `None` if the set is empty.
    pub fn pop_last(&mut self) -> Option<T> {
        self.map.pop_last().map(|(k, _)| k)
    }

    /// Removes a value from the set and returns it, or `None` if no such value
    /// was present.
    ///
    /// The probe type `Q` need not be identical to the set's element type `T`;
    /// it only has to borrow-compare against it (e.g. removing from a
    /// `BTreeSet<String>` with a `&str`).
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // The set stores its elements as keys of the underlying map, so a
        // successful removal yields a key — i.e. the stored value itself.
        self.map.remove_entry(value).map(|(k, _)| k)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In place, removes all elements from the set for which the predicate
    /// returns `false`. The predicate receives each element by reference and
    /// must return `true` to keep it. Elements for which the predicate returns
    /// `false` are removed and dropped.
    pub fn retain<P>(&mut self, mut keep: P)
    where
        P: FnMut(&T) -> bool,
    {
        self.map.retain(move |k, _| keep(k));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::super::super::invariant::{check_ascending_keys, check_tree_invariant};
    use super::*;
    use crate::test_helpers::{Ledger, TestRng, TrackedItem};

    type TestSet = BTreeSet<TrackedItem<u32>>;

    /// Builds a set of tracked items whose inners are `0..n`, registering each
    /// in the ledger. Because `TrackedItem` orders by its inner, the stored
    /// values are sorted by their inner payload.
    fn build_set(n: u32, ledger: &std::sync::Arc<Ledger>) -> TestSet {
        let mut set = TestSet::new();
        for i in 0..n {
            set.try_insert(TrackedItem::construct(ledger, i)).unwrap();
        }
        set
    }

    // --- clear ------------------------------------------------------------------

    #[test]
    fn clear_empty_map() {
        let mut set = BTreeSet::<i32>::new();
        assert!(set.is_empty());
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_single_element() {
        let mut set = BTreeSet::new();
        set.try_insert(1).unwrap();
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_multiple_elements() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 10);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_multilevel_tree() {
        let mut set = BTreeSet::new();
        for i in 0..100u32 {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 100);
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn clear_then_reinsert() {
        let mut set = BTreeSet::new();
        for i in 0..5u32 {
            set.try_insert(i).unwrap();
        }
        set.clear();
        assert!(set.is_empty());
        for i in 10..15u32 {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 5);
        assert!(!set.contains(&1));
        assert!(set.contains(&10));
        assert!(set.contains(&14));
    }

    #[test]
    fn clear_drops_every_key_once() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(20, &ledger);
        assert_eq!(set.len(), 20);
        set.clear();
        assert!(set.is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn clear_multilevel_drops_all_payloads_once() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(60, &ledger);
        assert_eq!(set.len(), 60);
        set.clear();
        assert!(set.is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn clear_drops_old_entries_but_keeps_new_ones_live() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(10, &ledger);
        set.clear();
        for i in 100..105u32 {
            set.try_insert(TrackedItem::construct(&ledger, i)).unwrap();
        }
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    // --- pop_first --------------------------------------------------------------

    #[test]
    fn pop_first_empty() {
        let mut set = BTreeSet::<i32>::new();
        assert_eq!(set.pop_first(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_single() {
        let mut set = BTreeSet::new();
        set.try_insert(42).unwrap();
        assert_eq!(set.pop_first(), Some(42));
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_ordered() {
        let mut set = BTreeSet::new();
        for i in [5, 1, 9, 3, 7] {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.pop_first(), Some(1));
        assert_eq!(set.pop_first(), Some(3));
        assert_eq!(set.pop_first(), Some(5));
        assert_eq!(set.pop_first(), Some(7));
        assert_eq!(set.pop_first(), Some(9));
        assert_eq!(set.pop_first(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_multilevel() {
        let mut set = BTreeSet::new();
        for i in 0..80u32 {
            set.try_insert(i).unwrap();
        }
        for expected in 0..80u32 {
            assert_eq!(set.pop_first(), Some(expected));
        }
        assert_eq!(set.pop_first(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_and_last_alternating() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.pop_first(), Some(0));
        assert_eq!(set.pop_last(), Some(9));
        assert_eq!(set.pop_first(), Some(1));
        assert_eq!(set.pop_last(), Some(8));
        assert_eq!(set.pop_first(), Some(2));
        assert_eq!(set.pop_last(), Some(7));
        assert_eq!(set.len(), 4);
        assert_eq!(set.pop_first(), Some(3));
        assert_eq!(set.pop_last(), Some(6));
        assert_eq!(set.pop_first(), Some(4));
        assert_eq!(set.pop_last(), Some(5));
        assert!(set.is_empty());
    }

    #[test]
    fn pop_first_drops_popped_items_once_no_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(15, &ledger);
        let popped = set.pop_first().expect("non-empty");
        assert_eq!(popped.inner, 0, "lowest inner should pop first");
        drop(popped);
        // 14 items are still live in the set; only double-frees matter here.
        assert!(ledger.double_dropped().is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    // --- pop_last ---------------------------------------------------------------

    #[test]
    fn pop_last_empty() {
        let mut set = BTreeSet::<i32>::new();
        assert_eq!(set.pop_last(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_last_single() {
        let mut set = BTreeSet::new();
        set.try_insert(7).unwrap();
        assert_eq!(set.pop_last(), Some(7));
        assert!(set.is_empty());
    }

    #[test]
    fn pop_last_ordered() {
        let mut set = BTreeSet::new();
        for i in [5, 1, 9, 3, 7] {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.pop_last(), Some(9));
        assert_eq!(set.pop_last(), Some(7));
        assert_eq!(set.pop_last(), Some(5));
        assert_eq!(set.pop_last(), Some(3));
        assert_eq!(set.pop_last(), Some(1));
        assert_eq!(set.pop_last(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_last_multilevel() {
        let mut set = BTreeSet::new();
        for i in 0..80u32 {
            set.try_insert(i).unwrap();
        }
        for expected in (0..80u32).rev() {
            assert_eq!(set.pop_last(), Some(expected));
        }
        assert_eq!(set.pop_last(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn pop_last_drops_popped_items_once_no_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(15, &ledger);
        let popped = set.pop_last().expect("non-empty");
        assert_eq!(popped.inner, 14, "highest inner should pop last");
        drop(popped);
        // 14 items are still live in the set; only double-frees matter here.
        assert!(ledger.double_dropped().is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    // --- remove -----------------------------------------------------------------

    #[test]
    fn remove_single_element() {
        let mut set = BTreeSet::new();
        set.try_insert(10).unwrap();
        assert_eq!(set.take(&10), Some(10));
        assert!(set.is_empty());
    }

    #[test]
    fn remove_nonexistent_value() {
        let mut set = BTreeSet::new();
        set.try_insert(1).unwrap();
        set.try_insert(2).unwrap();
        assert_eq!(set.take(&99), None);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn remove_all_values_reverse_order() {
        let mut set = BTreeSet::new();
        for i in 0..5u32 {
            set.try_insert(i).unwrap();
        }
        for i in (0..5u32).rev() {
            assert_eq!(set.take(&i), Some(i));
        }
        assert!(set.is_empty());
    }

    #[test]
    fn remove_preserves_sorted_invariant() {
        let mut set = BTreeSet::new();
        for i in 0..50u32 {
            set.try_insert(i).unwrap();
        }
        for i in (0..25u32).step_by(2) {
            assert_eq!(set.take(&(i * 2)), Some(i * 2));
        }
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
    }

    #[test]
    fn remove_interleaved_with_insert() {
        let mut set = BTreeSet::new();
        for i in 0..20u32 {
            set.try_insert(i).unwrap();
        }
        for i in 0..20u32 {
            if i % 2 == 0 {
                assert_eq!(set.take(&i), Some(i));
            } else {
                set.try_insert(100 + i).unwrap();
            }
        }
        assert_eq!(set.len(), 20);
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        for i in (0..20u32).filter(|x| x % 2 == 1) {
            assert!(set.contains(&(100 + i)));
            assert!(set.contains(&i));
        }
        for i in (0..20u32).filter(|x| x % 2 == 0) {
            assert!(!set.contains(&i));
        }
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
    }

    #[test]
    fn remove_borrowed_str_value() {
        let mut set: BTreeSet<std::string::String> = BTreeSet::new();
        set.try_insert(std::string::String::from("apple")).unwrap();
        set.try_insert(std::string::String::from("banana")).unwrap();
        set.try_insert(std::string::String::from("cherry")).unwrap();
        // Remove via &str probe exercises the real Q != T path.
        let removed = set.take("banana").expect("should find banana");
        assert_eq!(removed.as_str(), "banana");
        assert!(!set.contains("banana"));
        assert!(set.contains("apple"));
        assert!(set.contains("cherry"));
        assert_eq!(set.take("durian"), None);
    }

    #[test]
    fn remove_triggers_root_shrink() {
        let mut set = BTreeSet::new();
        for i in 0..50u32 {
            set.try_insert(i).unwrap();
        }
        while set.len() > 1 {
            let next = *set.map.first_key_value().unwrap().0;
            assert_eq!(set.take(&next), Some(next));
        }
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn remove_drops_removed_item_once_no_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(15, &ledger);
        // Probe with the raw inner (TrackedItem borrows through its inner).
        let removed = set.take(&5u32).expect("inner 5 should exist");
        assert_eq!(removed.inner, 5);
        drop(removed);
        // 14 items remain in the set; only double-frees matter here.
        assert!(ledger.double_dropped().is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn remove_drains_to_empty_ledger_clean() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set(20, &ledger);
        for i in 0..20u32 {
            let removed = set.take(&i).expect("value should exist");
            assert_eq!(removed.inner, i);
            // Second removal of the same inner must miss.
            assert!(set.take(&i).is_none());
            drop(removed);
            // Remaining items are still live; only double-frees matter mid-drain.
            assert!(ledger.double_dropped().is_empty());
        }
        assert!(set.is_empty());
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.double_dropped().is_empty());
    }

    #[test]
    fn remove_scrambled_preserves_invariant() {
        let rng = TestRng::new(0xC0FFEE);
        let mut set = BTreeSet::new();
        let n = 60u32;
        for i in 0..n {
            set.try_insert(i).unwrap();
        }
        for v in rng.permuted(0..n) {
            assert_eq!(set.take(&v), Some(v));
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
        }
        assert!(set.is_empty());
    }

    #[test]
    fn remove_half_preserves_invariant_and_values() {
        let mut set = BTreeSet::new();
        for i in 0..40u32 {
            set.try_insert(i).unwrap();
        }
        for i in (0..40u32).step_by(2) {
            assert_eq!(set.take(&i), Some(i));
        }
        assert_eq!(set.len(), 20);
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        for i in (0..40u32).filter(|x| x % 2 == 1) {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn remove_custom_order_multilevel_schedule() {
        let mut set = BTreeSet::new();
        for i in 0..80u32 {
            set.try_insert(i).unwrap();
        }
        // Deterministic scrambled order: stride 31 is coprime to 80, hitting
        // interior keys alongside the extremes across all levels.
        let rng = TestRng::new(0xC0FFEE);
        let order: std::vec::Vec<u32> = rng.permuted(0..80u32).collect();
        let mut seen = [false; 80];
        for &k in &order {
            assert!(!seen[k as usize], "duplicate in removal schedule");
            seen[k as usize] = true;
            assert_eq!(set.take(&k), Some(k));
            // The value is gone immediately: a second remove misses.
            assert!(set.take(&k).is_none());
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
        }
        assert!(set.is_empty());
    }

    // ── retain ─────────────────────────────────────────────────────────────────

    #[test]
    fn retain_empty_set() {
        let mut set = BTreeSet::<i32>::new();
        set.retain(|_| true);
        assert!(set.is_empty());
    }

    #[test]
    fn retain_keeps_matching_only() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        set.retain(|v| v % 2 == 1);
        assert_eq!(set.len(), 5);
        for i in (0..10u32).filter(|i| i % 2 == 1) {
            assert!(set.contains(&i), "kept {i} lost");
        }
        for i in (0..10u32).filter(|i| i % 2 == 0) {
            assert!(!set.contains(&i), "filtered-out {i} still present");
        }
    }

    #[test]
    fn retain_keeps_none() {
        let mut set = BTreeSet::new();
        for i in 0..20u32 {
            set.try_insert(i).unwrap();
        }
        set.retain(|_| false);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn retain_keeps_all() {
        let mut set = BTreeSet::new();
        for i in 0..20u32 {
            set.try_insert(i).unwrap();
        }
        set.retain(|_| true);
        assert_eq!(set.len(), 20);
        for i in 0..20u32 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn retain_multilevel() {
        let mut set = BTreeSet::new();
        for i in 0..300u32 {
            set.try_insert(i).unwrap();
        }
        assert!(set.map.root.as_ref().is_some_and(|r| r.height() >= 2));
        set.retain(|v| v % 3 != 0);
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        assert_eq!(set.len(), 200);
        for i in 0..300u32 {
            if i % 3 == 0 {
                assert!(!set.contains(&i), "{i} should be retained out");
            } else {
                assert!(set.contains(&i), "survivor {i} damaged");
            }
        }
    }

    #[test]
    fn retain_scrambled_schedule_preserves_invariant() {
        let mut set = BTreeSet::new();
        const N: u32 = 120;
        for i in 0..N {
            set.try_insert(i).unwrap();
        }
        let rng = TestRng::new(0xCAFE_F00D_DEAD_BEED);
        let keep_set: std::collections::BTreeSet<u32> =
            rng.permuted(0..N).take((N / 3) as usize).collect();
        set.retain(|v| keep_set.contains(v));
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        assert_eq!(set.len(), keep_set.len());
        for &v in &keep_set {
            assert!(set.contains(&v), "keeper {v} lost");
        }
        for i in 0..N {
            if !keep_set.contains(&i) {
                assert!(!set.contains(&i), "{i} should be pruned");
            }
        }
    }
}
