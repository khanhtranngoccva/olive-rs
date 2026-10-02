//! Conditional removal for [`BTreeSet`]: `extract_if`.
//!
//! Both are thin facades over the underlying [`BTreeMap`] counterparts; each
//! hides the zero-sized value marker so callers only ever see values. The set
//! variant of the extractor wraps the map's [`super::super::extract_if::ExtractIfInner`]
//! and adapts its `(K, SetValZST)` items down to bare `K`s.

use super::super::extract_if::ExtractIfInner;
use super::super::set_val::SetValZST;
use super::BTreeSet;
use crate::alloc::Allocator;
use core::fmt;
use core::iter::FusedIterator;
use core::ops::RangeBounds;

/// An iterator produced by calling [`BTreeSet::extract_if`](super::BTreeSet::extract_if).
///
/// Wraps the map-level [`ExtractIfInner`], yielding just the extracted values
/// (the keys of the underlying map) instead of key-marker pairs.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, T, R, F, A: Allocator>
where
    F: 'a + FnMut(&T) -> bool,
{
    pred: F,
    inner: ExtractIfInner<'a, T, SetValZST, R>,
    alloc: &'a A,
}

impl<T, R, F, A: Allocator> Iterator for ExtractIf<'_, T, R, F, A>
where
    T: PartialOrd,
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        // The marker is a ZST carrying no data, so it is discarded on unwrap.
        let pred = &mut self.pred;
        self.inner
            .next(&mut |k: &T, _: &mut SetValZST| pred(k), self.alloc)
            .map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, R, F, A: Allocator> FusedIterator for ExtractIf<'_, T, R, F, A>
where
    T: PartialOrd,
    R: RangeBounds<T>,
    F: FnMut(&T) -> bool,
{
}

impl<T, R, F, A: Allocator> fmt::Debug for ExtractIf<'_, T, R, F, A>
where
    T: fmt::Debug,
    F: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf")
            .field("peek", &self.inner.peek().map(|(k, _)| k))
            .finish_non_exhaustive()
    }
}

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Creates an iterator that extracts all elements matching the given
    /// predicate from this set and lying within the specified range.
    ///
    /// The returned iterator yields the extracted values in ascending order.
    ///
    /// # Panics
    ///
    /// On panic, this iterator stops functioning and yields no more values.
    pub fn extract_if<R, F>(&mut self, range: R, pred: F) -> ExtractIf<'_, T, R, F, A>
    where
        R: RangeBounds<T>,
        F: FnMut(&T) -> bool,
    {
        let (inner, alloc) = self.map.extract_if_inner(range);
        ExtractIf { pred, inner, alloc }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::collections::btree::invariant::{check_ascending_keys, check_tree_invariant};
    use crate::test_helpers::{Ledger, TestRng, TrackedItem};

    type TestSet = BTreeSet<TrackedItem<u32>>;

    /// Builds a set of tracked items whose inners are `0..n`, registering each
    /// in the ledger up front so drops are observable.
    fn build_set_with_ledger(n: u32, ledger: &std::sync::Arc<Ledger>) -> TestSet {
        let mut set = TestSet::new();
        for i in 0..n {
            set.try_insert(TrackedItem::construct(ledger, i)).unwrap();
        }
        set
    }

    /// Builds a set with integer values from a slice.
    fn build_set_from_slice(values: &[u32]) -> BTreeSet<u32> {
        let mut set = BTreeSet::new();
        for &v in values {
            set.try_insert(v).unwrap();
        }
        set
    }

    // ── Basic behavior ─────────────────────────────────────────────────────────

    #[test]
    fn extract_if_empty_set() {
        let mut set = BTreeSet::<i32>::new();
        let extracted: std::vec::Vec<i32> = set.extract_if(.., |_| true).collect();
        assert!(extracted.is_empty());
        assert!(set.is_empty());
    }

    #[test]
    fn extract_if_no_match() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        let extracted: std::vec::Vec<u32> = set.extract_if(.., |v| *v > 100).collect();
        assert!(extracted.is_empty());
        // Nothing was touched: all values survive intact.
        assert_eq!(set.len(), 10);
        for i in 0..10u32 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn extract_if_all_match() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        let extracted: std::vec::Vec<u32> = set.extract_if(.., |_| true).collect();
        assert_eq!(extracted, (0..10u32).collect::<std::vec::Vec<_>>());
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn extract_if_partial_matches() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        // Evens out.
        let extracted: std::vec::Vec<u32> = set.extract_if(.., |v| v % 2 == 0).collect();
        assert_eq!(
            extracted,
            [0, 2, 4, 6, 8].to_vec(),
            "extraction must yield matching values in ascending order"
        );
        // Odds remain, untouched.
        assert_eq!(set.len(), 5);
        for i in (0..10u32).filter(|i| i % 2 == 1) {
            assert!(set.contains(&i), "survivor {i} damaged");
        }
        for i in (0..10u32).filter(|i| i % 2 == 0) {
            assert!(!set.contains(&i), "extracted {i} still present");
        }
    }

    #[test]
    fn extract_if_single_element() {
        let mut set = BTreeSet::new();
        set.try_insert(7).unwrap();
        let extracted: std::vec::Vec<i32> = set.extract_if(.., |v| *v == 7).collect();
        assert_eq!(extracted, [7]);
        assert!(set.is_empty());
    }

    // ── Iterator semantics ─────────────────────────────────────────────────────

    #[test]
    fn extract_if_size_hint_bounds() {
        let mut set = BTreeSet::new();
        for i in 0..20u32 {
            set.try_insert(i).unwrap();
        }
        let mut iter = set.extract_if(.., |v| v % 2 == 0);
        // Lower bound is always 0; upper bound starts at the set length.
        assert_eq!(iter.size_hint(), (0, Some(20)));
        // Consume two matches (values 0 and 2): the upper bound tracks the live
        // length, which has dropped to 18.
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.size_hint(), (0, Some(18)));
        drop(iter);
        // The set itself reflects the live length the hint was tracking.
        assert_eq!(set.len(), 18);
    }

    #[test]
    fn extract_if_size_hint_exhausted() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        let collected: std::vec::Vec<u32> = set.extract_if(.., |_| true).collect();
        assert_eq!(collected.len(), 10);
        let mut empty_iter = set.extract_if(.., |_| true);
        assert_eq!(empty_iter.size_hint(), (0, Some(0)));
        assert_eq!(empty_iter.next(), None);
    }

    #[test]
    fn extract_if_fused_iterator() {
        let mut set = BTreeSet::new();
        for i in 0..5u32 {
            set.try_insert(i).unwrap();
        }
        let mut iter = set.extract_if(.., |_| true);
        assert_eq!(iter.next(), Some(0));
        // Exhaust.
        for _ in iter.by_ref() {}
        // A fused iterator keeps returning None once exhausted.
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn extract_if_lazy() {
        // Nothing happens until the iterator is driven: abandoning it after a
        // single step leaves the remainder in the set.
        let mut set = BTreeSet::new();
        for i in 0..5u32 {
            set.try_insert(i).unwrap();
        }
        let mut iter = set.extract_if(.., |_| true);
        assert_eq!(iter.next(), Some(0));
        drop(iter);
        // Abandoning the iterator mid-way leaves the remainder in the set.
        assert_eq!(set.len(), 4);
        assert!(!set.contains(&0));
        for i in 1..5u32 {
            assert!(set.contains(&i));
        }
    }

    // ── Memory safety: every extracted payload dies exactly once ───────────────

    #[test]
    fn extract_if_drops_extracted_items_once_no_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set_with_ledger(50, &ledger);
        // Extract evens among 0..40: 20 payloads.
        let mut extracted: std::vec::Vec<u32> = std::vec::Vec::new();
        for v in set.extract_if(.., |item| item.inner % 2 == 0 && item.inner < 40) {
            extracted.push(v.inner);
        }
        assert_eq!(extracted.len(), 20);
        assert_eq!(set.len(), 30);
        // Each extracted item died exactly once; nothing double-freed.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        let counts = ledger.drop_counts();
        assert_eq!(counts.values().sum::<usize>(), 20);
        assert!(
            counts.values().all(|&c| c == 1),
            "some payload dropped != once: {counts:?}"
        );
        // Survivors are still live: 30 items, none of them dead yet.
        assert_eq!(
            ledger.live_ids().len(),
            30,
            "unexpected live ids: {:?}",
            ledger.live_ids()
        );
        // Dropping the set frees the rest; everything dies exactly once.
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..50));
    }

    #[test]
    fn extract_if_full_drain_ledger_clean() {
        // Extracting everything must leave no payload alive and no double-free.
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set_with_ledger(60, &ledger);
        let collected: std::vec::Vec<u32> = set.extract_if(.., |_| true).map(|v| v.inner).collect();
        assert_eq!(collected.len(), 60);
        assert!(set.is_empty());
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
        assert!(ledger.all_dropped_once(0..60));
    }

    #[test]
    fn extract_if_abandoned_mid_iteration_survivors_stay_live() {
        // Stop iterating early: extracted items die, survivors stay live until
        // the set itself drops — proving partial consumption is memory-clean.
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut set = build_set_with_ledger(30, &ledger);
        let mut iter = set.extract_if(.., |_| true);
        // Take exactly three items.
        for i in 0..3u32 {
            let v = iter.next().expect("expected an item");
            assert_eq!(v.inner, i);
        }
        drop(iter);
        assert_eq!(set.len(), 27);
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(
            ledger.drop_counts().values().sum::<usize>(),
            3,
            "only the 3 taken items should be dead"
        );
        assert_eq!(
            ledger.live_ids().len(),
            27,
            "unexpected live ids: {:?}",
            ledger.live_ids()
        );
        drop(set);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..30));
    }

    // ── Structural invariants across multi-level trees ─────────────────────────

    /// Partial extraction from a multi-level tree
    #[test]
    fn extract_if_multilevel_preserves_invariant() {
        let mut set = BTreeSet::new();
        const N: u32 = 200;
        for i in 0..N {
            set.try_insert(i).unwrap();
        }
        assert!(
            set.map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        // Pull out every third value — interleaved across all levels.
        let extracted: std::vec::Vec<u32> = set.extract_if(.., |v| v % 3 == 0).collect();
        assert_eq!(extracted.len(), 67);
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        assert_eq!(set.len(), N as usize - extracted.len());
        for i in 0..N {
            if i % 3 == 0 {
                assert!(!set.contains(&i), "{i} should be extracted");
            } else {
                assert!(set.contains(&i), "survivor {i} damaged");
            }
        }
    }

    /// Drains a multi-level tree through extract_if one element at a time
    #[test]
    fn extract_if_drain_one_by_one_preserves_invariant() {
        let mut set = BTreeSet::new();
        const N: u32 = 200;
        for i in 0..N {
            set.try_insert(i).unwrap();
        }
        assert!(
            set.map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        let drained: std::vec::Vec<u32> = set.extract_if(.., |_| true).collect();
        assert_eq!(drained.len(), N as usize);
        for (i, v) in drained.iter().enumerate() {
            assert_eq!(*v, i as u32, "wrong value at step {i}");
            if i % 50 == 0 {
                check_tree_invariant(&set.map);
                check_ascending_keys(&set.map);
            }
        }
        assert!(set.is_empty());
    }

    /// Randomized (deterministic PRNG) partial extractions over many rounds:
    /// each round extracts a random subset of the remaining values, verifying
    /// the invariant and exact survivor set afterward.
    #[test]
    fn extract_if_random_subsets_preserve_invariant() {
        let mut set = BTreeSet::new();
        const N: u32 = 150;
        for i in 0..N {
            set.try_insert(i).unwrap();
        }
        let mut rng = TestRng::new(0xE7CABEEFCAFED00D);
        let mut expected: std::collections::BTreeSet<u32> = (0..N).collect();
        for round in 0..10u32 {
            // A deterministic hash-derived predicate computed outside the
            // closure so it can be re-checked after extraction.
            let threshold = (rng.next_u64() % 97) as u32;
            let seed = rng.next_u64();
            let seed32 = seed as u32;
            let pred = |v: u32| -> bool {
                let h = v
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(seed32)
                    .wrapping_mul(0xBF58_476D);
                (h >> 16) % 100 < threshold
            };
            let extracted: std::vec::Vec<u32> = set.extract_if(.., |v| pred(*v)).collect();
            for v in &extracted {
                assert!(pred(*v), "extracted {v} does not match its own predicate");
                assert!(expected.remove(v), "extracted {v} not expected");
            }
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
            assert_eq!(set.len(), expected.len(), "round {round}: length mismatch");
            for &v in &expected {
                assert!(set.contains(&v), "survivor {v} damaged");
            }
            if expected.is_empty() {
                break;
            }
        }
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
    }

    /// Extraction followed by re-insertion of new values must not disturb the
    /// tree: the classic churn pattern that stresses rebalancing.
    #[test]
    fn extract_if_then_reinsert_churn() {
        let mut set = BTreeSet::new();
        for i in 0..80u32 {
            set.try_insert(i).unwrap();
        }
        for _ in 0..5u32 {
            // Extract the lower half of whatever remains, then re-insert shifted values.
            let midpoint = set.map.last_key_value().map_or(0, |(k, _)| *k / 2);
            let n_extracted = set.extract_if(.., |v| *v < midpoint).count();
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
            for j in 0..n_extracted as u32 {
                set.try_insert(midpoint + 1 + j * 3).unwrap();
            }
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
        }
        // Whatever the mix, the tree is structurally sound and sorted.
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        assert!(!set.is_empty());
    }

    // ── Ranged extract_if ──────────────────────────────────────────────────────

    #[test]
    fn extract_if_range_middle() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        // Extract only within [3, 8): values 3..=7 that match the predicate.
        // The exclusive end bound excludes 8 even though it matches the predicate.
        let extracted: std::vec::Vec<u32> = set.extract_if(3..8, |v| v % 2 == 0).collect();
        assert_eq!(extracted, std::vec![4, 6]);
        // Values outside the range survive untouched; inside-range non-matches too.
        assert_eq!(set.len(), 8);
        for &v in &[0u32, 1, 2, 3, 5, 7, 8, 9] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
        for &v in &[4u32, 6] {
            assert!(!set.contains(&v), "extracted {v} still present");
        }
    }

    #[test]
    fn extract_if_range_inclusive_start_bound() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        let extracted: std::vec::Vec<u32> = set.extract_if(4.., |_| true).collect();
        assert_eq!(extracted, (4..10).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 4);
        for &v in &[0u32, 1, 2, 3] {
            assert!(set.contains(&v));
        }
    }

    #[test]
    fn extract_if_range_exclusive_start_bound() {
        use core::ops::Bound;
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        // Exclusive lower bound: start after value 4, take everything matching.
        let range = (Bound::Excluded(4u32), Bound::Unbounded);
        let extracted: std::vec::Vec<u32> = set.extract_if(range, |_| true).collect();
        assert_eq!(extracted, (5..10).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 5);
        for &v in &[0u32, 1, 2, 3, 4] {
            assert!(set.contains(&v));
        }
    }

    #[test]
    fn extract_if_range_inclusive_start_bound_but_value_absent() {
        let mut set = BTreeSet::new();
        for i in 0..10u32 {
            set.try_insert(i).unwrap();
        }
        set.take(&4);
        let extracted: std::vec::Vec<u32> = set.extract_if(4.., |_| true).collect();
        assert_eq!(extracted, (5..10).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 4);
        for &v in &[0u32, 1, 2, 3] {
            assert!(set.contains(&v));
        }
    }

    #[test]
    fn extract_if_upper_included_existing_end_key() {
        // End bound Included(7) where 7 is present: 7 must be extracted too.
        let mut set = build_set_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let extracted: std::vec::Vec<u32> = set.extract_if(..=7, |_| true).collect();
        assert_eq!(extracted, (0..=7).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 2);
        for &v in &[8u32, 9] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
    }

    #[test]
    fn extract_if_upper_excluded_existing_end_key() {
        // End bound Excluded(7) where 7 is present: 7 must NOT be extracted.
        let mut set = build_set_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let extracted: std::vec::Vec<u32> = set.extract_if(..7, |_| true).collect();
        assert_eq!(extracted, (0..7).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 3);
        for &v in &[7u32, 8, 9] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
    }

    #[test]
    fn extract_if_upper_unbounded_drains_to_top() {
        // No end bound: everything from the start bound up to the last element.
        let mut set = build_set_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let extracted: std::vec::Vec<u32> = set.extract_if(3.., |_| true).collect();
        assert_eq!(extracted, (3..10).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 3);
        for &v in &[0u32, 1, 2] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
    }

    #[test]
    fn extract_if_upper_included_missing_end_key() {
        // End bound Included(7) but 7 is absent; 6 is the largest present below it.
        // 6 must still be extracted; nothing above 7 exists to be touched.
        let mut set = build_set_from_slice(&[0, 1, 2, 3, 4, 5, 6, 8, 9]);
        let extracted: std::vec::Vec<u32> = set.extract_if(..=7, |_| true).collect();
        assert_eq!(extracted, (0..=6).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 2);
        for &v in &[8u32, 9] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
    }

    #[test]
    fn extract_if_upper_excluded_missing_end_key() {
        // End bound Excluded(7) but 7 is absent; 6 is the largest present below it.
        // Same visible outcome as the included case here, but exercised through
        // the exclusive branch of the termination check.
        let mut set = build_set_from_slice(&[0, 1, 2, 3, 4, 5, 6, 8, 9]);
        let extracted: std::vec::Vec<u32> = set.extract_if(..7, |_| true).collect();
        assert_eq!(extracted, (0..=6).collect::<std::vec::Vec<_>>());
        assert_eq!(set.len(), 2);
        for &v in &[8u32, 9] {
            assert!(set.contains(&v), "survivor {v} missing");
        }
    }

    #[test]
    fn extract_if_upper_bound_multilevel_matrix() {
        const N: u32 = 300;
        const END_VAL: u32 = 150;
        let cases: [(&str, bool); 4] = [
            ("included_existing", true),
            ("excluded_existing", true),
            ("included_missing", false),
            ("excluded_missing", false),
        ];
        for (cell, present) in cases {
            let mut set = BTreeSet::new();
            for i in 0..N {
                if i != END_VAL || present {
                    set.try_insert(i).unwrap();
                }
            }
            assert!(set.map.root.as_ref().is_some_and(|r| r.height() >= 2));

            let included = cell.starts_with("included");
            let extracted: std::vec::Vec<u32> = if included {
                set.extract_if(..=END_VAL, |_| true).collect()
            } else {
                set.extract_if(..END_VAL, |_| true).collect()
            };

            let expected_hi = if included && present {
                END_VAL
            } else {
                END_VAL - 1
            };
            let want: std::vec::Vec<u32> = (0..=expected_hi).collect();
            assert_eq!(extracted, want, "matrix cell ({cell}) mismatched");
            check_tree_invariant(&set.map);
            check_ascending_keys(&set.map);
        }
    }

    #[test]
    fn extract_if_range_unbounded_both_sides_is_full() {
        let mut set = BTreeSet::new();
        for i in 0..6u32 {
            set.try_insert(i).unwrap();
        }
        let collected: std::vec::Vec<u32> = set.extract_if(.., |_| true).collect();
        assert_eq!(collected, (0..6).collect::<std::vec::Vec<_>>());
        assert!(set.is_empty());
    }

    #[test]
    fn extract_if_range_multilevel_preserves_invariant() {
        let mut set = BTreeSet::new();
        const N: u32 = 200;
        for i in 0..N {
            set.try_insert(i).unwrap();
        }
        assert!(set.map.root.as_ref().is_some_and(|r| r.height() >= 2));
        // Extract a middle band: values 50..=149 that are multiples of 3.
        let extracted: std::vec::Vec<u32> = set.extract_if(50..150, |v| v % 3 == 0).collect();
        for &v in &extracted {
            assert!(
                (50..150).contains(&v) && v % 3 == 0,
                "out-of-band or mismatching {v}"
            );
        }
        check_tree_invariant(&set.map);
        check_ascending_keys(&set.map);
        assert_eq!(set.len(), N as usize - extracted.len());
        for i in 0..N {
            if (50..150).contains(&i) && i % 3 == 0 {
                assert!(!set.contains(&i), "{i} should be extracted");
            } else {
                assert!(set.contains(&i), "survivor {i} missing");
            }
        }
    }
}
