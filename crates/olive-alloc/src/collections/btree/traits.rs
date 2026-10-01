//! Trait implementations for [`BTreeMap`].
//!
//! Covers standard-library trait impls ([`Debug`](fmt::Debug), [`PartialEq`]/[`Eq`],
//! [`PartialOrd`]/[`Ord`], [`Hash`]) and the fallible construction traits from
//! `olive-core` ([`TryClone`], [`TryDefault`]).

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use super::TryBTreeMapWithCloneError;
use super::map::BTreeMap;
use crate::alloc::{AllocError, Allocator, AllocatorTryClone, Global};

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<K: fmt::Debug, V: fmt::Debug, A: Allocator> fmt::Debug for BTreeMap<K, V, A> {
    /// Formats the map as `{ key: value, ... }` in ascending key order.
    /// An empty map renders as `{}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_map();
        for (k, v) in self.iter() {
            d.entry(k, v);
        }
        d.finish()
    }
}

// ---------------------------------------------------------------------------
// PartialEq / Eq
// ---------------------------------------------------------------------------

impl<K: Ord + PartialEq, V: PartialEq, A: Allocator> PartialEq for BTreeMap<K, V, A> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        // Both maps have the same length; compare entries pairwise in key order.
        // Since both are sorted by key, zipping their iterators compares matching keys.
        self.iter()
            .zip(other.iter())
            .all(|((ak, av), (bk, bv))| ak == bk && av == bv)
    }
}

impl<K: Ord + Eq, V: Eq, A: Allocator> Eq for BTreeMap<K, V, A> {}

// ---------------------------------------------------------------------------
// PartialOrd / Ord
// ---------------------------------------------------------------------------

impl<K: Ord + PartialOrd, V: PartialOrd, A: Allocator> PartialOrd for BTreeMap<K, V, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Lexicographic comparison of (key, value) pairs in ascending key order.
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord, A: Allocator> Ord for BTreeMap<K, V, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

impl<K: Ord + Hash, V: Hash, A: Allocator> Hash for BTreeMap<K, V, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the length first so that a map which is a strict prefix of
        // another cannot collide with it.
        self.len().hash(state);
        for (k, v) in self.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<K: Ord + TryClone, V: TryClone, A: AllocatorTryClone> TryClone for BTreeMap<K, V, A> {
    /// Fallibly clones the map, entry by entry via [`TryClone`].
    ///
    /// The backing allocator is cloned first so the result lives on an
    /// equivalent allocator. Entries are inserted in ascending key order;
    /// because `try_insert` is fallible, any allocation failure during the
    /// rebuild propagates and the partially-built map is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator fails, or if
    /// inserting a cloned entry fails due to an allocation error.
    ///
    /// Note: std's `BTreeMap::clone` walks the tree structurally and copies
    /// nodes wholesale (avoiding per-key rebalancing). This port rebuilds via
    /// `try_insert` for simplicity; a structural copy can be added later.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = self.alloc.try_clone()?;
        let mut out = Self::new_in(alloc);
        for (k, v) in self.iter() {
            let ck = k.try_clone()?;
            let cv = v.try_clone()?;
            // `try_insert` returns `AllocError`; convert via `From<AllocError>`.
            out.try_insert(ck, cv).map_err(TryCloneError::from)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryFromIterator
// ---------------------------------------------------------------------------

impl<K: Ord, V> TryFromIterator<(K, V)> for BTreeMap<K, V, Global> {
    type Error = AllocError;

    /// Fallibly collect an iterator of key-value pairs into a [`BTreeMap`] on
    /// the default [`Global`] allocator. For a custom allocator use
    /// [`BTreeMap::try_from_iter_in`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if an internal allocation fails during insertion.
    fn try_from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

impl<K: Ord, V, A: Allocator> BTreeMap<K, V, A> {
    /// Fallibly collects an iterator of key-value pairs into a [`BTreeMap`] on
    /// the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if reserving capacity for an incoming element
    /// fails. On failure the partially constructed map is discarded.
    pub fn try_from_iter_in<I: IntoIterator<Item = (K, V)>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, AllocError> {
        let mut map = Self::new_in(alloc);
        for (k, v) in iter {
            map.try_insert(k, v)?;
        }
        Ok(map)
    }
}

// ---------------------------------------------------------------------------
// TryExtend / TryExtendFromSlice
// ---------------------------------------------------------------------------

impl<K: Ord, V, A: Allocator> TryExtend<(K, V)> for BTreeMap<K, V, A> {
    type Error = AllocError;

    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = (K, V)>,
    {
        let (head, mut inner, _hint) = source.decompose_with_size_hint();

        // Insert the stranded head first, if any.
        if let Some((k, v)) = head {
            if let Err((k, v, _e)) = self.try_insert_give_back(k, v) {
                return Err((Resume::new((k, v), inner), AllocError));
            }
        }

        // Insert the remainder one pair at a time. Each insertion is atomic:
        // on allocation failure the tree is left unmodified and we strand the
        // current pair in a `Resume` for retry.
        while let Some((k, v)) = inner.next() {
            if let Err((k, v, _e)) = self.try_insert_give_back(k, v) {
                return Err((Resume::new((k, v), inner), AllocError));
            }
        }
        Ok(())
    }
}

impl<K: Ord + TryClone, V: TryClone, A: Allocator> TryExtendFromSlice<(K, V)>
    for BTreeMap<K, V, A>
{
    type Error = TryBTreeMapWithCloneError;

    fn try_extend_from_slice<'s>(
        &mut self,
        other: &'s [(K, V)],
    ) -> Result<(), (&'s [(K, V)], Self::Error)> {
        let mut i = 0usize;
        for (k, v) in other {
            let cloned_k = k
                .try_clone()
                .map_err(|e| (&other[i..], TryBTreeMapWithCloneError::Clone(e)))?;
            let cloned_v = v
                .try_clone()
                .map_err(|e| (&other[i..], TryBTreeMapWithCloneError::Clone(e)))?;
            match self.try_insert_give_back(cloned_k, cloned_v) {
                Ok(_) => {
                    #[allow(clippy::arithmetic_side_effects, reason = "asserted i < other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(_) => {
                    return Err((&other[i..], TryBTreeMapWithCloneError::Alloc(AllocError)));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::super::invariant::{check_ascending_keys, check_tree_invariant};

    use super::*;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use std::format;

    // --- Debug ---------------------------------------------------------------

    #[test]
    fn debug_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert_eq!(format!("{map:?}"), "{}");
    }

    #[test]
    fn debug_nonempty_map() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        map.try_insert(2, 20).unwrap();
        map.try_insert(3, 30).unwrap();
        assert_eq!(format!("{map:?}"), "{1: 10, 2: 20, 3: 30}");
    }

    #[test]
    fn debug_multilevel_map() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i as i32, (i * 2) as i32).unwrap();
        }
        let s = format!("{map:?}");
        // Should start with '{' and end with '}'
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
        // First entry should be 0: 0
        assert!(s.contains("0: 0"));
        // Last entry should be 79: 158
        assert!(s.contains("79: 158"));
    }

    // --- PartialEq / Eq -------------------------------------------------------

    #[test]
    fn partial_eq_same_contents() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..10 {
            a.try_insert(i, i * 10).unwrap();
            b.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(a, b);
    }

    #[test]
    fn partial_eq_different_lengths_not_equal() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..10 {
            a.try_insert(i, i).unwrap();
            b.try_insert(i, i).unwrap();
        }
        a.try_insert(10, 100).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn partial_eq_different_values_not_equal() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(1, 20).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn partial_eq_both_empty() {
        let a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        assert_eq!(a, b);
    }

    #[test]
    fn partial_eq_multilevel_same_contents() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            a.try_insert(i as i32, (i * 3) as i32).unwrap();
            b.try_insert(i as i32, (i * 3) as i32).unwrap();
        }
        assert_eq!(a, b);
    }

    // --- PartialOrd / Ord ------------------------------------------------------

    #[test]
    fn partial_ord_less_than() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(2, 20).unwrap();
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn partial_ord_equal_maps() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(1, 10).unwrap();
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_eq!(a.cmp(&b), Ordering::Equal);
    }

    #[test]
    fn partial_ord_prefix_is_less() {
        // A map that is a prefix of another (fewer entries, all matching)
        // should sort less.
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(1, 10).unwrap();
        b.try_insert(2, 20).unwrap();
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn ord_multilevel() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            a.try_insert(i as i32, (i * 2) as i32).unwrap();
            b.try_insert(i as i32, (i * 2) as i32).unwrap();
        }
        assert_eq!(a.cmp(&b), Ordering::Equal);
        // Modify one value to make them differ.
        b.try_insert(40, 9999).unwrap();
        assert_ne!(a.cmp(&b), Ordering::Equal);
    }

    // --- Hash ------------------------------------------------------------------

    #[test]
    fn hash_equal_maps_produce_equal_hashes() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..20 {
            a.try_insert(i, i * 10).unwrap();
            b.try_insert(i, i * 10).unwrap();
        }
        let mut ha = std::collections::hash_map::DefaultHasher::default();
        let mut hb = std::collections::hash_map::DefaultHasher::default();
        use std::hash::Hasher;
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn hash_different_maps_produce_different_hashes() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        a.try_insert(1, 10).unwrap();
        b.try_insert(1, 20).unwrap();
        let mut ha = std::collections::hash_map::DefaultHasher::default();
        let mut hb = std::collections::hash_map::DefaultHasher::default();
        use std::hash::Hasher;
        a.hash(&mut ha);
        b.hash(&mut hb);
        // Extremely unlikely to collide for different values.
        assert_ne!(ha.finish(), hb.finish());
    }

    // --- TryClone --------------------------------------------------------------

    #[test]
    fn try_clone_empty_map() {
        let map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let cloned = map.try_clone().expect("clone ok");
        assert!(cloned.is_empty());
    }

    #[test]
    fn try_clone_preserves_entries() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let cloned = map.try_clone().expect("clone ok");
        assert_eq!(cloned.len(), 20);
        for i in 0..20 {
            assert_eq!(cloned.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn try_clone_multilevel() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i as i32, (i * 3) as i32).unwrap();
        }
        let cloned = map.try_clone().expect("clone ok");
        assert_eq!(cloned.len(), 80);
        for i in 0..80u32 {
            assert_eq!(cloned.get(&(i as i32)), Some(&((i * 3) as i32)));
        }
    }

    #[test]
    fn try_clone_original_unchanged() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i).unwrap();
        }
        let mut cloned = map.try_clone().expect("clone ok");
        cloned.try_insert(12, 12).unwrap();
        // Original must be intact.
        assert_eq!(map.len(), 10);
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn try_clone_fails_when_allocator_clone_exhausted() {
        use crate::test_helpers::{CloneBudget, FlakyCloneAlloc};
        use std::sync::Arc;

        let budget = Arc::new(CloneBudget::new(0));
        let alloc = FlakyCloneAlloc::new(budget.clone());
        let mut map: BTreeMap<i32, i32, FlakyCloneAlloc> = BTreeMap::new_in(alloc);
        map.try_insert(1, 10).unwrap();

        // Cloning the allocator must fail with zero budget.
        let result = map.try_clone();
        assert!(
            result.is_err(),
            "clone should fail when allocator clone is exhausted"
        );
    }

    #[test]
    fn try_clone_fails_midway_on_allocation_error() {
        use crate::test_helpers::BudgetedAlloc;

        // Strategy: build a multi-entry map under a BudgetedAlloc with enough
        // budget to construct it, then drain the remaining budget before
        // calling try_clone. The clone succeeds at cloning the allocator
        // (Arc clone is infallible) and begins re-inserting entries into a
        // fresh map, but the fresh map's node allocations will OOM once the
        // shared budget is exhausted.
        let alloc = BudgetedAlloc::new(5);
        let mut map: BTreeMap<i32, i32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        for i in 0..20i32 {
            match map.try_insert(i, i * 2) {
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let len_before = map.len();
        assert!(
            len_before > 1,
            "need multiple entries to exercise the clone path"
        );

        alloc.drain();
        // DEBUG: check map state before extend
        let result = map.try_clone();
        assert!(
            result.is_err(),
            "clone should fail when budget is exhausted"
        );
        assert!(matches!(result.unwrap_err(), TryCloneError::Alloc(_)));

        // Original must still be intact.
        assert_eq!(map.len(), len_before);
        for i in 0..len_before as i32 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn try_clone_preserves_original_on_failure() {
        use crate::test_helpers::{CloneBudget, FlakyCloneAlloc};
        use std::sync::Arc;

        // Build a map under a flaky allocator with enough budget for inserts,
        // then exhaust the budget so try_clone fails.
        let budget = Arc::new(CloneBudget::new(5));
        let alloc = FlakyCloneAlloc::new(budget.clone());
        let mut map: BTreeMap<i32, i32, FlakyCloneAlloc> = BTreeMap::new_in(alloc);
        for i in 0..4 {
            map.try_insert(i, i).unwrap();
        }

        // Manually exhaust any remaining budget units.
        while budget.try_consume() {}

        map.try_clone()
            .expect_err("should fail to clone the allocator"); // fails at allocator clone

        // Original must be completely intact.
        assert_eq!(map.len(), 4);
        for i in 0..4 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    // --- TryFromIterator ------------------------------------------------------

    #[test]
    fn try_from_iterator_builds_sorted_map() {
        // Insertion order is deliberately scrambled; the result must be keyed
        // in ascending order regardless of iterator order.
        let pairs: std::vec::Vec<(u32, u32)> = [3, 0, 5, 1, 4, 2]
            .into_iter()
            .map(|k| (k, k * 10))
            .collect();
        let map: BTreeMap<u32, u32> = TryFromIterator::try_from_iter(pairs).expect("iter ok");
        assert_eq!(map.len(), 6);
        let keys: std::vec::Vec<u32> = map.keys().copied().collect();
        assert_eq!(keys, [0, 1, 2, 3, 4, 5]);
        for (k, v) in map.iter() {
            assert_eq!(*v, *k * 10);
        }
    }

    #[test]
    fn try_from_iterator_empty_is_empty() {
        let map: BTreeMap<u32, u32> =
            TryFromIterator::try_from_iter(std::iter::empty()).expect("ok");
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn try_from_iterator_duplicate_keys_last_wins() {
        // Mirrors `Extend` semantics: a repeated key keeps its final value.
        let pairs = [(1u32, 100u32), (2, 200), (1, 111)];
        let map: BTreeMap<u32, u32> = TryFromIterator::try_from_iter(pairs).expect("iter ok");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&111));
        assert_eq!(map.get(&2), Some(&200));
    }

    #[test]
    fn try_from_iterator_alloc_failure_leaves_nothing_behind() {
        use crate::test_helpers::FailAlloc;

        let mut map: BTreeMap<u32, u32, FailAlloc> = BTreeMap::new_in(FailAlloc);
        let (resume, _err) = map
            .try_extend([(1u32, 10u32), (2, 20)].into_iter())
            .expect_err("allocation should fail");
        assert!(map.is_empty(), "no pair may survive a failed insert");
        // The resume carries the unconsumed remainder plus any stranded
        // element, so it must not be empty.
        let remaining: std::vec::Vec<(u32, u32)> = resume.into_remainder().collect();
        assert!(
            !remaining.is_empty(),
            "resume must carry the unconsumed tail"
        );
    }

    #[test]
    fn try_collect_into_btreemap_via_trait() {
        // Exercises the blanket `TryCollect` wiring end-to-end.
        use olive_core::try_traits::try_collect::TryCollect;

        let map: BTreeMap<u32, u32> = (0..5u32)
            .map(|k| (k, k + 100))
            .try_collect()
            .expect("collect ok");
        assert_eq!(map.len(), 5);
        for k in 0..5u32 {
            assert_eq!(map.get(&k), Some(&(k + 100)));
        }
    }

    // --- TryExtend ------------------------------------------------------------

    #[test]
    fn try_extend_success_merges_and_overwrites() {
        let mut map = BTreeMap::new();
        map.try_insert(1, 100).unwrap();
        map.try_insert(2, 200).unwrap();
        map.try_extend([(2u32, 201u32), (3, 300)].into_iter())
            .expect("extend ok");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&201), "existing key must be overwritten");
        assert_eq!(map.get(&3), Some(&300));
    }

    #[test]
    fn try_extend_retry_recovers() {
        let start = Resume::new((1u32, 10u32), [(2, 20), (3, 30)].into_iter());
        let mut map = BTreeMap::new();
        map.try_extend(start).expect("retry ok");
        assert_eq!(map.len(), 3);
        for (k, v) in map.iter() {
            assert_eq!(*v, *k * 10);
        }
    }

    #[test]
    fn try_extend_partial_commit_then_oom_strands_one_pair() {
        use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};

        // Build a populated map under a shared budget, then drain it so the next
        // insertion OOMs mid-extend. Because `try_extend` commits each successful
        // insert irreversibly, a partial merge is expected: some new keys land,
        // exactly one pair is stranded in the resume, and nothing is lost or
        // duplicated across map + stranded pair.
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut m: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        for i in (0..40u32).step_by(2) {
            let kid = ledger.allocate();
            ledger.register(kid);
            let vid = ledger.allocate();
            ledger.register(vid);
            m.try_insert(
                TrackedItem {
                    id: kid,
                    ledger: ledger.clone(),
                    inner: i,
                },
                TrackedItem {
                    id: vid,
                    ledger: ledger.clone(),
                    inner: i * 100,
                },
            )
            .unwrap();
        }
        let len_before = m.len();

        // Extend with 40 fresh odd keys; the first insert that needs a new node
        // will OOM, stranding its pair.
        let mut src_pairs: std::vec::Vec<(TrackedItem<u32>, TrackedItem<u32>)> =
            std::vec::Vec::new();
        for i in (1..80u32).step_by(2) {
            let kid = ledger.allocate();
            ledger.register(kid);
            let vid = ledger.allocate();
            ledger.register(vid);
            src_pairs.push((
                TrackedItem {
                    id: kid,
                    ledger: ledger.clone(),
                    inner: i,
                },
                TrackedItem {
                    id: vid,
                    ledger: ledger.clone(),
                    inner: i * 100,
                },
            ));
        }

        // Attempt to fail mid-extend. This figure is an educated guess - allocation demand depends on the shape
        // and not be easily predicted. However, 40 items easily require allocating more than 1 node.
        alloc.set_budget(1);

        let (resume, _e) = m
            .try_extend(src_pairs.into_iter())
            .expect_err("OOM mid-extend");

        // Map must remain a valid, sorted tree.
        check_tree_invariant(&m);
        check_ascending_keys(&m);
        assert!(m.len() >= len_before, "successful inserts are irreversible");

        // The stranded pair is carried by the resume and absent from the map.
        let stranded: std::vec::Vec<(TrackedItem<u32>, TrackedItem<u32>)> =
            resume.into_remainder().collect();
        assert!(
            !stranded.is_empty(),
            "resume must carry at least the stranded pair"
        );
        for (sk, _) in &stranded {
            assert!(m.get(sk).is_none(), "stranded key must not be committed");
        }

        // No double-drop / leak across the whole failure path.
        drop(m);
        drop(stranded);
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
    }

    // --- TryExtendFromSlice ---------------------------------------------------

    #[test]
    fn try_extend_from_slice_success() {
        let mut map = BTreeMap::new();
        map.try_insert(9, 90).unwrap();
        map.try_extend_from_slice(&[(1u32, 10u32), (5, 50), (9, 99)])
            .expect("slice extend ok");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&5), Some(&50));
        assert_eq!(map.get(&9), Some(&99), "existing key must be overwritten");
    }

    #[test]
    fn try_extend_from_slice_empty_is_noop() {
        let mut map = BTreeMap::new();
        map.try_extend_from_slice(&[] as &[(u32, u32)]).expect("ok");
        assert!(map.is_empty());
    }

    #[test]
    fn try_extend_from_slice_first_alloc_failure_carries_whole_tail() {
        use crate::test_helpers::FailAlloc;

        let mut map: BTreeMap<u32, u32, FailAlloc> = BTreeMap::new_in(FailAlloc);
        let src: &[(u32, u32)] = &[(1, 10), (2, 20), (3, 30)];
        let (rest, _e) = map
            .try_extend_from_slice(src)
            .expect_err("allocation should fail");
        assert!(map.is_empty(), "nothing may be committed under total OOM");
        assert!(
            !rest.is_empty(),
            "remainder must point at the unprocessed tail"
        );
        assert_eq!(rest.len(), 3, "the whole slice remains unprocessed");
    }

    #[test]
    fn try_extend_from_slice_clone_failure_returns_tail() {
        use crate::test_helpers::FlakyClone;

        // `FlakyClone::new(0)` never clones successfully, so the very first
        // source element's value clone fails. The returned tail must point at the
        // start of the slice (index 0) and nothing may be committed.
        let mut m: BTreeMap<u32, FlakyClone> = BTreeMap::new();
        let src: &[(u32, FlakyClone)] = &[
            (1, FlakyClone::new(0)),
            (2, FlakyClone::new(0)),
            (3, FlakyClone::new(0)),
        ];
        let (rest, e) = m
            .try_extend_from_slice(src)
            .expect_err("first clone should fail");
        assert!(matches!(
            e,
            TryBTreeMapWithCloneError::Clone(TryCloneError::Other(_))
        ));
        assert!(
            m.is_empty(),
            "nothing may be committed before a failed clone"
        );
        assert_eq!(rest.len(), 3, "the whole slice remains unprocessed");
    }

    #[test]
    fn try_extend_from_slice_commits_prefix_before_clone_failure() {
        use crate::test_helpers::FlakyClone;

        // Elements 0 and 1 clone cleanly (their values have headroom); element 2
        // has no headroom and its value clone fails. Only the successful prefix
        // may be committed, and the tail must begin exactly at the failing element.
        let mut m: BTreeMap<u32, FlakyClone> = BTreeMap::new();
        let src: &[(u32, FlakyClone)] = &[
            (
                1,
                FlakyClone {
                    count: 0,
                    threshold: 5,
                },
            ),
            (
                2,
                FlakyClone {
                    count: 0,
                    threshold: 5,
                },
            ),
            (
                3,
                FlakyClone {
                    count: 5,
                    threshold: 5,
                },
            ),
        ];
        let (rest, e) = m
            .try_extend_from_slice(src)
            .expect_err("third clone should fail");
        assert!(matches!(
            e,
            TryBTreeMapWithCloneError::Clone(TryCloneError::Other(_))
        ));
        assert_eq!(rest.len(), 1, "tail must start at the failing element");
        assert_eq!(m.len(), 2, "only the successful prefix is committed");
    }

    #[test]
    fn try_extend_from_slice_commits_prefix_before_alloc_failure() {
        use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};
        use std::sync::Arc;

        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        for i in 0..10u32 {
            map.try_insert(
                TrackedItem::construct(&ledger, i),
                TrackedItem::construct(&ledger, i * 100),
            )
            .unwrap();
        }
        assert_eq!(map.len(), 10);

        // Element 0 clones cleanly, element 1 clones, but cannot be inserted.
        let src: [(TrackedItem<u32>, TrackedItem<u32>); 2] = [
            (
                TrackedItem::construct(&ledger, 10),
                TrackedItem::construct(&ledger, 1000),
            ),
            (
                TrackedItem::construct(&ledger, 11),
                TrackedItem::construct(&ledger, 1000),
            ),
        ];

        alloc.drain();

        let result = map.try_extend_from_slice(&src);
        assert!(
            result.is_err(),
            "expected an allocation failure during the split"
        );
        let (rest, e) = result.unwrap_err();
        assert!(
            matches!(e, TryBTreeMapWithCloneError::Alloc(_)),
            "failure must be an allocation error, got {:?}",
            e
        );
        assert_eq!(rest.len(), 1, "the slice has one unprocessed element");

        // The map survived the failed split, but has an additional element.
        assert_eq!(map.len(), 11);
        check_tree_invariant(&map);
        check_ascending_keys(&map);

        drop(map);
        drop(src);
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
    }
}
