//! Trait implementations for [`BTreeSet`].

use super::super::TryBTreeSetWithCloneError;
use super::BTreeSet;
use crate::alloc::{Allocator, Global};
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::hash::{Hash, Hasher};
use olive_core::alloc::AllocError;
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::TryFromIterator;
use olive_core::try_traits::try_clone::TryClone;
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};

// ── Equality ─────────────────────────────────────────────────────────────────

impl<T: Ord, A: Allocator> PartialEq for BTreeSet<T, A> {
    /// Reports whether two sets contain exactly the same elements.
    ///
    /// Comparison is by membership only; insertion order and history are
    /// irrelevant. Sets of different lengths are unequal without inspecting
    /// any element.
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        // Both sets iterate in ascending order, so pairwise comparison of the
        // ordered sequences is equivalent to set equality.
        self.iter().eq(other.iter())
    }
}

impl<T: Ord, A: Allocator> Eq for BTreeSet<T, A> {}

// ── Hashing ──────────────────────────────────────────────────────────────────

impl<T: Hash + Ord, A: Allocator> Hash for BTreeSet<T, A> {
    /// Feeds the set into a hasher.
    ///
    /// Equal sets always produce equal digests regardless of the order their
    /// elements were inserted in.
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for value in self.iter() {
            value.hash(state);
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

impl<T: Ord, A: Allocator> PartialOrd for BTreeSet<T, A> {
    /// Compares two sets lexicographically over their sorted sequences.
    /// The first mismatched element decides; a shorter prefix sorts before its
    /// longer extension.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord, A: Allocator> Ord for BTreeSet<T, A> {
    /// Total-order comparison of two sets, lexicographic over their sorted
    /// sequences: the first mismatched element decides, and a shorter prefix
    /// sorts before its longer extension.
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ── TryFromIterator ────────────────────────────────────────────────────────────────────

impl<T: Ord> TryFromIterator<T> for BTreeSet<T, Global> {
    type Error = AllocError;

    /// Fallibly collect an iterator of values into a [`BTreeSet`] on the
    /// default [`Global`] allocator. For a custom allocator use
    /// [`BTreeSet::try_from_iter_in`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if an internal allocation fails during insertion.
    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Fallibly collects an iterator of values into a [`BTreeSet`] on the given
    /// allocator. Duplicate values collapse to a single entry, as with any set
    /// insertion.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if reserving capacity for an incoming element
    /// fails. On failure the partially constructed set is discarded.
    pub fn try_from_iter_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, AllocError> {
        let mut set = Self::new_in(alloc);
        for value in iter {
            set.try_insert(value)?;
        }
        Ok(set)
    }
}

// ── TryExtend / TryExtendFromSlice ───────────────────────────────────────────

impl<T: Ord, A: Allocator> TryExtend<T> for BTreeSet<T, A> {
    type Error = AllocError;

    /// Attempts to extend the set with elements from a resumable source.
    ///
    /// # Errors
    ///
    /// On allocation failure, returns the remaining items in a [`Resume`] so
    /// the caller can retry once memory is available, along with the [`AllocError`].
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, _hint) = source.decompose_with_size_hint();

        // Insert the stranded head first, if any.
        if let Some(val) = head {
            if let Err((val, e)) = self.try_insert_give_back(val) {
                return Err((Resume::new(val, inner), e));
            }
        }

        // Insert the remainder one element at a time. Each insertion is atomic:
        // on allocation failure the tree is left unmodified and we strand the
        // current element in a `Resume` for retry.
        while let Some(val) = inner.next() {
            if let Err((val, e)) = self.try_insert_give_back(val) {
                return Err((Resume::new(val, inner), e));
            }
        }
        Ok(())
    }
}

impl<T: Ord + TryClone, A: Allocator> TryExtendFromSlice<T> for BTreeSet<T, A> {
    type Error = TryBTreeSetWithCloneError;

    /// Extends the set with elements cloned from a slice.
    ///
    /// # Errors
    ///
    /// On failure, returns the unconsumed tail of the slice together with
    /// the [`TryBTreeSetWithCloneError`], so the caller can retry from
    /// where it stopped.
    fn try_extend_from_slice<'s>(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        let mut i = 0usize;
        for val in other {
            let cloned = val
                .try_clone()
                .map_err(|e| (&other[i..], TryBTreeSetWithCloneError::Clone(e)))?;
            match self.try_insert_give_back(cloned) {
                Ok(_) => {
                    #[allow(clippy::arithmetic_side_effects, reason = "asserted i < other.len()")]
                    {
                        i += 1;
                    }
                }
                Err(_) => {
                    return Err((&other[i..], TryBTreeSetWithCloneError::Alloc(AllocError)));
                }
            }
        }
        Ok(())
    }
}

// ── TryFrom: arrays and vectors ──────────────────────────────────────────────

impl<T: Ord, const N: usize> TryFrom<[T; N]> for BTreeSet<T> {
    type Error = AllocError;

    /// Fallibly converts an array into a set on the [`Global`] allocator,
    /// moving each element without cloning. Duplicate values are removed,
    /// so the resulting set may hold fewer than `N` elements.
    ///
    /// # Errors
    ///
    /// On allocation failure every element of the array is dropped and an
    /// [`AllocError`] is returned.
    fn try_from(array: [T; N]) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(array, Global)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::collections::btree::invariant::{check_ascending_keys, check_tree_invariant};

    fn make_set(values: &[i32]) -> BTreeSet<i32> {
        let mut set = BTreeSet::new();
        for &v in values {
            set.try_insert(v).unwrap();
        }
        set
    }

    // --- PartialEq / Eq -----------------------------------------------------

    #[test]
    fn equal_sets_are_equal() {
        assert_eq!(make_set(&[1, 2, 3]), make_set(&[3, 1, 2]));
        assert_eq!(make_set(&[]), make_set(&[]));
        assert_eq!(make_set(&[7]), make_set(&[7]));
    }

    #[test]
    fn different_membership_is_not_equal() {
        assert_ne!(make_set(&[1, 2]), make_set(&[1, 3]));
        assert_ne!(make_set(&[1, 2]), make_set(&[1, 2, 3]));
        assert_ne!(make_set(&[]), make_set(&[1]));
    }

    #[test]
    fn duplicates_collapse_before_comparison() {
        // Inserting the same value twice must not change the set.
        assert_eq!(make_set(&[1, 1, 2, 2, 2]), make_set(&[1, 2]));
    }

    #[test]
    fn empty_sets_are_equal() {
        let a: BTreeSet<i32> = BTreeSet::new();
        let b: BTreeSet<i32> = BTreeSet::new();
        assert_eq!(a, b);
    }

    #[test]
    fn eq_is_reflexive_and_symmetric() {
        let a = make_set(&[5, 1, 9]);
        let b = make_set(&[1, 5, 9]);
        assert_eq!(a, b);
        assert_eq!(b, a);
        assert_eq!(a, a);
    }

    // --- Hash ----------------------------------------------------------------

    #[test]
    fn equal_sets_hash_equally() {
        use core::hash::Hasher as _;
        use std::collections::hash_map::DefaultHasher;

        let a = make_set(&[3, 1, 2]);
        let b = make_set(&[2, 3, 1]);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn different_sets_hash_differently_mostly() {
        // Not a mathematical guarantee (hash collisions are possible), but for
        // small integer sets under DefaultHasher this holds reliably.
        use core::hash::Hasher as _;
        use std::collections::hash_map::DefaultHasher;

        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 4]);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn length_participates_in_hash() {
        // Same elements but different cardinality must produce different
        // digests even though one is a prefix of the other.
        use core::hash::Hasher as _;
        use std::collections::hash_map::DefaultHasher;

        let a = make_set(&[1, 2]);
        let b = make_set(&[1, 2, 3]);

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    // --- PartialOrd / Ord ----------------------------------------------------

    #[test]
    fn ord_matches_lexicographic_sequence_order() {
        let a = make_set(&[1, 2, 3]);
        let b = make_set(&[1, 2, 4]);
        let c = make_set(&[1, 2]);
        let d = make_set(&[]);

        assert!(c < a); // shorter prefix sorts first
        assert!(a < b); // first differing element decides
        assert!(d < c); // empty is the smallest
        assert!(b > a);
    }

    #[test]
    fn ord_is_consistent_with_partial_ord() {
        let a = make_set(&[1, 3]);
        let b = make_set(&[1, 2]);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Greater));
        assert_eq!(a.cmp(&b), Ordering::Greater);
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Less));
        assert_eq!(b.cmp(&a), Ordering::Less);
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn ord_is_transitive() {
        let a = make_set(&[1]);
        let b = make_set(&[1, 2]);
        let c = make_set(&[1, 2, 3]);
        assert!(a < b && b < c && a < c);
    }

    #[test]
    fn ord_total_order_over_disjoint_ranges() {
        let lo = make_set(&[0, 1, 2]);
        let hi = make_set(&[10, 11, 12]);
        assert!(lo < hi);
        assert!(hi > lo);
    }

    // --- TryFromIterator -------------------------------------------------------

    #[test]
    fn try_from_iter_builds_sorted_set() {
        let set: BTreeSet<i32> = TryFromIterator::try_from_iter((1..=5).rev()).unwrap();
        assert_eq!(set.len(), 5);
        assert_eq!(
            set.iter().copied().collect::<std::vec::Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn try_from_iter_collapses_duplicates() {
        let values = std::vec![3, 1, 3, 2, 1, 3];
        let set: BTreeSet<i32> = TryFromIterator::try_from_iter(values).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
    }

    #[test]
    fn try_from_iter_empty_is_empty() {
        let set: BTreeSet<i32> = TryFromIterator::try_from_iter(std::iter::empty::<i32>()).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn try_from_iter_in_budgeted_alloc_succeeds_within_budget() {
        use crate::test_helpers::BudgetedAlloc;

        // Generous budget: construction must succeed and the set must be
        // intact. The allocator is consumed by value into the set.
        let alloc = BudgetedAlloc::new(100);
        let set = BTreeSet::try_from_iter_in([7, 2, 9, 2], alloc).unwrap();
        assert_eq!(set.len(), 3);
        assert_eq!(set.iter().copied().collect::<std::vec::Vec<_>>(), [2, 7, 9]);
    }

    #[test]
    fn try_from_iter_in_drained_alloc_fails() {
        use crate::test_helpers::BudgetedAlloc;

        // Zero budget: the first internal allocation fails, so the whole
        // fallible collection must report an error rather than panic.
        let alloc = BudgetedAlloc::new(0);
        let result = BTreeSet::try_from_iter_in([1, 2, 3], alloc);
        assert!(result.is_err());
    }

    #[test]
    fn try_from_iter_single_element() {
        let set: BTreeSet<i32> = TryFromIterator::try_from_iter(std::iter::once(42)).unwrap();
        assert_eq!(set.len(), 1);
        assert!(set.contains(&42));
    }

    // --- Invariants after trait-driven reads ---------------------------------

    #[test]
    fn tree_still_valid_after_trait_queries() {
        let mut set = BTreeSet::new();
        for i in (0..80u32).rev() {
            set.try_insert(i as i32).unwrap();
        }
        check_ascending_keys(&set.map);
        check_tree_invariant(&set.map);

        // Exercise the trait paths; none mutate, but confirm the tree survives.
        let other = make_set(&(0..80).collect::<std::vec::Vec<_>>());
        assert_eq!(set, other);
        assert!(set <= other);
        check_ascending_keys(&set.map);
    }

    // --- TryExtend -----------------------------------------------------------

    #[test]
    fn try_extend_adds_new_elements() {
        let mut set = make_set(&[1, 3]);
        set.try_extend([2, 4, 5].into_iter()).expect("extend ok");
        assert_eq!(set.len(), 5);
        assert!(set.contains(&2));
        assert!(set.contains(&4));
        assert!(set.contains(&5));
    }

    #[test]
    fn try_extend_deduplicates_existing() {
        let mut set = make_set(&[1, 2, 3]);
        set.try_extend([2, 4].into_iter()).expect("extend ok");
        assert_eq!(set.len(), 4);
        assert!(set.contains(&4));
    }

    #[test]
    fn try_extend_empty_source_is_noop() {
        let mut set = make_set(&[1, 2]);
        set.try_extend(core::iter::empty::<i32>())
            .expect("extend ok");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn try_extend_retry_recovers() {
        use olive_core::recovery::Resume;
        let start = Resume::new(10i32, [20, 30].into_iter());
        let mut set = make_set(&[]);
        set.try_extend(start).expect("retry ok");
        assert_eq!(set.len(), 3);
        assert!(set.contains(&10));
        assert!(set.contains(&20));
        assert!(set.contains(&30));
    }

    #[test]
    fn try_extend_oom_strands_element() {
        use crate::test_helpers::BudgetedAlloc;
        // Build a populated set under a generous budget, then constrain the
        // remaining budget so the next insertion that needs a new node OOMs.
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut set: BTreeSet<i32, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        for i in 0..40i32 {
            set.try_insert(i).expect("insert ok");
        }
        // Constrain to exactly 1 allocation — enough for one leaf but not
        // enough for the rebalance, guaranteeing an OOM mid-extend.
        alloc.set_budget(1);
        let (resume, _err) = set.try_extend(40..200).expect_err("OOM mid-extend");
        let (_head, remainder) = resume.into_parts();
        let count = remainder.count();
        assert!(count > 0, "something must be stranded");
    }

    // --- TryExtendFromSlice ----------------------------------------------------

    #[test]
    fn try_extend_from_slice_adds_elements() {
        let mut set = make_set(&[1, 3]);
        let src = [2, 4, 5];
        set.try_extend_from_slice(&src).expect("extend ok");
        assert_eq!(set.len(), 5);
        assert!(set.contains(&2));
        assert!(set.contains(&4));
        assert!(set.contains(&5));
    }

    #[test]
    fn try_extend_from_slice_deduplicates() {
        let mut set = make_set(&[1, 2, 3]);
        let src = [2, 4];
        set.try_extend_from_slice(&src).expect("extend ok");
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn try_extend_from_slice_empty_is_noop() {
        let mut set = make_set(&[1]);
        let src: [i32; 0] = [];
        set.try_extend_from_slice(&src).expect("extend ok");
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn try_extend_from_slice_clone_failure_returns_tail() {
        use crate::test_helpers::FlakyClone;
        let mut set: BTreeSet<FlakyClone> = BTreeSet::new();
        let src = [FlakyClone::new(0), FlakyClone::new(0)];
        let (rest, e) = set
            .try_extend_from_slice(&src)
            .expect_err("first clone should fail");
        assert!(matches!(e, TryBTreeSetWithCloneError::Clone(_)));
        assert!(set.is_empty(), "nothing committed before failed clone");
        assert_eq!(rest.len(), 2, "whole slice remains unprocessed");
    }

    #[test]
    fn try_extend_from_slice_commits_prefix_before_clone_failure() {
        use crate::test_helpers::FlakyClone;
        let mut set: BTreeSet<FlakyClone> = BTreeSet::new();
        let src = [
            FlakyClone {
                count: 0,
                threshold: 5,
            },
            FlakyClone {
                count: 0,
                threshold: 0,
            },
            FlakyClone {
                count: 0,
                threshold: 5,
            },
        ];
        let (rest, e) = set
            .try_extend_from_slice(&src)
            .expect_err("second clone should fail");
        assert!(matches!(e, TryBTreeSetWithCloneError::Clone(_)));
        assert_eq!(set.len(), 1, "only first element committed");
        assert_eq!(rest.len(), 2, "tail starts at failing element");
    }

    #[test]
    fn try_extend_from_slice_oom_returns_tail() {
        use crate::test_helpers::BudgetedAlloc;
        // Build a populated set under a generous budget, then constrain the
        // remaining budget so the next insertion that needs a new node OOMs.
        let alloc = BudgetedAlloc::new(1 << 20);
        let mut set: BTreeSet<i32, BudgetedAlloc> = BTreeSet::new_in(alloc.clone());
        for i in 0..40i32 {
            set.try_insert(i).expect("insert ok");
        }
        alloc.set_budget(1);
        let src: std::vec::Vec<i32> = (40..200).collect();
        let (rest, e) = set.try_extend_from_slice(&src).expect_err("OOM mid-extend");
        assert!(matches!(e, TryBTreeSetWithCloneError::Alloc(_)));
        assert!(!rest.is_empty(), "tail must carry unprocessed elements");
    }
    // --- TryFrom: arrays -------------------------------------------------------

    #[test]
    fn try_from_array_builds_sorted_set() {
        let set: BTreeSet<i32> = [5, 1, 3, 2, 4].try_into().expect("array ok");
        assert_eq!(set.len(), 5);
        let collected: std::vec::Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, std::vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn try_from_array_collapses_duplicates() {
        let set: BTreeSet<i32> = [7, 7, 7, 7].try_into().expect("array ok");
        assert_eq!(set.len(), 1);
        assert!(set.contains(&7));
    }

    #[test]
    fn try_from_empty_array_is_empty() {
        let set: BTreeSet<i32> = [].try_into().expect("empty array ok");
        assert!(set.is_empty());
    }
}
