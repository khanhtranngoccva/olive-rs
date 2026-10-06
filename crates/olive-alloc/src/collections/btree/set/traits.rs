//! Trait implementations for [`BTreeSet`].
//!
//! This file hosts every trait impl that does not belong to an inherent-method
//! category (construction, insertion, query, …). The first batch mirrors the
//! corresponding [`BTreeMap`](super::super::map::BTreeMap) impls: equality,
//! hashing, and lexicographic ordering.

use super::BTreeSet;
use crate::alloc::{Allocator, Global};
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use olive_core::alloc::AllocError;
use olive_core::try_traits::TryFromIterator;

// ── Equality ─────────────────────────────────────────────────────────────────

impl<T: Ord, A: Allocator> PartialEq for BTreeSet<T, A> {
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Length first, then each element in ascending order. This makes the
        // digest depend on membership only, never on insertion history, and it
        // distinguishes multisets-with-duplicates from sets by construction.
        state.write_usize(self.len());
        for value in self.iter() {
            value.hash(state);
        }
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

impl<T: Ord, A: Allocator> PartialOrd for BTreeSet<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord, A: Allocator> Ord for BTreeSet<T, A> {
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
}
