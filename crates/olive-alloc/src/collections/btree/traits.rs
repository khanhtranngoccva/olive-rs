//! Trait implementations for [`BTreeMap`].
//!
//! Covers standard-library trait impls ([`Debug`](fmt::Debug), [`PartialEq`]/[`Eq`],
//! [`PartialOrd`]/[`Ord`], [`Hash`]) and the fallible construction traits from
//! `olive-core` ([`TryClone`], [`TryDefault`]).

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

use super::map::BTreeMap;
use crate::alloc::{AllocatorTryClone, Global};

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<K: fmt::Debug, V: fmt::Debug, A: AllocatorTryClone> fmt::Debug for BTreeMap<K, V, A> {
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

impl<K: Ord + PartialEq, V: PartialEq, A: AllocatorTryClone> PartialEq for BTreeMap<K, V, A> {
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

impl<K: Ord + Eq, V: Eq, A: AllocatorTryClone> Eq for BTreeMap<K, V, A> {}

// ---------------------------------------------------------------------------
// PartialOrd / Ord
// ---------------------------------------------------------------------------

impl<K: Ord + PartialOrd, V: PartialOrd, A: AllocatorTryClone> PartialOrd for BTreeMap<K, V, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Lexicographic comparison of (key, value) pairs in ascending key order.
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Ord, V: Ord, A: AllocatorTryClone> Ord for BTreeMap<K, V, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

impl<K: Ord + Hash, V: Hash, A: AllocatorTryClone> Hash for BTreeMap<K, V, A> {
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
// TryDefault
// ---------------------------------------------------------------------------

/// An empty `BTreeMap` never allocates, so its default construction is
/// infallible. The default allocator is [`Global`], matching std's `BTreeMap`
/// (which defaults to the global allocator).
impl<K: Ord, V> TryDefault for BTreeMap<K, V, Global> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(BTreeMap::new_in(Global))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
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

        // Give the allocator enough budget to build a small map (each insert
        // into an empty tree clones the alloc once for the leaf node). Then
        // exhaust the remaining budget so the final `try_clone` call fails.
        let budget = Arc::new(CloneBudget::new(2));
        let alloc = FlakyCloneAlloc::new(budget.clone());
        let mut map: BTreeMap<i32, i32, FlakyCloneAlloc> = BTreeMap::new_in(alloc);
        // First insert: new_in consumed 1 clone (for reserve_stack), this
        // insert needs another clone for the leaf → uses the last unit.
        map.try_insert(1, 10).unwrap();

        // Budget is now exhausted; cloning the allocator must fail.
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

    // --- TryDefault ------------------------------------------------------------

    #[test]
    fn try_default_creates_empty_map() {
        let map: BTreeMap<i32, i32, Global> = TryDefault::try_default().expect("default ok");
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }
}
