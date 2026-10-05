//! Set-algebra iterators for [`BTreeSet`]: `intersection`.
//!
//! The intersection of two sorted sets is computed with the same strategy as
//! std's `BTreeSet::intersection`: when the sets are similarly sized, both are
//! iterated jointly and matches spotted along the way (`Stitch`); when one set
//! is much smaller, its elements are looked up in the larger one (`Search`).
//! A degenerate case — one side empty, or exactly one candidate element
//! survives a length comparison — collapses to a single-element answer
//! (`Answer`).

use core::cmp::{Ordering, min};
use core::fmt;
use core::fmt::Debug;
use core::iter::{DoubleEndedIterator, FusedIterator, Iterator, Peekable};
use core::marker::PhantomData;

use super::super::merge_iter::MergeIterInner;
use super::BTreeSet;
use super::iter::Iter;
use crate::alloc::Global;
use olive_core::TryClone;
use olive_core::alloc::Allocator;
use olive_core::try_traits::TryClone;

// This constant is used by functions that compare two sets.
// It estimates the relative size at which searching performs better
// than iterating, based on the benchmarks in
// https://github.com/ssomers/rust_bench_btreeset_intersection.
// It's used to divide rather than multiply sizes, to rule out overflow,
// and it's a power of two to make that division cheap.
const ITER_PERFORMANCE_TIPPING_SIZE_DIFF: usize = 16;

// ── Intersection ──────────────────────────────────────────────────────────────

/// A lazy iterator producing elements in the intersection of `BTreeSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`intersection`]: BTreeSet::intersection
#[must_use = "this returns the intersection as an iterator, \
              without modifying either input set"]
#[derive(TryClone)]
pub struct Intersection<'a, T: 'a, A: Allocator = Global> {
    inner: IntersectionInner<'a, T, A>,
}

#[derive(TryClone)]
enum IntersectionInner<'a, T: 'a, A: Allocator> {
    /// Iterate similarly sized sets jointly, spotting matches along the way
    Stitch { a: Iter<'a, T>, b: Iter<'a, T> },
    /// Iterate a small set, look up in the large set
    Search {
        small_iter: Iter<'a, T>,
        large_set: &'a BTreeSet<T, A>,
    },
    /// Return a specific element or emptiness
    Answer(Option<&'a T>),
}

impl<T, A: Allocator> Intersection<'_, T, A> {
    fn empty() -> Self {
        Self {
            inner: IntersectionInner::Answer(None),
        }
    }
}

// TryClone is an infallible perfect macro, but Clone isn't
impl<T, A: Allocator> Clone for Intersection<'_, T, A> {
    fn clone(&self) -> Self {
        TryClone::try_clone(self).expect("should infallibly clone Intersection")
    }
}

impl<T: Debug, A: Allocator> Debug for IntersectionInner<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntersectionInner::Stitch { a, b } => f
                .debug_struct("Stitch")
                .field("a", a)
                .field("b", b)
                .finish(),
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => f
                .debug_struct("Search")
                .field("small_iter", small_iter)
                .field("large_set", large_set)
                .finish(),
            IntersectionInner::Answer(x) => f.debug_tuple("Answer").field(x).finish(),
        }
    }
}

impl<T: Debug, A: Allocator> Debug for Intersection<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Intersection").field(&self.inner).finish()
    }
}

impl<'a, T: Ord, A: Allocator> Iterator for Intersection<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            IntersectionInner::Stitch { a, b } => {
                let mut a_next = a.next()?;
                let mut b_next = b.next()?;
                // The iterator reporting "less" must advance.
                loop {
                    match a_next.cmp(b_next) {
                        Ordering::Less => a_next = a.next()?,
                        Ordering::Greater => b_next = b.next()?,
                        Ordering::Equal => return Some(a_next),
                    }
                }
            }
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => loop {
                let small_next = small_iter.next()?;
                if large_set.contains(small_next) {
                    return Some(small_next);
                }
            },
            IntersectionInner::Answer(answer) => answer.take(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            IntersectionInner::Stitch { a, b } => (0, Some(min(a.len(), b.len()))),
            IntersectionInner::Search { small_iter, .. } => (0, Some(small_iter.len())),
            IntersectionInner::Answer(None) => (0, Some(0)),
            IntersectionInner::Answer(Some(_)) => (1, Some(1)),
        }
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }

    fn max(mut self) -> Option<&'a T> {
        self.next_back()
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

impl<'a, T: Ord, A: Allocator> DoubleEndedIterator for Intersection<'a, T, A> {
    fn next_back(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            IntersectionInner::Stitch { a, b } => {
                let mut a_next_back = a.next_back()?;
                let mut b_next_back = b.next_back()?;
                // The iterator reporting "greater" must advance.
                loop {
                    match a_next_back.cmp(b_next_back) {
                        Ordering::Less => b_next_back = b.next_back()?,
                        Ordering::Greater => a_next_back = a.next_back()?,
                        Ordering::Equal => return Some(a_next_back),
                    }
                }
            }
            IntersectionInner::Search {
                small_iter,
                large_set,
            } => loop {
                let small_next_back = small_iter.next_back()?;
                if large_set.contains(small_next_back) {
                    return Some(small_next_back);
                }
            },
            IntersectionInner::Answer(answer) => answer.take(),
        }
    }
}

impl<T: Ord, A: Allocator> FusedIterator for Intersection<'_, T, A> {}

impl<'a, T: Ord, A: Allocator> Intersection<'a, T, A> {
    fn new(self_set: &'a BTreeSet<T, A>, other_set: &'a BTreeSet<T, A>) -> Self {
        let (Some(self_min), Some(self_max), Some(other_min), Some(other_max)) = (
            self_set.first(),
            self_set.last(),
            other_set.first(),
            other_set.last(),
        ) else {
            return Intersection::empty();
        };

        Intersection {
            inner: match (self_min.cmp(other_max), self_max.cmp(other_min)) {
                (Ordering::Greater, _) | (_, Ordering::Less) => IntersectionInner::Answer(None),
                (Ordering::Equal, _) => IntersectionInner::Answer(Some(self_min)),
                (_, Ordering::Equal) => IntersectionInner::Answer(Some(self_max)),
                _ if self_set.len() <= other_set.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                    IntersectionInner::Search {
                        small_iter: self_set.iter(),
                        large_set: other_set,
                    }
                }
                _ if other_set.len() <= self_set.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                    IntersectionInner::Search {
                        small_iter: other_set.iter(),
                        large_set: self_set,
                    }
                }
                _ => IntersectionInner::Stitch {
                    a: self_set.iter(),
                    b: other_set.iter(),
                },
            },
        }
    }
}

// ── Union ─────────────────────────────────────────────────────────────────────

/// A lazy iterator producing elements in the union of `BTreeSet`s.
///
/// This `struct` is created by the [`union`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`union`]: BTreeSet::union
#[must_use = "this returns the union as an iterator, \
              without modifying either input set"]
#[derive(TryClone)]
pub struct Union<'a, T: 'a, A: Allocator = Global> {
    inner: MergeIterInner<Iter<'a, T>>,
    _alloc: PhantomData<&'a A>,
}

impl<'a, T: Ord, A: Allocator> Union<'a, T, A> {
    pub(super) fn new(a: &'a BTreeSet<T, A>, b: &'a BTreeSet<T, A>) -> Self {
        let inner = MergeIterInner::new(a.iter(), b.iter());
        Union {
            inner,
            _alloc: PhantomData,
        }
    }
}

impl<T: Debug, A: Allocator> Debug for Union<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Union").field(&self.inner).finish()
    }
}

// TryClone is an infallible perfect macro, but Clone isn't
impl<T, A: Allocator> Clone for Union<'_, T, A> {
    fn clone(&self) -> Self {
        TryClone::try_clone(self).expect("should infallibly clone Union")
    }
}

impl<'a, T: Ord, A: Allocator> Iterator for Union<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let (a_next, b_next) = self.inner.nexts(Self::Item::cmp);
        a_next.or(b_next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_len, b_len) = self.inner.lens();
        (core::cmp::max(a_len, b_len), a_len.checked_add(b_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<T: Ord, A: Allocator> FusedIterator for Union<'_, T, A> {}

// ── Difference ────────────────────────────────────────────────────────────────

/// A lazy iterator producing elements in the difference of `BTreeSet`s.
///
/// This `struct` is created by the [`difference`] method on [`BTreeSet`].
/// See its documentation for more.
///
/// [`difference`]: BTreeSet::difference
#[must_use = "this returns the difference as an iterator, \
              without modifying either input set"]
#[derive(TryClone)]
pub struct Difference<'a, T: 'a, A: Allocator = Global> {
    inner: DifferenceInner<'a, T, A>,
}

enum DifferenceInner<'a, T: 'a, A: Allocator> {
    Stitch {
        self_iter: Iter<'a, T>,
        other_iter: Peekable<Iter<'a, T>>,
    },
    Search {
        self_iter: Iter<'a, T>,
        other_set: &'a BTreeSet<T, A>,
    },
    Iterate(Iter<'a, T>),
}

impl<T: Debug, A: Allocator> Debug for DifferenceInner<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => f
                .debug_struct("Stitch")
                .field("self_iter", self_iter)
                .field("other_iter", other_iter)
                .finish(),
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => f
                .debug_struct("Search")
                .field("self_iter", self_iter)
                .field("other_set", other_set)
                .finish(),
            DifferenceInner::Iterate(x) => f.debug_tuple("Iterate").field(x).finish(),
        }
    }
}

impl<T: Debug, A: Allocator> Debug for Difference<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Difference").field(&self.inner).finish()
    }
}

impl<T: Ord, A: Allocator> Clone for Difference<'_, T, A> {
    fn clone(&self) -> Self {
        Difference {
            inner: match &self.inner {
                DifferenceInner::Stitch {
                    self_iter,
                    other_iter,
                } => DifferenceInner::Stitch {
                    self_iter: self_iter.clone(),
                    other_iter: other_iter.clone(),
                },
                DifferenceInner::Search {
                    self_iter,
                    other_set,
                } => DifferenceInner::Search {
                    self_iter: self_iter.clone(),
                    other_set,
                },
                DifferenceInner::Iterate(iter) => DifferenceInner::Iterate(iter.clone()),
            },
        }
    }
}

impl<'a, T: Ord, A: Allocator> Iterator for Difference<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match &mut self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => {
                let mut self_next = self_iter.next()?;
                loop {
                    match other_iter
                        .peek()
                        .map_or(Ordering::Less, |other_next| self_next.cmp(other_next))
                    {
                        Ordering::Less => return Some(self_next),
                        Ordering::Equal => {
                            self_next = self_iter.next()?;
                            other_iter.next();
                        }
                        Ordering::Greater => {
                            other_iter.next();
                        }
                    }
                }
            }
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => loop {
                let self_next = self_iter.next()?;
                if !other_set.contains(self_next) {
                    return Some(self_next);
                }
            },
            DifferenceInner::Iterate(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (self_len, other_len) = match &self.inner {
            DifferenceInner::Stitch {
                self_iter,
                other_iter,
            } => (self_iter.len(), other_iter.len()),
            DifferenceInner::Search {
                self_iter,
                other_set,
            } => (self_iter.len(), other_set.len()),
            DifferenceInner::Iterate(iter) => (iter.len(), 0),
        };
        (self_len.saturating_sub(other_len), Some(self_len))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }

    fn max(self) -> Option<&'a T> {
        self.last()
    }
}

impl<T: Ord, A: Allocator> FusedIterator for Difference<'_, T, A> {}

impl<'a, T: Ord, A: Allocator> Difference<'a, T, A> {
    pub(super) fn new(self_set: &'a BTreeSet<T, A>, other_set: &'a BTreeSet<T, A>) -> Self {
        let (Some(self_min), Some(self_max), Some(other_min), Some(other_max)) = (
            self_set.first(),
            self_set.last(),
            other_set.first(),
            other_set.last(),
        ) else {
            return Difference {
                inner: DifferenceInner::Iterate(self_set.iter()),
            };
        };

        let inner = match (self_min.cmp(other_max), self_max.cmp(other_min)) {
            (Ordering::Greater, _) | (_, Ordering::Less) => {
                DifferenceInner::Iterate(self_set.iter())
            }
            (Ordering::Equal, _) => {
                let mut it = self_set.iter();
                it.next();
                DifferenceInner::Iterate(it)
            }
            (_, Ordering::Equal) => {
                let mut it = self_set.iter();
                it.next_back();
                DifferenceInner::Iterate(it)
            }
            _ if self_set.len() <= other_set.len() / ITER_PERFORMANCE_TIPPING_SIZE_DIFF => {
                DifferenceInner::Search {
                    self_iter: self_set.iter(),
                    other_set,
                }
            }
            _ => DifferenceInner::Stitch {
                self_iter: self_set.iter(),
                other_iter: other_set.iter().peekable(),
            },
        };
        Difference { inner }
    }
}

// ── BTreeSet methods ──────────────────────────────────────────────────────────

impl<T: Ord, A: Allocator> BTreeSet<T, A> {
    /// Returns an iterator over elements belonging to the intersection of
    /// `self` and `other`, in ascending order.
    ///
    /// The resulting iterator supports reverse iteration via
    /// [`DoubleEndedIterator::next_back`].
    pub fn intersection<'s>(&'s self, other: &'s BTreeSet<T, A>) -> Intersection<'s, T, A> {
        Intersection::new(self, other)
    }

    /// Returns an iterator over elements in the union of `self` and `other`,
    /// in ascending order.
    pub fn union<'s>(&'s self, other: &'s BTreeSet<T, A>) -> Union<'s, T, A> {
        Union::new(self, other)
    }

    /// Returns an iterator over elements in `self` but not in `other`,
    /// in ascending order.
    pub fn difference<'s>(&'s self, other: &'s BTreeSet<T, A>) -> Difference<'s, T, A> {
        Difference::new(self, other)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::vec::Vec;

    fn build(values: &[i32]) -> BTreeSet<i32> {
        let mut s = BTreeSet::new();
        for &v in values {
            s.try_insert(v).unwrap();
        }
        s
    }

    #[test]
    fn intersection_disjoint_sets_is_empty() {
        let a = build(&[1, 3, 5]);
        let b = build(&[2, 4, 6]);
        let result: Vec<&i32> = a.intersection(&b).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn intersection_identical_sets_yields_all_elements() {
        let a = build(&[1, 2, 3, 4, 5]);
        let result: Vec<&i32> = a.intersection(&a).collect();
        assert_eq!(result, [&1, &2, &3, &4, &5]);
    }

    #[test]
    fn intersection_partial_overlap() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[4, 5, 6, 7]);
        let result: Vec<&i32> = a.intersection(&b).collect();
        assert_eq!(result, [&4, &5]);
    }

    #[test]
    fn intersection_with_empty_set_is_empty() {
        let a = build(&[1, 2, 3]);
        let empty: BTreeSet<i32> = BTreeSet::new();
        let result: Vec<&i32> = a.intersection(&empty).collect();
        assert!(result.is_empty());

        let result_rev: Vec<&i32> = empty.intersection(&a).collect();
        assert!(result_rev.is_empty());
    }

    #[test]
    fn intersection_single_element() {
        let small = build(&[42]);
        let large = build(&(0..100).collect::<Vec<i32>>());
        let result: Vec<&i32> = small.intersection(&large).collect();
        assert_eq!(result, [&42]);

        let missing = build(&[999]);
        let result_missing: Vec<&i32> = missing.intersection(&large).collect();
        assert!(result_missing.is_empty());
    }

    #[test]
    fn intersection_search_path_answer() {
        let small = build(&[2, 4, 6, 8]);
        let large = build(&[8, 10, 12, 14]);
        let result: Vec<&i32> = small.intersection(&large).collect();
        assert_eq!(result, [&8]);
    }

    #[test]
    fn intersection_search_path_small_vs_large() {
        let small = build(&[2, 4, 6, 8]);
        let large = build(&(0..100).collect::<Vec<i32>>());
        let result: Vec<&i32> = small.intersection(&large).collect();
        assert_eq!(result, [&2, &4, &6, &8]);
    }

    #[test]
    fn intersection_stitch_path_similar_sizes() {
        let a = build(&[1, 3, 5, 7, 9, 11]);
        let b = build(&[2, 3, 5, 8, 9, 12]);
        let result: Vec<&i32> = a.intersection(&b).collect();
        assert_eq!(result, [&3, &5, &9]);
    }

    #[test]
    fn intersection_reverse_iteration() {
        let a = build(&[1, 2, 3, 4, 5, 6]);
        let b = build(&[3, 4, 5, 6, 7, 8]);
        let result: Vec<&i32> = a.intersection(&b).rev().collect();
        assert_eq!(result, [&6, &5, &4, &3]);
    }

    #[test]
    fn intersection_size_hint_stitch() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[3, 4, 5, 6, 7]);
        let inter = a.intersection(&b);
        let (low, high) = inter.size_hint();
        assert_eq!(high, Some(5));
        assert!(low == 0);
    }

    #[test]
    fn intersection_size_hint_search() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&(3..100).collect::<Vec<_>>());
        let inter = a.intersection(&b);
        let (low, high) = inter.size_hint();
        assert_eq!(high, Some(5));
        assert!(low == 0);
    }

    #[test]
    fn intersection_size_hint_answer() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[5, 6, 7, 8, 9]);
        let inter = a.intersection(&b);
        let (low, high) = inter.size_hint();
        assert_eq!(high, Some(1));
        assert!(low == 1);
    }

    #[test]
    fn intersection_size_hint_empty() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[6, 7, 8, 9]);
        let inter = a.intersection(&b);
        let (low, high) = inter.size_hint();
        assert_eq!(high, Some(0));
        assert!(low == 0);
    }

    #[test]
    fn intersection_fused_iterator() {
        let a = build(&[1, 2, 3]);
        let b = build(&[2, 3, 4]);
        let mut inter = a.intersection(&b);
        assert_eq!(inter.next(), Some(&2));
        assert_eq!(inter.next(), Some(&3));
        assert_eq!(inter.next(), None);
        // Fused: subsequent calls also return None
        assert_eq!(inter.next(), None);
    }

    #[test]
    fn intersection_debug_impl() {
        let a = build(&[1, 2, 3]);
        let b = build(&[2, 3, 4]);
        let inter = a.intersection(&b);
        // Debug must not panic and must produce non-empty output.
        use crate::string::String;
        use std::fmt::Write;
        let mut buf = String::new();
        write!(&mut buf, "{:?}", inter.inner).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn union_disjoint_sets() {
        let a = build(&[1, 3, 5]);
        let b = build(&[2, 4, 6]);
        let result: Vec<&i32> = a.union(&b).collect();
        assert_eq!(result, [&1, &2, &3, &4, &5, &6]);
    }

    #[test]
    fn union_identical_sets() {
        let a = build(&[1, 2, 3, 4, 5]);
        let result: Vec<&i32> = a.union(&a).collect();
        assert_eq!(result, [&1, &2, &3, &4, &5]);
    }

    #[test]
    fn union_partial_overlap() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[4, 5, 6, 7]);
        let result: Vec<&i32> = a.union(&b).collect();
        assert_eq!(result, [&1, &2, &3, &4, &5, &6, &7]);
    }

    #[test]
    fn union_with_empty_set() {
        let a = build(&[1, 2, 3]);
        let empty: BTreeSet<i32> = BTreeSet::new();
        let result: Vec<&i32> = a.union(&empty).collect();
        assert_eq!(result, [&1, &2, &3]);
        let result_rev: Vec<&i32> = empty.union(&a).collect();
        assert_eq!(result_rev, [&1, &2, &3]);
    }

    #[test]
    fn union_both_empty() {
        let a: BTreeSet<i32> = BTreeSet::new();
        let b: BTreeSet<i32> = BTreeSet::new();
        let result: Vec<&i32> = a.union(&b).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn union_size_hint_disjoint() {
        let a = build(&[1, 3, 5]);
        let b = build(&[2, 4, 6]);
        let u = a.union(&b);
        let (low, high) = u.size_hint();
        assert_eq!(low, 3);
        assert_eq!(high, Some(6));
    }

    #[test]
    fn union_size_hint_identical_sets() {
        let a = build(&[1, 2, 3]);
        let u = a.union(&a);
        let (low, high) = u.size_hint();
        assert_eq!(low, 3);
        assert_eq!(high, Some(6));
    }

    #[test]
    fn union_size_hint_one_empty() {
        let a: BTreeSet<i32> = BTreeSet::new();
        let b = build(&[1, 2, 3]);
        let u = a.union(&b);
        let (low, high) = u.size_hint();
        assert_eq!(low, 3);
        assert_eq!(high, Some(3));
    }

    #[test]
    fn union_size_hint_both_empty() {
        let a: BTreeSet<i32> = BTreeSet::new();
        let b: BTreeSet<i32> = BTreeSet::new();
        let u = a.union(&b);
        let (low, high) = u.size_hint();
        assert_eq!(low, 0);
        assert_eq!(high, Some(0));
    }

    #[test]
    fn union_size_hint_after_consumption() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[3, 4, 5, 6, 7]);
        let mut u = a.union(&b);
        assert_eq!(u.next(), Some(&1));
        let (low, high) = u.size_hint();
        assert_eq!(low, 5);
        assert_eq!(high, Some(9));
    }

    #[test]
    fn union_size_hint_unequal_sizes() {
        let a = build(&[1]);
        let b = build(&[10, 20, 30, 40]);
        let u = a.union(&b);
        let (low, high) = u.size_hint();
        assert_eq!(low, 4);
        assert_eq!(high, Some(5));
    }

    #[test]
    fn union_fused_iterator() {
        let a = build(&[1, 2, 3]);
        let b = build(&[2, 3, 4]);
        let mut u = a.union(&b);
        assert_eq!(u.next(), Some(&1));
        assert_eq!(u.next(), Some(&2));
        assert_eq!(u.next(), Some(&3));
        assert_eq!(u.next(), Some(&4));
        assert_eq!(u.next(), None);
        assert_eq!(u.next(), None);
    }

    // ── Difference tests ────────────────────────────────────────────────────────

    #[test]
    fn difference_disjoint_sets_returns_all_self() {
        let a = build(&[1, 2, 3]);
        let b = build(&[4, 5, 6]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&1, &2, &3]);
    }

    #[test]
    fn difference_identical_sets_is_empty() {
        let a = build(&[1, 2, 3, 4, 5]);
        let result: Vec<&i32> = a.difference(&a).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn difference_partial_overlap() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[3, 4, 5, 6, 7]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&1, &2]);
    }

    #[test]
    fn difference_with_empty_set_returns_all_self() {
        let a = build(&[1, 2, 3]);
        let empty: BTreeSet<i32> = BTreeSet::new();
        let result: Vec<&i32> = a.difference(&empty).collect();
        assert_eq!(result, [&1, &2, &3]);
    }

    #[test]
    fn difference_of_empty_set_is_empty() {
        let empty: BTreeSet<i32> = BTreeSet::new();
        let b = build(&[1, 2, 3]);
        let result: Vec<&i32> = empty.difference(&b).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn difference_stitch_path_similar_sizes() {
        let a = build(&[1, 3, 5, 7, 9, 11]);
        let b = build(&[3, 5, 9, 10, 11, 12]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&1, &7]);
    }

    #[test]
    fn difference_search_path_small_vs_large() {
        let small = build(&[2, 4, 6, 8]);
        let large = build(&(0..100).collect::<Vec<i32>>());
        let result: Vec<&i32> = small.difference(&large).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn difference_iterate_path_disjoint_ranges() {
        let a = build(&[1, 2, 3]);
        let b = build(&[10, 20, 30]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&1, &2, &3]);
    }

    #[test]
    fn difference_touching_at_min_skips_first() {
        let a = build(&[5, 6, 7]);
        let b = build(&[1, 2, 5]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&6, &7]);
    }

    #[test]
    fn difference_touching_at_max_skips_last() {
        let a = build(&[1, 2, 3]);
        let b = build(&[3, 4, 5]);
        let result: Vec<&i32> = a.difference(&b).collect();
        assert_eq!(result, [&1, &2]);
    }

    #[test]
    fn difference_size_hint_bounds() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[3, 4, 5, 6, 7]);
        let diff = a.difference(&b);
        let (low, high) = diff.size_hint();
        assert!(high.unwrap() <= 5);
        assert!(low == 0);
        assert!(low <= high.unwrap());
    }

    // ── Difference size_hint bound paths ────────────────────────────────────────

    #[test]
    fn difference_size_hint_iterate_exact() {
        // Disjoint ranges → Iterate path: other_len=0, so low=self_len, high=self_len (exact)
        let a = build(&[1, 2, 3]);
        let b = build(&[10, 20, 30]);
        let diff = a.difference(&b);
        let (low, high) = diff.size_hint();
        assert_eq!(low, 3);
        assert_eq!(high, Some(3));
    }

    #[test]
    fn difference_size_hint_empty_self() {
        // Empty self → Iterate with len 0: low=0, high=0
        let a: BTreeSet<i32> = BTreeSet::new();
        let b = build(&[1, 2, 3]);
        let diff = a.difference(&b);
        let (low, high) = diff.size_hint();
        assert_eq!(low, 0);
        assert_eq!(high, Some(0));
    }

    #[test]
    fn difference_size_hint_stitch_partial_overlap() {
        // Stitch path: a=[1..5], b=[3..7] → self_len=5, other_len=5
        // low = 5.saturating_sub(5) = 0, high = Some(5)
        let a = build(&[1, 2, 3, 4, 5]);
        let b = build(&[3, 4, 5, 6, 7]);
        let diff = a.difference(&b);
        let (low, high) = diff.size_hint();
        assert_eq!(low, 0);
        assert_eq!(high, Some(5));
    }

    #[test]
    fn difference_size_hint_search_path() {
        let small = build(&[2, 4, 6, 8]);
        let large = build(&(0..100).collect::<Vec<i32>>());
        let diff = small.difference(&large);
        let (low, high) = diff.size_hint();
        assert_eq!(low, 0);
        assert_eq!(high, Some(4));
    }

    #[test]
    fn difference_size_hint_stitch_mostly_disjoint() {
        let a = build(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let b = build(&[7, 8, 9, 10, 11, 12, 13, 14]);
        let diff = a.difference(&b);
        let (low, high) = diff.size_hint();
        assert_eq!(low, 0);
        assert_eq!(high, Some(8));
        let result: Vec<&i32> = diff.clone().collect();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn difference_size_hint_after_consumption() {
        let a = build(&[1, 2, 3, 4, 5]);
        let b: BTreeSet<i32> = BTreeSet::new();
        let mut diff = a.difference(&b);
        assert_eq!(diff.next(), Some(&1));
        assert_eq!(diff.next(), Some(&2));
        let (low, high) = diff.size_hint();
        assert_eq!(low, 3);
        assert_eq!(high, Some(3));
    }

    #[test]
    fn difference_fused_iterator() {
        let a = build(&[1, 2, 3]);
        let b = build(&[2]);
        let mut diff = a.difference(&b);
        assert_eq!(diff.next(), Some(&1));
        assert_eq!(diff.next(), Some(&3));
        assert_eq!(diff.next(), None);
        assert_eq!(diff.next(), None);
    }

    #[test]
    fn difference_clone() {
        let a = build(&[1, 2, 3, 4]);
        let b = build(&[2]);
        let diff = a.difference(&b);
        let cloned = diff.clone();
        let r1: Vec<&i32> = diff.collect();
        let r2: Vec<&i32> = cloned.collect();
        assert_eq!(r1, r2);
        assert_eq!(r1, [&1, &3, &4]);
    }

}
