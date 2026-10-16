//! Trait implementations for [`LinkedList`].
//!
//! This mirrors the std reference implementation, which keeps the derived
//! trait impls (`Clone`, `Debug` and friends) in a separate `traits` module.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

use olive_core::alloc::{Allocator, AllocatorTryClone};
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::try_traits::try_from_iterator::TryFromIterator;

use super::LinkedList;
use super::TryLinkedListWithCloneError;
use crate::alloc::{AllocError, Global};

impl<T: fmt::Debug, A: Allocator> fmt::Debug for LinkedList<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// SAFETY: the list is a plain chain of nodes reached only through `head` and
// `tail`; moving the list moves the whole chain, and the nodes are only ever
// touched from `&mut self` methods. Sending the list to another thread is
// sound exactly when `T` and the allocator are `Send`.
unsafe impl<T: Send, A: Allocator + Send> Send for LinkedList<T, A> {}

// SAFETY: shared access never creates mutable aliases into a node (the
// iteration methods hand out shared `&T` only), so sharing the list across
// threads is sound exactly when `T` and the allocator are `Sync`.
unsafe impl<T: Sync, A: Allocator + Sync> Sync for LinkedList<T, A> {}

// ---------------------------------------------------------------------------
// PartialEq / Eq
// ---------------------------------------------------------------------------

impl<T, U, A: Allocator, B: Allocator> PartialEq<LinkedList<U, B>> for LinkedList<T, A>
where
    T: PartialEq<U>,
{
    /// Compares two lists element by element, front to back. Lists of
    /// different lengths are never equal.
    fn eq(&self, other: &LinkedList<U, B>) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<T: Eq, A: Allocator> Eq for LinkedList<T, A> {}

// ---------------------------------------------------------------------------
// PartialOrd / Ord
// ---------------------------------------------------------------------------

impl<T: PartialOrd, A: Allocator, B: Allocator> PartialOrd<LinkedList<T, B>> for LinkedList<T, A> {
    /// Compares two lists lexicographically, front to back. A list that is a
    /// strict prefix of another orders before it.
    fn partial_cmp(&self, other: &LinkedList<T, B>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, A: Allocator> Ord for LinkedList<T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

impl<T: Hash, A: Allocator> Hash for LinkedList<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the length first. (We can't use the unstable `write_length_prefix`;
        // hashing the raw `usize` achieves the same disambiguation.)
        self.len.hash(state);
        self.iter().for_each(|elem| elem.hash(state));
    }
}

// ---------------------------------------------------------------------------
// TryClone
// ---------------------------------------------------------------------------

impl<T: TryClone, A: AllocatorTryClone> TryClone for LinkedList<T, A> {
    /// Fallibly clones the list, element by element via [`TryClone`].
    ///
    /// Each element is cloned and pushed to the back in order, so the copy
    /// preserves the source's element order.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if cloning the allocator reference or any element
    /// fails, or if allocating a node for a cloned element fails. On failure
    /// the partially built copy (elements included) is dropped and `self` is
    /// left untouched.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        let alloc = self.alloc.try_clone()?;
        let mut out = Self::new_in(alloc);
        for element in self.iter() {
            let cloned = element.try_clone()?;
            out.try_push_back(cloned).map_err(TryCloneError::Alloc)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TryFromIterator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> LinkedList<T, A> {
    /// Fallibly collects an iterator of values into a list on the given
    /// allocator. For the default [`Global`] allocator use
    /// [`TryFromIterator::try_from_iter`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating a node for an incoming element
    /// fails. On failure the partially constructed list is discarded
    /// (elements included).
    pub fn try_from_iter_in<I: IntoIterator<Item = T>>(
        iter: I,
        alloc: A,
    ) -> Result<Self, AllocError> {
        let mut list = Self::new_in(alloc);
        for element in iter {
            list.try_push_back(element)?;
        }
        Ok(list)
    }
}

impl<T> TryFromIterator<T> for LinkedList<T, Global> {
    type Error = AllocError;

    /// Fallibly collects an iterator into a list on the default [`Global`]
    /// allocator. For a custom allocator use
    /// [`LinkedList::try_from_iter_in`].
    ///
    /// This is the fallible analogue of std's `impl FromIterator<T> for
    /// LinkedList<T>`: on allocation failure the partially constructed list
    /// is dropped (elements included) and the error surfaces as
    /// [`AllocError`].
    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        Self::try_from_iter_in(iter, Global)
    }
}

// ---------------------------------------------------------------------------
// TryExtend / TryExtendFromSlice
// ---------------------------------------------------------------------------

impl<T, A: Allocator> TryExtend<T> for LinkedList<T, A> {
    type Error = AllocError;

    /// Attempts to extend the list with elements from a resumable source.
    ///
    /// # Errors
    ///
    /// On allocation failure, returns the remaining elements in a [`Resume`]
    /// so the caller can retry once memory is available, along with the
    /// [`AllocError`]. The stranded element comes first in the resume,
    /// followed by the untouched rest of the source.
    ///
    /// The elements committed before the failure are inside the list, so the
    /// call can be retried with the resume value to finish the job.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        let (head, mut inner, _hint) = source.decompose_with_size_hint();

        // Push the stranded head first, if any.
        if let Some(element) = head {
            if let Err((element, error)) = self.try_push_back_give_back(element) {
                return Err((Resume::new(element, inner), error));
            }
        }

        // Push the remainder one element at a time. Each push is atomic: on
        // allocation failure the list keeps exactly the elements committed so
        // far, and the current element is stranded in a `Resume` for retry.
        while let Some(element) = inner.next() {
            if let Err((element, error)) = self.try_push_back_give_back(element) {
                return Err((Resume::new(element, inner), error));
            }
        }
        Ok(())
    }
}

impl<'a, T: TryClone, A: Allocator> TryExtend<&'a T> for LinkedList<T, A> {
    type Error = TryLinkedListWithCloneError;

    /// Attempts to extend the list with borrowed references from a resumable
    /// source, cloning each element into the list.
    ///
    /// Only references pass through the source, so the stranded element in
    /// the [`Resume`] is a reference: retrying clones it again.
    ///
    /// # Errors
    ///
    /// On failure, returns the remaining elements in a [`Resume`] together
    /// with a [`TryLinkedListWithCloneError`] naming the cause, either a
    /// failed element clone or a node allocation failure, so the caller can
    /// retry from where it stopped. Elements cloned before the failure are
    /// inside the list.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = &'a T>,
    {
        let (head, mut inner, _hint) = source.decompose_with_size_hint();

        let mut pending = head.or_else(|| inner.next());
        while let Some(element) = pending {
            let cloned = match element.try_clone() {
                Ok(cloned) => cloned,
                Err(e) => {
                    return Err((
                        Resume::new(element, inner),
                        TryLinkedListWithCloneError::Clone(e),
                    ));
                }
            };
            if let Err((_cloned, error)) = self.try_push_back_give_back(cloned) {
                return Err((
                    Resume::new(element, inner),
                    TryLinkedListWithCloneError::Alloc(error),
                ));
            }
            pending = inner.next();
        }
        Ok(())
    }
}

impl<T: TryClone, A: Allocator> TryExtendFromSlice<T> for LinkedList<T, A> {
    type Error = TryLinkedListWithCloneError;

    /// Attempts to extend the list with elements cloned from a slice.
    ///
    /// # Errors
    ///
    /// On failure, returns the unconsumed tail of the slice together with
    /// the [`TryLinkedListWithCloneError`], so the caller can retry from
    /// where it stopped.
    ///
    /// The tail starts at the first element that failed to be cloned
    /// and stored. Elements cloned before the failure are inside the list.
    fn try_extend_from_slice<'s>(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        for (i, element) in other.iter().enumerate() {
            let cloned = element
                .try_clone()
                .map_err(|e| (&other[i..], TryLinkedListWithCloneError::Clone(e)))?;
            if let Err((_cloned, error)) = self.try_push_back_give_back(cloned) {
                return Err((&other[i..], TryLinkedListWithCloneError::Alloc(error)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    use std::sync::Arc;

    use olive_core::try_traits::TryFromIterator;
    use olive_core::try_traits::try_clone::TryCloneError;
    use olive_test_helpers::{BudgetedAlloc, Ledger, MaybeUnclonable, TrackedItem};

    use super::*;
    use crate::alloc::AllocError;

    // --- PartialEq / Eq ------------------------------------------------------

    #[test]
    fn equal_lists_compare_equal() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        for v in [1, 2, 3] {
            a.try_push_back(v).unwrap();
            b.try_push_back(v).unwrap();
        }
        assert_eq!(a, b);
        assert_eq!(b, a);
        // Two empty lists are equal too.
        assert_eq!(LinkedList::<i32>::new(), LinkedList::<i32>::new());
    }

    #[test]
    fn lists_of_different_lengths_or_contents_differ() {
        let build = |values: &[i32]| {
            let mut l: LinkedList<i32> = LinkedList::new();
            for v in values {
                l.try_push_back(*v).unwrap();
            }
            l
        };

        let short = build(&[1, 2]);
        let long = build(&[1, 2, 3]);
        assert_ne!(short, long);
        assert_ne!(long, short);

        // Same length, different order at the back.
        let a = build(&[1, 2, 3]);
        let b = build(&[1, 3, 2]);
        assert_ne!(a, b);

        // An empty list differs from any non-empty one.
        assert_ne!(LinkedList::<i32>::new(), build(&[1]));
    }

    #[test]
    fn lists_on_different_allocators_compare_by_elements() {
        let mut a = LinkedList::new_in(BudgetedAlloc::new(8));
        let mut b: LinkedList<i32> = LinkedList::new();
        for v in [4, 5, 6] {
            a.try_push_back(v).unwrap();
            b.try_push_back(v).unwrap();
        }
        assert_eq!(a, b);
    }

    // --- PartialOrd / Ord ----------------------------------------------------

    #[test]
    fn total_and_partial_ordering_is_lexicographic() {
        let build = |values: &[i32]| {
            let mut l: LinkedList<i32> = LinkedList::new();
            for v in values {
                l.try_push_back(*v).unwrap();
            }
            l
        };

        let empty = build(&[]);
        let one_two = build(&[1, 2]);
        let one_two_three = build(&[1, 2, 3]);
        let one_three = build(&[1, 3]);

        assert_eq!(one_two, one_two); // Ordering::Equal via Ord
        assert!(one_two < one_two_three, "a strict prefix orders first");
        assert!(one_two < one_three, "the first differing element decides");
        assert!(one_three > one_two);
        assert!(empty < one_two, "the empty list is the smallest prefix");

        assert_eq!(one_two.partial_cmp(&one_two), Some(Ordering::Equal)); // Ordering::Equal via Ord
        assert_eq!(
            one_two.partial_cmp(&one_two_three),
            Some(Ordering::Less),
            "a strict prefix orders first"
        );
        assert_eq!(
            one_two.partial_cmp(&one_three),
            Some(Ordering::Less),
            "the first differing element decides"
        );
        assert_eq!(one_three.partial_cmp(&one_two), Some(Ordering::Greater));
        assert_eq!(
            empty.partial_cmp(&one_two),
            Some(Ordering::Less),
            "the empty list is the smallest prefix"
        );
    }

    // --- Hash ----------------------------------------------------------------

    #[test]
    fn hash_equal_lists_have_equal_hashes() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        for v in [10, 20, 30] {
            a.try_push_back(v).unwrap();
            b.try_push_back(v).unwrap();
        }

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn hash_prefix_lists_differ() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        a.try_push_back(1).unwrap();
        b.try_push_back(1).unwrap();
        b.try_push_back(2).unwrap();

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        // Length prefix ensures these differ even though a is a prefix of b.
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn hash_same_length_different_content_lists_differ() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        a.try_push_back(1).unwrap();
        a.try_push_back(3).unwrap();
        b.try_push_back(1).unwrap();
        b.try_push_back(2).unwrap();

        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        // Length prefix ensures these differ even though a is a prefix of b.
        assert_ne!(ha.finish(), hb.finish());
    }

    // --- TryClone ------------------------------------------------------------

    #[test]
    fn try_clone_copies_the_elements_in_order() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for v in [5, 11, 3, 8, 1] {
            l.try_push_back(v).unwrap();
        }

        let clone = l.try_clone().expect("cloning i32 cannot fail");
        assert_eq!(clone, l);

        // The copy is independent of the source: consuming the original
        // leaves the clone's contents intact.
        assert_eq!(l.pop_front(), Some(5));
        assert_eq!(clone.len(), 5, "the clone is unaffected by the source");
        let collected: std::vec::Vec<i32> = clone.iter().copied().collect();
        assert_eq!(collected, std::vec![5, 11, 3, 8, 1]);
    }

    #[test]
    fn try_clone_of_a_tracked_list_owns_distinct_elements() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 0..3 {
            l.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        let clone = l.try_clone().expect("cloning u32 elements cannot fail");
        assert_eq!(clone.len(), 3);

        // The source's elements die with it, the clone's with the copy.
        drop(l);
        for id in 0..3u32 {
            assert_eq!(ledger.drop_count(id), 1, "id {id}");
        }
        assert_eq!(ledger.double_dropped(), std::vec::Vec::<u32>::new());

        drop(clone);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..6));
    }

    #[test]
    fn try_clone_failure_leaves_the_source_untouched() {
        let mut l: LinkedList<MaybeUnclonable<i32>> = LinkedList::new();
        for item in [
            MaybeUnclonable::clonable(3),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ] {
            l.try_push_back(item).unwrap();
        }

        let error = l
            .try_clone()
            .expect_err("the unclonable element fails the clone");
        assert!(matches!(error, TryCloneError::Other(_)));

        // `self` survives the failed clone intact and usable.
        assert_eq!(l.len(), 3, "no element was consumed");
        let collected: std::vec::Vec<i32> = l.iter().map(|item| item.inner).collect();
        assert_eq!(collected, std::vec![3, 9, 1]);
    }

    #[test]
    fn try_clone_oom_fails_cleanly() {
        let budget = BudgetedAlloc::new(2);
        let mut l: LinkedList<i32, _> = LinkedList::new_in(budget.clone());
        for item in [3i32, 1] {
            l.try_push_back(item).unwrap();
        }

        // The budget only has one allocation, so the second node of the copy cannot be
        // allocated even though the element clones may succeed.
        budget.set_budget(1);
        let error = l.try_clone().expect_err("the budget is spent");
        assert!(matches!(error, TryCloneError::Alloc(AllocError)));

        // `self` survives the failed clone intact and usable.
        assert_eq!(l.len(), 2, "no element was consumed");
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![3, 1]);
    }

    // --- TryFromIterator -----------------------------------------------------

    #[test]
    fn try_from_iter_collects_in_order() {
        let l: LinkedList<i32> = TryFromIterator::try_from_iter([1, 2, 3]).expect("allocation ok");
        assert_eq!(l.len(), 3);
        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&3));
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![1, 2, 3]);
    }

    #[test]
    fn try_from_iter_in_oom_drops_the_partials() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(2);
        let mut source = std::vec::Vec::new();
        for i in 0..5u32 {
            source.push(TrackedItem::construct(&ledger, i));
        }

        let outcome: Result<LinkedList<TrackedItem<u32>, _>, _> =
            LinkedList::try_from_iter_in(source, budget);
        assert!(matches!(outcome, Err(AllocError)));

        // The two committed elements and the three never-stored ones each
        // died exactly once when the partial list was discarded.
        drop(outcome);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    // --- TryExtend -----------------------------------------------------------

    #[test]
    fn try_extend_appends_in_order() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(7).unwrap();
        l.try_extend([5, 1, 9]).expect("extend ok");
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![7, 5, 1, 9]);

        // Extending from a vector works the same way.
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_extend(std::vec![4, 8, 2]).expect("extend ok");
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![4, 8, 2]);
    }

    #[test]
    fn try_extend_starving_midway_keeps_the_commit_and_returns_a_resume() {
        let alloc = BudgetedAlloc::new(1);
        let mut l: LinkedList<i32, _> = LinkedList::new_in(alloc.clone());

        let (resume, error) = l.try_extend(0..64).expect_err("one node is not enough");
        assert_eq!(error, AllocError);
        assert_eq!(l.len(), 1, "the committed element stays in the list");

        // The resume carries the stranded element followed by the untouched
        // rest of the range.
        assert_eq!(resume.head(), Some(&1));
        assert_eq!(resume.remainder().len(), 62);

        // With fresh budget the retry finishes the job.
        alloc.set_budget(63);
        l.try_extend(resume).expect("retry with a funded allocator");
        assert_eq!(l.len(), 64);
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        let expected: std::vec::Vec<i32> = (0..64).collect();
        assert_eq!(collected, expected);
    }

    // --- TryExtend<&T> -------------------------------------------------------

    #[test]
    fn try_extend_ref_clones_borrowed_elements() {
        let mut l: LinkedList<MaybeUnclonable<i32>> = LinkedList::new();
        let source = std::vec![
            MaybeUnclonable::clonable(5),
            MaybeUnclonable::clonable(1),
            MaybeUnclonable::clonable(9)
        ];
        l.try_extend(source.iter())
            .expect("cloning i32 cannot fail");
        let collected: std::vec::Vec<i32> = l.iter().map(|i| i.inner).collect();
        assert_eq!(collected, std::vec![5, 1, 9]);
    }

    #[test]
    fn try_extend_ref_reports_a_clone_failure_with_the_failed_element_first() {
        let mut l: LinkedList<MaybeUnclonable<i32>> = LinkedList::new();
        let source = std::vec![
            MaybeUnclonable::clonable(3),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ];
        let (resume, error) = l
            .try_extend(source.iter())
            .expect_err("the second element cannot clone");
        assert!(
            matches!(error, TryLinkedListWithCloneError::Clone(_)),
            "{error:?}"
        );
        assert_eq!(l.len(), 1, "the successful clone before the failure stays");

        // The failed element is stranded as the resume's head, the rest
        // follows in the remainder iterator.
        assert_eq!(resume.head().map(|item| item.inner), Some(9));
        assert_eq!(resume.remainder().len(), 1);
    }

    #[test]
    fn try_extend_ref_oom_strands_the_reference() {
        let budget = BudgetedAlloc::new(2);
        let mut l: LinkedList<i32, _> = LinkedList::new_in(budget.clone());
        let source = std::vec![0, 1, 2, 3];

        let (resume, error) = l
            .try_extend(source.iter())
            .expect_err("two nodes are not enough for four elements");
        assert!(
            matches!(error, TryLinkedListWithCloneError::Alloc(AllocError)),
            "{error:?}"
        );
        assert_eq!(l.len(), 2, "the committed elements stay in the list");

        // The reference to the failed element is stranded for a retry that
        // clones it again.
        assert_eq!(resume.head().copied(), Some(&2));
        assert_eq!(resume.remainder().len(), 1);

        budget.set_budget(2);
        l.try_extend(resume).expect("retry with a funded allocator");
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![0, 1, 2, 3]);
    }

    // --- TryExtendFromSlice --------------------------------------------------

    #[test]
    fn try_extend_from_slice_clones_the_whole_slice() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_extend_from_slice(&[4, 8, 2])
            .expect("cloning i32 cannot fail");
        l.try_extend_from_slice(&[])
            .expect("the empty slice is a no-op");
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![4, 8, 2]);
    }

    #[test]
    fn try_extend_from_slice_oom_returns_the_whole_slice() {
        let budget = BudgetedAlloc::new(0);
        let mut l: LinkedList<i32, _> = LinkedList::new_in(budget);
        let source = [1, 2, 3];
        let (rest, error) = l
            .try_extend_from_slice(&source)
            .expect_err("the budget is spent");
        assert!(
            matches!(error, TryLinkedListWithCloneError::Alloc(AllocError)),
            "{error:?}"
        );
        assert_eq!(rest, source, "nothing was consumed");
        assert!(l.is_empty());
    }

    #[test]
    fn try_extend_from_slice_clone_failure_halfway() {
        let mut l: LinkedList<MaybeUnclonable<i32>> = LinkedList::new();
        let source = std::vec![
            MaybeUnclonable::clonable(5),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ];
        let (rest, error) = l
            .try_extend_from_slice(&source)
            .expect_err("the middle element cannot clone");
        assert!(
            matches!(error, TryLinkedListWithCloneError::Clone(_)),
            "{error:?}"
        );
        assert_eq!(
            rest,
            &[MaybeUnclonable::unclonable(9), MaybeUnclonable::clonable(1)][..],
            "the remainder starts at the failing element"
        );
        assert_eq!(l.len(), 1, "the clone committed before the failure stays");
        assert_eq!(l.front().map(|item| item.inner), Some(5));
    }

    #[test]
    fn try_extend_from_slice_oom_halfway() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(1);
        let mut l: LinkedList<TrackedItem<u32>, _> = LinkedList::new_in(budget.clone());

        let mut source = std::vec::Vec::new();
        for i in 0..3u32 {
            source.push(TrackedItem::construct(&ledger, i));
        }

        let (rest, error) = l
            .try_extend_from_slice(&source)
            .expect_err("one node is not enough for three elements");
        assert!(
            matches!(error, TryLinkedListWithCloneError::Alloc(AllocError)),
            "{error:?}"
        );
        assert_eq!(rest.len(), 2, "the tail starts at the failed element");
        assert_eq!(l.len(), 1, "the committed element stays in the list");

        // Every tracked item dies exactly once: the three source items with
        // the vector, the committed one with the list, and the clone minted
        // for the failed push with the list's error path.
        drop(l);
        drop(source);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }
}
