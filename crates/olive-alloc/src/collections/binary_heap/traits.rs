//! Trait implementations for the binary heap.

use super::BinaryHeap;
use super::TryBinaryHeapWithCloneError;
use super::mutation::RebuildTailOnDrop;
use crate::{alloc::Global, vec::Vec};
use core::fmt;
use olive_core::recovery::{ResumableSource, Resume};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_extend::{TryExtend, TryExtendFromSlice};
use olive_core::{
    alloc::{Allocator, AllocatorTryClone},
    alloc_errors::TryReserveError,
    try_traits::TryFromIterator,
};

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<T: fmt::Debug, A: Allocator> fmt::Debug for BinaryHeap<T, A> {
    /// Formats the heap as a list of its elements in arbirary order (level order
    /// on clean heaps, arbitrary order on poisoned heaps), *not* in sorted order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Poisoned: the level order is not even a heap any more, so
        // listing it would mislead. std has no poison state and no
        // equivalent branch.
        if self.poisoned {
            return f.write_str("<poisoned>");
        }
        f.debug_list().entries(self.data.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// Cloning
// ---------------------------------------------------------------------------

impl<T: TryClone, A: AllocatorTryClone> TryClone for BinaryHeap<T, A> {
    /// Fallibly clones the heap, element by element, onto a freshly cloned
    /// allocator.
    ///
    /// The backing buffer is copied verbatim. The poisoned flag is
    /// carried across unchanged — cloning a poisoned heap yields a poisoned
    /// heap, which stays unusable for order-dependent operations until
    /// [`rebuild`](BinaryHeap::rebuild).
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocator or a capacity reservation
    /// fails, or if any element's [`try_clone`](TryClone::try_clone) fails. On
    /// failure the partially built copy (elements included) is dropped and
    /// `self` is left untouched.
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self {
            data: self.data.try_clone()?,
            poisoned: self.poisoned,
        })
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl<T: Ord> TryFromIterator<T> for BinaryHeap<T, Global> {
    type Error = TryReserveError;

    /// Collects the iterator into a vector on the [`Global`] allocator, then
    /// rebuilds it into a valid heap in linear time.
    ///
    /// This is the fallible analogue of std's `impl FromIterator<T> for
    /// BinaryHeap<T>`: on allocation failure the partially collected vector is
    /// dropped (elements included) and the error surfaces as
    /// [`TryReserveError`].
    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        // SAFETY: The heap is rebuilt after and can only be saved.
        let mut heap = unsafe { Self::with_data(Vec::try_from_iter_in(iter, Global)?) };
        heap.rebuild();
        Ok(heap)
    }
}

// ---------------------------------------------------------------------------
// Fallible extension
// ---------------------------------------------------------------------------

//<LLM-generated
impl<T: Ord, A: Allocator> TryExtend<T> for BinaryHeap<T, A> {
    type Error = TryReserveError;

    /// Attempts to extend the heap with the elements of a resumable source.
    ///
    /// # Errors
    ///
    /// On allocation failure, returns the remaining elements in a
    /// [`Resume`] together with the [`TryReserveError`]: the stranded
    /// element followed by the untouched rest of the source.
    ///
    /// The elements committed before the failure are inside the heap,
    /// so the call can be retried with the resume value to finish the job.
    ///
    /// Errors alone cannot poison the heap.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned ([`is_poisoned`](Self::is_poisoned)).
    ///
    /// If an [`Ord`] comparison panics during the closing rebuild, the heap
    /// is poisoned with every element already consumed from the source
    /// still inside it; [`rebuild`](Self::rebuild) restores the invariant.
    ///
    /// A panic from the source itself might not poison the heap as long as
    /// the [`Ord`] does not panic.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = T>,
    {
        self.assume_unpoisoned();
        let committed = self.data.len();
        let mut guard = RebuildTailOnDrop::new(self, committed);
        // Batch transfer under the guard: `Vec::try_extend` owns the
        // reservation strategy and the resume protocol.
        guard.data.try_extend(source)?;
        Ok(())
    }
}

impl<'a, T: Ord + TryClone, A: Allocator> TryExtend<&'a T> for BinaryHeap<T, A> {
    type Error = TryBinaryHeapWithCloneError;

    /// Attempts to extend the heap with borrowed references from a
    /// resumable source, cloning each element into the heap.
    ///
    /// # Errors
    ///
    /// On failure, returns the remaining elements in a [`Resume`] together
    /// with a [`TryBinaryHeapWithCloneError`] naming the cause: either a
    /// failed element clone or an allocation failure. Elements cloned
    /// before the failure are inside the heap.
    ///
    /// Errors alone cannot poison the heap.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned ([`is_poisoned`](Self::is_poisoned)).
    ///
    /// If an `Ord` comparison panics during the closing rebuild, the heap
    /// is poisoned with every element cloned so far still inside it;
    /// [`rebuild`](Self::rebuild) restores the invariant.
    ///
    /// A panic from the source itself might not poison the heap as long as
    /// the [`Ord`] does not panic.
    fn try_extend<S>(&mut self, source: S) -> Result<(), (Resume<S::Inner>, Self::Error)>
    where
        S: ResumableSource<Item = &'a T>,
    {
        self.assume_unpoisoned();
        let (head, mut inner, hint) = source.decompose_with_size_hint();

        // Reserve elements ahead of time if possible.
        let _ = self.try_reserve_adaptive(hint.estimated_total());

        let committed = self.data.len();
        let mut guard = RebuildTailOnDrop::new(self, committed);

        let mut pending = head.or_else(|| inner.next());
        while let Some(val) = pending {
            let cloned = match val.try_clone() {
                Ok(cloned) => cloned,
                Err(e) => {
                    return Err((
                        Resume::new(val, inner),
                        TryBinaryHeapWithCloneError::Clone(e),
                    ));
                }
            };
            if let Err((_cloned, err)) = guard.data.try_push_give_back(cloned) {
                return Err((
                    Resume::new(val, inner),
                    TryBinaryHeapWithCloneError::Reserve(err),
                ));
            }
            pending = inner.next();
        }

        Ok(())
    }
}

impl<T: Ord + TryClone, A: Allocator> TryExtendFromSlice<T> for BinaryHeap<T, A> {
    type Error = TryBinaryHeapWithCloneError;

    /// Attempts to extend the heap with elements cloned from a slice.
    ///
    /// # Errors
    ///
    /// If the reservation fails, the *whole* slice comes back together
    /// with [`TryBinaryHeapWithCloneError::Reserve`].
    ///
    /// If a clone fails, the unconsumed tail starting at that element
    /// comes back with [`TryBinaryHeapWithCloneError::Clone`];
    /// the clones committed so far are inside the heap, so the
    /// call can be retried with the returned remainder.
    ///
    /// Errors alone cannot poison the heap.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned ([`is_poisoned`](Self::is_poisoned)).
    ///
    /// If an `Ord` comparison panics during the closing rebuild, the heap
    /// is poisoned with every element cloned so far still inside it;
    /// [`rebuild`](Self::rebuild) restores the invariant.
    fn try_extend_from_slice<'s>(&mut self, other: &'s [T]) -> Result<(), (&'s [T], Self::Error)> {
        self.assume_unpoisoned();
        let committed = self.data.len();
        let mut guard = RebuildTailOnDrop::new(self, committed);
        // Same shape as `try_extend`: the vector performs the whole
        // reserve-and-clone transfer and owns the remainder protocol; the
        // clones it consumed join the heap when the guard gets out of scope.
        guard
            .data
            .try_extend_from_slice(other)
            .map_err(|(rest, e)| (rest, TryBinaryHeapWithCloneError::from(e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::TryFromIterator;
    use olive_test_helpers::BudgetedAlloc;
    use olive_test_helpers::PanicArmer;
    use olive_test_helpers::PanickyOrd;
    use std::sync::Arc;

    // --- Debug ---------------------------------------------------------------

    #[test]
    fn format_shows_the_elements_level_order() {
        let heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([100, 80, 70, 60, 50, 10]).expect("allocation ok");
        let text = std::format!("{heap:?}");
        assert!(
            text.starts_with('[') && text.ends_with(']'),
            "should render as a list: {text}"
        );
        assert!(text.contains("100"), "root should be shown: {text}");
    }

    // --- TryClone ------------------------------------------------------------

    #[test]
    fn try_clone_copies_the_elements_and_the_peek() {
        let mut heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([5, 11, 3, 8, 1]).expect("allocation ok");
        let mut clone = heap.try_clone().expect("cloning i32 cannot fail");

        assert_eq!(clone.len(), heap.len());
        assert_eq!(clone.peek(), heap.peek());

        // The copy is independent of the source: mutating the original
        // leaves the clone's contents and ordering intact.
        assert_eq!(heap.pop(), Some(11));
        assert_eq!(clone.len(), 5, "the clone is unaffected by the source");
        assert!(!clone.is_poisoned());
        assert_eq!(drain(&mut clone), std::vec![11, 8, 5, 3, 1]);
    }

    #[test]
    fn try_clone_preserves_the_poison_flag() {
        // Poison a heap through an armed comparison, then clone it.
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> = TryFromIterator::try_from_iter(
            [3, 1, 2]
                .iter()
                .map(|value| PanickyOrd::new(*value, armer.clone())),
        )
        .expect("allocation ok");

        armer.arm();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            heap.try_extend([10, 20, 30].iter().map(|value| PanickyOrd::calm(*value)))
                .expect("allocation succeeds; the detonation comes from the rebuild");
        }));
        assert!(outcome.is_err(), "the armed rebuild must detonate");
        assert!(heap.is_poisoned(), "the heap is quarantined");
        armer.disarm();

        let mut clone = heap.try_clone().expect("cloning runs no comparison");
        assert!(clone.is_poisoned(), "the quarantine is carried across");
        assert_eq!(clone.len(), heap.len());

        // Both copies recover through rebuild and drain the same multiset.
        heap.rebuild();
        clone.rebuild();
        assert!(!heap.is_poisoned() && !clone.is_poisoned());

        let mut expected = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            expected.push(item.value);
        }
        let mut actual = std::vec::Vec::new();
        while let Some(item) = clone.pop() {
            actual.push(item.value);
        }
        assert_eq!(expected, std::vec![30, 20, 10, 3, 2, 1]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_clone_failure_leaves_the_source_untouched() {
        use olive_test_helpers::MaybeUnclonable;
        let mut heap: BinaryHeap<MaybeUnclonable<i32>> = TryFromIterator::try_from_iter([
            MaybeUnclonable::clonable(3),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ])
        .expect("allocation ok");

        let error = heap
            .try_clone()
            .expect_err("the unclonable element fails the clone");
        assert!(matches!(error, TryCloneError::Other(_)));

        // `self` survives the failed clone intact and usable.
        assert_eq!(heap.len(), 3, "no element was consumed");
        assert_eq!(heap.peek().map(|item| item.inner), Some(9));
        assert!(!heap.is_poisoned());
        let mut actual = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            actual.push(item.inner);
        }
        assert_eq!(actual, [9i32, 3, 1]);
    }

    // --- TryFromIterator -------------------------------------------------------

    #[test]
    fn try_from_iter_produces_a_valid_max_heap() {
        // Unordered input: the constructor must heapify, so peek sees the
        // maximum and popping drains in decreasing order.
        let mut heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([5, 11, 3, 8, 1, 11, 0]).expect("allocation ok");
        assert_eq!(heap.len(), 7);
        assert_eq!(heap.peek(), Some(&11));
        let mut out = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, std::vec![11, 11, 8, 5, 3, 1, 0]);
    }

    // --- TryExtend -----------------------------------------------------------

    fn drain<T: Ord>(heap: &mut BinaryHeap<T>) -> std::vec::Vec<T> {
        let mut out = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        out
    }

    /// Hides the iterator's upper size bound so the batch append cannot
    /// buy the whole transfer in one reservation.
    #[derive(Debug)]
    struct NoUpper<I> {
        inner: I,
    }

    impl<I: Iterator> Iterator for NoUpper<I> {
        type Item = I::Item;

        fn next(&mut self) -> Option<I::Item> {
            self.inner.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, None)
        }
    }

    #[test]
    fn try_extend_appends_and_reheapifies() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        heap.try_extend([5, 1, 9, 0]).expect("extend ok");
        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some(&9));
        assert_eq!(drain(&mut heap), std::vec![9, 5, 1, 0]);
        // Extending a non-empty heap merges the multisets.
        let mut heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([7, 2]).expect("allocation ok");
        heap.try_extend(std::vec![8, 1]).expect("extend ok");
        assert_eq!(drain(&mut heap), std::vec![8, 7, 2, 1]);
    }

    #[test]
    fn try_extend_starving_midway_keeps_the_commit_and_returns_a_resume() {
        let alloc = BudgetedAlloc::new(1);
        let mut heap: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc.clone());

        // The NoUpper forces multiple reallocations.
        let (resume, error) = heap
            .try_extend(NoUpper { inner: 0..64 })
            .expect_err("one block is not enough");
        assert!(error.is_alloc(), "{error:?}");
        assert!(!heap.is_empty(), "consumed elements stay in the heap");
        assert!(
            heap.len() < 64,
            "the source cannot have been consumed wholly"
        );
        assert!(!heap.is_poisoned(), "the window rebuilt what it committed");

        // The resume carries the untouched rest; with fresh budget the
        // retry finishes the job.
        alloc.set_budget(64);
        heap.try_extend(resume)
            .expect("retry with a funded allocator");

        let expected: std::vec::Vec<i32> = (0..64).rev().collect();
        let mut out = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn try_extend_poisons_when_the_closing_rebuild_detonates() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> = TryFromIterator::try_from_iter(
            [3, 1, 2]
                .iter()
                .map(|value| PanickyOrd::new(*value, armer.clone())),
        )
        .expect("allocation ok");

        armer.arm();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            heap.try_extend(
                [10, 20, 30, 40, 50, 60]
                    .iter()
                    .map(|value| PanickyOrd::calm(*value)),
            )
            .expect("allocation succeeds; the detonation comes from the rebuild");
        }));
        assert!(outcome.is_err(), "the armed rebuild must detonate");
        assert!(heap.is_poisoned(), "the window quarantines the heap");
        assert_eq!(heap.len(), 9, "every consumed element is honored");

        armer.disarm();
        heap.rebuild();
        assert!(!heap.is_poisoned());
        let mut out = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item.value);
        }
        assert_eq!(out, std::vec![60, 50, 40, 30, 20, 10, 3, 2, 1]);
    }

    // --- TryExtend<&T> -------------------------------------------------------

    #[test]
    fn try_extend_ref_clones_borrowed_elements() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        let source = std::vec![5, 1, 9];
        heap.try_extend(source.iter())
            .expect("cloning i32 cannot fail");
        assert_eq!(source, std::vec![5, 1, 9], "the source is untouched");
        assert_eq!(drain(&mut heap), std::vec![9, 5, 1]);
    }

    #[test]
    fn try_extend_ref_reports_a_clone_failure_with_the_failed_element_first() {
        use olive_test_helpers::MaybeUnclonable;
        let mut heap: BinaryHeap<MaybeUnclonable<i32>> = BinaryHeap::new();
        let source = std::vec![
            MaybeUnclonable::clonable(3),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ];
        let (resume, error) = heap
            .try_extend(source.iter())
            .expect_err("the second element cannot clone");
        assert!(matches!(error, TryBinaryHeapWithCloneError::Clone(_)));
        assert_eq!(
            heap.len(),
            1,
            "the successful clone before the failure stays"
        );
        assert_eq!(heap.peek().map(|item| item.inner), Some(3));

        // The failed element is stranded as the resume's head, the rest
        // follows in the remainder iterator.
        let (head, tail) = resume.into_parts();
        assert_eq!(head.map(|item| item.inner), Some(9));
        assert_eq!(tail.len(), 1);

        // TODO: There is no way to make the affected element clonable (yet). May need to use an armer
        // (e.g. FailArmer) pattern instead.
    }

    // --- TryExtendFromSlice ----------------------------------------------------

    #[test]
    fn try_extend_from_slice_clones_the_whole_slice() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        heap.try_extend_from_slice(&[4, 8, 2])
            .expect("cloning i32 cannot fail");
        heap.try_extend_from_slice(&[])
            .expect("the empty slice is a no-op");
        assert_eq!(drain(&mut heap), std::vec![8, 4, 2]);
    }

    #[test]
    fn try_extend_from_slice_reserve_failure_returns_the_whole_slice() {
        let alloc = BudgetedAlloc::new(0);
        let mut heap: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc);
        let source = [1, 2, 3];
        let (rest, error) = heap
            .try_extend_from_slice(&source)
            .expect_err("the budget is spent");
        assert!(matches!(error, TryBinaryHeapWithCloneError::Reserve(_)));
        assert_eq!(rest, source, "nothing was consumed");
        assert!(heap.is_empty());
    }

    #[test]
    fn try_extend_from_slice_clone_failure_returns_the_remainder() {
        use olive_test_helpers::MaybeUnclonable;
        let mut heap: BinaryHeap<MaybeUnclonable<i32>> = BinaryHeap::new();
        let source = std::vec![
            MaybeUnclonable::clonable(5),
            MaybeUnclonable::unclonable(9),
            MaybeUnclonable::clonable(1),
        ];
        let (rest, error) = heap
            .try_extend_from_slice(&source)
            .expect_err("the middle element cannot clone");
        assert!(matches!(error, TryBinaryHeapWithCloneError::Clone(_)));
        assert_eq!(
            rest,
            &[MaybeUnclonable::unclonable(9), MaybeUnclonable::clonable(1)][..],
            "the remainder starts at the failing element"
        );
        assert_eq!(
            heap.len(),
            1,
            "the clone committed before the failure stays"
        );
        assert_eq!(heap.peek().map(|item| item.inner), Some(5));
        assert!(!heap.is_poisoned(), "the window rebuilt the committed tail");
    }
}
