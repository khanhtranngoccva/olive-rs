//! Growth operations: pushing elements and merging iterators or heaps.

use super::BinaryHeap;
use super::poison::PoisonOnUnwind;
use crate::alloc::Allocator;
use olive_core::alloc_errors::TryReserveError;

impl<T: Ord, A: Allocator> BinaryHeap<T, A> {
    /// Attempts to insert an item into the binary heap, returning an error on
    /// failure. The element is dropped on error - use
    /// [`try_push_give_back`](Self::try_push_give_back) to keep it.
    ///
    /// # Panics
    ///
    /// If the `Ord` comparison panics during the insertion, the element stays
    /// in the heap and the heap is poisoned. [`rebuild`](Self::rebuild) needs
    /// to be called to recover the heap.
    ///
    /// # Errors
    ///
    /// The [`TryReserveError`] is returned on failure - either the allocator failed
    /// or the capacity computation overflowed. The heap is unchanged in both cases.
    #[inline]
    pub fn try_push(&mut self, item: T) -> Result<(), TryReserveError> {
        match self.try_push_give_back(item) {
            Ok(()) => Ok(()),
            Err((_item, error)) => Err(error),
        }
    }

    /// Attempts to insert an item into the binary heap, giving both the item
    /// and the error back on failure.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned ([`is_poisoned`](Self::is_poisoned)),
    /// in this case one must restore with [`rebuild`](Self::rebuild) first.
    ///
    /// If the `Ord` comparison panics during the insertion, the element stays
    /// in the heap and the heap is poisoned. [`rebuild`](Self::rebuild) needs
    /// to be called to recover the heap.
    ///
    /// # Errors
    ///
    /// On failure, returns the element together with the [`TryReserveError`]
    /// The heap is unchanged in that case.
    pub fn try_push_give_back(&mut self, item: T) -> Result<(), (T, TryReserveError)> {
        self.assume_unpoisoned();

        let origin = self.data.len();
        if let Err((item, error)) = self.data.try_push_give_back(item) {
            return Err((item, error));
        }

        let mut guard = PoisonOnUnwind::new(self);
        // SAFETY: `origin` is the index the element was just written to, so
        // `origin < self.data.len()`.
        unsafe { guard.sift_up(0, origin) };
        core::mem::forget(guard);

        Ok(())
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// The elements cross as one comparison-free buffer move and the combined
    /// heap is then repaired by a single O(n) rebuild.
    ///
    /// # Errors
    ///
    /// Returns the [`TryReserveError`] if space for `other.len()` elements
    /// could not be reserved. In this case, both heaps are intact.
    ///
    /// # Panics
    ///
    /// Panics if either heap is poisoned or if the [`Ord`] implementation panics.
    ///
    /// If the `Ord` comparison panics during the rebuild, `self` is poisoned.
    /// Every element is already inside `self` - nothing is lost - and
    /// [`rebuild`](Self::rebuild) must be called to restore the invariant.
    pub fn try_append(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
        self.assume_unpoisoned();
        other.assume_unpoisoned();
        if self.is_empty() {
            core::mem::swap(self, other);
            return Ok(());
        }
        if other.is_empty() {
            return Ok(());
        }
        let old_len = self.len();
        // The vector-level transfer is comparison-free and reserves the exact
        // amount up front. On failure, `other` keeps its elements and the
        // allocations and both heaps stay valid.
        self.data.try_append(&mut other.data)?;
        // The vector-level transfer is atomic and comparison-free, so the critical
        // section starts semantically right here.
        let mut quarantine = PoisonOnUnwind::new(self);
        quarantine.rebuild_tail(old_len);
        core::mem::forget(quarantine);
        Ok(())
    }

    /// Attempts to reserve capacity for at least `additional` more elements.
    ///
    /// The collection may grow beyond `additional`, mirroring
    /// [`Vec::try_reserve`]'s amortized growth.
    ///
    /// [`Vec::try_reserve`]: crate::vec::Vec::try_reserve
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails or the
    /// capacity computation overflows. The heap is unchanged in both cases.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.data.try_reserve(additional)
    }

    /// Attempts to reserve the exact capacity for `additional` more
    /// elements.
    ///
    /// Prefer [`try_reserve`](Self::try_reserve) unless the exact element
    /// count is already known: an exact reservation defeats the amortized
    /// doubling and can make repeated growth quadratic.
    ///
    /// Like [`try_reserve`](Self::try_reserve), this runs no comparisons
    /// and leaves the poison flag alone.
    ///
    /// [`Vec::try_reserve_exact`]: crate::vec::Vec::try_reserve_exact
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if growing the buffer fails or the
    /// capacity computation overflows. The heap is unchanged in both cases.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.data.try_reserve_exact(additional)
    }

    /// Attempts to reserve capacity for at least `additional` more
    /// elements: amortized growth first, falling back to an exact
    /// reservation when the amortized one fails, mirroring
    /// [`Vec::try_reserve_adaptive`].
    ///
    /// Like [`try_reserve`](Self::try_reserve), this runs no comparisons
    /// and leaves the poison flag alone.
    ///
    /// [`Vec::try_reserve_adaptive`]: crate::vec::Vec::try_reserve_adaptive
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if both the amortized and the exact
    /// reservation fail. The heap is unchanged in that case.
    pub fn try_reserve_adaptive(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.data.try_reserve_adaptive(additional)
    }

    /// Attempts to shrink the capacity as much as possible while still
    /// holding `min_capacity` elements, mirroring [`Vec::try_shrink_to`].
    ///
    /// A `min_capacity` below the current length is treated as the
    /// length.
    ///
    /// [`Vec::try_shrink_to`]: crate::vec::Vec::try_shrink_to
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the shrink cannot reallocate. The
    /// heap keeps its old buffer and capacity in that case.
    pub fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
        self.data.try_shrink_to(min_capacity)
    }

    /// Attempts to shrink the capacity down to the element count,
    /// releasing every unused slot, mirroring [`Vec::try_shrink_to_fit`].
    ///
    /// [`Vec::try_shrink_to_fit`]: crate::vec::Vec::try_shrink_to_fit
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the shrink cannot reallocate. The
    /// heap keeps its old buffer and capacity in that case.
    pub fn try_shrink_to_fit(&mut self) -> Result<(), TryReserveError> {
        self.data.try_shrink_to_fit()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::BinaryHeap;
    use olive_core::{
        alloc_errors::TryReserveError, try_traits::try_from_iterator::TryFromIterator,
    };
    use olive_test_helpers::{BudgetedAlloc, PanicArmer, PanickyOrd};
    use std::panic;
    use std::sync::Arc;
    use std::vec;
    use std::vec::Vec;

    fn heap_of<I: IntoIterator<Item = i32>>(items: I) -> BinaryHeap<i32> {
        BinaryHeap::try_from_iter(items).expect("allocation ok")
    }

    fn budgeted_heap(values: &[i32], alloc: BudgetedAlloc) -> BinaryHeap<i32, BudgetedAlloc> {
        let mut heap = BinaryHeap::new_in(alloc);
        for value in values {
            heap.try_push(*value).expect("budget covers the fixture");
        }
        heap
    }

    fn drain_i32(heap: &mut BinaryHeap<i32>) -> Vec<i32> {
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        out
    }

    fn drain_i32_budgeted(heap: &mut BinaryHeap<i32, BudgetedAlloc>) -> Vec<i32> {
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        out
    }

    #[test]
    fn try_push_then_drain_is_sorted() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        for value in [5, 1, 9, 9, 0] {
            heap.try_push(value).expect("infallible on this budget");
        }
        assert_eq!(drain_i32(&mut heap), vec![9, 9, 5, 1, 0]);
    }

    #[test]
    fn try_push_give_back_hands_element_back_on_failure() {
        let alloc = BudgetedAlloc::new(1);
        let mut heap: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc.clone());
        // The one budgeted allocation buys the amortized first block,
        // which is wider than a single slot; fill it completely so the
        // next push is genuinely the one that must re-ask the allocator.
        let mut value = 0;
        while heap.is_empty() || heap.len() < heap.capacity() {
            heap.try_push(value)
                .expect("the one budgeted allocation covers the first block");
            value += 1;
        }

        let (returned, error) = heap
            .try_push_give_back(100)
            .expect_err("the exhausted budget must refuse a second block");
        assert!(error.is_alloc(), "{error:?}");
        assert_eq!(returned, 100, "the rejected element comes back untouched");

        assert_eq!(heap.len(), value as usize, "the heap did not move");
        assert_eq!(heap.peek(), Some(&(value - 1)));
        assert_eq!(
            alloc.remaining(),
            0,
            "the failed reserve must not debit the budget"
        );
    }

    #[test]
    fn try_push_uses_preexisting_capacity_without_allocating() {
        let alloc = BudgetedAlloc::new(2);
        let mut heap: BinaryHeap<i32, _> =
            BinaryHeap::try_with_capacity_in(4, alloc.clone()).expect("one budgeted allocation");
        for value in 0..4 {
            heap.try_push(value).expect("already-reserved slot");
        }
        assert_eq!(
            alloc.remaining(),
            1,
            "fills within capacity must not allocate"
        );
        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some(&3));
    }

    #[test]
    fn zst_pushes_need_no_allocation() {
        let alloc = BudgetedAlloc::new(0);
        let mut heap: BinaryHeap<(), _> = BinaryHeap::new_in(alloc.clone());
        heap.try_push(())
            .expect("a ZST slot never asks the allocator");
        heap.try_push(()).expect("nor does the second one");
        assert_eq!(heap.len(), 2);
        assert_eq!(alloc.remaining(), 0);
    }

    fn armed_heap(values: &[i32], armer: &Arc<PanicArmer>) -> BinaryHeap<PanickyOrd<i32>> {
        BinaryHeap::try_from_iter(
            values
                .iter()
                .map(|value| PanickyOrd::new(*value, Arc::clone(armer))),
        )
        .expect("allocation ok")
    }

    #[test]
    fn try_push_honors_the_element_when_the_comparison_panics() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap = armed_heap(&[1, 0], &armer);
        armer.arm();
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            heap.try_push(PanickyOrd::calm(2))
        }));
        assert!(
            heap.is_poisoned(),
            "the failed insertion must quarantine the heap loudly"
        );
        assert_eq!(heap.len(), 3, "the element is honored, not discarded");
        // With the fault withdrawn, the poisoned heap is fully repairable.
        armer.disarm();
        heap.rebuild();
        assert!(!heap.is_poisoned());
        assert_eq!(heap.peek().map(|item| item.value), Some(2));
    }

    #[test]
    fn try_push_give_back_honors_the_element_when_the_comparison_panics() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap = armed_heap(&[1, 0], &armer);
        armer.arm();
        let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            heap.try_push_give_back(PanickyOrd::calm(2))
        }));
        assert!(
            outcome.is_err(),
            "the armed comparison must detonate the sift"
        );
        assert!(
            heap.is_poisoned(),
            "the failed insertion must quarantine the heap loudly"
        );
        assert_eq!(heap.len(), 3, "the element is honored, not discarded");
        armer.disarm();
        heap.rebuild();
        assert!(!heap.is_poisoned());
        assert_eq!(heap.peek().map(|item| item.value), Some(2));
    }

    #[test]
    fn try_append_merges_two_heaps() {
        let mut left = heap_of(vec![9, 5, 1]);
        let mut right = heap_of(vec![8, 7, 6, 2]);
        left.try_append(&mut right).expect("reserve succeeds");
        assert!(right.is_empty());
        assert_eq!(drain_i32(&mut left), vec![9, 8, 7, 6, 5, 2, 1]);
    }

    #[test]
    fn try_append_failure_leaves_both_heaps_intact() {
        let alloc = BudgetedAlloc::new(8);
        // Fill the left heap to capacity so the merge genuinely needs a
        // new block: an append that fits in reserved capacity must not,
        // and does not, allocate.
        let mut left = budgeted_heap(&[9, 8, 7, 1], alloc.clone());
        let mut right = budgeted_heap(&[6, 5], alloc.clone());
        alloc.drain();

        let error: TryReserveError = left
            .try_append(&mut right)
            .expect_err("an exhausted budget must refuse the merge");
        assert!(error.is_alloc(), "{error:?}");
        assert_eq!(left.len(), 4, "the left heap must not have moved");
        assert_eq!(right.len(), 2, "the right heap must have kept its data");
        assert_eq!(drain_i32_budgeted(&mut left), vec![9, 8, 7, 1]);
        assert_eq!(drain_i32_budgeted(&mut right), vec![6, 5]);
    }

    #[test]
    fn try_append_of_empty_is_free() {
        let alloc = BudgetedAlloc::new(8);
        let mut left = budgeted_heap(&[3, 1, 2], alloc.clone());
        let mut right: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc.clone());
        alloc.drain();
        left.try_append(&mut right)
            .expect("empty appends are trivial");
        right
            .try_append(&mut left)
            .expect("empty appends are trivial");
        assert_eq!(drain_i32_budgeted(&mut right), vec![3, 2, 1]);
        assert_eq!(alloc.remaining(), 0);
    }

    // --- Capacity operations -------------------------------------------------

    #[test]
    fn try_reserve_satisfies_the_request_without_disturbing_the_heap() {
        let mut heap = heap_of([5, 1, 9, 9, 0]);
        heap.try_reserve(64).expect("global allocator has room");
        assert!(
            heap.capacity() >= heap.len() + 64,
            "the request must be met"
        );
        assert!(!heap.is_poisoned());
        assert_eq!(drain_i32(&mut heap), vec![9, 9, 5, 1, 0]);
    }

    #[test]
    fn try_reserve_zero_never_asks_the_allocator() {
        let alloc = BudgetedAlloc::new(8);
        let mut heap = budgeted_heap(&[3, 1, 2], alloc.clone());
        alloc.drain();
        heap.try_reserve(0).expect("no room is needed");
        heap.try_reserve_exact(0).expect("no room is needed");
        heap.try_reserve_adaptive(0).expect("no room is needed");
        assert_eq!(alloc.remaining(), 0);
        assert_eq!(drain_i32_budgeted(&mut heap), vec![3, 2, 1]);
    }

    #[test]
    fn try_reserve_reports_an_exhausted_budget_intact() {
        let alloc = BudgetedAlloc::new(8);
        let mut heap = budgeted_heap(&[9, 5, 1], alloc.clone());
        alloc.drain();
        let error = heap.try_reserve(1024).expect_err("the budget is spent");
        assert!(error.is_alloc(), "{error:?}");
        assert_eq!(
            heap.len(),
            3,
            "a refused reservation must not do anything else"
        );
        assert_eq!(drain_i32_budgeted(&mut heap), vec![9, 5, 1]);
    }

    #[test]
    fn capacity_overflow_is_reported_as_such_by_every_reserve_flavor() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        let error = heap.try_reserve(usize::MAX).expect_err("overflow");
        assert!(error.is_capacity_overflow(), "{error:?}");
        let error = heap.try_reserve_exact(usize::MAX).expect_err("overflow");
        assert!(error.is_capacity_overflow(), "{error:?}");
        let error = heap.try_reserve_adaptive(usize::MAX).expect_err("overflow");
        assert!(error.is_capacity_overflow(), "{error:?}");
        assert!(heap.is_empty());
    }

    #[test]
    fn try_reserve_exact_grows_to_exactly_the_need() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        for value in [1, 2, 3, 4, 5] {
            heap.try_push(value).expect("infallible on this budget");
        }
        // One slot beyond the current capacity forces the exact path to
        // reallocate; unlike the amortized path it lands on the need.
        let need = heap.capacity() - heap.len() + 1;
        heap.try_reserve_exact(need)
            .expect("global allocator has room");
        assert_eq!(heap.capacity(), heap.len() + need);
        assert_eq!(drain_i32(&mut heap), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn try_reserve_adaptive_satisfies_the_request() {
        let mut heap = heap_of([4, 2]);
        heap.try_reserve_adaptive(32)
            .expect("global allocator has room");
        assert!(
            heap.capacity() >= heap.len() + 32,
            "the request must be met"
        );
        assert_eq!(drain_i32(&mut heap), vec![4, 2]);
    }

    #[test]
    fn try_shrink_to_fit_leaves_just_enough_room() {
        let mut heap = heap_of([9, 8, 7, 6, 5, 4, 3, 2, 1]);
        heap.try_reserve(32).expect("global allocator has room");
        assert!(heap.capacity() > 9, "the fixture must carry slack");
        heap.try_shrink_to_fit()
            .expect("realloc to the live length");
        assert_eq!(heap.capacity(), heap.len());
        assert_eq!(
            drain_i32(&mut heap),
            vec![9, 8, 7, 6, 5, 4, 3, 2, 1],
            "shrinking must preserve every element"
        );
    }

    #[test]
    fn try_shrink_to_honors_the_bound_and_treats_small_bounds_as_the_length() {
        let mut heap = heap_of([9, 7, 5, 3, 1]);
        heap.try_reserve(32).expect("global allocator has room");
        // A bound far above the capacity is a no-op.
        heap.try_shrink_to(usize::MAX / 2)
            .expect("no work is required");
        // A real bound trims to max(bound, len).
        heap.try_shrink_to(8).expect("realloc to the bound");
        assert_eq!(heap.capacity(), 8);
        // A bound below the length is treated as the length.
        heap.try_shrink_to(0).expect("the floor is the live length");
        assert_eq!(heap.capacity(), heap.len());
        assert_eq!(drain_i32(&mut heap), vec![9, 7, 5, 3, 1]);
    }
}
