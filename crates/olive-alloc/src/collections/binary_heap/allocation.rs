//! Growth operations: pushing elements and merging iterators or heaps.

use core::ops::{Deref, DerefMut};

use super::BinaryHeap;
use super::poison::PoisonOnUnwind;
use crate::alloc::Allocator;
use olive_core::alloc_errors::TryReserveError;

impl<T: Ord, A: Allocator> BinaryHeap<T, A> {
    /// Attempts to insert an item into the binary heap, returning an error on
    /// failure. The element is dropped on error or panic.
    ///
    /// # Panics
    ///
    /// If the `Ord` comparison panics during the insertion, the buffer is
    /// truncated back over to restore the original state, and the element is
    /// lost.
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
    /// If the `Ord` comparison panics during the insertion, the buffer is
    /// truncated back over to restore the original state, and the element is
    /// lost.
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

        let mut guard = TruncateOnUnwind::new(self, origin);
        // SAFETY: `origin` is the index the element was just written to, so
        // `origin < self.data.len()`.
        unsafe { guard.sift_up(0, origin) };
        // Disable the guard.
        core::mem::forget(guard);

        Ok(())
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Errors
    ///
    /// Returns the [`TryReserveError`] if space for `other.len()` elements
    /// could not be reserved. In this case, both heaps are intact.
    ///
    /// # Panics
    ///
    /// Panics if either heap is poisoned or if the [`Ord`] implementation panics.
    /// However, panicking does not break invariants.
    ///
    /// # Atomicity and performance
    ///
    /// This method does not have full atomicity - it is guaranteed on errors, but not
    /// on panics. However, this method does not poison either heap, for the method uses
    /// the *slow* algorithm that is resilient to poisoning on panics.
    ///
    /// To improve performance, you may use [`try_append_fast`](Self::try_append_fast),
    /// which partially sacrifices panic safety (but may poison the heap).
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
        self.data.try_reserve_adaptive(other.len())?;
        while let Some(item) = other.pop() {
            self.try_push(item)
                .expect("capacity should already be reserved");
        }
        Ok(())
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty. This is
    /// a faster implementation (taken from std) that sacrifices panic safety
    /// (may leave the heap poisoned).
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
    /// [`rebuild`](Self::rebuild) must be called.
    pub fn try_append_fast(&mut self, other: &mut Self) -> Result<(), TryReserveError> {
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
        quarantine.defuse();
        Ok(())
    }
}

/// Guard that truncates a vector back to a recorded length if the region it
/// guards unwinds.
///
/// The guard is trivial and can be defused with `mem::forget`.
struct TruncateOnUnwind<'a, T, A: Allocator> {
    heap: &'a mut BinaryHeap<T, A>,
    keep: usize,
}

impl<'a, T, A: Allocator> TruncateOnUnwind<'a, T, A> {
    #[inline]
    const fn new(heap: &'a mut BinaryHeap<T, A>, keep: usize) -> Self {
        Self { heap, keep }
    }
}

impl<T, A: Allocator> Deref for TruncateOnUnwind<'_, T, A> {
    type Target = BinaryHeap<T, A>;

    fn deref(&self) -> &Self::Target {
        self.heap
    }
}

impl<T, A: Allocator> DerefMut for TruncateOnUnwind<'_, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.heap
    }
}

impl<T, A: Allocator> Drop for TruncateOnUnwind<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        self.heap.data.truncate(self.keep);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::BinaryHeap;
    use crate::test_helpers::BudgetedAlloc;
    use core::sync::atomic::Ordering;
    use olive_core::{
        alloc_errors::TryReserveError, try_traits::try_from_iterator::TryFromIterator,
    };
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
            alloc.remaining.load(Ordering::SeqCst),
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
            alloc.remaining.load(Ordering::SeqCst),
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
        assert_eq!(alloc.remaining.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn try_append_merges_two_heaps() {
        // Slow variant
        let mut left = heap_of(vec![9, 5, 1]);
        let mut right = heap_of(vec![8, 7, 6, 2]);
        left.try_append(&mut right).expect("reserve succeeds");
        assert!(right.is_empty());
        assert_eq!(drain_i32(&mut left), vec![9, 8, 7, 6, 5, 2, 1]);

        // Fast variant
        let mut left = heap_of(vec![9, 5, 1]);
        let mut right = heap_of(vec![8, 7, 6, 2]);
        left.try_append_fast(&mut right).expect("reserve succeeds");
        assert!(right.is_empty());
        assert_eq!(drain_i32(&mut left), vec![9, 8, 7, 6, 5, 2, 1]);
    }

    #[test]
    fn try_append_failure_leaves_both_heaps_intact() {
        // Slow variant
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

        // Fast variant
        let alloc = BudgetedAlloc::new(8);
        let mut left = budgeted_heap(&[9, 8, 7, 1], alloc.clone());
        let mut right = budgeted_heap(&[6, 5], alloc.clone());
        alloc.drain();

        let error: TryReserveError = left
            .try_append_fast(&mut right)
            .expect_err("an exhausted budget must refuse the merge");
        assert!(error.is_alloc(), "{error:?}");
        assert_eq!(left.len(), 4, "the left heap must not have moved");
        assert_eq!(right.len(), 2, "the right heap must have kept its data");
        assert_eq!(drain_i32_budgeted(&mut left), vec![9, 8, 7, 1]);
        assert_eq!(drain_i32_budgeted(&mut right), vec![6, 5]);
    }

    #[test]
    fn try_append_of_empty_is_free() {
        // Slow variant
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
        assert_eq!(alloc.remaining.load(Ordering::SeqCst), 0);

        // Fast variant
        let alloc = BudgetedAlloc::new(8);
        let mut left = budgeted_heap(&[3, 1, 2], alloc.clone());
        let mut right: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc.clone());
        alloc.drain();
        left.try_append_fast(&mut right)
            .expect("empty appends are trivial");
        right
            .try_append_fast(&mut left)
            .expect("empty appends are trivial");
        assert_eq!(drain_i32_budgeted(&mut right), vec![3, 2, 1]);
        assert_eq!(alloc.remaining.load(Ordering::SeqCst), 0);
    }
}
