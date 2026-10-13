//! Inspecting the maximum and removing it from the front.
//!
//! This is the read-and-shrink half of the [`BinaryHeap`] API, the
//! counterpart of the growth operations in [`allocation`](super::allocation).
//! [`BinaryHeap::pop`] is the only operation here that moves elements: it
//! swaps the root with the last slot, truncates, and re-sifts the root down,
//! which turns the removal into exactly one sifting pass.
//!
//! The re-sift is atomic ([`sift_down`](super::sifting::BinaryHeap::sift_down)
//! computes its plan before touching storage) and `pop` wraps it in its own
//! rollback: if a comparison panics, the extracted element is pushed back
//! and the initial swap is undone, so the heap is restored bit-for-bit
//! before the panic continues unwinding. A heap that is *already*
//! poisoned — quarantined by a panicking rebuild; see
//! [`poison`](super::poison) — refuses to pop: its children are not valid
//! heaps, so the sifting would silently produce wrong orderings.

use core::ops::{Deref, DerefMut};

use super::BinaryHeap;
use super::poison::PoisonOnUnwind;
use crate::alloc::Allocator;
use crate::vec::Vec;

impl<T: Ord, A: Allocator> BinaryHeap<T, A> {
    /// Removes the biggest item from the binary heap and returns it, or
    /// `None` if it is empty.
    ///
    /// # Panics
    ///
    /// Panics if the `Ord` comparison panics while sifting the new root
    /// down. It will not poison the heap.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.assume_unpoisoned();
        if self.is_empty() {
            return None;
        }
        // Perform a swap and remove - the last item is moved to the first position and
        // ready to be sifted down.
        let item = self
            .data
            .try_swap_remove(0)
            .expect("length checked just above");
        if self.data.is_empty() {
            return Some(item);
        }
        let mut guard = PopRollback::new(self, item);
        // SAFETY: `0 < self.data.len()` because the empty case returned
        // just above.
        unsafe { guard.sift_down(0) };
        let item = guard.defuse();
        Some(item)
    }

    /// Clears the binary heap, removing all elements. The allocated capacity
    /// is retained and the poison flag, if any, is lifted.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.poisoned = false;
    }

    /// Consumes the binary heap, returning the underlying vector.
    ///
    /// The elements are returned in their internal level-order arrangement
    /// and are *not* sorted. A poisoned heap can be converted into the
    /// vector.
    #[must_use]
    #[inline]
    pub fn into_vec(self) -> Vec<T, A> {
        self.data
    }

    /// Retains only the elements the predicate accepts.
    ///
    /// In other words, removes all elements `e` for which `f(&e)` returns
    /// `false`, dropping the displaced values.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned. If an `Ord` comparison panics
    /// during the closing rebuild, the heap is poisoned with every
    /// surviving element still present; [`rebuild`](BinaryHeap::rebuild)
    /// restores the invariant.
    ///
    /// A panicking *predicate* does not poison the heap, for the tail
    /// behind the first element will always be rebuilt.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.assume_unpoisoned();
        let mut guard = RebuildTailOnDrop::new(self, self.len());
        let mut i = 0usize;
        // The closure may only narrow the window to the first removed
        // slot; the prefix before it was never touched and is still a
        // proper heap.
        guard.heap.data.retain(|e| {
            let keep = f(e);
            if !keep && i < guard.start {
                guard.start = i;
            }
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "one step per element of a finite slice"
            )]
            {
                i += 1;
            }
            keep
        });
    }

    /// Removes and returns the largest item, but only if the predicate
    /// accepts it.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned, or if an `Ord` comparison panics
    /// while popping — [`pop`](BinaryHeap::pop)'s rollback restores the
    /// heap without triggering the poison flag.
    pub fn pop_if<F>(&mut self, mut predicate: F) -> Option<T>
    where
        F: FnMut(&T) -> bool,
    {
        let first = self.peek()?;
        if predicate(first) { self.pop() } else { None }
    }
}

/// Guard that reverses a `pop`'s swap-and-truncate if the sift it wraps
/// unwinds.
struct PopRollback<'a, T, A: Allocator> {
    heap: &'a mut BinaryHeap<T, A>,
    item: Option<T>,
}

impl<T, A: Allocator> Deref for PopRollback<'_, T, A> {
    type Target = BinaryHeap<T, A>;

    fn deref(&self) -> &Self::Target {
        self.heap
    }
}

impl<T, A: Allocator> DerefMut for PopRollback<'_, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.heap
    }
}

impl<'a, T, A: Allocator> PopRollback<'a, T, A> {
    #[inline]
    const fn new(data: &'a mut BinaryHeap<T, A>, item: T) -> Self {
        Self {
            heap: data,
            item: Some(item),
        }
    }

    /// The sift completed: take the element back out for `pop` to return.
    #[inline]
    fn defuse(&mut self) -> T {
        self.item
            .take()
            .expect("a fresh guard always holds the element")
    }
}

impl<T, A: Allocator> Drop for PopRollback<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let last_index = self.heap.len();
            self.heap
                .data
                .try_push(item)
                .expect("the slot the pop freed is still reserved");
            self.heap
                .data
                .try_swap(0, last_index)
                .expect("swapped offsets should be in bounds");
        }
    }
}

/// Critical section shared by [`retain`](BinaryHeap::retain) and the
/// [`try_extend`](olive_core::try_traits::TryExtend::try_extend) family:
/// it owns the heap for as long as `data[start..]` is
/// known not to be a proper heap, and its `Drop` closes the window by
/// rebuilding the tail.
pub(super) struct RebuildTailOnDrop<'a, T: Ord, A: Allocator> {
    /// First index whose heap property is no longer guaranteed.
    start: usize,
    heap: &'a mut BinaryHeap<T, A>,
}

impl<'a, T: Ord, A: Allocator> RebuildTailOnDrop<'a, T, A> {
    /// Opens a rebuild window over `data[start..]` for the lifetime of the
    /// returned guard. `start` must not exceed the heap's length.
    pub(super) fn new(heap: &'a mut BinaryHeap<T, A>, start: usize) -> Self {
        Self { start, heap }
    }
}

impl<T: Ord, A: Allocator> Deref for RebuildTailOnDrop<'_, T, A> {
    type Target = BinaryHeap<T, A>;

    fn deref(&self) -> &Self::Target {
        self.heap
    }
}

impl<T: Ord, A: Allocator> DerefMut for RebuildTailOnDrop<'_, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.heap
    }
}

impl<T: Ord, A: Allocator> Drop for RebuildTailOnDrop<'_, T, A> {
    fn drop(&mut self) {
        // The window may only cover a tail of the current view. The caller may then lower
        // `start` or increase the number of actual elements, as long as this invariant still holds.
        debug_assert!(self.start <= self.heap.data.len());
        let mut quarantine = PoisonOnUnwind::new(&mut *self.heap);
        quarantine.rebuild_tail(self.start);
        core::mem::forget(quarantine);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use olive_test_helpers::{PanicArmer, PanickyOrd};

    use super::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use std::panic;
    use std::sync::Arc;
    use std::vec;
    use std::vec::Vec;

    fn heap_of<I: IntoIterator<Item = i32>>(items: I) -> BinaryHeap<i32> {
        BinaryHeap::try_from_iter(items).expect("allocation ok")
    }

    #[test]
    fn pop_keep_max_on_top() {
        let mut heap = heap_of(vec![1, 5, 3, 9, 2]);
        assert_eq!(heap.pop(), Some(9));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn pop_on_empty_returns_none_and_stays_empty() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        assert_eq!(heap.pop(), None);
        assert!(heap.is_empty());
    }

    #[test]
    fn single_element_roundtrip() {
        let mut heap = heap_of(vec![42]);
        assert_eq!(heap.pop(), Some(42));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn drain_yields_elements_in_non_increasing_order() {
        let data = [5, 100, 23, 0, 100, 1, 23, 100];
        let mut heap = heap_of(data.iter().copied());
        let mut last = i32::MAX;
        while let Some(item) = heap.pop() {
            assert!(item <= last, "pop yielded {item} after {last}");
            last = item;
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn into_vec_test() {
        let heap = heap_of([7, 7, 1, 4]);
        let mut storage = heap.into_vec();
        storage.sort_unstable();
        assert_eq!(storage.as_slice(), [1, 4, 7, 7]);
    }

    #[test]
    fn clear_keeps_capacity_and_repush_does_not_allocate() {
        use olive_test_helpers::BudgetedAlloc;

        let alloc = BudgetedAlloc::new(8);
        let mut heap: BinaryHeap<i32, _> =
            BinaryHeap::try_with_capacity_in(8, alloc.clone()).expect("budgeted capacity");
        heap.try_push(1).expect("budgeted push");
        heap.try_push(2).expect("budgeted push");
        heap.clear();
        // `clear` retains the allocation, so refilling stays within budget
        // without the allocator being asked for anything.
        for value in 0..8 {
            heap.try_push(value).expect("reused capacity");
        }
        assert_eq!(alloc.remaining(), 7, "refill must not allocate");
        assert_eq!(heap.pop(), Some(7));
    }

    #[test]
    fn clear_lifts_the_quarantine() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::new(1, Arc::clone(&armer))])
                .expect("allocation ok");
        let mut other: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::new(2, Arc::clone(&armer))])
                .expect("allocation ok");
        armer.arm();
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| heap.try_append(&mut other)));
        assert!(heap.is_poisoned());
        heap.clear();
        assert!(!heap.is_poisoned());
        assert!(heap.pop().is_none());
    }

    // --- retain --------------------------------------------------------------

    #[test]
    fn retain_keeps_exactly_the_matching_multiset() {
        let mut heap = heap_of([9, 5, 1, 7, 3, 8, 2, 4]);
        heap.retain(|value| *value >= 5);
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, std::vec![9, 8, 7, 5]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn retain_never_and_always_stay_valid() {
        let mut heap = heap_of([3, 1, 2]);
        heap.retain(|_| true);
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, std::vec![3, 2, 1]);

        let mut heap = heap_of([3, 1, 2]);
        heap.retain(|_| false);
        assert!(heap.is_empty());
        assert!(!heap.is_poisoned());
        // The emptied heap stays usable.
        heap.try_push(7).expect("the heap is usable after emptying");
        assert_eq!(heap.peek(), Some(&7));
    }

    #[test]
    fn retain_poisons_when_the_closing_rebuild_detonates() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap = BinaryHeap::try_from_iter(
            [7, 6, 5, 4, 3, 2, 1]
                .iter()
                .map(|value| PanickyOrd::new(*value, Arc::clone(&armer))),
        )
        .expect("allocation ok");

        // Removing the root forces the closing rebuild to compare.
        armer.arm();
        let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            heap.retain(|&PanickyOrd { value, .. }| value != 7);
        }));
        assert!(outcome.is_err(), "the armed rebuild must detonate");
        assert!(heap.is_poisoned(), "the window quarantines the heap");
        assert_eq!(heap.len(), 6, "survivors are honored, not discarded");

        armer.disarm();
        heap.rebuild();
        assert!(!heap.is_poisoned());
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item.value);
        }
        assert_eq!(out, std::vec![6, 5, 4, 3, 2, 1]);
    }

    // --- pop_if --------------------------------------------------------------

    #[test]
    fn pop_if_takes_the_head_when_the_predicate_agrees() {
        let mut heap = heap_of([9, 5, 7]);
        assert_eq!(heap.pop_if(|value| *value > 8), Some(9));
        assert_eq!(heap.peek(), Some(&7));
        let mut out = Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, std::vec![7, 5]);
    }

    #[test]
    fn pop_if_leaves_the_heap_untouched_when_the_predicate_does_not_match() {
        let mut heap = heap_of([9, 5, 7]);
        assert_eq!(heap.pop_if(|value| *value > 100), None);
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some(&9));
    }

    #[test]
    fn pop_if_on_an_empty_heap_never_asks_the_predicate() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        let mut calls = 0usize;
        assert_eq!(
            heap.pop_if(|_| {
                calls += 1;
                true
            }),
            None
        );
        assert_eq!(calls, 0);
    }
}
