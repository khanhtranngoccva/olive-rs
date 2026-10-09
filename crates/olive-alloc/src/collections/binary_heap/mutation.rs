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

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::test_helpers::{PanicArmer, PanickyOrd};

    use super::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use std::panic;
    use std::sync::Arc;
    use std::vec;

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
        use crate::test_helpers::BudgetedAlloc;
        use core::sync::atomic::Ordering;

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
        assert_eq!(
            alloc.remaining.load(Ordering::SeqCst),
            7,
            "refill must not allocate"
        );
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
}
