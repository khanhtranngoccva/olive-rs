//! The borrowing drain iterator produced by [`BinaryHeap::drain`].

use super::BinaryHeap;
use crate::alloc::{Allocator, Global};
use crate::vec::Drain as VecDrain;
use core::fmt::{self, Debug};
use core::iter::FusedIterator;

/// A borrowing iterator that removes every element from a
/// [`BinaryHeap`](super::BinaryHeap), yielding each exactly once.
///
/// This `struct` is created by the [`drain`] method on
/// [`BinaryHeap`](super::BinaryHeap).
///
/// Draining empties the heap while keeping its allocated capacity, and the
/// elements come out in the heap's internal buffer order — which is *not*
/// sorted. Use [`pop`](BinaryHeap::pop) to walk them from largest to
/// smallest.
///
/// [`drain`]: BinaryHeap::drain
pub struct Drain<'a, T, A: Allocator = Global> {
    inner: VecDrain<'a, T, A>,
}

impl<'a, T, A: Allocator> Drain<'a, T, A> {
    /// Wraps the underlying vector drainer, which covers the heap's whole
    /// buffer.
    #[inline]
    pub(super) const fn new(inner: VecDrain<'a, T, A>) -> Self {
        Self { inner }
    }

    /// Returns the number of elements still held by this iterator.
    #[inline]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Views the remaining, not-yet-yielded elements as a slice, in raw
    /// buffer order.
    ///
    /// Borrowing the window reads no order and runs no [`Ord`]. Elements are
    /// in arbitrary layer.
    ///
    /// [`Ord`]: core::cmp::Ord
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    /// Yields the next remaining element, in buffer order.
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<T>
    where
        Self: Sized,
    {
        self.inner.next_back()
    }
}

impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    /// Yields the next remaining element in the backwards direction, in
    /// buffer order.
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {
    /// Returns the number of elements still held by this iterator.
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}

impl<T: Debug, A: Allocator> Debug for Drain<'_, T, A> {
    /// Lists the elements not yet yielded, in buffer order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.as_slice()).finish()
    }
}

impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Removes every element from the heap, yielding them by value in
    /// arbitrary order.
    ///
    /// This is the yielding counterpart of [`clear`](BinaryHeap::clear): the
    /// heap is left empty with its allocated capacity retained, and any
    /// quarantine flag is lifted. Like `clear`, draining moves elements out
    /// without reading order, so it is permitted on a poisoned heap.
    ///
    /// The elements are yielded in the heap's internal buffer order, which is
    /// *not* sorted. To visit them from largest to smallest, [`pop`] them
    /// instead.
    ///
    /// If the returned iterator is dropped before being fully consumed, the
    /// remaining elements are dropped with it; the heap is still left empty.
    ///
    /// [`pop`]: BinaryHeap::pop
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T, A> {
        // Emptying the heap leaves it in a valid (empty) state, so a previous
        // quarantine is lifted exactly as `clear` does. This must happen before
        // the buffer is borrowed, while `self` is still exclusively ours.
        self.poisoned = false;
        let inner = self
            .data
            .try_drain(..)
            .expect("the full range `..` is always in bounds");
        Drain::new(inner)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::collections::binary_heap::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use olive_test_helpers::PanicArmer;
    use olive_test_helpers::PanickyOrd;
    use std::sync::Arc;
    use std::{format, panic, vec, vec::Vec};

    fn heap_of<I: IntoIterator<Item = i32>>(items: I) -> BinaryHeap<i32> {
        BinaryHeap::try_from_iter(items).expect("allocation ok")
    }

    #[test]
    fn drain_yields_every_element_once_and_empties_the_heap() {
        let mut heap = heap_of([4, 8, 15, 16, 23, 42]);
        let mut drained: Vec<i32> = heap.drain().collect();
        drained.sort_unstable();
        assert_eq!(drained, vec![4, 8, 15, 16, 23, 42]);
        assert!(heap.is_empty());
        assert!(!heap.is_poisoned(), "a drained heap is unpoisoned");
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn drain_on_empty_heap_is_empty() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        let drain = heap.drain();
        assert!(drain.is_empty());
        assert_eq!(drain.len(), 0);
        assert_eq!(drain.count(), 0);
    }

    #[test]
    fn drain_keeps_capacity() {
        let mut heap = heap_of([1, 2, 3, 4]);
        let capacity = heap.capacity();
        let taken: usize = heap.drain().count();
        assert_eq!(taken, 4);
        assert_eq!(heap.capacity(), capacity, "drain must not shrink");
        // The emptied-but-capable heap stays usable.
        heap.try_push(7).expect("the heap is usable after draining");
        assert_eq!(heap.peek(), Some(&7));
    }

    #[test]
    fn drain_is_double_ended() {
        let mut heap = heap_of([3, 1, 4, 1, 8]);
        let mut it = heap.drain();
        let mut taken = Vec::new();
        let mut from_front = true;
        loop {
            let item = if from_front {
                it.next()
            } else {
                it.next_back()
            };
            from_front = !from_front;
            match item {
                Some(item) => taken.push(item),
                None => break,
            }
        }
        taken.sort_unstable();
        assert_eq!(taken, vec![1, 1, 3, 4, 8]);
        assert!(it.is_empty());
        // Release the exclusive borrow before inspecting the heap again: the
        // drainer holds the buffer until it is dropped.
        drop(it);
        assert!(heap.is_empty());
    }

    #[test]
    fn drain_len_and_window_track_consumption() {
        let mut heap = heap_of([2, 4, 6, 8]);
        let mut it = heap.drain();
        assert_eq!(it.len(), 4);
        assert!(!it.is_empty());
        let window = it.as_slice().to_vec();
        let front = it.next().expect("non-empty");
        let back = it.next_back().expect("non-empty");
        assert_eq!(front, window[0], "the front walk starts at the root slot");
        assert_eq!(back, window[window.len() - 1]);
        assert_eq!(it.as_slice(), &window[1..window.len() - 1]);
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn drain_last_returns_the_window_tail() {
        let mut heap = heap_of([1, 7, 3]);
        let expected = heap.iter().next_back().copied().expect("non-empty");
        assert_eq!(heap.drain().last(), Some(expected));
        assert!(heap.is_empty());
    }

    #[test]
    fn drop_early_leaves_the_heap_empty() {
        // A drain dropped before exhaustion must still empty the heap and
        // never leak the elements it never yielded.
        let mut heap = heap_of([5, 4, 3, 2, 1]);
        {
            let mut it = heap.drain();
            let _ = it.next();
            assert_eq!(it.len(), 4, "four elements are left when it is dropped");
        } // `it` dropped here with four elements left.
        assert!(heap.is_empty());
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn drain_lifts_the_quarantine() {
        // Like `clear`, draining a poisoned heap moves every element out and
        // leaves a fresh, unpoisoned, empty heap behind.
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::new(1, Arc::clone(&armer))])
                .expect("allocation ok");
        let mut other: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::new(2, Arc::clone(&armer))])
                .expect("allocation ok");
        armer.arm();
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| heap.try_append(&mut other)));
        assert!(heap.is_poisoned(), "the append poisoned the heap");

        let count = heap.drain().count();
        assert_eq!(count, 2, "draining a quarantined heap yields everything");
        assert!(!heap.is_poisoned(), "draining lifts the quarantine");
        assert!(heap.is_empty());
    }

    #[test]
    fn debug_lists_the_remaining_elements() {
        let mut heap = heap_of([42, 1, 1, 1]);
        let mut it = heap.drain();
        let first = it.next().expect("non-empty");
        let text = format!("{it:?}");
        // The element we pulled is gone from the remaining window.
        assert!(!text.contains("42"));
        assert!(format!("{first:?}").contains("42"));
    }
}
