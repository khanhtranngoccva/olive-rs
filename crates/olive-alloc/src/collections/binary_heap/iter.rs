//! Iterator types for the [`BinaryHeap`].

use super::BinaryHeap;
use crate::alloc::{Allocator, Global};
use crate::vec::IntoIter as VecIntoIter;
use core::fmt::{self, Debug};
use core::iter::FusedIterator;
use core::slice;

/// A borrowing iterator over the elements of a [`BinaryHeap`](super::BinaryHeap),
/// yielding each exactly once.
///
/// This `struct` is created by the [`iter`] method on
/// [`BinaryHeap`](super::BinaryHeap).
///
/// The iterator is a *view* of the heap's buffer, not a sorted walk,
/// elements come out in arbitrary order.
///
/// [`iter`]: BinaryHeap::iter
pub struct Iter<'a, T> {
    inner: slice::Iter<'a, T>,
}

impl<'a, T> Iter<'a, T> {
    /// Wraps a slice iterator over the heap's buffer.
    #[inline]
    pub(super) const fn new(inner: slice::Iter<'a, T>) -> Self {
        Self { inner }
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Views the remaining, not-yet-yielded elements as a slice, in
    /// raw buffer order.
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        self.inner.as_slice()
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    /// Yields the next remaining element, in buffer order.
    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.inner.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    /// Yields the next remaining element in the backwards direction,
    /// in buffer order.
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        self.inner.next_back()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T: Debug> Debug for Iter<'_, T> {
    /// Lists the elements not yet yielded, in buffer order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.clone()).finish()
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Default for Iter<'_, T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

/// An iterator that moves out of a [`BinaryHeap`](super::BinaryHeap) by
/// value, yielding every element exactly once.
///
/// This `struct` is created by the [`into_iter`] method on
/// [`BinaryHeap`](super::BinaryHeap) (provided by [`IntoIterator`]).
///
/// [`into_iter`]: IntoIterator::into_iter
pub struct IntoIter<T, A: Allocator = Global> {
    inner: VecIntoIter<T, A>,
}

impl<T, A: Allocator> IntoIter<T, A> {
    /// Wraps the vector's owned iterator; the heap itself is gone by
    /// then, its quarantine flag discarded along with the wrapper it
    /// lived in.
    #[inline]
    pub(super) const fn new(inner: VecIntoIter<T, A>) -> Self {
        Self { inner }
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Views the remaining, not-yet-yielded elements as a slice, in
    /// raw buffer order.
    ///
    /// Borrowing the window reads no order and runs no [`Ord`]. Elements
    /// are in arbitrary layer.
    ///
    /// [`Ord`]: core::cmp::Ord
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Borrows the remaining elements as an iterator, in raw buffer order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self.inner.as_slice().iter())
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    /// Yields the next remaining element, in buffer order.
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.inner.last()
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    /// Yields the next remaining element in the backwards direction,
    /// in buffer order.
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    /// Returns the number of elements still held by this iterator.
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T: Debug, A: Allocator> Debug for IntoIter<T, A> {
    /// Lists the elements not yet yielded, in buffer order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Yields every element exactly once, in arbitrary order.
    ///
    /// The iterator is a view of the heap's buffer: elements come out in
    /// the raw array order, not sorted.
    ///
    /// Iteration is permitted even on a poisoned heap.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self.data.iter())
    }
}

impl<T, A: Allocator> IntoIterator for BinaryHeap<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    /// Consumes the heap, returning an [`IntoIter`] over its elements in
    /// an arbitrary order.
    ///
    /// Iteration is permitted even on a quarantined heap, for it only moves
    /// elements out and the iterator does not guarantee any ordering.
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.data.into_iter())
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a BinaryHeap<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    /// Borrows the heap, returning an [`Iter`] over its element references in an
    /// arbitrary order.
    ///
    /// Iteration is permitted even on a quarantined heap, for the iterator
    /// does not guarantee any ordering.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::IntoIter;
    use crate::collections::binary_heap::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use olive_test_helpers::{PanicArmer, PanickyOrd};
    use std::sync::Arc;
    use std::{format, panic, vec, vec::Vec};

    fn iter_of(items: impl IntoIterator<Item = i32>) -> IntoIter<i32> {
        let heap = BinaryHeap::try_from_iter(items).expect("allocation ok");
        heap.into_iter()
    }

    #[test]
    fn into_iterator_yields_every_element_once() {
        let mut original = vec![4, 8, 15, 16, 23, 42];
        let heap = BinaryHeap::try_from_iter(original.iter().copied()).expect("allocation ok");
        let mut drained: Vec<i32> = heap.into_iter().collect();
        drained.sort_unstable();
        original.sort_unstable();
        assert_eq!(drained, original);
    }

    #[test]
    fn double_ended_drain_meets_in_the_middle() {
        let it = iter_of([3, 1, 4, 1, 8]);
        let mut taken = Vec::new();
        let mut from_front = true;
        let mut it = it;
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
    }

    #[test]
    fn next_back_walks_the_buffer_backwards() {
        let mut it = iter_of([9, 7, 5, 1, 12]);
        let mut expected: Vec<i32> = it.as_slice().to_vec();
        expected.reverse();
        let mut reversed = Vec::new();
        while let Some(item) = it.next_back() {
            reversed.push(item);
        }
        assert_eq!(
            reversed, expected,
            "back drain is the exact reverse of the window"
        );
    }

    #[test]
    fn len_tracks_the_live_window() {
        let mut it = iter_of([2, 4, 6, 8]);
        assert_eq!(it.len(), 4);
        assert!(!it.is_empty());
        it.next();
        it.next_back();
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn as_slice_and_iter_view_the_same_window() {
        let mut it = iter_of([5, 6, 7, 8, 9]);
        assert_eq!(it.as_slice().len(), it.len());
        assert_eq!(it.iter().len(), it.len());
        it.next();
        assert_eq!(
            it.iter().copied().collect::<Vec<_>>(),
            it.as_slice().to_vec()
        );
    }

    #[test]
    fn debug_lists_the_remaining_elements() {
        let mut it = iter_of([42, 1, 1, 1]);
        it.next();
        let text = format!("{it:?}");
        assert!(text.contains('1'));
        // Largest element is removed in the next() call.
        assert!(!text.contains("42"));
    }

    // A heap whose comparisons panic once `armer` is armed, so the test
    // can poison it on purpose through the guarded append.
    fn panicky_heap_of(
        items: impl IntoIterator<Item = i32>,
        armer: &Arc<PanicArmer>,
    ) -> BinaryHeap<PanickyOrd<i32>> {
        BinaryHeap::try_from_iter(
            items
                .into_iter()
                .map(|value| PanickyOrd::new(value, Arc::clone(armer))),
        )
        .expect("allocation ok")
    }

    #[test]
    fn into_iterator_salvages_a_poisoned_heap() {
        // The quarantine must never cost data: draining by value is the
        // documented salvage path, and it reads no order.
        let armer = Arc::new(PanicArmer::new());
        let mut heap = panicky_heap_of([1, 4, 3], &armer);
        let mut other: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::new(9, Arc::clone(&armer))])
                .expect("allocation ok");
        armer.arm();
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| heap.try_append(&mut other)));
        assert!(heap.is_poisoned());
        let count = heap.into_iter().count();
        assert_eq!(count, 4, "draining a quarantined heap yields everything");
    }

    // --- borrowing Iter ------------------------------------------------------

    #[test]
    fn iter_yields_every_element_exactly_once() {
        let heap = BinaryHeap::try_from_iter([4, 8, 15, 16]).expect("allocation ok");
        let mut seen: Vec<i32> = heap.iter().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![4, 8, 15, 16]);
    }

    #[test]
    fn iter_len_and_window_track_consumption() {
        let heap = BinaryHeap::try_from_iter([2, 4, 6, 8]).expect("allocation ok");
        let mut it = heap.iter();
        let window = it.as_slice().to_vec();
        assert_eq!(it.len(), 4);
        assert!(!it.is_empty());
        let front = *it.next().expect("non-empty");
        let back = *it.next_back().expect("non-empty");
        assert_eq!(front, window[0], "the front walk starts at the root slot");
        assert_eq!(back, window[window.len() - 1]);
        assert_eq!(it.as_slice(), &window[1..3]);
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn iter_works_on_a_poisoned_heap() {
        // Like `into_iter`, borrowing iteration must survive the
        // quarantine: it moves nothing and compares nothing.
        let armer = Arc::new(PanicArmer::new());
        let mut heap = panicky_heap_of([1, 4, 3], &armer);
        let mut other: BinaryHeap<PanickyOrd<i32>> =
            BinaryHeap::try_from_iter([PanickyOrd::calm(9)]).expect("allocation ok");
        armer.arm();
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| heap.try_append(&mut other)));
        assert!(heap.is_poisoned());
        let mut values: Vec<i32> = heap.iter().map(|item| item.value).collect();
        values.sort_unstable();
        assert_eq!(
            values,
            vec![1, 3, 4, 9],
            "borrowed iteration yields everything"
        );
    }

    #[test]
    fn iter_debug_lists_the_remaining_elements() {
        let heap = BinaryHeap::try_from_iter([42, 1, 1, 1]).expect("allocation ok");
        let mut it = heap.iter();
        it.next();
        let text = format!("{it:?}");
        assert!(text.contains('1'));
        assert!(!text.contains("42"));
    }
}
