//! The owned iterator produced by `VecDeque::into_iter`.
use super::VecDeque;
use crate::alloc::{Allocator, Global};
use core::iter::FusedIterator;

/// An iterator that moves out of a `VecDeque`.
///
/// This struct is created by the `into_iter` method on
/// [`VecDeque`] (provided by the [`IntoIterator`] trait).
///
/// # Example
///
/// ```
/// use olive_alloc::collections::vec_deque::VecDeque;
/// let mut dq = VecDeque::new();
/// for i in 0..3 { dq.try_push_back(i).unwrap(); }
/// let collected: std::vec::Vec<i32> = dq.into_iter().collect();
/// assert_eq!(collected, std::vec![0, 1, 2]);
/// ```
pub struct IntoIter<T, A: Allocator = Global> {
    /// The consumed-by-value deque. Each `next` pops from its front; whatever
    /// remains when the iterator is dropped is destroyed by the deque's own
    /// `Drop` impl.
    deque: VecDeque<T, A>,
}

// SAFETY: mirroring `VecDeque`, moving the iterator moves the whole allocation.
// Sound iff `T` and `A` are `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for IntoIter<T, A> {}

impl<T, A: Allocator> IntoIter<T, A> {
    pub(super) fn new(deque: VecDeque<T, A>) -> Self {
        Self { deque }
    }

    /// Consumes this iterator and returns the underlying [`VecDeque`], still
    /// holding any elements that have not yet been yielded.
    pub fn into_vecdeque(self) -> VecDeque<T, A> {
        self.deque
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.deque.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    // Consuming the iterator just discards the deque; its length is exactly
    // the number of elements still held, so no popping is needed.
    #[inline]
    fn count(self) -> usize {
        self.deque.len()
    }

    // The last element is at the back: pop everything but it off the front
    // implicitly by reaching straight for the back in one step.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.deque.pop_back()
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.deque.pop_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn len(&self) -> usize {
        self.deque.len()
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

// ---------------------------------------------------------------------------
// IntoIterator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> IntoIterator for VecDeque<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::sync::Arc;

    fn build(n: usize) -> VecDeque<i32> {
        let mut dq = VecDeque::new();
        for i in 0..n {
            dq.try_push_back(i as i32).unwrap();
        }
        dq
    }

    #[test]
    fn into_iter_yields_all_front_to_back() {
        let collected: std::vec::Vec<_> = build(5).into_iter().collect();
        assert_eq!(collected, std::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn count_matches_length_without_consuming_visibly() {
        // `count` consumes the iterator but just reads the length.
        assert_eq!(build(7).into_iter().count(), 7);
        assert_eq!(build(0).into_iter().count(), 0);
    }

    #[test]
    fn last_returns_back_element_and_drops_rest() {
        // Track each element individually in a ledger so we can prove the front
        // elements are destroyed exactly once when `last` consumes the iterator
        // and pops only the back element — catching leaks and double-frees, not
        // just aggregate counts.
        use crate::test_helpers::{CloneBudget, FlakyTrackedItem, Ledger};
        let ledger = Arc::new(Ledger::new());
        let budget = Arc::new(CloneBudget::new(u32::MAX));
        let mut dq: VecDeque<FlakyTrackedItem> = VecDeque::new();
        for i in 0..4u32 {
            ledger.register(i);
            dq.try_push_back(FlakyTrackedItem {
                id: i,
                ledger: ledger.clone(),
                inner: (*budget).share(),
            })
            .unwrap();
        }
        assert!(ledger.live_ids().len() == 4);
        let last = dq.into_iter().last();
        // `last` yields the back element (id 3) and leaves the rest to the
        // deque's Drop, which destroys the three front elements.
        assert_eq!(last.as_ref().map(|t| t.id), Some(3));
        assert_eq!(ledger.live_ids(), std::vec![3]);
        assert!(ledger.double_dropped().is_empty());
        drop(last);
        // All four dropped exactly once; nothing leaked, nothing doubled.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn into_vecdeque_preserves_remaining_elements() {
        let mut it = build(5).into_iter();
        // Consume two from the front.
        assert_eq!(it.next(), Some(0));
        assert_eq!(it.next(), Some(1));
        let rest = it.into_vecdeque();
        let collected: std::vec::Vec<_> = rest.into_iter().collect();
        assert_eq!(collected, std::vec![2, 3, 4]);
    }
}
