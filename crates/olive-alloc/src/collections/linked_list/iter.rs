//! Iterators over a [`LinkedList`].
//!
//! This mirrors the std reference implementation, whose iterators likewise
//! track the head and tail pointers and the remaining length.

use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

use olive_core::prelude::{TryClone, TryDefault};
use olive_core::try_traits::try_clone::TryCloneError;
use olive_core::try_traits::try_default::TryDefaultError;

use super::LinkedList;
use super::node::Node;
use crate::alloc::{Allocator, Global};

/// An iterator over the elements of a `LinkedList`.
///
/// This struct is created by the [`iter`](LinkedList::iter) method on
/// `LinkedList`. See its documentation for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a Node<T>>,
}

impl<T> Default for Iter<'_, T> {
    /// Creates an empty `linked_list::Iter`.
    ///
    /// The returned iterator iterates over no elements.
    #[inline]
    fn default() -> Self {
        Iter {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }
}

// No allocation is involved, so the fallible form can never fail.
impl<T> TryClone for Iter<'_, T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

// An empty iterator holds no storage, so its default construction is
// infallible.
impl<T> TryDefault for Iter<'_, T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::default())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            return None;
        }
        self.head.map(|node| {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "cannot underflow, we only decrement when the iterator is not empty"
            )]
            {
                self.len -= 1;
            }
            // SAFETY: `node` points to a live node of the list this iterator
            // borrows, whose links and element stay initialized and unchanged
            // for as long as the list is only shared-borrowed.
            unsafe {
                let node = &*node.as_ptr();
                self.head = node.next;
                &node.element
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            return None;
        }
        self.tail.map(|node| {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "cannot underflow, we only decrement when the iterator is not empty"
            )]
            {
                self.len -= 1;
            }
            // SAFETY: `node` points to a live node of the list this iterator
            // borrows, whose links and element stay initialized and unchanged
            // for as long as the list is only shared-borrowed.
            unsafe {
                let node = &*node.as_ptr();
                self.tail = node.prev;
                &node.element
            }
        })
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter")
            .field(&*mem::ManuallyDrop::new(LinkedList {
                head: self.head,
                tail: self.tail,
                len: self.len,
                alloc: Global,
                marker: PhantomData,
            }))
            .field(&self.len)
            .finish()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a LinkedList<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    /// Creates a front-to-back iterator from an immutably borrowed list.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A mutable iterator over the elements of a `LinkedList`.
///
/// This struct is created by the [`iter_mut`](LinkedList::iter_mut) method on
/// `LinkedList`. See its documentation for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IterMut<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a mut Node<T>>,
}

impl<T> Default for IterMut<'_, T> {
    /// Creates an empty `linked_list::IterMut`.
    ///
    /// The returned iterator iterates over no elements.
    #[inline]
    fn default() -> Self {
        IterMut {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
        }
    }
}

// An empty iterator holds no storage, so its default construction is
// infallible.
impl<T> TryDefault for IterMut<'_, T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::default())
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        // The `len` check also guards the pointer walk: when it reaches
        // zero, the front and the back have met and `head` is stale.
        if self.len == 0 {
            return None;
        }
        self.head.map(|node| {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "cannot underflow, we only decrement when the iterator is not empty"
            )]
            {
                self.len -= 1;
            }
            // SAFETY: `node` points to a live node of the list this iterator
            // exclusively borrows, and its element is initialized for as long
            // as the node is in the list. The iterator yields at most one
            // mutable reference per node.
            unsafe {
                let node = &mut *node.as_ptr();
                self.head = node.next;
                &mut node.element
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<&'a mut T> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            return None;
        }
        self.tail.map(|node| {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "cannot underflow, we only decrement when the iterator is not empty"
            )]
            {
                self.len -= 1;
            }
            // SAFETY: `node` points to a live node of the list this iterator
            // exclusively borrows, and its element is initialized for as long
            // as the node is in the list. The iterator yields at most one
            // mutable reference per node.
            unsafe {
                let node = &mut *node.as_ptr();
                self.tail = node.prev;
                &mut node.element
            }
        })
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> FusedIterator for IterMut<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut")
            .field(&*mem::ManuallyDrop::new(LinkedList {
                head: self.head,
                tail: self.tail,
                len: self.len,
                alloc: Global,
                marker: PhantomData,
            }))
            .field(&self.len)
            .finish()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut LinkedList<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    /// Creates a front-to-back iterator from a mutably-borrowed list.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// An owning iterator over the elements of a `LinkedList`.
///
/// This struct is created by the `into_iter` method on `LinkedList` (provided
/// by the [`IntoIterator`] trait). See its documentation for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IntoIter<T, A: Allocator = Global> {
    /// The by-value list being consumed. `next` pops its front node and
    /// `next_back` its back node; whatever remains when the iterator is
    /// dropped is destroyed by the list's own guarded `Drop`.
    list: LinkedList<T, A>,
}

// SAFETY: the iterator owns the whole list, so it carries the exact same
// thread safety requirements as the list itself.
unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}

// SAFETY: the iterator owns the whole list, so it carries the exact same
// thread safety requirements as the list itself.
unsafe impl<T: Sync, A: Allocator + Sync> Sync for IntoIter<T, A> {}

impl<T, A: Allocator> IntoIter<T, A> {
    pub(super) fn new(list: LinkedList<T, A>) -> Self {
        Self { list }
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.list.pop_front_node().map(Node::into_element)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<T> {
        self.next_back()
    }
}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.list.pop_back_node().map(Node::into_element)
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn len(&self) -> usize {
        self.list.len
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.list).finish()
    }
}

impl<T, A: Allocator> IntoIterator for LinkedList<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    /// Creates a consuming iterator, that is, one that moves each element out
    /// of the list (from front to back). The list cannot be used afterwards.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<T, A: Allocator> LinkedList<T, A> {
    /// Returns a front-to-back iterator over the list's elements.
    ///
    /// The iterator is double-ended and exact-size: it tracks the head and
    /// tail pointers plus the remaining length.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use olive_alloc::collections::linked_list::LinkedList;
    ///
    /// let mut list = LinkedList::new();
    /// for value in 0..3 {
    ///     list.try_push_back(value).unwrap();
    /// }
    ///
    /// let mut iterator = list.iter();
    /// assert_eq!(iterator.next(), Some(&0));
    /// assert_eq!(iterator.next_back(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }

    /// Returns a front-to-back iterator over mutable references to the list's
    /// elements.
    ///
    /// The iterator is double-ended and exact-size: it tracks the head and
    /// tail pointers plus the remaining length.
    ///
    /// This operation should compute in O(1) time.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_test_helpers::{Ledger, TrackedItem};
    use std::format;
    use std::sync::Arc;
    use std::vec::Vec;

    #[test]
    fn iter_yields_elements_front_to_back() {
        let mut l: LinkedList<i32> = LinkedList::new();
        let mut iter = l.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        for value in 1..=4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_is_double_ended() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_references_coexist() {
        // Stress for Miri: a double-ended iterator must be able to hold live
        // shared references to both ends of the list at once.
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter();
        let front: &i32 = iter.next().unwrap();
        let back: &i32 = iter.next_back().unwrap();
        assert_eq!(front, &1);
        assert_eq!(back, &4);
    }

    #[test]
    fn iter_and_iter_mut_size_hint_and_len_are_exact() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.len(), 2);
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.len(), 1);

        let mut iter = l.iter_mut();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.len(), 2);
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.len(), 1);
    }

    #[test]
    fn iter_last_returns_the_back_element() {
        let mut l: LinkedList<i32> = LinkedList::new();
        assert_eq!(l.iter().last(), None);
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(l.iter().last(), Some(&2));
    }

    #[test]
    fn iter_clone_continues_independently() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(&0));
        let mut cloned = iter.clone();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(cloned.next(), Some(&1));
        // The two iterators advance over the same nodes independently.
        assert_eq!(cloned.next_back(), Some(&3));
        assert_eq!(iter.next_back(), Some(&3));
        assert_eq!(cloned.next(), Some(&2));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(cloned.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_default_is_empty() {
        let iter: Iter<'_, i32> = Iter::default();
        assert_eq!(iter.count(), 0);

        let iter: IterMut<'_, i32> = IterMut::default();
        assert_eq!(iter.count(), 0);
    }

    #[test]
    fn iter_mut_modifies_elements_in_place() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..5 {
            l.try_push_back(value).unwrap();
        }
        for element in l.iter_mut() {
            *element *= 2;
        }
        let collected: Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, [0, 2, 4, 6, 8]);
    }

    #[test]
    fn iter_mut_is_double_ended() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter_mut();
        assert_eq!(iter.next(), Some(&mut 0));
        assert_eq!(iter.next_back(), Some(&mut 3));
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next_back(), Some(&mut 2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn iter_mut_references_coexist_and_are_disjoint() {
        // Stress for Miri: the double-ended mutable iterator must be able to
        // hand out live mutable references to every element of the list at once.
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=4 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.iter_mut();
        let front: &mut i32 = iter.next().unwrap();
        let back: &mut i32 = iter.next_back().unwrap();
        *front += 10;
        *back += 100;
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next_back(), Some(&mut 3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        // `front` and `back` stayed live across the interleaved iterations
        // above, so the three mutable references truly coexisted.
        assert_eq!(front, &11);
        assert_eq!(back, &104);
    }

    #[test]
    fn iter_mut_last_returns_the_back_element() {
        let mut l: LinkedList<i32> = LinkedList::new();
        assert_eq!(l.iter_mut().last(), None);
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(l.iter_mut().last(), Some(&mut 2));
    }

    #[test]
    fn into_iter_yields_all_front_to_back() {
        let l: LinkedList<i32> = LinkedList::new();
        assert_eq!(l.into_iter().count(), 0);

        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..5 {
            l.try_push_back(value).unwrap();
        }
        let collected: Vec<i32> = l.into_iter().collect();
        assert_eq!(collected, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn into_iter_is_double_ended() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..5 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn into_iter_size_hint_and_len_are_exact() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        let mut iter = l.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.len(), 2);
        iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.len(), 1);
    }

    #[test]
    fn into_iter_count_matches_the_length() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(l.into_iter().count(), 3);
    }

    #[test]
    fn into_iter_last_yields_the_back_element() {
        let l: LinkedList<i32> = LinkedList::new();
        assert_eq!(l.into_iter().last(), None);

        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(l.into_iter().last(), Some(2));
    }

    #[test]
    fn into_iter_drops_only_unconsumed_elements() {
        let ledger = Arc::new(Ledger::new());
        let mut l = LinkedList::new();
        for value in 0..4 {
            l.try_push_back(TrackedItem::construct(&ledger, value))
                .unwrap();
        }
        let mut iter = l.into_iter();
        iter.next();
        iter.next_back();
        // The remaining two elements are destroyed by the list's own guarded
        // `Drop` when the iterator goes away.
        drop(iter);

        assert_eq!(ledger.total_allocated(), 4);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }

    #[test]
    fn into_iterator_works_in_for_loops() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        let mut collected: Vec<i32> = Vec::new();
        for value in &l {
            collected.push(*value);
        }
        assert_eq!(collected, [0, 1, 2]);

        for value in &mut l {
            *value *= 2;
        }
        let mut doubled: Vec<i32> = Vec::new();
        for value in &l {
            doubled.push(*value);
        }
        assert_eq!(doubled, [0, 2, 4]);

        let mut moved: Vec<i32> = Vec::new();
        for value in l {
            moved.push(value);
        }
        assert_eq!(moved, [0, 2, 4]);
    }

    #[test]
    fn debug_matches_std() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..9 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(format!("{l:?}"), "[0, 1, 2, 3, 4, 5, 6, 7, 8]");

        assert_eq!(
            format!("{:?}", l.iter()),
            "Iter([0, 1, 2, 3, 4, 5, 6, 7, 8], 9)"
        );

        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..3 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(format!("{:?}", l.iter_mut()), "IterMut([0, 1, 2], 3)");

        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 0..5 {
            l.try_push_back(value).unwrap();
        }
        assert_eq!(format!("{:?}", l.into_iter()), "IntoIter([0, 1, 2, 3, 4])");
    }
}
