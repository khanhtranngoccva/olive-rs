//! Mutation operations for [`LinkedList`].

use crate::alloc::Allocator;

use super::LinkedList;
use super::node::Node;

impl<T, A: Allocator> LinkedList<T, A> {
    /// Removes the first element and returns it, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        self.pop_front_node().map(Node::into_element)
    }

    /// Removes the last element and returns it, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        self.pop_back_node().map(Node::into_element)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_test_helpers::{Ledger, TrackedItem};
    use std::sync::Arc;

    #[test]
    fn pop_front_and_pop_back_are_none_on_an_empty_list() {
        let mut l: LinkedList<i32> = LinkedList::new();
        assert!(l.pop_front().is_none());
        assert!(l.pop_back().is_none());
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn pop_front_removes_elements_front_to_back() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            l.try_push_back(value).unwrap();
        }

        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.front(), Some(&2));
        assert_eq!(l.back(), Some(&3));
        assert_eq!(l.len(), 2);

        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));

        // The list is empty again: both ends report no element, and the
        // pops keep reporting `None`.
        assert!(l.front().is_none());
        assert!(l.back().is_none());
        assert_eq!(l.len(), 0);
        assert!(l.pop_front().is_none());
    }

    #[test]
    fn pop_back_removes_elements_back_to_front() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            l.try_push_back(value).unwrap();
        }

        assert_eq!(l.pop_back(), Some(3));
        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&2));
        assert_eq!(l.len(), 2);

        assert_eq!(l.pop_back(), Some(2));
        assert_eq!(l.pop_back(), Some(1));

        // The list is empty again: both ends report no element, and the
        // pops keep reporting `None`.
        assert!(l.front().is_none());
        assert!(l.back().is_none());
        assert_eq!(l.len(), 0);
        assert!(l.pop_back().is_none());
    }

    #[test]
    fn popping_drops_each_element_exactly_once() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 0..3u32 {
            l.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        // Popping moves the element out of its node: dropping the popped
        // value drops it exactly once, while the remaining elements stay
        // owned by the list.
        assert!(l.pop_front().is_some());
        assert_eq!(ledger.drop_count(0), 1);
        assert!(l.pop_back().is_some());
        assert_eq!(ledger.drop_count(2), 1);
        assert_eq!(ledger.drop_count(1), 0);
        assert_eq!(l.len(), 1);

        // Dropping the rest of the list tears down the remaining element.
        drop(l);
        assert_eq!(ledger.drop_count(1), 1);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..3));
    }
}
