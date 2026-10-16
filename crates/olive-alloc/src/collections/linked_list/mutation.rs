//! Mutation operations for [`LinkedList`].

use core::marker::PhantomData;
use core::mem;

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

    /// Removes all elements from the list.
    ///
    /// This operation should compute in O(n) time.
    pub fn clear(&mut self) {
        // Construct an ad-hoc list and drop it.
        drop(LinkedList {
            head: self.head.take(),
            tail: self.tail.take(),
            len: mem::take(&mut self.len),
            alloc: &self.alloc,
            marker: PhantomData,
        });
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_test_helpers::{Ledger, PanicArmer, TrackedItem};
    use std::panic::{AssertUnwindSafe, catch_unwind};
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

    #[test]
    fn clear_on_an_empty_list_is_a_no_op() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.clear();

        assert_eq!(l.len(), 0);
        assert!(l.front().is_none());
        assert!(l.back().is_none());
    }

    #[test]
    fn clear_removes_all_elements_and_leaves_the_list_reusable() {
        let ledger = Arc::new(Ledger::new());
        let mut l = LinkedList::new();
        for value in 0..4 {
            l.try_push_back(TrackedItem::construct(&ledger, value))
                .unwrap();
        }

        l.clear();

        assert_eq!(l.len(), 0);
        assert!(l.front().is_none());
        assert!(l.back().is_none());
        assert!(l.pop_front().is_none());
        assert!(l.pop_back().is_none());

        // The clear destroyed every stored element exactly once.
        assert_eq!(ledger.total_allocated(), 4);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));

        // The list still works afterwards: it can take new elements, tracked
        // by ids continuing the same ledger.
        l.try_push_back(TrackedItem::construct(&ledger, 4)).unwrap();
        assert_eq!(l.front().unwrap().inner, 4);
        assert_eq!(l.back().unwrap().inner, 4);
        assert_eq!(l.len(), 1);

        drop(l);
        assert_eq!(ledger.total_allocated(), 5);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn clear_drops_each_element_exactly_once() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 0..3u32 {
            l.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        l.clear();

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..3));
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn clearing_with_a_panicking_element_still_empties_the_list() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        struct PanicDrop {
            armer: Arc<PanicArmer>,
        }

        impl Drop for PanicDrop {
            fn drop(&mut self) {
                if self.armer.is_armed() {
                    self.armer.disarm();
                    panic!("forced panic in drop");
                }
            }
        }

        let mut l = LinkedList::new();
        for _ in 0..4 {
            l.try_push_back(TrackedItem::construct(
                &ledger,
                PanicDrop {
                    armer: armer.clone(),
                },
            ))
            .unwrap();
        }

        armer.arm();
        let result = catch_unwind(AssertUnwindSafe(|| l.clear()));
        assert!(result.is_err(), "expected the clear to panic");

        // The teardown ran through the ad-hoc list's guarded `Drop`, so the
        // panic left `l` empty rather than half-cleared, and it stays
        // usable afterwards.
        assert_eq!(l.len(), 0);
        assert!(l.front().is_none());
        assert!(l.back().is_none());
        // The list accepts new elements again, and clearing it a second
        // time works now that nothing is armed.
        l.try_push_back(TrackedItem::construct(
            &ledger,
            PanicDrop {
                armer: armer.clone(),
            },
        ))
        .unwrap();
        assert_eq!(l.len(), 1);
        l.clear();
        assert_eq!(l.len(), 0);

        assert_eq!(ledger.total_allocated(), 5);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }
}
