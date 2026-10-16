//! Mutation operations for [`LinkedList`].

use core::marker::PhantomData;
use core::mem;

use crate::alloc::{Allocator, Global};

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

impl<T> LinkedList<T, Global> {
    /// Moves all elements from `other` to the back of the list, reusing
    /// `other`'s nodes without deallocating or reallocating anything.
    ///
    /// Afterwards, `other` is empty and reusable. No allocation happens, so
    /// this cannot fail.
    ///
    /// This method is only available to lists backed by the [`Global`]
    /// allocator. This operation should compute in O(1) time and O(1) memory.
    pub fn append(&mut self, other: &mut Self) {
        match self.tail {
            // `self` is empty: `other`'s chain becomes the whole list.
            None => {
                self.head = other.head.take();
                self.tail = other.tail.take();
            }
            Some(tail) => match other.head.take() {
                // `other` is empty: there is nothing to link.
                None => (),
                Some(node) => {
                    // SAFETY: `tail` is a live node of this list and `node`
                    // is the live first node of `other`. The two lists are
                    // distinct `&mut` borrows, so their nodes are disjoint,
                    // and linking the tail of one to the head of the other
                    // cannot create a cycle.
                    unsafe {
                        (*tail.as_ptr()).next = Some(node);
                        (*node.as_ptr()).prev = Some(tail);
                    }
                    self.tail = other.tail.take();
                }
            },
        }

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "overflow requires at least 2 * 2 ^ usize::BITS bytes of memory, which cannot fit in memory"
        )]
        {
            self.len += mem::take(&mut other.len);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::AllocError;
    use olive_test_helpers::{BudgetedAlloc, Ledger, MaybeUnclonable, PanicArmer, TrackedItem};
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

    #[test]
    fn append_moves_all_elements_to_the_back() {
        let mut a: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            a.try_push_back(value).unwrap();
        }
        let mut b: LinkedList<i32> = LinkedList::new();
        for value in 4..=5 {
            b.try_push_back(value).unwrap();
        }

        a.append(&mut b);

        assert_eq!(
            a.iter().copied().collect::<std::vec::Vec<_>>(),
            std::vec![1, 2, 3, 4, 5]
        );
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&5));
        assert_eq!(a.len(), 5);

        // The appended list is empty and reusable.
        assert_eq!(b.len(), 0);
        assert!(b.front().is_none());
        assert!(b.back().is_none());
        assert!(b.pop_front().is_none());
        assert!(b.pop_back().is_none());

        b.try_push_back(6).unwrap();
        assert_eq!(b.front(), Some(&6));
        assert_eq!(b.back(), Some(&6));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn append_to_an_empty_list_adopts_the_other_list() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            b.try_push_back(value).unwrap();
        }

        a.append(&mut b);

        assert_eq!(
            a.iter().copied().collect::<std::vec::Vec<_>>(),
            std::vec![1, 2, 3]
        );
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&3));
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 0);
        assert!(b.front().is_none());
        assert!(b.back().is_none());

        // The adopted nodes are deallocated through `a`, exercising that the
        // move kept them valid.
        drop(a);
    }

    #[test]
    fn append_an_empty_list_is_a_no_op() {
        let mut a: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            a.try_push_back(value).unwrap();
        }
        let mut b: LinkedList<i32> = LinkedList::new();

        a.append(&mut b);

        assert_eq!(
            a.iter().copied().collect::<std::vec::Vec<_>>(),
            std::vec![1, 2, 3]
        );
        assert_eq!(a.front(), Some(&1));
        assert_eq!(a.back(), Some(&3));
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn append_two_empty_lists_is_a_no_op() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();

        a.append(&mut b);

        assert_eq!(a.len(), 0);
        assert!(a.front().is_none());
        assert!(a.back().is_none());
        assert_eq!(b.len(), 0);
        assert!(b.front().is_none());
        assert!(b.back().is_none());
    }

    #[test]
    fn append_moves_elements_without_double_drops_or_leaks() {
        let ledger = Arc::new(Ledger::new());
        let mut a: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 0..2u32 {
            a.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }
        let mut b: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 2..5u32 {
            b.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        a.append(&mut b);

        // The move transferred ownership without dropping anything: all five
        // elements are still alive, owned by `a`.
        assert_eq!(ledger.total_allocated(), 5);
        assert_eq!(ledger.live_ids(), std::vec![0, 1, 2, 3, 4]);

        // Dropping the receiving list tears down every element exactly once,
        // including the ones that were allocated by `b`.
        drop(a);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));

        // The emptied list owns nothing: dropping it drops no element twice.
        drop(b);
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }
}
