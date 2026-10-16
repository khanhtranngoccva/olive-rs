//! Allocating growth operations for [`LinkedList`].

use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ptr;

use crate::alloc::{AllocError, Allocator};
use crate::boxed::Box;

use super::LinkedList;
use super::node::Node;

impl<T, A: Allocator> LinkedList<T, A> {
    /// Prepends `element` to the front of the list, allocating its node in the process.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the node fails. The list is then
    /// unchanged and the element is dropped.
    pub fn try_push_front(&mut self, element: T) -> Result<(), AllocError> {
        // The element half of the error tuple is dropped here, as documented.
        self.try_push_front_mut_give_back(element)
            .map(|_| ())
            .map_err(|(_, error)| error)
    }

    /// Prepends `element` to the front of the list, allocating its node in the process,
    /// and returns a mutable reference to the newly inserted element.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the node fails. The list is then
    /// unchanged and the element is dropped.
    pub fn try_push_front_mut(&mut self, element: T) -> Result<&mut T, AllocError> {
        // The element half of the error tuple is dropped here, as documented.
        self.try_push_front_mut_give_back(element)
            .map_err(|(_, error)| error)
    }

    /// Like [`try_push_front`](Self::try_push_front), but hands the element
    /// back on failure instead of dropping it.
    ///
    /// # Errors
    ///
    /// Returns `Err((element, AllocError))` if allocating the node fails. The
    /// list is then unchanged and the element is returned untouched.
    pub fn try_push_front_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        self.try_push_front_mut_give_back(element).map(|_| ())
    }

    /// Prepends `element` to the front of the list, allocating its node in the process,
    /// and returns a mutable reference to the newly inserted element.
    ///
    /// The reference stays valid for as long as the list is mutably borrowed:
    /// nodes are heap-allocated, so later pushes never move it.
    ///
    /// # Errors
    ///
    /// Returns `Err((element, AllocError))` if allocating the node fails. The
    /// list is then unchanged and the element is returned untouched.
    pub fn try_push_front_mut_give_back(&mut self, element: T) -> Result<&mut T, (T, AllocError)> {
        // Allocate the node before moving the element in, so a failed
        // allocation leaves the element untouched for the caller.
        let node = match Box::try_new_uninit_in(&self.alloc) {
            Ok(node) => node,
            Err(error) => return Err((element, error)),
        };
        // The allocation succeeded, so no failure is possible from here on.
        let node = node.write(Node {
            next: None,
            prev: None,
            element,
        });

        // SAFETY: the node was allocated with `&self.alloc` and is not
        // linked anywhere.
        let (node, _) = Box::into_non_null_with_allocator(node);
        // SAFETY: the node was just allocated with `&self.alloc`, the same
        // allocator this list deallocates with, and its element was fully
        // initialized by the `write` above.
        Ok(unsafe { self.push_front_node(node) })
    }

    /// Appends `element` to the back of the list, allocating its node in the process.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the node fails. The list is then
    /// unchanged and the element is dropped.
    pub fn try_push_back(&mut self, element: T) -> Result<(), AllocError> {
        self.try_push_back_mut_give_back(element)
            .map(|_| ())
            .map_err(|(_, error)| error)
    }

    /// Appends `element` to the back of the list, allocating its node in the process,
    /// and returns a mutable reference to the newly inserted element.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the node fails. The list is then
    /// unchanged and the element is dropped.
    pub fn try_push_back_mut(&mut self, element: T) -> Result<&mut T, AllocError> {
        self.try_push_back_mut_give_back(element)
            .map_err(|(_, error)| error)
    }

    /// Like [`try_push_back`](Self::try_push_back), but hands the element
    /// back on failure instead of dropping it.
    ///
    /// # Errors
    ///
    /// Returns `Err((element, AllocError))` if allocating the node fails. The
    /// list is then unchanged and the element is returned untouched.
    pub fn try_push_back_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        self.try_push_back_mut_give_back(element).map(|_| ())
    }

    /// Appends `element` to the back of the list, allocating its node in the process,
    /// and returns a mutable reference to the newly inserted element.
    ///
    /// The reference stays valid for as long as the list is mutably borrowed:
    /// nodes are heap-allocated, so later pushes never move it.
    ///
    /// # Errors
    ///
    /// Returns `Err((element, AllocError))` if allocating the node fails. The
    /// list is then unchanged and the element is returned untouched.
    pub fn try_push_back_mut_give_back(&mut self, element: T) -> Result<&mut T, (T, AllocError)> {
        // Allocate the node before moving the element in, so a failed
        // allocation leaves the element untouched for the caller.
        let node = match Box::try_new_uninit_in(&self.alloc) {
            Ok(node) => node,
            Err(error) => return Err((element, error)),
        };
        // The allocation succeeded, so no failure is possible from here on.
        let node = node.write(Node {
            next: None,
            prev: None,
            element,
        });

        // SAFETY: the node was allocated with `&self.alloc` and is not
        // linked anywhere.
        let (node, _) = Box::into_non_null_with_allocator(node);
        // SAFETY: the node was just allocated with `&self.alloc`, the same
        // allocator this list deallocates with, and its element was fully
        // initialized by the `write` above.
        Ok(unsafe { self.push_back_node(node) })
    }

    /// Moves all elements from `other` to the back of the list, but it works
    /// with arbitrary allocator instances. [`append`](Self::append) moves whole
    /// nodes instead and is restricted to the global allocator for that
    /// reason.
    ///
    /// This operation computes in O(n) time and O(n) memory.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if reserving a node fails. Both lists are then
    /// unchanged: the reserved space is freed again with `self`'s allocator,
    /// and no element has been moved out of `other` yet.
    pub fn try_append<B: Allocator>(
        &mut self,
        other: &mut LinkedList<T, B>,
    ) -> Result<(), AllocError> {
        // 1. Reserve - create a brand-new list of uninitialized nodes all at once.
        let mut reserved: LinkedList<MaybeUninit<T>, &A> = LinkedList::new_in(&self.alloc);
        for _ in 0..other.len {
            reserved.try_push_back(MaybeUninit::uninit())?;
        }

        // 2. Fill: move every element out of `other` and into the reserved
        // nodes, front to back.
        for (source, target) in other.iter().zip(reserved.iter_mut()) {
            // SAFETY: `source` points at a live element of `other` and no
            // reference to it exists, so reading it moves the element out.
            // The emptied slot is never dropped as a `T` again: the teardown
            // in step 4 views it as a `MaybeUninit`.
            let element = unsafe { ptr::read(source) };
            target.write(element);
        }

        // 3. Append: append the `reserved` the back of `self`, and clear `reserved` so that it no longer has
        // any owned nodes.
        match self.tail {
            // `self` is empty: the reserved chain becomes the whole list.
            None => {
                self.head = reserved.head.take().map(|node| node.cast());
                self.tail = reserved.tail.take().map(|node| node.cast());
            }
            Some(tail) => match reserved.head.take() {
                // `other` was empty: there is nothing to link.
                None => (),
                // This branch is sound because it only executes when both lists have at least 1 element.
                Some(node) => {
                    // SAFETY: `tail` is a live node of this list and `node`
                    // is the live first node of the reserved list. The two
                    // lists are distinct values, so their nodes are
                    // disjoint, and linking them cannot create a cycle.
                    unsafe {
                        (*tail.as_ptr()).next = Some(node.cast());
                        (*node.cast::<Node<T>>().as_ptr()).prev = Some(tail);
                    }
                    self.tail = reserved.tail.take().map(|node| node.cast());
                }
            },
        }

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "overflow requires at least 2 * 2 ^ usize::BITS bytes of memory, which cannot fit in memory"
        )]
        {
            self.len += mem::take(&mut reserved.len);
        }
        // `reserved` is empty now, so dropping it deallocates nothing.

        // 4. Empty `other` and tear its nodes down through a fake,
        // ad-hoc list of `MaybeUninit` items on `other`'s own allocator.
        drop(LinkedList::<MaybeUninit<T>, &B> {
            head: other.head.take().map(|node| node.cast()),
            tail: other.tail.take().map(|node| node.cast()),
            len: mem::take(&mut other.len),
            alloc: &other.alloc,
            marker: PhantomData,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::sync::Arc;

    use olive_test_helpers::{BudgetedAlloc, FailAlloc, Ledger, MaybeUnclonable, TrackedItem};

    use super::*;
    use crate::alloc::AllocError;

    #[test]
    fn try_push_front_grows_the_list() {
        let mut l = LinkedList::new();
        for i in 0..3 {
            l.try_push_front(i).unwrap();
            assert_eq!(l.len(), i + 1);
        }
        assert_eq!(
            l.front(),
            Some(&2),
            "the last pushed front element is the head"
        );
        assert!(!l.is_empty());
    }

    #[test]
    fn try_push_front_prepends_at_the_front() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            l.try_push_front(value).unwrap();
        }

        assert_eq!(
            l.front(),
            Some(&3),
            "the first appended element is the head"
        );
        assert_eq!(l.back(), Some(&1), "the last appended element is the tail");
        // A correct append preserves front-to-back order; a prepend would
        // have reversed it.
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![3, 2, 1]);
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn try_push_front_oom_leaves_the_list_unchanged() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(4);
        let mut l = LinkedList::new_in(budget.clone());
        for i in 0..4u32 {
            l.try_push_front(TrackedItem::construct(&ledger, i))
                .unwrap();
        }

        assert!(matches!(
            l.try_push_front(TrackedItem::construct(&ledger, 4)),
            Err(AllocError)
        ));
        assert_eq!(l.len(), 4);

        drop(l);

        // The four stored elements plus the failed push's element (dropped on
        // failure, as documented) each died exactly once.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_push_front_give_back_returns_the_element_on_oom() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, FailAlloc> = LinkedList::new_in(FailAlloc);
        let Err((item, AllocError)) =
            l.try_push_front_give_back(TrackedItem::construct(&ledger, 0))
        else {
            panic!("expected an allocation failure");
        };

        assert!(l.is_empty());
        drop(item);
        drop(l);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..1));
    }

    #[test]
    fn try_push_front_mut_returns_a_reference_to_the_stored_element() {
        let mut l: LinkedList<i32> = LinkedList::new();
        let front = l.try_push_front_mut(1).unwrap();
        assert_eq!(*front, 1);
        *front = 2;
        assert_eq!(l.len(), 1);
        assert!(!l.is_empty());
    }

    #[test]
    fn try_push_front_mut_drops_the_element_on_oom() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, FailAlloc> = LinkedList::new_in(FailAlloc);
        assert!(matches!(
            l.try_push_front_mut(TrackedItem::construct(&ledger, 0)),
            Err(AllocError)
        ));

        assert!(l.is_empty());
        drop(l);

        // The failed push's element was dropped on failure, as documented.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..1));
    }

    #[test]
    fn try_push_front_mut_give_back_returns_a_reference_to_the_stored_element() {
        let mut l: LinkedList<i32> = LinkedList::new();

        {
            let front = l.try_push_front_mut_give_back(1).unwrap();
            assert_eq!(*front, 1);
            *front = 42;
        }

        assert_eq!(l.len(), 1);
        // The mutation through the returned reference reached the stored
        // element.
        assert_eq!(l.front(), Some(&42));
        assert_eq!(l.back(), Some(&42));
    }

    #[test]
    fn try_push_front_mut_give_back_returns_the_element_on_oom() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, FailAlloc> = LinkedList::new_in(FailAlloc);
        let Err((item, AllocError)) =
            l.try_push_front_mut_give_back(TrackedItem::construct(&ledger, 0))
        else {
            panic!("expected an allocation failure");
        };

        assert!(l.is_empty());
        drop(item);
        drop(l);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..1));
    }

    // This test is written to run under Miri. Pushing a second element to the front
    // should not invalidate the first element's aliasing reference when writing the old
    // head's `prev` pointer.
    #[test]
    fn pointer_to_first_element_survives_a_push_front() {
        let mut l: LinkedList<i32> = LinkedList::new();
        let first = l.try_push_front_mut_give_back(1).unwrap();
        assert_eq!(*first, 1);
        let pointer: *mut i32 = first;
        assert_eq!(unsafe { *pointer }, 1);

        let reference: &mut i32 = unsafe { &mut *pointer };
        l.try_push_front(2).unwrap();

        // The first element's storage was untouched by the push:
        // the reference must still read the value it was created from.
        assert_eq!(*reference, 1);
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn try_push_back_grows_the_list() {
        let mut l = LinkedList::new();
        for i in 0..3 {
            l.try_push_back(i).unwrap();
            assert_eq!(l.len(), i + 1);
        }
        assert!(!l.is_empty());
    }

    #[test]
    fn try_push_back_appends_at_the_back() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            l.try_push_back(value).unwrap();
        }

        assert_eq!(
            l.front(),
            Some(&1),
            "the first appended element is the head"
        );
        assert_eq!(l.back(), Some(&3), "the last appended element is the tail");
        // A correct append preserves front-to-back order; a prepend would
        // have reversed it.
        let collected: std::vec::Vec<i32> = l.iter().copied().collect();
        assert_eq!(collected, std::vec![1, 2, 3]);
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn try_push_back_oom_leaves_the_list_unchanged() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(4);
        let mut l = LinkedList::new_in(budget.clone());
        for i in 0..4u32 {
            l.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        assert!(matches!(
            l.try_push_back(TrackedItem::construct(&ledger, 4)),
            Err(AllocError)
        ));
        assert_eq!(l.len(), 4);

        drop(l);

        // The four stored elements plus the failed push's element (dropped on
        // failure, as documented) each died exactly once.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_push_back_give_back_returns_the_element_on_oom() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, FailAlloc> = LinkedList::new_in(FailAlloc);
        let Err((item, AllocError)) = l.try_push_back_give_back(TrackedItem::construct(&ledger, 0))
        else {
            panic!("expected an allocation failure");
        };

        assert!(l.is_empty());
        drop(item);
        drop(l);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..1));
    }

    #[test]
    fn try_push_back_mut_returns_a_reference_to_the_stored_element() {
        let mut l: LinkedList<i32> = LinkedList::new();
        let back = l.try_push_back_mut(1).unwrap();
        assert_eq!(*back, 1);
        *back = 2;
        assert_eq!(l.len(), 1);
        assert!(!l.is_empty());
    }

    #[test]
    fn try_push_back_mut_drops_the_element_on_oom() {
        let ledger = Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, FailAlloc> = LinkedList::new_in(FailAlloc);
        assert!(matches!(
            l.try_push_back_mut(TrackedItem::construct(&ledger, 0)),
            Err(AllocError)
        ));

        assert!(l.is_empty());
        drop(l);

        // The failed push's element was dropped on failure, as documented.
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..1));
    }

    // This test is written to run under Miri. Pushing a new last element
    // rewrites the old tail's `next` field, which must not tamper with the
    // aliasing reference held against the old last element.
    #[test]
    fn pointer_to_last_element_survives_a_push_back() {
        let mut l: LinkedList<i32> = LinkedList::new();
        let last = l.try_push_back_mut_give_back(1).unwrap();
        assert_eq!(*last, 1);
        let pointer: *mut i32 = last;
        assert_eq!(unsafe { *pointer }, 1);

        let reference: &mut i32 = unsafe { &mut *pointer };
        l.try_push_back(2).unwrap();

        // The old last element's storage was untouched by the push: the
        // reference must still read the value it was created from.
        assert_eq!(*reference, 1);
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn try_append_moves_all_elements_to_the_back_without_double_drops_or_leaks() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(5);

        let mut a: LinkedList<TrackedItem<u32>, _> = LinkedList::new_in(budget.clone());
        for i in 0..2u32 {
            a.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }
        let mut b: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 2..5u32 {
            b.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        a.try_append(&mut b)
            .expect("the budget covers exactly five nodes");

        assert_eq!(
            a.iter()
                .map(|item| item.inner)
                .collect::<std::vec::Vec<_>>(),
            std::vec![0, 1, 2, 3, 4]
        );
        assert_eq!(a.front().unwrap().inner, 0);
        assert_eq!(a.back().unwrap().inner, 4);
        assert_eq!(a.len(), 5);

        // The elements were moved out of `b`, not cloned: it is empty.
        assert_eq!(b.len(), 0);
        assert!(b.front().is_none());
        assert!(b.back().is_none());

        // Nothing was dropped along the way.
        assert_eq!(ledger.total_allocated(), 5);
        assert_eq!(ledger.live_ids(), std::vec![0, 1, 2, 3, 4]);

        drop(a);
        drop(b);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_append_to_an_empty_list_adopts_the_other_list() {
        let mut a: LinkedList<i32> = LinkedList::new();
        let mut b: LinkedList<i32> = LinkedList::new();
        for value in 1..=3 {
            b.try_push_back(value).unwrap();
        }

        a.try_append(&mut b).expect("moving i32 cannot fail");

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
    fn try_append_an_empty_list_is_a_no_op() {
        // The budget covers `a`'s three nodes exactly, so a successful
        // append of an empty list proves that no node was allocated.
        let budget = BudgetedAlloc::new(3);
        let mut a: LinkedList<i32, _> = LinkedList::new_in(budget.clone());
        for value in 1..=3 {
            a.try_push_back(value).unwrap();
        }
        let mut b: LinkedList<i32, _> = LinkedList::new_in(budget.clone());

        a.try_append(&mut b)
            .expect("appending an empty list cannot fail");

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
    fn try_append_oom_leaves_both_lists_unchanged() {
        let ledger = Arc::new(Ledger::new());
        let budget = BudgetedAlloc::new(4);
        let mut a: LinkedList<TrackedItem<u32>, _> = LinkedList::new_in(budget.clone());
        for i in 0..2u32 {
            a.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }
        let mut b: LinkedList<TrackedItem<u32>> = LinkedList::new();
        for i in 2..5u32 {
            b.try_push_back(TrackedItem::construct(&ledger, i)).unwrap();
        }

        // Two nodes are still funded, so two reserve and the third fails.
        let error = a.try_append(&mut b).expect_err("the budget is spent");
        assert!(matches!(error, AllocError), "{error:?}");

        // `a` is unchanged.
        assert_eq!(a.len(), 2);
        assert_eq!(
            a.iter()
                .map(|item| item.inner)
                .collect::<std::vec::Vec<_>>(),
            std::vec![0, 1]
        );

        // `b` is unchanged too: the failure happened entirely in the
        // reserve phase, before any element was moved out of it.
        assert_eq!(b.len(), 3);
        assert_eq!(
            b.iter()
                .map(|item| item.inner)
                .collect::<std::vec::Vec<_>>(),
            std::vec![2, 3, 4]
        );

        budget.set_budget(3);
        a.try_append(&mut b)
            .expect("the budget now covers three nodes");
        assert_eq!(a.len(), 5);
        assert_eq!(
            a.iter()
                .map(|item| item.inner)
                .collect::<std::vec::Vec<_>>(),
            std::vec![0, 1, 2, 3, 4]
        );
        assert_eq!(b.len(), 0);

        drop(a);
        drop(b);
        assert_eq!(ledger.total_allocated(), 5);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..5));
    }

    #[test]
    fn try_append_moves_elements_without_cloning_them() {
        let mut a: LinkedList<MaybeUnclonable<u32>> = LinkedList::new();
        for i in 0..2u32 {
            a.try_push_back(MaybeUnclonable::unclonable(i)).unwrap();
        }
        let mut b: LinkedList<MaybeUnclonable<u32>> = LinkedList::new();
        for i in 2..5u32 {
            b.try_push_back(MaybeUnclonable::unclonable(i)).unwrap();
        }

        a.try_append(&mut b)
            .expect("no clone takes place, so nothing can fail");

        // Ownership moved: `a` holds all five elements in order.
        assert_eq!(
            a.iter()
                .map(|item| item.inner)
                .collect::<std::vec::Vec<_>>(),
            std::vec![0, 1, 2, 3, 4]
        );
        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 0);

        // Dropping the lists tears every element down exactly once: no
        // element was dropped twice by the emptied `b`.
        drop(a);
        drop(b);
    }
}
