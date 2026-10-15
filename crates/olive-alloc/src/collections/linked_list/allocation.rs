//! Allocating growth operations for [`LinkedList`].

use crate::alloc::{AllocError, Allocator};
use crate::boxed::Box;

use super::LinkedList;
use super::node::Node;

impl<T, A: Allocator> LinkedList<T, A> {
    /// Prepends `element` to the front of the list, allocating its node.
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

    /// Prepends `element` to the front of the list, allocating its node, and
    /// returns a mutable reference to the newly inserted element.
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

    /// Prepends `element` to the front of the list, allocating its node, and
    /// returns a mutable reference to the newly inserted element.
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
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::cell::Cell;
    use std::sync::Arc;

    use olive_test_helpers::{BudgetedAlloc, FailAlloc, Ledger, TrackedItem};

    use super::*;
    use crate::alloc::AllocError;

    #[test]
    fn try_push_front_grows_the_list() {
        let mut l = LinkedList::new();
        for i in 0..3 {
            l.try_push_front(i).unwrap();
            assert_eq!(l.len(), i + 1);
        }
        assert!(!l.is_empty());
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

    // FIXME: add query methods to simplify this test
    #[test]
    fn try_push_front_mut_give_back_returns_a_reference_to_the_stored_element() {
        // A `Drop` observer: whatever value the element holds when the list
        // tears it down lands in `sink`, which is how the test reads back a
        // stored element before any pop or iteration API exists.
        #[derive(Debug)]
        struct DropCell<'a> {
            value: i32,
            sink: &'a Cell<i32>,
        }

        impl Drop for DropCell<'_> {
            fn drop(&mut self) {
                self.sink.set(self.value);
            }
        }

        let sink = Cell::new(0);
        let mut l: LinkedList<DropCell<'_>> = LinkedList::new();

        {
            let front = l
                .try_push_front_mut_give_back(DropCell {
                    value: 1,
                    sink: &sink,
                })
                .unwrap();
            assert_eq!(front.value, 1);
            front.value = 42;
        }

        assert_eq!(l.len(), 1);
        drop(l);
        // The mutation through the returned reference reached the stored
        // element: teardown dropped it with the updated value.
        assert_eq!(sink.get(), 42);
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

    // This test is written to run under Miri. Pushing a second element should not
    // invalidate the first element's aliasing reference.
    #[test]
    fn pointer_to_first_element_survives_a_second_push() {
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
}
