//! The drop implementation for [`LinkedList`].
//!
//! Teardown pops from the front until the list is empty.
use super::LinkedList;
use crate::alloc::Allocator;
use core::mem;

impl<T, A: Allocator> Drop for LinkedList<T, A> {
    fn drop(&mut self) {
        struct DropGuard<'a, T, A: Allocator>(&'a mut LinkedList<T, A>);

        impl<T, A: Allocator> Drop for DropGuard<'_, T, A> {
            fn drop(&mut self) {
                // Continue the same loop as below. This only runs when a
                // destructor has panicked. If another one panics, this aborts.
                while self.0.pop_front_node().is_some() {}
            }
        }

        // Wrap `self` so that if an element destructor panics, the guard
        // keeps popping and the list still empties.
        let guard = DropGuard(self);
        while guard.0.pop_front_node().is_some() {}
        mem::forget(guard);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use olive_test_helpers::{Counter, DropCountingAlloc, Ledger, PanicArmer, TrackedItem};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    #[test]
    fn dropping_a_populated_list_drops_every_element_once() {
        let ledger = Arc::new(Ledger::new());
        let mut l = LinkedList::new();
        for i in 0..8u32 {
            l.try_push_front(TrackedItem::construct(&ledger, i))
                .unwrap();
        }
        assert_eq!(l.len(), 8);

        drop(l);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..8));
    }

    #[test]
    fn dropping_an_empty_list_drops_the_allocator_once() {
        let drops = Counter::shared();
        let l: LinkedList<i32, _> = LinkedList::new_in(DropCountingAlloc::new(drops.clone()));
        assert!(l.is_empty());

        drop(l);

        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn dropping_a_populated_list_drops_the_allocator_once() {
        let drops = Counter::shared();
        let mut l: LinkedList<i32, _> = LinkedList::new_in(DropCountingAlloc::new(drops.clone()));
        for i in 0..4 {
            l.try_push_front(i).unwrap();
        }

        drop(l);

        // Every node was deallocated and the allocator field destroyed
        // exactly once.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn dropping_with_a_panicking_element_still_empties_the_list() {
        let ledger = Arc::new(Ledger::new());
        let armer = Arc::new(PanicArmer::new());

        /// Simple panicking payload: panics exactly once, the first time it
        /// drops while armed. Registration is handled by the wrapping
        /// [`TrackedItem`].
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

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut l = LinkedList::new();
            for _ in 0..4 {
                l.try_push_front(TrackedItem::construct(
                    &ledger,
                    PanicDrop {
                        armer: armer.clone(),
                    },
                ))
                .unwrap();
            }
            armer.arm();
            drop(l);
        }));

        assert!(result.is_err(), "expected the drop to panic");

        assert_eq!(ledger.total_allocated(), 4);
        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..4));
    }
}
