//! Mutable access to the maximum, with the re-sift deferred until the guard
//! drops.
use core::fmt;
use core::ops::{Deref, DerefMut};

use super::BinaryHeap;
use super::poison::PoisonOnUnwind;
use crate::alloc::Allocator;

impl<T: Ord, A: Allocator> BinaryHeap<T, A> {
    /// Returns a guard over the maximum element of the binary heap, or
    /// `None` if it is empty.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned (see [`BinaryHeap::rebuild`]). A
    /// comparison that panics during the drop of the `PeekMut` poisons
    /// the heap.
    ///
    /// # Leaking
    ///
    /// Do not [`core::mem::forget`] the PeekMut. Otherwise, any tail
    /// elements will be leaked (which is better than leaving the heap's
    /// invariants silently broken after modifying the peeked element
    /// and attempting to restore it).
    #[inline]
    #[must_use]
    pub fn peek_mut(&mut self) -> Option<PeekMut<'_, T, A>> {
        self.assume_unpoisoned();
        (!self.is_empty()).then(|| PeekMut {
            heap: self,
            original_len: None,
        })
    }
}

/// A wrapper over the maximum element of a [`BinaryHeap`], created by
/// [`peek_mut`].
///
/// # Leaking
///
/// The guard should not be leaked. See [`peek_mut`] for details.
///
/// # Panics
///
/// Panics and poisons the heap if the comparison implementation panics.
///
/// [`peek_mut`]: BinaryHeap::peek_mut
pub struct PeekMut<'a, T: Ord, A: Allocator = crate::alloc::Global> {
    /// The borrowed heap.
    heap: &'a mut BinaryHeap<T, A>,
    /// `Some(len)` while a leak amplification is active.
    original_len: Option<usize>,
}

impl<T: Ord, A: Allocator> Deref for PeekMut<'_, T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // We do nothing - the element is not mutated.
        self.heap.assume_unpoisoned();
        self.heap
            .data
            .as_slice()
            .first()
            .expect("a PeekMut always guards a non-empty heap")
    }
}

impl<T: Ord, A: Allocator> DerefMut for PeekMut<'_, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // Performs a leak amplification - prevents the element from
        // being modified followed by PeekMut being forgotten, which
        // breaks invariants without ever being able to trigger the
        // poison flag.
        self.heap.assume_unpoisoned();
        self.leak();
        self.heap
            .data
            .as_mut_slice()
            .first_mut()
            .expect("a PeekMut always guards a non-empty heap")
    }
}

impl<T: Ord, A: Allocator> PeekMut<'_, T, A> {
    /// Re-establishes the heap order.
    ///
    /// # Panics
    ///
    /// Panics if the Ord implementation panics. In this case, the heap will be poisoned.
    fn restore(&mut self) -> bool {
        let mut quarantine = PoisonOnUnwind::new(&mut *self.heap);
        let new_pos = unsafe { quarantine.sift_down(0) };
        // Defuse the quarantine.
        core::mem::forget(quarantine);
        new_pos != 0
    }

    /// Leaks the tail.
    fn leak(&mut self) {
        if self.original_len.is_none() {
            let original_len = self.heap.len();
            unsafe { self.heap.data.set_len(1) };
            self.original_len = Some(original_len);
        }
    }

    /// Unleaks the leaked tail by restoring the length.
    /// The operation must do it first.
    ///
    /// Returns `true` if the leak was in place, implying the
    /// element was modified.
    fn unleak(&mut self) -> bool {
        if let Some(original_len) = self.original_len.take() {
            unsafe { self.heap.data.set_len(original_len) };
            true
        } else {
            false
        }
    }

    /// Removes the guarded element from the heap and returns it.
    ///
    /// The value returned is the one currently at the root, which
    /// includes any mutation made through [`DerefMut`].
    ///
    /// # Panics
    ///
    /// If a comparison panics during the re-sift, the heap is
    /// poisoned; [`rebuild`] recovers it.
    ///
    /// [`rebuild`]: BinaryHeap::rebuild
    #[inline]
    #[must_use]
    pub fn pop(mut this: Self) -> T {
        let modified = this.unleak();
        this.heap.assume_unpoisoned();
        if modified {
            let mut quarantine = PoisonOnUnwind::new(&mut *this.heap);
            // We are interested in the first element.
            // On the happy path, the first element is removed
            // regardless of state, and the swapped element is sifted down.
            // On the panic path, the first element is put back in, and
            // because it has changed, the heap is poisoned.
            let res = quarantine
                .pop()
                .expect("a PeekMut always guards a non-empty heap");
            core::mem::forget(quarantine);
            res
        } else {
            this.heap
                .pop()
                .expect("a PeekMut always guards a non-empty heap")
        }
    }

    /// Refresh the binary heap after an in-place mutation of the top element.
    /// The guard continue to sits at the top of the heap, viewing the largest
    /// element.
    ///
    /// Returns `true` if the element moved, `false` otherwise.
    ///
    /// # Panics
    ///
    /// If a comparison panics, the heap is poisoned.
    #[inline]
    #[must_use]
    pub fn refresh(&mut self) -> bool {
        let modified = self.unleak();
        self.heap.assume_unpoisoned();
        if modified { self.restore() } else { false }
    }
}

impl<T: Ord + fmt::Debug, A: Allocator> fmt::Debug for PeekMut<'_, T, A> {
    /// Formats the guarded element.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.heap.poisoned {
            return f.write_str("PeekMut(<poisoned>)");
        }
        f.debug_tuple("PeekMut")
            .field(
                self.heap
                    .data
                    .as_slice()
                    .first()
                    .expect("a PeekMut always guards a non-empty heap"),
            )
            .finish()
    }
}

impl<T: Ord, A: Allocator> Drop for PeekMut<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        let modified = self.unleak();
        // This prevents panicking twice.
        if self.heap.poisoned {
            return;
        }
        if modified {
            self.restore();
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::PeekMut;
    use crate::collections::binary_heap::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
    use olive_test_helpers::{PanicArmer, PanickyOrd};
    use std::sync::Arc;
    use std::{panic, vec, vec::Vec};

    fn heap_of(items: impl IntoIterator<Item = i32>) -> BinaryHeap<i32> {
        BinaryHeap::try_from_iter(items).expect("allocation ok")
    }

    fn drain_heap(heap: &mut BinaryHeap<i32>) -> Vec<i32> {
        let mut out = Vec::new();
        while let Some(top) = heap.pop() {
            out.push(top);
        }
        out
    }

    #[test]
    fn peek_mut_guards_the_maximum() {
        let mut heap = heap_of([1, 5, 3, 9, 2]);
        let peek = heap.peek_mut();
        assert_eq!(*peek.expect("non-empty heap"), 9);
    }

    #[test]
    fn empty_heap_yields_no_guard() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        assert!(heap.peek_mut().is_none());
    }

    #[test]
    fn guard_dropped_without_mutation_leaves_the_heap_untouched() {
        let mut heap = heap_of([9, 8, 7, 6]);
        {
            let peek = heap.peek_mut().expect("non-empty heap");
            assert_eq!(*peek, 9);
        }
        assert_eq!(heap.peek(), Some(&9));
        assert_eq!(drain_heap(&mut heap), vec![9, 8, 7, 6]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn replaced_root_re_sifts_on_drop() {
        let mut heap = heap_of([1, 2, 3, 4, 5]);
        {
            let mut peek = heap.peek_mut().expect("non-empty heap");
            assert_eq!(*peek, 5);
            *peek = 0;
        }
        // The guard's drop re-sifted the demoted root to the bottom.
        // Popping must now yield every element in descending order, poison-free.
        assert_eq!(drain_heap(&mut heap), vec![4, 3, 2, 1, 0]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn guard_leak_amplification_works() {
        let mut heap = heap_of([1, 2, 3, 4, 5]);
        {
            let mut peek = heap.peek_mut().expect("non-empty heap");
            assert_eq!(*peek, 5);
            *peek = 0;
            // Upon modifying, the guard should leak.
            assert_eq!(peek.heap.len(), 1);
        }
    }

    #[test]
    fn single_element_heap_survives_mutation() {
        let mut heap = heap_of([7]);
        {
            let mut peek = heap.peek_mut().expect("non-empty heap");
            *peek = 1;
        }
        // A mutated one-element root still matches its original length at
        // drop, so the re-sift is skipped - and a heap of one is always a
        // heap.
        assert_eq!(heap.pop(), Some(1));
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn panicking_sift_poisons_and_rebuild_repairs() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> = BinaryHeap::try_from_iter(
            [9, 8, 7].map(|value| PanickyOrd::new(value, Arc::clone(&armer))),
        )
        .expect("allocation ok");
        let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let mut peek = heap.peek_mut().expect("non-empty heap");
            peek.value = 0;
            armer.arm();
            drop(peek);
        }));
        assert!(outcome.is_err());
        assert!(heap.is_poisoned());

        // Rebuilding should work when disarmed.
        armer.disarm();
        heap.rebuild();
        assert!(!heap.is_poisoned());
        let mut popped = Vec::new();
        while let Some(top) = heap.pop() {
            popped.push(top.value);
        }
        assert_eq!(popped, vec![8, 7, 0]);
    }

    #[test]
    fn pop_without_mutation_removes_the_maximum() {
        let mut heap = heap_of([5, 4, 3]);
        let peek = heap.peek_mut().expect("non-empty heap");
        assert_eq!(PeekMut::pop(peek), 5);
        assert_eq!(drain_heap(&mut heap), vec![4, 3]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn pop_after_mutation_removes_the_mutated_root() {
        let mut heap = heap_of([9, 8, 7, 6]);
        let mut peek = heap.peek_mut().expect("non-empty heap");
        *peek += 6;
        assert_eq!(PeekMut::pop(peek), 15);
        assert_eq!(drain_heap(&mut heap), vec![8, 7, 6]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn pop_on_a_single_element_empties_the_heap() {
        let mut heap = heap_of([7]);
        let mut peek = heap.peek_mut().expect("non-empty heap");
        *peek = 1;
        assert_eq!(PeekMut::pop(peek), 1);
        assert!(heap.is_empty());
    }

    #[test]
    fn refresh_of_an_untouched_root_reports_no_movement() {
        let mut heap = heap_of([9, 8, 7, 6]);
        let mut guard = heap.peek_mut().expect("non-empty heap");
        assert!(!guard.refresh(), "the untouched root has nothing to sift");
        assert_eq!(*guard, 9);
        drop(guard);
        assert_eq!(drain_heap(&mut heap), vec![9, 8, 7, 6]);
        assert!(!heap.is_poisoned());
    }

    #[test]
    fn refresh_clamps_every_element_at_the_bound() {
        // std's doc example for `PeekMut::refresh`: keep overwriting the
        // top and re-sifting until nothing moves, which clamps the whole
        // heap at the bound.
        let mut heap = heap_of(0..128);
        let mut guard = heap.peek_mut().expect("non-empty heap");
        while {
            *guard = 99;
            guard.refresh()
        } {}
        assert_eq!(*guard, 99);
        drop(guard);
        assert_eq!(heap.peek(), Some(&99));
        let drained = drain_heap(&mut heap);
        assert!(drained.iter().all(|&value| value <= 99));
        assert_eq!(drained.len(), 128);
    }

    #[test]
    fn panicking_refresh_poisons_and_the_guard_lifts_cleanly() {
        let armer = Arc::new(PanicArmer::new());
        let mut heap: BinaryHeap<PanickyOrd<i32>> = BinaryHeap::try_from_iter(
            [9, 8, 7].map(|value| PanickyOrd::new(value, Arc::clone(&armer))),
        )
        .expect("allocation ok");
        {
            let mut peek = heap.peek_mut().expect("non-empty heap");
            peek.value = 0;
            armer.arm();
            let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| peek.refresh()));
            assert!(outcome.is_err());
            // The guard is still live while poisoned; dropping it must
            // restore the hidden tail without panicking a second time.
        }
        assert!(heap.is_poisoned());

        armer.disarm();
        heap.rebuild();
        let mut values = Vec::new();
        while let Some(top) = heap.pop() {
            values.push(top.value);
        }
        assert_eq!(values, vec![8, 7, 0]);
    }

    #[test]
    fn peek_mut_debug_shows_the_root_or_the_borrow() {
        let mut heap = heap_of([9, 8, 7]);
        let guard = heap.peek_mut().expect("non-empty heap");
        assert_eq!(std::format!("{guard:?}"), "PeekMut(9)");
        let mut guard = guard;
        *guard = 5;
        assert_eq!(std::format!("{guard:?}"), "PeekMut(5)");
    }

    #[test]
    fn poisoned_heap_peek_mut_debug_says_so() {
        let mut heap = heap_of([1]);
        let guard = heap.peek_mut().expect("non-empty heap");
        guard.heap.poisoned = true;
        assert_eq!(std::format!("{guard:?}"), "PeekMut(<poisoned>)");
    }
}
