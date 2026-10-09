//! Mutable access to the maximum, with the re-sift deferred until the guard
//! drops.
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
            orig_len: self.len(),
            heap: self,
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
    /// The length the buffer had at creation. If [`DerefMut`] is called,
    /// the heap's length is set to 1, and this indicates a leak.
    ///
    /// If this field is 1, dropping the peek is trivial.
    orig_len: usize,
}

impl<T: Ord, A: Allocator> Deref for PeekMut<'_, T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // We do nothing - the element is not mutated.
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
        unsafe { self.heap.data.set_len(1) };
        self.heap
            .data
            .as_mut_slice()
            .first_mut()
            .expect("a PeekMut always guards a non-empty heap")
    }
}

impl<T: Ord, A: Allocator> Drop for PeekMut<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        // `DerefMut` is the only operation that may hide the tail, so a buffer
        // still at its original length means the root was never handed out
        // mutably and the invariant never moved. (A one-element heap that
        // *was* mutated also matches here, and skipping the sift is sound:
        // a single root is always a valid heap.)
        if self.heap.len() == self.orig_len {
            return;
        }

        let mut quarantine = PoisonOnUnwind::new(self.heap);
        // SAFETY: the heap originally had the exact length.
        unsafe { quarantine.data.set_len(self.orig_len) };
        // SAFETY: the restored length is at least 1, so the guarded root at
        // index 0 is live.
        unsafe { quarantine.sift_down(0) };
        quarantine.poisoned = false;
        core::mem::forget(quarantine);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::collections::binary_heap::BinaryHeap;
    use crate::test_helpers::{PanicArmer, PanickyOrd};
    use olive_core::try_traits::try_from_iterator::TryFromIterator;
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
}
