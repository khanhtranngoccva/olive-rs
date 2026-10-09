//! Quarantine for heaps that a panicking [`Ord`](core::cmp::Ord) comparison
//! left out of order.

use super::BinaryHeap;
use crate::alloc::Allocator;
use core::ops::{Deref, DerefMut};

/// Message carried by the panic that order-trusting operations raise on a
/// poisoned heap.
pub(super) const POISON_MSG: &str = "BinaryHeap is poisoned: a panicking Ord comparison left the heap invariant unestablished; call BinaryHeap::rebuild to recover or BinaryHeap::clear to discard";
impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Whether a panicking comparison left the heap invariant unestablished.
    ///
    /// A poisoned heap still owns exactly its elements (nothing is lost or
    /// duplicated) and can be destroyed, drained through [`clear`], or
    /// recovered through [`rebuild`](BinaryHeap::rebuild).
    ///
    /// [`clear`]: BinaryHeap::clear
    #[inline]
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Panic if the heap is poisoned: helper for operations whose
    /// only failure mode is trusting an invariant that no longer holds.
    pub(super) fn assume_unpoisoned(&self) {
        if self.poisoned {
            panic!("{POISON_MSG}");
        }
    }
}

/// Guard that poisons the borrowed heap if the region it wraps unwinds.
///
/// The guard acts in the poisoning direction only: its drop path can set
/// the flag, and nothing the guard does clears it. Lifting a quarantine is
/// an act of restored trust, so it is confined to the two operations that
/// re-establish or discard the data — [`rebuild`](BinaryHeap::rebuild) and
/// [`clear`](BinaryHeap::clear) — and those write the field directly.
///
/// The guard is handed out by whichever operation opens the critical
/// section semantically (for the merge shapes,
/// [`try_append_fast`](BinaryHeap::try_append_fast) arms it as soon as
/// foreign elements enter the buffer) and dereferences to the heap for the
/// duration of the guarded region. Holding a real `&mut` is what keeps any
/// other borrow of the heap from existing; this is not a synchronization
/// structure, so there is no raw pointer and no atomic involved.
pub(super) struct PoisonOnUnwind<'a, T, A: Allocator> {
    heap: &'a mut BinaryHeap<T, A>,
    armed: bool,
}

impl<'a, T, A: Allocator> PoisonOnUnwind<'a, T, A> {
    /// Arm the guard over `heap`: unless [`defuse`](Self::defuse) runs
    /// first, dropping this guard poisons the heap.
    #[inline]
    pub(super) const fn new(heap: &'a mut BinaryHeap<T, A>) -> Self {
        Self { heap, armed: true }
    }

    /// Mark the guarded region as completed: the drop becomes a no-op.
    /// It deliberately does not touch the poison flag — see the
    /// one-direction rule on the type.
    #[inline]
    pub(super) fn defuse(&mut self) {
        self.armed = false;
    }
}

impl<T, A: Allocator> Deref for PoisonOnUnwind<'_, T, A> {
    type Target = BinaryHeap<T, A>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.heap
    }
}

impl<T, A: Allocator> DerefMut for PoisonOnUnwind<'_, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.heap
    }
}

impl<T, A: Allocator> Drop for PoisonOnUnwind<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            self.heap.poisoned = true;
        }
    }
}
