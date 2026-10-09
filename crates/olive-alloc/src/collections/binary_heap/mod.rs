//! A binary max-heap - a complete binary tree packed into a vector.
mod allocation;
mod construction;
mod iter;
mod mutation;
mod poison;
mod query;
mod sifting;
mod traits;

use crate::alloc::Allocator;
use crate::vec::Vec;
pub use iter::{IntoIter, Iter};

/// A max-heap binary tree stored in a growable [`Vec`].
///
/// # Fallibility
///
/// Every operation that may grow or shrink the buffer comes with fallibility
/// ([`try_with_capacity_in`], [`try_push`], [`try_push_give_back`],
/// [`try_append`], [`try_append_fast`]). These operations report
/// an [`olive_core::alloc_errors::TryReserveError`] on failure instead
/// of aborting.
///
/// A failed growth never touches the heap, for capacity is secured before
/// any element is consumed, so the heap keeps the exact contents (and
/// contents-derived ordering) it had before the failed call.
///
/// # Atomicity
///
/// There are a few atomicity aspects for the BinaryHeap. It is usually not possible to
/// satisfy all of them.
///
/// - Atomicity on errors - demonstrated on normal errors during algorithm execution.
/// - Atomicity on panics - demonstrated on a panic.
/// - Data atomicity - whether all elements is pushed into the heap in an atomic manner.
///   This aspect is irrelevant to tree invariants - a panic may break invariant and poison
///   the heap, but appended elements may not be lost.
/// - Invariant atomicity - whether the binary tree invariants for the heap are satisfied.
///   This is applicable only on panics.
///
/// For example:
/// - [`try_append`] and [`try_append_fast`] offer both data and invariant atomicity
///   on errors. On panics, [`try_append_fast`] satisfies data atomicity but not invariant
///   atomicity (if the [`Ord`] implementation panics, the data stays but the invariant breaks),
///   while [`try_append`] satisfies invariant atomicity, but data atomicity breaks during a
///   panic.
/// - [`try_push`], [`try_push_give_back`] also offer data and invariant atomicity
///   on errors. However, on a panic, both of them satisfy invariant atomicity, but
///   not data atomicity.
///
/// It is recommended to focus on error atomicity and not panic atomicity - panic
/// denotes an unrecoverable error and should warrant a fix.
///
/// These quirks are being worked through and are subject to change.
///
/// # Panic safety and poisoning
///
/// [`Ord`] is user code and may panic mid-comparison. In most cases, this will
/// lead to the heap being poisoned. Call [`rebuild`] to reconstruct the heap
/// to restore its invariants.
///
/// Since the Ord implementation should not panic, it is best to avoid logic bugs
/// in the first place - these issues are usually not recoverable.
///
/// [`try_with_capacity_in`]: BinaryHeap::try_with_capacity_in
/// [`try_push`]: BinaryHeap::try_push
/// [`try_push_give_back`]: BinaryHeap::try_push_give_back
/// [`try_append`]: BinaryHeap::try_append
/// [`try_append_fast`]: BinaryHeap::try_append_fast
/// [`rebuild`]: BinaryHeap::rebuild
pub struct BinaryHeap<T, A: Allocator = crate::alloc::Global> {
    data: Vec<T, A>,
    poisoned: bool,
}

impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Wraps a vector as a heap *without* re-establishing order.
    ///
    /// # Safety
    ///
    /// - The provided payload must satisfy the binary heap invariants,
    ///   or [`rebuild`] must be called afterwards. On failure, the value
    ///   must be discarded.
    ///
    /// [`rebuild`]: Self::rebuild
    pub(super) const unsafe fn with_data(data: Vec<T, A>) -> Self {
        Self {
            data,
            poisoned: false,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::BinaryHeap;
    use crate::test_helpers::PanickyOrd;
    use std::panic;

    // UnwindSafe and RefUnwindSafe assertions
    const _: fn() = || {
        fn assert_unwind_safe<T: panic::UnwindSafe + panic::RefUnwindSafe>() {}
        assert_unwind_safe::<BinaryHeap<PanickyOrd<i32>>>();
    };
}
