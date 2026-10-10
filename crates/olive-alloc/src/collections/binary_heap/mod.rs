//! A binary max-heap - a complete binary tree packed into a vector.
mod allocation;
mod construction;
mod drain;
mod iter;
pub(super) mod mutation;
mod peek;
mod poison;
mod query;
mod sifting;
mod traits;

use crate::alloc::Allocator;
use crate::vec::TryVecWithCloneError;
use crate::vec::Vec;
use core::fmt;
pub use drain::Drain;
pub use iter::{IntoIter, Iter};
use olive_core::alloc_errors::TryReserveError;
use olive_core::try_traits::try_clone::TryCloneError;
pub use peek::PeekMut;

/// A max-heap binary tree stored in a growable [`Vec`].
///
/// # Fallibility
///
/// Every operation that may grow or shrink the buffer comes with fallibility
/// ([`try_with_capacity_in`], [`try_push`], [`try_push_give_back`],
/// [`try_append`]). These operations report
/// an [`olive_core::alloc_errors::TryReserveError`] on failure instead
/// of aborting.
///
/// A failed growth never touches the heap, for capacity is secured before
/// any element is consumed, so the heap keeps the exact contents (and
/// contents-derived ordering) it had before the failed call.
///
/// # Atomicity
///
/// There are a few atomicity aspects for the BinaryHeap, and they usually
/// cannot all be satisfied at once.
///
/// - Atomicity on errors - demonstrated on normal errors during algorithm execution.
/// - Atomicity on panics - demonstrated on a panic.
/// - Data atomicity - whether all elements is pushed into the heap in an atomic manner.
///   This aspect is irrelevant to tree invariants - a panic may break invariant and poison
///   the heap, but appended elements may not be lost.
/// - Invariant atomicity - whether the binary tree invariants for the heap are satisfied.
///   This is applicable only on panics. When invariants break, the heap will be poisoned.
///
/// For example:
/// - [`pop`] is fully-atomic: its sifting plan is computed before the sift initiates,
///   and `PopRollback` restores the heap bit-for-bit, so a panic in the middle of a pop
///   breaks neither data nor invariant and poisons nothing.
/// - [`try_append`] offers both data and invariant atomicity on errors. On a
///   panic, it loses no data: a panicking [`Ord`] implementation leaves every
///   element inside `self` but breaks the invariant, so `self` ends up poisoned.
/// - [`try_push`] and [`try_push_give_back`] also offer data and invariant
///   atomicity on errors. On a panic, the pushed element stays in the heap
///   (data atomicity holds) and the heap is poisoned (invariant atomicity
///   breaks).
///
/// It is recommended to focus on error and data atomicity and not panic atomicity -
/// panic denotes an unrecoverable error and should warrant a fix.
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
/// [`pop`]: BinaryHeap::pop
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

/// Error type for binary-heap operations that can fail either for lack of
/// memory or for a failed element clone.
///
/// Covers `try_extend` over borrowed elements and `try_extend_from_slice`
/// — or any operation whose failure modes are limited to a capacity
/// reservation ([`TryReserveError`]) or an element clone failure
/// ([`TryCloneError`]).
#[derive(Clone, PartialEq, Eq)]
pub enum TryBinaryHeapWithCloneError {
    /// A capacity reservation on the backing vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires `TryClone`.
    Clone(TryCloneError),
}

impl fmt::Debug for TryBinaryHeapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryBinaryHeapWithCloneError::Reserve")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryBinaryHeapWithCloneError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryBinaryHeapWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "binary heap operation failed: {e}"),
            Self::Clone(e) => write!(f, "binary heap operation failed: {e}"),
        }
    }
}

impl core::error::Error for TryBinaryHeapWithCloneError {}

impl From<TryReserveError> for TryBinaryHeapWithCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryBinaryHeapWithCloneError {
    #[inline]
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

impl From<TryVecWithCloneError> for TryBinaryHeapWithCloneError {
    #[inline]
    fn from(err: TryVecWithCloneError) -> Self {
        match err {
            TryVecWithCloneError::Reserve(e) => Self::Reserve(e),
            TryVecWithCloneError::Clone(e) => Self::Clone(e),
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
