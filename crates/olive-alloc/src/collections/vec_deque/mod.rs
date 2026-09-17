//! A fully-fallible port of the standard library's `VecDeque`.
//!
//! Compared with the std original, three things differ:
//!
//! * Every operation that can grow the buffer returns a [`Result`] carrying an
//!   error instead of panicking on out-of-memory.
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Element cloning uses the fallible [`TryClone`](olive_core::try_traits::try_clone::TryClone)
//!   trait throughout.

mod allocation;
mod construction;
mod drain;
mod into_iter;
mod iter;
mod mutation;
mod query;
mod traits;
mod wrapped_index;

use crate::alloc::{Allocator, Global};
use crate::collections::vec_deque::wrapped_index::WrappedIndex;
use crate::raw_vec::RawVec;

pub use allocation::{TryVecDequeInsertError, TryVecDequeWithClosureError};
pub use drain::Drain;
pub use iter::{Iter, IterMut};
pub use mutation::{
    TryVecDequeInsertWithinCapacityError, TryVecDequePushWithinCapacityError,
    TryVecDequeRemoveError, TryVecDequeSwapError,
};
pub use traits::TryVecDequeWithCloneError;

// ---------------------------------------------------------------------------
// VecDeque
// ---------------------------------------------------------------------------

/// A double-ended queue backed by a circular heap buffer.
///
/// Elements are stored in a single contiguous allocation interpreted as a
/// circular buffer. Logical index 0 corresponds to physical slot `head`;
/// logical index `n` corresponds to physical slot `(head + n) % capacity`.
pub struct VecDeque<T, A: Allocator = Global> {
    buf: RawVec<T, A>,
    /// Physical index of the first (front) element.
    head: WrappedIndex,
    /// Number of live elements. Invariant: `len <= capacity`.
    len: usize,
}

// ---------------------------------------------------------------------------
// Drop
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Drop for VecDeque<T, A> {
    fn drop(&mut self) {
        /// Runs the destructor for all items in the slice when it gets dropped (normally or
        /// during unwinding).
        struct Dropper<'a, T>(&'a mut [T]);

        // SAFETY: `Dropper` only ever calls `drop_in_place` on its held slice;
        // it never stores or returns the reference, so the lifetime is purely
        // a borrow-duration marker and can be erased from the type signature.
        #[allow(
            clippy::needless_lifetimes,
            reason = "lifetime is a borrow-duration marker"
        )]
        impl<'a, T> Drop for Dropper<'a, T> {
            fn drop(&mut self) {
                unsafe {
                    core::ptr::drop_in_place(self.0);
                }
            }
        }

        let (front, back) = self.as_mut_slices();
        unsafe {
            let _back_dropper = Dropper(back);
            // use drop for [T]
            core::ptr::drop_in_place(front);
        }
        // RawVec handles deallocation
    }
}
