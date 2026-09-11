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

mod query;
mod wrapped_index;

use crate::alloc::{Allocator, Global};
use crate::collections::vec_deque::wrapped_index::WrappedIndex;
use crate::raw_vec::RawVec;
use core::ptr;

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
        todo!();
        let cap = self.buf.capacity();
        if cap == 0 || self.len == 0 {
            return;
        }

        // The live elements occupy `self.len` consecutive slots starting at
        // physical index `self.head`, wrapping around the buffer boundary.
        // There are at most two contiguous regions to drop:
        //   Region 1: [head .. min(head + len, cap))
        //   Region 2 (if wrapped): [0 .. (head + len) - cap)
        //
        // Arithmetic safety: `head < cap` and `len <= cap`, so `head + len < 2*cap`.
        // For any realistic allocation `2*cap < usize::MAX`, so no overflow.
        // All subtractions are ordered (larger minus smaller).
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "head < cap, len <= cap ⇒ head+len < 2*cap < usize::MAX; subtractions are ordered"
        )]
        unsafe {
            let head = self.head.as_index();
            let end = head + self.len;

            if end <= cap {
                // Non-wrapped: single contiguous region [head..end).
                let count = end - head;
                let start = self.buf.ptr().add(head);
                let slice = ptr::slice_from_raw_parts_mut(start, count);
                ptr::drop_in_place(slice);
            } else {
                // Wrapped: two regions [head..cap) and [0..end-cap).
                let first_count = cap - head;
                let first_start = self.buf.ptr().add(head);
                let first_slice = ptr::slice_from_raw_parts_mut(first_start, first_count);
                ptr::drop_in_place(first_slice);

                let second_count = end - cap;
                let second_slice = ptr::slice_from_raw_parts_mut(self.buf.ptr(), second_count);
                ptr::drop_in_place(second_slice);
            }
        }
        // The buffer itself is freed by `RawVec`'s `Drop` immediately after
        // this method returns.
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

// SAFETY: `VecDeque` never hands out references that outlive the buffer, and
// moving a `VecDeque` moves its whole allocation. Sound iff `T` itself is
// `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for VecDeque<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for VecDeque<T, A> {}
