//! The owned iterator produced by `Vec::into_iter`.
//!
//! Unlike the borrowed iterators, this consumes the vector by value and yields
//! its elements one at a time, moving them out of the buffer.

use core::fmt;
use core::fmt::Debug;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::slice;

use crate::alloc::{Allocator, Global};
use crate::raw_vec::RawVec;

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the [`into_iter`](super::Vec::into_iter) method
/// on [`Vec`](super::Vec) (provided by the [`IntoIterator`] trait).
///
/// # Example
///
/// ```
/// use olive_alloc::vec::Vec;
/// let v = {
///     let mut v = Vec::new();
///     for i in 0..3 { v.try_push(i).unwrap(); }
///     v
/// };
/// let collected: std::vec::Vec<i32> = v.into_iter().collect();
/// assert_eq!(collected, std::vec![0, 1, 2]);
/// ```
pub struct IntoIter<T, A: Allocator = Global> {
    /// Owns the backing allocation. Its `Drop` frees the block and drops the
    /// allocator. We read element pointers from it lazily (via `ptr()`) but
    /// never advance a stored head pointer.
    raw: RawVec<T, A>,
    /// Number of elements already yielded from the front (via `next`). Together
    /// with `taken_back` this partitions the buffer into three regions:
    /// `[consumed, len - taken_back)` is still live.
    consumed: usize,
    /// Number of elements already yielded from the back (via `next_back`).
    taken_back: usize,
    /// Total number of elements originally held (`len` at construction).
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: mirroring `Vec`, the iterator never hands out references that
// outlive the buffer, and moving it moves the whole allocation. Sound iff `T`
// and `A` are `Send`/`Sync`.
unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for IntoIter<T, A> {}

impl<T, A: Allocator> IntoIter<T, A> {
    /// Constructs an iterator from the raw parts of a consumed `Vec`.
    ///
    /// # Safety
    ///
    /// `start` must be the base of a valid allocation of at least `cap`
    /// elements made with `alloc`, and the first `len` slots must be
    /// initialized. For zero-sized `T`, `start` may be a dangling aligned
    /// pointer and `cap` may be an arbitrary value (it is completely ignored).
    #[inline]
    pub(super) const unsafe fn new_from_parts(
        start: NonNull<T>,
        len: usize,
        cap: usize,
        alloc: A,
    ) -> Self {
        // SAFETY: preconditions passed through from the caller.
        Self {
            raw: unsafe { RawVec::from_nonnull_in(start, cap, alloc) },
            consumed: 0,
            taken_back: 0,
            len,
            _marker: PhantomData,
        }
    }

    /// Number of elements still held in the live window.
    #[inline]
    const fn remaining(&self) -> usize {
        // `consumed + taken_back` can never exceed `len`: each yield shrinks the
        // live window by exactly one and stops at zero.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "consumed + taken_back <= len"
        )]
        let taken = { self.consumed + self.taken_back };
        #[allow(clippy::arithmetic_side_effects, reason = "taken <= len")]
        {
            self.len - taken
        }
    }

    /// Returns the number of elements still held by this iterator.
    #[inline]
    pub const fn len(&self) -> usize {
        self.remaining()
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Views the remaining elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // The computed front pointer is invalid only when it lands one-past-the-
        // end of the buffer: `front_offset() == len`. When that case happens,
        // the fallback must be used. This fallback behavior is triggered
        // sparingly for conformance with std. (this also happens to work with ZSTs)
        if self.front_offset() == self.len {
            return unsafe { slice::from_raw_parts(self.raw.ptr(), 0) };
        }
        // SAFETY: `front_offset() < len`; the front of the live window addresses a
        // real, initialized slot (or a valid dangling-aligned ZST address),
        // and `[front, front + rem)` lies entirely within the initialized region
        // of the buffer.
        let rem = self.remaining();
        unsafe {
            let front = self.raw.ptr().add(self.front_offset());
            slice::from_raw_parts(front, rem)
        }
    }

    /// Element offset (from the buffer base) of the current front of the live
    /// window.
    #[inline]
    fn front_offset(&self) -> usize {
        self.consumed
    }

    /// Element offset (from the buffer base) just past the current back of the
    /// live window (the last live element is at `back_offset() - 1`).
    #[inline]
    fn back_offset(&self) -> usize {
        #[allow(clippy::arithmetic_side_effects, reason = "taken_back <= len")]
        {
            self.len - self.taken_back
        }
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // Record the front offset *before* advancing; it addresses a live,
        // initialized slot because we just confirmed `remaining > 0`.
        let offset = self.front_offset();
        #[allow(clippy::arithmetic_side_effects, reason = "consumed < len")]
        {
            self.consumed += 1;
        }
        // SAFETY: `offset` is within the initialized region and the value is now
        // ours to move out. Reading a ZST from its (possibly dangling-aligned)
        // pointer is well defined.
        Some(unsafe { ptr::read(self.raw.ptr().add(offset)) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining(), Some(self.remaining()))
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.remaining() == 0 {
            return None;
        }
        // The last live element sits just below the current back boundary.
        // Compute its offset *before* recording the take so it addresses a
        // live, initialized slot.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "remaining > 0 implies back_offset >= 1"
        )]
        let offset = { self.back_offset() - 1 };
        #[allow(clippy::arithmetic_side_effects, reason = "taken_back < len")]
        {
            self.taken_back += 1;
        }
        // SAFETY: as in `next`, `offset` is within the initialized region and
        // the value becomes the caller's.
        Some(unsafe { ptr::read(self.raw.ptr().add(offset)) })
    }
}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        // Destroy the unconsumed tail.
        // SAFETY: the addressed slots lie within the initialized region and hold
        // valid values; for ZSTs `drop_in_place` performs no memory access but
        // still runs any `Drop` glue. To avoid OOB overflow, only drop when rem > 0
        let rem = self.remaining();
        if rem > 0 {
            unsafe {
                let front = self.raw.ptr().add(self.consumed);
                ptr::drop_in_place(slice::from_raw_parts_mut(front, rem));
            }
        }
        // The owned `RawVec` field is dropped automatically after this body
        // returns, freeing the backing block and dropping the allocator. No
        // explicit free call or guard is needed: `RawVec::drop` handles both.
    }
}

impl<T: Debug, A: Allocator> Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debugger = f.debug_list();
        if self.remaining() > 0 {
            debugger.entries(self.as_slice());
        }
        debugger.finish()
    }
}
