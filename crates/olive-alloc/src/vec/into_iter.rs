//! The owned iterator produced by `Vec::into_iter`.
//!
//! Unlike the borrowed iterators, this consumes the vector by value and yields
//! its elements one at a time, moving them out of the buffer.

use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
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
    /// Base address of the original allocation. Kept separately from `ptr` so
    /// we can deallocate the exact block even after `ptr` has advanced.
    start: NonNull<T>,
    /// Pointer to the next element to yield from the front. Advances toward the
    /// end for non-ZSTs; for ZSTs it stays fixed (position lives in `remaining`).
    ptr: NonNull<T>,
    /// Number of elements not yet yielded. This is the single source of truth
    /// for progress, which avoids any pointer arithmetic on the dangling base
    /// pointer used for zero-sized types.
    remaining: usize,
    /// Capacity of the underlying allocation, used to free it by reconstituting a RawVec.
    cap: usize,
    /// Wrapped in `ManuallyDrop` because our `Drop` impl frees the allocation
    /// explicitly; we must not also run the allocator's destructor.
    alloc: ManuallyDrop<A>,
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
    /// `ptr` must be the base of a valid allocation of at least `cap` elements
    /// made with `alloc`, and the first `len` slots must be initialized. For
    /// zero-sized `T`, `ptr` may be a dangling aligned pointer and `cap` may be
    /// an arbitrary value (it is completely ignored)
    #[inline]
    pub(super) unsafe fn new_from_parts(ptr: NonNull<T>, len: usize, cap: usize, alloc: A) -> Self {
        Self {
            start: ptr,
            ptr,
            remaining: len,
            cap,
            alloc: ManuallyDrop::new(alloc),
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements still held by this iterator.
    #[inline]
    pub const fn len(&self) -> usize {
        self.remaining
    }

    /// Returns `true` if all elements have been yielded.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.remaining == 0
    }

    /// Views the remaining elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `[ptr, ptr + remaining)` lies within the initialized region
        // of the buffer. For ZSTs `from_raw_parts` on a dangling aligned
        // pointer is well defined.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.remaining) }
    }

    /// Releases the exact block described by `start`/`cap` using `alloc`.
    ///
    /// SAFETY:
    /// - The block must satisfy safety conditions of [`RawVec::from_nonnull_in`].
    unsafe fn dealloc_block(start: NonNull<T>, cap: usize, alloc: &A) {
        unsafe { drop(RawVec::from_nonnull_in(start, cap, alloc)) }
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.remaining > 0"
        )]
        {
            self.remaining -= 1;
        }
        // For non-ZSTs advance the read pointer *before* reading so that a
        // panicking consumer cannot cause the same slot to be yielded twice.
        // For ZSTs the pointer is left alone (advancing it would be UB on a
        // dangling pointer); position is carried entirely by `remaining`.
        let item = if size_of::<T>() == 0 {
            self.ptr
        } else {
            let old = self.ptr;
            self.ptr = unsafe { old.add(1) };
            old
        };
        // SAFETY: the slot addressed by `item` was part of the initialized
        // region and is now ours to move out. Reading a ZST from its dangling
        // aligned pointer is well defined.
        Some(unsafe { ptr::read(item.as_ptr()) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "asserted self.remaining > 0"
        )]
        {
            self.remaining -= 1;
        }
        let item = if size_of::<T>() == 0 {
            // ZSTs occupy no memory; the front pointer is the only valid one to
            // "read" from (it is dangling), and it never moves.
            self.ptr
        } else {
            // Indexing `ptr` at `old_remaining` is out of bounds (like indexing an array
            // at length), at so the live window is `[ptr, ptr[old_remaining])` or
            // `[ptr, ptr[new_remaining]]`, or new_remaining indicating the tail.
            // SAFETY: `ptr + remaining` is within the allocation and non-null,
            // so wrapping it in a `NonNull` is valid.
            unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(self.remaining)) }
        };
        // SAFETY: as in `next`, the slot is initialized and becomes the caller's.
        Some(unsafe { ptr::read(item.as_ptr()) })
    }
}

/// Owns the raw parts of an [`IntoIter`] being dropped and frees the backing
/// block in its own `Drop`.
///
/// Dropping the unconsumed tail (`ptr::drop_in_place`) can panic. Without this
/// guard a panic would unwind past the deallocation line and leak the block.
/// Arming the guard *before* dropping the tail guarantees the free runs both on
/// the happy path (guard falls out of scope normally) and on unwind (the guard's
/// `Drop` runs during stack teardown). There is exactly one free site. This
/// mirrors the `BoxDeallocGuard` used by [`crate::boxed::Box`].
struct IntoIterDeallocGuard<'a, T, A: Allocator> {
    start: NonNull<T>,
    cap: usize,
    alloc: &'a A,
    _marker: PhantomData<T>,
}

impl<T, A: Allocator> Drop for IntoIterDeallocGuard<'_, T, A> {
    fn drop(&mut self) {
        unsafe { IntoIter::dealloc_block(self.start, self.cap, self.alloc) };
    }
}

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        // Arm the deallocation guard *before* dropping the tail. It will free
        // the block whether or not `drop_in_place` panics. ZSTs have no backing
        // allocation, so the guard becomes a no-op for them.
        let _dealloc_guard = IntoIterDeallocGuard {
            start: self.start,
            cap: self.cap,
            alloc: &*self.alloc,
            _marker: PhantomData,
        };
        // Destroy only the unconsumed range `[ptr, ptr + remaining)`. Everything
        // before `ptr` has already been moved out by `next`/`next_back` and
        // belongs to the caller, so it must not be dropped again.
        // SAFETY: `[ptr, ptr + remaining)` is within the initialized region of
        // the buffer; for ZSTs this is a no-op drop over a dangling pointer.
        unsafe {
            ptr::drop_in_place(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.remaining));
        }
    }
}
