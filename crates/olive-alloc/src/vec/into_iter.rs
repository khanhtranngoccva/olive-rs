//! The owned iterator produced by [`Vec::into_iter`](super::Vec::into_iter).
//!
//! Unlike the borrowed iterators, this consumes the vector by value and yields
//! its elements one at a time, moving them out of the buffer. It owns the
//! backing allocation directly (a raw pointer plus capacity and allocator)
//! rather than embedding a whole [`Vec`](super::Vec); that keeps the drop path
//! trivially correct — there is no nested `Vec` whose own destructor could
//! re-drop elements that have already been handed to the caller.
//!
//! # Zero-sized types
//!
//! For ZSTs the underlying "allocation" is a well-aligned dangling pointer with
//! no provenance, so we must never perform in-bounds pointer arithmetic on it.
//! Position is therefore tracked with an explicit element count (`remaining`)
//! instead of pointer offsets; the pointer itself is only ever dereferenced
//! (which is fine for a ZST) and never advanced.

use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr::{self, NonNull};
use core::slice;

use crate::alloc::{Allocator, Global};

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
    /// Pointer to the next element to yield. Advances toward the end for
    /// non-ZSTs; for ZSTs it stays fixed (position lives in `remaining`).
    ptr: NonNull<T>,
    /// Number of elements not yet yielded. This is the single source of truth
    /// for progress, which avoids any pointer arithmetic on the dangling base
    /// pointer used for zero-sized types.
    remaining: usize,
    /// Capacity of the underlying allocation, used only to free it.
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
    /// `usize::MAX`; both are handled.
    #[inline]
    pub(super) unsafe fn new_from_parts(ptr: NonNull<T>, len: usize, cap: usize, alloc: A) -> Self {
        Self {
            start: ptr,
            ptr,
            remaining: len,
            // For ZSTs the reported capacity is `usize::MAX`, which cannot be
            // stored in a `RawVec` (caps must be `<= isize::MAX`). We never
            // actually deallocate for ZSTs anyway, so normalize it to zero here
            // to keep the field sane if it is ever inspected.
            cap: if size_of::<T>() == 0 { 0 } else { cap },
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

    /// Frees the backing allocation without dropping any remaining elements.
    ///
    /// Used by `Drop` once the tail has been destroyed.
    #[inline]
    unsafe fn dealloc_only(&mut self) {
        // Zero-sized types never allocate: their pointer is dangling and there
        // is nothing to free.
        if size_of::<T>() == 0 {
            return;
        }
        // SAFETY: the caller guarantees this iterator will not be dropped or
        // used again, so taking the allocator out is sound.
        let alloc = unsafe { ManuallyDrop::take(&mut self.alloc) };
        // Reconstruct just enough of a `RawVec` to release the block. Dropping
        // the empty `RawVec` frees the allocation without touching any element
        // (none remain at this point).
        let raw = unsafe { crate::raw_vec::RawVec::from_nonnull_in(self.start, self.cap, alloc) };
        drop(raw);
    }
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
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
        self.remaining -= 1;
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

impl<T, A: Allocator> Drop for IntoIter<T, A> {
    // FIXME: dealloc_only needs an unconditional panic-aware guard to ensure unconditional execution, 
    // similar to Box
    fn drop(&mut self) {
        // Destroy only the unconsumed range `[ptr, ptr + remaining)`. Everything
        // before `ptr` has already been moved out by `next`/`next_back` and
        // belongs to the caller, so it must not be dropped again.
        //
        // The allocation is freed explicitly via `dealloc_only`.
        // SAFETY: `[ptr, ptr + remaining)` is within the initialized region of
        // the buffer; for ZSTs this is a no-op drop over a dangling pointer.
        unsafe {
            ptr::drop_in_place(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.remaining));
            self.dealloc_only();
        }
    }
}
