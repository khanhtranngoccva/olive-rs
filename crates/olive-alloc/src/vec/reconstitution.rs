use core::{
    mem::ManuallyDrop,
    ptr::{self, NonNull},
};

use olive_core::alloc::Allocator;

use super::Vec;
use crate::{alloc::Global, raw_vec::RawVec};

// ---------------------------------------------------------------------------
// Reconstitution — global allocator
// ---------------------------------------------------------------------------

impl<T> Vec<T, Global> {
    /// Creates a `Vec<T>` directly from a pointer, a length, and a capacity.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// - If T is not a zero-sized type and the capacity is nonzero, ptr must have been allocated using the global allocator.
    ///   If T is a zero-sized type or the capacity is zero, ptr need only be non-null and aligned.
    /// - T needs to have the same alignment as what ptr was allocated with, if the pointer is required to be allocated.
    ///   (T having a less strict alignment is not sufficient, the alignment really needs to be equal to satisfy the dealloc requirement that memory must be allocated and deallocated with the same layout.)
    /// - The size of T times the `capacity` (i.e. the allocated size in bytes), if nonzero, needs to be the same size as the pointer was allocated with.
    ///   (Because similar to alignment, dealloc must be called with the same layout size.)
    /// - `length` needs to be less than or equal to `capacity`.
    /// - The first `length` values must be properly initialized values of type T.
    /// - `capacity` needs to be the capacity that the pointer was allocated with, if the pointer is required to be allocated.
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of `ptr.offset()`.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_raw_parts requires that length <= capacity"
        );
        unsafe { Self::from_raw_parts_in(ptr, length, capacity, Global) }
    }

    /// Creates a `Vec<T>` from a non-null pointer, length, and capacity.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::from_raw_parts`], except `ptr` is already
    /// known to be non-null.
    #[inline]
    pub unsafe fn from_parts(ptr: NonNull<T>, length: usize, capacity: usize) -> Self {
        unsafe { Self::from_parts_in(ptr, length, capacity, Global) }
    }

    /// Decomposes a `Vec<T>` into its constituent parts: a raw pointer, a
    /// length, and a capacity.
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Vec`. Most often, one does this by converting
    /// the raw pointer, length, and capacity back into a `Vec` with the
    /// [`from_raw_parts`] function.
    ///
    /// The returned pointer is non-null if and only if the capacity is non-zero.
    ///
    /// [`from_raw_parts`]: Self::from_raw_parts
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        // Mirrors std's `Vec::into_raw_parts`: wrap in `ManuallyDrop` so that a
        // panic during unwinding cannot cause the implicit drop at scope end to
        // free the buffer after we have already handed ownership to the caller.
        let mut this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        // We still want to extract the allocator for consistency even if it is a ZST.
        let _alloc = unsafe { ptr::read(&this.allocator()) };
        (this.as_mut_ptr(), this.len(), this.capacity())
    }

    /// Decomposes a `Vec<T>` into its constituent parts: a non-null pointer, a
    /// length, and a capacity.
    ///
    /// This is the [`NonNull`] counterpart of [`Self::into_raw_parts`].
    ///
    /// After calling this function, the caller is responsible for the memory
    /// previously managed by the `Vec`. Most often, one does this by converting
    /// the raw pointer, length, and capacity back into a `Vec` with the
    /// [`from_parts`] function.
    ///
    /// [`from_parts`]: Self::from_parts
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_parts(self) -> (NonNull<T>, usize, usize) {
        // Mirrors std's `Vec::into_parts`: wrap in `ManuallyDrop` so that a
        // panic during unwinding cannot cause the implicit drop at scope end to
        // free the buffer after we have already handed ownership to the caller.
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller, who
        // takes over ownership of the allocation.
        // We still want to extract the allocator for consistency even if it is a ZST.
        let _alloc = unsafe { ptr::read(&this.allocator()) };
        (this.raw.non_null(), this.len(), this.capacity())
    }
}

// ---------------------------------------------------------------------------
// Reconstitution — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Creates a `Vec<T>` directly from a pointer, a length, a capacity, and an
    /// allocator.
    ///
    /// # Safety
    ///
    /// - If T is not a zero-sized type and the capacity is nonzero, ptr must have been allocated using the `alloc`.
    ///   If T is a zero-sized type or the capacity is zero, ptr need only be non-null and aligned.
    /// - T needs to have the same alignment as what ptr was allocated with, if the pointer is required to be allocated.
    ///   (T having a less strict alignment is not sufficient, the alignment really needs to be equal to satisfy the dealloc
    ///   requirement that memory must be allocated and deallocated with the same layout.)
    /// - The size of T times the `capacity` (i.e. the allocated size in bytes), if nonzero, needs to be the same size as
    ///   the pointer was allocated with. (Because similar to alignment, dealloc must be called with the same layout size.)
    /// - `length` needs to be less than or equal to `capacity`.
    /// - The first `length` values must be properly initialized values of type T.
    /// - `capacity` needs to be the capacity that the pointer was allocated with, if the pointer is required to be allocated.
    /// - The allocated size in bytes must be no larger than `isize::MAX`. See the safety documentation of `ptr.offset()`.
    #[inline]
    pub unsafe fn from_raw_parts_in(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_raw_parts_in requires that length <= capacity"
        );
        // SAFETY: preconditions passed to the caller.
        let raw = unsafe { RawVec::from_raw_parts_in(ptr, capacity, alloc) };
        Self { raw, len: length }
    }

    /// Creates a `Vec<T>` from a non-null pointer, length, capacity, and
    /// allocator.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts_in`] for the full list of invariants.
    #[inline]
    pub unsafe fn from_parts_in(ptr: NonNull<T>, length: usize, capacity: usize, alloc: A) -> Self {
        debug_assert!(
            length <= capacity,
            "Vec::from_parts_in requires that length <= capacity"
        );
        // SAFETY: preconditions passed to the caller.
        let raw = unsafe { RawVec::from_nonnull_in(ptr, capacity, alloc) };
        Self { raw, len: length }
    }

    /// Decomposes a `Vec<T, A>` into its constituent parts: a raw pointer, a
    /// length, a capacity, and the allocator.
    ///
    /// After calling this function, the caller assumes responsibility for the
    /// allocation: it must eventually deallocate `ptr` with `alloc`.
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw_parts_with_allocator(self) -> (*mut T, usize, usize, A) {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller.
        unsafe {
            (
                this.as_mut_ptr(),
                this.len(),
                this.capacity(),
                ptr::read(this.raw.allocator()),
            )
        }
    }

    /// Decomposes a `Vec<T, A>` into a non-null pointer, length, capacity, and
    /// allocator.
    ///
    /// See [`Self::into_raw_parts_with_alloc`].
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_parts_with_allocator(self) -> (NonNull<T>, usize, usize, A) {
        let this = ManuallyDrop::new(self);
        // SAFETY: we consume `self`; the pointer is handed to the caller.
        unsafe {
            (
                this.raw.non_null(),
                this.len,
                this.capacity(),
                ptr::read(this.allocator()),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use core::slice;

    use super::*;

    #[test]
    fn into_raw_parts_roundtrip() {
        let v = {
            let mut v = Vec::new();
            for i in 0..4 {
                v.try_push(i * 10).unwrap();
            }
            v
        };
        let (base, len, cap) = v.into_raw_parts();
        assert_eq!(len, 4);
        assert!(cap >= 4);
        assert_eq!(
            unsafe { slice::from_raw_parts(base, len) },
            &[10, 20, 30, 40]
        );
        let rebuilt = unsafe { Vec::<i32>::from_raw_parts(base, len, cap) };
        assert_eq!(rebuilt, [10, 20, 30, 40]);
        assert_eq!(rebuilt.len(), len);
        assert_eq!(rebuilt.capacity(), cap);
    }

    #[test]
    fn into_raw_parts_with_allocator_roundtrip() {
        // FIXME: use CountingAllocator
        let v = {
            let mut v = Vec::new();
            for i in 0..4 {
                v.try_push(i * 10).unwrap();
            }
            v
        };
        let (base, len, cap, alloc) = v.into_raw_parts_with_allocator();
        assert_eq!(len, 4);
        assert!(cap >= 4);
        assert_eq!(
            unsafe { slice::from_raw_parts(base, len) },
            &[10, 20, 30, 40]
        );
        let rebuilt = unsafe { Vec::<i32>::from_raw_parts_in(base, len, cap, alloc) };
        assert_eq!(rebuilt, [10, 20, 30, 40]);
        assert_eq!(rebuilt.len(), len);
        assert_eq!(rebuilt.capacity(), cap);
    }

    // FIXME: add NonNull test counterparts
}
