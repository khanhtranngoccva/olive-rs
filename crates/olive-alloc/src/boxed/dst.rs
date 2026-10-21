use super::Box;
use core::mem::{self, MaybeUninit};
use core::ptr::NonNull;

use crate::alloc::{AllocError, Allocator, Global, Layout};
use crate::raw_vec::RawVec;
use olive_core::alloc::LayoutExt;
use olive_core::ptr::PointerExt;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError, TryCloneToUninit};

// ---------------------------------------------------------------------------
// Clone constructors (unsized T — slices, str, dyn)
// ---------------------------------------------------------------------------

impl<T: ?Sized + TryCloneToUninit> Box<T, Global> {
    /// Clones a `&T` into a freshly allocated `Box<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation or any element clone fails.
    #[inline]
    pub fn try_clone_from_ref(t: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(t, Global)
    }
}

impl<T: ?Sized + TryCloneToUninit, A: Allocator> Box<T, A> {
    /// Clones a `&T` into a freshly allocated `Box<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation or any element clone fails.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        struct DeallocDropGuard<'a, A: Allocator>(Layout, &'a A, NonNull<u8>);
        impl<A: Allocator> Drop for DeallocDropGuard<'_, A> {
            fn drop(&mut self) {
                let &mut DeallocDropGuard(layout, alloc, ptr) = self;
                // Safety: `ptr` was allocated by `*alloc` with layout `layout`
                unsafe {
                    alloc.deallocate(ptr, layout);
                }
            }
        }
        let layout = Layout::for_value::<T>(src);
        let (ptr, guard) = if layout.size() == 0 {
            (layout.dangling_pointer(), None)
        } else {
            // Safety: layout is non-zero-sized
            let ptr = alloc.allocate(layout)?.cast();
            (ptr, Some(DeallocDropGuard(layout, &alloc, ptr)))
        };
        let ptr = ptr.as_ptr();
        // Safety: `*ptr` is newly allocated, correctly aligned to `align_of_val(src)`,
        // and is valid for writes for `size_of_val(src)`.
        // If this panics, then `guard` will deallocate for us (if allocation occuured)
        unsafe {
            <T as TryCloneToUninit>::try_clone_to_uninit(src, ptr)?;
        }
        // Defuse the deallocate guard
        mem::forget(guard);
        // Safety: We just initialized `*ptr` as a clone of `src`. Relocate the
        // freshly allocated mutable base `ptr` onto `src`'s metadata (length /
        // vtable), yielding a valid `*mut T` fat pointer to the clone.
        Ok(unsafe { Box::from_raw_in(ptr.cast_with_metadata(src as *const T), alloc) })
    }
}

impl<T> Box<[T], Global> {
    /// Creates a new boxed slice from a `&[T]`, cloning the elements.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, or propagates the
    /// first element-level [`TryCloneError`] encountered.
    #[inline]
    pub fn try_from_slice(slice: &[T]) -> Result<Self, TryCloneError>
    where
        T: TryClone,
    {
        Self::try_from_slice_in(slice, Global)
    }

    /// Allocates a boxed slice of uninitialized memory on the global allocator.
    ///
    /// The elements are left uninitialized; use [`MaybeUninit::write`] to fill them before
    /// reinterpreting as `[T]`. This is the slice analogue of
    /// [`try_new_uninit`](Self::try_new_uninit).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails or `n * size_of::<T>()` overflows.
    #[inline]
    pub fn try_new_uninit_slice(n: usize) -> Result<Box<[MaybeUninit<T>]>, AllocError> {
        Self::try_new_uninit_slice_in(n, Global)
    }

    /// Allocates a boxed slice of zero-initialized memory on the global allocator.
    ///
    /// The elements are zero-filled; use [`MaybeUninit::assume_init`] to reinterpret
    /// them as `[T]` (but only if zeroes are valid values; otherwise, the elements are
    /// always uninitialized).
    ///
    /// This is the slice analogue of [`try_new_zeroed`](Self::try_new_zeroed).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails or `n * size_of::<T>()` overflows.
    #[inline]
    pub fn try_new_zeroed_slice(n: usize) -> Result<Box<[MaybeUninit<T>]>, AllocError> {
        Self::try_new_zeroed_slice_in(n, Global)
    }
}

impl<T, A: Allocator> Box<[T], A> {
    /// Clones a slice into a new boxed slice using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation or any element clone fails.
    #[inline]
    pub fn try_from_slice_in(slice: &[T], alloc: A) -> Result<Self, TryCloneError>
    where
        T: TryClone,
    {
        Self::try_clone_from_ref_in(slice, alloc)
    }

    /// Allocates a boxed slice of uninitialized memory, parameterized over the choice of
    /// allocator for the returned `Box`.
    ///
    /// The elements are left uninitialized; use [`MaybeUninit::write`] to fill them before
    /// reinterpreting as `[T]`. This is the slice analogue of
    /// [`try_new_uninit`](Self::try_new_uninit).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails or `n * size_of::<T>()` overflows.
    #[inline]
    pub fn try_new_uninit_slice_in(
        n: usize,
        alloc: A,
    ) -> Result<Box<[MaybeUninit<T>], A>, AllocError> {
        let buf = RawVec::<T, A>::try_with_capacity_in(n, alloc).map_err(|_| AllocError)?;
        // SAFETY: `buf` holds exactly `n` reserved slots and we wrap all of them, so the
        // length equals the requested capacity (within the allowed range).
        Ok(unsafe { buf.into_box(n) })
    }

    /// Allocates a boxed slice of zero-initialized memory, parameterized over the choice of
    /// allocator for the returned `Box`.
    ///
    /// The elements are zero-filled; use [`MaybeUninit::assume_init`] to reinterpret them as `[T]`
    /// (or [`MaybeUninit::write`]). (but only if zeroes are valid values; otherwise, the elements are
    /// always uninitialized).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the allocation fails or `n * size_of::<T>()` overflows.
    #[inline]
    pub fn try_new_zeroed_slice_in(
        n: usize,
        alloc: A,
    ) -> Result<Box<[MaybeUninit<T>], A>, AllocError> {
        let buf = RawVec::<T, A>::try_with_capacity_zeroed_in(n, alloc).map_err(|_| AllocError)?;
        // SAFETY: as in `try_new_uninit_slice_in`; the buffer is additionally zero-filled.
        Ok(unsafe { buf.into_box(n) })
    }
}
