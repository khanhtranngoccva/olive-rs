use super::Rc;
use super::uninit::UniqueRcUninit;
use crate::alloc::{Allocator, Global};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError, TryCloneToUninit};

// ---------------------------------------------------------------------------
// Unsized construction via TryCloneToUninit
// ---------------------------------------------------------------------------

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit> Rc<T, Global> {
    /// Clones a `&T` into a freshly allocated `Rc<T, Global>` for potentially
    /// unsized `T`.
    ///
    /// This is the fallible analogue of std's `Rc::from(&slice[..])`,
    /// `Rc::from("literal")`, etc. It works for any type implementing
    /// [`TryCloneToUninit`]: in practice that means sized types, `str`, and
    /// slices `[T]`. Trait-object payloads (`dyn Trait`) are not yet supported.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, an element clone
    /// fails, or the payload layout is absurd (see
    /// [`try_clone_from_ref_in`](Self::try_clone_from_ref_in)).
    #[inline]
    pub fn try_clone_from_ref(src: &T) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, Global)
    }
}

#[allow(private_bounds)]
impl<T: ?Sized + TryCloneToUninit, A: Allocator> Rc<T, A> {
    /// Clones a `&T` into a freshly allocated `Rc<T, A>` for potentially
    /// unsized `T`, using the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails, any element clone
    /// fails, or the payload layout is absurd (e.g. a slice whose length would
    /// overflow when combined with the header). The last case is reported as
    /// [`TryCloneError::Other`], not an OOM-style error, because it can never
    /// succeed on retry.
    #[inline]
    pub fn try_clone_from_ref_in(src: &T, alloc: A) -> Result<Self, TryCloneError> {
        let uninit = UniqueRcUninit::try_new_for_value(src, alloc)?;
        // SAFETY: `uninit.data_ptr()` points to the uninitialized payload slot
        // within a live allocation of the correct size and alignment.
        unsafe { <T as TryCloneToUninit>::try_clone_to_uninit(src, uninit.data_ptr()) }?;
        // SAFETY: the value is just cloned into the uninit item.
        Ok(unsafe { uninit.into_rc() })
    }
}

// Convenience wrappers for common unsized types.
impl<T: TryClone, A: Allocator> Rc<[T], A> {
    /// Creates a new `Rc<[T]>` by cloning the slice of T items.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_slice_in(src: &[T], alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl<T: TryClone> Rc<[T], Global> {
    /// Convenience wrapper around [`Self::try_from_slice_in`]
    /// using the global allocator.
    #[inline]
    pub fn try_from_slice(src: &[T]) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
    }
}

impl<A: Allocator> Rc<str, A> {
    /// Allocates a new `Rc<str>` by copying the UTF-8 bytes from `src` into
    /// fresh heap memory. Fallible analogue of std's `Rc::from("literal")`.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] if the allocation fails.
    #[inline]
    pub fn try_from_str_in(src: &str, alloc: A) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref_in(src, alloc)
    }
}

impl Rc<str, Global> {
    /// Convenience wrapper around [`Self::try_from_str_in`] using
    /// the global allocator.
    #[inline]
    pub fn try_from_str(src: &str) -> Result<Self, TryCloneError> {
        Self::try_clone_from_ref(src)
    }
}
