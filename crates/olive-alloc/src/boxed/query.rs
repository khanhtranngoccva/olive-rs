//! Module that contains APIs that query the [`Box`] type.

use core::ptr::NonNull;

use super::Box;
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// Query block
// ---------------------------------------------------------------------------

impl<T: ?Sized, A: Allocator> Box<T, A> {
    /// Gets a mutable raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_mut_ptr(b: &mut Self) -> *mut T {
        b.inner.as_ptr()
    }

    /// Gets a shared raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_ptr(b: &Self) -> *const T {
        b.inner.as_ptr()
    }

    /// Gets a `NonNull<T>` pointing to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_non_null(b: &mut Self) -> NonNull<T> {
        b.inner
    }

    /// Gets a shared reference to the allocator backing this `Box`.
    #[must_use]
    #[inline]
    pub fn allocator(b: &Self) -> &A {
        &b.alloc
    }
}
