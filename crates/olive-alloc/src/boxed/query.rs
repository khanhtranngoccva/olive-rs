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
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_ptr()
    }

    /// Gets a shared raw pointer to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Gets a `NonNull<T>` pointing to the underlying data.
    #[must_use]
    #[inline]
    pub fn as_non_null(&self) -> NonNull<T> {
        self.inner
    }

    /// Gets a shared reference to the allocator backing this `Box`.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }
}
