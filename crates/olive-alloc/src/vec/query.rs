use core::slice;
use super::Vec;
use olive_core::alloc::Allocator;

// ---------------------------------------------------------------------------
// Query methods — generic allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Vec<T, A> {
    /// Returns a reference to the allocator backing this vector.
    #[inline]
    pub const fn allocator(&self) -> &A {
        self.raw.allocator()
    }

    /// The maximum number of elements the vector can hold without reallocating.
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.raw.capacity()
    }

    /// Number of elements in the vector, also referred to as its 'length'.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the vector's buffer.
    ///
    /// The caller must not use this pointer after the vector is mutated or
    /// dropped.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.raw.ptr()
    }

    /// Returns a mutable raw pointer to the vector's buffer.
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.raw.ptr()
    }

    /// Views the vector as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.len <= self.capacity()` is an invariant, and the first
        // `self.len` slots are always initialized.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Views the vector as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see `as_slice`.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}
