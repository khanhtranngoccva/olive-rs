//! Implementations of [`String`] APIs that query string information.

use core::{mem, ptr};

use olive_core::alloc::Allocator;

use super::String;
use crate::{alloc::Global, vec::Vec};

impl String {
    /// Reconstructs a `String` from the components of its underlying buffer:
    /// a raw pointer allocated with [`Global`], a length, and a capacity.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts`](crate::vec::Vec::from_raw_parts) for the
    /// full list of preconditions. In addition, the first `length` bytes must
    /// be valid UTF-8.
    pub unsafe fn from_raw_parts(ptr: *mut u8, length: usize, capacity: usize) -> Self {
        // SAFETY: preconditions passed to the caller.
        let buf = unsafe { Vec::<u8, Global>::from_raw_parts(ptr, length, capacity) };
        debug_assert!(core::str::from_utf8(&buf).is_ok());
        Self { buf }
    }

    /// Decomposes a `String` into its raw components: a raw pointer allocated
    /// with [`Global`], a length, and a capacity.
    ///
    /// The caller owns the memory afterwards and must eventually reconstruct
    /// it via [`from_raw_parts`](Self::from_raw_parts) (or free it
    /// consistently with the `Global` allocator) to avoid a leak.
    ///
    /// The first `length` bytes are valid UTF-8.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut u8, usize, usize) {
        self.buf.into_raw_parts()
    }
}

impl<A: Allocator> String<A> {
    /// Same as [`from_raw_parts`](Self::from_raw_parts), but takes a generic
    /// allocator.
    ///
    /// # Safety
    ///
    /// See [`Vec::from_raw_parts_in`](crate::vec::Vec::from_raw_parts_in) for
    /// the full list of preconditions. In addition, the first `length` bytes
    /// must be valid UTF-8. This condition is unchecked.
    pub unsafe fn from_raw_parts_in(
        ptr: *mut u8,
        length: usize,
        capacity: usize,
        alloc: A,
    ) -> Self {
        // SAFETY: preconditions passed to the caller.
        let buf = unsafe { Vec::<u8, A>::from_raw_parts_in(ptr, length, capacity, alloc) };
        debug_assert!(core::str::from_utf8(&buf).is_ok());
        Self { buf }
    }

    /// Same as [`into_raw_parts`](Self::into_raw_parts), but for any
    /// allocator: decomposes the string into its raw components plus the
    /// allocator, which the caller must use for the eventual deallocation.
    ///
    /// # Safety
    ///
    /// See
    /// [`Vec::into_raw_parts_with_alloc`](crate::vec::Vec::into_raw_parts_with_alloc)
    /// for the full list of preconditions.
    #[inline]
    pub unsafe fn into_raw_parts_with_alloc(self) -> (*mut u8, usize, usize, A) {
        // SAFETY: forwarding the caller's obligation unchanged.
        unsafe { self.buf.into_raw_parts_with_alloc() }
    }

    /// Leaks the string and its allocator instance, returning a string reference
    /// over the allocator's lifetime.
    ///
    /// The allocation is intentionally leaked; the caller owns the memory forever.
    /// The underlying bytes remain valid UTF-8.
    ///
    /// This method does not try to shrink the underlying allocation.
    pub fn leak<'a>(self) -> &'a mut str
    where
        // Unlike the original String implementation, this implementation allows arbitrary
        // allocators to deal with complex allocation scenarios. The string must *not* move
        // out of allocator scope.
        A: 'a,
    {
        let (ptr, len, _cap, alloc) = unsafe { self.buf.into_raw_parts_with_alloc() };
        // Must leak the allocator to prevent the backing storage from being lost.
        mem::forget(alloc);
        // SAFETY: `ptr` points to `len` bytes of valid UTF-8 allocated by
        // `alloc`; reinterpreting the fat pointer as `str` preserves the
        // invariant and keeps the same allocator for eventual deallocation.
        unsafe {
            let str_ptr: *mut str = ptr::slice_from_raw_parts_mut(ptr, len) as *mut str;
            str_ptr.as_mut().expect("pointer is non-nul l")
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Deref;

    use super::*;
    use crate::alloc::CountingAllocator;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    #[test]
    fn into_raw_parts_roundtrips_through_from_raw_parts() {
        let s = mk("raw parts \u{2713}");
        let ptr = s.as_bytes().as_ptr() as *mut u8;
        let (p, len, cap) = s.into_raw_parts();
        assert_eq!(len, 13); // 10 ASCII bytes plus the 3-byte item
        assert!(cap >= 13);
        assert_eq!(p, ptr);
        // SAFETY: the components come straight from a live string, so they
        // satisfy `from_raw_parts`' preconditions.
        let rebuilt = unsafe { String::from_raw_parts(p, len, cap) };
        assert_eq!(rebuilt.deref(), "raw parts \u{2713}");
    }

    #[test]
    fn into_raw_parts_with_alloc_roundtrips_through_from_raw_parts_in() {
        let alloc = CountingAllocator::new();
        let s: String<&CountingAllocator> = String::try_from_str_in("payload", &alloc).unwrap();
        let (ptr, len, cap, returned) = unsafe { s.into_raw_parts_with_alloc() };
        assert_eq!(len, 7);
        // The allocator travels with the parts: reconstruction deallocates
        // through the very same instance.
        assert!(ptr::eq(returned, &alloc));
        // SAFETY: the components come straight from a live string.
        let rebuilt = unsafe { String::from_raw_parts_in(ptr, len, cap, returned) };
        assert_eq!(rebuilt.deref(), "payload");
    }
}
