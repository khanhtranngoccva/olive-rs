//! Module that pertains to splitting and reconstitution of [`CString`] structures
//! into and from raw pointers and parts.

use core::ffi::c_char;
use core::ptr;

use super::CString;
use crate::alloc::{Allocator, Global};
use crate::boxed::Box;

/// Measures the length of the NUL-terminated buffer at `p`, excluding the
/// terminating NUL byte.
///
/// # Safety
///
/// `p` must point to a NUL-terminated buffer, i.e. reading forward from `p`
/// must reach a NUL byte within the allocation.
unsafe fn raw_len(p: *const c_char) -> usize {
    unsafe extern "C" {
        /// Provided by `libc` or `compiler_builtins`.
        fn strlen(s: *const c_char) -> usize;
    }
    unsafe { strlen(p) }
}

// ---------------------------------------------------------------------------
// Global pointer transformation block
// ---------------------------------------------------------------------------

impl CString<Global> {
    /// Converts the `CString` into a raw pointer to the underlying buffer.
    ///
    /// The pointer addresses the first byte of the string, and the buffer ends
    /// in exactly one NUL byte, so it is suitable for handing to a foreign
    /// `char *` or `const char *` interface.
    ///
    /// Ownership of the buffer moves to the caller, which must eventually
    /// return the buffer to Rust to reconstruct a `CString` from it
    /// via [`from_raw`](CString::from_raw) or free it through the global allocator.
    ///
    /// The C side must *not* modify the length of the string by writing a nul
    /// byte inside the string or removing the NUL terminator.
    #[must_use = "losing the pointer will leak memory"]
    #[inline]
    pub fn into_raw(self) -> *mut c_char {
        // Delegate to the boxed slice's `into_raw`, which performs the
        // Miri-friendly retag dance, then drop the slice metadata.
        Box::into_raw(self.inner).cast()
    }

    /// Reconstitutes a [`CString`] from a raw pointer.
    ///
    /// The length of the string is measured by scanning the buffer for its
    /// terminating NUL byte.
    ///
    /// # Safety
    ///
    /// - The pointer must have been previously obtained from
    ///   [`into_raw`](Self::into_raw),
    ///   [`into_raw_with_allocator`](Self::into_raw_with_allocator)
    ///   (on a [`CString`] backed by the [`Global`] allocator), or
    ///   equivalent methods, and must still be valid (not yet freed).
    /// - The C side must not have written a NUL byte inside the string
    ///   or removed the terminating NUL byte.
    pub unsafe fn from_raw(p: *mut c_char) -> Self {
        // SAFETY: the caller guarantees the buffer is NUL-terminated.
        let len = unsafe { raw_len(p) };
        // SAFETY: the caller guarantees the pointer originates from
        // `into_raw`, whose boxed slice is exactly `len + 1` bytes long and
        // owned by the global allocator.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "no buffer can overflow the usize counter"
        )]
        let inner =
            unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(p.cast::<u8>(), len + 1)) };
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// Generic pointer transformation block
// ---------------------------------------------------------------------------

impl<A: Allocator> CString<A> {
    /// Converts the `CString` into a raw pointer, retaining its allocator.
    ///
    /// The pointer addresses the first byte of the string, and the buffer ends
    /// in exactly one NUL byte.
    ///
    /// Ownership of the buffer moves to the caller, which must eventually
    /// return the buffer to Rust to reconstruct a `CString` from it
    /// via [`from_raw_in`](CString::from_raw_in) or free it through
    /// the allocator `A`.
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw_with_allocator(self) -> (*mut c_char, A) {
        let (ptr, alloc) = Box::into_raw_with_allocator(self.inner);
        (ptr.cast(), alloc)
    }

    /// Reconstitutes a [`CString`] from a raw pointer and its allocator.
    ///
    /// The length of the string is measured by scanning the buffer for its
    /// terminating NUL byte.
    ///
    /// # Safety
    ///
    /// - The pointer must have been previously obtained from
    ///   [`into_raw_with_allocator`](Self::into_raw_with_allocator) on a
    ///   `CString<A>` with the same allocator, and must still be valid (not
    ///   yet freed).
    /// - The pointer can also come from [`into_raw`](Self::into_raw);
    ///   should this be the case, `alloc` must be [`Global`].
    /// - The C side must not have written a NUL byte inside the
    ///   string or removed the terminating NUL byte.
    pub unsafe fn from_raw_in(p: *mut c_char, alloc: A) -> Self {
        // SAFETY: the caller guarantees the buffer is NUL-terminated.
        let len = unsafe { raw_len(p) };
        // SAFETY: the caller guarantees the pointer originates from
        // `into_raw_with_allocator` with the same allocator, whose boxed
        // slice is exactly `len + 1` bytes long.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "no buffer can overflow the usize counter"
        )]
        let inner =
            unsafe { Box::from_raw_in(ptr::slice_from_raw_parts_mut(p.cast(), len + 1), alloc) };
        Self { inner }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::CString;
    use crate::alloc::{CountingAllocator, Global};
    use crate::vec::Vec;

    fn cstring(bytes: &[u8]) -> CString {
        CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(bytes, Global)
                .expect("allocation cannot fail for the global allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string")
    }

    #[test]
    fn from_raw_reconstitutes_the_global_string() {
        let ptr = cstring(b"foo\0").into_raw();
        let rebuilt = unsafe { CString::from_raw(ptr) };
        assert_eq!(rebuilt.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn from_raw_reconstitutes_the_empty_string() {
        let ptr = cstring(b"\0").into_raw();
        let rebuilt = unsafe { CString::from_raw(ptr) };
        assert_eq!(rebuilt.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn from_raw_in_reconstitutes_the_string_with_its_allocator() {
        let alloc = CountingAllocator::new();
        let c_string = CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(b"foo\0", &alloc)
                .expect("allocation cannot fail for the counting allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string");
        let (ptr, returned) = c_string.into_raw_with_allocator();
        let rebuilt = unsafe { CString::from_raw_in(ptr, returned) };
        assert_eq!(rebuilt.as_bytes_with_nul(), b"foo\0");
        drop(rebuilt);
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(alloc.deallocations(), 1);
    }
}
