//! Module that contains APIs that query the [`CString`] type.

use core::ffi::CStr;

use super::CString;
use crate::alloc::Allocator;
use crate::boxed::Box;

// ---------------------------------------------------------------------------
// Query block
// ---------------------------------------------------------------------------

impl<A: Allocator> CString<A> {
    /// Returns the contents of the C string as a slice of bytes, without the
    /// trailing NUL terminator.
    ///
    /// The returned slice never contains a NUL byte\.
    /// The full buffer, terminator included, is available through
    /// [`as_bytes_with_nul`](CString::as_bytes_with_nul).
    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        let bytes: &[u8] = &self.inner;
        // A `CString` always ends in its trailing NUL byte, so `split_last`
        // succeeds; the fallback only keeps the method panic-free.
        bytes.split_last().map_or(bytes, |(_, body)| body)
    }

    /// Returns the contents of the C string as a slice of bytes, trailing NUL
    /// terminator included.
    ///
    /// The last byte of the returned slice is always the NUL terminator, and
    /// it is the only NUL byte in the slice.
    #[must_use]
    #[inline]
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.inner
    }

    /// Extracts a [`CStr`] slice containing the entire string.
    pub fn as_c_str(&self) -> &CStr {
        // SAFETY: Ensured by `as_bytes_with_nul`.
        unsafe { CStr::from_bytes_with_nul_unchecked(self.as_bytes_with_nul()) }
    }

    /// Gets a shared reference to the allocator backing this `CString`.
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        Box::allocator(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::alloc::{CountingAllocator, Global};
    use crate::vec::Vec;

    use super::CString;

    fn cstring(bytes: &[u8]) -> CString {
        CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(bytes, Global)
                .expect("allocation cannot fail for the global allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string")
    }

    #[test]
    fn as_bytes_excludes_the_nul_terminator() {
        assert_eq!(cstring(b"foo\0").as_bytes(), b"foo");
        assert_eq!(cstring(b"\0").as_bytes(), b"");
    }

    #[test]
    fn as_bytes_with_nul_includes_exactly_one_nul() {
        assert_eq!(cstring(b"foo\0").as_bytes_with_nul(), b"foo\0");
        assert_eq!(cstring(b"\0").as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn allocator_returns_the_backing_allocator() {
        let alloc = CountingAllocator::new();
        let c_string = CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(b"x\0", &alloc)
                .expect("allocation cannot fail for the counting allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string");
        assert_eq!(c_string.allocator().allocations(), 1);
    }
}
