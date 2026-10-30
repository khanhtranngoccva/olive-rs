//! Module that pertains to conversion of [`CString`] structures into other
//! owned types.

use super::CString;
use crate::alloc::Allocator;
use crate::boxed::Box;
use crate::string::String;
use crate::vec::Vec;
use core::ffi::CStr;
use core::fmt::{self, Display, Formatter};

// ---------------------------------------------------------------------------
// Owned conversion block
// ---------------------------------------------------------------------------

impl<A: Allocator> CString<A> {
    /// Consumes the [`CString`], returning its bytes without the trailing NUL
    /// terminator.
    ///
    /// The buffer is reused as-is: the conversion hands the boxed slice to a
    /// vector and pops the terminator, so no reallocation happens.
    #[must_use]
    #[inline]
    pub fn into_bytes(self) -> Vec<u8, A> {
        let mut bytes = Vec::from(self.inner);
        let popped = bytes.pop();
        debug_assert_eq!(
            popped,
            Some(0),
            "a CString always ends in exactly one NUL byte",
        );
        bytes
    }

    /// Consumes the [`CString`], returning its bytes with the trailing NUL
    /// terminator.
    ///
    /// The returned vector is exactly the C string's buffer, so its last byte
    /// is the terminator and no other byte is a NUL.
    #[must_use]
    #[inline]
    pub fn into_bytes_with_nul(self) -> Vec<u8, A> {
        Vec::from(self.inner)
    }

    /// Consumes the [`CString`], returning a boxed [`CStr`].
    #[must_use]
    #[inline]
    pub fn into_boxed_c_str(self) -> Box<CStr, A> {
        let (ptr, alloc) = Box::into_raw_with_allocator(self.inner);
        let casted = ptr as *mut CStr;
        // SAFETY: a valid `CStr` is a NUL-terminated byte slice without
        // interior NUL bytes, which is the invariant a `CString` upholds;
        // the fat pointer keeps the length, and the allocation and allocator
        // are moved unchanged.
        unsafe { Box::from_raw_in(casted, alloc) }
    }

    /// Attempts to convert the [`CString`] into a [`String`].
    ///
    /// The bytes are validated as UTF-8 without the trailing NUL terminator.
    /// On success the buffer is reused as the string's own.
    ///
    /// # Errors
    ///
    /// Returns a [`CStringIntoStringError`] if the bytes before the terminator are not
    /// valid UTF-8; the original string is discarded.
    pub fn into_string(self) -> Result<String<A>, CStringIntoStringError> {
        self.into_string_give_back().map_err(|(_cstr, e)| e)
    }

    /// Attempts to convert the [`CString`] into a [`String`].
    ///
    /// The bytes are validated as UTF-8 without the trailing NUL terminator.
    /// On success the buffer is reused as the string's own.
    ///
    /// # Errors
    ///
    /// Returns the original [`CString`] alongside [`CStringIntoStringError`]
    /// if the bytes before the terminator are not valid UTF-8; the pair keeps
    /// the string and the cause apart, so nothing is discarded.
    pub fn into_string_give_back(self) -> Result<String<A>, (CString<A>, CStringIntoStringError)> {
        let bytes = self.into_bytes();
        match String::from_utf8_in(bytes) {
            Ok(string) => Ok(string),
            Err(error) => {
                let cause = *error.utf8_error();
                // Rebuild the original `CString` from its own bytes.
                // The from_utf8_in function does not change capacity,
                // so the pushing back does not need to perform any
                // sort of allocation.
                // SAFETY: the bytes are the buffer of a valid `CString`
                // without its terminator, so they contain no interior NUL
                // byte.
                let original = unsafe { CString::try_from_vec_unchecked(error.into_bytes()) }
                    .unwrap_or_else(|_| {
                        unreachable!("rebuilding a `CString` from its own bytes cannot allocate")
                    });
                Err((original, CStringIntoStringError { cause }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion relevant to Vec/Boxed CStr
// ---------------------------------------------------------------------------

impl<A: Allocator> From<CString<A>> for Vec<u8, A> {
    /// Consumes the [`CString`] into its full byte buffer, removing the
    /// terminating NUL byte, on the same allocator.
    #[inline]
    fn from(value: CString<A>) -> Self {
        value.into_bytes()
    }
}

impl<A: Allocator> From<CString<A>> for Box<CStr, A> {
    /// Consumes the [`CString`] and converts it into a boxed [`CStr`].
    #[inline]
    fn from(value: CString<A>) -> Self {
        value.into_boxed_c_str()
    }
}

// ---------------------------------------------------------------------------
// IntoStringError
// ---------------------------------------------------------------------------

/// Error returned by [`CString::into_string`] when the bytes before the
/// terminator are not valid UTF-8.
///
/// The error carries no string: the original string travels back to the
/// caller alongside the error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CStringIntoStringError {
    cause: core::str::Utf8Error,
}

impl Display for CStringIntoStringError {
    /// Formats a human-friendly message for this error.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("C string contained non-utf8 bytes")
    }
}

impl core::error::Error for CStringIntoStringError {
    /// Retrieves the type-erased validation failure, which is the source of this error.
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl CStringIntoStringError {
    /// Returns the underlying UTF-8 validation error.
    #[must_use]
    #[inline]
    pub fn utf8_error(&self) -> &core::str::Utf8Error {
        &self.cause
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::CString;
    use crate::alloc::{CountingAllocator, Global};
    use crate::boxed::Box;
    use crate::vec::Vec;
    use core::ffi::CStr;
    use olive_test_helpers::BudgetedAlloc;

    fn cstring(bytes: &[u8]) -> CString {
        CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(bytes, Global)
                .expect("allocation cannot fail for the global allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string")
    }

    #[test]
    fn into_bytes_drops_the_terminator() {
        assert_eq!(cstring(b"foo\0").into_bytes(), b"foo");
        assert_eq!(cstring(b"\0").into_bytes(), b"");
        let b: Vec<_> = cstring(b"foo\0").into();
        assert_eq!(b, b"foo");
    }

    #[test]
    fn into_bytes_with_nul_keeps_the_terminator() {
        assert_eq!(cstring(b"foo\0").into_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn into_boxed_c_str_keeps_the_terminator() {
        let boxed: Box<CStr> = cstring(b"foo\0").into_boxed_c_str();
        assert_eq!(boxed.to_bytes_with_nul(), b"foo\0");
        let boxed: Box<CStr> = cstring(b"foo\0").into();
        assert_eq!(boxed.to_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn into_string_accepts_valid_utf8() {
        let string = cstring("héllo\0".as_bytes())
            .into_string()
            .expect("the input is valid utf-8");
        assert_eq!(string.as_str(), "héllo");
        let string = cstring("héllo\0".as_bytes())
            .into_string_give_back()
            .expect("the input is valid utf-8");
        assert_eq!(string.as_str(), "héllo");
    }

    #[test]
    fn into_string_reports_the_error_even_when_the_allocator_is_exhausted() {
        let alloc = BudgetedAlloc::new(1);
        let c_string = CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(b"fo\xffo\0", &alloc).expect("allocation cannot fail"),
        )
        .expect("the checked input is exactly one nul-terminated string");
        // Rebuilding the original string on failure never allocates: the
        // popped terminator leaves exactly one spare slot in the buffer.
        alloc.drain();
        let error = c_string
            .into_string()
            .expect_err("the input holds an invalid utf-8 byte");
        assert_eq!(
            error.utf8_error().valid_up_to(),
            2,
            "the cause points at the offending byte"
        );
    }

    #[test]
    fn into_string_give_back_returns_the_original_even_when_the_allocator_is_exhausted() {
        let alloc = BudgetedAlloc::new(1);
        let c_string = CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(b"fo\xffo\0", &alloc).expect("allocation cannot fail"),
        )
        .expect("the checked input is exactly one nul-terminated string");
        // Rebuilding the original string on failure never allocates: the
        // popped terminator leaves exactly one spare slot in the buffer.
        alloc.drain();
        let (original, error) = c_string
            .into_string_give_back()
            .expect_err("the input holds an invalid utf-8 byte");
        assert_eq!(original.into_bytes_with_nul(), b"fo\xffo\0");
        assert_eq!(
            error.utf8_error().valid_up_to(),
            2,
            "the cause points at the offending byte"
        );
    }

    #[test]
    fn into_string_error_is_displayed_and_sourced() {
        let (_original, error) = cstring(b"\xff\0")
            .into_string_give_back()
            .expect_err("the input holds an invalid utf-8 byte");
        std::assert_eq!(std::format!("{error}"), "C string contained non-utf8 bytes");
        assert!(core::error::Error::source(&error).is_some());
    }

    #[test]
    fn owned_conversions_move_the_single_allocation() {
        let alloc = CountingAllocator::new();
        let c_string = CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(b"foo\0", &alloc)
                .expect("allocation cannot fail for the counting allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string");
        drop(c_string.into_bytes());
        assert_eq!(alloc.allocations(), 1);
        assert_eq!(alloc.deallocations(), 1);
    }
}
