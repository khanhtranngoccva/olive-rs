//! Module that pertains to conversion of [`CString`] structures into other
//! owned types.

use super::CString;
use crate::alloc::Allocator;
use crate::borrow::Cow;
use crate::boxed::Box;
use crate::string::String;
use crate::vec::Vec;
use core::ffi::CStr;
use core::fmt::{self, Display, Formatter};
use core::num::NonZero;
use olive_core::alloc_errors::TryReserveError;

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
// Conversion relevant to Cow
// ---------------------------------------------------------------------------

impl From<CString> for Cow<'_, CStr> {
    /// Moves the owned [`CString`] into the owned arm of a [`Cow`], without
    /// copying.
    #[inline]
    fn from(value: CString) -> Self {
        Cow::Owned(value)
    }
}

impl<'a> From<&'a CStr> for Cow<'a, CStr> {
    /// Moves a [`CStr`] reference into the borrowed arm of a [`Cow`], without copying.
    #[inline]
    fn from(value: &'a CStr) -> Self {
        Cow::Borrowed(value)
    }
}

impl<'a> TryFrom<Cow<'a, CStr>> for CString {
    type Error = TryReserveError;

    /// Turns a [`Cow`] of a [`CStr`] into an owned [`CString`].
    ///
    /// The owned arm is moved through unchanged; the borrowed arm is copied
    /// onto the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if copying the borrowed arm fails.
    #[inline]
    fn try_from(value: Cow<'a, CStr>) -> Result<Self, Self::Error> {
        match value {
            Cow::Borrowed(borrowed) => CString::try_from_c_str(borrowed),
            Cow::Owned(owned) => Ok(owned),
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion from borrowed references
// ---------------------------------------------------------------------------

impl<'a> TryFrom<&'a CStr> for CString {
    type Error = TryReserveError;

    /// Copies a borrowed [`CStr`] into an owned [`CString`] on the global
    /// allocator.
    ///
    /// See [`CString::try_from_c_str_in`] for the allocator-generic variant.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if copying the bytes fails.
    #[inline]
    fn try_from(value: &'a CStr) -> Result<Self, Self::Error> {
        Self::try_from_c_str(value)
    }
}

impl<A: Allocator> TryFrom<&CString<A>> for CString {
    type Error = TryReserveError;

    /// Copies a reference to a [`CString`] into a new [`CString`] on the
    /// global allocator, regardless of the source's allocator.
    ///
    /// To clone onto the source's own allocator instead, use
    /// [`TryClone`](olive_core::try_traits::try_clone::TryClone), which keeps
    /// the allocator handle, or [`CString::try_from_c_str_in`] with the
    /// source's allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if copying the bytes fails.
    #[inline]
    fn try_from(value: &CString<A>) -> Result<Self, Self::Error> {
        Self::try_from_c_str(value.as_c_str())
    }
}

// ---------------------------------------------------------------------------
// Conversion from non-zero bytes
// ---------------------------------------------------------------------------

impl<A: Allocator> TryFrom<Vec<NonZero<u8>, A>> for CString<A> {
    type Error = TryReserveError;

    /// Creates a C string from a vector of non-zero bytes, appending a
    /// terminating NUL byte.
    ///
    /// The bytes are reinterpreted in place without copying: a [`NonZero`]
    /// can never hold the value zero, so the vector is guaranteed to contain
    /// no NUL byte at all.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if appending the terminating NUL byte or
    /// shrinking the buffer into a boxed slice fails. On failure the vector
    /// is discarded.
    fn try_from(bytes: Vec<NonZero<u8>, A>) -> Result<Self, Self::Error> {
        let (ptr, length, capacity, alloc) = bytes.into_raw_parts_with_allocator();
        // SAFETY: `NonZero<u8>` is `#[repr(transparent)]` over `u8`, so the
        // buffer may be reinterpreted as plain bytes with the same length,
        // capacity and allocator.
        let bytes = unsafe { Vec::from_raw_parts_in(ptr.cast::<u8>(), length, capacity, alloc) };
        // SAFETY: a `NonZero<u8>` can never be a NUL byte, so the buffer
        // contains no interior NUL byte.
        unsafe { Self::try_from_vec_unchecked(bytes) }
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
    use crate::borrow::Cow;
    use crate::boxed::Box;
    use crate::vec::Vec;
    use core::ffi::CStr;
    use core::num::NonZero;
    use olive_core::alloc_errors::TryReserveErrorKind;
    use olive_test_helpers::BudgetedAlloc;

    fn cstring(bytes: &[u8]) -> CString {
        CString::try_from_vec_with_nul(
            Vec::try_from_slice_in(bytes, Global)
                .expect("allocation cannot fail for the global allocator under test input"),
        )
        .expect("the checked input is exactly one nul-terminated string")
    }

    fn borrowed_cow(bytes: &'static [u8]) -> Cow<'static, CStr> {
        Cow::Borrowed(CStr::from_bytes_with_nul(bytes).expect("the test bytes end in one nul"))
    }

    fn owned_cow(bytes: &[u8]) -> Cow<'static, CStr> {
        let c_str = CStr::from_bytes_with_nul(bytes).expect("the test bytes end in one nul");
        Cow::Owned(
            CString::try_from_c_str(c_str)
                .expect("allocation cannot fail for the global allocator"),
        )
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

    #[test]
    fn c_string_moves_into_the_owned_arm_of_a_cow() {
        let source = cstring(b"foo\0");
        let cow = Cow::<'static, CStr>::from(source);
        match cow {
            Cow::Owned(owned) => assert_eq!(owned.as_bytes_with_nul(), b"foo\0"),
            Cow::Borrowed(_) => panic!("an owned CString must land in the owned arm"),
        }
    }

    #[test]
    fn c_str_borrows_into_the_borrowed_arm_of_a_cow() {
        let source = c"foo";
        let cow = Cow::from(source);
        match cow {
            Cow::Borrowed(borrowed) => assert!(core::ptr::eq(borrowed, source)),
            Cow::Owned(_) => panic!("a borrowed CStr must land in the borrowed arm"),
        }
    }

    #[test]
    fn cow_try_from_moves_the_owned_arm_without_copying() {
        let cow = owned_cow(b"foo\0");
        let c_string: CString = CString::try_from(cow).expect("moving an owned arm cannot fail");
        assert_eq!(c_string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn cow_try_from_copies_the_borrowed_arm() {
        let cow = borrowed_cow(b"foo\0");
        let c_string: CString = CString::try_from(cow).expect("allocation cannot fail");
        assert_eq!(c_string.as_bytes_with_nul(), b"foo\0");
        assert_eq!(c_string.as_bytes(), b"foo");
    }

    #[test]
    fn c_str_try_from_copies_the_bytes() {
        let string = CString::try_from(c"foo").expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");

        let empty = CString::try_from(c"").expect("allocation cannot fail");
        assert_eq!(empty.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn c_string_reference_try_from_copies_across_allocators() {
        let alloc = CountingAllocator::new();
        let source = CString::try_from_c_str_in(c"foo", &alloc)
            .expect("allocation cannot fail for the counting allocator under test input");
        let baseline = alloc.allocations();

        let copy: CString = CString::try_from(&source).expect("allocation cannot fail");
        assert_eq!(copy.as_bytes_with_nul(), b"foo\0");
        // The copy never touched the source's allocator.
        assert_eq!(alloc.allocations(), baseline);
        // The source is untouched by the conversion.
        assert_eq!(source.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn non_zero_vec_try_from_appends_the_terminator() {
        let mut bytes: Vec<NonZero<u8>> = Vec::new();
        for byte in b"foo" {
            bytes
                .try_push(NonZero::new(*byte).expect("the byte is not zero"))
                .expect("allocation cannot fail");
        }

        let string = CString::try_from(bytes).expect("allocation in test harness cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");

        let string = CString::try_from(Vec::<NonZero<u8>>::new())
            .expect("allocation in test harness cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn non_zero_vec_try_from_moves_the_source_allocator() {
        let alloc = CountingAllocator::new();
        let mut bytes: Vec<NonZero<u8>, &CountingAllocator> =
            Vec::try_with_capacity_in(4, &alloc).expect("allocation cannot fail");
        for byte in b"foo" {
            bytes
                .try_push(NonZero::new(*byte).expect("the byte is not zero"))
                .expect("allocation cannot fail");
        }
        let baseline = alloc.allocations();

        let string: CString<&CountingAllocator> =
            CString::try_from(bytes).expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
        // The terminator exactly fills the vector's spare capacity, so the
        // conversion neither grew nor shrank the buffer.
        assert_eq!(alloc.allocations(), baseline);
        drop(string);
        assert_eq!(alloc.deallocations(), 1);
    }

    #[test]
    fn non_zero_vec_try_from_fails_when_the_terminator_cannot_be_appended() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes: Vec<NonZero<u8>, BudgetedAlloc> = Vec::new_in(alloc.clone());
        bytes
            .try_push(NonZero::new(b'f').expect("the byte is not zero"))
            .expect("allocation cannot fail");
        // The vector's own allocation consumed the budget, so appending the
        // terminator cannot succeed.
        alloc.drain();

        let error = CString::try_from(bytes).unwrap_err();
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn non_zero_vec_try_from_fails_when_an_overreserved_vec_cannot_shrink() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes: Vec<NonZero<u8>, BudgetedAlloc> =
            Vec::try_with_capacity_in(4, alloc.clone()).expect("allocation cannot fail");
        for byte in b"fo" {
            bytes
                .try_push(NonZero::new(*byte).expect("the byte is not zero"))
                .expect("allocation cannot fail");
        }
        // The terminator fits in the vector's spare capacity, so the only
        // allocation the conversion still needs is the shrink from four
        // slots down to three, which the exhausted budget cannot provide.
        alloc.drain();

        let error = CString::try_from(bytes).unwrap_err();
        assert!(matches!(error.kind(), TryReserveErrorKind::AllocError { .. }));
    }
}
