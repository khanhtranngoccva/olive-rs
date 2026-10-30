//! Module that pertains to construction of [`CString`] structures.

use olive_core::alloc_errors::{TryReserveError, TryReserveErrorKind};
use olive_core::try_traits::{TryDefault, TryDefaultError};

use super::{CString, CStringNulError, TryCStringFromVecWithNulError, TryCStringNewError};
use crate::alloc::{Allocator, AllocatorTryDefault, Global};
use crate::vec::Vec;
use core::ffi::CStr;
use core::ptr;

// ──────────────────────────────────────────────────────────────────────────────
// Construction block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> CString<A> {
    /// Creates a C string from a vector of bytes (or anything that can
    /// trivially convert into it), appending a terminating NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryCStringNewError::Nul`] if the vector contains a NUL byte,
    /// or [`TryCStringNewError::Reserve`] if appending the terminating NUL
    /// byte or shrinking the buffer into a boxed slice fails.
    ///
    /// On failure the vector is discarded; use [`CString::try_new_give_back`]
    /// to receive it back alongside the error.
    #[inline]
    pub fn try_new(bytes: impl Into<Vec<u8, A>>) -> Result<Self, TryCStringNewError> {
        Self::try_new_give_back(bytes).map_err(|(_payload, e)| e)
    }

    /// Creates a C string from a vector of bytes (or anything that can
    /// trivially convert into it), appending a terminating NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryCStringNewError::Nul`] if the vector contains a NUL byte,
    /// or [`TryCStringNewError::Reserve`] if appending the terminating NUL
    /// byte or shrinking the buffer into a boxed slice fails.
    ///
    /// On failure the vector is returned unchanged alongside the error.
    #[inline]
    pub fn try_new_give_back(
        bytes: impl Into<Vec<u8, A>>,
    ) -> Result<Self, (Vec<u8, A>, TryCStringNewError)> {
        let bytes = bytes.into();
        if let Some(position) = bytes.iter().position(|&byte| byte == 0) {
            return Err((bytes, TryCStringNewError::Nul(CStringNulError(position))));
        }
        // SAFETY: the no-null condition was checked above.
        unsafe { Self::append_nul_and_box(bytes) }
            .map_err(|(bytes, e)| (bytes, TryCStringNewError::Reserve(e)))
    }

    /// Creates a C string from a vector of bytes without any NUL bytes,
    /// appending the terminating NUL byte, without any checking for NUL bytes.
    ///
    /// # Safety
    ///
    /// The vector must not have any NUL bytes.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if appending the terminating NUL byte or
    /// shrinking the buffer into a boxed slice fails.
    ///
    /// On failure the vector is discarded; use
    /// [`CString::try_from_vec_unchecked_give_back`] to receive it back
    /// alongside the error.
    #[inline]
    pub unsafe fn try_from_vec_unchecked(bytes: Vec<u8, A>) -> Result<Self, TryReserveError> {
        // SAFETY: precondition from the caller.
        unsafe { Self::append_nul_and_box(bytes) }.map_err(|(_bytes, e)| e)
    }

    /// Creates a C string from a vector of bytes, appending a terminating NUL
    /// byte, without checking for NUL bytes.
    ///
    /// # Safety
    ///
    /// The provided vector must contain no interior NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if appending the terminating NUL byte or
    /// shrinking the buffer into a boxed slice fails.
    ///
    /// On failure the vector is returned unchanged alongside the error.
    #[inline]
    pub unsafe fn try_from_vec_unchecked_give_back(
        bytes: Vec<u8, A>,
    ) -> Result<Self, (Vec<u8, A>, TryReserveError)> {
        // SAFETY: precondition from the caller.
        unsafe { Self::append_nul_and_box(bytes) }
    }

    /// Creates a C string from a vector of bytes that already ends with
    /// exactly one NUL byte at the end, without checking for NUL bytes.
    ///
    /// # Safety
    ///
    /// The provided vector must end with a NUL byte and contain no interior
    /// NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if shrinking the buffer into a boxed slice
    /// fails.
    ///
    /// On failure the vector is discarded; use
    /// [`CString::try_from_vec_with_nul_unchecked_give_back`] to receive it
    /// back alongside the error.
    #[inline]
    pub unsafe fn try_from_vec_with_nul_unchecked(
        bytes: Vec<u8, A>,
    ) -> Result<Self, TryReserveError> {
        match bytes.try_into_boxed_slice_give_back() {
            Ok(boxed) => Ok(Self { inner: boxed }),
            Err((_bytes, e)) => Err(e),
        }
    }

    /// Creates a C string from a vector of bytes that already ends with
    /// exactly one NUL byte at the end, without checking for NUL bytes.
    ///
    /// # Safety
    ///
    /// The provided vector must end with a NUL byte and contain no interior
    /// NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if shrinking the buffer into a boxed slice
    /// fails.
    ///
    /// On failure the vector is returned unchanged alongside the error.
    #[inline]
    pub unsafe fn try_from_vec_with_nul_unchecked_give_back(
        bytes: Vec<u8, A>,
    ) -> Result<Self, (Vec<u8, A>, TryReserveError)> {
        match bytes.try_into_boxed_slice_give_back() {
            Ok(boxed) => Ok(Self { inner: boxed }),
            Err((bytes, e)) => Err((bytes, e)),
        }
    }

    /// Creates a C string by copying the bytes of a borrowed [`CStr`],
    /// trailing NUL terminator included, on the given allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if reserving capacity for the copy fails.
    #[inline]
    pub fn try_from_c_str_in(src: &CStr, alloc: A) -> Result<Self, TryReserveError> {
        // One extra byte for the terminating NUL.
        let capacity = src
            .count_bytes()
            .checked_add(1)
            .ok_or(TryReserveErrorKind::CapacityOverflow)?;
        let mut bytes = Vec::<u8, _>::try_with_capacity_in(capacity, alloc)?;
        // SAFETY: `capacity` reflects the payload length with a NUL terminator, and the destination
        // covers enough bytes.
        unsafe {
            ptr::copy_nonoverlapping::<u8>(
                src.to_bytes_with_nul().as_ptr(),
                bytes.as_mut_ptr(),
                capacity,
            );
        }
        // SAFETY: the NUL has already been copied, there is no interior NUL byte in a CStr.
        unsafe { Self::try_from_vec_with_nul_unchecked(bytes) }
    }

    /// Creates a C string from a vector of bytes that already ends with
    /// exactly one NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryCStringFromVecWithNulError::NotNulTerminated`] if the vector
    /// is empty or does not end with a NUL byte,
    /// [`TryCStringFromVecWithNulError::InteriorNul`] if it contains an interior
    /// NUL byte, or [`TryCStringFromVecWithNulError::Reserve`] if shrinking the
    /// buffer into a boxed slice fails.
    ///
    /// On failure the vector is discarded; use
    /// [`CString::try_from_vec_with_nul_give_back`] to receive it back alongside
    /// the error.
    #[inline]
    pub fn try_from_vec_with_nul(bytes: Vec<u8, A>) -> Result<Self, TryCStringFromVecWithNulError> {
        Self::try_from_vec_with_nul_give_back(bytes).map_err(|(_payload, e)| e)
    }

    /// Creates a C string from a vector of bytes that already ends with
    /// exactly one NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryCStringFromVecWithNulError::NotNulTerminated`] if the vector
    /// is empty or does not end with a NUL byte,
    /// [`TryCStringFromVecWithNulError::InteriorNul`] if it contains an interior
    /// NUL byte, or [`TryCStringFromVecWithNulError::Reserve`] if shrinking the
    /// buffer into a boxed slice fails.
    ///
    /// On failure the vector is returned unchanged alongside the error.
    #[inline]
    pub fn try_from_vec_with_nul_give_back(
        bytes: Vec<u8, A>,
    ) -> Result<Self, (Vec<u8, A>, TryCStringFromVecWithNulError)> {
        let Some((&0, body)) = bytes.split_last() else {
            return Err((bytes, TryCStringFromVecWithNulError::NotNulTerminated));
        };
        if let Some(position) = body.iter().position(|&byte| byte == 0) {
            return Err((bytes, TryCStringFromVecWithNulError::InteriorNul(position)));
        }
        match bytes.try_into_boxed_slice_give_back() {
            Ok(boxed) => Ok(Self { inner: boxed }),
            Err((bytes, e)) => Err((bytes, TryCStringFromVecWithNulError::Reserve(e))),
        }
    }

    /// Appends the terminating NUL byte to the vector and shrinks it into
    /// the boxed slice backing the string.
    ///
    /// On failure the vector is returned unchanged: appending never removes
    /// bytes, and the shrink reallocation leaves the contents intact.
    ///
    /// # Safety
    ///
    /// - The vector must not have any NUL/zeroes.
    unsafe fn append_nul_and_box(
        mut bytes: Vec<u8, A>,
    ) -> Result<Self, (Vec<u8, A>, TryReserveError)> {
        // Reserving exactly by 1 allows try_into_boxed_slice_give_back to shrink for free
        // in certain scenarios.
        if let Err(e) = bytes.try_reserve_exact(1) {
            return Err((bytes, e));
        }
        if let Err(e) = bytes.try_push(0) {
            return Err((bytes, e));
        }
        match bytes.try_into_boxed_slice_give_back() {
            Ok(boxed) => Ok(Self { inner: boxed }),
            Err((mut bytes, e)) => {
                // Must pop the NUL terminator from last step.
                bytes.pop();
                Err((bytes, e))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Construction from a borrowed CStr — global allocator
// ---------------------------------------------------------------------------

impl CString<Global> {
    /// Creates a C string by copying the bytes of a borrowed [`CStr`],
    /// trailing NUL terminator included, using the global allocator.
    ///
    /// See [`CString::try_from_c_str_in`] for the allocator-generic variant.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the allocation fails.
    #[inline]
    pub fn try_from_c_str(src: &CStr) -> Result<Self, TryReserveError> {
        Self::try_from_c_str_in(src, Global)
    }
}

// ---------------------------------------------------------------------------
// Default / TryDefault
// ---------------------------------------------------------------------------

impl<A: AllocatorTryDefault> TryDefault for CString<A> {
    /// Creates the default empty `CString`, which consists of a lone NUL byte.
    ///
    /// # Errors
    ///
    /// Returns [`TryDefaultError::Reserve`] if obtaining the default allocator
    /// handle or reserving the buffer for the terminating NUL byte fails.
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        let bytes = Vec::<u8, A>::try_with_capacity_in(1, A::try_default()?)
            .map_err(TryDefaultError::Reserve)?;
        // SAFETY: the byte vector is empty.
        unsafe { Self::try_from_vec_unchecked(bytes) }.map_err(TryDefaultError::Reserve)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{CString, TryCStringFromVecWithNulError, TryCStringNewError};
    use crate::alloc::{CountingAllocator, Global};
    use crate::vec::Vec;
    use olive_core::alloc_errors::TryReserveErrorKind;
    use olive_core::try_traits::{TryDefault, TryDefaultError, TryExtendFromSlice};
    use olive_test_helpers::{BudgetedAlloc, FailAlloc, FailDefaultAlloc};

    fn checked_vec(bytes: &[u8]) -> Vec<u8, Global> {
        Vec::try_from_slice_in(bytes, Global).expect("allocation cannot fail")
    }

    #[test]
    fn try_from_c_str_copies_the_bytes() {
        let string = CString::try_from_c_str(c"foo").expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"foo");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");

        let empty = CString::try_from_c_str(c"").expect("allocation cannot fail");
        assert_eq!(empty.as_bytes(), b"");
        assert_eq!(empty.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn try_from_c_str_in_uses_the_given_allocator() {
        let alloc = CountingAllocator::new();
        let string = CString::try_from_c_str_in(c"foo", &alloc)
            .expect("allocation cannot fail for the counting allocator under test input");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
        assert_eq!(string.allocator().allocations(), 1);
        drop(string);
        assert_eq!(alloc.allocations(), alloc.deallocations());
    }

    #[test]
    fn try_from_c_str_in_fails_when_the_allocator_is_exhausted() {
        let alloc = BudgetedAlloc::new(0);
        let error = CString::try_from_c_str_in(c"foo", &alloc)
            .expect_err("the zero budget cannot allocate");
        assert!(
            matches!(error.kind(), TryReserveErrorKind::AllocError { .. }),
            "expected an allocation error, got {error:?}"
        );
    }

    #[test]
    fn try_new_appends_the_terminator() {
        let string = CString::try_new(checked_vec(b"foo")).expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"foo");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_new_discards_the_bytes_on_a_nul_error() {
        let error = CString::try_new(checked_vec(b"fo\0o")).unwrap_err();
        match error {
            TryCStringNewError::Nul(nul) => assert_eq!(nul.nul_position(), 2),
            other @ TryCStringNewError::Reserve(_) => {
                panic!("expected a nul error, got {other:?}")
            }
        }
    }

    #[test]
    fn try_new_discards_the_bytes_on_a_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let bytes = Vec::try_from_slice_in(b"foo", &alloc).expect("allocation cannot fail");
        // The budget was spent building the vector, so appending the
        // terminator cannot allocate; the plain variant discards the bytes.
        match CString::try_new(bytes) {
            Err(TryCStringNewError::Reserve(reserve)) => {
                assert!(matches!(
                    reserve.kind(),
                    TryReserveErrorKind::AllocError { .. }
                ));
            }
            other => panic!("expected a reserve error, got {other:?}"),
        }

        alloc.set_budget(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"foo")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the oversized vector, so shrinking
        // the buffer into a boxed slice cannot allocate.
        match CString::try_new(bytes) {
            Err(TryCStringNewError::Reserve(reserve)) => {
                assert!(matches!(
                    reserve.kind(),
                    TryReserveErrorKind::AllocError { .. }
                ));
            }
            other => panic!("expected a reserve error, got {other:?}"),
        }
    }

    #[test]
    fn try_new_give_back_appends_the_terminator() {
        let string =
            CString::try_new_give_back(checked_vec(b"foo")).expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_new_give_back_returns_the_bytes_alongside_the_nul_error() {
        let (bytes, error) = CString::try_new_give_back(checked_vec(b"fo\0o")).unwrap_err();
        assert_eq!(bytes.as_slice(), &b"fo\0o"[..]);
        match error {
            TryCStringNewError::Nul(nul) => assert_eq!(nul.nul_position(), 2),
            other @ TryCStringNewError::Reserve(_) => {
                panic!("expected a nul error, got {other:?}")
            }
        }
    }

    #[test]
    fn try_new_give_back_returns_the_bytes_alongside_the_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let bytes = Vec::try_from_slice_in(b"foo", &alloc).expect("allocation cannot fail");
        // The budget was spent building the vector, so appending the
        // terminator cannot allocate.
        let (mut bytes, error) = CString::try_new_give_back(bytes).unwrap_err();
        assert_eq!(bytes.as_slice(), &b"foo"[..]);
        match error {
            TryCStringNewError::Reserve(_) => {}
            other @ TryCStringNewError::Nul(_) => panic!("expected a reserve error, got {other:?}"),
        }

        alloc.set_budget(1);
        // The budget was spent building an oversized vector, so shrinking cannot allocate.
        let mut bytes_2 = Vec::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes_2
            .try_append(&mut bytes)
            .expect("should push into pre-allocated buffer");
        let (bytes_2, error) = CString::try_new_give_back(bytes_2).unwrap_err();
        assert_eq!(bytes_2.as_slice(), &b"foo"[..]);
        match error {
            TryCStringNewError::Reserve(_) => {}
            other @ TryCStringNewError::Nul(_) => panic!("expected a reserve error, got {other:?}"),
        }
    }

    #[test]
    fn try_from_vec_with_nul_keeps_the_terminator() {
        let string =
            CString::try_from_vec_with_nul(checked_vec(b"foo\0")).expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"foo");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_with_nul_discards_the_bytes_on_a_validation_error() {
        assert!(matches!(
            CString::try_from_vec_with_nul(checked_vec(b"foo")).unwrap_err(),
            TryCStringFromVecWithNulError::NotNulTerminated
        ));
        assert!(matches!(
            CString::try_from_vec_with_nul(checked_vec(b"fo\0o\0")).unwrap_err(),
            TryCStringFromVecWithNulError::InteriorNul(2)
        ));
    }

    #[test]
    fn try_from_vec_with_nul_discards_the_bytes_on_a_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"fo\0")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the oversized vector, so shrinking
        // the buffer into a boxed slice cannot allocate; the plain variant
        // discards the bytes.
        match CString::try_from_vec_with_nul(bytes) {
            Err(TryCStringFromVecWithNulError::Reserve(reserve)) => {
                assert!(matches!(
                    reserve.kind(),
                    TryReserveErrorKind::AllocError { .. }
                ));
            }
            other => panic!("expected a reserve error, got {other:?}"),
        }
    }

    #[test]
    fn try_from_vec_with_nul_give_back_keeps_the_terminator() {
        let string = CString::try_from_vec_with_nul_give_back(checked_vec(b"foo\0"))
            .expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_with_nul_give_back_returns_the_bytes_alongside_the_errors() {
        let (bytes, error) =
            CString::try_from_vec_with_nul_give_back(checked_vec(b"fo\0o\0")).unwrap_err();
        assert_eq!(bytes.as_slice(), &b"fo\0o\0"[..]);
        assert!(matches!(
            error,
            TryCStringFromVecWithNulError::InteriorNul(2)
        ));

        let (bytes, error) =
            CString::try_from_vec_with_nul_give_back(checked_vec(b"foo")).unwrap_err();
        assert_eq!(bytes.as_slice(), &b"foo"[..]);
        assert!(matches!(
            error,
            TryCStringFromVecWithNulError::NotNulTerminated
        ));
    }

    #[test]
    fn try_from_vec_with_nul_give_back_returns_the_bytes_alongside_the_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes = Vec::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"fo\0")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the vector, so shrinking the buffer
        // into a boxed slice cannot allocate.
        let (bytes, error) = CString::try_from_vec_with_nul_give_back(bytes).unwrap_err();
        assert_eq!(bytes.as_slice(), &b"fo\0"[..]);
        match error {
            TryCStringFromVecWithNulError::Reserve(_) => {}
            other @ (TryCStringFromVecWithNulError::InteriorNul(_)
            | TryCStringFromVecWithNulError::NotNulTerminated) => {
                panic!("expected a reserve error, got {other:?}")
            }
        }
    }

    #[test]
    fn try_from_vec_unchecked_appends_the_terminator() {
        // SAFETY: the byte literal has no NUL bytes.
        let string = unsafe { CString::try_from_vec_unchecked(checked_vec(b"foo")) }
            .expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"foo");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_unchecked_discards_the_bytes_when_appending_fails() {
        let alloc = BudgetedAlloc::new(1);
        let bytes = Vec::try_from_slice_in(b"foo", &alloc).expect("allocation cannot fail");
        // The budget was spent building the vector, so appending the
        // terminator cannot allocate; the plain variant discards the bytes.
        let error = unsafe {
            // SAFETY: the byte literal has no NUL bytes.
            CString::try_from_vec_unchecked(bytes)
        }
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_from_vec_unchecked_discards_the_bytes_when_shrinking_fails() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"foo")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the oversized vector, so the shrink
        // reallocation cannot allocate; the plain variant discards the bytes.
        let error = unsafe {
            // SAFETY: the byte literal has no NUL bytes.
            CString::try_from_vec_unchecked(bytes)
        }
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_from_vec_unchecked_give_back_appends_the_terminator() {
        // SAFETY: the byte literal has no NUL bytes.
        let string = unsafe { CString::try_from_vec_unchecked_give_back(checked_vec(b"foo")) }
            .expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_unchecked_give_back_returns_the_bytes_alongside_the_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let bytes = Vec::try_from_slice_in(b"foo", &alloc).expect("allocation cannot fail");
        // The budget was spent building the vector, so appending the
        // terminator cannot allocate.
        let (bytes, error) = unsafe {
            // SAFETY: the byte literal has no NUL bytes.
            CString::try_from_vec_unchecked_give_back(bytes)
        }
        .unwrap_err();
        assert_eq!(bytes.as_slice(), &b"foo"[..]);
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_from_vec_unchecked_give_back_returns_the_bytes_when_shrinking_fails() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"foo")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the oversized vector, so the shrink
        // reallocation cannot allocate.
        let (bytes, error) = unsafe {
            // SAFETY: the byte literal has no NUL bytes.
            CString::try_from_vec_unchecked_give_back(bytes)
        }
        .unwrap_err();
        assert_eq!(bytes.as_slice(), &b"foo"[..]);
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_from_vec_with_nul_unchecked_keeps_the_terminator() {
        // SAFETY: the vector ends with exactly one NUL byte and has no
        // interior NUL bytes.
        let string = unsafe { CString::try_from_vec_with_nul_unchecked(checked_vec(b"foo\0")) }
            .expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"foo");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_with_nul_unchecked_discards_the_bytes_when_shrinking_fails() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"fo\0")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the oversized vector, so shrinking
        // the buffer into a boxed slice cannot allocate; the plain variant
        // discards the bytes.
        let error = unsafe {
            // SAFETY: the vector ends with exactly one NUL byte and has no
            // interior NUL bytes.
            CString::try_from_vec_with_nul_unchecked(bytes)
        }
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_from_vec_with_nul_unchecked_give_back_keeps_the_terminator() {
        // SAFETY: the vector ends with exactly one NUL byte and has no
        // interior NUL bytes.
        let string =
            unsafe { CString::try_from_vec_with_nul_unchecked_give_back(checked_vec(b"foo\0")) }
                .expect("allocation cannot fail");
        assert_eq!(string.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn try_from_vec_with_nul_unchecked_give_back_returns_the_bytes_alongside_the_reserve_error() {
        let alloc = BudgetedAlloc::new(1);
        let mut bytes =
            Vec::<u8, _>::try_with_capacity_in(8, &alloc).expect("allocation cannot fail");
        bytes
            .try_extend_from_slice(b"fo\0")
            .expect("should extend from byte slice with reserved capacity");
        // The budget was spent building the vector, so shrinking the buffer
        // into a boxed slice cannot allocate.
        let (bytes, error) = unsafe {
            // SAFETY: the vector ends with exactly one NUL byte and has no
            // interior NUL bytes.
            CString::try_from_vec_with_nul_unchecked_give_back(bytes)
        }
        .unwrap_err();
        assert_eq!(bytes.as_slice(), &b"fo\0"[..]);
        assert!(matches!(
            error.kind(),
            TryReserveErrorKind::AllocError { .. }
        ));
    }

    #[test]
    fn try_default_yields_a_lone_nul_byte() {
        let string = <CString as TryDefault>::try_default().expect("allocation cannot fail");
        assert_eq!(string.as_bytes(), b"");
        assert_eq!(string.as_bytes_with_nul(), b"\0");
    }

    #[test]
    fn try_default_failure_propagates_allocator_error() {
        let error = <CString<FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            error,
            TryDefaultError::Other("allocator default failed")
        ));
    }

    #[test]
    fn try_default_fails_when_the_buffer_allocation_fails() {
        // `FailAlloc` defaults fine but fails every non-zero-size
        // allocation, so reserving the buffer for the lone terminator
        // fails.
        let error = <CString<FailAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(error, TryDefaultError::Reserve(_)));
    }
}
