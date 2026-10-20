//! Implementations of APIs that convert [`String`] to and from other types
//! and don't directly perform decoding logic.

use crate::{boxed::Box, vec::Vec};
use core::ptr;

use super::FromUtf8Error;
use super::String;
use olive_core::{alloc::Allocator, alloc_errors::TryReserveError};

impl<A: Allocator> String<A> {
    /// Decomposes the `String` into its underlying byte buffer.
    ///
    /// This consumes the `String` and hands over ownership of the allocation;
    /// the caller is responsible for eventually freeing the buffer.
    pub fn into_bytes(self) -> Vec<u8, A> {
        self.buf
    }

    /// Infallibly consumes the `String`, returning its contents as a `Box<str>`
    /// on the same allocator.
    ///
    /// Unlike the fallible cousin [`Self::try_into_boxed_str`], this does not shrink
    /// the string and may cause the string to hold more memory than needed.
    pub fn into_boxed_str(self) -> Box<str, A> {
        let (ptr, len, _cap, alloc) = unsafe { self.buf.into_raw_parts_with_alloc() };
        // SAFETY: `ptr` points to `len` bytes of valid UTF-8 allocated by
        // `alloc`; reinterpreting the fat pointer as `str` preserves the
        // invariant and keeps the same allocator for eventual deallocation.
        unsafe {
            let str_ptr: *mut str = ptr::slice_from_raw_parts_mut(ptr, len) as *mut str;
            Box::from_raw_in(str_ptr, alloc)
        }
    }

    /// Shrinks and consumes the `String`, returning its contents as a `Box<str>` on the
    /// same allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the shrink reallocation fails.
    pub fn try_into_boxed_str(self) -> Result<Box<str, A>, TryReserveError> {
        self.try_into_boxed_str_give_back()
            .map_err(|(_returned, err)| err)
    }

    /// Shrinks and consumes the `String`, returning its contents as a `Box<str>` on the
    /// same allocator — or, if the shrink reallocation fails, returns the original
    /// `String` alongside the error so no data is lost.
    ///
    /// # Errors
    ///
    /// Returns `(Self, TryReserveError)` if the shrink reallocation fails.
    pub fn try_into_boxed_str_give_back(mut self) -> Result<Box<str, A>, (Self, TryReserveError)> {
        // Shrink-to-fit first so the boxed allocation can be exactly `len` bytes.
        match self.buf.try_shrink_to_fit() {
            Ok(()) => Ok(self.into_boxed_str()),
            Err(e) => Err((self, e)),
        }
    }
}

// We intentionally do not allow easy trivial conversions to a Box because it
// misses a shrinking opportunity.
impl<A: Allocator> TryFrom<String<A>> for Box<str, A> {
    type Error = TryReserveError;

    fn try_from(value: String<A>) -> Result<Self, Self::Error> {
        value.try_into_boxed_str()
    }
}

/// Fallible conversion from a borrowed `&str` into an owned [`String`].
///
/// Equivalent to [`String::try_from_str`]; surfaced as a trait impl so that
/// generic code can write `String::try_from(some_str)?`.
impl<'a> TryFrom<&'a str> for String {
    type Error = TryReserveError;

    #[inline]
    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        Self::try_from_str(s)
    }
}

impl<A: Allocator> TryFrom<Vec<u8, A>> for String<A> {
    type Error = FromUtf8Error<A>;

    /// Fallible conversion from a byte vector into a [`String`], validating UTF-8.
    ///
    /// On success the [`Vec<u8>`] is moved into the new [`String`] with no copy or
    /// re-allocation. On failure the offending bytes are returned inside the error
    /// so the caller can inspect or recover them.
    #[inline]
    fn try_from(v: Vec<u8, A>) -> Result<Self, Self::Error> {
        Self::from_utf8_in(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{AllocError, Global};
    use core::alloc::Layout;
    use core::ptr::NonNull;

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    #[test]
    fn into_bytes_moves_buffer() {
        let s = mk("héllo ✓");
        let v = s.into_bytes();
        assert_eq!(&v[..], "héllo ✓".as_bytes());
    }

    #[test]
    fn into_bytes_empty() {
        let s = String::new();
        let v = s.into_bytes();
        assert!(v.is_empty());
    }

    #[test]
    fn into_boxed_str_preserves_contents() {
        let s = mk("boxed ✓");
        let b = s.into_boxed_str();
        assert_eq!(&*b, "boxed ✓");
    }

    #[test]
    fn into_boxed_str_empty() {
        let s = String::new();
        let b = s.into_boxed_str();
        assert_eq!(&*b, "");
    }

    #[test]
    fn try_into_boxed_str_shrinks_and_boxes() {
        let mut s = String::try_with_capacity(64).unwrap();
        s.try_push_str("abc").unwrap();
        let b = s.try_into_boxed_str().unwrap();
        assert_eq!(&*b, "abc");
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn try_into_boxed_str_give_back_success() {
        let mut s = String::try_with_capacity(64).unwrap();
        s.try_push_str("giveback").unwrap();
        let b = s.try_into_boxed_str_give_back().unwrap();
        assert_eq!(&*b, "giveback");
    }

    /// Delegates to `Global` but refuses any allocation of 8 bytes or fewer, so
    /// the shrink reallocation inside `try_into_boxed_str_give_back` fails while
    /// the original (larger) allocation succeeds.
    struct FailSmallAlloc;
    unsafe impl Allocator for FailSmallAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if layout.size() <= 8 {
                Err(AllocError)
            } else {
                Global.allocate(layout)
            }
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    // FIXME: non-giveback variant should be tested too

    #[test]
    fn try_into_boxed_str_give_back_returns_string_on_failure() {
        let mut s = String::try_with_capacity_in(16, FailSmallAlloc).unwrap();
        s.try_push_str("abc").unwrap();
        let (returned, err) = s.try_into_boxed_str_give_back().unwrap_err();
        assert!(err.is_alloc());
        // No data is lost: the original string comes back intact.
        assert_eq!(&*returned, "abc");
        assert_eq!(returned.capacity(), 16);
    }

    #[test]
    fn try_from_string_for_boxed_str_ok() {
        let mut s = String::try_with_capacity(32).unwrap();
        s.try_push_str("via-try-from").unwrap();
        let b: Box<str> = s.try_into().unwrap();
        assert_eq!(&*b, "via-try-from");
    }

    #[test]
    fn try_from_ref_str_ok() {
        let s: String = "via-try-from-ref".try_into().unwrap();
        assert_eq!(&*s, "via-try-from-ref");
    }

    #[test]
    fn try_from_ref_str_empty() {
        let s: String = "".try_into().unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_from_vec_u8_valid_utf8_moves_buffer() {
        let v = Vec::<u8>::try_from_slice_in("moved ✓".as_bytes(), Global).unwrap();
        let ptr = v.as_ptr();
        let s: String = v.try_into().unwrap();
        assert_eq!(&*s, "moved ✓");
        // The byte vector is consumed in place, not copied.
        assert_eq!(s.as_bytes().as_ptr(), ptr);
    }

    #[test]
    fn try_from_vec_u8_invalid_returns_bytes() {
        let v = Vec::<u8>::try_from_slice_in(&[0x61, 0xFF, 0x62], Global).unwrap();
        let err = String::try_from(v).unwrap_err();
        assert!(err.utf8_error().error_len().is_some());
        let bytes = err.into_bytes();
        assert_eq!(&bytes[..], &[0x61, 0xFF, 0x62]);
    }

    #[test]
    fn try_from_vec_u8_empty() {
        let v = Vec::<u8>::try_from_slice_in(&[], Global).unwrap();
        let s: String = v.try_into().unwrap();
        assert!(s.is_empty());
    }
}
