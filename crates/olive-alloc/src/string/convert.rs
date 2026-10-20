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
