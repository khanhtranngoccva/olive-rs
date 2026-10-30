//! [`TryToOwned`]: a fallible analogue of [`alloc::borrow::ToOwned`].
//!
//! This modules declares the [`TryToOwned`] trait, which fallibly converts
//! borrowed references to owned types.

use crate::ffi::CString;
use crate::string::String;
use crate::vec::{TryVecWithCloneError, Vec};
use core::borrow::Borrow;
use core::ffi::CStr;
use core::fmt;
use olive_core::alloc_errors::{AllocError, TryReserveError};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

/// Error returned by [`TryToOwned::try_to_owned`].
#[derive(Clone, PartialEq, Eq)]
pub enum TryToOwnedError {
    /// A capacity reservation on a collection failed (overflow or OOM).
    Reserve(TryReserveError),
    /// A single heap allocation failed (no reserve phase — e.g. a leaf
    /// allocation such as a `Box`, `Arc`, or `Rc` node).
    Alloc(AllocError),
    /// A logic-level failure with a static diagnostic message.
    Other(&'static str),
}

impl fmt::Debug for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f.debug_tuple("TryToOwnedError::Reserve").field(e).finish(),
            Self::Alloc(e) => f.debug_tuple("TryToOwnedError::Alloc").field(e).finish(),
            Self::Other(msg) => f.debug_tuple("TryToOwnedError::Other").field(msg).finish(),
        }
    }
}

impl fmt::Display for TryToOwnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "to_owned failed: {e}"),
            Self::Alloc(_) => write!(f, "to_owned failed: memory allocation failed"),
            Self::Other(msg) => write!(f, "to_owned failed: {msg}"),
        }
    }
}

impl core::error::Error for TryToOwnedError {}

impl From<TryReserveError> for TryToOwnedError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<AllocError> for TryToOwnedError {
    #[inline]
    fn from(err: AllocError) -> Self {
        Self::Alloc(err)
    }
}

impl From<TryCloneError> for TryToOwnedError {
    /// Lifts a [`TryCloneError`] into an identically-shaped [`TryToOwnedError`].
    #[inline]
    fn from(err: TryCloneError) -> Self {
        match err {
            TryCloneError::Reserve(e) => Self::Reserve(e),
            TryCloneError::Alloc(e) => Self::Alloc(e),
            TryCloneError::Other(msg) => Self::Other(msg),
        }
    }
}

/// A fallible analogue of [`alloc::borrow::ToOwned`].
///
/// Types implementing this trait guarantee that constructing their owned variant
/// will not panic/abort on allocation failure.
pub trait TryToOwned {
    /// The owned type produced by [`Self::try_to_owned`].
    type Owned: Borrow<Self>;

    /// Construct the owned version of `self`, falling back to an error on
    /// allocation failure rather than panicking.
    ///
    /// # Errors
    ///
    /// Returns [`TryToOwnedError`] if a capacity reservation or allocation fails.
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError>;
}

impl<T: TryClone> TryToOwned for T {
    type Owned = Self;

    #[inline]
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        Ok(self.try_clone()?)
    }
}

impl TryToOwned for str {
    type Owned = String;

    /// Turns a borrowed [`str`] into an owned [`String`], copying the bytes
    /// and appending the trailing NUL terminator on the global allocator.
    /// # Errors
    ///
    /// Returns [`TryToOwnedError`] if allocating or copying the string bytes fails.
    #[inline]
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        Ok(String::try_from_str(self)?)
    }
}

impl TryToOwned for CStr {
    type Owned = CString;

    /// Turns a borrowed [`CStr`] into an owned [`CString`], copying the bytes
    /// and appending the trailing NUL terminator on the global allocator.
    ///
    /// # Errors
    ///
    /// Returns [`TryToOwnedError`] if allocating or copying the C string bytes fails.
    #[inline]
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        Ok(CString::try_from_c_str(self)?)
    }
}

impl<T: TryClone> TryToOwned for [T] {
    type Owned = Vec<T>;

    /// Turns a borrowed `&[T]` into an owned [`Vec<T>`], cloning each element.
    ///
    /// # Errors
    ///
    /// Returns [`TryToOwnedError`] if allocating the buffer or cloning any element fails.
    #[inline]
    fn try_to_owned(&self) -> Result<Self::Owned, TryToOwnedError> {
        Vec::try_from(self).map_err(|e| match e {
            TryVecWithCloneError::Reserve(e) => e.into(),
            TryVecWithCloneError::Clone(e) => e.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::vec::Vec;

    #[test]
    fn try_to_owned_error_conversions() {
        let e: TryToOwnedError = TryReserveError::new_capacity_overflow().into();
        assert!(matches!(e, TryToOwnedError::Reserve(_)));
        let e: TryToOwnedError = AllocError.into();
        assert!(matches!(e, TryToOwnedError::Alloc(_)));
    }

    #[test]
    fn blanket_impl_covers_try_clone_types() {
        // Primitives implement `TryClone` in olive-core, so they inherit
        // `TryToOwned` through the blanket impl defined above.
        let x: u32 = 42;
        let owned: u32 = x.try_to_owned().unwrap();
        assert_eq!(owned, 42);

        // A heap-backed collection (`Vec`) likewise inherits it via its own
        // `TryClone` impl.
        let mut v: Vec<i32> = Vec::new();
        v.try_push(1).unwrap();
        v.try_push(2).unwrap();
        v.try_push(3).unwrap();
        let owned: Vec<i32> = v.try_to_owned().unwrap();
        assert_eq!(owned.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn c_str_try_to_owned_copies_the_bytes() {
        let borrowed: &CStr = c"foo";
        let owned: CString = borrowed.try_to_owned().unwrap();
        assert_eq!(owned.as_bytes(), borrowed.to_bytes());
        assert_eq!(owned.as_bytes_with_nul(), b"foo\0");
    }

    #[test]
    fn c_str_try_to_owned_handles_the_empty_string() {
        let borrowed: &CStr = c"";
        let owned: CString = borrowed.try_to_owned().unwrap();
        assert_eq!(owned.as_bytes_with_nul(), b"\0");
        assert_eq!(owned.as_bytes(), borrowed.to_bytes());
    }
}
