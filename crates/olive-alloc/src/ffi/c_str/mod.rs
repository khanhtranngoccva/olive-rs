//! A fully-fallible port of the standard library's `CString`.
//!
//! Differences from the standard library:
//!
//! - The type is generic over an [`Allocator`], defaulting to [`Global`].
//! - All allocation is fallible; constructors return a
//!   [`Result`](core::result::Result) instead of panicking on OOM.

mod cmp;
mod construction;
mod convert;
mod query;
mod reconstitution;
mod traits;

pub use convert::CStringIntoStringError;

use crate::alloc::{Allocator, Global};
use crate::boxed::Box;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use olive_core::alloc_errors::TryReserveError;

// ---------------------------------------------------------------------------
// CString
// ---------------------------------------------------------------------------

/// An owned, heap-allocated, NUL-terminated C string.
///
/// A `CString` is safe to hand to a foreign `char *` or `const char *`
/// interface: its bytes are guaranteed to end with exactly one NUL byte and to
/// contain no interior NUL bytes, so a `strlen` over the buffer terminates on
/// the first and only NUL.
///
/// Construction is guaranteed to be fallible when applicable.
pub struct CString<A: Allocator = Global> {
    inner: Box<[u8], A>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error returned by [`CString::try_new`]: the provided bytes contained an
/// interior NUL byte. Rust strings can contain NUL bytes in the middle, but C
/// strings can't.
///
/// The error carries no payload, it is returned with the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CStringNulError(usize);

impl CStringNulError {
    /// Returns the position of the interior NUL byte in the provided byte slice.
    #[must_use]
    pub fn nul_position(&self) -> usize {
        self.0
    }
}

impl Display for CStringNulError {
    /// Formats a message that shows the position of the interior NUL byte.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

impl Error for CStringNulError {}

/// Error returned by [`CString::try_new`]: either the input bytes contained a
/// NUL byte ([`TryCStringNewError::Nul`]) or an allocation failed
/// ([`TryCStringNewError::Reserve`]).
///
/// The error carries no payload, it is returned with the caller.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryCStringNewError {
    /// An interior NUL byte was found.
    Nul(CStringNulError),
    /// Appending the terminating NUL byte or shrinking the buffer into a
    /// boxed slice failed.
    Reserve(TryReserveError),
}

impl Debug for TryCStringNewError {
    ///  Formats a Debug message that shows the underlying failure.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nul(e) => Debug::fmt(e, f),
            Self::Reserve(..) => f
                .debug_tuple("TryCStringNewError::Reserve")
                .field(&format_args!(".."))
                .finish(),
        }
    }
}

impl Display for TryCStringNewError {
    /// Formats a message that mirrors the underlying failure's message.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nul(e) => Display::fmt(e, f),
            Self::Reserve(e) => Display::fmt(e, f),
        }
    }
}

impl Error for TryCStringNewError {}

/// Error returned by [`CString::try_from_vec_with_nul`], due to
/// either the provided bytes not being exactly one NUL-terminated
/// sequence or an allocation failure.
///
/// The error carries no payload, it is returned with the caller.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TryCStringFromVecWithNulError {
    /// An interior NUL byte was found at the contained position.
    InteriorNul(usize),
    /// The vector was empty or did not end with a NUL byte.
    NotNulTerminated,
    /// Shrinking the buffer into a boxed slice failed.
    Reserve(TryReserveError),
}

impl Debug for TryCStringFromVecWithNulError {
    /// Formats a Debug message that shows the underlying failure.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InteriorNul(position) => f
                .debug_tuple("TryCStringFromVecWithNulError::InteriorNul")
                .field(position)
                .finish(),
            Self::NotNulTerminated => f
                .debug_tuple("TryCStringFromVecWithNulError::NotNulTerminated")
                .finish(),
            Self::Reserve(..) => f
                .debug_tuple("TryCStringFromVecWithNulError::Reserve")
                .field(&format_args!(".."))
                .finish(),
        }
    }
}

/// Mirrors the underlying failure's message.
impl Display for TryCStringFromVecWithNulError {
    /// Formats a human-friendly display message for the error.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InteriorNul(position) => {
                write!(f, "nul byte found in provided data at position: {position}")
            }
            Self::NotNulTerminated => write!(f, "provided data was not terminated by nul"),
            Self::Reserve(e) => Display::fmt(e, f),
        }
    }
}

impl Error for TryCStringFromVecWithNulError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::alloc::CountingAllocator;
    use crate::vec::Vec;

    use super::CString;

    #[test]
    fn dropping_a_string_deallocates_its_buffer_once() {
        let alloc = CountingAllocator::new();

        drop(
            CString::try_from_vec_with_nul(
                Vec::try_from_slice_in(b"foo\0", &alloc)
                    .expect("allocation cannot fail for the counting allocator under test input"),
            )
            .expect("the checked input is exactly one nul-terminated string"),
        );

        assert_eq!(
            1,
            alloc.allocations(),
            "one allocate, one deallocate expected"
        );
        assert_eq!(
            1,
            alloc.deallocations(),
            "one allocate, one deallocate expected"
        );
    }
}
