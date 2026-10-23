//! A fully-fallible port of the standard library's `Vec`.
//!
//! Compared with the std original, three things differ:
//!
//! * Every operation that can grow the buffer — `push`, `insert`, `extend`,
//!   `resize`, capacity reservations, and every allocating constructor — returns
//!   a [`Result`] carrying an error instead of panicking on out-of-memory. The
//!   primary error is [`TryReserveError`]; operations that also clone elements
//!   additionally surface [`TryCloneError`].
//! * All allocation goes through the [`Allocator`] trait rather than free
//!   functions, giving a single swappable seam for custom allocators and OOM
//!   simulation in tests.
//! * Compiler-internal attributes (`#[lang = "exchange_vector_like"]`) and
//!   unstable helpers are omitted; this is a pure type with no compiler magic.
//!
//! Element cloning uses the fallible [`TryClone`] trait throughout, so a `Vec<T>`
//! can hold values whose own construction can fail (nested collections, boxes,
//! …) without ever panicking or aborting.

mod allocation;
mod construction;
mod convert;
mod drain;
mod iter;
mod mutation;
mod query;
mod reconstitution;
mod traits;
pub use allocation::{TryVecExtendFromWithinError, TryVecInsertError, TryVecSplitOffError};
pub use convert::TryVecIntoArrayError;
pub use drain::Drain;
pub use iter::IntoIter;
pub use mutation::{TryPushWithinCapacityError, TrySwapError, TryVecRemoveError};

use core::fmt;
use core::ptr;

use crate::alloc::{Allocator, Global};
use crate::raw_vec::RawVec;
use olive_core::alloc_errors::TryReserveError;
use olive_core::try_traits::try_clone::TryCloneError;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by fallible vector operations that may both reserve capacity
/// and clone elements.
///
/// Covers `try_from_elem`, `try_from_slice`, `try_resize`,
/// `try_extend_from_slice_with_rollback`, and so on — any operation whose failure
/// modes are limited to a capacity reservation ([`TryReserveError`]) or an element
/// clone failure ([`TryCloneError`]).
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecWithCloneError {
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// An element clone failed during a method that requires [`TryClone`].
    Clone(TryCloneError),
}

impl fmt::Debug for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecWithCloneError::Reserve")
                .field(e)
                .finish(),
            Self::Clone(e) => f
                .debug_tuple("TryVecWithCloneError::Clone")
                .field(e)
                .finish(),
        }
    }
}

impl fmt::Display for TryVecWithCloneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "vector operation failed: {e}"),
            Self::Clone(e) => write!(f, "vector operation failed: {e}"),
        }
    }
}

impl core::error::Error for TryVecWithCloneError {}

impl From<TryReserveError> for TryVecWithCloneError {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

impl From<TryCloneError> for TryVecWithCloneError {
    #[inline]
    fn from(err: TryCloneError) -> Self {
        Self::Clone(err)
    }
}

/// Error returned by [`Vec::try_from_fn`] when constructing or extending a vector
/// from a fallible closure.
///
/// The closure may fail with any error type `E`, and the capacity reservation
/// itself may also fail independently.
#[derive(Clone, PartialEq, Eq)]
pub enum TryVecWithClosureError<E> {
    /// A capacity reservation on the vector failed (overflow or OOM).
    Reserve(TryReserveError),
    /// The closure returned an error.
    Closure(E),
}

impl<E: fmt::Debug> fmt::Debug for TryVecWithClosureError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => f
                .debug_tuple("TryVecWithClosureError::Reserve")
                .field(e)
                .finish(),
            Self::Closure(e) => f
                .debug_tuple("TryVecWithClosureError::Closure")
                .field(e)
                .finish(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for TryVecWithClosureError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve(e) => write!(f, "vector construction failed: {e}"),
            Self::Closure(e) => write!(f, "vector construction failed: {e}"),
        }
    }
}

impl<E: core::error::Error> core::error::Error for TryVecWithClosureError<E> {}

impl<E> From<TryReserveError> for TryVecWithClosureError<E> {
    #[inline]
    fn from(err: TryReserveError) -> Self {
        Self::Reserve(err)
    }
}

// ---------------------------------------------------------------------------
// Vec
// ---------------------------------------------------------------------------

/// A growable contiguous list of items backed by the heap.
///
/// This is the fully-fallible analogue of `std::vec::Vec`: every operation that
/// could allocate or fail returns a [`Result`] instead of panicking. It is generic over
/// the [`Allocator`] used for all heap traffic (defaulting to [`Global`]).
pub struct Vec<T, A: Allocator = Global> {
    raw: RawVec<T, A>,
    len: usize,
}

/// Panic-aware rollback guard for fallible mutation loops.
///
/// If the guarded section unwinds (e.g. a clone or closure panics), `Drop`
/// truncates the vector back to `original_len`, ensuring any partially-written
/// tail is dropped exactly once.
///
/// Stores a raw `*mut Vec` to avoid conflicting with the `&mut self` borrows
/// used by the guarded loop body. The pointer is derived directly from the
/// enclosing `&mut self` (via `&raw mut *self`), so it carries mutable provenance
/// — no `*const → *mut` reborrow round-trip, which keeps Miri's Stacked Borrows
/// model happy.
struct RollbackGuard<T, A: Allocator>(*mut Vec<T, A>, usize);

impl<T, A: Allocator> Drop for RollbackGuard<T, A> {
    fn drop(&mut self) {
        // SAFETY: the pointer was obtained from a valid `&mut Vec` at guard
        // construction time; the Vec is alive for the entire scope (the guard
        // is a local that drops before the function returns).
        let vec_ref = unsafe { &mut *self.0 };
        vec_ref.truncate(self.1);
    }
}

// ---------------------------------------------------------------------------
// Fallible constructors — explicit allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        // SAFETY: the first `self.len` slots are initialized; dropping them in
        // place runs each element's destructor exactly once. The buffer itself
        // is freed by `RawVec`'s `Drop` immediately afterward
        // (even when drop_in_place panics).
        unsafe {
            let slice = ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len);
            ptr::drop_in_place(slice);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::{alloc::Layout, convert::Infallible};
    use olive_core::try_traits::try_clone::TryCloneError;
    use std::format;

    #[test]
    fn error_types_display() {
        let wce = TryVecWithCloneError::Clone(TryCloneError::Other("boom"));
        assert!(format!("{wce}").contains("boom"));

        let wcle = TryVecWithClosureError::Closure(TryCloneError::Other("boom"));
        assert!(format!("{wcle}").contains("boom"));

        let wcle =
            TryVecWithClosureError::<Infallible>::Reserve(TryReserveError::new_capacity_overflow());
        assert!(format!("{wcle}").contains("capacity overflow"));

        let wcle = TryVecWithClosureError::<Infallible>::Reserve(TryReserveError::new_alloc(
            Layout::new::<i32>(),
        ));
        assert!(
            format!("{wcle}").contains("4"),
            "should include layout size in an alloc error"
        );
    }
}
