//! Collection-level allocation errors.
//!
//! The low-level [`AllocError`] is defined here (it must live in `olive_core`,
//! since this crate has no dependency on `olive_alloc`) and is re-exported by
//! [`olive_alloc::allocator`] alongside the
//! [`Allocator`](olive_alloc::allocator::Allocator) trait it originates from.
//! This module also defines the higher-level [`TryReserveError`], which a
//! *collection* returns when reserving capacity fails:
//!
//! * [`TryReserveError`] — a capacity *reservation* on a collection failed. It
//!   distinguishes between an arithmetic overflow while computing the new
//!   capacity and an actual out-of-memory from the allocator.
//!
//! Together these let downstream code reason about failure the same way it does
//! with `std`, except that the value is returned to the caller instead of
//! triggering an abort.

use core::alloc::Layout;
use core::error::Error;
use core::fmt;

/// Indicates an allocation failure that may be due to resource exhaustion or to
/// something wrong when combining the given input arguments with this allocator.
///
/// A unit struct, matching the shape of the (unstable) standard library's
/// `core::alloc::AllocError`. It carries no payload because the allocator
/// reports OOM without additional context.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl Error for AllocError {}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// The kind of failure behind a [`TryReserveError`].
///
/// Mirrors the standard library's private `TryReserveErrorKind`: either the
/// computed capacity overflowed before any allocation was attempted, or the
/// allocator refused the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TryReserveErrorKind {
    /// The computed capacity exceeded the collection's maximum.
    CapacityOverflow,
    /// The memory allocator returned an error for the given [`Layout`].
    AllocError {
        /// The allocation layout that failed.
        layout: Layout,
    },
}

impl TryReserveErrorKind {
    /// Returns `true` if the reservation failed because the allocator ran out
    /// of memory.
    #[inline]
    #[must_use]
    pub const fn is_alloc(&self) -> bool {
        matches!(self, Self::AllocError { .. })
    }

    /// Returns `true` if the reservation failed due to capacity arithmetic
    /// overflowing.
    #[inline]
    #[must_use]
    pub const fn is_capacity_overflow(&self) -> bool {
        matches!(self, Self::CapacityOverflow)
    }
}

/// Error returned when reserving capacity on a collection fails.
///
/// This is the fallible analogue of the panic that `Vec::reserve`,
/// `String::reserve`, and friends raise on out-of-memory. Unlike those, the
/// value is returned to the caller so the program can react (free memory, back
/// off, retry, or degrade gracefully).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

impl TryReserveError {
    /// Construct an error representing a failed heap allocation, carrying the
    /// [`Layout`] of the allocation that failed.
    #[inline]
    #[must_use]
    pub const fn new_alloc(layout: Layout) -> Self {
        Self {
            kind: TryReserveErrorKind::AllocError { layout },
        }
    }

    /// Construct an error representing a capacity computation that overflowed
    /// before any allocation was attempted.
    #[inline]
    #[must_use]
    pub const fn new_capacity_overflow() -> Self {
        Self {
            kind: TryReserveErrorKind::CapacityOverflow,
        }
    }

    /// Enumerate which kind of reservation failure occurred.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> TryReserveErrorKind {
        self.kind
    }

    /// Returns `true` if the failure was a failed heap allocation.
    #[inline]
    #[must_use]
    pub const fn is_alloc(&self) -> bool {
        self.kind.is_alloc()
    }

    /// Returns `true` if the failure was a capacity arithmetic overflow.
    #[inline]
    #[must_use]
    pub const fn is_capacity_overflow(&self) -> bool {
        self.kind.is_capacity_overflow()
    }
}

impl fmt::Debug for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TryReserveErrorKind::CapacityOverflow => {
                f.write_str("TryReserveError::CapacityOverflow")
            }
            TryReserveErrorKind::AllocError { .. } => f
                .debug_struct("TryReserveError::AllocError")
                .finish_non_exhaustive(),
        }
    }
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TryReserveErrorKind::CapacityOverflow => {
                f.write_str("capacity overflow when reserving space for a collection")
            }
            TryReserveErrorKind::AllocError { .. } => {
                f.write_str("memory allocation failed when reserving capacity")
            }
        }
    }
}

impl Error for TryReserveError {}

impl From<TryReserveErrorKind> for TryReserveError {
    #[inline]
    fn from(kind: TryReserveErrorKind) -> Self {
        Self { kind }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::format;

    #[test]
    fn capacity_overflow_round_trip() {
        let err = TryReserveError::new_capacity_overflow();
        assert!(err.is_capacity_overflow());
        assert!(!err.is_alloc());
        assert_eq!(err.kind(), TryReserveErrorKind::CapacityOverflow);
    }

    #[test]
    fn alloc_error_carries_layout() {
        let layout = Layout::new::<u64>();
        let err = TryReserveError::new_alloc(layout);
        assert!(err.is_alloc());
        assert!(!err.is_capacity_overflow());
        match err.kind() {
            TryReserveErrorKind::AllocError { layout } => {
                assert_eq!(layout.size(), 8);
                assert_eq!(layout.align(), 8);
            }
            _ => panic!("expected AllocError kind"),
        }
    }

    #[test]
    fn display_mentions_allocation() {
        let co = TryReserveError::new_capacity_overflow();
        let al = TryReserveError::new_alloc(Layout::new::<u8>());
        assert!(format!("{co}").contains("overflow"));
        assert!(format!("{al}").contains("allocation"));
    }
}
