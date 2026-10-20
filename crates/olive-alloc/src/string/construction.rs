//! Module that pertains to construction of [`String`] structures.

use olive_core::{
    alloc::{Allocator, AllocatorTryDefault},
    alloc_errors::TryReserveError,
    try_traits::{TryDefault, TryDefaultError},
};

use super::String;
use crate::{alloc::Global, vec::Vec};

// ──────────────────────────────────────────────────────────────────────────────
// Global allocator implementation block
// ──────────────────────────────────────────────────────────────────────────────

impl String {
    /// Creates an empty `String`.
    ///
    /// The returned string contains no characters and has no allocated
    /// capacity.
    #[inline]
    pub const fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Creates an empty `String` with exactly the given capacity.
    ///
    /// The string can hold `cap` bytes without reallocating; unlike the
    /// fallible growth methods, this constructor does not allocate more than
    /// requested. If `cap` is zero, no memory is allocated.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned string is deterministic. Since the
    /// backing buffer stores `u8` (a non-zero-sized type), the capacity is
    /// exactly `cap`, so callers can rely on `self.capacity() == cap` when
    /// asserting on allocations in tests.
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `cap`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity(cap: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(cap, Global)
    }

    /// Converts a `&str` into a `String` by copying its contents.
    #[inline]
    pub fn try_from_str(s: &str) -> Result<Self, TryReserveError> {
        Self::try_from_str_in(s, Global)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Generic allocator implementation block
// ──────────────────────────────────────────────────────────────────────────────

impl<A: Allocator> String<A> {
    /// Creates an empty `String` using the given allocator.
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            buf: Vec::new_in(alloc),
        }
    }

    /// Creates an empty `String` with exactly the given capacity, allocating
    /// through `alloc`.
    ///
    /// The string can hold `cap` bytes without reallocating; unlike the
    /// fallible growth methods, this constructor does not allocate more than
    /// requested. If `cap` is zero, no memory is allocated.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned string is deterministic. Since the
    /// backing buffer stores `u8` (a non-zero-sized type), the capacity is
    /// exactly `cap`, so callers can rely on `self.capacity() == cap` when
    /// asserting on allocations in tests.
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `cap`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// initial allocation fails.
    #[inline]
    pub fn try_with_capacity_in(cap: usize, alloc: A) -> Result<Self, TryReserveError> {
        let buf = Vec::<u8, A>::try_with_capacity_in(cap, alloc)?;
        Ok(Self { buf })
    }

    /// Converts a `&str` into a `String` by copying its contents, allocating
    /// through `alloc`.
    #[inline]
    pub fn try_from_str_in(s: &str, alloc: A) -> Result<Self, TryReserveError> {
        let mut buf = Vec::<u8, A>::try_with_capacity_in(s.len(), alloc)?;
        // Reserve succeeded, so appending cannot fail; each byte is `Copy`.
        for b in s.as_bytes() {
            buf.try_push(*b).expect("capacity was just reserved");
        }
        Ok(Self { buf })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Default implementation block
// ──────────────────────────────────────────────────────────────────────────────

impl Default for String {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<A: AllocatorTryDefault> TryDefault for String<A> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new_in(A::try_default()?))
    }
}
