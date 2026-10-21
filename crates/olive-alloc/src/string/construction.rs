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

#[cfg(test)]
mod tests {
    use core::ops::Deref;

    use super::*;
    use crate::alloc::CountingAllocator;
    use olive_test_helpers::{FailAlloc, FailDefaultAlloc};

    fn mk(s: &str) -> String {
        String::try_from_str(s).unwrap()
    }

    // ── String::new ───────────────────────────────────────────────────────────

    #[test]
    fn new_is_empty() {
        let s = String::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    // ── String::try_with_capacity ─────────────────────────────────────────────

    #[test]
    fn try_with_capacity_reserves_at_least_requested() {
        let s = String::try_with_capacity(16).unwrap();
        assert!(s.capacity() == 16);
        assert!(s.is_empty());
    }

    #[test]
    fn try_with_capacity_rejects_overflow() {
        // Deterministic failure: no allocator can satisfy a `usize::MAX` request.
        let err = String::try_with_capacity(usize::MAX).unwrap_err();
        assert!(err.is_capacity_overflow());
    }

    // ── String::try_from_str ──────────────────────────────────────────────────

    #[test]
    fn from_str_roundtrip() {
        let s = mk("héllo wörld ✓");
        assert_eq!(s.deref(), "héllo wörld ✓");
    }

    // ── String::new_in ────────────────────────────────────────────────────────

    #[test]
    fn new_in_failing_allocator_is_empty_without_allocation() {
        // Constructing an empty string must not touch the allocator at all,
        // so even a failing allocator yields a usable empty string.
        let s: String<FailAlloc> = String::new_in(FailAlloc);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    // ── String::try_with_capacity_in ──────────────────────────────────────────

    #[test]
    fn try_with_capacity_in_uses_given_allocator() {
        // Pass a reference: `CountingAllocator` is not `Copy`, but its counters
        // live behind interior mutability, so the string's stored
        // `&CountingAllocator` observes the same instance we hold here.
        let alloc = CountingAllocator::new();
        let mut s = String::try_with_capacity_in(16, &alloc).unwrap();
        s.try_push_str("hello").unwrap();
        assert!(alloc.allocations() >= 1);
        drop(s);
        assert!(alloc.deallocations() >= 1);
    }

    #[test]
    fn try_with_capacity_in_oom_returns_error() {
        // Deterministic failure: FailAlloc refuses every non-zero-size
        // allocation.
        let err = String::try_with_capacity_in(16, FailAlloc).unwrap_err();
        assert!(err.is_alloc());
    }

    // ── String::try_from_str_in ───────────────────────────────────────────────

    #[test]
    fn try_from_str_in_copies_content_into_custom_allocator() {
        let alloc = CountingAllocator::new();
        let s: String<&CountingAllocator> = String::try_from_str_in("payload", &alloc).unwrap();
        assert_eq!(s.deref(), "payload");
        // The content was copied through our counting allocator, not leaked
        // from the source slice.
        assert!(alloc.allocations() >= 1);
    }

    #[test]
    fn try_from_str_in_oom_returns_error() {
        let err = String::try_from_str_in("payload", FailAlloc).unwrap_err();
        assert!(err.is_alloc());
    }

    // ── Default / TryDefault ──────────────────────────────────────────────────

    #[test]
    fn default_is_empty() {
        let s = String::default();
        assert!(s.is_empty());
        assert_eq!(s.deref(), "");
    }

    #[test]
    fn try_default_success_yields_empty_string() {
        // FailAlloc's TryDefault yields an allocator without allocating, so
        // even a failing allocator can produce an empty string.
        let s = <String<FailAlloc> as TryDefault>::try_default().unwrap();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn try_default_failure_propagates_allocator_error() {
        let e = <String<FailDefaultAlloc> as TryDefault>::try_default().unwrap_err();
        assert!(matches!(
            e,
            TryDefaultError::Other("allocator default failed")
        ));
    }
}
