//! Construction methods for [`BinaryHeap`].
use super::BinaryHeap;
use crate::alloc::Global;
use crate::vec::Vec;
use olive_core::alloc::{Allocator, AllocatorTryDefault};
use olive_core::alloc_errors::TryReserveError;
use olive_core::prelude::TryDefault;

// ---------------------------------------------------------------------------
// Constructors — global allocator
// ---------------------------------------------------------------------------

impl<T> BinaryHeap<T, Global> {
    /// Constructs a new, empty `BinaryHeap<T>` in the [`Global`] allocator.
    ///
    /// The heap will not allocate until elements are pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        // SAFETY: empty heaps always satisfy invariants.
        unsafe { Self::with_data(Vec::new()) }
    }

    /// Constructs a new, empty `BinaryHeap<T>` with exactly the given capacity
    /// on the global allocator.
    ///
    /// The heap will contain no elements and can hold `capacity` of them
    /// before needing to reallocate; unlike the fallible growth methods, this
    /// constructor does not allocate more than requested.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned heap is deterministic. For
    /// non-zero-sized `T`, the capacity is exactly `capacity`, so callers can
    /// rely on `self.capacity() == capacity` when asserting on allocations in
    /// tests. For zero-sized `T`, the capacity is always reported as
    /// `usize::MAX` regardless of the request (and no memory is allocated).
    ///
    /// The method does not ask for more allocation memory than needed, and if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        // SAFETY: empty heaps always satisfy invariants.
        Ok(unsafe { Self::with_data(Vec::try_with_capacity(capacity)?) })
    }
}

// ---------------------------------------------------------------------------
// Constructors — custom allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Like [`Self::new`], but parameterized over the choice of allocator for
    /// the returned `BinaryHeap`.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        // SAFETY: empty heaps always satisfy invariants.
        unsafe { Self::with_data(Vec::new_in(alloc)) }
    }

    /// Like [`Self::try_with_capacity`], but parameterized over the choice of
    /// allocator for the returned `BinaryHeap`.
    ///
    /// The heap will contain no elements and can hold `capacity` of them
    /// before needing to reallocate; unlike the fallible growth methods, this
    /// constructor does not allocate more than requested. If `capacity` is
    /// zero, the heap will not allocate.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned heap is deterministic, exactly as
    /// described on [`Self::try_with_capacity`].
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        // SAFETY: empty heaps always satisfy invariants.
        Ok(unsafe { Self::with_data(Vec::try_with_capacity_in(capacity, alloc)?) })
    }
}

// ---------------------------------------------------------------------------
// Default / TryDefault
// ---------------------------------------------------------------------------

impl<T, A: AllocatorTryDefault> TryDefault for BinaryHeap<T, A> {
    #[inline]
    fn try_default() -> Result<Self, olive_core::prelude::TryDefaultError> {
        let alloc = A::try_default()?;
        Ok(Self::new_in(alloc))
    }
}

impl<T> Default for BinaryHeap<T, Global> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::test_helpers::{DropCounter, DropCountingAlloc, FailAlloc};
    use std::sync::Arc;

    // --- new / new_in --------------------------------------------------------

    #[test]
    fn new_is_empty_and_unallocated() {
        let heap: BinaryHeap<i32> = BinaryHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.capacity(), 0);
    }

    #[test]
    fn new_in_carries_the_given_allocator_and_drops_it_exactly_once() {
        let drops = Arc::new(DropCounter::new());
        let alloc = DropCountingAlloc::new(drops.clone());
        let heap: BinaryHeap<i32, _> = BinaryHeap::new_in(alloc);
        assert!(heap.is_empty());
        assert_eq!(heap.capacity(), 0);
        drop(heap);
        // The allocator handle was consumed by the heap and dropped with it.
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn new_in_does_not_touch_the_allocator() {
        let heap: BinaryHeap<i32, FailAlloc> = BinaryHeap::new_in(FailAlloc);
        assert!(heap.is_empty());
        assert_eq!(heap.capacity(), 0);
        // Dropping an unallocated heap must also skip the allocator.
        drop(heap);
    }

    // --- try_with_capacity / try_with_capacity_in ----------------------------

    #[test]
    fn try_with_capacity_is_deterministic() {
        // `Vec::try_with_capacity` clamps to the request, so the capacity is
        // exact — the property the Determinism docs promise.
        let heap = BinaryHeap::<i32>::try_with_capacity(10).expect("allocation ok");
        assert!(heap.is_empty());
        assert_eq!(heap.capacity(), 10);
    }

    #[test]
    fn try_with_capacity_zero_never_allocates() {
        // A zero-capacity request behaves exactly like `new`.
        let heap = BinaryHeap::<i32>::try_with_capacity(0).expect("empty allocation ok");
        assert_eq!(heap.capacity(), 0);
        assert!(heap.is_empty());
    }

    #[test]
    fn try_with_capacity_zst_succeeds_without_allocation() {
        // Zero-sized types report `usize::MAX` capacity without ever touching
        // the heap.
        let heap = BinaryHeap::<()>::try_with_capacity(1_000_000).expect("ZST allocation ok");
        assert_eq!(heap.capacity(), usize::MAX);
        assert!(heap.is_empty());
    }

    #[test]
    fn try_with_capacity_overflow_returns_capacity_overflow() {
        // A capacity whose byte size overflows must be reported as a capacity
        // overflow, not an allocation error.
        let err = match BinaryHeap::<u8>::try_with_capacity(usize::MAX) {
            Err(e) => e,
            Ok(_) => panic!("expected capacity overflow error"),
        };
        assert!(err.is_capacity_overflow());
    }

    #[test]
    fn try_with_capacity_oom_returns_alloc_error_kind() {
        // Prove an allocation failure surfaces as an AllocError-kind
        // `TryReserveError` (distinct from CapacityOverflow).
        let err = match BinaryHeap::<i32, FailAlloc>::try_with_capacity_in(8, FailAlloc) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(!err.is_capacity_overflow());
    }

    #[test]
    fn try_with_capacity_in_uses_custom_allocator() {
        let drops = Arc::new(DropCounter::new());
        let alloc = DropCountingAlloc::new(drops.clone());
        let heap = BinaryHeap::<i32, _>::try_with_capacity_in(16, alloc).expect("allocation ok");
        assert_eq!(heap.capacity(), 16);
        assert!(heap.is_empty());
        drop(heap);
        assert_eq!(drops.get(), 1);
    }

    // --- Default / TryDefault ------------------------------------------------

    #[test]
    fn default_matches_new() {
        let d: BinaryHeap<i32> = Default::default();
        let n: BinaryHeap<i32> = BinaryHeap::new();
        assert_eq!(d.len(), n.len());
        assert_eq!(d.capacity(), n.capacity());
        assert!(d.is_empty());
    }

    #[test]
    fn try_default_global_produces_empty_heap() {
        let d: BinaryHeap<i32, Global> = TryDefault::try_default().expect("infallible");
        assert!(d.is_empty());
        assert_eq!(d.capacity(), 0);
    }

    // --- Cross-checks ----------------------------------------------------------

    #[test]
    fn global_and_generic_constructors_agree() {
        // Both routes produce equivalent empty heaps on the global allocator.
        let g = BinaryHeap::<i32>::new();
        let generic = BinaryHeap::<i32>::new_in(Global);
        assert_eq!(g.len(), generic.len());
        assert_eq!(g.capacity(), generic.capacity());

        let g = BinaryHeap::<i32>::try_with_capacity(7).expect("global ok");
        let generic = BinaryHeap::<i32>::try_with_capacity_in(7, Global).expect("generic ok");
        assert_eq!(g.len(), generic.len());
        assert_eq!(g.capacity(), generic.capacity());
    }
}
