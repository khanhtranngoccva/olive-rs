//! Construction methods for [`VecDeque`].
//!
//! # Invariants
//!
//! Every constructor establishes the canonical empty state: `len == 0`,
//! `head == WrappedIndex::zero()`, and a buffer whose capacity is at least the
//! requested amount (exactly `0` for `new`/`new_in`). No constructor allocates
//! more than requested, and none can fail except the `try_with_capacity*`
//! pair, which surface allocation failure and capacity overflow as
//! [`TryReserveError`] without leaking or leaving a half-built deque.

use super::VecDeque;
use super::wrapped_index::WrappedIndex;
use crate::alloc::Global;
use crate::raw_vec::RawVec;
use olive_core::alloc::Allocator;
use olive_core::alloc_errors::TryReserveError;
use olive_core::prelude::TryDefault;

// ---------------------------------------------------------------------------
// Constructors — global allocator
// ---------------------------------------------------------------------------

impl<T> VecDeque<T, Global> {
    /// Constructs a new, empty `VecDeque<T>`.
    ///
    /// The deque will not allocate until elements are pushed onto it.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }

    /// Constructs a new, empty `VecDeque<T>` with exactly the given capacity
    /// on the global allocator.
    ///
    /// The deque will contain no elements and can hold `capacity` of them
    /// before needing to reallocate; unlike the fallible growth methods, this
    /// constructor does not allocate more than requested. If `capacity` is
    /// zero, the deque will not allocate.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned deque is deterministic. For
    /// non-zero-sized `T`, the capacity is exactly `capacity`, so callers can
    /// rely on `self.capacity() == capacity` when asserting on allocations in
    /// tests. For zero-sized `T`, the capacity is always reported as
    /// `usize::MAX` regardless of the request (and no memory is allocated).
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, Global)
    }
}

// ---------------------------------------------------------------------------
// Constructors — custom allocator
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Like [`Self::new`], but parameterized over the choice of allocator for
    /// the returned `VecDeque`.
    #[must_use]
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            buf: RawVec::new_in(alloc),
            head: WrappedIndex::zero(),
            len: 0,
        }
    }

    /// Like [`Self::try_with_capacity`], but parameterized over the choice of
    /// allocator for the returned `VecDeque`.
    ///
    /// The deque will contain no elements and can hold `capacity` of them
    /// before needing to reallocate; unlike the fallible growth methods, this
    /// constructor does not allocate more than requested. If `capacity` is
    /// zero, the deque will not allocate.
    ///
    /// ## Determinism
    ///
    /// The capacity of the returned deque is deterministic. For
    /// non-zero-sized `T`, the capacity is exactly `capacity`, so callers can
    /// rely on `self.capacity() == capacity` when asserting on allocations in
    /// tests. For zero-sized `T`, the capacity is always reported as
    /// `usize::MAX` regardless of the request (and no memory is allocated).
    ///
    /// The method does not ask for more allocation memory than needed: if
    /// the allocator returns a buffer larger than the request, the reported
    /// capacity is still clamped to the requested `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`TryReserveError`] if the requested capacity overflows or the
    /// allocation fails.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let buf = RawVec::try_with_capacity_in(capacity, alloc)?;
        Ok(Self {
            buf,
            head: WrappedIndex::zero(),
            len: 0,
        })
    }
}

// An empty deque never allocates, so its default construction is infallible —
// even when the allocator itself can be defaulted fallibly. The bound on `A`
// mirrors std's `Default` impls for allocator-parameterized types: the
// allocator must have a canonical default of its own.
impl<T, A: Allocator + TryDefault> TryDefault for VecDeque<T, A> {
    #[inline]
    fn try_default() -> Result<Self, olive_core::prelude::TryDefaultError> {
        let alloc = A::try_default()?;
        Ok(Self::new_in(alloc))
    }
}

impl<T> Default for VecDeque<T, Global> {
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
    use crate::alloc::{AllocError, Layout};
    use crate::test_helpers::FailAlloc;
    use core::mem::size_of;
    use std::sync::Arc;

    // --- new / new_in --------------------------------------------------------

    #[test]
    fn new_is_empty_and_unallocated() {
        let dq: VecDeque<i32> = VecDeque::new();
        assert!(dq.is_empty());
        assert_eq!(dq.len(), 0);
        assert_eq!(dq.capacity(), 0);
        // Empty slices in both halves: nothing has been allocated yet.
        let (a, b) = dq.as_slices();
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn new_in_carries_the_given_allocator_and_drops_it_exactly_once() {
        let drops = Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let dq: VecDeque<i32, _> = VecDeque::new_in(alloc);
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), 0);
        drop(dq);
        // The allocator handle was consumed by the deque and dropped with it.
        assert_eq!(drops.get(), 1);
    }

    /// An allocator that fails every allocation, so any accidental
    /// allocation during construction is caught immediately.
    #[derive(Debug)]
    struct NoAllocProbe;
    unsafe impl Allocator for NoAllocProbe {
        fn allocate(&self, _layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
            Err(AllocError)
        }
        unsafe fn deallocate(&self, _ptr: core::ptr::NonNull<u8>, _layout: Layout) {}
    }

    #[test]
    fn new_in_does_not_touch_the_allocator() {
        // `new_in` must never call the allocator — not even to probe or
        // pre-allocate — because an empty deque owns no memory. A failing
        // allocator proves this: if `new_in` touched it, we'd see an error.
        let dq: VecDeque<i32, NoAllocProbe> = VecDeque::new_in(NoAllocProbe);
        assert!(dq.is_empty());
        assert_eq!(dq.capacity(), 0);
        // Dropping an unallocated deque must also skip the allocator.
        drop(dq);
    }

    // --- try_with_capacity / try_with_capacity_in ----------------------------

    #[test]
    fn try_with_capacity_allocates_at_least_requested() {
        let dq = VecDeque::<i32>::try_with_capacity(10).expect("allocation ok");
        assert!(dq.is_empty());
        assert_eq!(dq.len(), 0);
        assert!(dq.capacity() >= 10);
    }

    #[test]
    fn try_with_capacity_zero_never_allocates() {
        // A zero-capacity request behaves exactly like `new`: no allocation,
        // capacity reported as 0 for non-ZSTs.
        let dq = VecDeque::<i32>::try_with_capacity(0).expect("empty allocation ok");
        assert_eq!(dq.capacity(), 0);
        assert!(dq.is_empty());
    }

    #[test]
    fn try_with_capacity_zst_succeeds_without_allocation() {
        // Zero-sized types report `usize::MAX` capacity without ever touching
        // the heap.
        let dq = VecDeque::<()>::try_with_capacity(1_000_000).expect("ZST allocation ok");
        assert_eq!(dq.capacity(), usize::MAX);
        assert!(dq.is_empty());
    }

    #[test]
    fn try_with_capacity_overflow_returns_capacity_overflow() {
        // A capacity whose byte size overflows must be reported as a capacity
        // overflow, not an allocation error. (`VecDeque` is not `Debug` yet, so we
        // match rather than use `expect_err`.)
        let err = match VecDeque::<u8>::try_with_capacity(usize::MAX) {
            Err(e) => e,
            Ok(_) => panic!("expected capacity overflow"),
        };
        assert!(err.is_capacity_overflow());
    }

    #[test]
    fn try_with_capacity_oom_returns_alloc_error_kind() {
        // Prove an allocation failure surfaces as an AllocError-kind
        // `TryReserveError` (distinct from CapacityOverflow).
        let err = match VecDeque::<i32, FailAlloc>::try_with_capacity_in(8, FailAlloc) {
            Err(e) => e,
            Ok(_) => panic!("expected allocation failure"),
        };
        assert!(err.is_alloc());
        assert!(!err.is_capacity_overflow());
    }

    #[test]
    fn try_with_capacity_in_uses_custom_allocator() {
        let drops = Arc::new(crate::test_helpers::DropCounter::new());
        let alloc = crate::test_helpers::DropCountingAlloc::new(drops.clone());
        let dq = VecDeque::<i32, _>::try_with_capacity_in(16, alloc).expect("allocation ok");
        assert!(dq.capacity() >= 16);
        assert!(dq.is_empty());
        drop(dq);
        assert_eq!(drops.get(), 1);
    }

    // --- Default / TryDefault ------------------------------------------------

    #[test]
    fn default_matches_new() {
        let d: VecDeque<i32> = Default::default();
        let n: VecDeque<i32> = VecDeque::new();
        assert_eq!(d.len(), n.len());
        assert_eq!(d.capacity(), n.capacity());
        assert!(d.is_empty());
    }

    #[test]
    fn try_default_matches_new() {
        let t: VecDeque<i32> = TryDefault::try_default().expect("infallible");
        assert!(t.is_empty());
        assert_eq!(t.capacity(), 0);
    }

    #[test]
    fn try_default_with_custom_allocator_defaults_the_allocator() {
        // An allocator whose `try_default` fails must propagate as an error —
        // proving the impl actually calls `A::try_default()` rather than using
        // a hardcoded default.
        #[derive(Debug)]
        struct NoDefaultAlloc;
        unsafe impl Allocator for NoDefaultAlloc {
            fn allocate(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
                Global.allocate(layout)
            }
            unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
                unsafe { Global.deallocate(ptr, layout) };
            }
        }
        impl TryDefault for NoDefaultAlloc {
            fn try_default() -> Result<Self, olive_core::prelude::TryDefaultError> {
                Err(olive_core::prelude::TryDefaultError::Other(
                    "no canonical allocator",
                ))
            }
        }

        let res: Result<VecDeque<i32, NoDefaultAlloc>, _> = TryDefault::try_default();
        assert!(res.is_err(), "allocator default failure must propagate");
    }

    // --- Cross-checks ---------------------------------------------------------

    #[test]
    fn global_and_generic_constructors_agree() {
        // Both routes produce equivalent empty deques on the global allocator.
        let g = VecDeque::<i32>::new();
        let generic = VecDeque::<i32>::new_in(Global);
        assert_eq!(g.len(), generic.len());
        assert_eq!(g.capacity(), generic.capacity());

        let g = VecDeque::<i32>::try_with_capacity(7).expect("global ok");
        let generic = VecDeque::<i32>::try_with_capacity_in(7, Global).expect("generic ok");
        assert_eq!(g.len(), generic.len());
        assert!(g.capacity() >= 7);
        assert!(generic.capacity() >= 7);
    }

    #[test]
    fn zst_new_reports_max_capacity() {
        // Consistency with `RawVec`: a ZST deque reports `usize::MAX` capacity
        // even when constructed empty.
        let dq: VecDeque<()> = VecDeque::new();
        assert_eq!(dq.capacity(), usize::MAX);
        assert_eq!(size_of::<()>(), 0);
    }
}
