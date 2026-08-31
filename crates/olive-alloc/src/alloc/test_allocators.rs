//! Shared test-only allocators.
//!
//! These are available to every test target in the crate (declared from
//! [`super::mod`] under `#[cfg(test)]`) so that tests which need to observe
//! allocation traffic — e.g. verifying a block is freed even when a pointee's
//! destructor panics — can share one well-tested implementation instead of each
//! rolling its own.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::{AllocError, Allocator, Global, Layout, StaticAllocator};

/// An [`Allocator`] that records how many times it has allocated and deallocated,
/// forwarding actual memory management to [`Global`].
///
/// The counters are per-instance [`AtomicUsize`]s (rather than process-global
/// statics), so concurrent or repeated tests cannot interfere with one another
/// and no reset step is required: construct a fresh allocator per test and read
/// its counters at the end. Because the counters live behind interior mutability
/// and the methods take `&self`, a test may pass `&allocator` into the `_in`
/// constructors and still hold onto the original to inspect the counts after the
/// box (which stores a copy of the reference) is dropped.
///
/// Being a [`StaticAllocator`], both the value and a reference to it can be used
/// with the `_in` constructors without installing a process-wide global allocator.
pub struct CountingAllocator {
    /// Number of non-zero-size blocks handed out by [`Self::allocate`].
    allocations: AtomicUsize,
    /// Number of non-zero-size blocks released by [`Self::deallocate`].
    deallocations: AtomicUsize,
}

impl Default for CountingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CountingAllocator {
    /// Creates a freshly-zeroed counter.
    #[inline]
    pub const fn new() -> Self {
        Self {
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
        }
    }

    /// The number of non-zero-size blocks allocated so far.
    #[inline]
    pub fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Acquire)
    }

    /// The number of non-zero-size blocks deallocated so far.
    #[inline]
    pub fn deallocations(&self) -> usize {
        self.deallocations.load(Ordering::Acquire)
    }
}

unsafe impl Allocator for CountingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Zero-sized layouts never touch the heap, so they are not counted.
        if layout.size() != 0 {
            self.allocations.fetch_add(1, Ordering::Release);
        }
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            self.deallocations.fetch_add(1, Ordering::Release);
        }
        // SAFETY: forwarded from a valid drop site honoring `Global`'s contract.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

// SAFETY: the only state held is a pair of monotonic counters mutated through
// `&self`; no two instances ever manage the same block, so there is no aliasing
// hazard, and dropping the allocator invalidates nothing. A reference to it is
// therefore equivalent to the allocator itself.
unsafe impl StaticAllocator for CountingAllocator {}
