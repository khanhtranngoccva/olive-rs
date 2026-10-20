//! A pass-through allocator that counts allocation and deallocation traffic.

use core::alloc::Layout;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, StaticAllocator};

/// An [`Allocator`] that records how many times it has allocated and deallocated,
/// forwarding actual memory management to [`Global`].
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
    #[must_use]
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
        // Zero-sized layouts never touch the heap, so they are not counted,
        // and neither are failed allocations (only handed-out blocks count).
        let block = Global.allocate(layout)?;
        if layout.size() != 0 {
            self.allocations.fetch_add(1, Ordering::Release);
        }
        Ok(block)
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
