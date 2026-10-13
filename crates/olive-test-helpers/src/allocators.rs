//! Shared test-only allocators.
//!
//! These are available to every test target in the workspace except the core
//! crate, so that tests which need to observe allocation traffic — e.g.
//! verifying a block is freed even when a pointee's destructor panics —
//! can share one well-known implementation.

use core::alloc::Layout;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, StaticAllocator};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// An allocator whose `TryDefault` always fails, used to exercise the
/// allocator-default-failure path in `Arc::try_default` and
/// `Weak::try_default`.
#[derive(Debug)]
pub struct FailDefaultAlloc;

// SAFETY: delegates all operations to `Global`; no extra invariants.
unsafe impl Allocator for FailDefaultAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl TryDefault for FailDefaultAlloc {
    fn try_default() -> Result<Self, TryDefaultError> {
        Err(TryDefaultError::Other("allocator default failed"))
    }
}

/// An allocator that rejects any single allocation whose size exceeds a
/// configured byte cap, delegating smaller requests to
/// [`allocator_api2::alloc::Global`].
#[derive(Debug, Clone)]
pub struct ByteCapAlloc {
    /// Maximum bytes allowed in a single allocation. Requests larger than this
    /// are rejected with [`AllocError`].
    pub max_bytes: usize,
}

impl ByteCapAlloc {
    /// Builds an allocator that permits single allocations up to `max_bytes`.
    pub const fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }
}

// SAFETY: passes through to `Global` for allocations within the cap; freed
// blocks were originally handed out by `Global`, so deallocation is safe.
unsafe impl Allocator for ByteCapAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() > self.max_bytes {
            return Err(AllocError);
        }
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl TryDefault for ByteCapAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new(usize::MAX))
    }
}

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
