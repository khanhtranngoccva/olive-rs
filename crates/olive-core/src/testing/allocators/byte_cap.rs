//! A pass-through allocator that rejects any allocation above a byte cap.

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

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
    #[must_use]
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
