//! Specialized test allocators that implement [`TryDefault`] in addition to
//! [`Allocator`], used to exercise allocator-specific code paths in tests.

use core::alloc::Layout;
use core::ptr::NonNull;
use olive_core::alloc::{AllocError, Allocator};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// An allocator whose `TryDefault` always fails, used to exercise the
/// allocator-default-failure path in `Arc::try_default` and
/// `Weak::try_default`.
#[derive(Debug)]
pub struct FailDefaultAlloc;

// SAFETY: delegates all operations to `Global`; no extra invariants.
unsafe impl Allocator for FailDefaultAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        crate::alloc::Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl TryDefault for FailDefaultAlloc {
    fn try_default() -> Result<Self, TryDefaultError> {
        Err(TryDefaultError::Other("allocator default failed"))
    }
}

/// An allocator that rejects any single allocation whose size exceeds a
/// configured byte cap, delegating smaller requests to [`crate::alloc::Global`].
///
/// Useful for exercising the over-hint absorption in `try_from_iter_in`: an
/// iterator advertising a large upper bound triggers a big upfront batch
/// reserve that this allocator refuses; the constructor absorbs that failure and
/// falls back to incremental per-element growth where each small reserve
/// succeeds or use exact reservation.
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
        crate::alloc::Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl TryDefault for ByteCapAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new(usize::MAX))
    }
}
