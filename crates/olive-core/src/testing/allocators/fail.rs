//! A pass-through allocator that always fails, for OOM-path tests.

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone, AllocatorTryDefault};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// An allocator whose every allocation fails. Used to exercise OOM paths.
#[derive(Debug, Default)]
pub struct FailAlloc;

// SAFETY: never hands out memory for non-zero-sized layouts, so there is
// nothing to free on deallocate; zero-sized layouts delegate to `Global`,
// which handles them without touching the heap.
unsafe impl Allocator for FailAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Zero-sized allocations never touch the heap, so the failure hook
        // does not fire for them, matching the other test allocators.
        if layout.size() == 0 {
            return Global.allocate(layout);
        }
        Err(AllocError)
    }
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

impl TryClone for FailAlloc {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(Self)
    }
}

// SAFETY: all operations are trivially safe; cloning is infallible.
unsafe impl AllocatorTryClone for FailAlloc {}

impl TryDefault for FailAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self)
    }
}

// SAFETY: `FailAlloc` is a stateless ZST that delegates every zero-sized
// layout to `Global`, the process-wide singleton backing store.
unsafe impl AllocatorTryDefault for FailAlloc {}
