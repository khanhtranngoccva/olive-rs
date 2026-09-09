//! Specialized test allocators that implement [`TryDefault`] in addition to
//! [`Allocator`], used to exercise failure paths in `TryDefault` impls for
//! smart pointers.

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

