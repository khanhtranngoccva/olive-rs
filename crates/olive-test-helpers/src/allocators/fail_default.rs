//! A pass-through allocator whose `TryDefault` fails.

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::Global;
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
