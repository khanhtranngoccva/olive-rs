//! A pass-through allocator whose `TryClone` succeeds only while a shared clone
//! budget lasts.

extern crate std;

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

use super::super::fallible_ops::CloneBudget;

/// An [`Allocator`] whose allocation forwards to `Global` but whose
/// [`TryClone`] succeeds only while a shared [`CloneBudget`] has remaining units.
#[derive(Debug)]
pub struct FlakyCloneAlloc {
    pub(crate) budget: CloneBudget,
}

impl FlakyCloneAlloc {
    /// Builds an allocator sharing one clone budget with the test.
    #[must_use]
    pub fn new(budget: CloneBudget) -> Self {
        Self { budget }
    }
}

// SAFETY: all allocation operations delegate to `Global`; no extra invariants.
unsafe impl Allocator for FlakyCloneAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl TryClone for FlakyCloneAlloc {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.budget.try_consume() {
            Ok(Self {
                budget: self.budget.share(),
            })
        } else {
            Err(TryCloneError::Other("clone budget exhausted"))
        }
    }
}

// SAFETY: allocation delegates to `Global` (a valid allocator) and cloning is
// handled by the `TryClone` impl above; together they satisfy the marker.
unsafe impl AllocatorTryClone for FlakyCloneAlloc {}
