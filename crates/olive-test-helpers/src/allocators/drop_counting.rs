//! A pass-through allocator whose own `Drop` is observable through a
//! shared counter.

extern crate std;

use core::alloc::Layout;
use core::ptr::NonNull;

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
use std::sync::Arc;

use crate::counters::Counter;

/// A pass-through allocator whose own `Drop` is recorded by a shared counter, so
/// a test can verify the allocator instance was destroyed exactly once (neither
/// leaked nor double-dropped), e.g. after being moved into a container and back
/// out. Delegates all memory operations to `Global`. Per-test isolation via the
/// `Arc` shared with the test body.
#[derive(Debug, Clone)]
pub struct DropCountingAlloc {
    drops: Arc<Counter>,
}

impl DropCountingAlloc {
    /// Builds a counting allocator sharing one drop counter with the test.
    pub fn new(drops: Arc<Counter>) -> Self {
        Self { drops }
    }
}

// SAFETY: delegates all operations to `Global`; no additional invariants.
unsafe impl Allocator for DropCountingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl Drop for DropCountingAlloc {
    fn drop(&mut self) {
        self.drops.record();
    }
}

impl TryClone for DropCountingAlloc {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning an `Arc` never fails in this framework's test harness.
        Ok(Self {
            drops: self.drops.clone(),
        })
    }
}

// SAFETY: cloning is infallible (the shared `Arc` clone cannot fail here) and all
// memory operations delegate to `Global`, so a cloned handle is equivalent to
// the original.
unsafe impl AllocatorTryClone for DropCountingAlloc {}

impl TryDefault for DropCountingAlloc {
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new(Arc::new(Counter::new())))
    }
}
