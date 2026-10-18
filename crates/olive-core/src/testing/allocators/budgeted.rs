//! A pass-through allocator with a settable allocation budget.

extern crate std;

use core::alloc::Layout;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use allocator_api2::alloc::Global;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
use std::sync::Arc;

/// An allocator that succeeds for its first `budget` allocations and fails
/// forever after. Lets a test build a populated collection under a working
/// allocator and then deterministically trigger an OOM on the next growth.
#[derive(Debug, Clone)]
pub struct BudgetedAlloc {
    pub(crate) remaining: Arc<AtomicUsize>,
}

impl BudgetedAlloc {
    /// Builds an allocator allowing exactly `budget` successful allocations.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(budget)),
        }
    }

    /// Sets the remaining allocation budget to exactly `budget`, letting a
    /// test tighten or loosen the limit on the fly — e.g. allow just enough
    /// allocations for the next operation and no more
    pub fn set_budget(&self, budget: usize) {
        self.remaining.store(budget, Ordering::SeqCst);
    }

    /// Removes all remaining allocation budget, so every subsequent
    /// [`Allocator::allocate`] call fails. Tests use this to force a
    /// deterministic OOM on the next growth.
    pub fn drain(&self) {
        self.set_budget(0);
    }

    /// The number of allocations still permitted. Reads the shared counter, so
    /// every handle observing this budget sees the same value.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::SeqCst)
    }
}

// SAFETY: while budget remains, delegates all operations to `Global`; once
// exhausted, no new blocks are handed out and only previously allocated ones
// (all owned by `Global`) are freed.
unsafe impl Allocator for BudgetedAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Zero-sized allocations never touch the heap, so they don't draw
        // from the budget, matching the other test allocators.
        if layout.size() == 0 {
            return Global.allocate(layout);
        }
        // Atomically draw one unit; `fetch_update` rejects when already zero so
        // concurrent allocators can't overdraw past the budget.
        let had_budget = self
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                #[allow(clippy::arithmetic_side_effects, reason = "asserted cur > 0")]
                if cur > 0 { Some(cur - 1) } else { None }
            })
            .is_ok();
        if !had_budget {
            return Err(AllocError);
        }
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl TryClone for BudgetedAlloc {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning the budget handle is infallible (Arc clone); the remaining
        // counter is shared, so the clone observes the same budget.
        Ok(Self {
            remaining: self.remaining.clone(),
        })
    }
}

// SAFETY: delegates all memory ops to `Global`; cloning shares the same
// budget counter via Arc, so a cloned handle is equivalent.
unsafe impl AllocatorTryClone for BudgetedAlloc {}

impl TryDefault for BudgetedAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new(0))
    }
}
