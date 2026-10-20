//! A pass-through allocator with a settable allocation budget.

extern crate std;

use allocator_api2::alloc::Global;
use core::alloc::Layout;
use core::ptr::NonNull;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Shared, mutex-guarded budget state.
///
/// The lock keeps `set_budget` and the refunds of in-flight failed
/// allocations mutually exclusive: a bare atomic refund would land in
/// whatever counter value happens to be current, crediting a budget
/// installed after the attempt began (e.g. undoing a concurrent
/// [`BudgetedAlloc::drain`]). The generation counter lets a refund tell
/// whether the budget it would credit is the same one its unit was drawn
/// from; it wraps after `usize::MAX` replacements, at which point a stale
/// refund could be misattributed, which no realistic test reaches.
#[derive(Debug)]
struct BudgetState {
    remaining: usize,
    generation: usize,
}

/// An allocator that succeeds for its first `budget` allocations and fails
/// forever after. Lets a test build a populated collection under a working
/// allocator and then deterministically trigger an OOM on the next growth.
#[derive(Debug, Clone)]
pub struct BudgetedAlloc {
    state: Arc<Mutex<BudgetState>>,
}

impl BudgetedAlloc {
    /// Builds an allocator allowing exactly `budget` successful allocations.
    /// Failed attempts (e.g. a rejected huge layout) do not consume budget.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(BudgetState {
                remaining: budget,
                generation: 0,
            })),
        }
    }

    /// Sets the remaining allocation budget to exactly `budget`, letting a
    /// test tighten or loosen the limit on the fly — e.g. allow just enough
    /// allocations for the next operation and no more.
    ///
    /// A budget installed while an allocation is in flight is never inflated
    /// by that attempt's refund: the refund is dropped instead of credited.
    pub fn set_budget(&self, budget: usize) {
        let mut state = self.lock();
        state.remaining = budget;
        state.generation = state.generation.wrapping_add(1);
    }

    /// Removes all remaining allocation budget, so every subsequent
    /// [`Allocator::allocate`] call fails. Tests use this to force a
    /// deterministic OOM on the next growth. In-flight attempts do not
    /// resurrect budget: their refunds are dropped.
    pub fn drain(&self) {
        self.set_budget(0);
    }

    /// The number of allocations still permitted. Reads the shared state, so
    /// every handle observing this budget sees the same value.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.lock().remaining
    }

    fn lock(&self) -> MutexGuard<'_, BudgetState> {
        // A panicking test may have poisoned the lock; the guarded state is
        // two plain integers that are always internally consistent, so
        // poisoning carries no information and is ignored.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
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
        // Draw one unit under the lock, so concurrent allocators cannot
        // overdraw past the budget. The lock is released before the actual
        // heap call, so concurrent allocations are not serialized.
        let generation;
        {
            let mut state = self.lock();
            generation = state.generation;
            if state.remaining == 0 {
                return Err(AllocError);
            }
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the zero check above guarantees remaining > 0"
            )]
            {
                state.remaining -= 1;
            }
        }
        // A failed `Global.allocate` must not consume budget: give the drawn
        // unit back, but only if the budget was not replaced since the draw.
        // A refund landing on a newer budget would let a test see more
        // successful allocations than it asked for, so stale refunds are
        // dropped instead.
        match Global.allocate(layout) {
            Ok(block) => Ok(block),
            Err(e) => {
                let mut state = self.lock();
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "generation ensures that this does not affect new budget"
                )]
                if state.generation == generation {
                    state.remaining += 1;
                }
                Err(e)
            }
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) };
    }
}

impl TryClone for BudgetedAlloc {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning the budget handle is infallible (Arc clone); the budget
        // state is shared, so the clone observes the same budget.
        Ok(Self {
            state: self.state.clone(),
        })
    }
}

// SAFETY: delegates all memory ops to `Global`; cloning shares the same
// budget state via Arc, so a cloned handle is equivalent.
unsafe impl AllocatorTryClone for BudgetedAlloc {}

impl TryDefault for BudgetedAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        // Matches the sibling test allocators: a usable default, not an
        // OOM-by-default one. Use `BudgetedAlloc::new(0)` / `drain()` to
        // deliberately start with an empty budget.
        Ok(Self::new(usize::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::alloc::Layout;

    const LAYOUT: Layout = match Layout::array::<u8>(8) {
        Ok(l) => l,
        Err(_) => panic!("unreachable"),
    };

    #[test]
    fn budget_is_consumed_by_successful_allocations_only() {
        let alloc = BudgetedAlloc::new(1);

        // A failed attempt does not consume the unit. `isize::MAX` bytes is a
        // valid layout that `Global` can never satisfy.
        let huge = Layout::from_size_align(isize::MAX as usize, 1).unwrap();
        assert!(alloc.allocate(huge).is_err());
        assert_eq!(alloc.remaining(), 1, "failed attempt must not consume");

        // The legitimate allocation consumes it.
        assert!(alloc.allocate(LAYOUT).is_ok());
        assert_eq!(alloc.remaining(), 0);

        // Past the budget every non-ZST allocation fails.
        assert!(alloc.allocate(LAYOUT).is_err());
        assert!(alloc.allocate(LAYOUT).is_err());
        assert_eq!(alloc.remaining(), 0);
    }

    #[test]
    fn set_budget_resets_and_handles_share_state() {
        let a = BudgetedAlloc::new(2);
        let b = a.try_clone().unwrap();

        assert!(a.allocate(LAYOUT).is_ok());
        b.set_budget(7);
        assert_eq!(a.remaining(), 7);
        assert_eq!(b.remaining(), 7);

        a.drain();
        assert_eq!(b.remaining(), 0);
        assert!(b.allocate(LAYOUT).is_err());
    }

    #[test]
    fn zero_sized_allocations_bypass_the_budget() {
        let alloc = BudgetedAlloc::new(0);
        let zst = Layout::new::<()>();
        assert!(alloc.allocate(zst).is_ok());
        assert_eq!(alloc.remaining(), 0);
    }
}
