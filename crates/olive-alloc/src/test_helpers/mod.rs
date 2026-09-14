//! Shared test helpers for the `olive-alloc` crate.
//!
//! These utilities replace per-test-file static atomics with per-test
//! `Rc<RefCell<_>>` instances, eliminating cross-test interference when tests
//! run in parallel threads.

pub mod allocators;
mod ledger;
pub use ledger::{Ledger, FlakyTrackedItem};

extern crate std;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use core::alloc::Layout;
use core::ptr::NonNull;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// A per-test drop counter. Each test constructs its own instance so there is
/// no cross-test interference from parallel execution.
#[derive(Debug, Default)]
pub struct DropCounter {
    count: RefCell<usize>,
}

impl DropCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the counter (call from a `Drop` impl via the shared `Rc`).
    pub fn record_drop(&self) {
        *self.count.borrow_mut() += 1;
    }

    /// Returns the current drop count.
    pub fn get(&self) -> usize {
        *self.count.borrow()
    }
}

/// A per-test panic armer. Starts disarmed; the test arms it before triggering
/// the code path under test and can disarm it afterwards.
#[derive(Debug, Default)]
pub struct PanicArmer {
    armed: RefCell<bool>,
}

impl PanicArmer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arm(&self) {
        *self.armed.borrow_mut() = true;
    }

    pub fn disarm(&self) {
        *self.armed.borrow_mut() = false;
    }

    pub fn is_armed(&self) -> bool {
        *self.armed.borrow()
    }
}

/// A per-test clone budget shared by all instances of [`BudgetedFlaky`].
/// Counts down from a configured threshold; when it reaches zero, clones fail.
#[derive(Debug, PartialEq, Eq)]
pub struct CloneBudget {
    remaining: RefCell<u32>,
}

impl CloneBudget {
    pub fn new(remaining: u32) -> Self {
        Self {
            remaining: RefCell::new(remaining),
        }
    }

    /// Attempts to consume one unit of budget. Returns `true` if a clone is
    /// allowed, `false` if the budget is exhausted.
    pub fn try_consume(&self) -> bool {
        let mut r = self.remaining.borrow_mut();
        if *r == 0 {
            false
        } else {
            *r -= 1;
            true
        }
    }
}

/// A value whose `try_clone` succeeds while the shared budget has remaining
/// units, then fails forever. Per-test isolation via `Rc<CloneBudget>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedFlaky {
    pub(crate) budget: Rc<CloneBudget>,
}

impl TryClone for BudgetedFlaky {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.budget.try_consume() {
            Ok(BudgetedFlaky {
                budget: self.budget.clone(),
            })
        } else {
            Err(TryCloneError::Other("budget exhausted"))
        }
    }
}

/// A pass-through allocator whose own `Drop` is recorded by a shared counter, so
/// a test can verify the allocator instance was destroyed exactly once (neither
/// leaked nor double-dropped), e.g. after being moved into a container and back
/// out. Delegates all memory operations to `Global`. Per-test isolation via the
/// `Rc` shared with the test body.
#[derive(Debug, Clone)]
pub struct LocalCountingAlloc {
    drops: Rc<DropCounter>,
}

impl LocalCountingAlloc {
    /// Builds a counting allocator sharing one drop counter with the test.
    pub fn new(drops: Rc<DropCounter>) -> Self {
        Self { drops }
    }
}

// SAFETY: delegates all operations to `Global`; no additional invariants.
unsafe impl Allocator for LocalCountingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        crate::alloc::Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl Drop for LocalCountingAlloc {
    fn drop(&mut self) {
        self.drops.record_drop();
    }
}

impl TryClone for LocalCountingAlloc {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning an `Rc` never fails in this framework's test harness.
        Ok(Self {
            drops: self.drops.clone(),
        })
    }
}

// SAFETY: cloning is infallible (the shared `Rc` clone cannot fail) and all
// memory operations delegate to `Global`, so a cloned handle is equivalent to
// the original.
unsafe impl AllocatorTryClone for LocalCountingAlloc {}

/// An [`Allocator`] whose allocation forwards to `Global` but whose
/// [`TryClone`] succeeds only while a shared [`CloneBudget`] has remaining units.
#[derive(Debug, Clone)]
pub struct FlakyCloneAlloc {
    pub(crate) budget: Rc<CloneBudget>,
}

impl FlakyCloneAlloc {
    /// Builds an allocator sharing one clone budget with the test.
    pub fn new(budget: Rc<CloneBudget>) -> Self {
        Self { budget }
    }
}

// SAFETY: all allocation operations delegate to `Global`; no extra invariants.
unsafe impl Allocator for FlakyCloneAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        crate::alloc::Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl TryClone for FlakyCloneAlloc {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.budget.try_consume() {
            Ok(Self {
                budget: self.budget.clone(),
            })
        } else {
            Err(TryCloneError::Other("clone budget exhausted"))
        }
    }
}

// SAFETY: allocation delegates to `Global` (a valid allocator) and cloning is
// handled by the `TryClone` impl above; together they satisfy the marker.
unsafe impl AllocatorTryClone for FlakyCloneAlloc {}

/// An allocator whose every allocation fails. Used to exercise OOM paths.
#[derive(Debug, Default)]
pub struct FailAlloc;

// SAFETY: never hands out memory, so there is nothing to free on deallocate.
unsafe impl Allocator for FailAlloc {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

impl TryDefault for FailAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self)
    }
}

/// An allocator that succeeds for its first `budget` allocations and fails
/// forever after. Lets a test build a populated collection under a working
/// allocator and then deterministically trigger an OOM on the next growth —
/// without any buffer surgery or transmute.
///
/// The budget is shared via [`Rc`] so the same limit can be observed from
/// multiple handles if needed.
#[derive(Debug, Clone)]
pub struct BudgetedAlloc {
    pub(crate) remaining: Rc<Cell<usize>>,
}

impl BudgetedAlloc {
    /// Builds an allocator allowing exactly `budget` successful allocations.
    pub fn new(budget: usize) -> Self {
        Self {
            remaining: Rc::new(Cell::new(budget)),
        }
    }
}

// SAFETY: while budget remains, delegates all operations to `Global`; once
// exhausted, no new blocks are handed out and only previously allocated ones
// (all owned by `Global`) are freed.
unsafe impl Allocator for BudgetedAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let had_budget = self.remaining.get() > 0;
        if had_budget {
            self.remaining.set(self.remaining.get() - 1);
        }
        if !had_budget {
            return Err(AllocError);
        }
        crate::alloc::Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl TryDefault for BudgetedAlloc {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::new(0))
    }
}

/// A value whose `try_clone` succeeds while its internal counter is below
/// `threshold`, incrementing it each successful clone, then fails forever once
/// the counter reaches `threshold`. Parametrized as `(start, threshold)` so a
/// test can place exactly where in a sequence of clones the failure lands: an
/// element seeded with `start` will fail to clone after `threshold - start`
/// further successful clones. Lets us simulate a mid-operation clone failure at
/// a deterministic point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlakyClone {
    pub count: u32,
    pub threshold: u32,
}

impl FlakyClone {
    /// Shorthand constructor for the common case of starting at 0.
    pub const fn new(threshold: u32) -> Self {
        Self {
            count: 0,
            threshold,
        }
    }
}

impl TryClone for FlakyClone {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if self.count >= self.threshold {
            Err(TryCloneError::Other("flaky"))
        } else {
            Ok(FlakyClone {
                count: self.count + 1,
                threshold: self.threshold,
            })
        }
    }
}
