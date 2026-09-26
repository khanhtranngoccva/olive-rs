//! Shared test helpers for the `olive-alloc` crate.
//!
//! These utilities replace per-test-file static atomics with per-test
//! thread-safe shared handles (`Arc` over atomics / `RwLock`), eliminating
//! cross-test interference when tests run in parallel threads and allowing the
//! scaffolding to be moved across threads for multi-threaded tests.

pub mod allocators;
pub mod counters;
pub mod ledger;
pub mod rng;
// `TrackedItem` is the generic base behind the `FlakyTrackedItem` alias; it's
// exported so future tests can instantiate it with other clone policies (e.g.
// an infallible one) without touching the tracking machinery. Not referenced by
// name in current tests, hence the allow.
#[allow(unused_imports)]
pub use counters::CloneCounter;
#[allow(unused_imports)]
pub use ledger::{FlakyTrackedItem, Ledger, TrackedItem};
pub use rng::TestRng;

extern crate std;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use core::alloc::Layout;
use core::ptr::NonNull;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

/// A per-test drop counter. Each test constructs its own instance so there is
/// no cross-test interference from parallel execution. Backed by an atomic so a
/// shared [`Arc<DropCounter>`] can be dropped from multiple threads.
#[derive(Debug, Default)]
pub struct DropCounter {
    count: AtomicUsize,
}

impl DropCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the counter (call from a `Drop` impl via the shared `Arc`).
    pub fn record_drop(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current drop count.
    pub fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A per-test panic armer. Starts disarmed; the test arms it before triggering
/// the code path under test and can disarm it afterwards. Backed by an atomic
/// so a shared [`Arc<PanicArmer>`] can be observed across threads.
#[derive(Debug, Default)]
pub struct PanicArmer {
    armed: AtomicBool,
}

impl PanicArmer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }
}

/// A per-test clone budget shared by all instances of [`BudgetedFlaky`].
/// Counts down from a configured threshold; when it reaches zero, clones fail.
///
/// The counter lives in an [`Arc<AtomicU32>`] so every handle produced by
/// [`TryClone`] observes and mutates the *same* remaining budget — that sharing
/// is what lets a single collection draw down one budget across all of its
/// elements, safely across threads.
#[derive(Debug)]
pub struct CloneBudget {
    remaining: Arc<AtomicU32>,
}

impl CloneBudget {
    pub fn new(remaining: u32) -> Self {
        Self {
            remaining: Arc::new(AtomicU32::new(remaining)),
        }
    }

    /// Attempts to consume one unit of budget. Returns `true` if a clone is
    /// allowed, `false` if the budget is exhausted. Atomic so concurrent
    /// consumers can't overdraw the shared budget.
    pub fn try_consume(&self) -> bool {
        // `fetch_update` retries on contention; the closure rejects only when
        // the counter is already zero, so the sole failure value is `Err(0)`.
        self.remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                if cur > 0 { Some(cur - 1) } else { None }
            })
            .is_ok()
    }

    /// Obtains a new handle sharing this budget's counter *without* drawing
    /// from it. Used to seed collection elements at construction time, where
    /// creating the payload must not count against the clone budget — only
    /// actual [`TryClone`]s do.
    #[inline]
    pub fn share(&self) -> Self {
        Self {
            remaining: self.remaining.clone(),
        }
    }
}

// A `CloneBudget` doubles as a clone *policy*: cloning it draws one unit from
// the shared counter and hands back a handle that observes the same remaining
// budget. When the budget is exhausted the clone fails — this is exactly the
// deterministic mid-operation failure the flaky tests drive. Sharing the
// counter via `Arc` keeps every instance in a collection on one budget.
impl TryClone for CloneBudget {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if !self.try_consume() {
            return Err(TryCloneError::Other("budget exhausted"));
        }
        Ok(Self {
            remaining: self.remaining.clone(),
        })
    }
}

/// A pass-through allocator whose own `Drop` is recorded by a shared counter, so
/// a test can verify the allocator instance was destroyed exactly once (neither
/// leaked nor double-dropped), e.g. after being moved into a container and back
/// out. Delegates all memory operations to `Global`. Per-test isolation via the
/// `Arc` shared with the test body.
#[derive(Debug, Clone)]
pub struct DropCountingAlloc {
    drops: Arc<DropCounter>,
}

impl DropCountingAlloc {
    /// Builds a counting allocator sharing one drop counter with the test.
    pub fn new(drops: Arc<DropCounter>) -> Self {
        Self { drops }
    }
}

// SAFETY: delegates all operations to `Global`; no additional invariants.
unsafe impl Allocator for DropCountingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        crate::alloc::Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
    }
}

impl Drop for DropCountingAlloc {
    fn drop(&mut self) {
        self.drops.record_drop();
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
        Ok(Self::new(Arc::new(DropCounter::new())))
    }
}

/// An [`Allocator`] whose allocation forwards to `Global` but whose
/// [`TryClone`] succeeds only while a shared [`CloneBudget`] has remaining units.
#[derive(Debug, Clone)]
pub struct FlakyCloneAlloc {
    pub(crate) budget: Arc<CloneBudget>,
}

impl FlakyCloneAlloc {
    /// Builds an allocator sharing one clone budget with the test.
    pub fn new(budget: Arc<CloneBudget>) -> Self {
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

/// An allocator that succeeds for its first `budget` allocations and fails
/// forever after. Lets a test build a populated collection under a working
/// allocator and then deterministically trigger an OOM on the next growth —
/// without any buffer surgery or transmute.
///
/// The budget is shared via [`Arc<AtomicUsize>`] so the same limit can be
/// observed from multiple handles, safely across threads.
#[derive(Debug, Clone)]
pub struct BudgetedAlloc {
    pub(crate) remaining: Arc<AtomicUsize>,
}

impl BudgetedAlloc {
    /// Builds an allocator allowing exactly `budget` successful allocations.
    pub fn new(budget: usize) -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(budget)),
        }
    }

    /// Sets the remaining allocation budget to exactly `budget`, letting a
    /// test tighten or loosen the limit on the fly — e.g. allow just enough
    /// allocations for the next operation and no more. The change is visible
    /// to every handle sharing the same budget (clones share the counter via
    /// [`Arc`]).
    pub fn set_budget(&self, budget: usize) {
        self.remaining.store(budget, Ordering::SeqCst);
    }

    /// Removes all remaining allocation budget, so every subsequent
    /// [`Allocator::allocate`] call fails. Tests use this to force a
    /// deterministic OOM on the next growth without leaking any memory (the
    /// old pattern of looping `allocate` until failure leaked one block per
    /// iteration).
    pub fn drain(&self) {
        self.set_budget(0);
    }
}

// SAFETY: while budget remains, delegates all operations to `Global`; once
// exhausted, no new blocks are handed out and only previously allocated ones
// (all owned by `Global`) are freed.
unsafe impl Allocator for BudgetedAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Atomically draw one unit; `fetch_update` rejects when already zero so
        // concurrent allocators can't overdraw past the budget.
        let had_budget = self
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                if cur > 0 { Some(cur - 1) } else { None }
            })
            .is_ok();
        if !had_budget {
            return Err(AllocError);
        }
        crate::alloc::Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { crate::alloc::Global.deallocate(ptr, layout) };
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
