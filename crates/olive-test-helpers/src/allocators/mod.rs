//! Shared test-only allocators.
//!
//! These are available to every test target in the workspace except the core
//! crate, so that tests which need to observe allocation traffic — e.g.
//! verifying a block is freed even when a pointee's destructor panics —
//! can share one well-known implementation.
//!
//! Allocation hooks ignore zero-sized layouts: those never touch the heap,
//! so they neither count (CountingAllocator), draw from a budget
//! (BudgetedAlloc), trip the cap (ByteCapAlloc, by construction) nor fail
//! (FailAlloc); they delegate straight to `Global`.

pub mod budgeted;
pub mod byte_cap;
pub mod counting;
pub mod drop_counting;
pub mod fail;
pub mod fail_default;
pub mod flaky_clone;

pub use budgeted::BudgetedAlloc;
pub use byte_cap::ByteCapAlloc;
pub use counting::CountingAllocator;
pub use drop_counting::DropCountingAlloc;
pub use fail::FailAlloc;
pub use fail_default::FailDefaultAlloc;
pub use flaky_clone::FlakyCloneAlloc;
