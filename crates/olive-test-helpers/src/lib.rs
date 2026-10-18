//! Shared test fixtures for the Olive stack.

pub mod allocators;
pub mod counters;
pub mod fallible_ops;
pub mod ledger;
pub mod panicky_ops;
pub mod rng;

pub use allocators::{
    BudgetedAlloc, ByteCapAlloc, CountingAllocator, DropCountingAlloc, FailAlloc, FailDefaultAlloc,
    FlakyCloneAlloc,
};
pub use counters::Counter;
pub use fallible_ops::{CloneBudget, FailArmer, FlakyClone, MaybeUnclonable, NoDefault};
#[allow(unused_imports)]
pub use ledger::{FlakyCloneTrackedItem, Ledger, TrackedItem};
pub use panicky_ops::{PanicArmer, PanickyDrop, PanickyOrd};
pub use rng::TestRng;
