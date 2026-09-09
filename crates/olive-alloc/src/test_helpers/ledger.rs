//! Per-test ledger for tracking individual payload ids by count.

extern crate std;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::vec::Vec;

use super::CloneBudget;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

/// A per-test ledger tracking individual payload ids by count. On construction
/// an id is registered as "live" (`live[id] == true`); on drop its count is
/// incremented and it is marked no longer live. Because drops are counted
/// rather than merely logged, the ledger detects all three failure classes a
/// container bug could introduce after a caught panic or mid-operation abort:
/// 1. Leaks — an id still marked live when it should have been dropped.
/// 2. Double-frees — an id whose drop count exceeds one.
/// 3. Wrong totals — the sum of drop counts differs from expectation.
///
/// Ids are allocated monotonically via [`Ledger::allocate`] so the set of ids
/// ever created is exactly `0..total_allocated()`, which lets
/// [`Ledger::all_dropped_once`] check every id without callers enumerating them.
#[derive(Debug)]
pub struct Ledger {
    live: RefCell<HashSet<u32>>,
    drop_counts: RefCell<HashMap<u32, usize>>,
    next_id: RefCell<u32>,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            live: RefCell::new(HashSet::new()),
            drop_counts: RefCell::new(HashMap::new()),
            next_id: RefCell::new(0u32),
        }
    }

    /// Allocates and returns a fresh unique id.
    pub fn allocate(&self) -> u32 {
        let mut next = self.next_id.borrow_mut();
        let id = *next;
        *next += 1;
        id
    }

    /// Total number of ids handed out by [`Self::allocate`] so far. The set of
    /// all created ids is therefore exactly `0..total_allocated()`.
    pub fn total_allocated(&self) -> u32 {
        *self.next_id.borrow()
    }

    /// Registers an id as currently alive.
    pub fn register(&self, id: u32) {
        self.live.borrow_mut().insert(id);
    }

    /// Records that `id` was dropped: bumps its drop count and clears its live
    /// flag. Called from `Drop` impls. Calling this twice for the same id is
    /// what makes double-frees observable (the count goes to 2+).
    pub fn unregister(&self, id: u32) {
        *self.drop_counts.borrow_mut().entry(id).or_insert(0) += 1;
        self.live.borrow_mut().remove(&id);
    }

    /// Number of times `id` has been dropped (0 if never).
    pub fn drop_count(&self, id: u32) -> usize {
        self.drop_counts.borrow().get(&id).copied().unwrap_or(0)
    }

    /// Map of every id that has been dropped at least once, to its count.
    pub fn drop_counts(&self) -> HashMap<u32, usize> {
        self.drop_counts.borrow().clone()
    }

    /// Snapshot of currently-live ids (non-empty ⇒ leak).
    pub fn live_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.live.borrow().iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Ids that were allocated but never dropped (a subset of the live set,
    /// excluding any that also dropped — impossible here since dropping clears
    /// live, but kept explicit for clarity). Empty ⇒ no leaks.
    pub fn leaked_ids(&self) -> Vec<u32> {
        self.live_ids()
    }

    /// Ids dropped more than once. Empty ⇒ no double-free.
    pub fn double_dropped(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .drop_counts
            .borrow()
            .iter()
            .filter(|(_, c)| **c > 1)
            .map(|(id, _)| *id)
            .collect();
        v.sort_unstable();
        v
    }

    /// True iff every id in `expected` was dropped exactly once and nothing
    /// else was dropped. Handy for the common "these N elements must each die
    /// exactly once" assertion.
    pub fn all_dropped_once(&self, expected: impl IntoIterator<Item = u32>) -> bool {
        let counts = self.drop_counts.borrow();
        let n_expected: usize = expected.into_iter().count();
        if counts.len() != n_expected {
            return false;
        }
        for (_, c) in counts.iter() {
            if *c != 1 {
                return false;
            }
        }
        true
    }
}

/// A tracked item registering a fresh id in a shared [`Ledger`] on
/// construction and unregistering it on drop, so a test can observe every
/// transient instance individually — catching leaks, double-frees, and wrong
/// totals rather than merely aggregate drop counts. Its `try_clone` succeeds
/// while a shared [`CloneBudget`] has room (each clone gets its own new id),
/// then fails — letting a test drive a deterministic mid-operation clone
/// failure while still observing every instance.
pub struct FlakyTrackedItem {
    pub id: u32,
    pub ledger: Rc<Ledger>,
    pub budget: Rc<CloneBudget>,
}

impl Drop for FlakyTrackedItem {
    fn drop(&mut self) {
        self.ledger.unregister(self.id);
    }
}

impl TryClone for FlakyTrackedItem {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        if !self.budget.try_consume() {
            return Err(TryCloneError::Other("budget exhausted"));
        }
        let id = self.ledger.allocate();
        self.ledger.register(id);
        Ok(FlakyTrackedItem {
            id,
            ledger: self.ledger.clone(),
            budget: self.budget.clone(),
        })
    }
}
