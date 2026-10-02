//! Per-test ledger for tracking individual payload ids by count.

extern crate std;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::vec::Vec;

use super::CloneBudget;
use olive_core::try_traits::try_clone::{TryClone, TryCloneError};

/// A tracked item carrying a unique [`Ledger`] id plus a pluggable inner payload
/// `C`.
///
/// Drop registration/unregistration is fixed: on construction the caller has
/// already registered `id` as live (via [`Ledger::register`]), and dropping the
/// item unregisters it — so a test observes every transient instance
/// individually, catching leaks, double-frees, and wrong totals rather than mere
/// aggregate counts.
pub struct TrackedItem<C> {
    pub id: u32,
    pub ledger: Arc<Ledger>,
    pub inner: C,
}

impl<C> Drop for TrackedItem<C> {
    fn drop(&mut self) {
        self.ledger.unregister(self.id);
    }
}

impl<C> TrackedItem<C> {
    /// Constructs a tracked item with the next available id from `ledger`,
    /// registering it as live in one step. Convenience wrapper around
    /// [`Ledger::allocate`] + [`Ledger::register`].
    pub fn construct(ledger: &Arc<Ledger>, inner: C) -> Self {
        let id = ledger.allocate();
        ledger.register(id);
        Self {
            id,
            ledger: ledger.clone(),
            inner,
        }
    }
}

impl<C: TryClone> TryClone for TrackedItem<C> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Gate on the inner first so a failed clone never mints a stray id.
        let inner = self.inner.try_clone()?;
        let id = self.ledger.allocate();
        self.ledger.register(id);
        Ok(TrackedItem {
            id,
            ledger: self.ledger.clone(),
            inner,
        })
    }
}

impl<C: core::fmt::Debug> core::fmt::Debug for TrackedItem<C> {
    /// Prints only the fields that carry meaning in a test failure — the ledger
    /// id and the inner payload.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TrackedItem")
            .field("id", &self.id)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<C> core::ops::Deref for TrackedItem<C> {
    type Target = C;

    #[inline]
    fn deref(&self) -> &C {
        &self.inner
    }
}

impl<C> core::ops::DerefMut for TrackedItem<C> {
    #[inline]
    fn deref_mut(&mut self) -> &mut C {
        &mut self.inner
    }
}

impl<C> core::borrow::Borrow<C> for TrackedItem<C> {
    #[inline]
    fn borrow(&self) -> &C {
        &self.inner
    }
}

impl<C: PartialEq> PartialEq for TrackedItem<C> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<C: Eq> Eq for TrackedItem<C> {}

impl<C: PartialOrd> PartialOrd for TrackedItem<C> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl<C: Ord> Ord for TrackedItem<C> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

/// The flaky variant used throughout existing tests: a tracked item whose clone
/// policy is a shared [`CloneBudget`] (internally `Rc`-backed, so every handle
/// observes the same remaining budget). A test can therefore place a
/// deterministic clone failure at a known point across a whole collection.
/// Retained as an alias so current call sites keep their familiar shape.
pub type FlakyTrackedItem = TrackedItem<CloneBudget>;

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
/// Thread-safe per-test ledger. Interior mutability uses `AtomicU32` for the
/// monotonic id counter (lock-free read-modify-write via `fetch_add`) and
/// `RwLock` for the two maps, so a shared [`Arc<Ledger>`] can be moved across
/// threads. Lock scopes are kept minimal and never held across calls into user
/// code.
#[derive(Debug)]
pub struct Ledger {
    live: RwLock<HashSet<u32>>,
    drop_counts: RwLock<HashMap<u32, usize>>,
    next_id: AtomicU32,
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            live: RwLock::new(HashSet::new()),
            drop_counts: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Allocates and returns a fresh unique id.
    pub fn allocate(&self) -> u32 {
        // Relaxed is sufficient: `fetch_add` is internally atomic and we do not
        // order other memory operations around it here.
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Total number of ids handed out by [`Self::allocate`] so far. The set of
    /// all created ids is therefore exactly `0..total_allocated()`.
    pub fn total_allocated(&self) -> u32 {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Registers an id as currently alive.
    pub fn register(&self, id: u32) {
        self.live.write().unwrap().insert(id);
    }

    /// Records that `id` was dropped: bumps its drop count and clears its live
    /// flag. Called from `Drop` impls. Calling this twice for the same id is
    /// what makes double-frees observable (the count goes to 2+).
    pub fn unregister(&self, id: u32) {
        *self.drop_counts.write().unwrap().entry(id).or_insert(0) += 1;
        self.live.write().unwrap().remove(&id);
    }

    /// Number of times `id` has been dropped (0 if never).
    pub fn drop_count(&self, id: u32) -> usize {
        self.drop_counts
            .read()
            .unwrap()
            .get(&id)
            .copied()
            .unwrap_or(0)
    }

    /// Map of every id that has been dropped at least once, to its count.
    pub fn drop_counts(&self) -> HashMap<u32, usize> {
        self.drop_counts.read().unwrap().clone()
    }

    /// Snapshot of currently-live ids (non-empty ⇒ leak).
    pub fn live_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.live.read().unwrap().iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Ids that were allocated but never dropped (a subset of the live set,
    /// excluding any that also dropped). Empty ⇒ no leaks.
    pub fn leaked_ids(&self) -> Vec<u32> {
        self.live_ids()
    }

    /// Ids dropped more than once. Empty ⇒ no double-free.
    pub fn double_dropped(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .drop_counts
            .read()
            .unwrap()
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
        let counts = self.drop_counts.read().unwrap();
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
