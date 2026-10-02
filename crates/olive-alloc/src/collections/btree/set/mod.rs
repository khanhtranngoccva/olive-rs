//! A fallible port of `std::collections::BTreeSet`.
//!
//! This is a thin wrapper over [`BTreeMap`] that stores each element as a key
//! paired with a zero-sized marker ([`SetValZST`]). All ordering, insertion,
//! removal and iteration semantics are inherited from the underlying map; the
//! set-specific surface hides the marker so callers only ever see values.
//!
//! # Fallibility
//!
//! Every operation that may allocate returns a `Result` whose error carries the
//! offending value plus an [`AllocError`]. On failure the set is left unmodified.

use super::map::BTreeMap;
use super::set_val::SetValZST;
use crate::alloc::{Allocator, Global};
pub use entry::{Entry, OccupiedEntry, VacantEntry};
pub use extract_if::ExtractIf;

mod construction;
mod entry;
mod extract_if;
mod insertion;
mod mutation;
mod query;

/// An ordered set based on a B-tree.
///
/// Values are stored in sorted order according to their [`Ord`] implementation.
/// Internally this is a [`BTreeMap<T, SetValZST>`]; the marker on the right-hand
/// side is never exposed through the public API.
pub struct BTreeSet<T, A: Allocator = Global> {
    map: BTreeMap<T, SetValZST, A>,
}
