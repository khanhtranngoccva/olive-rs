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

use crate::alloc::{AllocError, Allocator, AllocatorTryDefault, Global};
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use olive_core::try_traits::try_default::{TryDefault, TryDefaultError};

use super::iter::{IntoKeys, Keys};
use super::map::BTreeMap;
use super::set_val::SetValZST;
pub use entry::{Entry, OccupiedEntry, VacantEntry};

mod entry;

/// An ordered set based on a B-tree.
///
/// Values are stored in sorted order according to their [`Ord`] implementation.
/// Internally this is a [`BTreeMap<T, SetValZST>`]; the marker on the right-hand
/// side is never exposed through the public API.
pub struct BTreeSet<T, A: Allocator = Global> {
    map: BTreeMap<T, SetValZST, A>,
}
