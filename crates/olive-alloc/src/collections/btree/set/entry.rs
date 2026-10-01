//! The set-specific entry API for [`BTreeSet`](super::BTreeSet).
//!
//! Mirrors `std`'s Entry API: an [`Entry`] is a view into a
//! single slot of the set, either vacant or occupied.
//!
//! # Error shape
//!
//! Each mutating method comes in two flavours:
//!
//! - The plain form (`try_insert`, `or_try_insert`) reports a bare
//!   [`AllocError`]; the probed value is dropped on failure.
//! - The `_give_back` form (`try_insert_give_back`, `or_try_insert_give_back`)
//!   reports `(T, AllocError)`, handing the value back so the caller can
//!   recover it after an allocation failure.

use super::super::entry::{
    Entry as MapEntry, OccupiedEntry as MapOccupied, VacantEntry as MapVacant,
};
use super::super::set_val::SetValZST;
use crate::alloc::{AllocError, Allocator, Global};
use core::fmt;

/// A view into a single entry in a [`BTreeSet`](super::BTreeSet), which may
/// either be vacant or occupied.
pub enum Entry<'a, T: 'a, A: Allocator = Global> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, T, A>),
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, T, A>),
}

impl<'a, T: Ord, A: Allocator> Entry<'a, T, A> {
    /// Wraps a map-level entry over `BTreeMap<T, SetValZST>` as a set entry.
    pub(super) fn from_map_entry(inner: MapEntry<'a, T, SetValZST, A>) -> Self {
        match inner {
            MapEntry::Vacant(v) => Self::Vacant(VacantEntry { inner: v }),
            MapEntry::Occupied(o) => Self::Occupied(OccupiedEntry { inner: o }),
        }
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &T {
        match self {
            Entry::Vacant(ve) => ve.get(),
            Entry::Occupied(oe) => oe.get(),
        }
    }

    /// Inserts the value into the set if absent, and returns an [`OccupiedEntry`].
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during insertion of a
    /// previously-absent value. The probed value is dropped on failure and the
    /// set is left unmodified. For a variant that recovers the value, use
    /// [`Self::try_insert_give_back`].
    pub fn try_insert(self) -> Result<OccupiedEntry<'a, T, A>, AllocError> {
        match self {
            Entry::Occupied(oe) => Ok(oe),
            Entry::Vacant(ve) => ve
                .insert_marker()
                .map(|inner| OccupiedEntry { inner })
                .map_err(|(_, e)| e),
        }
    }

    /// If the value is not present, inserts it into the set. Does nothing if it
    /// is already present.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during insertion of a
    /// previously-absent value. The probed value is dropped on failure and the
    /// set is left unmodified. For a variant that recovers the value, use
    /// [`Self::or_try_insert_give_back`].
    pub fn or_try_insert(self) -> Result<(), AllocError> {
        match self {
            Entry::Occupied(_) => Ok(()),
            Entry::Vacant(ve) => {
                ve.insert_marker().map_err(|(_, e)| e)?;
                Ok(())
            }
        }
    }

    /// Inserts the value into the set if absent, and returns an [`OccupiedEntry`].
    /// On allocation failure the value is handed back alongside the error so the
    /// caller can recover it.
    ///
    /// # Errors
    ///
    /// Returns `(value, AllocError)` if memory allocation fails during insertion
    /// of a previously-absent value. The set is left unmodified on failure.
    pub fn try_insert_give_back(self) -> Result<OccupiedEntry<'a, T, A>, (T, AllocError)> {
        match self {
            Entry::Occupied(oe) => Ok(oe),
            Entry::Vacant(ve) => ve.insert_marker().map(|inner| OccupiedEntry { inner }),
        }
    }

    /// If the value is not present, inserts it into the set. Does nothing if it
    /// is already present. On allocation failure the value is handed back
    /// alongside the error so the caller can recover it.
    ///
    /// # Errors
    ///
    /// Returns `(value, AllocError)` if memory allocation fails during insertion
    /// of a previously-absent value. The set is left unmodified on failure.
    pub fn or_try_insert_give_back(self) -> Result<(), (T, AllocError)> {
        match self {
            Entry::Occupied(_) => Ok(()),
            Entry::Vacant(ve) => {
                ve.insert_marker()?;
                Ok(())
            }
        }
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for Entry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Vacant(ve) => ve.fmt(f),
            Entry::Occupied(oe) => oe.fmt(f),
        }
    }
}

/// A vacant entry in a [`BTreeSet`](super::BTreeSet).
pub struct VacantEntry<'a, T: 'a, A: Allocator = Global> {
    inner: MapVacant<'a, T, SetValZST, A>,
}

impl<'a, T: Ord, A: Allocator> VacantEntry<'a, T, A> {
    /// Returns a reference to the value that was probed.
    pub fn get(&self) -> &T {
        self.inner.key()
    }

    /// Inserts the value into the set.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during the reserve
    /// phase. The probed value is dropped on failure and the tree is left
    /// unmodified. For a variant that recovers the value, use
    /// [`Self::try_insert_give_back`].
    pub fn try_insert(self) -> Result<(), AllocError> {
        self.insert_marker().map(|_| ()).map_err(|(_, e)| e)
    }

    /// Inserts the value into the set. On allocation failure the value is
    /// handed back alongside the error so the caller can recover it.
    ///
    /// # Errors
    ///
    /// Returns `(value, AllocError)` if memory allocation fails during the
    /// reserve phase. The tree is left unmodified on failure.
    pub fn try_insert_give_back(self) -> Result<(), (T, AllocError)> {
        self.insert_marker().map(|_| ())
    }

    /// Takes ownership of the value that was originally passed to
    /// [`BTreeSet::entry`](super::BTreeSet::entry).
    pub fn into_value(self) -> T {
        self.inner.into_key()
    }

    /// Commits the probed value into the underlying tree by inserting the
    /// zero-sized marker. Pure implementation detail — see [`Self::try_insert`]
    /// and [`Self::try_insert_give_back`] for the public surface.
    fn insert_marker(self) -> Result<MapOccupied<'a, T, SetValZST, A>, (T, AllocError)> {
        self.inner
            .try_insert_entry(SetValZST)
            .map_err(|(k, _, e)| (k, e))
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for VacantEntry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry")
            .field(self.inner.key())
            .finish()
    }
}

/// An occupied entry in a [`BTreeSet`](super::BTreeSet).
pub struct OccupiedEntry<'a, T: 'a, A: Allocator = Global> {
    inner: MapOccupied<'a, T, SetValZST, A>,
}

impl<T: Ord, A: Allocator> OccupiedEntry<'_, T, A> {
    /// Gets a reference to the value corresponding to the entry in the set.
    pub fn get(&self) -> &T {
        self.inner.key()
    }

    /// Removes the value from the set, returning it.
    pub fn remove(self) -> T {
        self.inner.remove_entry().0
    }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for OccupiedEntry<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OccupiedEntry")
            .field(self.inner.key())
            .finish()
    }
}
