//! Public insertion API for BTreeMap.
//!
//! Thin facade exposing the user-facing insertion variants. All actual
//! insertion logic lives in [`entry`](super::entry) and
//! [`node`](super::node).

use crate::alloc::{AllocError, AllocatorTryClone};

use super::entry::Entry;
use super::map::BTreeMap;

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Inserts a key-value pair into the map, attempting allocation as needed.
    ///
    /// If the key already existed, the old value is replaced and returned.
    /// Otherwise, `None` is returned.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, AllocError> {
        match self.entry(key) {
            Entry::Occupied(mut occ) => Ok(Some(occ.insert(value))),
            Entry::Vacant(vac) => vac
                .try_insert_entry(value)
                .map(|_| None)
                .map_err(|(_, _, e)| e),
        }
    }

    /// Attempts to insert a key-value pair, returning the key and value back
    /// on allocation failure so the caller can retry or handle the error.
    ///
    /// On success returns `Ok(Some(old_value))` if the key was already present,
    /// or `Ok(None)` if it was newly inserted.
    ///
    /// # Errors
    ///
    /// Returns `Err((key, value))` if memory allocation fails. The tree is
    /// left unmodified on failure.
    pub fn try_insert_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Result<Option<V>, ()>, (K, V)> {
        match self.entry(key) {
            Entry::Occupied(mut occ) => Ok(Ok(Some(occ.insert(value)))),
            Entry::Vacant(vac) => match vac.try_insert_entry(value) {
                Ok(_) => Ok(Ok(None)),
                Err((k, v, _)) => Err((k, v)),
            },
        }
    }

    /// Inserts a key-value pair only if the key does not already exist.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was
    /// already present (in which case the existing value is unchanged).
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails.
    pub fn try_insert_unique(&mut self, key: K, value: V) -> Result<bool, AllocError> {
        match self.entry(key) {
            Entry::Occupied(_) => Ok(false),
            Entry::Vacant(vac) => vac
                .try_insert_entry(value)
                .map(|_| true)
                .map_err(|(_, _, e)| e),
        }
    }

    /// Like [`Self::try_insert_unique`], but returns the key and value back
    /// on allocation failure.
    ///
    /// Returns `Ok(true)` if the key was newly inserted, `Ok(false)` if it was
    /// already present.
    ///
    /// # Errors
    ///
    /// Returns `Err((key, value))` if memory allocation fails. The tree is
    /// left unmodified on failure.
    pub fn try_insert_unique_give_back(
        &mut self,
        key: K,
        value: V,
    ) -> Result<Result<bool, ()>, (K, V)> {
        match self.entry(key) {
            Entry::Occupied(_) => Ok(Ok(false)),
            Entry::Vacant(vac) => match vac.try_insert_entry(value) {
                Ok(_) => Ok(Ok(true)),
                Err((k, v, _)) => Err((k, v)),
            },
        }
    }
}
