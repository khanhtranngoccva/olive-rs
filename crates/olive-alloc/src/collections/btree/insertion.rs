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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::map::BTreeMap;
    use crate::alloc::Global;

    #[test]
    fn insert_and_get_single() {
        let mut map = BTreeMap::new_in(Global);
        assert!(map.is_empty());

        map.try_insert(1, "one").unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn insert_multiple_no_split() {
        let mut map = BTreeMap::new_in(Global);
        // CAPACITY is 11, so 11 inserts fit in one leaf without splitting.
        for i in 0..11 {
            map.try_insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), 11);
        for i in 0..11 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_triggers_leaf_split() {
        let mut map = BTreeMap::new_in(Global);
        // 12th insert should trigger a leaf split.
        for i in 0..12 {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 12);
        let mut missing_count = 0;
        let mut first_missing = 0;
        for i in 0..12 {
            if map.get(&i).is_none() {
                missing_count += 1;
                if missing_count == 1 {
                    first_missing = i;
                }
            }
        }
        let h = map.root.as_ref().map_or(0, |r| r.height());
        assert_eq!(
            missing_count, 0,
            "Missing {} keys after leaf split, first missing={}, height={}",
            missing_count, first_missing, h
        );
    }

    #[test]
    fn insert_triggers_root_growth() {
        let mut map = BTreeMap::new_in(Global);
        // Measured for ascending sequential insertion: the first leaf split
        // (height 0 -> 1) happens at len 12, and the internal root itself
        // splits (height 1 -> 2, i.e. genuine root growth) at len 89. So we
        // insert past 89 to actually exercise the taller tree.
        for i in 0..95u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert_eq!(map.len(), 95);
        // Root growth means the tree is now at least two levels deep.
        let h = map.root.as_ref().map_or(0, |r| r.height());
        assert!(
            h >= 2,
            "expected multi-level tree (root grew), got height {}",
            h
        );
        for i in 0..95u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)), "missing key {}", i);
        }
    }

    #[test]
    fn insert_overwrite_existing_key() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, "first").unwrap();
        let old = map.try_insert(1, "second").unwrap();
        assert_eq!(old, Some("first"));
        assert_eq!(map.get(&1), Some(&"second"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn get_mut_returns_correct_value() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(5, 50).unwrap();
        map.try_insert(10, 100).unwrap();

        if let Some(v) = map.get_mut(&5) {
            *v = 55;
        }
        assert_eq!(map.get(&5), Some(&55));
        assert_eq!(map.get(&10), Some(&100));
    }

    #[test]
    fn reverse_order_insertion() {
        let mut map = BTreeMap::new_in(Global);
        for i in (0..100).rev() {
            map.try_insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }
}
