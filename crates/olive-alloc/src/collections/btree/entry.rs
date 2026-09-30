//! Fallible B-tree map entry operations via direct node manipulation.
//!
//! The central abstraction is [`Entry`], which mirrors std's
//! `alloc::collections::btree_map::entry::Entry`.
//!
//! # Reserve-and-Commit Architecture
//!
//! Insertion with cascading splits uses a strict two-phase approach:
//!
//! 1. **Reserve phase** — walk the tree bottom-up (reads only) to learn exactly
//!    which nodes will split and how deep the cascade goes. After that, allocate
//!    every node the commit needs in one batch.
//!    If any single allocation fails we drop the already-reserved nodes and
//!    return `Err`; because no mutation has touched the original tree yet, it
//!    remains completely intact.
//! 2. **Commit phase** — with every node already allocated, the actual splits
//!    are performed as pure pointer surgery. No allocation occurs here, so
//!    failure is impossible.

use crate::alloc::{AllocError, Allocator, Global};
use crate::collections::btree::scratch;
use core::borrow::Borrow;
use core::fmt;
use core::marker::PhantomData;
use olive_core::try_traits::try_default::TryDefault;

use super::borrow::DormantMutRef;
use super::map::BTreeMap;
use super::node::{CAPACITY, Handle, NodeRef, marker};
use super::search::SearchResult;
use super::{TryBTreeMapEntryWithDefaultError, TryBTreeMapEntryWithError};

impl<K: Ord, V, A: Allocator> BTreeMap<K, V, A> {
    /// Gets an [`Entry`] to a single entry in the map, which may either be
    /// occupied or vacant.
    ///
    /// This is the standard entry API, mirroring `std::collections::BTreeMap::entry`.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, A> {
        let (map, dormant_map) = DormantMutRef::new(self);
        match map.root {
            None => Entry::Vacant(VacantEntry {
                key,
                handle: None,
                dormant_map,
                _marker: PhantomData,
            }),
            Some(ref mut root) => match root.borrow_mut().search_tree(&key) {
                SearchResult::Found(handle) => Entry::Occupied(OccupiedEntry {
                    handle,
                    dormant_map,
                    _marker: PhantomData,
                }),
                SearchResult::GoDown(handle) => Entry::Vacant(VacantEntry {
                    key,
                    handle: Some(handle),
                    dormant_map,
                    _marker: PhantomData,
                }),
            },
        }
    }

    /// Like [`entry`](Self::entry) but accepts a borrowed key via `Borrow`.
    /// Only useful for lookups that don't need to insert (e.g. `remove`).
    pub(super) fn entry_ref<Q>(&mut self, key: &Q) -> Option<Entry<'_, K, V, A>>
    where
        Q: Ord + ?Sized,
        K: Ord + Borrow<Q>,
    {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root = map.root.as_mut()?;
        match root.borrow_mut().search_tree(key) {
            SearchResult::Found(handle) => Some(Entry::Occupied(OccupiedEntry {
                handle,
                dormant_map,
                _marker: PhantomData,
            })),
            SearchResult::GoDown(_) => None, // not found
        }
    }

    /// Gets an [`Entry`] to the first (lowest-keyed) entry in the map, or
    /// returns `None` if the map is empty.
    pub fn first_entry(&mut self) -> Option<Entry<'_, K, V, A>> {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root = map.root.as_mut()?;
        match root.borrow_mut().first_leaf_edge().right_kv() {
            Ok(handle) => Some(Entry::Occupied(OccupiedEntry {
                handle: handle.forget_node_type(),
                dormant_map,
                _marker: PhantomData,
            })),
            Err(_) => None,
        }
    }

    /// Gets an [`Entry`] to the last (highest-keyed) entry in the map, or
    /// returns `None` if the map is empty.
    pub fn last_entry(&mut self) -> Option<Entry<'_, K, V, A>> {
        let (map, dormant_map) = DormantMutRef::new(self);
        let root = map.root.as_mut()?;
        match root.borrow_mut().last_leaf_edge().left_kv() {
            Ok(handle) => Some(Entry::Occupied(OccupiedEntry {
                handle: handle.forget_node_type(),
                dormant_map,
                _marker: PhantomData,
            })),
            Err(_) => None,
        }
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`BTreeMap`].
pub enum Entry<'a, K: 'a, V: 'a, A: Allocator = Global> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, A>),
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, A>),
}

impl<'a, K: Ord, V, A: Allocator> Entry<'a, K, V, A> {
    /// Applies `f` to the value in the entry if it exists, returning the entry.
    /// Does nothing if the entry is vacant.
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(oe) = &mut self {
            f(oe.get_mut());
        }
        self
    }

    /// Applies `f` to the value in the entry if it exists, returning the entry.
    /// Does nothing if the entry is vacant.
    ///
    /// # Errors
    ///
    /// Returns `Err(e)` if the closure `f` returns an error.
    pub fn and_try_modify<E, F>(mut self, f: F) -> Result<Self, E>
    where
        E: core::error::Error,
        F: FnOnce(&mut V) -> Result<(), E>,
    {
        if let Entry::Occupied(oe) = &mut self {
            f(oe.get_mut())?;
        }
        Ok(self)
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Vacant(ve) => ve.key(),
            Entry::Occupied(oe) => oe.key(),
        }
    }

    /// Sets the value of the entry, and returns an [`OccupiedEntry`].
    ///
    /// # Errors
    ///
    /// Returns `(K, V, AllocError)` if memory allocation fails during
    /// insertion into a vacant entry. The tree is left unmodified on failure.
    pub fn try_insert_entry(
        self,
        value: V,
    ) -> Result<OccupiedEntry<'a, K, V, A>, (K, V, AllocError)> {
        match self {
            Entry::Occupied(mut oe) => {
                oe.insert(value);
                Ok(oe)
            }
            Entry::Vacant(ve) => ve.try_insert_entry(value),
        }
    }

    /// If a key maps to a value in this map, return a mutable reference to
    /// that value. Otherwise, construct a default value using [`TryDefault`],
    /// insert it into the map, and return a mutable reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`TryBTreeMapEntryWithDefaultError`] if either the default
    /// construction fails or an allocation error occurs during insertion.
    pub fn or_try_default(self) -> Result<&'a mut V, TryBTreeMapEntryWithDefaultError>
    where
        V: TryDefault,
    {
        match self {
            Entry::Occupied(oe) => Ok(oe.into_mut()),
            Entry::Vacant(ve) => {
                let value = V::try_default()?;
                match ve.try_insert(value) {
                    Ok(v) => Ok(v),
                    Err((_, _, e)) => Err(TryBTreeMapEntryWithDefaultError::Alloc(e)),
                }
            }
        }
    }

    /// If a key maps to a value in this map, return a mutable reference to
    /// that value. Otherwise, insert the given value and return a mutable
    /// reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during insertion.
    pub fn or_try_insert(self, value: V) -> Result<&'a mut V, AllocError> {
        match self {
            Entry::Occupied(oe) => Ok(oe.into_mut()),
            Entry::Vacant(ve) => match ve.try_insert(value) {
                Ok(v) => Ok(v),
                Err((_, _, e)) => Err(e),
            },
        }
    }

    /// If a key maps to a value in this map, return a mutable reference to
    /// that value. Otherwise, call `f` to compute a value, insert it into the
    /// map, and return a mutable reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`TryBTreeMapEntryWithError`] if either the closure fails or an
    /// allocation error occurs during insertion.
    pub fn or_try_insert_with<E, F>(self, f: F) -> Result<&'a mut V, TryBTreeMapEntryWithError<E>>
    where
        E: core::error::Error,
        F: FnOnce() -> Result<V, E>,
    {
        match self {
            Entry::Occupied(oe) => Ok(oe.into_mut()),
            Entry::Vacant(ve) => {
                let value = match f() {
                    Ok(v) => v,
                    Err(e) => return Err(TryBTreeMapEntryWithError::Closure(e)),
                };
                match ve.try_insert(value) {
                    Ok(v) => Ok(v),
                    Err((_, _, e)) => Err(TryBTreeMapEntryWithError::Alloc(e)),
                }
            }
        }
    }

    /// If a key maps to a value in this map, return a mutable reference to
    /// that value. Otherwise, call `f` with a reference to the key to compute
    /// a value, insert it into the map, and return a mutable reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`TryBTreeMapEntryWithError`] if either the closure fails or an
    /// allocation error occurs during insertion.
    pub fn or_try_insert_with_key<E, F>(
        self,
        f: F,
    ) -> Result<&'a mut V, TryBTreeMapEntryWithError<E>>
    where
        E: core::error::Error,
        F: FnOnce(&K) -> Result<V, E>,
    {
        match self {
            Entry::Occupied(oe) => Ok(oe.into_mut()),
            Entry::Vacant(ve) => {
                let value = match f(ve.key()) {
                    Ok(v) => v,
                    Err(e) => return Err(TryBTreeMapEntryWithError::Closure(e)),
                };
                match ve.try_insert(value) {
                    Ok(v) => Ok(v),
                    Err((_, _, e)) => Err(TryBTreeMapEntryWithError::Alloc(e)),
                }
            }
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, A: Allocator> fmt::Debug for Entry<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Vacant(ve) => ve.fmt(f),
            Entry::Occupied(oe) => oe.fmt(f),
        }
    }
}

/// A vacant entry in a [`BTreeMap`].
pub struct VacantEntry<'a, K, V, A: Allocator = Global> {
    pub(super) key: K,
    /// The edge handle in the target leaf where the key belongs, if the map
    /// is non-empty. `None` when the map is empty and a new root must be created.
    pub(super) handle: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

/// An occupied entry in a [`BTreeMap`].
pub struct OccupiedEntry<'a, K, V, A: Allocator = Global> {
    pub(super) handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV>,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K, V, A: Allocator> VacantEntry<'a, K, V, A> {
    /// Returns a reference to the key that was probed.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes back the key that was originally passed to [`BTreeMap::entry`].
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns a mutable reference to it.
    ///
    /// # Errors
    ///
    /// Returns `(K, V, AllocError)` if memory allocation fails during the
    /// reserve phase. The tree is left unmodified on failure.
    pub fn try_insert(self, value: V) -> Result<&'a mut V, (K, V, AllocError)> {
        self.try_insert_entry(value).map(|oe| oe.into_mut())
    }

    /// Sets the value of the entry with the `VacantEntry`'s key,
    /// and returns an [`OccupiedEntry`].
    ///
    /// # Errors
    ///
    /// Returns `(K, V, AllocError)` if memory allocation fails during the
    /// reserve phase. The tree is left unmodified on failure.
    pub fn try_insert_entry(
        mut self,
        value: V,
    ) -> Result<OccupiedEntry<'a, K, V, A>, (K, V, AllocError)> {
        // This guard is in place because overflowing is a possibility on ZSTs
        if unsafe { self.dormant_map.reborrow() }.length == usize::MAX {
            return Err((self.key, value, AllocError));
        }
        let leaf_kv_handle = match self.handle {
            None => {
                // SAFETY: There is no tree yet so no reference to it exists.
                let map = unsafe { self.dormant_map.reborrow() };
                let node_ref = match NodeRef::new_leaf(&*map.alloc) {
                    Ok(a) => a,
                    Err(e) => return Err((self.key, value, e)),
                };
                let root = map.root.insert(node_ref.forget_type());
                // SAFETY: We *just* created the root as a leaf, and we're
                // stacking the new handle on the original borrow lifetime.
                unsafe {
                    let mut leaf = root.borrow_mut().cast_to_leaf_unchecked();
                    leaf.push_with_handle(self.key, value)
                }
            }
            Some(handle) => {
                // Special case - no need for the two phase routine
                if handle.node.len() < CAPACITY {
                    unsafe { handle.insert_fit(self.key, value) }
                } else {
                    do_two_phase(&mut self.dormant_map, handle, self.key, value)?
                }
            }
        };

        // SAFETY: modifying the length doesn't invalidate handles to existing nodes.
        #[allow(clippy::arithmetic_side_effects, reason = "length is capped above")]
        unsafe {
            self.dormant_map.reborrow().length += 1
        };
        Ok(OccupiedEntry {
            handle: leaf_kv_handle.forget_node_type(),
            dormant_map: self.dormant_map,
            _marker: PhantomData,
        })
    }
}

impl<K: fmt::Debug, V: fmt::Debug, A: Allocator> fmt::Debug for VacantEntry<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry")
            .field("key", &self.key)
            .finish()
    }
}

impl<'a, K, V, A: Allocator> OccupiedEntry<'a, K, V, A> {
    /// Gets a reference to the key corresponding to the entry in the map.
    pub fn key(&self) -> &K {
        self.handle.reborrow().into_kv().0
    }

    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.handle.reborrow().into_kv().1
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// Use [`Self::into_mut`] if you want a reference that is independent
    /// from this entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.handle.kv_mut().1
    }

    /// Converts the occupied entry into a mutable reference to the value.
    ///
    /// If you need to reuse the [`OccupiedEntry`], please use [`Self::get_mut`]
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_mut(self) -> &'a mut V {
        self.handle.into_val_mut()
    }

    /// Sets the value of the entry with the `OccupiedEntry`'s key,
    /// and returns the entry's old value.
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }

    /// Removes the key-value pair from the map, returning it as a tuple.
    // The removed pair is taken out of the tree; if its removal leaves an
    // underfull node, the tree is rebalanced (via merging or stealing) so that
    // all invariants hold again.
    //
    // The general invariant is that if a node is not a root node, it must not have fewer
    // than MIN_LEN elements.
    pub fn remove_entry(mut self) -> (K, V) {
        let mut emptied_internal_root = false;
        let map = unsafe { self.dormant_map.reborrow() };
        // Use references here to avoid cloning.
        let (old_kv, _) = self
            .handle
            .remove_kv_tracking(|| emptied_internal_root = true, &*map.alloc);
        // SAFETY: we consumed the intermediate root borrow held by `self.handle`.
        let map = unsafe { self.dormant_map.awaken() };
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "we just removed an existing element"
        )]
        {
            map.length -= 1;
        }
        if emptied_internal_root {
            let root = map.root.as_mut().unwrap();
            root.pop_internal_level(&*map.alloc);
        }
        old_kv
    }

    /// Removes the key-value pair from the map, returning only the value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

impl<K: fmt::Debug, V: fmt::Debug, A: Allocator> fmt::Debug for OccupiedEntry<'_, K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (k, v) = self.handle.reborrow().into_kv();
        f.debug_list().entry(k).entry(v).finish()
    }
}

// ── Two-phase insertion (reserve + commit) ─────────────────────────────────

/// Performs a reserve-and-commit insertion at a full leaf.
///
/// Phase 1 (probe + reserve): allocates a fresh leaf plus one internal node per full
/// ancestor (plus one more if the root itself splits). On any allocation
/// failure the reserved nodes are dropped and the untouched tree is returned
/// unchanged.
///
/// Phase 2 (commit): performs the splits bottom-up using only the reserved
/// nodes, then splices the resulting subtree back into the map's root. Because
/// every allocation succeeded, this phase cannot fail.
#[allow(
    clippy::type_complexity,
    reason = "this type declaration is inherently complex"
)]
fn do_two_phase<'a, K, V, A: Allocator>(
    map: &mut DormantMutRef<'a, BTreeMap<K, V, A>>,
    handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    key: K,
    value: V,
) -> Result<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV>, (K, V, AllocError)> {
    let immut_reborrow = handle.reborrow();
    // Reserve phase: allocate all needed nodes with a borrowed allocator reference.
    // Zero allocator clones are performed. On any allocation failure the partially
    // allocated nodes are dropped and the untouched tree is returned unchanged.
    let map_ref = unsafe { map.reborrow() };
    let mut nodes = match scratch::reserve_for_insertion(immut_reborrow, &*map_ref.alloc) {
        Ok(nodes) => nodes,
        Err(e) => return Err((key, value, e)),
    };
    // Commit phase: perform the splits bottom-up using the reserved nodes.
    let new_handle = handle.insert_recursing(key, value, &mut nodes, |ins, new_node| {
        // SAFETY: Pushing a new root node doesn't invalidate
        // handles to existing nodes.
        let map = unsafe { map.reborrow() };
        let root = map.root.as_mut().unwrap(); // same as ins.left
        root.push_internal_level(new_node)
            .push(ins.kv.0, ins.kv.1, ins.right)
    });
    Ok(new_handle)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use core::convert::Infallible;

    use olive_core::alloc::AllocError;

    use super::super::map::BTreeMap;
    use super::{Entry, OccupiedEntry, VacantEntry};
    use crate::alloc::Global;
    use crate::collections::btree::{TryBTreeMapEntryWithDefaultError, TryBTreeMapEntryWithError};
    use crate::test_helpers::NoDefault;

    // An error type for and_try_modify / or_try_insert_with closures.
    #[derive(Debug, PartialEq, Eq)]
    struct MyErr(u8);
    impl core::fmt::Display for MyErr {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "MyErr({})", self.0)
        }
    }
    impl core::error::Error for MyErr {}

    // ── Entry::and_modify / and_try_modify ────────────────────────────────────

    #[test]
    fn and_modify_changes_value_when_occupied() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        map.entry(1).and_modify(|v| *v += 5);
        assert_eq!(map.get(&1), Some(&15));
    }

    #[test]
    fn and_modify_is_noop_when_vacant() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        map.entry(42).and_modify(|v| *v += 5);
        assert!(map.is_empty());
    }

    #[test]
    fn and_try_modify_applies_closure_on_success() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let _entry = map
            .entry(1)
            .and_try_modify(|v| -> Result<(), Infallible> {
                *v *= 2;
                Ok(())
            })
            .unwrap();
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn and_try_modify_returns_error_from_closure() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let res: Result<_, MyErr> = map.entry(1).and_try_modify(|_| Err(MyErr(7)));
        assert_eq!(res.err(), Some(MyErr(7)));
        // Value untouched on error.
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn and_try_modify_ok_when_vacant() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let entry = map
            .entry(99)
            .and_try_modify(|_| -> Result<(), Infallible> { Ok(()) })
            .unwrap();
        assert!(matches!(entry, Entry::Vacant(_)));
        assert!(map.is_empty());
    }

    // ── Entry::key ────────────────────────────────────────────────────────────

    #[test]
    fn key_of_both_variants() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(3, "three").unwrap();
        assert_eq!(map.entry(3).key(), &3);
        assert_eq!(map.entry(4).key(), &4);
    }

    // ── Entry::try_insert_entry ───────────────────────────────────────────────

    #[test]
    fn try_insert_entry_into_vacant_slot() {
        let mut map = BTreeMap::new_in(Global);
        let oe = map.entry(1).try_insert_entry(100).unwrap();
        assert_eq!(oe.key(), &1);
        assert_eq!(oe.get(), &100);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&100));
    }

    #[test]
    fn try_insert_entry_replaces_existing_value() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 1).unwrap();
        let oe = map.entry(1).try_insert_entry(2).unwrap();
        assert_eq!(oe.get(), &2);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&2));
    }

    #[test]
    fn try_insert_entry_across_leaf_splits() {
        // CAPACITY is 11 (B = 6), so the 12th insert forces a leaf split.
        let mut map = BTreeMap::new_in(Global);
        for i in 0..11 {
            map.entry(i).try_insert_entry(i * 10).unwrap();
        }
        let oe = map.entry(11).try_insert_entry(110).unwrap();
        assert_eq!(oe.get(), &110);
        assert_eq!(map.len(), 12);
        for i in 0..12 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    // ── Entry::or_try_default ─────────────────────────────────────────────────

    #[test]
    fn or_try_default_returns_existing_value() {
        let mut map: BTreeMap<i32, u64> = BTreeMap::new_in(Global);
        map.try_insert(1, 42).unwrap();
        let v = map.entry(1).or_try_default().unwrap();
        assert_eq!(*v, 42);
    }

    #[test]
    fn or_try_default_constructs_and_inserts() {
        let mut map: BTreeMap<i32, u64> = BTreeMap::new_in(Global);
        let v = map.entry(1).or_try_default().unwrap();
        assert_eq!(*v, 0);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&0));
    }

    #[test]
    fn or_try_default_propagates_construction_failure() {
        let mut map: BTreeMap<i32, NoDefault> = BTreeMap::new_in(Global);
        let err = map.entry(1).or_try_default().unwrap_err();
        assert!(matches!(err, TryBTreeMapEntryWithDefaultError::Default(_)));
        assert!(map.is_empty());
    }

    // ── Entry::or_try_insert ──────────────────────────────────────────────────

    #[test]
    fn or_try_insert_keeps_existing_value() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 7).unwrap();
        let v = map.entry(1).or_try_insert(99).unwrap();
        assert_eq!(*v, 7);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn or_try_insert_inserts_when_vacant() {
        let mut map = BTreeMap::new_in(Global);
        let v = map.entry(1).or_try_insert(99).unwrap();
        assert_eq!(*v, 99);
        assert_eq!(map.get(&1), Some(&99));
    }

    #[test]
    fn or_try_insert_alloc_error_on_split() {
        use crate::test_helpers::BudgetedAlloc;

        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        // Fill the sole leaf to CAPACITY (11) so the next insert forces a split.
        for i in 0..11i32 {
            map.entry(i).or_try_insert((i * 10) as u32).unwrap();
        }
        assert_eq!(map.len(), 11);

        // Exhaust the budget so the split's allocation fails mid-insertion.
        alloc.drain();

        let err = map.entry(11).or_try_insert(110).unwrap_err();
        assert!(matches!(err, AllocError));
        // Tree left unmodified: the failed pair is not present.
        assert_eq!(map.len(), 11);
        assert_eq!(map.get(&11), None);
    }

    // ── Entry::or_try_insert_with ─────────────────────────────────────────────

    #[test]
    fn or_try_insert_with_skips_closure_when_occupied() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 5).unwrap();
        let v = map
            .entry(1)
            .or_try_insert_with(|| -> Result<u32, MyErr> {
                panic!("closure must not run when occupied")
            })
            .unwrap();
        assert_eq!(*v, 5);
    }

    #[test]
    fn or_try_insert_with_computes_and_inserts() {
        let mut map = BTreeMap::new_in(Global);
        let v = map
            .entry(1)
            .or_try_insert_with(|| -> Result<u32, Infallible> { Ok(123) })
            .unwrap();
        assert_eq!(*v, 123);
        assert_eq!(map.get(&1), Some(&123));
    }

    #[test]
    fn or_try_insert_with_reports_closure_error() {
        let mut map = BTreeMap::new_in(Global);
        let err = map
            .entry(1)
            .or_try_insert_with(|| -> Result<u32, MyErr> { Err(MyErr(3)) })
            .unwrap_err();
        assert!(matches!(err, TryBTreeMapEntryWithError::Closure(MyErr(3))));
        assert!(map.is_empty());
    }

    // FIXME: should have a test case with allocation error

    // ── Entry::or_try_insert_with_key ─────────────────────────────────────────

    #[test]
    fn or_try_insert_with_key_passes_key_to_closure() {
        let mut map = BTreeMap::new_in(Global);
        let v = map
            .entry(10)
            .or_try_insert_with_key(|k| -> Result<u32, Infallible> { Ok(*k * 2) })
            .unwrap();
        assert_eq!(*v, 20);
        assert_eq!(map.get(&10), Some(&20));
    }

    #[test]
    fn or_try_insert_with_key_skips_closure_when_occupied() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(10, 5).unwrap();
        let v = map
            .entry(10)
            .or_try_insert_with_key(|_| -> Result<u32, MyErr> {
                panic!("closure must not run when occupied")
            })
            .unwrap();
        assert_eq!(*v, 5);
    }

    #[test]
    fn or_try_insert_with_key_reports_closure_error() {
        let mut map = BTreeMap::new_in(Global);
        let err = map
            .entry(10)
            .or_try_insert_with_key(|_| -> Result<u32, MyErr> { Err(MyErr(9)) })
            .unwrap_err();
        assert!(matches!(err, TryBTreeMapEntryWithError::Closure(MyErr(9))));
        assert!(map.is_empty());
    }

    #[test]
    fn or_try_insert_with_key_alloc_error_on_split() {
        use crate::test_helpers::BudgetedAlloc;

        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        for i in 0..11i32 {
            map.entry(i).or_try_insert((i * 10) as u32).unwrap();
        }
        assert_eq!(map.len(), 11);

        alloc.drain();

        let err = map
            .entry(11)
            .or_try_insert_with_key(|k| -> Result<u32, Infallible> { Ok((*k * 10) as u32) })
            .unwrap_err();
        assert!(matches!(err, TryBTreeMapEntryWithError::Alloc(_)));
        assert_eq!(map.len(), 11);
        assert_eq!(map.get(&11), None);
    }

    // ── VacantEntry::try_insert / try_insert_entry ────────────────────────────

    #[test]
    fn vacant_try_insert_returns_mut_ref() {
        let mut map = BTreeMap::new_in(Global);
        if let Entry::Vacant(v) = map.entry(1) {
            let val = v.try_insert(37).unwrap();
            assert_eq!(*val, 37);
        } else {
            panic!("expected vacant entry");
        }
        assert_eq!(map.get(&1), Some(&37));
    }

    #[test]
    fn vacant_try_insert_into_empty_map_creates_root() {
        let mut map = BTreeMap::new_in(Global);
        assert!(map.is_empty());
        if let Entry::Vacant(v) = map.entry(0) {
            let val = v.try_insert(0).unwrap();
            assert_eq!(*val, 0);
        } else {
            panic!("expected vacant entry");
        }
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn vacant_try_insert_triggers_split() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..11 {
            map.entry(i).try_insert_entry(i).unwrap();
        }
        if let Entry::Vacant(v) = map.entry(11) {
            let val = v.try_insert(11).unwrap();
            assert_eq!(*val, 11);
        } else {
            panic!("expected vacant entry");
        }
        assert_eq!(map.len(), 12);
        for i in 0..12 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn vacant_try_insert_entry_returns_occupied_entry() {
        let mut map = BTreeMap::new_in(Global);
        if let Entry::Vacant(v) = map.entry(5) {
            let oe = v.try_insert_entry(50).unwrap();
            assert_eq!(oe.key(), &5);
            assert_eq!(oe.get(), &50);
        } else {
            panic!("expected vacant entry");
        }
        assert_eq!(map.get(&5), Some(&50));
    }

    #[test]
    fn vacant_try_insert_entry_alloc_error_on_split() {
        use crate::test_helpers::BudgetedAlloc;

        let alloc = BudgetedAlloc::new(1 << 20);
        let mut map: BTreeMap<i32, u32, BudgetedAlloc> = BTreeMap::new_in(alloc.clone());
        for i in 0..11i32 {
            map.entry(i).try_insert_entry((i * 10) as u32).unwrap();
        }
        assert_eq!(map.len(), 11);

        alloc.drain();

        let (k, v, e) = map.entry(11).try_insert_entry(110).unwrap_err();
        assert_eq!((k, v), (11, 110));
        assert!(matches!(e, AllocError));
        assert_eq!(map.len(), 11);
        assert_eq!(map.get(&11), None);
    }

    // ── OccupiedEntry::insert / into_mut / remove ─────────────────────────────

    #[test]
    fn occupied_insert_replaces_and_returns_old() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 1).unwrap();
        if let Entry::Occupied(mut o) = map.entry(1) {
            let old = o.insert(2);
            assert_eq!(old, 1);
            assert_eq!(*o.get(), 2);
        } else {
            panic!("expected occupied entry");
        }
        assert_eq!(map.get(&1), Some(&2));
    }

    // FIXME: missing get_mut

    #[test]
    fn occupied_into_mut_gives_live_reference() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        if let Entry::Occupied(o) = map.entry(1) {
            *o.into_mut() += 1;
        }
        assert_eq!(map.get(&1), Some(&11));
    }

    #[test]
    fn occupied_remove_entry_removes_pair() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 100).unwrap();
        map.try_insert(2, 200).unwrap();
        if let Entry::Occupied(o) = map.entry(1) {
            let (k, v) = o.remove_entry();
            assert_eq!((k, v), (1, 100));
        }
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), Some(&200));
    }

    #[test]
    fn occupied_remove_returns_only_value() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 100).unwrap();
        if let Entry::Occupied(o) = map.entry(1) {
            assert_eq!(o.remove(), 100);
        }
        assert!(map.is_empty());
    }

    // ── VacantEntry::key / into_key ───────────────────────────────────────────

    #[test]
    fn vacant_key_and_into_key() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        match map.entry(7) {
            Entry::Vacant(v) => {
                assert_eq!(v.key(), &7);
                assert_eq!(v.into_key(), 7);
            }
            _ => panic!("expected vacant entry"),
        }
        assert!(map.is_empty());
    }

    // ── Type-level sanity ─────────────────────────────────────────────────────

    #[test]
    fn entry_variant_types_are_public() {
        let mut map: BTreeMap<i32, u32> = BTreeMap::new_in(Global);
        let _: Option<VacantEntry<i32, u32>> = None;
        let _: Option<OccupiedEntry<i32, u32>> = None;
        match map.entry(1) {
            Entry::Vacant(_) | Entry::Occupied(_) => {}
        }
    }
}
