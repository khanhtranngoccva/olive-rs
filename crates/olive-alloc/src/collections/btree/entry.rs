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

use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::collections::btree::scratch;
use core::borrow::Borrow;
use core::marker::PhantomData;

use super::borrow::DormantMutRef;
use super::map::BTreeMap;
use super::node::{CAPACITY, Handle, NodeRef, marker};
use super::search::SearchResult;

impl<K: Ord, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
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
pub enum Entry<'a, K: 'a, V: 'a, A: AllocatorTryClone = Global> {
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, A>),
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, A>),
}

/// A vacant entry in a [`BTreeMap`].
pub struct VacantEntry<'a, K, V, A: AllocatorTryClone = Global> {
    pub(super) key: K,
    /// The edge handle in the target leaf where the key belongs, if the map
    /// is non-empty. `None` when the map is empty and a new root must be created.
    pub(super) handle: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

/// An occupied entry in a [`BTreeMap`].
pub struct OccupiedEntry<'a, K, V, A: AllocatorTryClone = Global> {
    pub(super) handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV>,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

impl<K, V, A: AllocatorTryClone> VacantEntry<'_, K, V, A> {
    /// Returns a reference to the key that was probed.
    pub(super) fn key(&self) -> &K {
        &self.key
    }
}

impl<K, V, A: AllocatorTryClone> OccupiedEntry<'_, K, V, A> {
    /// Gets a reference to the value in the entry.
    pub(super) fn get(&self) -> &V {
        self.handle.reborrow().into_kv().1
    }

    /// Gets a mutable reference to the value in the entry.
    pub(super) fn get_mut(&mut self) -> &mut V {
        self.handle.kv_mut().1
    }

    /// Sets the value of the entry with the `OccupiedEntry`'s key,
    /// and returns the entry's old value.
    pub(super) fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }

    /// Removes the key-value pair from the map, returning it as a tuple.
    ///
    /// The removed pair is taken out of the tree; if its removal leaves an
    /// underfull node, the tree is rebalanced (via merging or stealing) so that
    /// all invariants hold again.
    ///
    /// The general invariant is that if a node has siblings, it must not have fewer
    /// than MIN_LEN elements.
    pub(super) fn remove_entry(mut self) -> (K, V) {
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
fn do_two_phase<'a, K, V, A: AllocatorTryClone>(
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

impl<'a, K, V, A: AllocatorTryClone> VacantEntry<'a, K, V, A> {
    /// Inserts the value into the vacant slot, performing the full
    /// reserve-and-commit insertion if a split cascade is needed.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if memory allocation fails during the reserve
    /// phase. The tree is left unmodified on failure.
    pub(super) fn try_insert_entry(
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
