//! Fallible B-tree map entry operations via direct node manipulation.
//!
//! The central abstraction is [`VacantEntry`], which mirrors std's
//! `alloc::collections::btree_map::entry::VacantEntry`. It holds a mutable
//! leaf handle obtained during a read-only probe, plus a dormant reference
//! back to the owning map for fallible allocation.
//!
//! # Reserve-and-Commit Architecture
//!
//! Insertion with cascading splits uses a strict three-phase approach:
//!
//! 1. **Probe phase** — walk the tree bottom-up (reads only) to learn exactly
//!    which nodes will split and how deep the cascade goes.
//! 2. **Reserve phase** — allocate every node the commit needs in one batch.
//!    If any single allocation fails we drop the already-reserved nodes and
//!    return `Err`; because no mutation has touched the original tree yet, it
//!    remains completely intact.
//! 3. **Commit phase** — with every node already allocated, the actual splits
//!    are performed as pure pointer surgery. No allocation occurs here, so
//!    failure is impossible.

use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::collections::btree::scratch;
use crate::vec::Vec;
use core::marker::PhantomData;

use super::borrow::DormantMutRef;
use super::map::BTreeMap;
use super::node::{self, CAPACITY, Handle, NodeRef, Root, marker};
use super::scratch::reserve_for_insertion;

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
    /// `None` for a (empty) map without root
    pub(super) handle: Option<
        Handle<NodeRef<node::marker::Mut<'a>, K, V, node::marker::Leaf>, node::marker::Edge>,
    >,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    /// The BTreeMap will outlive this IntoIter so we don't care about drop order for `alloc`.
    pub(super) alloc: A,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

/// An occupied entry in a [`BTreeMap`].
pub struct OccupiedEntry<'a, K, V, A: AllocatorTryClone = Global> {
    pub(super) handle: Handle<
        NodeRef<node::marker::Mut<'a>, K, V, node::marker::LeafOrInternal>,
        node::marker::KV,
    >,
    pub(super) dormant_map: DormantMutRef<'a, BTreeMap<K, V, A>>,
    /// The BTreeMap will outlive this IntoIter so we don't care about drop order for `alloc`.
    pub(super) alloc: A,
    // Be invariant in `K` and `V`
    pub(super) _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K, V, A: AllocatorTryClone> VacantEntry<'a, K, V, A> {
    /// Returns a reference to the key that was probed.
    pub(super) fn key(&self) -> &K {
        &self.key
    }

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
        let handle = match self.handle {
            None => {
                let alloc = match self.alloc.try_clone() {
                    Ok(a) => a,
                    Err(_) => return Err((self.key, value, AllocError)),
                };
                let node_ref = match NodeRef::new_leaf(alloc) {
                    Ok(a) => a,
                    Err(e) => return Err((self.key, value, e)),
                };
                // SAFETY: There is no tree yet so no reference to it exists.
                let map = unsafe { self.dormant_map.reborrow() };
                let root = map.root.insert(node_ref.forget_type());
                // SAFETY: We *just* created the root as a leaf, and we're
                // stacking the new handle on the original borrow lifetime.
                unsafe {
                    let mut leaf = root.borrow_mut().cast_to_leaf_unchecked();
                    leaf.push_with_handle(self.key, value)
                }
            }
            Some(mut handle) => {
                // Special case - no need for the three phase routine
                if handle.node.len() < CAPACITY {
                    unsafe { handle.node.push_with_handle(self.key, value) }
                } else {
                    do_two_phase(&mut self.dormant_map, handle, &self.alloc, self.key, value)?
                }
            }
        };

        // SAFETY: modifying the length doesn't invalidate handles to existing nodes.
        unsafe { self.dormant_map.reborrow().length += 1 };
        Ok(OccupiedEntry {
            handle: handle.forget_node_type(),
            dormant_map: self.dormant_map,
            alloc: self.alloc,
            _marker: PhantomData,
        })
    }
}

// ── Two-phase insertion (reserve + commit) ─────────────────────────────────

/// Performs a reserve-and-commit insertion at a full leaf.
///
/// Phase 1 (probe): re-walks the tree immutably from the root to determine
/// exactly which ancestors are full and thus must split. This produces an
/// immutable edge handle onto the target leaf.
///
/// Phase 2 (reserve): allocates a fresh leaf plus one internal node per full
/// ancestor (plus one more if the root itself splits). On any allocation
/// failure the reserved nodes are dropped and the untouched tree is returned
/// unchanged.
///
/// Phase 3 (commit): performs the splits bottom-up using only the reserved
/// nodes, then splices the resulting subtree back into the map's root. Because
/// every allocation succeeded, this phase cannot fail.
fn do_two_phase<'a, K, V, A: AllocatorTryClone>(
    map: &mut DormantMutRef<'a, BTreeMap<K, V, A>>,
    handle: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    alloc: &A,
    // FIXME: should return KV on error for give back semantics
    key: K,
    value: V,
) -> Result<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV>, (K, V, AllocError)> {
    let immut_reborrow = handle.reborrow();
    let reserve_stack = &mut unsafe { map.reborrow() }.reserve_stack;
    if reserve_stack.is_none() {
        let cloned = match alloc.try_clone() {
            Ok(allocator) => allocator,
            Err(_) => return Err((key, value, AllocError)),
        };
        *reserve_stack = Some(Vec::new_in(cloned));
    }
    let stack = reserve_stack
        .as_mut()
        .expect("reserve stack is just initialized");
    let nodes = match scratch::reserve_for_insertion(stack, immut_reborrow, alloc) {
        Ok(nodes) => nodes,
        Err(e) => return Err((key, value, e)),
    };
    let new_handle = handle.insert_recursing(key, value, nodes, |ins, new_node| {
        // SAFETY: Pushing a new root node doesn't invalidate
        // handles to existing nodes.
        let map = unsafe { map.reborrow() };
        let root = map.root.as_mut().unwrap(); // same as ins.left
        root.push_internal_level(new_node)
            .push(ins.kv.0, ins.kv.1, ins.right)
    });
    Ok(new_handle)
}

/// Finds the edge index in a node where the given key should be inserted.
fn find_edge_index<K: Ord, V, Q: Ord + ?Sized>(
    leaf_ptr: *mut node::LeafNode<K, V>,
    key: &Q,
) -> usize
where
    K: core::borrow::Borrow<Q>,
{
    let len = unsafe { (*leaf_ptr).len as usize };
    // SAFETY: the first `len` elements of `keys` are initialized.
    let keys = unsafe { core::slice::from_raw_parts((*leaf_ptr).keys.as_ptr().cast::<K>(), len) };
    keys.partition_point(|k| k.borrow() < key)
}
