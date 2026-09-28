use super::node::marker::{Edge, Immut, Leaf};
use super::node::{CAPACITY, Handle, InternalNode, LeafNode, NodeRef};
use crate::boxed::Box;
use crate::vec::Vec;
use olive_core::alloc::{AllocError, AllocatorTryClone};

/// Pre-reserved nodes for a split-and-insert operation.
pub(super) struct Nodes<K, V, A: AllocatorTryClone> {
    /// The leaf node to be reserved.
    pub(super) leaf: Option<Box<LeafNode<K, V>, A>>,
    /// The intermediate internal nodes.
    /// Note that these nodes are *not* fully initialized - they do not have any edges yet.
    pub(super) internals: Vec<Box<InternalNode<K, V>, A>, A>,
}

impl<K, V, A: AllocatorTryClone> Nodes<K, V, A> {
    /// Pops the next pre-reserved internal node, if any remain.
    pub(super) fn pop_internal(&mut self) -> Option<Box<InternalNode<K, V>, A>> {
        self.internals.pop()
    }
}

/// Given a cached buffer stored on a BTreeMap, allocate a sufficient number
/// of nodes to accommodate the next split-and-insert on the specified leaf.
///
/// Any previously cached nodes in `map_buffer` are reused (drained first),
/// and newly allocated nodes are appended. After this call, `map_buffer`
/// is empty — all nodes are owned by the returned `Nodes`.
///
/// # Errors
///
/// Returns [`AllocError`] if reservation of any node or storing space fails.
pub(super) fn reserve_for_insertion<K, V, A: AllocatorTryClone>(
    map_buffer: &mut Vec<Box<InternalNode<K, V>, A>, A>,
    handle: Handle<NodeRef<Immut<'_>, K, V, Leaf>, Edge>,
    alloc: &A,
) -> Result<Nodes<K, V, A>, AllocError> {
    let leaf: Box<LeafNode<K, V>, A> = LeafNode::new(alloc.try_clone().map_err(|_| AllocError)?)?;

    // Count how many internal nodes will be needed.
    let mut current = handle.node.ascend().ok();
    let mut internals_needed = 0usize;
    loop {
        match current {
            None => {
                internals_needed = internals_needed.checked_add(1).ok_or(AllocError)?;
                break;
            }
            Some(h) => {
                if h.node.len() < CAPACITY {
                    break;
                }
                internals_needed = internals_needed.checked_add(1).ok_or(AllocError)?;
                current = h.node.ascend().ok();
            }
        }
    }

    // Drain any previously cached nodes from the map's buffer.
    let cloned_alloc: A = alloc.try_clone().map_err(|_| AllocError)?;
    let mut internals: Vec<Box<InternalNode<K, V>, A>, A> = Vec::new_in(cloned_alloc);
    while let Some(node) = map_buffer.pop() {
        internals.try_push(node).map_err(|_| AllocError)?;
    }

    // Allocate additional nodes if we need more than what was cached.
    let new_count = internals_needed.saturating_sub(internals.len());
    for _ in 0..new_count {
        let intermediate: Box<InternalNode<K, V>, A> =
            unsafe { InternalNode::new(alloc.try_clone().map_err(|_| AllocError)?)? };
        internals.try_push(intermediate).map_err(|_| AllocError)?;
    }

    Ok(Nodes {
        leaf: Some(leaf),
        internals,
    })
}
