use super::node::marker::{Edge, Immut, Leaf};
use super::node::{CAPACITY, Handle, InternalNode, LeafNode, NodeRef};
use crate::{boxed::Box, vec::Vec};
use core::ops::{Deref, DerefMut};
use olive_core::alloc::{AllocError, AllocatorTryClone};

/// An internal reusable buffer that stores pending node allocations on the heap.
/// When a bulk allocation operation fails or panics, the entire buffer gets drained
/// (or stored, whichever is faster).
pub(super) struct InternalBuffer<'map, K, V, A: AllocatorTryClone> {
    // Vector uses the same allocator as the parent to avoid issues.
    vec: &'map mut Vec<Box<InternalNode<K, V>, A>, A>,
}

impl<'map, K, V, A: AllocatorTryClone> InternalBuffer<'map, K, V, A> {
    pub(super) fn new(vec: &'map mut Vec<Box<InternalNode<K, V>, A>, A>) -> Self {
        Self { vec }
    }
}

impl<'map, K, V, A: AllocatorTryClone> Deref for InternalBuffer<'map, K, V, A> {
    type Target = Vec<Box<InternalNode<K, V>, A>, A>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<'map, K, V, A: AllocatorTryClone> DerefMut for InternalBuffer<'map, K, V, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<'map, K, V, A: AllocatorTryClone> Drop for InternalBuffer<'map, K, V, A> {
    fn drop(&mut self) {
        self.vec.clear();
    }
}

pub(super) struct Nodes<'map, K, V, A: AllocatorTryClone> {
    /// The leaf node to be reserved.
    pub(super) leaf: Option<Box<LeafNode<K, V>, A>>,
    /// The intermediate internal nodes.
    /// Note that these notes are *not* fully initialized - they do not have any edges yet.
    // FIXME: Which is the better approach? This is more performant, but it risks compiler and Miri errors due to janky borrows.
    // OOM deadlocks are also an issue because failed operations do not completely release all memory
    pub(super) internals: InternalBuffer<'map, K, V, A>,
}

/// Given a cached buffer stored on a BTreeMap, allocate a sufficient number
/// of nodes to accommodate the next split-and-insert on the specified leaf.
///
/// # Errors
///
/// Returns [`AllocError`] if reservation of any node or storing space fails.
pub(super) fn reserve_for_insertion<'map, 'unbounded, K, V, A: AllocatorTryClone>(
    map_buffer: &'map mut Vec<Box<InternalNode<K, V>, A>, A>,
    handle: Handle<NodeRef<Immut<'unbounded>, K, V, Leaf>, Edge>,
    alloc: &A,
) -> Result<Nodes<'map, K, V, A>, AllocError> {
    let leaf: Box<LeafNode<K, V>, A> = LeafNode::new(alloc.try_clone().map_err(|_| AllocError)?)?;
    let mut current = handle.node.ascend().ok();
    let mut internals = 0usize;
    let mut buf = InternalBuffer::new(map_buffer);
    loop {
        match current {
            // The current node is a root node. If this point is reached, a root node must be allocated.
            None => {
                internals = internals.checked_add(1).ok_or(AllocError)?;
                break;
            }
            Some(handle) => {
                // No node splitting is necessary if handle is full.
                if handle.node.len() < CAPACITY {
                    break;
                }
                internals = internals.checked_add(1).ok_or(AllocError)?;
                current = handle.node.ascend().ok();
            }
        }
    }
    // Reserve enough space to store `internals` elements.
    let cap = buf.capacity();
    buf.try_reserve_adaptive(internals.saturating_sub(cap))
        .map_err(|_| AllocError)?;
    let new_buffers = internals.saturating_sub(buf.len());
    for _ in 0..new_buffers {
        let intermediate: Box<InternalNode<K, V>, A> =
            unsafe { InternalNode::new(alloc.try_clone().map_err(|_| AllocError)?)? };
        buf.try_push_within_capacity(intermediate)
            .expect("reservation is sufficient for all intermediate node allocations");
    }
    Ok(Nodes {
        leaf: Some(leaf),
        internals: buf,
    })
}
