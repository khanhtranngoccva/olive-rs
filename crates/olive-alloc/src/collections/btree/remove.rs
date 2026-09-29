use super::map::MIN_LEN;
use super::node::{ForceResult, Handle, LeftOrRight, NodeRef, marker};
use crate::alloc::Allocator;

// ── Deletion (remove + rebalance) ───────────────────────────────────────────

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::KV> {
    /// Removes a key-value pair from the tree, returning that pair along with the
    /// leaf edge where it collapsed into. If this empties an internal root node,
    /// the caller-supplied closure is invoked so the map can pop the extra level.
    ///
    /// Rebalancing is performed bottom-up: first the immediate underfull child of
    /// the removed pair's parent is fixed, then any ancestors left underfull by a
    /// merge are fixed recursively.
    #[allow(
        clippy::type_complexity,
        reason = "this is the best type representation"
    )]
    pub(super) fn remove_kv_tracking<F: FnOnce(), A: Allocator>(
        self,
        handle_emptied_internal_root: F,
        alloc: &A,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    ) {
        match self.force() {
            ForceResult::Leaf(node) => node.remove_leaf_kv(handle_emptied_internal_root, alloc),
            ForceResult::Internal(node) => {
                node.remove_internal_kv(handle_emptied_internal_root, alloc)
            }
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    /// Removes a key-value pair from a leaf node and rebalances the tree if the
    /// resulting leaf is underfull. Returns the removed pair and the leaf edge
    /// the pair collapsed into.
    #[allow(
        clippy::type_complexity,
        reason = "this is the best type representation"
    )]
    fn remove_leaf_kv<F: FnOnce(), A: Allocator>(
        self,
        handle_emptied_internal_root: F,
        // Difference from std: allocator references are used, cloning failures are not acceptable here.
        alloc: &A,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    ) {
        let (old_kv, mut pos) = self.remove();
        let len = pos.reborrow().into_node().len();
        if len < MIN_LEN {
            let idx = pos.idx();
            // We have to temporarily forget the child type, because there is no
            // distinct node type for the immediate parents of a leaf.
            let new_pos = match pos.into_node().forget_type().choose_parent_kv() {
                Ok(LeftOrRight::Left(left_parent_kv)) => {
                    debug_assert!(left_parent_kv.right_child_len() == MIN_LEN - 1);
                    if left_parent_kv.can_merge() {
                        left_parent_kv.merge_tracking_child_edge(LeftOrRight::Right(idx), alloc)
                    } else {
                        // right == MIN_LEN - 1,
                        // total > 2 * MIN_LEN + 1
                        // and left == total - 1 - right == total - 1 - (MIN_LEN - 1) == total - MIN_LEN.
                        // left > 2 * MIN_LEN + 1 - MIN_LEN == MIN_LEN + 1.
                        debug_assert!(left_parent_kv.left_child_len() > MIN_LEN);
                        left_parent_kv.steal_left(idx)
                    }
                }
                Ok(LeftOrRight::Right(right_parent_kv)) => {
                    debug_assert!(right_parent_kv.left_child_len() == MIN_LEN - 1);
                    if right_parent_kv.can_merge() {
                        right_parent_kv.merge_tracking_child_edge(LeftOrRight::Left(idx), alloc)
                    } else {
                        debug_assert!(right_parent_kv.right_child_len() > MIN_LEN);
                        right_parent_kv.steal_right(idx)
                    }
                }
                // We are at the root node. If it does not happen, there is always a node to its left or right.
                // ignore-tidy-undocumented-unsafe
                Err(pos) => unsafe { Handle::new_edge(pos, idx) },
            };
            // SAFETY: `new_pos` is the leaf we started from or a sibling.
            pos = unsafe { new_pos.cast_to_leaf_unchecked() };

            // Only if we merged, the parent (if any) has shrunk, but skipping
            // the following step otherwise does not pay off in benchmarks.
            //
            // SAFETY: We won't destroy or rearrange the leaf where `pos` is at
            // by handling its parent recursively; at worst we will destroy or
            // rearrange the parent through the grandparent, thus change the
            // link to the parent inside the leaf.
            if let Ok(parent) = unsafe { pos.reborrow_mut() }.into_node().ascend() {
                // SAFETY: the parent returned by `ascend` is always an internal node.
                if !parent
                    .into_node()
                    .forget_type()
                    .fix_node_and_affected_ancestors(alloc)
                {
                    handle_emptied_internal_root();
                }
            }
        }
        (old_kv, pos)
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Removes a key-value pair from an internal node. It swaps in an adjacent
    /// KV from a descendant leaf (preferring the left one) in place of the
    /// removed pair, then removes that leaf KV, which triggers the rebalancing.
    #[allow(
        clippy::type_complexity,
        reason = "this is the best type representation"
    )]
    fn remove_internal_kv<F: FnOnce(), A: Allocator>(
        self,
        handle_emptied_internal_root: F,
        alloc: &A,
    ) -> (
        (K, V),
        Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>,
    ) {
        // Remove an adjacent KV from its leaf and then put it back in place of
        // the element we were asked to remove. Prefer the left adjacent KV,
        // for the reasons listed in `choose_parent_kv`.
        let left_leaf_kv = self.left_edge().descend().last_leaf_edge().left_kv();
        // SAFETY: the left subtree of an internal KV always contains at least
        // one leaf element, so `left_kv` on the last leaf edge succeeds.
        let left_leaf_kv = unsafe { left_leaf_kv.ok().unwrap_unchecked() };

        let (left_kv, left_hole) = left_leaf_kv.remove_leaf_kv(handle_emptied_internal_root, alloc);

        // The internal node may have been stolen from or merged. Go back right
        // to find where the original KV ended up.
        // SAFETY: there is always a KV to the right of the left hole.
        let mut internal = unsafe { left_hole.next_kv().ok().unwrap_unchecked() };
        let old_kv = internal.replace_kv(left_kv.0, left_kv.1);
        let pos = internal.next_leaf_edge();
        (old_kv, pos)
    }
}
