use core::ptr::NonNull;

use super::node::marker::{Edge, Immut, Leaf};
use super::node::{B, CAPACITY, Handle, InternalNode, LeafNode, NodeRef};
use crate::boxed::Box;
use olive_core::alloc::{AllocError, AllocatorTryClone};

/// Worst-case height of a B-tree with branching factor `B` and up to
/// `usize::MAX` keys, computed as `floor(log_B((n + 1) / 2))` (Wikipedia).
/// Height counts edges from root to leaf (a single-node tree has height 0).
/// With `n = usize::MAX`, `(n + 1) / 2 = 2^(BITS-1) = isize::MAX + 1`.
const MAX_HEIGHT: usize = {
    const BASE: u64 = B as u64;
    // (usize::MAX + 1) / 2 = 2^(BITS-1) = isize::MAX + 1.
    const N_PLUS_1_HALF: u64 = (isize::MAX as u64) + 1;
    let mut depth = 0usize;
    let mut power = 1u64;
    loop {
        // Stop before power overflows or exceeds N_HALF.
        if power > N_PLUS_1_HALF / BASE {
            break;
        }
        power *= BASE;
        depth += 1;
    }
    depth
};

/// Maximum number of internal nodes a single reserve-and-commit insertion
/// can need, for any tree that fits in this address space.
///
/// A tree of height `h` has `h` internal levels on the path from root to leaf.
/// In the worst case every one of those `h` ancestors is full and splits, each
/// producing one new internal node. At the maximum height (`MAX_HEIGHT`), the
/// tree already holds `usize::MAX` keys, so an insertion attempt cannot
/// trigger a root split (there is no room for more keys).
///
/// The bound is thus `MAX_HEIGHT`, plus one as a safety leeway against an
/// algorithm that over-reserves past the theoretical maximum — if one ever did,
/// [`reserve_for_insertion`] returns [`AllocError`] rather than overflowing
/// the fixed array.
const MAX_RESERVE_INTERNALS: usize = MAX_HEIGHT + 1;

/// Total number of nodes (internal + leaf) a single reserve-and-commit
/// insertion can reserve.
#[allow(
    unused,
    reason = "documentary constant; asserted in __assert_reserve_bounds"
)]
const MAX_RESERVE_NODES: usize = MAX_RESERVE_INTERNALS + 1;

#[allow(
    clippy::assertions_on_constants,
    reason = "intentional compile-time sanity check"
)]
const fn __assert_reserve_bounds() {
    // The total must be large enough to hold a real tree's worst case.
    assert!(MAX_RESERVE_NODES >= 5);
    // Internals + 1 leaf must equal total.
    assert!(MAX_RESERVE_INTERNALS + 1 == MAX_RESERVE_NODES);
}
const _ASSERT_RESERVE_BOUNDS: () = __assert_reserve_bounds();

/// Ephemeral reservation buffer for a single split-and-insert operation.
///
/// Stores raw `NonNull` pointers to pre-allocated nodes alongside a single
/// shared allocator reference to reduce memory overhead.
///
/// All nodes are allocated with a **borrowed** allocator reference (`&'a A`)
/// rather than an owned clone. This eliminates all allocator cloning during
/// both the reserve and commit phases.
///
/// This struct is **not persisted** across inserts. It is created locally in the
/// insert path, used during the commit phase, and dropped when the insert
/// completes. Unconsumed nodes are freed immediately, avoiding the deadlock risk
/// of holding many live allocations while contending for other resources.
///
/// Index `depth - 1` holds the next node to pop, so popping walks downward from
/// the highest reserved level.
pub(super) struct Nodes<'a, K, V, A: AllocatorTryClone> {
    /// Raw pointer to the pre-allocated leaf node.
    leaf: Option<NonNull<LeafNode<K, V>>>,
    /// Fixed-capacity stack of pre-reserved internal node pointers.
    internals: [Option<NonNull<InternalNode<K, V>>>; MAX_RESERVE_INTERNALS],
    /// Number of internal nodes currently stacked (top is at index `depth - 1`).
    depth: usize,
    /// Shared allocator reference used to hydrate `Box`es on pop.
    alloc: &'a A,
}

/// Panic-aware drop guard that borrows a [`Nodes`] buffer. If a panic occurs
/// mid-drop, the guard's `Drop` re-runs [`Nodes::do_drop`] on whatever remains.
struct DropGuard<'n, 'a, K, V, A: AllocatorTryClone> {
    nodes: &'n mut Nodes<'a, K, V, A>,
}

impl<K, V, A: AllocatorTryClone> Drop for DropGuard<'_, '_, K, V, A> {
    fn drop(&mut self) {
        // Called during unwind after a panic in the main drop loop.
        // Free everything still held. A second panic here aborts the process.
        self.nodes.do_drop();
    }
}

impl<K, V, A: AllocatorTryClone> Nodes<'_, K, V, A> {
    /// Destroys and deallocates every node still held in this buffer.
    /// Slots already taken to `None` are skipped.
    fn do_drop(&mut self) {
        while let Some(_node) = self.pop_internal() {}
        drop(self.take_leaf());
    }
}

impl<K, V, A: AllocatorTryClone> Drop for Nodes<'_, K, V, A> {
    fn drop(&mut self) {
        // Arm the guard before doing any work. The guard's `Drop`
        // re-runs `do_drop` on whatever is left if the implementation panics.
        let guard = DropGuard { nodes: self };
        guard.nodes.do_drop();
    }
}

impl<'a, K, V, A: AllocatorTryClone> Nodes<'a, K, V, A> {
    /// Creates a new empty buffer bound to the given allocator reference.
    pub(super) fn with_alloc(alloc: &'a A) -> Self {
        Self {
            leaf: None,
            internals: [None; MAX_RESERVE_INTERNALS],
            depth: 0,
            alloc,
        }
    }

    /// Takes the pre-allocated leaf node, hydrating it into a `Box`.
    ///
    /// Returns `None` if the leaf has already been taken.
    pub(super) fn take_leaf(&mut self) -> Option<Box<LeafNode<K, V>, &'a A>> {
        let ptr = self.leaf.take()?;
        // SAFETY: `ptr` was obtained from a valid `Box` allocation made by
        // `self.alloc`; alignment and liveness are preserved.
        Some(unsafe { Box::from_non_null_in(ptr, self.alloc) })
    }

    /// Pops the next pre-reserved internal node, hydrating it into a `Box`.
    pub(super) fn pop_internal(&mut self) -> Option<Box<InternalNode<K, V>, &'a A>> {
        if self.depth == 0 {
            return None;
        }
        #[allow(clippy::arithmetic_side_effects, reason = "guarded by depth != 0")]
        let idx = self.depth - 1;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "depth > 0, decrementing is safe"
        )]
        {
            self.depth -= 1;
        }
        let ptr = self.internals[idx].take()?;
        // SAFETY: `ptr` was obtained from a valid `Box` allocation made by
        // `self.alloc`; alignment and liveness are preserved.
        Some(unsafe { Box::from_non_null_in(ptr, self.alloc) })
    }

    /// Pushes a pre-reserved internal node onto the stack, dehydrating the
    /// `Box` into a raw pointer.
    ///
    /// # Safety
    ///
    /// The allocator used to allocate `node` must be identical to the one
    /// stored in this buffer (`self.alloc`). Dehydration strips the allocator
    /// reference, so a mismatch would cause deallocation through the wrong
    /// allocator later.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the stack already holds [`MAX_RESERVE_INTERNALS`]
    /// nodes. For a valid tree this is unreachable; it is a last-ditch backstop
    /// against an algorithm over-reserving past the theoretical maximum.
    pub(super) unsafe fn push_internal(
        &mut self,
        node: Box<InternalNode<K, V>, &'a A>,
    ) -> Result<(), AllocError> {
        debug_assert!(
            core::ptr::eq(*node.allocator(), self.alloc),
            "push_internal: allocator mismatch"
        );
        if self.depth >= MAX_RESERVE_INTERNALS {
            return Err(AllocError);
        }
        let (ptr, _) = Box::into_non_null_with_allocator(node);
        self.internals[self.depth] = Some(ptr);
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "depth < MAX_RESERVE_INTERNALS, increment is safe"
        )]
        {
            self.depth += 1;
        }
        Ok(())
    }

    /// Sets the leaf pointer, dehydrating the `Box` into a raw pointer.
    ///
    /// # Safety
    ///
    /// The allocator used to allocate `leaf` must be identical to the one
    /// stored in this buffer (`self.alloc`). Dehydration strips the allocator
    /// reference, so a mismatch would cause deallocation through the wrong
    /// allocator later.
    ///
    /// # Panics
    ///
    /// Panics if a leaf has already been set. Calling this twice would otherwise
    /// leak the first allocation; it is a caller bug and unrecoverable.
    pub(super) unsafe fn set_leaf(&mut self, leaf: Box<LeafNode<K, V>, &'a A>) {
        debug_assert!(
            core::ptr::eq(*leaf.allocator(), self.alloc),
            "set_leaf: allocator mismatch"
        );
        assert!(
            self.leaf.is_none(),
            "set_leaf called on a buffer that already holds a leaf"
        );
        let (ptr, _) = Box::into_non_null_with_allocator(leaf);
        self.leaf = Some(ptr);
    }
}

/// Allocate a sufficient number of nodes to accommodate the next split-and-insert
/// on the specified leaf. All nodes are allocated with a borrowed reference to
/// `alloc` — **zero** allocator clones are performed.
///
/// On success, the returned [`Nodes`] owns every reserved node. On failure, no
/// nodes leak (partially-filled buffers are dropped). The caller is responsible
/// for consuming the nodes during the commit phase; unconsumed nodes are freed
/// when the `Nodes` value is dropped.
///
/// # Errors
///
/// Returns [`AllocError`] if allocation of any node fails, or if the required
/// number of internals would exceed [`MAX_RESERVE_INTERNALS`] (impossible for a
/// valid tree).
pub(super) fn reserve_for_insertion<'a, K, V, A: AllocatorTryClone>(
    handle: Handle<NodeRef<Immut<'_>, K, V, Leaf>, Edge>,
    alloc: &'a A,
) -> Result<Nodes<'a, K, V, A>, AllocError> {
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

    // A valid tree can never demand more than MAX_RESERVE_INTERNALS internals.
    if internals_needed > MAX_RESERVE_INTERNALS {
        return Err(AllocError);
    }

    // Allocate the leaf first.
    let leaf: Box<LeafNode<K, V>, &A> = LeafNode::new(alloc)?;

    // Build the buffer, allocating each internal node with the borrowed reference.
    let mut buffer = Nodes::with_alloc(alloc);
    for _ in 0..internals_needed {
        // SAFETY: we initialize the node's data field immediately; edges are
        // filled by the subsequent `split` call in the same commit step.
        let intermediate: Box<InternalNode<K, V>, &A> = unsafe { InternalNode::new(alloc)? };
        // SAFETY: `intermediate` was allocated with `alloc`, which is identical
        // to the allocator stored in `buffer`.
        unsafe { buffer.push_internal(intermediate)? };
    }

    // SAFETY: `leaf` was allocated with `alloc`, which is identical to the
    // allocator stored in `buffer`.
    unsafe { buffer.set_leaf(leaf) };
    Ok(buffer)
}
