//! A fallible port of `alloc::collections::BTreeMap`.
//!
//! This implementation uses a reserve-and-commit architecture for insertion.
use olive_core::mem::ManuallyDrop;
use olive_core::ptr;

use super::node::{self, Root};
use crate::alloc::{AllocError, AllocatorTryClone, Global};
use crate::boxed::Box;
use crate::collections::btree::node::InternalNode;
use crate::vec::Vec;

/// Minimum number of key-value pairs a non-root node must retain after removal.
/// A node with fewer than this is underfull and needs rebalancing.
pub(super) const MIN_LEN: usize = node::MIN_LEN_AFTER_SPLIT;

/// Converts a `TryCloneError` to an `AllocError`.
pub(super) fn try_clone_err_to_alloc_error(
    _e: olive_core::try_traits::try_clone::TryCloneError,
) -> AllocError {
    AllocError
}

/// A B-tree based implementation of a ordered map, similar to std's BTreeMap.
pub struct BTreeMap<K, V, A: AllocatorTryClone = Global> {
    pub(super) root: Option<Root<K, V>>,
    pub(super) length: usize,
    pub(super) alloc: ManuallyDrop<A>,
    /// Stack of reserved internal nodes awaiting commitment.
    /// During the reserve phase, newly allocated internal nodes are pushed here.
    /// The commit phase pops them as it climbs the tree.
    #[allow(clippy::type_complexity)]
    pub(super) reserve_stack: ManuallyDrop<Option<Vec<Box<InternalNode<K, V>, A>, A>>>,
}

impl<K, V, A: AllocatorTryClone> Drop for BTreeMap<K, V, A> {
    fn drop(&mut self) {
        // SAFETY: Mirrors std.
        // All fields are either trivially copyable or are stored in ManuallyDrop.
        drop(unsafe { ptr::read(self) }.into_iter())
    }
}
