//! A fallible port of `alloc::collections::BTreeMap`.
//!
//! This implementation uses a reserve-and-commit architecture for insertion.
use olive_core::mem::ManuallyDrop;
use olive_core::ptr;

use super::node::{self, Root};
use crate::alloc::{Allocator, Global};

/// Minimum number of key-value pairs a non-root node must retain after removal.
/// A node with fewer than this is underfull and needs rebalancing.
pub(super) const MIN_LEN: usize = node::MIN_LEN_AFTER_SPLIT;

/// A B-tree based implementation of a ordered map, similar to std's BTreeMap.
pub struct BTreeMap<K, V, A: Allocator = Global> {
    pub(super) root: Option<Root<K, V>>,
    pub(super) length: usize,
    pub(super) alloc: ManuallyDrop<A>,
}

impl<K, V, A: Allocator> Drop for BTreeMap<K, V, A> {
    fn drop(&mut self) {
        // SAFETY: Mirrors std.
        // All fields are either trivially copyable or are stored in ManuallyDrop.
        drop(unsafe { ptr::read(self) }.into_iter())
    }
}
