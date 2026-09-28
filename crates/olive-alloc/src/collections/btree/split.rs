use core::borrow::Borrow;
use core::mem::ManuallyDrop;

use super::map::BTreeMap;
use super::node::ForceResult::*;
use super::node::{InternalNode, Root};
use super::search::SearchResult::*;
use crate::vec::Vec;
use olive_core::alloc::AllocError;
use olive_core::alloc::AllocatorTryClone;

impl<K, V> Root<K, V> {
    /// Calculates the length of both trees that result from splitting up
    /// a given number of distinct key-value pairs.
    pub(super) fn calc_split_length(
        total_num: usize,
        root_a: &Root<K, V>,
        root_b: &Root<K, V>,
    ) -> (usize, usize) {
        let (length_a, length_b);
        if root_a.height() < root_b.height() {
            length_a = root_a.reborrow().calc_length();
            assert!(
                length_a <= total_num,
                "BTree*: left root length must not exceed original tree length"
            );
            #[allow(clippy::arithmetic_side_effects, reason = "asserted above")]
            {
                length_b = total_num - length_a;
            }
            debug_assert_eq!(length_b, root_b.reborrow().calc_length());
        } else {
            length_b = root_b.reborrow().calc_length();
            assert!(
                length_b <= total_num,
                "BTree*: right root length must not exceed original tree length"
            );
            #[allow(clippy::arithmetic_side_effects, reason = "asserted above")]
            {
                length_a = total_num - length_b;
            }
            debug_assert_eq!(length_a, root_a.reborrow().calc_length());
        }
        (length_a, length_b)
    }

    /// Split off a tree with key-value pairs at and after the given key.
    /// The result is meaningful only if the tree is ordered by key,
    /// and if the ordering of `Q` corresponds to that of `K`.
    /// If `self` respects all `BTreeMap` tree invariants, then both
    /// `self` and the returned tree will respect those invariants.
    // FIXME: there are a lot of panic points.
    pub(super) fn split_off<Q: ?Sized + Ord, A: AllocatorTryClone>(
        &mut self,
        key: &Q,
        alloc: &A,
    ) -> Result<Self, AllocError>
    where
        K: Borrow<Q>,
    {
        let left_root = self;
        let mut right_root = Root::new_pillar(left_root.height(), &alloc)?;
        let mut left_node = left_root.borrow_mut();
        let mut right_node = right_root.borrow_mut();

        loop {
            let mut split_edge = match left_node.search_node(key) {
                // key is going to the right tree
                Found(kv) => kv.left_edge(),
                GoDown(edge) => edge,
            };

            split_edge.move_suffix(&mut right_node);

            match (split_edge.force(), right_node.force()) {
                (Internal(edge), Internal(node)) => {
                    left_node = edge.descend();
                    right_node = node.first_edge().descend();
                }
                (Leaf(_), Leaf(_)) => break,
                _ => unreachable!(),
            }
        }

        // What happens if &alloc panics?
        left_root.fix_right_border(&alloc);
        right_root.fix_left_border(&alloc);
        Ok(right_root)
    }

    /// Creates a tree consisting of empty nodes.
    fn new_pillar<A: AllocatorTryClone>(height: usize, alloc: &A) -> Result<Self, AllocError> {
        let mut root = Root::new(alloc)?;
        let new_count = height;
        let mut ephemeral_stack =
            Vec::try_with_capacity_in(new_count, alloc).map_err(|_| AllocError)?;
        // Allocate additional nodes if we need more than what was cached.
        for _ in 0..new_count {
            let intermediate = unsafe { InternalNode::new(alloc)? };
            ephemeral_stack
                .try_push(intermediate)
                .expect("we just reserved enough items");
        }
        for _ in 0..height {
            root.push_internal_level(
                ephemeral_stack
                    .pop()
                    .expect("internal node should be available"),
            );
        }
        Ok(root)
    }
}

impl<K, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Splits the collection into two at the given key. Returns everything after the given key,
    /// including the key. If the key is not present, the split will occur at the nearest
    /// greater key, or return an empty map if no such key exists.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use olive_alloc::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(17, "d");
    /// a.insert(41, "e");
    ///
    /// let b = a.split_off(&3).unwrap();
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 3);
    ///
    /// assert_eq!(a[&1], "a");
    /// assert_eq!(a[&2], "b");
    ///
    /// assert_eq!(b[&3], "c");
    /// assert_eq!(b[&17], "d");
    /// assert_eq!(b[&41], "e");
    /// ```
    pub fn split_off<Q: ?Sized + Ord>(&mut self, key: &Q) -> Result<Self, AllocError>
    where
        K: Borrow<Q> + Ord,
    {
        let alloc = self.alloc.try_clone().map_err(|_| AllocError)?;
        if self.is_empty() {
            return Ok(Self::new_in(alloc));
        }

        let total_num = self.len();
        let left_root = self.root.as_mut().unwrap(); // unwrap succeeds because not empty
        let right_root = left_root.split_off(key, &alloc)?;

        let (new_left_len, right_len) = Root::calc_split_length(total_num, left_root, &right_root);
        self.length = new_left_len;

        Ok(BTreeMap {
            root: Some(right_root),
            length: right_len,
            alloc: ManuallyDrop::new(alloc),
            reserve_stack: ManuallyDrop::new(None),
        })
    }
}
