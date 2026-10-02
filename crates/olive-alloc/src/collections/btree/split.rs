use core::borrow::Borrow;
use core::mem::ManuallyDrop;

use super::map::BTreeMap;
use super::node::ForceResult::*;
use super::node::{Handle, InternalNode, Root};
use super::search::SearchResult::*;
use crate::vec::Vec;
use olive_core::alloc::{AllocError, Allocator, AllocatorTryClone};

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

    /// Splits off a tree with key-value pairs at and after the given key.
    /// The result is meaningful only if the tree is ordered by key,
    /// and if the ordering of `Q` corresponds to that of `K`.
    /// If `self` respects all `BTreeMap` tree invariants, then both
    /// `self` and the returned tree will respect those invariants.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the right tree's pillar of internal
    /// nodes fails. In that case `self` is left completely unmodified.
    ///
    /// # Panics
    ///
    /// May panic if the user's `Ord::cmp` implementation panics (e.g. due to
    /// arithmetic overflow). In that case `self` is left completely unmodified,
    /// because the comparison phase is read-only and no mutations have occurred yet.
    pub(super) fn try_split_off<Q: ?Sized + Ord, A: Allocator>(
        &mut self,
        key: &Q,
        alloc: &A,
    ) -> Result<Self, AllocError>
    where
        K: Borrow<Q>,
    {
        // Maximum tree height for B=6: even 2^64 entries give height ≤ 17.
        // 32 is a generous upper bound that fits comfortably on the stack.
        const MAX_DEPTH: usize = 32;
        let mut path: [usize; MAX_DEPTH] = [0; MAX_DEPTH];
        let mut depth: usize = 0;

        // ── Phase 1: read-only descent ──────────────────────────────────────
        // Walk down the left tree recording the edge index at each level.
        // No mutations occur here, so a panic in `Ord::cmp` leaves `self` intact.
        {
            let mut node = self.reborrow();
            loop {
                let edge_idx = match node.search_node(key) {
                    Found(kv) => kv.idx(),
                    GoDown(edge) => edge.idx(),
                };
                debug_assert!(depth < MAX_DEPTH);
                path[depth] = edge_idx;
                #[allow(clippy::arithmetic_side_effects, reason = "depth + 1 <= MAX_DEPTH")]
                {
                    depth += 1;
                }

                if node.height() == 0 {
                    break;
                }
                // Descend to the child at the recorded edge index.
                let internal = unsafe { node.cast_to_internal_unchecked() };
                node = unsafe { Handle::new_edge(internal, edge_idx) }.descend();
            }
        }

        // ── Phase 2: allocate + mutate ──────────────────────────────────────
        // All comparisons are done; this phase performs only pointer chases,
        // mem-moves, and length writes, so it cannot panic.
        let mut right_root = Root::new_pillar(self.height(), &alloc)?;
        let mut left_node = self.borrow_mut();
        let mut right_node = right_root.borrow_mut();

        // Process all levels except the last (leaf) level.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "depth is nonzero - a leaf analysis already increments depth"
        )]
        for &idx in &path[..depth - 1] {
            let mut split_edge = unsafe { Handle::new_edge(left_node, idx) };
            split_edge.move_suffix(&mut right_node);

            match (split_edge.force(), right_node.force()) {
                (Internal(edge), Internal(rnode)) => {
                    left_node = edge.descend();
                    right_node = rnode.first_edge().descend();
                }
                _ => unreachable!(),
            }
        }

        // Final (leaf) level.
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "depth is nonzero - a leaf analysis already increments depth"
        )]
        {
            let idx = path[depth - 1];
            let mut split_edge = unsafe { Handle::new_edge(left_node, idx) };
            split_edge.move_suffix(&mut right_node);
        }

        // Deallocation may not panic.
        self.fix_right_border(&alloc);
        right_root.fix_left_border(&alloc);
        Ok(right_root)
    }

    /// Creates a tree consisting of empty nodes.
    fn new_pillar<A: Allocator>(height: usize, alloc: &A) -> Result<Self, AllocError> {
        let new_count = height;
        let mut ephemeral_stack =
            Vec::try_with_capacity_in(new_count, alloc).map_err(|_| AllocError)?;
        // Allocate additional nodes if we need more.
        for _ in 0..new_count {
            let intermediate = unsafe { InternalNode::new(alloc)? };
            ephemeral_stack
                .try_push(intermediate)
                .expect("we just reserved enough items");
        }
        // Allocate the root node here.
        let mut root = Root::new(alloc)?;
        for _ in 0..height {
            root.push_internal_level(
                ephemeral_stack
                    .pop()
                    .expect("internal node should be available"),
            );
        }
        debug_assert!(ephemeral_stack.is_empty());
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
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if allocating the right tree's internal nodes fails.
    /// In that case `self` is left completely unmodified.
    ///
    /// # Panics
    ///
    /// May panic if the user's `Ord::cmp` implementation panics (e.g. due to
    /// bugs in a custom key type). In that case `self` is left completely unmodified.
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
        let right_root = left_root.try_split_off(key, &alloc)?;

        let (new_left_len, right_len) = Root::calc_split_length(total_num, left_root, &right_root);
        self.length = new_left_len;

        Ok(BTreeMap {
            root: Some(right_root),
            length: right_len,
            alloc: ManuallyDrop::new(alloc),
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::super::invariant::{check_ascending_keys, check_tree_invariant};
    use super::super::map::BTreeMap;
    use crate::alloc::Global;
    use crate::test_helpers::{BudgetedAlloc, Ledger, TrackedItem};

    /// Runs all structural invariant checks on both maps after a split.
    fn verify_both<K, V>(left: &BTreeMap<K, V>, right: &BTreeMap<K, V>)
    where
        K: Ord + core::fmt::Debug,
    {
        check_tree_invariant(left);
        check_tree_invariant(right);
        check_ascending_keys(left);
        check_ascending_keys(right);
    }

    /// Basic correctness: split at an existing key keeps that key in the right map.
    #[test]
    fn split_at_existing_key() {
        let mut a = BTreeMap::new_in(Global);
        for i in 1..=5 {
            a.try_insert(i, i * 10).unwrap();
        }
        let b = a.split_off(&3).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 3);
        assert_eq!(a.get(&1), Some(&10));
        assert_eq!(a.get(&2), Some(&20));
        assert_eq!(b.get(&3), Some(&30));
        assert_eq!(b.get(&4), Some(&40));
        assert_eq!(b.get(&5), Some(&50));
        verify_both(&a, &b);
    }

    /// Split at a non-existent key splits before the next greater key.
    #[test]
    fn split_at_missing_key() {
        let mut a = BTreeMap::new_in(Global);
        // Keys have gaps: 1, 3, 7, 9. Splitting at 5 lands between 3 and 7.
        for i in [1, 3, 7, 9] {
            a.try_insert(i, i * 10).unwrap();
        }
        let b = a.split_off(&5).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        let mut it_a = a.iter();
        assert_eq!(it_a.next(), Some((&1, &10)));
        assert_eq!(it_a.next(), Some((&3, &30)));
        assert_eq!(it_a.next(), None);
        let mut it_b = b.iter();
        assert_eq!(it_b.next(), Some((&7, &70)));
        assert_eq!(it_b.next(), Some((&9, &90)));
        assert_eq!(it_b.next(), None);
        verify_both(&a, &b);
    }

    /// Splitting off everything leaves `self` empty and returns all entries.
    #[test]
    fn split_all_to_right() {
        let mut a = BTreeMap::new_in(Global);
        for i in 0..10 {
            a.try_insert(i, i).unwrap();
        }
        let b = a.split_off(&(-1i32)).unwrap();
        assert!(a.is_empty());
        assert_eq!(b.len(), 10);
        verify_both(&a, &b);
    }

    /// Splitting with a key past the end yields an empty right map.
    #[test]
    fn split_all_to_left() {
        let mut a = BTreeMap::new_in(Global);
        for i in 0..10 {
            a.try_insert(i, i).unwrap();
        }
        let b = a.split_off(&999).unwrap();
        assert_eq!(a.len(), 10);
        assert!(b.is_empty());
        verify_both(&a, &b);
    }

    /// Empty map split returns an empty map.
    #[test]
    fn split_empty_map() {
        let mut a: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let b = a.split_off(&0).unwrap();
        assert!(a.is_empty());
        assert!(b.is_empty());
        verify_both(&a, &b);
    }

    /// Multi-level tree (forces internal nodes) splits correctly.
    #[test]
    fn split_multi_level_tree() {
        let mut a = BTreeMap::new_in(Global);
        // Enough entries to guarantee at least one level of internal nodes.
        for i in 0..60 {
            a.try_insert(i, i * 2).unwrap();
        }
        let b = a.split_off(&30).unwrap();
        assert_eq!(a.len(), 30);
        assert_eq!(b.len(), 30);
        // Verify boundaries via iteration.
        let a_keys: std::vec::Vec<i32> = a.iter().map(|(k, _)| *k).collect();
        let b_keys: std::vec::Vec<i32> = b.iter().map(|(k, _)| *k).collect();
        assert_eq!(a_keys, (0..30).collect::<std::vec::Vec<_>>());
        assert_eq!(b_keys, (30..60).collect::<std::vec::Vec<_>>());
        // Spot-check a few values on each side.
        assert_eq!(a.get(&0), Some(&0));
        assert_eq!(a.get(&29), Some(&58));
        assert_eq!(b.get(&30), Some(&60));
        assert_eq!(b.get(&59), Some(&118));
        verify_both(&a, &b);
    }

    /// Allocation failure during `new_pillar` leaves `self` completely
    /// unmodified and leaks nothing.
    #[test]
    fn split_alloc_failure_leaves_self_intact_and_no_leak() {
        let alloc = BudgetedAlloc::new(1 << 20);
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut a: BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc> =
            BTreeMap::new_in(alloc.clone());
        // Build a multi-level tree so `new_pillar` actually has to allocate
        // several internal nodes.
        for i in 0..60u32 {
            insert_tracked_split_pair(&mut a, i, i * 100, &ledger);
        }
        let snapshot: std::vec::Vec<(u32, u32)> = a.iter().map(|(k, v)| (**k, **v)).collect();
        assert_eq!(snapshot.len(), 60);

        // Set the budget so `new_pillar`'s second allocation fails.
        alloc.set_budget(1);

        let result = a.split_off(&30u32);
        assert!(result.is_err(), "expected allocation failure");

        // `self` must be structurally valid and byte-for-byte identical.
        check_tree_invariant(&a);
        check_ascending_keys(&a);
        let after: std::vec::Vec<(u32, u32)> = a.iter().map(|(k, v)| (**k, **v)).collect();
        assert_eq!(
            after, snapshot,
            "self must be unmodified after failed split"
        );
        assert_eq!(a.len(), 60);

        // Drop `a`; every registered id must die exactly once — no leak from
        // the partially-built right tree, no double-drop.
        drop(a);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
    }

    fn insert_tracked_split_pair(
        map: &mut BTreeMap<TrackedItem<u32>, TrackedItem<u32>, BudgetedAlloc>,
        key_val: u32,
        val_val: u32,
        ledger: &std::sync::Arc<Ledger>,
    ) {
        let kid = ledger.allocate();
        ledger.register(kid);
        let vid = ledger.allocate();
        ledger.register(vid);
        map.try_insert(
            TrackedItem {
                id: kid,
                ledger: ledger.clone(),
                inner: key_val,
            },
            TrackedItem {
                id: vid,
                ledger: ledger.clone(),
                inner: val_val,
            },
        )
        .unwrap();
    }
}
