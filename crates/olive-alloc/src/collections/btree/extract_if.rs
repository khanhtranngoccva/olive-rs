use super::borrow::DormantMutRef;
use super::node::marker;
use super::node::{NodeRef, Root};
use crate::{alloc::Global, collections::btree::node::Handle};
use core::fmt;
use core::iter::FusedIterator;
use olive_core::alloc::Allocator;

/// An iterator produced by calling `extract_if` on BTreeMap.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, K, V, F, A: Allocator = Global>
where
    F: 'a + FnMut(&K, &mut V) -> bool,
{
    pub(super) pred: F,
    pub(super) inner: ExtractIfInner<'a, K, V>,
    pub(super) alloc: &'a A,
}

/// Most of the implementation of ExtractIf are generic over the type
/// of the predicate, thus also serving for BTreeSet::ExtractIf.
pub(super) struct ExtractIfInner<'a, K, V> {
    /// Reference to the length field in the borrowed map, updated live.
    pub(super) length: &'a mut usize,
    /// Buried reference to the root field in the borrowed map.
    /// Wrapped in `Option` to allow drop handler to `take` it.
    pub(super) dormant_root: Option<DormantMutRef<'a, Root<K, V>>>,
    /// Contains a leaf edge preceding the next element to be returned, or the last leaf edge.
    /// Empty if the map has no root, if iteration went beyond the last leaf edge,
    /// or if a panic occurred in the predicate.
    pub(super) cur_leaf_edge:
        Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
}

impl<K, V, F> fmt::Debug for ExtractIf<'_, K, V, F>
where
    K: fmt::Debug,
    V: fmt::Debug,
    F: FnMut(&K, &mut V) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExtractIf")
            .field(&self.inner.peek())
            .finish()
    }
}

impl<K, V, F, A: Allocator> Iterator for ExtractIf<'_, K, V, F, A>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next(&mut self.pred, &self.alloc)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExtractIfInner<'_, K, V> {
    /// Allow Debug implementations to predict the next element.
    pub(super) fn peek(&self) -> Option<(&K, &V)> {
        let edge = self.cur_leaf_edge.as_ref()?;
        edge.reborrow().next_kv().ok().map(Handle::into_kv)
    }

    /// Implementation of a typical `ExtractIf::next` method, given the predicate.
    pub(super) fn next<F, A: Allocator>(
        &mut self,
        pred: &mut F,
        alloc: &A,
    ) -> Option<(K, V)>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        while let Ok(mut kv) = self.cur_leaf_edge.take()?.next_kv() {
            let (k, v) = kv.kv_mut();
            if pred(k, v) {
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "kv available implies positive length"
                )]
                {
                    *self.length -= 1;
                }
                let (kv, pos) = kv.remove_kv_tracking(
                    || {
                        // SAFETY: we will touch the root in a way that will not
                        // invalidate the position returned.
                        let root = unsafe { self.dormant_root.take().unwrap().awaken() };
                        root.pop_internal_level(alloc);
                        self.dormant_root = Some(DormantMutRef::new(root).1);
                    },
                    alloc,
                );
                self.cur_leaf_edge = Some(pos);
                return Some(kv);
            }
            self.cur_leaf_edge = Some(kv.next_leaf_edge());
        }
        None
    }

    /// Implementation of a typical `ExtractIf::size_hint` method.
    pub(super) fn size_hint(&self) -> (usize, Option<usize>) {
        // In most of the btree iterators, `self.length` is the number of elements
        // yet to be visited. Here, it includes elements that were visited and that
        // the predicate decided not to drain. Making this upper bound more tight
        // during iteration would require an extra field.
        (0, Some(*self.length))
    }
}

impl<K, V, F> FusedIterator for ExtractIf<'_, K, V, F> where F: FnMut(&K, &mut V) -> bool {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::super::invariant::{check_ascending_keys, check_tree_invariant};
    use super::super::map::BTreeMap;
    use std::sync::Arc;
    use std::vec::Vec;

    use crate::alloc::Global;
    use crate::test_helpers::{Ledger, TestRng, TrackedItem};

    /// Inserts a tracked key + tracked value pair into the map, registering both
    /// ids with the ledger up front so drops are observable.
    fn insert_tracked_pair(
        map: &mut BTreeMap<TrackedItem<u32>, TrackedItem<u32>>,
        key_val: u32,
        val_val: u32,
        ledger: &Arc<Ledger>,
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

    // ── Basic behavior ─────────────────────────────────────────────────────────

    #[test]
    fn extract_if_empty_map() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        let extracted: Vec<(i32, i32)> = map.extract_if(|_, _| true).collect();
        assert!(extracted.is_empty());
        assert!(map.is_empty());
    }

    #[test]
    fn extract_if_no_match() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        let extracted: Vec<(u32, u32)> = map.extract_if(|&k, _| k > 100).collect();
        assert!(extracted.is_empty());
        // Nothing was touched: all entries survive intact.
        assert_eq!(map.len(), 10);
        for i in 0..10u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn extract_if_all_match() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        let extracted: Vec<(u32, u32)> = map.extract_if(|&k, _| k < 100).collect();
        assert_eq!(extracted.len(), 10);
        for (i, (k, v)) in extracted.iter().enumerate() {
            assert_eq!(*k, i as u32);
            assert_eq!(*v, i as u32 * 10);
        }
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn extract_if_partial_matches() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        // Evens out.
        let extracted: Vec<(u32, u32)> = map.extract_if(|&k, _| k % 2 == 0).collect();
        assert_eq!(
            extracted,
            [(0, 0), (2, 20), (4, 40), (6, 60), (8, 80)].to_vec(),
            "extraction must yield matching keys in ascending order"
        );
        // Odds remain, untouched and still ordered.
        assert_eq!(map.len(), 5);
        for i in (0..10u32).filter(|i| i % 2 == 1) {
            assert_eq!(map.get(&i), Some(&(i * 10)), "survivor {i} damaged");
        }
        for i in (0..10u32).filter(|i| i % 2 == 0) {
            assert_eq!(map.get(&i), None, "extracted {i} still present");
        }
    }

    #[test]
    fn extract_if_predicate_sees_live_value_and_can_mutate_it() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        // The predicate receives `&mut V`: mutate the value before deciding.
        // Keys divisible by 2 get their value doubled, then are extracted.
        // Keys 0,2,4 are even → extracted with doubled values; 1,3 survive.
        let extracted: Vec<(u32, u32)> = map
            .extract_if(|&k, v| {
                if k % 2 == 0 {
                    *v *= 2;
                    true
                } else {
                    false
                }
            })
            .collect();
        assert_eq!(extracted, [(0, 0), (2, 40), (4, 80)].to_vec());
        // Survivors were never visited or mutated.
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn extract_if_single_element() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(7, 70).unwrap();
        let extracted: Vec<(i32, i32)> = map.extract_if(|&k, _| k == 7).collect();
        assert_eq!(extracted, [(7, 70)].to_vec());
        assert!(map.is_empty());
    }

    // ── Iterator semantics ─────────────────────────────────────────────────────

    #[test]
    fn extract_if_size_hint_bounds() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20u32 {
            map.try_insert(i, i).unwrap();
        }
        let mut iter = map.extract_if(|&k, _| k % 2 == 0);
        // Lower bound is always 0; upper bound starts at the map length.
        assert_eq!(iter.size_hint(), (0, Some(20)));
        // Consume two matches (keys 0 and 2): the upper bound tracks the live
        // length, which has dropped to 18.
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.size_hint(), (0, Some(18)));
        drop(iter);
        // The map itself reflects the live length the hint was tracking.
        assert_eq!(map.len(), 18);
    }

    #[test]
    fn extract_if_size_hint_exhausted() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i).unwrap();
        }
        // After a full drain the hint collapses to (0, Some(0)); on an empty
        // map it starts there.
        let collected: Vec<(u32, u32)> = map.extract_if(|_, _| true).collect();
        assert_eq!(collected.len(), 10);
        let mut empty_iter = map.extract_if(|_, _| true);
        assert_eq!(empty_iter.size_hint(), (0, Some(0)));
        assert_eq!(empty_iter.next(), None);
    }

    #[test]
    fn extract_if_fused_iterator() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5u32 {
            map.try_insert(i, i).unwrap();
        }
        let mut iter = map.extract_if(|_, _| true);
        assert_eq!(iter.next(), Some((0, 0)));
        // Exhaust.
        for _ in iter.by_ref() {}
        // A fused iterator keeps returning None once exhausted.
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn extract_if_debug_format() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        map.try_insert(2, 20).unwrap();
        let mut iter = map.extract_if(|_, _| true);
        // Before any next(): the peek shows the first pending element.
        assert_eq!(std::format!("{:?}", iter), "ExtractIf(Some((1, 10)))");
        assert_eq!(iter.next(), Some((1, 10)));
        assert_eq!(std::format!("{:?}", iter), "ExtractIf(Some((2, 20)))");
        assert_eq!(iter.next(), Some((2, 20)));
        assert_eq!(std::format!("{:?}", iter), "ExtractIf(None)");
    }

    #[test]
    fn extract_if_lazy() {
        // Nothing happens until the iterator is driven: abandoning it after a
        // single step leaves the remainder in the map.
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut iter = map.extract_if(|_, _| true);
        assert_eq!(iter.next(), Some((0, 0)));
        drop(iter);
        // Abandoning the iterator mid-way leaves the remainder in the map.
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&0), None);
        for i in 1..5u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    // ── retain ─────────────────────────────────────────────────────────────────

    #[test]
    fn retain_empty_map() {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new_in(Global);
        map.retain(|_, _| true);
        assert!(map.is_empty());
    }

    #[test]
    fn retain_keeps_matching_only() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        map.retain(|&k, _| k % 2 == 1);
        assert_eq!(map.len(), 5);
        for i in (0..10u32).filter(|i| i % 2 == 1) {
            assert_eq!(map.get(&i), Some(&(i * 10)), "kept {i} damaged");
        }
        for i in (0..10u32).filter(|i| i % 2 == 0) {
            assert_eq!(map.get(&i), None, "filtered-out {i} still present");
        }
    }

    #[test]
    fn retain_keeps_none() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20u32 {
            map.try_insert(i, i).unwrap();
        }
        map.retain(|_, _| false);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn retain_keeps_all() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        map.retain(|_, _| true);
        assert_eq!(map.len(), 20);
        for i in 0..20u32 {
            assert_eq!(map.get(&i), Some(&(i * 3)));
        }
    }

    #[test]
    fn retain_predicate_receives_mutable_value() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..4u32 {
            map.try_insert(i, i * 10).unwrap();
        }
        // Mutate every visited value, keeping only the even-keyed ones.
        map.retain(|&k, v| {
            *v += 1;
            k % 2 == 0
        });
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&0), Some(&1));
        assert_eq!(map.get(&2), Some(&21));
    }

    #[test]
    fn retain_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..300u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        assert!(map.root.as_ref().is_some_and(|r| r.height() >= 2));
        map.retain(|&k, _| k % 3 != 0);
        check_tree_invariant(&map);
        check_ascending_keys(&map);
        assert_eq!(map.len(), 200);
        for i in 0..300u32 {
            if i % 3 == 0 {
                assert_eq!(map.get(&i), None, "{i} should be retained out");
            } else {
                assert_eq!(map.get(&i), Some(&(i * 2)), "survivor {i} damaged");
            }
        }
    }

    // ── Memory safety: every extracted payload dies exactly once ───────────────

    #[test]
    fn extract_if_drops_extracted_pairs_once_no_leak() {
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        // Extract evens among 0..40: 20 pairs, 40 payloads (ids 0..80).
        let mut extracted: Vec<u32> = Vec::new();
        for (k, v) in map.extract_if(|k, _| k.inner % 2 == 0 && k.inner < 40) {
            assert_eq!(v.inner, k.inner * 10);
            extracted.push(k.inner);
        }
        assert_eq!(extracted.len(), 20);
        assert_eq!(map.len(), 30);
        // Each extracted key/value died exactly once; nothing double-freed.
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        let counts = ledger.drop_counts();
        assert_eq!(counts.values().sum::<usize>(), 40);
        assert!(
            counts.values().all(|&c| c == 1),
            "some payload dropped != once: {counts:?}"
        );
        // Survivors are still live: 30 keys/values, none of them dead yet.
        assert_eq!(
            ledger.live_ids().len(),
            60,
            "unexpected live ids: {:?}",
            ledger.live_ids()
        );
        // Dropping the map frees the rest; everything dies exactly once.
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..100));
    }

    #[test]
    fn extract_if_full_drain_ledger_clean() {
        // Extracting everything must leave no payload alive and no double-free.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..60u32 {
            insert_tracked_pair(&mut map, i, i * 2, &ledger);
        }
        let collected: Vec<u32> = map.extract_if(|_, _| true).map(|(k, _)| k.inner).collect();
        assert_eq!(collected.len(), 60);
        assert!(map.is_empty());
        assert!(
            ledger.live_ids().is_empty(),
            "survivors still live after full drain: {:?}",
            ledger.live_ids()
        );
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
        assert!(ledger.all_dropped_once(0..120));
    }

    #[test]
    fn extract_if_abandoned_mid_iteration_survivors_stay_live() {
        // Stop iterating early: extracted pairs die, survivors stay live until
        // the map itself drops — proving partial consumption is memory-clean.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..30u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        let mut iter = map.extract_if(|_, _| true);
        // Take exactly three pairs (6 payloads).
        for i in 0..3u32 {
            let (k, _) = iter.next().expect("expected a pair");
            assert_eq!(k.inner, i);
        }
        drop(iter);
        assert_eq!(map.len(), 27);
        assert!(
            ledger.double_dropped().is_empty(),
            "double-dropped: {:?}",
            ledger.double_dropped()
        );
        assert_eq!(
            ledger.drop_counts().values().sum::<usize>(),
            6,
            "only the 3 taken pairs should be dead"
        );
        assert_eq!(
            ledger.live_ids().len(),
            54,
            "unexpected live ids: {:?}",
            ledger.live_ids()
        );
        drop(map);
        assert!(
            ledger.leaked_ids().is_empty(),
            "leaked ids: {:?}",
            ledger.leaked_ids()
        );
        assert!(ledger.all_dropped_once(0..60));
    }

    // ── Structural invariants across multi-level trees ─────────────────────────

    /// Partial extraction from a multi-level tree: the structural invariant
    /// must hold while the tree is being pruned, and afterwards.
    #[test]
    fn extract_if_multilevel_preserves_invariant() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 200;
        for i in 0..N {
            map.try_insert(i, i * 2).unwrap();
        }
        assert!(
            map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        // Pull out every third key — interleaved across all levels.
        let extracted: Vec<u32> = map.extract_if(|&k, _| k % 3 == 0).map(|(k, _)| k).collect();
        assert_eq!(extracted.len(), 67);
        check_tree_invariant(&map);
        check_ascending_keys(&map);
        assert_eq!(map.len(), N as usize - extracted.len());
        for i in 0..N {
            if i % 3 == 0 {
                assert_eq!(map.get(&i), None, "{i} should be extracted");
            } else {
                assert_eq!(map.get(&i), Some(&(i * 2)), "survivor {i} damaged");
            }
        }
    }

    /// Drains a multi-level tree through extract_if one element at a time,
    /// verifying exact values per step and the structural invariant afterwards.
    /// (Mid-drain invariant checks aren't possible while the iterator holds its
    /// mutable borrow.)
    #[test]
    fn extract_if_drain_one_by_one_preserves_invariant() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 200;
        for i in 0..N {
            map.try_insert(i, i).unwrap();
        }
        assert!(
            map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        let drained: Vec<(u32, u32)> = map.extract_if(|_, _| true).collect();
        assert_eq!(drained.len(), N as usize);
        for (i, (k, v)) in drained.iter().enumerate() {
            assert_eq!(*k, i as u32, "wrong key at step {i}");
            assert_eq!(*v, i as u32, "wrong value at step {i}");
        }
        assert!(map.is_empty());
        check_tree_invariant(&map);
        check_ascending_keys(&map);
    }

    /// Repeated partial drains over shrinking trees: each round extracts half of
    /// what remains, checking the structural invariant between rounds — the
    /// closest analogue to per-extraction invariant checks that the borrow rules
    /// allow.
    #[test]
    fn extract_if_repeated_partial_drains_preserve_invariant() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 200;
        for i in 0..N {
            map.try_insert(i, i * 2).unwrap();
        }
        assert!(
            map.root.as_ref().is_some_and(|r| r.height() >= 2),
            "expected multi-level tree"
        );
        let mut remaining = N as usize;
        while !map.is_empty() {
            if remaining == 1 {
                // Single element left: extract it outright to terminate.
                let n_extracted = map.extract_if(|_, _| true).count();
                assert_eq!(n_extracted, 1);
                remaining -= n_extracted;
                check_tree_invariant(&map);
                continue;
            }
            // Extract roughly the lower half of whatever remains. Use a strict
            // inequality on a midpoint so we never sweep out the entire set.
            let first = *map.first_key_value().expect("non-empty").0;
            let last = *map.last_key_value().expect("non-empty").0;
            let cutoff = first + (last - first) / 2;
            let n_extracted = map.extract_if(|&k, _| k <= cutoff).count();
            assert!(n_extracted > 0 && n_extracted < remaining, "bad split");
            remaining -= n_extracted;
            check_tree_invariant(&map);
            check_ascending_keys(&map);
            assert_eq!(map.len(), remaining);
        }
        check_tree_invariant(&map);
    }

    /// Randomized (deterministic PRNG) partial extractions over many rounds:
    /// each round extracts a random subset of the remaining keys, verifying the
    /// invariant and exact survivor set afterward.
    #[test]
    fn extract_if_random_subsets_preserve_invariant() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 150;
        for i in 0..N {
            map.try_insert(i, i * 7).unwrap();
        }
        let mut rng = TestRng::new(0xE7CABEEFCAFED00D);
        let mut expected: std::collections::BTreeSet<u32> = (0..N).collect();
        for round in 0..10u32 {
            // A deterministic hash-derived predicate computed outside the
            // closure so it can be re-checked after extraction.
            let threshold = (rng.next_u64() % 97) as u32;
            let seed = rng.next_u64();
            let seed32 = seed as u32;
            let pred = |k: u32| -> bool {
                let h = k
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(seed32)
                    .wrapping_mul(0xBF58_476D);
                (h >> 16) % 100 < threshold
            };
            let extracted: Vec<u32> = map.extract_if(|&k, _| pred(k)).map(|(k, _)| k).collect();
            for k in &extracted {
                assert!(pred(*k), "extracted {k} does not match its own predicate");
                assert!(expected.remove(k), "extracted {k} not expected");
            }
            check_tree_invariant(&map);
            check_ascending_keys(&map);
            assert_eq!(map.len(), expected.len(), "round {round}: length mismatch");
            for &k in &expected {
                assert_eq!(map.get(&k), Some(&(k * 7)), "survivor {k} damaged");
            }
            if expected.is_empty() {
                break;
            }
        }
        check_tree_invariant(&map);
    }

    /// Extraction followed by re-insertion of new values must not disturb the
    /// tree: the classic churn pattern that stresses rebalancing.
    #[test]
    fn extract_if_then_reinsert_churn() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i).unwrap();
        }
        for round in 0..5u32 {
            // Extract the lower half of whatever remains, then re-insert shifted keys.
            let midpoint = map.last_key_value().map_or(0, |(k, _)| *k / 2);
            let n_extracted = map.extract_if(|&k, _| k < midpoint).count();
            check_tree_invariant(&map);
            check_ascending_keys(&map);
            for j in 0..n_extracted as u32 {
                map.try_insert(midpoint + 1 + j * 3, round).unwrap();
            }
            check_tree_invariant(&map);
            check_ascending_keys(&map);
        }
        // Whatever the mix, the tree is structurally sound and sorted.
        check_tree_invariant(&map);
        check_ascending_keys(&map);
        assert!(!map.is_empty());
    }

    #[test]
    fn retain_scrambled_schedule_preserves_invariant() {
        let mut map = BTreeMap::new_in(Global);
        const N: u32 = 120;
        for i in 0..N {
            map.try_insert(i, i).unwrap();
        }
        let rng = TestRng::new(0xCAFE_F00D_DEAD_BEED);
        let keep_set: std::collections::BTreeSet<u32> =
            rng.permuted(0..N).take((N / 3) as usize).collect();
        map.retain(|&k, _| keep_set.contains(&k));
        check_tree_invariant(&map);
        check_ascending_keys(&map);
        assert_eq!(map.len(), keep_set.len());
        for &k in &keep_set {
            assert_eq!(map.get(&k), Some(&k), "keeper {k} lost");
        }
        for i in 0..N {
            if !keep_set.contains(&i) {
                assert_eq!(map.get(&i), None, "{i} should be pruned");
            }
        }
    }
}
