use super::borrow::DormantMutRef;
use super::node::marker;
use super::node::{NodeRef, Root};
use crate::{alloc::Global, collections::btree::node::Handle};
use core::fmt;
use core::iter::FusedIterator;
use olive_core::alloc::AllocatorTryClone;

/// An iterator produced by calling `extract_if` on BTreeMap.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct ExtractIf<'a, K, V, F, A: AllocatorTryClone = Global>
where
    F: 'a + FnMut(&K, &mut V) -> bool,
{
    pred: F,
    inner: ExtractIfInner<'a, K, V>,
    alloc: &'a A,
}

/// Most of the implementation of ExtractIf are generic over the type
/// of the predicate, thus also serving for BTreeSet::ExtractIf.
pub(super) struct ExtractIfInner<'a, K, V> {
    /// Reference to the length field in the borrowed map, updated live.
    length: &'a mut usize,
    /// Buried reference to the root field in the borrowed map.
    /// Wrapped in `Option` to allow drop handler to `take` it.
    dormant_root: Option<DormantMutRef<'a, Root<K, V>>>,
    /// Contains a leaf edge preceding the next element to be returned, or the last leaf edge.
    /// Empty if the map has no root, if iteration went beyond the last leaf edge,
    /// or if a panic occurred in the predicate.
    cur_leaf_edge: Option<Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>>,
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

impl<K, V, F, A: AllocatorTryClone> Iterator for ExtractIf<'_, K, V, F, A>
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
    pub(super) fn next<F, A: AllocatorTryClone>(
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
