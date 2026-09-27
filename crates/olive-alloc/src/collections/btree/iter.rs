//! Iterators for BTreeMap.

use olive_core::try_traits::{TryClone, TryCloneError};

use super::map::BTreeMap;
use super::navigate::LazyLeafRange;
use super::node::marker;
use super::node::{Handle, NodeRef};
use crate::alloc::{AllocatorTryClone, Global};
use core::iter::{DoubleEndedIterator, FusedIterator, Iterator};
use core::mem::ManuallyDrop;

// ── Iter ─────────────────────────────────────────────────────────────────────

/// An iterator yielding immutable key-value references over the entries of a `BTreeMap`.
///
/// Returned by [`BTreeMap::iter`].
pub struct Iter<'a, K: 'a, V: 'a> {
    range: LazyLeafRange<marker::Immut<'a>, K, V>,
    length: usize,
}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            // SAFETY: a non-zero length guarantees at least one KV remains.
            Some(unsafe { self.range.next_unchecked() })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    fn last(mut self) -> Option<(&'a K, &'a V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a V)>
    where
        (&'a K, &'a V): Ord,
    {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            // SAFETY: a non-zero length guarantees at least one KV remains.
            Some(unsafe { self.range.next_back_unchecked() })
        }
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a, K: 'a, V: 'a> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Self {
        Iter {
            range: self.range.clone(),
            length: self.length,
        }
    }
}

impl<'a, K: 'a, V: 'a> TryClone for Iter<'a, K, V> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning is trivial.
        Ok(self.clone())
    }
}

// ── Keys ─────────────────────────────────────────────────────────────────────

/// An iterator yielding immutable key references over the entries of a `BTreeMap`.
///
/// Returned by [`BTreeMap::keys`].
pub struct Keys<'a, K: 'a, V: 'a> {
    iter: Iter<'a, K, V>,
}

impl<'a, K: 'a, V: 'a> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<&'a K> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a K>
    where
        &'a K: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a K>
    where
        &'a K: Ord,
    {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for Keys<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<K, V> FusedIterator for Keys<'_, K, V> {}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: 'a, V: 'a> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Self {
        Keys {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K: 'a, V: 'a> TryClone for Keys<'a, K, V> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning is trivial.
        Ok(self.clone())
    }
}

// ── Values ───────────────────────────────────────────────────────────────────

/// An iterator yielding immutable value references over the entries of a `BTreeMap`.
///
/// Returned by [`BTreeMap::values`].
pub struct Values<'a, K: 'a, V: 'a> {
    iter: Iter<'a, K, V>,
}

impl<'a, K: 'a, V: 'a> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<&'a V> {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for Values<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, v)| v)
    }
}

impl<K, V> FusedIterator for Values<'_, K, V> {}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: 'a, V: 'a> Clone for Values<'a, K, V> {
    fn clone(&self) -> Self {
        Values {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K: 'a, V: 'a> TryClone for Values<'a, K, V> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning is trivial.
        Ok(self.clone())
    }
}

// ── IterMut ──────────────────────────────────────────────────────────────────

/// An iterator over mutable references to the values of a `BTreeMap`.
///
/// Returned by [`BTreeMap::iter_mut`].
pub struct IterMut<'a, K: 'a, V: 'a> {
    range: LazyLeafRange<marker::ValMut<'a>, K, V>,
    length: usize,
}

impl<'a, K: 'a, V: 'a> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            // SAFETY: a non-zero length guarantees at least one KV remains.
            Some(unsafe { self.range.next_unchecked() })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    fn last(mut self) -> Option<(&'a K, &'a mut V)> {
        self.next_back()
    }

    fn min(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<(&'a K, &'a mut V)>
    where
        (&'a K, &'a mut V): Ord,
    {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for IterMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            // SAFETY: a non-zero length guarantees at least one KV remains.
            Some(unsafe { self.range.next_back_unchecked() })
        }
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.length
    }
}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

// ── ValuesMut ────────────────────────────────────────────────────────────────

/// An iterator yielding mutable value references over the entries of a `BTreeMap`.
///
/// Returned by [`BTreeMap::values_mut`].
pub struct ValuesMut<'a, K: 'a, V: 'a> {
    iter: IterMut<'a, K, V>,
}

impl<'a, K: 'a, V: 'a> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<&'a mut V> {
        self.next_back()
    }
}

impl<K, V> DoubleEndedIterator for ValuesMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, v)| v)
    }
}

impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

// ── IntoIter ─────────────────────────────────────────────────────────────────

/// An owning iterator over the entries of a `BTreeMap`, consuming the map.
///
/// Returned by iterating over a `BTreeMap` directly ([`IntoIterator`]) or calling
/// [`IntoIterator::into_iter`].
/// Nodes are deallocated as they are visited.
pub struct IntoIter<K, V, A: AllocatorTryClone = Global> {
    range: LazyLeafRange<marker::Dying, K, V>,
    length: usize,
    alloc: A,
}

impl<K, V, A: AllocatorTryClone> Iterator for IntoIter<K, V, A> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.dying_next()
            .map(|kv_handle| unsafe { kv_handle.into_key_val() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<K, V, A: AllocatorTryClone> DoubleEndedIterator for IntoIter<K, V, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.dying_next_back()
            .map(|kv_handle| unsafe { kv_handle.into_key_val() })
    }
}

impl<K, V, A: AllocatorTryClone> ExactSizeIterator for IntoIter<K, V, A> {
    fn len(&self) -> usize {
        self.length
    }
}

impl<K, V, A: AllocatorTryClone> Drop for IntoIter<K, V, A> {
    fn drop(&mut self) {
        // Exhaust the remaining entries, dropping each key/value in place and deallocating
        // every node as we climb back up to the root. The walk is panic-free — `dying_next`
        // is guarded by the `length` counter and `deallocating_end` only fires once it hits
        // zero — so no unwind guard is needed. Unlike std's `BTreeMap` drop, we never read the
        // reserve stack here: it is provably empty (see `move_fields_to_iterator`).
        while let Some(kv) = self.dying_next() {
            // SAFETY: we consume the dying handle immediately and don't touch the tree first.
            unsafe { kv.drop_key_val() };
        }
    }
}

impl<K, V, A: AllocatorTryClone> FusedIterator for IntoIter<K, V, A> {}

impl<K, V, A: AllocatorTryClone> IntoIter<K, V, A> {
    /// Core of a `next` method returning a dying KV handle,
    /// invalidated by further calls to this function and some others.
    fn dying_next(
        &mut self,
    ) -> Option<Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>> {
        if self.length == 0 {
            self.range.deallocating_end(&self.alloc);
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            Some(unsafe { self.range.deallocating_next_unchecked(&self.alloc) })
        }
    }

    /// Core of a `next_back` method returning a dying KV handle,
    /// invalidated by further calls to this function and some others.
    fn dying_next_back(
        &mut self,
    ) -> Option<Handle<NodeRef<marker::Dying, K, V, marker::LeafOrInternal>, marker::KV>> {
        if self.length == 0 {
            self.range.deallocating_end(&self.alloc);
            None
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "asserted self.length > 0 above"
            )]
            {
                self.length -= 1;
            }
            Some(unsafe { self.range.deallocating_next_back_unchecked(&self.alloc) })
        }
    }
}

// ── IntoKeys ─────────────────────────────────────────────────────────────────

/// An owning iterator over the keys of a `BTreeMap`, consuming the map.
///
/// Returned by [`BTreeMap::into_keys`].
pub struct IntoKeys<K, V, A: AllocatorTryClone = Global> {
    iter: IntoIter<K, V, A>,
}

impl<K, V, A: AllocatorTryClone> Iterator for IntoKeys<K, V, A> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<K, V, A: AllocatorTryClone> DoubleEndedIterator for IntoKeys<K, V, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<K, V, A: AllocatorTryClone> ExactSizeIterator for IntoKeys<K, V, A> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V, A: AllocatorTryClone> FusedIterator for IntoKeys<K, V, A> {}

// ── IntoValues ───────────────────────────────────────────────────────────────

/// An owning iterator over the values of a `BTreeMap`, consuming the map.
///
/// Returned by [`BTreeMap::into_values`].
pub struct IntoValues<K, V, A: AllocatorTryClone = Global> {
    iter: IntoIter<K, V, A>,
}

impl<K, V, A: AllocatorTryClone> Iterator for IntoValues<K, V, A> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn last(mut self) -> Option<V> {
        self.next_back()
    }
}

impl<K, V, A: AllocatorTryClone> DoubleEndedIterator for IntoValues<K, V, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, v)| v)
    }
}

impl<K, V, A: AllocatorTryClone> ExactSizeIterator for IntoValues<K, V, A> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V, A: AllocatorTryClone> FusedIterator for IntoValues<K, V, A> {}

// ── BTreeMap methods ─────────────────────────────────────────────────────────

impl<K, V, A: AllocatorTryClone> BTreeMap<K, V, A> {
    /// Returns an iterator over the key-value pairs in ascending key order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        let range = match &self.root {
            Some(root) => root.reborrow().full_range(),
            None => LazyLeafRange::none(),
        };
        Iter {
            range,
            length: self.length,
        }
    }

    /// Returns an iterator over immutable key references in ascending key order.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator over immutable value references in ascending key order.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { iter: self.iter() }
    }

    /// Returns an iterator over mutable value references in ascending key order.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let range = match &mut self.root {
            Some(root) => root.borrow_valmut().full_range(),
            None => LazyLeafRange::none(),
        };
        IterMut {
            range,
            length: self.length,
        }
    }

    /// Returns an iterator over mutable value references in ascending key order.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }

    /// Consumes the map and returns an iterator over its keys in ascending key order.
    pub fn into_keys(self) -> IntoKeys<K, V, A> {
        IntoKeys {
            iter: self.into_iter(),
        }
    }

    /// Consumes the map and returns an iterator over its values in ascending key order.
    pub fn into_values(self) -> IntoValues<K, V, A> {
        IntoValues {
            iter: self.into_iter(),
        }
    }
}

impl<K, V, A: AllocatorTryClone> IntoIterator for BTreeMap<K, V, A> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, A>;

    fn into_iter(self) -> Self::IntoIter {
        let mut me = ManuallyDrop::new(self);
        let _reserve_stack = unsafe { ManuallyDrop::take(&mut me.reserve_stack) };
        if let Some(root) = me.root.take() {
            let full_range = root.into_dying().full_range();
            IntoIter {
                range: full_range,
                length: me.length,
                alloc: unsafe { ManuallyDrop::take(&mut me.alloc) },
            }
        } else {
            IntoIter {
                range: LazyLeafRange::none(),
                length: 0,
                alloc: unsafe { ManuallyDrop::take(&mut me.alloc) },
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::alloc::Global;
    use crate::test_helpers::{Ledger, TrackedItem};
    use std::sync::Arc;

    /// A simple comparable wrapper around a fixed-size array, used in place of
    /// `String` (which is fallible in this crate) to exercise non-Copy keys.
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key([u8; 4]);
    impl From<&str> for Key {
        fn from(s: &str) -> Self {
            let mut buf = [0u8; 4];
            for (i, b) in s.bytes().take(4).enumerate() {
                buf[i] = b;
            }
            Key(buf)
        }
    }

    // ── Iter tests ────────────────────────────────────────────────────────────

    #[test]
    fn iter_empty_map() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        assert_eq!(map.iter().next(), None);
        assert_eq!(map.iter().next_back(), None);
        assert_eq!(map.iter().count(), 0);
    }

    #[test]
    fn iter_single_element() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let mut it = map.iter();
        assert_eq!(it.next(), Some((&1, &10)));
        assert_eq!(it.next(), None);
        let mut it = map.iter();
        assert_eq!(it.next_back(), Some((&1, &10)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn iter_forward_order_small() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.iter();
        for i in 0..5i32 {
            assert_eq!(it.next(), Some((&i, &(i * 10))), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_reverse_order_small() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.iter();
        for i in (0..5i32).rev() {
            assert_eq!(it.next_back(), Some((&i, &(i * 10))), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn iter_multilevel_forward() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        let mut it = map.iter();
        for i in 0..80u32 {
            assert_eq!(it.next(), Some((&i, &(i * 2))), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn iter_multilevel_reverse() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        let mut it = map.iter();
        for i in (0..80u32).rev() {
            assert_eq!(it.next_back(), Some((&i, &(i * 2))), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn iter_interleaved_forward_backward() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..30u32 {
            map.try_insert(i, i).unwrap();
        }
        let mut it = map.iter();
        let mut fwd_count = 0usize;
        let mut bwd_count = 0usize;
        let mut prev_fwd: Option<u32> = None;
        let mut prev_bwd: Option<u32> = None;
        // Alternate: 2 forward, 2 backward, until exhausted.
        'control_loop: loop {
            for _ in 0..2 {
                match it.next() {
                    Some((k, _)) => {
                        if let Some(p) = prev_fwd {
                            assert!(p < *k, "forward order violated: {} !< {}", p, k);
                        }
                        prev_fwd = Some(*k);
                        fwd_count += 1;
                    }
                    None => break 'control_loop,
                }
            }
            match it.next_back() {
                Some((k, _)) => {
                    if let Some(p) = prev_bwd {
                        assert!(p > *k, "backward order violated: {} !> {}", p, k);
                    }
                    prev_bwd = Some(*k);
                    bwd_count += 1;
                }
                None => break 'control_loop,
            }
        }
        assert_eq!(fwd_count + bwd_count, 30, "total visited != map size");
    }

    #[test]
    fn iter_clone_independent() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i).unwrap();
        }
        let mut it1 = map.iter();
        let mut it2 = it1.clone();
        // Advance it1 by 3.
        for _ in 0..3 {
            it1.next();
        }
        // it2 should still start from the beginning.
        assert_eq!(it2.next().map(|(k, _)| *k), Some(0));
        // it1 should continue from where it left off.
        assert_eq!(it1.next().map(|(k, _)| *k), Some(3));
    }

    #[test]
    fn iter_fused() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let mut it = map.iter();
        assert_eq!(it.next(), Some((&1, &10)));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None, "fused: repeated next after exhaustion");
    }

    // ── IterMut tests ─────────────────────────────────────────────────────────

    #[test]
    fn iter_mut_modify_values() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        for (_, v) in map.iter_mut() {
            *v *= 2;
        }
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&(i * 20)), "key {}", i);
        }
    }

    #[test]
    fn iter_mut_reverse_modify() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i).unwrap();
        }
        for (_, v) in map.iter_mut().rev() {
            *v += 100;
        }
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&(i + 100)), "key {}", i);
        }
    }

    #[test]
    fn iter_mut_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i).unwrap();
        }
        for (_, v) in map.iter_mut() {
            *v += 100;
        }
        for i in 0..80u32 {
            assert_eq!(map.get(&i), Some(&(i + 100)), "key {}", i);
        }
    }

    #[test]
    fn iter_mut_multilevel_reverse() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i).unwrap();
        }
        for (_, v) in map.iter_mut().rev() {
            *v += 100;
        }
        for i in 0..80u32 {
            assert_eq!(map.get(&i), Some(&(i + 100)), "key {}", i);
        }
    }

    // ── IntoIter tests ────────────────────────────────────────────────────────

    #[test]
    fn into_iter_empty() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        let mut it = map.into_iter();
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
        assert_eq!(it.count(), 0);
    }

    #[test]
    fn into_iter_single() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(42, 420).unwrap();
        let mut it = map.into_iter();
        assert_eq!(it.next(), Some((42, 420)));
        assert_eq!(it.next(), None);

        let mut map = BTreeMap::new_in(Global);
        map.try_insert(42, 420).unwrap();
        let mut it = map.into_iter();
        assert_eq!(it.next_back(), Some((42, 420)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn into_iter_forward_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_iter();
        for i in 0..20i32 {
            assert_eq!(it.next(), Some((i, i * 10)), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_iter_reverse_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_iter();
        for i in (0..20i32).rev() {
            assert_eq!(it.next_back(), Some((i, i * 10)), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn into_iter_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..100u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        let mut it = map.into_iter();
        for i in 0..100u32 {
            assert_eq!(it.next(), Some((i, i * 3)), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    /// Helper to insert a tracked key + tracked value pair into the map.
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

    #[test]
    fn into_iter_partial_then_drop() {
        // Take a few elements, then drop the iterator — Drop should clean up
        // remaining KVs and nodes without leaking or double-freeing.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        let mut it = map.into_iter();
        assert_eq!(it.next().map(|(k, v)| (k.inner, v.inner)), Some((0, 0)));
        assert_eq!(it.next().map(|(k, v)| (k.inner, v.inner)), Some((1, 10)));
        assert_eq!(it.next().map(|(k, v)| (k.inner, v.inner)), Some((2, 20)));
        // Drop the iterator here — remaining 47 KV pairs should be cleaned up.
        drop(it);
        // All 100 items (50 keys + 50 values) must have been dropped exactly once.
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
        assert_eq!(ledger.total_allocated(), 100);
    }

    #[test]
    fn into_iter_partial_rev_then_drop() {
        // Same but advancing from the back.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..50u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        let mut it = map.into_iter();
        assert_eq!(
            it.next_back().map(|(k, v)| (k.inner, v.inner)),
            Some((49, 490))
        );
        assert_eq!(
            it.next_back().map(|(k, v)| (k.inner, v.inner)),
            Some((48, 480))
        );
        drop(it);
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
        assert_eq!(ledger.total_allocated(), 100);
    }

    #[test]
    fn into_iter_non_copy_keys() {
        // Exercise non-Copy key types to catch any assumption about Copy.
        let mut map: BTreeMap<Key, u32> = BTreeMap::new_in(Global);
        map.try_insert(Key::from("banana"), 1).unwrap();
        map.try_insert(Key::from("apple"), 2).unwrap();
        map.try_insert(Key::from("cherry"), 3).unwrap();
        let mut it = map.into_iter();
        assert_eq!(it.next(), Some((Key::from("apple"), 2)));
        assert_eq!(it.next(), Some((Key::from("banana"), 1)));
        assert_eq!(it.next(), Some((Key::from("cherry"), 3)));
        assert_eq!(it.next(), None);
    }

    // ── Keys tests ────────────────────────────────────────────────────────────

    #[test]
    fn keys_empty_map() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        assert_eq!(map.keys().next(), None);
        assert_eq!(map.keys().next_back(), None);
        assert_eq!(map.keys().count(), 0);
    }

    #[test]
    fn keys_forward_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.keys();
        for i in 0..5i32 {
            assert_eq!(it.next(), Some(&i), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn keys_reverse_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.keys();
        for i in (0..5i32).rev() {
            assert_eq!(it.next_back(), Some(&i), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn keys_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        let mut it = map.keys();
        for i in 0..80u32 {
            assert_eq!(it.next(), Some(&i), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn keys_exact_size_and_clone() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i).unwrap();
        }
        let it = map.keys();
        assert_eq!(it.len(), 10);
        let mut it1 = it.clone();
        let mut it2 = it;
        // Advance it1 by 3.
        for _ in 0..3 {
            it1.next();
        }
        assert_eq!(it1.len(), 7);
        assert_eq!(it2.len(), 10);
        assert_eq!(it2.next(), Some(&0));
        assert_eq!(it1.next(), Some(&3));
    }

    #[test]
    fn keys_fused() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let mut it = map.keys();
        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None, "fused: repeated next after exhaustion");
    }

    // ── Values tests ──────────────────────────────────────────────────────────

    #[test]
    fn values_empty_map() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        assert_eq!(map.values().next(), None);
        assert_eq!(map.values().next_back(), None);
        assert_eq!(map.values().count(), 0);
    }

    #[test]
    fn values_forward_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.values();
        for i in 0..5i32 {
            assert_eq!(it.next(), Some(&(i * 10)), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn values_reverse_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.values();
        for i in (0..5i32).rev() {
            assert_eq!(it.next_back(), Some(&(i * 10)), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn values_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..80u32 {
            map.try_insert(i, i * 2).unwrap();
        }
        let mut it = map.values();
        for i in 0..80u32 {
            assert_eq!(it.next(), Some(&(i * 2)), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn values_exact_size_and_clone() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i * 7).unwrap();
        }
        let it = map.values();
        assert_eq!(it.len(), 10);
        let mut it1 = it.clone();
        let mut it2 = it;
        for _ in 0..3 {
            it1.next();
        }
        assert_eq!(it1.len(), 7);
        assert_eq!(it2.len(), 10);
        assert_eq!(it2.next(), Some(&0));
        assert_eq!(it1.next(), Some(&(3 * 7)));
    }

    #[test]
    fn values_fused() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let mut it = map.values();
        assert_eq!(it.next(), Some(&10));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None, "fused: repeated next after exhaustion");
    }

    // ── ValuesMut tests ───────────────────────────────────────────────────────

    #[test]
    fn values_mut_modify_values() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i * 10).unwrap();
        }
        for v in map.values_mut() {
            *v *= 2;
        }
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&(i * 20)), "key {}", i);
        }
    }

    #[test]
    fn values_mut_reverse_modify() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..5 {
            map.try_insert(i, i).unwrap();
        }
        for v in map.values_mut().rev() {
            *v += 100;
        }
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&(i + 100)), "key {}", i);
        }
    }

    #[test]
    fn values_mut_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            map.try_insert(i, i).unwrap();
        }
        for v in map.values_mut() {
            *v += 100;
        }
        for i in 0..200u32 {
            assert_eq!(map.get(&i), Some(&(i + 100)), "key {}", i);
        }
    }

    #[test]
    fn values_mut_exact_size() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..10 {
            map.try_insert(i, i).unwrap();
        }
        let mut it = map.values_mut();
        assert_eq!(it.len(), 10);
        it.next();
        assert_eq!(it.len(), 9);
        it.next_back();
        assert_eq!(it.len(), 8);
    }

    #[test]
    fn values_mut_fused() {
        let mut map = BTreeMap::new_in(Global);
        map.try_insert(1, 10).unwrap();
        let mut it = map.values_mut();
        assert_eq!(*it.next().unwrap(), 10);
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None, "fused: repeated next after exhaustion");
    }

    // ── IntoKeys tests ────────────────────────────────────────────────────────

    #[test]
    fn into_keys_empty() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        let mut it = map.into_keys();
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
        assert_eq!(it.count(), 0);
    }

    #[test]
    fn into_keys_forward_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_keys();
        for i in 0..20i32 {
            assert_eq!(it.next(), Some(i), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_keys_reverse_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_keys();
        for i in (0..20i32).rev() {
            assert_eq!(it.next_back(), Some(i), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn into_keys_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        let mut it = map.into_keys();
        for i in 0..200u32 {
            assert_eq!(it.next(), Some(i), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_keys_partial_then_drop() {
        // Take a few keys, then drop the iterator — Drop should clean up
        // remaining KVs and nodes without leaking or double-freeing.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        let mut it = map.into_keys();
        assert_eq!(it.next().map(|k| k.inner), Some(0));
        assert_eq!(it.next().map(|k| k.inner), Some(1));
        // Drop the iterator here — remaining KV pairs should be cleaned up.
        drop(it);
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
        assert_eq!(ledger.total_allocated(), 400);
    }

    #[test]
    fn into_keys_non_copy_keys() {
        let mut map: BTreeMap<Key, u32> = BTreeMap::new_in(Global);
        map.try_insert(Key::from("banana"), 1).unwrap();
        map.try_insert(Key::from("apple"), 2).unwrap();
        map.try_insert(Key::from("cherry"), 3).unwrap();
        let mut it = map.into_keys();
        assert_eq!(it.next(), Some(Key::from("apple")));
        assert_eq!(it.next(), Some(Key::from("banana")));
        assert_eq!(it.next(), Some(Key::from("cherry")));
        assert_eq!(it.next(), None);
    }

    // ── IntoValues tests ──────────────────────────────────────────────────────

    #[test]
    fn into_values_empty() {
        let map = BTreeMap::<i32, i32>::new_in(Global);
        let mut it = map.into_values();
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
        assert_eq!(it.count(), 0);
    }

    #[test]
    fn into_values_forward_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_values();
        for i in 0..20i32 {
            assert_eq!(it.next(), Some(i * 10), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_values_reverse_order() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..20 {
            map.try_insert(i, i * 10).unwrap();
        }
        let mut it = map.into_values();
        for i in (0..20i32).rev() {
            assert_eq!(it.next_back(), Some(i * 10), "position {}", i);
        }
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn into_values_multilevel() {
        let mut map = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            map.try_insert(i, i * 3).unwrap();
        }
        let mut it = map.into_values();
        for i in 0..200u32 {
            assert_eq!(it.next(), Some(i * 3), "position {}", i);
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_values_partial_then_drop() {
        // Take a few values, then drop the iterator — Drop should clean up
        // remaining KVs and nodes without leaking or double-freeing.
        let ledger = Arc::new(Ledger::new());
        let mut map: BTreeMap<TrackedItem<u32>, TrackedItem<u32>> = BTreeMap::new_in(Global);
        for i in 0..200u32 {
            insert_tracked_pair(&mut map, i, i * 10, &ledger);
        }
        let mut it = map.into_values();
        assert_eq!(it.next().map(|v| v.inner), Some(0));
        assert_eq!(it.next().map(|v| v.inner), Some(10));
        // Drop the iterator here — remaining KV pairs should be cleaned up.
        drop(it);
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
        assert_eq!(ledger.total_allocated(), 400);
    }
}
