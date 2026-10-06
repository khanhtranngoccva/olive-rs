//! Iterator adaptors layered on top of [`core::iter`].
//!
//! This module mirrors the layout of [`core::iter`] and re-exports its entire
//! surface, then adds Olive's own adaptor types on top.

use core::fmt::{Debug, Formatter};
pub use core::iter::*;
use olive_macros::TryClone;

/// A two-way analogue of [`core::iter::Peekable`]: peek at the next element
/// from either end without consuming it.
///
/// Exposes [`peek_front`](Self::peek_front) / [`peek_back`](Self::peek_back)
/// (and their `_mut` variants) alongside normal iteration.
///
/// Requires the wrapped iterator to be [`DoubleEndedIterator`] + [`FusedIterator`].
///
/// [`FusedIterator`] allows correct behavior: once the iterator returns `None`
/// it stays `None`, so when the main iterator yields `None` and there is an
/// element on the opposite side, that element is semantically guaranteed to be
/// the final element. It is not correct behavior if the "back" element is popped by front methods,
/// and the main iterator could resurrect, because that element may no longer
/// be the back. The same applies in the reverse direction.
///
/// ```
/// use olive_core::iter::{DoubleEndedPeekable, DoubleEndedPeekableExt};
///
/// let mut it = [10, 20, 30].into_iter().double_ended_peekable();
/// assert_eq!(it.peek_front(), Some(&10));
/// assert_eq!(it.peek_back(), Some(&30));
/// assert_eq!(it.next(), Some(10));
/// assert_eq!(it.next_back(), Some(30));
/// assert_eq!(it.peek_front(), Some(&20));
/// assert_eq!(it.peek_back(), Some(&20));
/// assert_eq!(it.next(), Some(20));
/// assert_eq!(it.peek_front(), None);
/// assert_eq!(it.peek_back(), None);
/// ```
#[must_use = "adaptors do nothing unless used"]
#[derive(TryClone)]
pub struct DoubleEndedPeekable<I: DoubleEndedIterator + FusedIterator> {
    /// The primary iterator.
    iter: I,
    /// Element cached by [`peek_front`](Self::peek_front), if any.
    front: Option<I::Item>,
    /// Element cached by [`peek_back`](Self::peek_back), if any.
    back: Option<I::Item>,
}

impl<I: DoubleEndedIterator + FusedIterator> Clone for DoubleEndedPeekable<I>
where
    I: Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
        }
    }
}

impl<I: DoubleEndedIterator + FusedIterator> Debug for DoubleEndedPeekable<I>
where
    I: Debug,
    I::Item: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DoubleEndedPeekable")
            .field("iter", &self.iter)
            .field("front", &self.front)
            .field("back", &self.back)
            .finish()
    }
}

impl<I: DoubleEndedIterator + FusedIterator> DoubleEndedPeekable<I> {
    /// Creates a new [`DoubleEndedPeekable`] from any fusable double-ended
    /// iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            front: None,
            back: None,
        }
    }

    /// Borrows the inner iterator.
    #[inline]
    pub fn get_ref(&self) -> &I {
        &self.iter
    }

    /// Mutably borrows the inner iterator.
    #[inline]
    pub fn get_mut(&mut self) -> &mut I {
        &mut self.iter
    }

    /// Consumes this adapter, returning the inner iterator and any buffered
    /// elements so that none are lost.
    ///
    /// The returned tuple is `(inner, front, back)`. Each buffer
    /// is `Some(_)` only if a peek was outstanding when the adapter was
    /// consumed.
    #[inline]
    pub fn into_inner(self) -> (I, Option<I::Item>, Option<I::Item>) {
        (self.iter, self.front, self.back)
    }

    /// Returns a reference to the element the next [`next`](Iterator::next)
    /// would yield, or `None` if the front is exhausted.
    #[inline]
    pub fn peek_front(&mut self) -> Option<&I::Item> {
        if self.front.is_some() {
            return self.front.as_ref();
        }
        // Probe the inner iterator from the front.
        match self.iter.next() {
            Some(item) => {
                self.front = Some(item);
                self.front.as_ref()
            }
            None => {
                // Front direction is exhausted (fused: permanently). Fall
                // through to the back buffer, reading without filling.
                self.back.as_ref()
            }
        }
    }

    /// Like [`peek_front`](Self::peek_front) but returns `&mut I::Item`.
    #[inline]
    pub fn peek_front_mut(&mut self) -> Option<&mut I::Item> {
        if self.front.is_some() {
            return self.front.as_mut();
        }
        match self.iter.next() {
            Some(item) => {
                self.front = Some(item);
                self.front.as_mut()
            }
            None => self.back.as_mut(),
        }
    }

    /// Returns a reference to the element the next
    /// [`next_back`](DoubleEndedIterator::next_back) would yield, or `None` if
    /// the back is exhausted.
    #[inline]
    pub fn peek_back(&mut self) -> Option<&I::Item> {
        if self.back.is_some() {
            return self.back.as_ref();
        }
        // Probe the inner iterator from the back.
        match self.iter.next_back() {
            Some(item) => {
                self.back = Some(item);
                self.back.as_ref()
            }
            None => {
                // Back direction is exhausted (fused: permanently). Fall
                // through to the front buffer, reading without filling.
                self.front.as_ref()
            }
        }
    }

    /// Like [`peek_back`](Self::peek_back) but returns `&mut I::Item`.
    #[inline]
    pub fn peek_back_mut(&mut self) -> Option<&mut I::Item> {
        if self.back.is_some() {
            return self.back.as_mut();
        }
        match self.iter.next_back() {
            Some(item) => {
                self.back = Some(item);
                self.back.as_mut()
            }
            None => self.front.as_mut(),
        }
    }
}

impl<I: DoubleEndedIterator + FusedIterator> Iterator for DoubleEndedPeekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Pop the front buffer if present.
        if let Some(item) = self.front.take() {
            return Some(item);
        }
        // Ask the inner iterator from the front.
        match self.iter.next() {
            Some(item) => Some(item),
            None => {
                // Front direction is exhausted (fused: permanently).
                // Fall through to the back buffer.
                self.back.take()
            }
        }
    }

    #[inline]
    fn count(self) -> usize {
        // Drain both buffers plus the remainder of the inner iterator.
        #[allow(clippy::arithmetic_side_effects, reason = "n_cached is at most 2")]
        let n_cached = usize::from(self.front.is_some()) + usize::from(self.back.is_some());
        n_cached
            .checked_add(self.iter.count())
            .expect("underlying iterator of DoubleEndedPeekable yielded too many elements")
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        // Seed with whichever end is already buffered, then fold over the
        // inner iterator.
        let mut best = self.front.take();
        if let Some(b) = self.back.take() {
            best = Some(match best {
                Some(a) => core::cmp::min(a, b),
                None => b,
            });
        }
        for item in self.iter.by_ref() {
            best = Some(match best {
                Some(a) => core::cmp::min(a, item),
                None => item,
            });
        }
        best
    }

    #[inline]
    fn max(mut self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        let mut best = self.front.take();
        if let Some(b) = self.back.take() {
            best = Some(match best {
                Some(a) => core::cmp::max(a, b),
                None => b,
            });
        }
        for item in self.iter.by_ref() {
            best = Some(match best {
                Some(a) => core::cmp::max(a, item),
                None => item,
            });
        }
        best
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The size hint is total of elements in the main iterator plus 0-2 elements in
        // the front and back.
        // Drain both buffers plus the remainder of the inner iterator.
        #[allow(clippy::arithmetic_side_effects, reason = "n_cached is at most 2")]
        let n_cached = usize::from(self.front.is_some()) + usize::from(self.back.is_some());
        // In case the iterator lies about its bounds, the saturating amd checked adds do the job.
        let (lower, upper) = self.iter.size_hint();
        (
            lower.saturating_add(n_cached),
            upper.and_then(|u| u.checked_add(n_cached)),
        )
    }
}

impl<I: DoubleEndedIterator + FusedIterator> DoubleEndedIterator for DoubleEndedPeekable<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // Pop the back buffer if present.
        if let Some(item) = self.back.take() {
            return Some(item);
        }
        // Ask the inner iterator from the back.
        match self.iter.next_back() {
            Some(item) => Some(item),
            None => {
                // Back direction is exhausted (fused: permanently).
                //    Fall through to the front buffer.
                self.front.take()
            }
        }
    }
}

impl<I: DoubleEndedIterator + FusedIterator> FusedIterator for DoubleEndedPeekable<I> {}

impl<I: DoubleEndedIterator + FusedIterator + ExactSizeIterator> ExactSizeIterator
    for DoubleEndedPeekable<I>
{
    fn len(&self) -> usize {
        // The size hint is total of elements in the main iterator plus 0-2 elements in
        // the front and back.
        // Drain both buffers plus the remainder of the inner iterator.
        #[allow(clippy::arithmetic_side_effects, reason = "n_cached is at most 2")]
        let n_cached = usize::from(self.front.is_some()) + usize::from(self.back.is_some());
        self.iter
            .len()
            .checked_add(n_cached)
            .expect("inner iterator does not adhere to ExactSizeIterator contract")
    }
}

/// Convenience constructor mirroring [`core::iter::Peekable::peekable`].
pub trait DoubleEndedPeekableExt: DoubleEndedIterator + FusedIterator {
    /// Wraps `self` in a [`DoubleEndedPeekable`], enabling
    /// [`peek_front`](DoubleEndedPeekable::peek_front) and
    /// [`peek_back`](DoubleEndedPeekable::peek_back), so that
    /// the iterator can be peeked both ways.
    fn double_ended_peekable(self) -> DoubleEndedPeekable<Self>
    where
        Self: Sized,
    {
        DoubleEndedPeekable::new(self)
    }
}

impl<I: DoubleEndedIterator + FusedIterator + ?Sized> DoubleEndedPeekableExt for I {}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn basic_two_way_peek() {
        let mut it = [10, 20, 30].into_iter().double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&10));
        assert_eq!(it.peek_back(), Some(&30));
        assert_eq!(it.next(), Some(10));
        assert_eq!(it.next_back(), Some(30));
        assert_eq!(it.peek_front(), Some(&20));
        assert_eq!(it.peek_back(), Some(&20));
        assert_eq!(it.next(), Some(20));
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn repeated_peeks_are_stable() {
        let mut it = [1, 2, 3, 4].into_iter().double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&1));
        assert_eq!(it.peek_front(), Some(&1));
        assert_eq!(it.peek_back(), Some(&4));
        assert_eq!(it.peek_back(), Some(&4));
        // Neither peek consumed anything.
        assert_eq!((&mut it).count(), 4);
    }

    #[test]
    fn drain_from_front_only() {
        let mut it = (1..=5).double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&1));
        let v: std::vec::Vec<i32> = (&mut it).collect();
        assert_eq!(v, std::vec![1, 2, 3, 4, 5]);
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn drain_from_back_only() {
        let mut it = (1..=5).double_ended_peekable();
        assert_eq!(it.peek_back(), Some(&5));
        let mut v = std::vec::Vec::new();
        while let Some(x) = it.next_back() {
            v.push(x);
        }
        assert_eq!(v, std::vec![5, 4, 3, 2, 1]);
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn alternating_ends_meet_in_middle() {
        let mut it = (1..=6).double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&1));
        assert_eq!(it.peek_back(), Some(&6));
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next_back(), Some(6));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next_back(), Some(5));
        assert_eq!(it.peek_front(), Some(&3));
        assert_eq!(it.peek_back(), Some(&4));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.peek_back(), None);
    }

    #[test]
    fn single_element_shared_by_both_eyes_and_front_invalidates_back_eye() {
        let mut it = [7].into_iter().double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&7));
        // Back eye sees the same lone element too.
        assert_eq!(it.peek_back(), Some(&7));
        // Front consumes it; the back eye must now see exhaustion.
        assert_eq!(it.next(), Some(7));
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn single_element_consumed_from_back_invalidates_front_eye() {
        let mut it = [7].into_iter().double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&7));
        assert_eq!(it.peek_back(), Some(&7));
        // Back consumes it; the front eye must now see exhaustion.
        assert_eq!(it.next_back(), Some(7));
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn empty_iterator_is_fused_at_both_ends() {
        let mut it: DoubleEndedPeekable<Empty<i32>> = empty().double_ended_peekable();
        assert_eq!(it.peek_front(), None);
        assert_eq!(it.peek_back(), None);
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn into_inner_preserves_buffers() {
        let mut it = [1, 2, 3, 4].into_iter().double_ended_peekable();
        let _ = it.peek_front();
        let _ = it.peek_back();
        let (inner, front, back) = it.into_inner();
        assert_eq!(front, Some(1));
        assert_eq!(back, Some(4));
        assert_eq!(inner.collect::<std::vec::Vec<_>>(), std::vec![2, 3]);
    }

    #[test]
    fn get_ref_and_get_mut_forward() {
        let mut it = (1..=3).double_ended_peekable();
        assert_eq!(
            it.get_ref().clone().collect::<std::vec::Vec<_>>(),
            std::vec![1, 2, 3]
        );
        *it.get_mut() = 5..=7;
        assert_eq!(it.next(), Some(5));
    }

    #[test]
    fn try_clone_carries_state() {
        let mut it = (1..=4).double_ended_peekable();
        let _ = it.peek_front();
        let mut cloned = it.try_clone().expect("infallible clone");
        // Both views share the same logical position.
        assert_eq!(cloned.peek_front(), Some(&1));
        assert_eq!(it.peek_back(), Some(&4));
        assert_eq!(cloned.next(), Some(1));
        assert_eq!(it.next(), Some(1));
    }

    #[test]
    fn debug_impl_mentions_buffers() {
        let mut it = (1..=3).double_ended_peekable();
        let _ = it.peek_front();
        let s = std::format!("{it:?}");
        assert!(s.contains("DoubleEndedPeekable"));
        assert!(s.contains("Some(1)"), "debug output was: {s}");
    }

    #[test]
    fn size_hint_adds_cached_elements_to_inner_hint() {
        let mut it = (1..=5).double_ended_peekable();
        // Exact hint from RangeInclusive, nothing cached yet.
        assert_eq!(it.size_hint(), (5, Some(5)));
        let _ = it.peek_front();
        // Inner now reports 4 remaining; +1 cached front = 5 total.
        assert_eq!(it.size_hint(), (5, Some(5)));
        let _ = it.peek_back();
        // Inner now reports 3 remaining between the eyes; +2 cached = 5 total.
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.next(), Some(1));
        // Front buffer drained; inner still reports 3; +1 cached back = 4.
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.next_back(), Some(5));
        // Both buffers drained; hint reflects only what remains in the middle.
        assert_eq!(it.size_hint(), (3, Some(3)));
    }

    #[test]
    fn works_through_generic_paths() {
        // Slice iterators are fused double-ended; the adaptor composes.
        let data = [9, 8, 7];
        let mut it = data.iter().copied().double_ended_peekable();
        assert_eq!(it.peek_front(), Some(&9));
        assert_eq!(it.peek_back(), Some(&7));
        assert_eq!((&mut it).map(|x| x * 2).sum::<i32>(), 9 * 2 + 8 * 2 + 7 * 2);
    }

    #[test]
    fn peek_front_mut_returns_mutable_reference() {
        let mut it = [1, 2, 3].into_iter().double_ended_peekable();
        // Peek and mutate the front element.
        if let Some(front) = it.peek_front_mut() {
            *front += 100;
        }
        // The mutation should be visible when we consume.
        assert_eq!(it.next(), Some(101));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(3));
    }

    #[test]
    fn peek_back_mut_returns_mutable_reference() {
        let mut it = [1, 2, 3].into_iter().double_ended_peekable();
        // Peek and mutate the back element.
        if let Some(back) = it.peek_back_mut() {
            *back += 100;
        }
        // The mutation should be visible when we consume from the back.
        assert_eq!(it.next_back(), Some(103));
        assert_eq!(it.next_back(), Some(2));
        assert_eq!(it.next_back(), Some(1));
    }

    #[test]
    fn peek_mut_front_visible_in_back() {
        let mut it = [42].into_iter().double_ended_peekable();
        // Peek from back, exhausting the iterator and leaving one item at the back.
        assert_eq!(it.peek_back(), Some(&42));
        // Now peek_front_mut should read the back buffer (which holds 42).
        if let Some(v) = it.peek_front_mut() {
            *v += 1;
        }
        // Consume and verify the mutation.
        assert_eq!(it.next(), Some(43));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn peek_mut_back_visible_in_front() {
        let mut it = [7].into_iter().double_ended_peekable();
        // Both eyes see the same element. The item is stored at the front.
        assert_eq!(it.peek_front(), Some(&7));
        assert_eq!(it.peek_back(), Some(&7));
        // Mutate via peek_back_mut.
        if let Some(v) = it.peek_back_mut() {
            *v *= 2;
        }
        // The mutation should be visible from the front too.
        assert_eq!(it.peek_front(), Some(&14));
        assert_eq!(it.next(), Some(14));
    }
    #[test]
    fn count_drains_both_buffers_and_inner() {
        let mut it = (1..=5).double_ended_peekable();
        let _ = it.peek_front();
        let _ = it.peek_back();
        // Two buffered + three inner = five total. count consumes self.
        assert_eq!(it.count(), 5);
    }

    #[test]
    fn last_prefers_back_buffer_then_inner_tail() {
        let it = (1..=5).double_ended_peekable();
        // No peek yet: last() must reach into the inner tail.
        assert_eq!(it.last(), Some(5));
    }

    #[test]
    fn last_with_back_buffer_returns_buffered_value() {
        let mut it = (1..=5).double_ended_peekable();
        let _ = it.peek_back();
        // Back buffer holds 5; last() should return it without touching inner.
        assert_eq!(it.last(), Some(5));
    }

    #[test]
    fn last_with_only_front_buffer_falls_through_to_front() {
        let mut it = [42].into_iter().double_ended_peekable();
        let _ = it.peek_front();
        // Inner is empty; front buffer holds the sole element.
        assert_eq!(it.last(), Some(42));
    }

    #[test]
    fn min_max_fold_over_all_three_sources() {
        let mut a = [3, 1, 4, 1, 5, 9, 2, 6].into_iter().double_ended_peekable();
        let _ = a.peek_front();
        let _ = a.peek_back();
        assert_eq!(a.min(), Some(1));

        let mut c = [3, 1, 4, 1, 5, 9, 2, 6].into_iter().double_ended_peekable();
        let _ = c.peek_front();
        let _ = c.peek_back();
        assert_eq!(c.max(), Some(9));

        // min() and max() should work even if the elements are separated from the main iterator
        let mut b = [1, 4, 5, 9, 2, 6].into_iter().double_ended_peekable();
        let _ = b.peek_front();
        let _ = b.peek_back();
        assert_eq!(b.min(), Some(1));

        let mut d = [1, 4, 5, 2, 6, 9].into_iter().double_ended_peekable();
        let _ = d.peek_front();
        let _ = d.peek_back();
        assert_eq!(d.max(), Some(9));
    }

    #[test]
    fn min_max_last_empty_are_none() {
        let a: DoubleEndedPeekable<Empty<i32>> = empty().double_ended_peekable();
        let b: DoubleEndedPeekable<Empty<i32>> = empty().double_ended_peekable();
        let c: DoubleEndedPeekable<Empty<i32>> = empty().double_ended_peekable();
        assert_eq!(a.min(), None);
        assert_eq!(b.max(), None);
        assert_eq!(c.last(), None);
    }
}
