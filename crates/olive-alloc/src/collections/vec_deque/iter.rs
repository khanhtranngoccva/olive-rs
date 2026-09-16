//! Front-to-back iterators over a [`VecDeque`].
//!
//! A deque's elements may occupy two disjoint regions of its circular buffer,
//! so both `Iter` and `IterMut` chain two slice iterators — one for each region.
//! This mirrors the std reference implementation.

use core::fmt;
use core::mem;
use core::ops::RangeBounds;
use core::slice;

use super::VecDeque;
use olive_core::alloc::Allocator;
use olive_core::prelude::{TryClone, TryDefault};
use olive_core::slice::TrySliceRangeError;
use olive_core::try_traits::try_clone::TryCloneError;
use olive_core::try_traits::try_default::TryDefaultError;

/// An iterator over the elements of a `VecDeque`.
///
/// This struct is created by the [`iter`](VecDeque::iter) method on `VecDeque`.
pub struct Iter<'a, T: 'a> {
    i1: slice::Iter<'a, T>,
    i2: slice::Iter<'a, T>,
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter")
            .field(&self.i1.as_slice())
            .field(&self.i2.as_slice())
            .finish()
    }
}

impl<T> Default for Iter<'_, T> {
    /// Creates an empty `vec_deque::Iter`.
    fn default() -> Self {
        Iter {
            i1: Default::default(),
            i2: Default::default(),
        }
    }
}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter {
            i1: self.i1.clone(),
            i2: self.i2.clone(),
        }
    }
}

// Cloning a slice iterator only copies two pointer-and-length pairs — no
// allocation is involved, so the fallible form can never fail.
impl<T> TryClone for Iter<'_, T> {
    #[inline]
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(self.clone())
    }
}

// An empty iterator holds no storage, so its default construction is infallible.
impl<T> TryDefault for Iter<'_, T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::default())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        match self.i1.next() {
            Some(val) => Some(val),
            None => {
                // Most of the time the iterator will either always call
                // `next()` or always call `next_back()`. By swapping the
                // iterators once the first one is empty, we ensure that the
                // first branch is taken as often as possible, without
                // sacrificing correctness, as `i1` is empty anyway.
                mem::swap(&mut self.i1, &mut self.i2);
                self.i1.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn fold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i1.fold(accum, &mut f);
        self.i2.fold(accum, &mut f)
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        match self.i2.next_back() {
            Some(val) => Some(val),
            None => {
                // Same swap rationale as in `next()`, applied to the back.
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.next_back()
            }
        }
    }

    fn rfold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i2.rfold(accum, &mut f);
        self.i1.rfold(accum, &mut f)
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        // The two halves partition the deque's live elements, so their
        // lengths sum to at most the deque length and cannot overflow.
        let (l1, l2) = (self.i1.len(), self.i2.len());
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "l1 + l2 == number of live elements <= usize::MAX"
        )]
        let total = l1 + l2;
        total
    }
}

impl<T> core::iter::FusedIterator for Iter<'_, T> {}

// Borrowed iteration delegates to the slice methods, mirroring std's
// `IntoIterator` impls for `&VecDeque` and `&mut VecDeque`.
impl<'a, T, A: Allocator> IntoIterator for &'a VecDeque<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ---------------------------------------------------------------------------
// IterMut
// ---------------------------------------------------------------------------

/// An iterator over mutable references to the elements of a `VecDeque`.
///
/// This struct is created by the
/// [`iter_mut`](VecDeque::iter_mut) method on `VecDeque`.
pub struct IterMut<'a, T: 'a> {
    i1: slice::IterMut<'a, T>,
    i2: slice::IterMut<'a, T>,
}

impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut")
            .field(&self.i1.as_slice())
            .field(&self.i2.as_slice())
            .finish()
    }
}

impl<T> Default for IterMut<'_, T> {
    /// Creates an empty `vec_deque::IterMut`.
    fn default() -> Self {
        IterMut {
            i1: Default::default(),
            i2: Default::default(),
        }
    }
}

// An empty iterator holds no storage, so its default construction is infallible.
impl<T> TryDefault for IterMut<'_, T> {
    #[inline]
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Self::default())
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        match self.i1.next() {
            Some(val) => Some(val),
            None => {
                // Swap once the front slice is exhausted so subsequent
                // `next()` calls hit the populated half directly.
                mem::swap(&mut self.i1, &mut self.i2);
                self.i1.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn fold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i1.fold(accum, &mut f);
        self.i2.fold(accum, &mut f)
    }

    #[inline]
    fn last(mut self) -> Option<&'a mut T> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        match self.i2.next_back() {
            Some(val) => Some(val),
            None => {
                mem::swap(&mut self.i1, &mut self.i2);
                self.i2.next_back()
            }
        }
    }

    fn rfold<Acc, F>(self, accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.i2.rfold(accum, &mut f);
        self.i1.rfold(accum, &mut f)
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize {
        // The two halves partition the deque's live elements, so their
        // lengths sum to at most the deque length and cannot overflow.
        let (l1, l2) = (self.i1.len(), self.i2.len());
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "l1 + l2 == number of live elements <= usize::MAX"
        )]
        let total = l1 + l2;
        total
    }
}

impl<T> core::iter::FusedIterator for IterMut<'_, T> {}

impl<'a, T, A: Allocator> IntoIterator for &'a mut VecDeque<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T, A: Allocator> VecDeque<T, A> {
    /// Returns a front-to-back iterator over the deque's elements.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        let (a, b) = self.as_slices();
        Iter::new(a.iter(), b.iter())
    }

    /// Returns a front-to-back iterator over mutable references to the
    /// deque's elements.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let (a, b) = self.as_mut_slices();
        IterMut::new(a.iter_mut(), b.iter_mut())
    }

    /// Returns an [`Iter`] over the elements in the given range of the deque,
    /// mirroring std's [`VecDeque::range`](stock_alloc::collections::VecDeque::range).
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the resolved range is out of bounds —
    /// an excluded/inclusive edge overflows, the start exceeds the end, or the
    /// end exceeds the deque's length.
    pub fn try_range<R>(&self, range: R) -> Result<Iter<'_, T>, TrySliceRangeError>
    where
        R: RangeBounds<usize>,
    {
        // Resolve the logical range into one or two contiguous physical runs.
        let (a_range, b_range) = self.try_slice_ranges(range, self.len)?;
        // SAFETY: `try_slice_ranges` returns valid ranges into the physical
        // buffer over initialized elements.
        unsafe {
            let a = &*self.buffer_range(a_range);
            let b = &*self.buffer_range(b_range);
            Ok(Iter::new(a.iter(), b.iter()))
        }
    }

    /// Returns an [`IterMut`] over mutable references to the elements in the
    /// given range of the deque, mirroring std's
    /// [`VecDeque::range_mut`](stock_alloc::collections::VecDeque::range_mut).
    ///
    /// # Errors
    ///
    /// Returns [`TrySliceRangeError`] if the resolved range is out of bounds —
    /// an excluded/inclusive edge overflows, the start exceeds the end, or the
    /// end exceeds the deque's length.
    pub fn try_range_mut<R>(&mut self, range: R) -> Result<IterMut<'_, T>, TrySliceRangeError>
    where
        R: RangeBounds<usize>,
    {
        // Resolve the logical range into one or two contiguous physical runs.
        let (a_range, b_range) = self.try_slice_ranges(range, self.len)?;
        // SAFETY: `try_slice_ranges` returns valid ranges into the physical
        // buffer over initialized elements.
        unsafe {
            let a = &mut *self.buffer_range(a_range);
            let b = &mut *self.buffer_range(b_range);
            Ok(IterMut::new(a.iter_mut(), b.iter_mut()))
        }
    }
}

impl<'a, T> Iter<'a, T> {
    pub(super) fn new(i1: slice::Iter<'a, T>, i2: slice::Iter<'a, T>) -> Self {
        Self { i1, i2 }
    }
}

impl<'a, T> IterMut<'a, T> {
    pub(super) fn new(i1: slice::IterMut<'a, T>, i2: slice::IterMut<'a, T>) -> Self {
        Self { i1, i2 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::super::VecDeque;
    use olive_core::slice::TrySliceRangeError;

    /// Collects the elements yielded by an iterator into a freshly grown
    /// `std::vec::Vec`, for easy comparison against expected values.
    fn collect<'a, I: Iterator<Item = &'a i32>>(it: I) -> std::vec::Vec<i32> {
        it.copied().collect()
    }

    /// Builds a wrapped deque holding `[5, 4, 1, 2, 3]` with capacity 6, so
    /// that logical indices straddle the physical wrap point.
    fn wrapped_deque() -> VecDeque<i32> {
        let mut dq = VecDeque::<i32>::try_with_capacity(6).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        assert_eq!(dq.try_push_front_within_capacity(4), Ok(()));
        assert_eq!(dq.try_push_front_within_capacity(5), Ok(()));
        dq
    }

    #[test]
    fn try_range_full_non_wrapped() {
        let mut dq = VecDeque::<i32>::try_with_capacity(4).expect("allocation ok");
        for v in [1, 2, 3] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let got = collect(dq.try_range(..).expect("full range is resolvable"));
        assert_eq!(got, [1, 2, 3]);
    }

    #[test]
    fn try_range_partial_middle() {
        let mut dq = VecDeque::<i32>::try_with_capacity(5).expect("allocation ok");
        for v in [10, 20, 30, 40, 50] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        let got = collect(dq.try_range(1..4).expect("in-bounds range"));
        assert_eq!(got, [20, 30, 40]);
    }

    #[test]
    fn try_range_inclusive_end_bound() {
        let mut dq = VecDeque::<i32>::try_with_capacity(5).expect("allocation ok");
        for v in [10, 20, 30, 40, 50] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // `(..=1)` resolves to logical `0..2`.
        let got = collect(dq.try_range(..=1).expect("inclusive end in bounds"));
        assert_eq!(got, [10, 20]);
    }

    #[test]
    fn try_range_spanning_wrap_point() {
        // Logical layout: [5, 4, 1, 2, 3]; physical split puts the tail of the
        // first run and the head of the second on opposite sides of the buffer.
        let dq = wrapped_deque();
        // Indices 1..=4 => [4, 1, 2, 3], crossing the wrap boundary.
        let got = collect(dq.try_range(1..=4).expect("wrap-spanning range"));
        assert_eq!(got, [4, 1, 2, 3]);
    }

    #[test]
    fn try_range_single_element_across_wrap() {
        let dq = wrapped_deque();
        // Index 2 sits at the very start of the second physical run.
        let got = collect(dq.try_range(2..3).expect("single-element range"));
        assert_eq!(got, [1]);
    }

    #[test]
    fn try_range_empty_slice() {
        let dq = wrapped_deque();
        // A zero-width range yields no elements but still validates.
        let got = collect(dq.try_range(2..2).expect("empty range is valid"));
        assert!(got.is_empty());
    }

    #[test]
    fn try_range_out_of_bounds_rejects() {
        let dq = wrapped_deque();
        // len == 5, so ending at index 7 exceeds the bound.
        let err = match dq.try_range(0..7) {
            Err(e) => e,
            Ok(_) => panic!("expected out-of-bounds error"),
        };
        assert!(matches!(err, TrySliceRangeError::EndExceedsBound { .. }));
    }

    #[test]
    fn try_range_reversed_rejects() {
        let dq = wrapped_deque();
        #[allow(
            clippy::reversed_empty_ranges,
            reason = "we are deliberately testing error scenario"
        )]
        let err = match dq.try_range(3..1) {
            Err(e) => e,
            Ok(_) => panic!("expected reversed-range error"),
        };
        assert!(matches!(err, TrySliceRangeError::StartExceedsEnd { .. }));
    }

    #[test]
    fn try_range_mut_yields_writable_refs() {
        let mut dq = wrapped_deque();
        // Double every element in logical indices 0..=2 ([5, 4, 1]).
        for x in dq.try_range_mut(0..=2).expect("mutable range") {
            *x *= 2;
        }
        // Expected after doubling: [10, 8, 2, 2, 3].
        let (a, b) = dq.as_slices();
        let mut all = [0i32; 5];
        all[..a.len()].copy_from_slice(a);
        all[a.len()..].copy_from_slice(b);
        assert_eq!(all, [10, 8, 2, 2, 3]);
    }

    #[test]
    fn try_range_exact_size() {
        let dq = wrapped_deque();
        let it = dq.try_range(1..=4).expect("wrap-spanning range");
        assert_eq!(it.len(), 4);
        assert_eq!(it.size_hint(), (4, Some(4)));
    }

    #[test]
    fn try_range_double_ended() {
        let dq = wrapped_deque();
        // Logical layout: [5, 4, 1, 2, 3]; walk both ends of the full range.
        let mut it = dq.try_range(..).expect("full range is resolvable");
        assert_eq!(it.next(), Some(&5));
        assert_eq!(it.next_back(), Some(&3));
        assert_eq!(it.next(), Some(&4));
        assert_eq!(it.next_back(), Some(&2));
        assert_eq!(it.next(), Some(&1));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn try_range_mut_exact_size() {
        let mut dq = wrapped_deque();
        let it = dq
            .try_range_mut(1..=4)
            .expect("mutable wrap-spanning range");
        assert_eq!(it.len(), 4);
        assert_eq!(it.size_hint(), (4, Some(4)));
    }

    #[test]
    fn try_range_mut_double_ended() {
        let mut dq = VecDeque::<i32>::try_with_capacity(5).expect("allocation ok");
        for v in [1, 2, 3, 4, 5] {
            assert_eq!(dq.try_push_back_within_capacity(v), Ok(()));
        }
        // Walk both ends inward, tagging each visited element distinctly.
        let mut it = dq.try_range_mut(0..5).expect("full mutable range");
        assert_eq!(it.next(), Some(&mut 1));
        assert_eq!(it.next_back(), Some(&mut 5));
        assert_eq!(it.next(), Some(&mut 2));
        assert_eq!(it.next_back(), Some(&mut 4));
        assert_eq!(it.next(), Some(&mut 3));
        assert_eq!(it.next(), None);
    }
}
