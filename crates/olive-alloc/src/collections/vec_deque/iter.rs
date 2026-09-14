//! Front-to-back iterators over a [`VecDeque`].
//!
//! A deque's elements may occupy two disjoint regions of its circular buffer,
//! so both `Iter` and `IterMut` chain two slice iterators — one for each region.
//! This mirrors the std reference implementation.

use core::fmt;
use core::mem;
use core::slice;

use super::VecDeque;
use olive_core::alloc::Allocator;
use olive_core::prelude::{TryClone, TryDefault};
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
