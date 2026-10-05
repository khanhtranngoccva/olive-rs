//! Iterators for [`BTreeSet`].

use super::super::iter::{IntoIter as MapIntoIter, Iter as MapIter};
use super::super::set_val::SetValZST;
use super::BTreeSet;
use crate::alloc::{Allocator, Global};
use core::fmt;
use core::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator};
use olive_core::try_traits::{TryClone, TryCloneError};

// ── Iter ─────────────────────────────────────────────────────────────────────

/// An iterator yielding immutable references to the values of a [`BTreeSet`].
///
/// Returned by [`BTreeSet::iter`].
pub struct Iter<'a, T: 'a> {
    iter: MapIter<'a, T, SetValZST>,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    fn min(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next()
    }

    fn max(mut self) -> Option<&'a T>
    where
        &'a T: Ord,
    {
        self.next_back()
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T: 'a> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T: 'a> TryClone for Iter<'a, T> {
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        // Cloning is trivial.
        Ok(self.clone())
    }
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter.clone().map(|(k, _v)| k))
            .finish()
    }
}

// ── IntoIter ─────────────────────────────────────────────────────────────────

/// An owning iterator over the values of a [`BTreeSet`], consuming the set.
///
/// Returned by iterating over a `BTreeSet` directly ([`IntoIterator`]) or
/// calling [`IntoIterator::into_iter`].
pub struct IntoIter<T, A: Allocator = Global> {
    iter: MapIntoIter<T, SetValZST, A>,
}

impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

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

impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(k, _)| k)
    }
}

impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter.iter().map(|(k, _v)| k))
            .finish()
    }
}

// ── BTreeSet methods ─────────────────────────────────────────────────────────

impl<T, A: Allocator> BTreeSet<T, A> {
    /// Returns an iterator visiting the values in ascending order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.map.iter(),
        }
    }
}

impl<T, A: Allocator> IntoIterator for BTreeSet<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.map.into_iter(),
        }
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a BTreeSet<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    /// Smoke test for the iterator wiring. The underlying traversal logic is
    /// covered by the map's own tests; this only verifies that the set's
    /// newtypes delegate correctly and project away the ZST value.
    #[test]
    fn iter_and_into_iter_smoke() {
        let mut set = BTreeSet::new();
        for i in [3, 1, 4, 1, 5, 9, 2, 6] {
            set.try_insert(i).unwrap();
        }
        assert_eq!(set.len(), 7); // duplicate 1 collapsed

        // Borrowed iteration yields references in ascending order.
        let collected: Vec<&i32> = set.iter().collect();
        assert_eq!(collected, [&1, &2, &3, &4, &5, &6, &9]);

        // Reverse iteration.
        let reversed: Vec<&i32> = set.iter().rev().collect();
        assert_eq!(reversed, [&9, &6, &5, &4, &3, &2, &1]);

        // ExactSizeIterator.
        assert_eq!(set.iter().len(), 7);

        // Owned iteration consumes the set and yields values in order.
        let owned: Vec<i32> = set.into_iter().collect();
        assert_eq!(owned, [1, 2, 3, 4, 5, 6, 9]);
    }
}
