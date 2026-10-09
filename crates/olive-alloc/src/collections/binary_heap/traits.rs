//! Trait implementations for the binary heap.

use super::BinaryHeap;
use crate::{alloc::Global, vec::Vec};
use core::fmt;
use olive_core::{alloc::Allocator, alloc_errors::TryReserveError, try_traits::TryFromIterator};

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<T: fmt::Debug, A: Allocator> fmt::Debug for BinaryHeap<T, A> {
    /// Formats the heap as a list of its elements in arbirary order (level order
    /// on clean heaps, arbitrary order on poisoned heaps), *not* in sorted order.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Poisoned: the level order is not even a heap any more, so
        // listing it would mislead. std has no poison state and no
        // equivalent branch.
        if self.poisoned {
            return f.write_str("<poisoned>");
        }
        f.debug_list().entries(self.data.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl<T: Ord> TryFromIterator<T> for BinaryHeap<T, Global> {
    type Error = TryReserveError;

    /// Collects the iterator into a vector on the [`Global`] allocator, then
    /// rebuilds it into a valid heap in linear time.
    ///
    /// This is the fallible analogue of std's `impl FromIterator<T> for
    /// BinaryHeap<T>`: on allocation failure the partially collected vector is
    /// dropped (elements included) and the error surfaces as
    /// [`TryReserveError`].
    fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Self::Error> {
        // SAFETY: The heap is rebuilt after and can only be saved.
        let mut heap = unsafe { Self::with_data(Vec::try_from_iter_in(iter, Global)?) };
        heap.rebuild();
        Ok(heap)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use olive_core::try_traits::TryFromIterator;

    // --- Debug ---------------------------------------------------------------

    #[test]
    fn format_shows_the_elements_level_order() {
        let heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([100, 80, 70, 60, 50, 10]).expect("allocation ok");
        let text = std::format!("{heap:?}");
        assert!(
            text.starts_with('[') && text.ends_with(']'),
            "should render as a list: {text}"
        );
        assert!(text.contains("100"), "root should be shown: {text}");
    }

    // --- TryFromIterator -------------------------------------------------------

    #[test]
    fn try_from_iter_produces_a_valid_max_heap() {
        // Unordered input: the constructor must heapify, so peek sees the
        // maximum and popping drains in decreasing order.
        let mut heap: BinaryHeap<i32> =
            TryFromIterator::try_from_iter([5, 11, 3, 8, 1, 11, 0]).expect("allocation ok");
        assert_eq!(heap.len(), 7);
        assert_eq!(heap.peek(), Some(&11));
        let mut out = std::vec::Vec::new();
        while let Some(item) = heap.pop() {
            out.push(item);
        }
        assert_eq!(out, std::vec![11, 11, 8, 5, 3, 1, 0]);
    }
}
