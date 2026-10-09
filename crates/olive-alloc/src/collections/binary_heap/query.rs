use super::BinaryHeap;
use olive_core::alloc::Allocator;

impl<T, A: Allocator> BinaryHeap<T, A> {
    /// Returns the biggest item in the binary heap without removing it, or
    /// `None` if the heap is empty.
    ///
    /// # Panics
    ///
    /// Panics if the heap is poisoned ([`is_poisoned`](Self::is_poisoned)),
    /// in which case [`rebuild`](Self::rebuild) is needed.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.assume_unpoisoned();
        self.data.as_slice().first()
    }

    /// Returns the number of elements the heap can hold without
    /// reallocating.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the number of elements in the heap.
    #[inline]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the heap contains no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Retrieves a reference over the heap's allocator instance.
    #[inline]
    pub const fn allocator(&self) -> &A {
        self.data.allocator()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::BinaryHeap;
    use olive_core::try_traits::try_from_iterator::TryFromIterator;

    fn heap_of(items: impl IntoIterator<Item = i32>) -> BinaryHeap<i32> {
        BinaryHeap::try_from_iter(items).expect("allocation ok")
    }

    #[test]
    fn peek_agrees_with_first_pop() {
        let mut heap = heap_of([3, 9, 1]);
        let top = *heap.peek().expect("non-empty");
        assert_eq!(top, heap.pop().expect("non-empty"));
    }
}
