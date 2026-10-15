//! Query operations for [`LinkedList`].

use crate::alloc::Allocator;

use super::LinkedList;

impl<T, A: Allocator> LinkedList<T, A> {
    /// Returns `true` if the list has no elements.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    /// Returns the number of elements in the list.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
}
