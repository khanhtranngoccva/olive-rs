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

    /// Provides a reference to the front element, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    #[must_use]
    #[inline]
    pub fn front(&self) -> Option<&T> {
        // SAFETY: `head` is nonempty and points to a live node of this
        // list whose `element` is initialized for as long as the node is in the list.
        self.head.map(|node| unsafe { &node.as_ref().element })
    }

    /// Provides a mutable reference to the front element, or `None` if the
    /// list is empty.
    ///
    /// This operation should compute in O(1) time.
    #[must_use]
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        // SAFETY: `head` is nonempty and points to a live node of this
        // list whose `element` is initialized for as long as the node is in the list.
        self.head
            .map(|mut node| unsafe { &mut node.as_mut().element })
    }

    /// Provides a reference to the back element, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    #[must_use]
    #[inline]
    pub fn back(&self) -> Option<&T> {
        // SAFETY: `tail` is nonempty and points to a live node of this
        // list whose `element` is initialized for as long as the node is in the list.
        self.tail.map(|node| unsafe { &node.as_ref().element })
    }

    /// Provides a mutable reference to the back element, or `None` if the
    /// list is empty.
    ///
    /// This operation should compute in O(1) time.
    #[must_use]
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        // SAFETY: `tail` is nonempty and points to a live node of this
        // list whose `element` is initialized for as long as the node is in the list.
        self.tail
            .map(|mut node| unsafe { &mut node.as_mut().element })
    }

    /// Returns `true` if the list contains an element equal to the given
    /// value.
    ///
    /// This operation should compute in O(n) time.
    // FIXME: switch to Iter implementation soon
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq<T>,
    {
        let mut node = self.head;
        while let Some(ptr) = node {
            // SAFETY: `ptr` points to a live node of this list, whose
            // `element` and links stay initialized and unchanged.
            let node_ref = unsafe { ptr.as_ref() };
            if node_ref.element == *x {
                return true;
            }
            node = node_ref.next;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn front_and_back_are_none_on_an_empty_list() {
        let l: LinkedList<i32> = LinkedList::new();
        assert!(l.front().is_none());
        assert!(l.back().is_none());

        let mut l: LinkedList<i32> = LinkedList::new();
        assert!(l.front_mut().is_none());
        assert!(l.back_mut().is_none());
    }

    #[test]
    fn front_and_back_see_both_ends() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(1).unwrap();
        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&1));

        l.try_push_back(2).unwrap();
        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&2));

        l.try_push_front(3).unwrap();
        assert_eq!(l.front(), Some(&3));
        assert_eq!(l.back(), Some(&2));
    }

    #[test]
    fn front_mut_and_back_mut_change_only_their_end() {
        let mut l: LinkedList<i32> = LinkedList::new();
        for value in 1..=4 {
            l.try_push_back(value).unwrap();
        }

        *l.front_mut().unwrap() = 10;
        *l.back_mut().unwrap() = 40;

        assert_eq!(l.front(), Some(&10));
        assert_eq!(l.back(), Some(&40));
        assert_eq!(l.len(), 4);
    }

    // This test is written to run under Miri.
    // The back element is held through a raw pointer, cast back into a
    // mutable reference, while the front is mutated through a fresh
    // mutable reference, and the two references must not conflict.
    // The back element must also not be the front element.
    #[test]
    fn reference_to_back_survives_a_front_mutation() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(1).unwrap();
        l.try_push_back(2).unwrap();

        let back: *mut i32 = l.back_mut().unwrap();
        assert_eq!(unsafe { *back }, 2);

        let reference: &mut i32 = unsafe { &mut *back };

        let front: &mut i32 = l.front_mut().unwrap();
        *front = 10;

        // The mutation touched only the front node.
        assert_eq!(*reference, 2);
        assert_eq!(l.front(), Some(&10));
        assert_eq!(l.back(), Some(&2));
    }

    // This test is written to run under Miri.
    // The front element is held through a raw pointer, cast back into a
    // mutable reference, while the front is mutated through a fresh
    // mutable reference, and the two references must not conflict.
    // The back element must also not be the front element.
    #[test]
    fn reference_to_front_survives_a_back_mutation() {
        let mut l: LinkedList<i32> = LinkedList::new();
        l.try_push_back(1).unwrap();
        l.try_push_back(2).unwrap();

        let front: *mut i32 = l.front_mut().unwrap();
        assert_eq!(unsafe { *front }, 1);

        let reference: &mut i32 = unsafe { &mut *front };

        let back: &mut i32 = l.back_mut().unwrap();
        *back = 10;

        // The mutation touched only the back node.
        assert_eq!(*reference, 1);
        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&10));
    }

    #[test]
    fn contains_finds_only_matching_elements() {
        let mut l: LinkedList<i32> = LinkedList::new();
        assert!(!l.contains(&1));

        for value in 1..=4 {
            l.try_push_back(value).unwrap();
        }

        assert!(l.contains(&1));
        assert!(l.contains(&2));
        assert!(l.contains(&4));
        assert!(!l.contains(&5));
    }
}
