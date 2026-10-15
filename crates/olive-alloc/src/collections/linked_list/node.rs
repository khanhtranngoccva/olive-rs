//! The node underlying [`LinkedList`](super::LinkedList).
use core::ptr::NonNull;

use crate::alloc::Allocator;
use crate::boxed::Box;

use super::LinkedList;

/// A node in a [`LinkedList`](super::LinkedList).
///
/// Nodes are individually allocated as `Box<Node<T>, &A>` with the list's
/// allocator and linked together through raw [`NonNull`] pointers: the list
/// owns every node reachable from its head or tail, and each node owns its element.
pub(super) struct Node<T> {
    /// The node after this one, or `None` when this is the last node.
    pub(super) next: Option<NonNull<Node<T>>>,
    /// The node before this one, or `None` when this is the first node.
    pub(super) prev: Option<NonNull<Node<T>>>,
    /// The element stored in this node.
    #[allow(
        dead_code,
        reason = "no operation reads elements yet; they are only written at \
                  construction and destroyed by the drop glue once popping or \
                  iteration lands"
    )]
    pub(super) element: T,
}

impl<T, A: Allocator> LinkedList<T, A> {
    /// Prepends an already-allocated node to the front of the list and
    /// returns a mutable reference to its element.
    ///
    /// # Safety
    ///
    /// - `node` must point to a `Node<T>` that was allocated with `self.alloc`
    ///   (as a `Box<Node<T>, A>` whose ownership the caller has released) and
    ///   must not be linked into this or any other list yet.
    /// - This method takes ownership of the node, so the pointer should not be used
    ///   again.
    /// - Its `element` must be fully initialized, because the returned reference
    ///   exposes it.
    pub(super) unsafe fn push_front_node(&mut self, mut node: NonNull<Node<T>>) -> &mut T {
        // SAFETY: by the caller's guarantee the node is a live, unlinked
        // allocation made through `self.alloc`, so writing through the raw
        // pointer is sound. Touching only the link fields keeps validity of
        // aliasing pointers into `element`.
        unsafe {
            (*node.as_ptr()).next = self.head;
            (*node.as_ptr()).prev = None;
        }
        let linked = Some(node);

        match self.head {
            None => self.tail = linked,
            // SAFETY: `head` points to a live node of this list.
            Some(head) => unsafe { (*head.as_ptr()).prev = linked },
        }

        self.head = linked;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "overflow requires at least 2 * 2 ^ usize::BITS memory which is impossible"
        )]
        {
            self.len += 1;
        }

        // SAFETY: `node` is now a part of the list and its
        // `element` is initialized per the caller's guarantee.
        unsafe { &mut node.as_mut().element }
    }

    /// Removes the node at the front of the list and returns it as a box
    /// borrowing the list's allocator, or `None` when the list is empty.
    ///
    /// Dropping the returned box drops the element and deallocates the node
    /// with `&self.alloc`.
    pub(super) fn pop_front_node(&mut self) -> Option<Box<Node<T>, &A>> {
        // Takes care not to create mutable references to whole nodes
        // keeps validity of aliasing pointers into `element`.
        self.head.map(|node| {
            // SAFETY: `node` is a live node allocated with `self.alloc`, so re-boxing
            // it here is sound; the box then owns both the node and its element.
            let node = unsafe { Box::from_raw_in(node.as_ptr(), &self.alloc) };

            self.head = node.next;

            match self.head {
                None => self.tail = None,
                // SAFETY: the new head is a live node of this list.
                Some(head) => unsafe { (*head.as_ptr()).prev = None },
            }

            #[allow(
                clippy::arithmetic_side_effects,
                reason = "cannot underflow, we are popping an element"
            )]
            {
                self.len -= 1;
            }

            node
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // This test is written to run under Miri (`cargo xtask miri-test`).
    // The second element has a pointer, and popping should not
    // tamper with the aliasing reference.
    #[test]
    fn pointer_to_second_element_survives_a_pop() {
        let mut l: LinkedList<i32> = LinkedList::new();
        // After the second push, element 1 is the second element of the
        // list, so the reference returned here points at it.
        let second = l.try_push_front_mut_give_back(1).unwrap();
        assert_eq!(*second, 1);
        let pointer: *mut i32 = second;

        l.try_push_front(2).unwrap();

        // Before popping, the pointer is cast back into a reference, and
        // the read afterwards goes through that reference.
        let reference: &mut i32 = unsafe { &mut *pointer };

        // Popping the first element drops its node and element. The
        // returned box borrows the list's allocator, so the list is not
        // queried until the box is gone.
        drop(l.pop_front_node().unwrap());
        assert_eq!(l.len(), 1);

        // The second element's storage was untouched by the pop: the
        // reference must still read the value it was created from.
        assert_eq!(*reference, 1);
    }
}
