//! Cursors over a [`LinkedList`].
//!
//! Cursors are pointers to the list that can be freely seeked back-and-forth.
//! These cursors are circular, and point to a "ghost" non-element after the end
//! of the list or when the list is empty.
use super::LinkedList;
use super::node::Node;
use crate::alloc::{AllocError, Allocator, Global};
use crate::boxed::Box;
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;
use olive_core::TryClone;
use olive_core::alloc::AllocatorTryClone;
use olive_core::try_traits::try_clone::TryCloneError;

/// The non-null head, tail and length of a chain of detached nodes.
type DetachedNodes<T> = (NonNull<Node<T>>, NonNull<Node<T>>, usize);

// ── Public list methods ─────────────────────────────────────────────────────────────────────

impl<T, A: Allocator> LinkedList<T, A> {
    /// Provides a cursor at the front element of the list.
    ///
    /// The cursor starts pointing at the front element. If the list is empty,
    /// the cursor points at the "ghost" non-element.
    #[inline]
    #[must_use]
    pub fn cursor_front(&self) -> Cursor<'_, T, A> {
        Cursor {
            index: 0,
            current: self.head,
            list: self,
        }
    }

    /// Provides a mutable cursor at the front element of the list.
    ///
    /// The cursor starts pointing at the front element. If the list is empty,
    /// the cursor points at the "ghost" non-element.
    #[inline]
    #[must_use]
    pub fn cursor_front_mut(&mut self) -> CursorMut<'_, T, A> {
        CursorMut {
            index: 0,
            current: self.head,
            list: self,
        }
    }

    /// Provides a cursor at the back element of the list.
    ///
    /// The cursor starts pointing at the back element. If the list is empty,
    /// the cursor points at the "ghost" non-element.
    #[inline]
    #[must_use]
    pub fn cursor_back(&self) -> Cursor<'_, T, A> {
        Cursor {
            index: self.len.saturating_sub(1),
            current: self.tail,
            list: self,
        }
    }

    /// Provides a mutable cursor at the back element of the list.
    ///
    /// The cursor starts pointing at the back element. If the list is empty,
    /// the cursor points at the "ghost" non-element.
    #[inline]
    #[must_use]
    pub fn cursor_back_mut(&mut self) -> CursorMut<'_, T, A> {
        CursorMut {
            index: self.len.saturating_sub(1),
            current: self.tail,
            list: self,
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────────────────

impl<T, A: Allocator> LinkedList<T, A> {
    /// Unlinks the specified node from the list.
    ///
    /// This method takes care not to create mutable references to `element`,
    /// to maintain validity of aliasing pointers to `element`.
    ///
    /// This method does not clear the link fields of the unlinked node.
    ///
    /// # Safety
    ///
    /// `node` must point to a valid node that belongs to the list, and no
    /// other existing aliasing reference into that node may be used afterwards.
    #[inline]
    unsafe fn unlink_node(&mut self, mut node: NonNull<Node<T>>) {
        // SAFETY: this one is ours now, we can create an &mut.
        let node = unsafe { node.as_mut() };

        // Not creating new mutable (unique!) references overlapping `element`.
        match node.prev {
            Some(prev) => unsafe { (*prev.as_ptr()).next = node.next },
            // this node is the head node
            None => self.head = node.next,
        };

        match node.next {
            Some(next) => unsafe { (*next.as_ptr()).prev = node.prev },
            // this node is the tail node
            None => self.tail = node.prev,
        };

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "cannot underflow, we are unlinking a node that belongs to the list"
        )]
        {
            self.len -= 1;
        }
    }

    /// Splices a series of nodes between two existing nodes.
    ///
    /// This method takes care not to create multiple mutable references to
    /// whole nodes at the same time, to maintain validity of aliasing
    /// pointers to `element`.
    ///
    /// # Safety
    ///
    /// The nodes from `splice_start` to `splice_end` must form a valid chain
    /// of `splice_length` nodes allocated with this list's allocator, and
    /// they must not be linked into the list.
    #[inline]
    unsafe fn splice_nodes(
        &mut self,
        existing_prev: Option<NonNull<Node<T>>>,
        existing_next: Option<NonNull<Node<T>>>,
        splice_start: NonNull<Node<T>>,
        splice_end: NonNull<Node<T>>,
        splice_length: usize,
    ) {
        // This method takes care not to create multiple mutable references to
        // whole nodes at the same time, to maintain validity of aliasing
        // pointers into `element`.
        if let Some(existing_prev) = existing_prev {
            unsafe {
                (*existing_prev.as_ptr()).next = Some(splice_start);
            }
        } else {
            self.head = Some(splice_start);
        }
        if let Some(existing_next) = existing_next {
            unsafe {
                (*existing_next.as_ptr()).prev = Some(splice_end);
            }
        } else {
            self.tail = Some(splice_end);
        }
        unsafe {
            (*splice_start.as_ptr()).prev = existing_prev;
            (*splice_end.as_ptr()).next = existing_next;
        }

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "the list cannot hold more nodes than `usize::MAX`, so adding \
                      `splice_length` nodes cannot overflow the length"
        )]
        {
            self.len += splice_length;
        }
    }

    /// Consumes the list and returns the non-null head, tail and length of
    /// the detached chain of nodes, or `None` if the list was empty.
    #[inline]
    fn detach_all_nodes(mut self) -> Option<DetachedNodes<T>> {
        let head = self.head.take();
        let tail = self.tail.take();
        let len = mem::replace(&mut self.len, 0);
        if let Some(head) = head {
            // Safety: in a `LinkedList`, either both the head and tail are
            // `None` because the list is empty, or both head and tail are
            // `Some` because the list is populated. Since we have verified
            // the head is `Some`, we are sure the tail is `Some` too.
            let tail = unsafe { tail.unwrap_unchecked() };
            Some((head, tail, len))
        } else {
            None
        }
    }

    /// Splits the list into two parts after the node `at` places from the front,
    /// where `split_node` is the first node of the second part.
    ///
    /// Returns the part before `split_node`, leaving the part from `split_node` on in
    /// `self`. If `split_node` is `None` the whole list is moved into the
    /// returned one and `self` is left empty, using `alloc`.
    ///
    /// # Safety
    ///
    /// - `split_node` must either be `None`, or point to a valid node belonging
    ///   to the list.
    /// - `at` must be the index of `split_node` in the list, or the list's length
    ///   when `split_node` is `None` (ghost non-element).
    /// - `alloc` must point to the same backing store as `self.alloc` (usually
    ///   acquired via [`AllocatorTryClone`]).
    #[inline]
    unsafe fn split_off_before_node(
        &mut self,
        split_node: Option<NonNull<Node<T>>>,
        at: usize,
        alloc: A,
    ) -> Self {
        let split_node = match split_node {
            None => return mem::replace(self, LinkedList::new_in(alloc)),
            Some(split_node) => split_node,
        };

        // This branch can only be taken if the linked list is non-empty.
        unsafe {
            let old_len = self.len;

            // Modify the second part (for self) and extract the first part (for returned).
            let first_part_head;
            let first_part_tail = (*split_node.as_ptr()).prev.take();
            if let Some(first_part_tail) = first_part_tail {
                (*first_part_tail.as_ptr()).next = None;
                first_part_head = self.head;
            }
            // Edge case: the split node is the first node, leaving behind no node for the first part.
            else {
                first_part_head = None;
            }

            self.head = Some(split_node);
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "caller guarantees at <= old_len"
            )]
            {
                self.len = old_len - at;
            }

            // Bootstrap the first part.
            LinkedList {
                head: first_part_head,
                tail: first_part_tail,
                len: at,
                alloc,
                marker: PhantomData,
            }
        }
    }

    /// Splits the list into two before the node `at` places from the front,
    /// where `split_node` is the last node of the first part.
    ///
    /// Returns the part after `split_node`, leaving the part up to `split_node` in
    /// `self`. If `split_node` is `None` the whole list is moved into the
    /// returned one and `self` is left empty, using `alloc`.
    ///
    /// # Safety
    ///
    /// - `split_node` must either be `None`, or point to a valid node belonging
    ///   to the list.
    /// - `at` must be the index just past `split_node` in the list,
    ///    or `0` when `split_node` is `None` (ghost non-element).
    /// - `alloc` must point to the same backing store as `self.alloc` (usually
    ///   acquired via [`AllocatorTryClone`]).
    #[inline]
    unsafe fn split_off_after_node(
        &mut self,
        split_node: Option<NonNull<Node<T>>>,
        at: usize,
        alloc: A,
    ) -> Self {
        let split_node = match split_node {
            None => return mem::replace(self, LinkedList::new_in(alloc)),
            Some(split_node) => split_node,
        };
        unsafe {
            let old_len = self.len;
            let second_part_head = (*split_node.as_ptr()).next.take();
            let second_part_tail;
            // Edge case: the split node is the last node, leaving behind no node for the second part.
            if let Some(second_part_head) = second_part_head {
                (*second_part_head.as_ptr()).prev = None;
                second_part_tail = self.tail;
            } else {
                second_part_tail = None;
            }

            // Fix the first part
            self.tail = Some(split_node);
            self.len = at;

            // Bootstrap the second part.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "caller guarantees at <= old_len (index of `at` is one past the element)"
            )]
            return LinkedList {
                head: second_part_head,
                tail: second_part_tail,
                len: old_len - at,
                alloc,
                marker: PhantomData,
            };
        }
    }
}

// ── Cursor ─────────────────────────────────────────────────────────────────────

/// A cursor over a [`LinkedList`] that looks at one element.
///
/// A `Cursor` is like an iterator, except that it can freely seek
/// back-and-forth, and always points to either:
///
/// 1. One of the elements of the underlying `LinkedList`.
/// 2. The "ghost" non-element if the cursor is past the end of the list or
///    if the list is empty.
///
/// A `Cursor` is obtained with [`LinkedList::cursor_front`] or
/// [`LinkedList::cursor_back`].
///
/// Mutating methods are only available on [`CursorMut`].
#[derive(Debug, TryClone)]
pub struct Cursor<'a, T: 'a, A: Allocator = Global> {
    index: usize,
    current: Option<NonNull<Node<T>>>,
    list: &'a LinkedList<T, A>,
}

impl<T, A: Allocator> Clone for Cursor<'_, T, A> {
    fn clone(&self) -> Self {
        let Cursor {
            index,
            current,
            list,
        } = *self;
        Cursor {
            index,
            current,
            list,
        }
    }
}

impl<'a, T, A: Allocator> Cursor<'a, T, A> {
    /// Returns the index of the cursor's element in the list, or `None` when
    /// the cursor points at the "ghost" non-element.
    #[must_use]
    pub fn index(&self) -> Option<usize> {
        let _ = self.current?;
        Some(self.index)
    }

    /// Moves the cursor to the next element of the list.
    ///
    /// If the cursor is pointing at the "ghost" non-element then this moves
    /// it to the front element of the list. If it is pointing at the back
    /// element of the list then this moves it to the "ghost" non-element.
    pub fn move_next(&mut self) {
        if let Some(current) = self.current {
            unsafe { self.current = (*current.as_ptr()).next };
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is bounded by the list's length, so incrementing it \
                      cannot overflow"
            )]
            {
                self.index += 1;
            }
        } else {
            self.current = self.list.head;
            self.index = 0;
        }
    }

    /// Moves the cursor to the previous element of the list.
    ///
    /// If the cursor is pointing at the "ghost" non-element then this moves
    /// it to the back element of the list. If it is pointing at the front
    /// element of the list then this moves it to the "ghost" non-element.
    pub fn move_prev(&mut self) {
        if let Some(current) = self.current {
            unsafe {
                self.current = current.as_ref().prev;
                self.index = self.index.checked_sub(1).unwrap_or_else(|| self.list.len());
            }
        } else {
            self.current = self.list.tail;
            self.index = self.list.len().saturating_sub(1);
        }
    }

    /// Returns a reference to the element that the cursor is currently
    /// pointing to.
    ///
    /// This returns `None` if the cursor is currently pointing at the
    /// "ghost" non-element.
    #[must_use]
    pub fn current(&self) -> Option<&'a T> {
        // SAFETY: `self.current` either points to a node that belongs to the
        // list, or is `None`.
        unsafe { self.current.map(|current| &(*current.as_ptr()).element) }
    }

    /// Returns a reference to the next element, without moving the cursor.
    ///
    /// If the cursor is pointing to the "ghost" non-element then this returns
    /// the front element of the list.
    #[must_use]
    pub fn peek_next(&self) -> Option<&'a T> {
        // SAFETY: `self.current` either points to a node that belongs to the
        // list, or is `None`.
        unsafe {
            let next = match self.current {
                None => self.list.head,
                Some(node) => node.as_ref().next,
            };
            next.map(|node| &(*node.as_ptr()).element)
        }
    }

    /// Returns a reference to the previous element, without moving the
    /// cursor.
    ///
    /// If the cursor is pointing to the "ghost" non-element then this returns
    /// the back element of the list.
    #[must_use]
    pub fn peek_prev(&self) -> Option<&'a T> {
        // SAFETY: `self.current` either points to a node that belongs to the
        // list, or is `None`.
        unsafe {
            let prev = match self.current {
                None => self.list.tail,
                Some(node) => node.as_ref().prev,
            };
            prev.map(|node| &(*node.as_ptr()).element)
        }
    }

    /// Returns a reference to the front element of the cursor's parent list,
    /// or `None` if the list is empty.
    #[must_use]
    pub fn front(&self) -> Option<&T> {
        self.list.front()
    }

    /// Returns a reference to the back element of the cursor's parent list,
    /// or `None` if the list is empty.
    #[must_use]
    pub fn back(&self) -> Option<&T> {
        self.list.back()
    }

    /// Returns a read-only reference to the cursor's parent list.
    #[must_use]
    pub fn as_list(&self) -> &'a LinkedList<T, A> {
        self.list
    }
}

// ── CursorMut ─────────────────────────────────────────────────────────────────────

/// A mutable cursor over a [`LinkedList`] that looks at one element.
///
/// A `CursorMut` is like an iterator, except that it can freely seek
/// back-and-forth, positions itself in the list, and can mutate or reorder
/// the elements it visits. It always points to either:
///
/// 1. One of the elements of the underlying `LinkedList`. This means that the
///    cursor always has a predecessor and successor element.
/// 2. The "ghost" non-element. This is the first element in the
///    `LinkedList`, or the last one in it, if the list is empty.
///
/// A `CursorMut` is obtained with [`LinkedList::cursor_front_mut`] or
/// [`LinkedList::cursor_back_mut`].
#[derive(Debug)]
pub struct CursorMut<'a, T: 'a, A: Allocator = Global> {
    index: usize,
    current: Option<NonNull<Node<T>>>,
    list: &'a mut LinkedList<T, A>,
}

impl<T, A: Allocator> CursorMut<'_, T, A> {
    /// Returns the index of the cursor's element in the list, or `None` when
    /// the cursor points at the "ghost" non-element.
    #[must_use]
    pub fn index(&self) -> Option<usize> {
        let _ = self.current?;
        Some(self.index)
    }

    /// Moves the cursor to the next element of the list.
    ///
    /// If the cursor is pointing at the "ghost" non-element then this moves
    /// it to the front element of the list. If it is pointing at the back
    /// element of the list then this moves it to the "ghost" non-element.
    pub fn move_next(&mut self) {
        if let Some(current) = self.current {
            unsafe { self.current = current.as_ref().next };
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is bounded by the list's length, so incrementing it \
                      cannot overflow"
            )]
            {
                self.index += 1;
            }
        } else {
            self.current = self.list.head;
            self.index = 0;
        }
    }

    /// Moves the cursor to the previous element of the list.
    ///
    /// If the cursor is pointing at the "ghost" non-element then this moves
    /// it to the back element of the list. If it is pointing at the front
    /// element of the list then this moves it to the "ghost" non-element.
    pub fn move_prev(&mut self) {
        if let Some(current) = self.current {
            unsafe {
                self.current = current.as_ref().prev;
                self.index = self.index.checked_sub(1).unwrap_or_else(|| self.list.len());
            }
        } else {
            self.current = self.list.tail;
            self.index = self.list.len().saturating_sub(1);
        }
    }

    /// Returns a reference to the element that the cursor is currently
    /// pointing to.
    ///
    /// This returns `None` if the cursor is currently pointing at the
    /// "ghost" non-element.
    #[must_use]
    pub fn current(&mut self) -> Option<&mut T> {
        // SAFETY: `self.current` either points to a node that belongs to the
        // list, or is `None`. We have a `&mut self` so no other reference
        // into that node can exist at the same time.
        unsafe { self.current.map(|current| &mut (*current.as_ptr()).element) }
    }

    /// Returns a reference to the front element of the cursor's parent list,
    /// or `None` if the list is empty.
    #[must_use]
    pub fn front(&self) -> Option<&T> {
        self.list.front()
    }

    /// Returns a mutable reference to the front element of the cursor's
    /// parent list, or `None` if the list is empty.
    #[must_use]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.list.front_mut()
    }

    /// Returns a reference to the back element of the cursor's parent list,
    /// or `None` if the list is empty.
    #[must_use]
    pub fn back(&self) -> Option<&T> {
        self.list.back()
    }

    /// Returns a mutable reference to the back element of the cursor's
    /// parent list, or `None` if the list is empty.
    #[must_use]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.list.back_mut()
    }

    /// Returns a reference to the next element, without moving the cursor.
    ///
    /// If the cursor is pointing to the "ghost" non-element then this returns
    /// the front element of the list. If it is pointing to the last
    /// element of the `LinkedList` then this returns `None`.
    #[must_use]
    pub fn peek_next(&mut self) -> Option<&mut T> {
        // SAFETY: `self.current` either points to a node thfat belongs to the
        // list, or is `None`. We have a `&mut self` so no other reference
        // into that node can exist at the same time.
        unsafe {
            let next = match self.current {
                None => self.list.head,
                Some(node) => node.as_ref().next,
            };
            next.map(|node| &mut (*node.as_ptr()).element)
        }
    }

    /// Returns a reference to the previous element, without moving the
    /// cursor.
    ///
    /// If the cursor is pointing to the "ghost" non-element then this returns
    /// the back element of the list.
    #[must_use]
    pub fn peek_prev(&mut self) -> Option<&mut T> {
        // SAFETY: `self.current` either points to a node that belongs to the
        // list, or is `None`. We have a `&mut self` so no other reference
        // into that node can exist at the same time.
        unsafe {
            let prev = match self.current {
                None => self.list.tail,
                Some(node) => node.as_ref().prev,
            };
            prev.map(|node| &mut (*node.as_ptr()).element)
        }
    }

    /// Returns a read-only reference to the cursor's parent list.
    #[must_use]
    pub fn as_list(&mut self) -> &mut LinkedList<T, A> {
        self.list
    }

    /// Converts the cursor into an immutable one, borrowing the list.
    #[must_use]
    pub fn as_cursor(&mut self) -> Cursor<'_, T, A> {
        Cursor {
            index: self.index,
            current: self.current,
            list: self.list,
        }
    }
}

impl<T, A: Allocator> CursorMut<'_, T, A> {
    /// Inserts a new element after the current one.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the new
    /// element is inserted at the front of the list.
    ///
    /// The operation is fallible: when allocating the node fails, the
    /// element is dropped and the list and the cursor are unchanged.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] when allocating the node fails.
    pub fn try_insert_after(&mut self, element: T) -> Result<(), AllocError> {
        self.try_insert_after_give_back(element)
            .map_err(|(_, error)| error)
    }

    /// Inserts a new element after the current one, returning the element
    /// when the node cannot be allocated.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the new
    /// element is inserted at the front of the list.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns `(element, AllocError)` when allocating the node fails. The
    /// list and the cursor are unchanged, and the element is handed back.
    pub fn try_insert_after_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        let node = match Box::try_new_uninit_in(&self.list.alloc) {
            Ok(node) => node,
            Err(error) => return Err((element, error)),
        };
        let node = node.write(Node {
            next: None,
            prev: None,
            element,
        });
        let (spliced_node, _) = Box::into_non_null_with_allocator(node);

        // SAFETY: `spliced_node` points to a node allocated with the list's
        // allocator, and it is not linked into the list yet.
        unsafe {
            let node_next = match self.current {
                None => self.list.head,
                Some(node) => node.as_ref().next,
            };
            self.list
                .splice_nodes(self.current, node_next, spliced_node, spliced_node, 1);
            if self.current.is_none() {
                // The "ghost" non-element's index has changed.
                self.index = self.list.len;
            }
        }

        Ok(())
    }

    /// Inserts a new element before the current one.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the new
    /// element is inserted at the back of the list.
    ///
    /// The operation is fallible: when allocating the node fails, the
    /// element is dropped and the list and the cursor are unchanged.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] when allocating the node fails.
    pub fn try_insert_before(&mut self, element: T) -> Result<(), AllocError> {
        self.try_insert_before_give_back(element)
            .map_err(|(_, error)| error)
    }

    /// Inserts a new element before the current one, returning the element
    /// when the node cannot be allocated.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the new
    /// element is inserted at the back of the list.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns `(element, AllocError)` when allocating the node fails. The
    /// list and the cursor are unchanged, and the element is handed back.
    pub fn try_insert_before_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        let node = match Box::try_new_uninit_in(&self.list.alloc) {
            Ok(node) => node,
            Err(error) => return Err((element, error)),
        };
        let node = node.write(Node {
            next: None,
            prev: None,
            element,
        });
        let (spliced_node, _) = Box::into_non_null_with_allocator(node);

        // SAFETY: `spliced_node` points to a node allocated with the list's
        // allocator, and it is not linked into the list yet.
        unsafe {
            let node_prev = match self.current {
                None => self.list.tail,
                Some(node) => node.as_ref().prev,
            };
            self.list
                .splice_nodes(node_prev, self.current, spliced_node, spliced_node, 1);

            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is bounded by the list's length, so incrementing it \
                          cannot overflow"
            )]
            {
                self.index += 1;
            }
        }

        Ok(())
    }

    /// Removes the current element from the list.
    ///
    /// The element that was removed is returned, and the cursor is moved to
    /// point to the next element in the list.
    ///
    /// If the cursor is currently pointing to the "ghost" non-element then no
    /// element is removed and `None` is returned.
    pub fn remove_current(&mut self) -> Option<T> {
        let unlinked_node = self.current?;
        // SAFETY: `unlinked_node` points to a node that belongs to the list,
        // and the cursor is the only holder of a pointer into it.
        unsafe {
            self.current = unlinked_node.as_ref().next;
            self.list.unlink_node(unlinked_node);
            let unlinked_node = Box::from_raw_in(unlinked_node.as_ptr(), &self.list.alloc);
            Some(Node::into_element(unlinked_node))
        }
    }

    /// Removes the current element from the list without deallocating the
    /// node, returning it as a new list containing only this node.
    ///
    /// The cursor is moved to point to the next element in the list.
    ///
    /// If the cursor is currently pointing to the "ghost" non-element then no
    /// element is removed and `None` is returned.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] when cloning the allocator fails. The list
    /// and the cursor are unchanged in that case.
    pub fn try_remove_current_as_list(&mut self) -> Result<Option<LinkedList<T, A>>, TryCloneError>
    where
        A: AllocatorTryClone,
    {
        let Some(mut unlinked_node) = self.current else {
            return Ok(None);
        };

        // Clone the allocator up front so that a failed removal leaves both
        // the list and the cursor untouched.
        let alloc = self.list.alloc.try_clone()?;

        // SAFETY: `unlinked_node` points to a node that belongs to the list,
        // and the cursor is the only holder of a pointer into it.
        unsafe {
            self.current = unlinked_node.as_ref().next;
            self.list.unlink_node(unlinked_node);

            unlinked_node.as_mut().prev = None;
            unlinked_node.as_mut().next = None;

            Ok(Some(LinkedList {
                head: Some(unlinked_node),
                tail: Some(unlinked_node),
                len: 1,
                alloc,
                marker: PhantomData,
            }))
        }
    }

    /// Splits the list into two after the current element. This will return
    /// a new list consisting of everything after the cursor, with the
    /// original list retaining everything before.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the entire
    /// contents of the list are moved.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] when cloning the allocator fails. The list
    /// and the cursor are unchanged in that case.
    pub fn try_split_after(&mut self) -> Result<LinkedList<T, A>, TryCloneError>
    where
        A: AllocatorTryClone,
    {
        // Clone the allocator up front so that a failed split leaves both
        // the list and the cursor untouched.
        let alloc = self.list.alloc.try_clone()?;

        let split_off_idx = if self.index == self.list.len {
            0
        } else {
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is smaller than the list's length here, so adding \
                          one cannot overflow"
            )]
            {
                self.index + 1
            }
        };
        if self.index == self.list.len {
            // The "ghost" non-element's index has changed to 0.
            self.index = 0;
        }

        // SAFETY: `self.current` points to a node belonging to the list, or
        // is `None`, and `split_off_idx` is the index just past it, or `0`
        // when at the "ghost" non-element, and the allocator matches.
        Ok(unsafe {
            self.list
                .split_off_after_node(self.current, split_off_idx, alloc)
        })
    }

    /// Splits the list into two before the current element. This will return
    /// a new list consisting of everything before the cursor, with the
    /// original list retaining everything after.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the entire
    /// contents of the list are moved.
    ///
    /// # Errors
    ///
    /// Returns [`TryCloneError`] when cloning the allocator fails. The list
    /// and the cursor are unchanged in that case.
    pub fn try_split_before(&mut self) -> Result<LinkedList<T, A>, TryCloneError>
    where
        A: AllocatorTryClone,
    {
        // Clone the allocator up front so that a failed split leaves both
        // the list and the cursor untouched.
        let alloc = self.list.alloc.try_clone()?;

        let split_off_idx = self.index;
        self.index = 0;

        // SAFETY: `self.current` points to a node belonging to the list, or
        // is `None`, and `split_off_idx` is the index of that node, or the
        // list's length when at the "ghost" non-element, and the allocator
        // matches.
        Ok(unsafe {
            self.list
                .split_off_before_node(self.current, split_off_idx, alloc)
        })
    }

    /// Appends an element to the front of the cursor's parent list. The node
    /// that the cursor points to is unchanged, even if it is the "ghost"
    /// node.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] when allocating the node fails. The list and
    /// the cursor are unchanged in that case.
    ///
    // `push_front` continues to point to the "ghost" when it adds a node, to
    // mimic the behavior of `CursorMut::try_insert_before` on an empty
    // list.
    pub fn try_push_front(&mut self, element: T) -> Result<(), AllocError> {
        self.list.try_push_front(element)?;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "the index is bounded by the list's length, so incrementing it \
                      cannot overflow"
        )]
        {
            self.index += 1;
        }
        Ok(())
    }

    /// Appends an element to the front of the cursor's parent list,
    /// returning the element when the node cannot be allocated.
    ///
    /// The node that the cursor points to is unchanged, even if it is
    /// the "ghost" node.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns `(element, AllocError)` when allocating the node fails. The
    /// list and the cursor are unchanged, and the element is handed back.
    pub fn try_push_front_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        self.list.try_push_front_give_back(element)?;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "the index is bounded by the list's length, so incrementing it \
                      cannot overflow"
        )]
        {
            self.index += 1;
        }
        Ok(())
    }

    /// Appends an element to the back of the cursor's parent list. The node
    /// that the cursor points to is unchanged, even if it is the "ghost"
    /// node.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] when allocating the node fails. The list and
    /// the cursor are unchanged in that case.
    pub fn try_push_back(&mut self, element: T) -> Result<(), AllocError> {
        self.list.try_push_back(element)?;
        if self.current().is_none() {
            // The index of the "ghost" is the length of the list, so we just
            // need to increment self.index to reflect the new length of the
            // list.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is bounded by the list's length, so incrementing it \
                          cannot overflow"
            )]
            {
                self.index += 1;
            }
        }
        Ok(())
    }

    /// Appends an element to the back of the cursor's parent list, returning
    /// the element when the node cannot be allocated.
    ///
    /// The node that the cursor points to is unchanged, even if it is
    /// the "ghost" node.
    ///
    /// This operation should compute in *O*(1) time.
    ///
    /// # Errors
    ///
    /// Returns `(element, AllocError)` when allocating the node fails. The
    /// list and the cursor are unchanged, and the element is handed back.
    pub fn try_push_back_give_back(&mut self, element: T) -> Result<(), (T, AllocError)> {
        self.list.try_push_back_give_back(element)?;
        if self.current().is_none() {
            // The index of the "ghost" is the length of the list, so we just
            // need to increment self.index to reflect the new length of the
            // list.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index is bounded by the list's length, so incrementing it \
                          cannot overflow"
            )]
            {
                self.index += 1;
            }
        }
        Ok(())
    }

    /// Removes the first element from the cursor's parent list and returns
    /// it, or `None` if the list is empty. The element the cursor points to
    /// remains unchanged, unless it was pointing to the front element. In
    /// that case, it points to the new front element.
    ///
    /// This operation should compute in *O*(1) time.
    pub fn pop_front(&mut self) -> Option<T> {
        // We can't check if current is empty, we must check the list
        // directly. It is possible for `self.current == None` and the list to
        // be non-empty.
        if self.list.is_empty() {
            None
        } else {
            // We can't point to the node that we pop. Copying the behavior of
            // `remove_current`, we move on to the next node in the sequence.
            // If the list is of length 1 then we end up pointing to the "ghost"
            // node at index 1 -> 0, which is expected.
            if self.list.head == self.current {
                self.move_next();
            }
            // An element was removed before our current position, so
            // the index must be decremented.
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the list is non-empty and the cursor is not at head, so index > 0"
            )]
            {
                self.index -= 1;
            }
            self.list.pop_front()
        }
    }

    /// Removes the last element from the cursor's parent list and returns
    /// it, or `None` if the list is empty. The element the cursor points to
    /// remains unchanged, unless it was pointing to the back element. In
    /// that case, it points to the "ghost" element.
    ///
    /// This operation should compute in *O*(1) time.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.list.is_empty() {
            None
        } else {
            if self.list.tail == self.current {
                // The index after popping reflects the length of the list.
                // It was the length of the list minus 1, but now the list
                // is 1 smaller.
                // No change is needed for `index`.
                self.current = None;
            } else if self.current.is_none() {
                #[allow(clippy::arithmetic_side_effects, reason = "the list is non-empty")]
                {
                    self.index = self.list.len - 1;
                }
            }
            self.list.pop_back()
        }
    }
}

impl<T> CursorMut<'_, T> {
    /// Splices all the elements from `list` into the parent list after the
    /// current one. The `list` is emptied and dropped.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the
    /// elements are spliced at the front of the parent list.
    ///
    /// This operation should compute in *O*(1) time. It moves the nodes of
    /// `list` into the parent list, without allocating or deallocating.
    ///
    /// This operation is only available for the [`Global`] allocator: the
    /// nodes would otherwise have to be reallocated when the two lists do
    /// not share the same allocator.
    pub fn splice_after(&mut self, list: LinkedList<T>) {
        // The list is empty -> nothing to splice in.
        if list.len == 0 {
            return;
        }

        // SAFETY: `splice_nodes` is given a valid chain of nodes allocated
        // with the parent list's allocator.
        unsafe {
            let Some((splice_head, splice_tail, splice_len)) = list.detach_all_nodes() else {
                return;
            };
            let node_next = match self.current {
                None => self.list.head,
                Some(node) => node.as_ref().next,
            };
            self.list.splice_nodes(
                self.current,
                node_next,
                splice_head,
                splice_tail,
                splice_len,
            );

            // The cursor must continue pointing at the ghost element.
            if self.current.is_none() {
                self.index = self.list.len;
            }
        }
    }

    /// Splices all the elements from `list` into the parent list before the
    /// current one. The `list` is emptied and dropped.
    ///
    /// If the cursor is pointing at the "ghost" non-element then the
    /// elements are spliced at the back of the parent list.
    ///
    /// This operation should compute in *O*(1) time. It moves the nodes of
    /// `list` into the parent list, without allocating or deallocating.
    ///
    /// This operation is only available for the [`Global`] allocator: the
    /// nodes would otherwise have to be reallocated when the two lists do
    /// not share the same allocator.
    pub fn splice_before(&mut self, list: LinkedList<T>) {
        // The list is empty -> nothing to splice in.
        if list.len == 0 {
            return;
        }

        // SAFETY: `splice_nodes` is given a valid chain of nodes allocated
        // with the parent list's allocator, and the cursor's pointer stays
        // valid because splicing does not move the existing nodes.
        unsafe {
            let Some((splice_head, splice_tail, splice_len)) = list.detach_all_nodes() else {
                return;
            };
            let node_prev = match self.current {
                None => self.list.tail,
                Some(node) => node.as_ref().prev,
            };
            self.list.splice_nodes(
                node_prev,
                self.current,
                splice_head,
                splice_tail,
                splice_len,
            );

            #[allow(
                clippy::arithmetic_side_effects,
                reason = "the index and the spliced length are both bounded by the \
                          lists' lengths, so adding them cannot overflow"
            )]
            {
                self.index += splice_len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use olive_test_helpers::{BudgetedAlloc, CloneBudget, FlakyCloneAlloc, Ledger, TrackedItem};
    use std::vec;
    use std::vec::Vec;

    fn make_list(values: &[u32]) -> LinkedList<u32> {
        let mut l = LinkedList::new();
        for &v in values {
            l.try_push_back(v).unwrap();
        }
        l
    }

    /// Walks the chain in both directions, checking the links and the
    /// recorded length.
    fn check_links<T, A: Allocator>(list: &LinkedList<T, A>) {
        // SAFETY: the test only walks nodes that belong to the list.
        unsafe {
            let mut len = 0_usize;
            let Some(mut node_ptr) = list.head else {
                assert!(list.tail.is_none());
                assert_eq!(list.len, 0);
                return;
            };
            assert!(list.tail.is_some());

            let mut last_ptr: Option<NonNull<Node<T>>> = None;
            loop {
                let node = &*node_ptr.as_ptr();
                match (last_ptr, node.prev) {
                    // Head pointer
                    (None, None) => {}
                    (None, Some(_)) => panic!("prev link for head"),
                    // Non-head pointer
                    (Some(p), Some(pptr)) => assert_eq!(p, pptr),
                    (Some(_), None) => panic!("prev link is none, not good"),
                }
                len = len.saturating_add(1);
                match node.next {
                    Some(next) => {
                        last_ptr = Some(node_ptr);
                        node_ptr = next;
                    }
                    None => break,
                }
            }

            let tail = list.tail.unwrap().as_ptr();
            assert_eq!(tail, node_ptr.as_ptr());
            assert_eq!(len, list.len);
        }
    }

    #[test]
    fn test_cursor_move_peek() {
        let mut m = make_list(&[1, 2, 3, 4, 5, 6]);
        let mut cursor = m.cursor_front();
        assert_eq!(cursor.current(), Some(&1));
        assert_eq!(cursor.peek_next(), Some(&2));
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.index(), Some(0));
        cursor.move_prev();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.peek_next(), Some(&1));
        assert_eq!(cursor.peek_prev(), Some(&6));
        assert_eq!(cursor.index(), None);
        cursor.move_next();
        cursor.move_next();
        assert_eq!(cursor.current(), Some(&2));
        assert_eq!(cursor.peek_next(), Some(&3));
        assert_eq!(cursor.peek_prev(), Some(&1));
        assert_eq!(cursor.index(), Some(1));

        let mut cursor = m.cursor_back();
        assert_eq!(cursor.current(), Some(&6));
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.peek_prev(), Some(&5));
        assert_eq!(cursor.index(), Some(5));
        cursor.move_next();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.peek_next(), Some(&1));
        assert_eq!(cursor.peek_prev(), Some(&6));
        assert_eq!(cursor.index(), None);
        cursor.move_prev();
        cursor.move_prev();
        assert_eq!(cursor.current(), Some(&5));
        assert_eq!(cursor.peek_next(), Some(&6));
        assert_eq!(cursor.peek_prev(), Some(&4));
        assert_eq!(cursor.index(), Some(4));

        let mut cursor = m.cursor_front_mut();
        assert_eq!(cursor.current(), Some(&mut 1));
        assert_eq!(cursor.peek_next(), Some(&mut 2));
        assert_eq!(cursor.peek_prev(), None);
        assert_eq!(cursor.index(), Some(0));
        cursor.move_prev();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.peek_next(), Some(&mut 1));
        assert_eq!(cursor.peek_prev(), Some(&mut 6));
        assert_eq!(cursor.index(), None);
        cursor.move_next();
        cursor.move_next();
        assert_eq!(cursor.current(), Some(&mut 2));
        assert_eq!(cursor.peek_next(), Some(&mut 3));
        assert_eq!(cursor.peek_prev(), Some(&mut 1));
        assert_eq!(cursor.index(), Some(1));
        let mut cursor2 = cursor.as_cursor();
        assert_eq!(cursor2.current(), Some(&2));
        assert_eq!(cursor2.index(), Some(1));
        cursor2.move_next();
        assert_eq!(cursor2.current(), Some(&3));
        assert_eq!(cursor2.index(), Some(2));
        assert_eq!(cursor.current(), Some(&mut 2));
        assert_eq!(cursor.index(), Some(1));

        let mut cursor = m.cursor_back_mut();
        assert_eq!(cursor.current(), Some(&mut 6));
        assert_eq!(cursor.peek_next(), None);
        assert_eq!(cursor.peek_prev(), Some(&mut 5));
        assert_eq!(cursor.index(), Some(5));
        cursor.move_next();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.peek_next(), Some(&mut 1));
        assert_eq!(cursor.peek_prev(), Some(&mut 6));
        assert_eq!(cursor.index(), None);
        cursor.move_prev();
        cursor.move_prev();
        assert_eq!(cursor.current(), Some(&mut 5));
        assert_eq!(cursor.peek_next(), Some(&mut 6));
        assert_eq!(cursor.peek_prev(), Some(&mut 4));
        assert_eq!(cursor.index(), Some(4));
        let mut cursor2 = cursor.as_cursor();
        assert_eq!(cursor2.current(), Some(&5));
        assert_eq!(cursor2.index(), Some(4));
        cursor2.move_prev();
        assert_eq!(cursor2.current(), Some(&4));
        assert_eq!(cursor2.index(), Some(3));
        assert_eq!(cursor.current(), Some(&mut 5));
        assert_eq!(cursor.index(), Some(4));
    }

    #[test]
    fn test_cursor_mut_insert_and_splice() {
        let mut m = make_list(&[1, 2, 3, 4, 5, 6]);
        let mut cursor = m.cursor_front_mut();
        cursor.try_insert_before(7).unwrap();
        cursor.try_insert_after(8).unwrap();
        check_links(&m);
        assert_eq!(
            m.iter().copied().collect::<Vec<_>>(),
            vec![7, 1, 8, 2, 3, 4, 5, 6]
        );
        let mut cursor = m.cursor_front_mut();
        cursor.move_prev();
        cursor.try_insert_before_give_back(9).unwrap();
        cursor.try_insert_after_give_back(10).unwrap();
        check_links(&m);
        assert_eq!(
            m.iter().copied().collect::<Vec<_>>(),
            vec![10, 7, 1, 8, 2, 3, 4, 5, 6, 9]
        );
        let mut cursor = m.cursor_front_mut();
        cursor.move_prev();
        assert_eq!(cursor.remove_current(), None);
        cursor.move_next();
        cursor.move_next();
        assert_eq!(cursor.remove_current(), Some(7));
        cursor.move_prev();
        cursor.move_prev();
        cursor.move_prev();
        assert_eq!(cursor.remove_current(), Some(9));
        cursor.move_next();
        assert_eq!(cursor.remove_current(), Some(10));
        check_links(&m);
        assert_eq!(
            m.iter().copied().collect::<Vec<_>>(),
            vec![1, 8, 2, 3, 4, 5, 6]
        );
        let mut cursor = m.cursor_front_mut();
        let p = make_list(&[100, 101, 102, 103]);
        let q = make_list(&[200, 201, 202, 203]);
        cursor.splice_after(p);
        cursor.splice_before(q);
        check_links(&m);
        assert_eq!(
            m.iter().copied().collect::<Vec<_>>(),
            vec![200, 201, 202, 203, 1, 100, 101, 102, 103, 8, 2, 3, 4, 5, 6]
        );
        let mut cursor = m.cursor_front_mut();
        cursor.move_prev();
        let tmp = cursor.try_split_before().unwrap();
        assert_eq!(m.iter().count(), 0);
        let mut m = tmp;
        let mut cursor = m.cursor_front_mut();
        cursor.move_next();
        cursor.move_next();
        cursor.move_next();
        cursor.move_next();
        cursor.move_next();
        cursor.move_next();
        let tmp = cursor.try_split_after().unwrap();
        assert_eq!(
            tmp.iter().copied().collect::<Vec<_>>(),
            vec![102, 103, 8, 2, 3, 4, 5, 6]
        );
        check_links(&m);
        assert_eq!(
            m.iter().copied().collect::<Vec<_>>(),
            vec![200, 201, 202, 203, 1, 100, 101]
        );
    }

    #[test]
    fn test_cursor_push_front_back() {
        let mut ll = make_list(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut c = ll.cursor_front_mut();
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.index(), Some(0));
        c.try_push_front(0).unwrap();
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.peek_prev(), Some(&mut 0));
        assert_eq!(c.index(), Some(1));
        c.try_push_back(11).unwrap();
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.peek_prev(), Some(&mut 0));
        assert_eq!(c.index(), Some(1));
        let cursor = ll.cursor_back();
        let p = cursor.front().unwrap();
        assert_eq!(p, &0);
        assert_eq!(
            ll.iter().copied().collect::<Vec<_>>(),
            (0..12).collect::<Vec<_>>()
        );
        check_links(&ll);
    }

    #[test]
    fn test_cursor_pop_front_back() {
        let mut ll = make_list(&[1, 2, 3, 4, 5, 6]);
        let mut c = ll.cursor_back_mut();
        assert_eq!(c.pop_front(), Some(1));
        c.move_prev();
        c.move_prev();
        c.move_prev();
        assert_eq!(c.pop_back(), Some(6));
        let c = c.as_cursor();
        assert_eq!(c.front(), Some(&2));
        assert_eq!(c.back(), Some(&5));
        assert_eq!(c.index(), Some(1));
        assert_eq!(c.current(), Some(&3));
        assert_eq!(ll.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4, 5]);
        check_links(&ll);
        let mut c = ll.cursor_back_mut();
        assert_eq!(c.current(), Some(&mut 5));
        assert_eq!(c.index, 3);
        assert_eq!(c.pop_back(), Some(5));
        assert_eq!(c.current(), None);
        assert_eq!(c.index, 3);
        let mut c = ll.cursor_front_mut();
        assert_eq!(c.pop_front(), Some(2));
        assert_eq!(c.current(), Some(&mut 3));
        assert_eq!(c.index, 0);
    }

    #[test]
    fn test_cursor_oom_insert_keeps_state() {
        let mut l = LinkedList::new_in(BudgetedAlloc::new(2));
        l.try_push_back(1).unwrap();
        l.try_push_back(2).unwrap();
        let mut c = l.cursor_back_mut();
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        // The budget is spent, so the insertion has to fail cleanly.
        let error = c.try_insert_after(3).unwrap_err();
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));
        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![1, 2]);

        let error = c.try_insert_before(3).unwrap_err();
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        let error = c.try_push_front(3).unwrap_err();
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        let error = c.try_push_back(3).unwrap_err();
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        check_links(c.as_list());
    }

    #[test]
    fn test_cursor_oom_insert_give_back_returns_element() {
        let mut l = LinkedList::new_in(BudgetedAlloc::new(1));
        l.try_push_back(1).unwrap();
        let mut c = l.cursor_back_mut();

        let (element, error) = c.try_insert_after_give_back(7).unwrap_err();
        assert_eq!(element, 7);
        assert!(matches!(error, AllocError));
        assert_eq!(c.as_list().len(), 1);
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.index(), Some(0));

        let (element, error) = c.try_insert_before_give_back(8).unwrap_err();
        assert_eq!(element, 8);
        assert!(matches!(error, AllocError));
        assert_eq!(c.as_list().len(), 1);
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.index(), Some(0));

        let (element, error) = c.try_push_front_give_back(9).unwrap_err();
        assert_eq!(element, 9);
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.index(), Some(0));

        let (element, error) = c.try_push_back_give_back(10).unwrap_err();
        assert_eq!(element, 10);
        assert!(matches!(error, AllocError));
        assert_eq!(c.current(), Some(&mut 1));
        assert_eq!(c.index(), Some(0));

        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![1]);
        check_links(c.as_list());
    }

    #[test]
    fn test_cursor_oom_does_not_leak() {
        let ledger = std::sync::Arc::new(Ledger::new());
        let mut l: LinkedList<TrackedItem<u32>, _> = LinkedList::new_in(BudgetedAlloc::new(1));
        l.try_push_back(TrackedItem::construct(&ledger, 1)).unwrap();

        let mut c = l.cursor_front_mut();
        let orphan = TrackedItem::construct(&ledger, 2);
        let (orphan, error) = c.try_insert_after_give_back(orphan).unwrap_err();
        assert!(matches!(error, AllocError));
        let (orphan, error) = c.try_insert_before_give_back(orphan).unwrap_err();
        assert!(matches!(error, AllocError));
        let (orphan, error) = c.try_push_front_give_back(orphan).unwrap_err();
        assert!(matches!(error, AllocError));
        let (orphan, error) = c.try_push_back_give_back(orphan).unwrap_err();
        assert!(matches!(error, AllocError));
        drop(orphan);

        // Nothing may leak: the surviving element must drop with the list.
        drop(l);

        assert!(ledger.leaked_ids().is_empty());
        assert!(ledger.double_dropped().is_empty());
        assert!(ledger.all_dropped_once(0..2));
    }

    #[test]
    fn test_cursor_allocator_clone_failure_leaves_state_untouched() {
        // No clone budget means the allocator cannot be cloned.
        let mut l: LinkedList<u32, _> =
            LinkedList::new_in(FlakyCloneAlloc::new(CloneBudget::new(0)));
        for v in 1..=3u32 {
            l.try_push_back(v).unwrap();
        }
        let mut c = l.cursor_front_mut();
        c.move_next();
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        let error = c.try_split_after().unwrap_err();
        assert!(matches!(error, TryCloneError::Other(_)));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        let error = c.try_split_before().unwrap_err();
        assert!(matches!(error, TryCloneError::Other(_)));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        let error = c.try_remove_current_as_list().unwrap_err();
        assert!(matches!(error, TryCloneError::Other(_)));
        assert_eq!(c.current(), Some(&mut 2));
        assert_eq!(c.index(), Some(1));

        assert_eq!(
            c.as_list().iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        check_links(c.as_list());
    }

    #[test]
    fn test_cursor_split_success() {
        let mut l: LinkedList<u32, _> =
            LinkedList::new_in(FlakyCloneAlloc::new(CloneBudget::new(2)));
        for v in 1..=5u32 {
            l.try_push_back(v).unwrap();
        }

        let mut c = l.cursor_front_mut();
        c.move_next();
        c.move_next();
        let after = c.try_split_after().unwrap();
        assert_eq!(
            c.as_list().iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(after.iter().copied().collect::<Vec<_>>(), vec![4, 5]);
        check_links(c.as_list());
        check_links(&after);

        let before = c.try_split_before().unwrap();
        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![3]);
        assert_eq!(before.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        check_links(c.as_list());
        check_links(&before);
        assert_eq!(c.current(), Some(&mut 3));
        assert_eq!(c.index(), Some(0));
    }

    #[test]
    fn test_cursor_remove_current_as_list() {
        let mut l: LinkedList<u32, _> =
            LinkedList::new_in(FlakyCloneAlloc::new(CloneBudget::new(1)));
        for v in 1..=3u32 {
            l.try_push_back(v).unwrap();
        }

        let mut c = l.cursor_front_mut();
        c.move_next();
        let single = c.try_remove_current_as_list().unwrap().unwrap();
        assert_eq!(c.current(), Some(&mut 3));
        assert_eq!(c.index(), Some(1));
        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(single.iter().copied().collect::<Vec<_>>(), vec![2]);
        check_links(c.as_list());
        check_links(&single);

        // Trying to remove while being unable to clone the allocator results in an error.
        let mut c = l.cursor_back_mut();
        let _err = c.try_remove_current_as_list().expect_err("should return TryCloneError");

        // At the "ghost" non-element there is nothing to remove, so cloning is not needed.
        let mut c = l.cursor_back_mut();
        c.move_next();
        let removed = c.try_remove_current_as_list().unwrap();
        assert!(removed.is_none());
        assert_eq!(c.as_list().iter().copied().collect::<Vec<_>>(), vec![1, 3]);
    }
}
