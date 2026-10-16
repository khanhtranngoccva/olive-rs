//! A fallible port of `alloc::collections::LinkedList`.

mod allocation;
mod construction;
mod drop;
mod iter;
mod mutation;
mod node;
mod query;
mod traits;

pub use iter::{IntoIter, Iter, IterMut};

use self::node::Node;
use crate::alloc::{Allocator, Global};
use crate::boxed::Box;
use core::marker::PhantomData;
use core::ptr::NonNull;

/// A doubly-linked list with owned nodes.
///
/// `LinkedList` allows prepending and appending elements in constant time
/// and allows popping at either end.
///
/// Growing the list is fallible: allocation failures leave the list unchanged
/// instead of panicking or aborting.
///
/// NOTE: It is almost always better to use [`Vec`](crate::vec::Vec) or
/// [`VecDeque`](crate::collections::vec_deque::VecDeque) because
/// array-based containers are generally faster, more memory efficient, and
/// make better use of CPU cache.
pub struct LinkedList<T, A: Allocator = Global> {
    /// Pointer to the first node, or `None` when the list is empty.
    ///
    /// Invariants: `head.is_some()` exactly when `len > 0`; the node it
    /// points to has `prev == None`; every node reachable from `head` was
    /// allocated with `self.alloc` and is reachable from exactly one
    /// position in the list.
    head: Option<NonNull<Node<T>>>,
    /// Pointer to the last node, or `None` when the list is empty.
    ///
    /// Invariants: `tail.is_some()` exactly when `head.is_some()`; the node
    /// it points to has `next == None`.
    tail: Option<NonNull<Node<T>>>,
    /// The number of elements in the list; the length of the `head` to
    /// `tail` chain.
    len: usize,
    /// The allocator each node was allocated with and will be deallocated
    /// with.
    alloc: A,
    /// Tells the drop checker that the list owns `T` values (the elements
    /// inside its nodes), even though it reaches them only through raw
    /// pointers.
    marker: PhantomData<Box<Node<T>, A>>,
}
