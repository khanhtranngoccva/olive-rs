//! Trait implementations for [`LinkedList`].
//!
//! This mirrors the std reference implementation, which keeps the derived
//! trait impls (`Clone`, `Debug` and friends) in a separate `traits` module.

use core::fmt;

use super::LinkedList;
use crate::alloc::Allocator;

impl<T: fmt::Debug, A: Allocator> fmt::Debug for LinkedList<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// SAFETY: the list is a plain chain of nodes reached only through `head` and
// `tail`; moving the list moves the whole chain, and the nodes are only ever
// touched from `&mut self` methods. Sending the list to another thread is
// sound exactly when `T` and the allocator are `Send`.
unsafe impl<T: Send, A: Allocator + Send> Send for LinkedList<T, A> {}

// SAFETY: shared access never creates mutable aliases into a node (the
// iteration methods hand out shared `&T` only), so sharing the list across
// threads is sound exactly when `T` and the allocator are `Sync`.
unsafe impl<T: Sync, A: Allocator + Sync> Sync for LinkedList<T, A> {}

#[cfg(test)]
mod tests {
    use super::LinkedList;

    #[test]
    fn list_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<LinkedList<i32>>();
        assert_sync::<LinkedList<i32>>();
    }
}
