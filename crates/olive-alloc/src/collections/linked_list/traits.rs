//! Trait implementations for [`LinkedList`].

use core::fmt;

use super::LinkedList;
use crate::alloc::Allocator;

impl<T: fmt::Debug, A: Allocator> fmt::Debug for LinkedList<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
